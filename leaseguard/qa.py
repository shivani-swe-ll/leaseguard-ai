"""Answer tenant questions with citations, using RAG over the rule knowledge base.

Flow: question -> scope check -> hybrid retrieval (top rules, plus the user's own
lease clauses if a lease was checked) -> answer.
* Rules mode (no LLM): the answer is built from the top rule's own text, so it is
  always faithful, and it cites [R05]-style ids.
* LLM mode: the LLM may only use the retrieved sources and must cite them. Code then
  checks: every citation is a retrieved source, at least one citation exists, and
  every number appears in the sources. If any check fails, the rules answer is shown.
* Abstain: out-of-scope questions (other provinces, commercial leases) or questions
  with no matching rule get "I can't answer that" instead of a guess.
"""

from __future__ import annotations

import re
from typing import Iterator

from leaseguard.guardrails import citations_valid, numbers_not_in
from leaseguard.knowledge import DISCLAIMER
from leaseguard.models import Answer, Citation, LeaseReport, Verdict
from leaseguard.retrieval.bm25 import tokenize
from leaseguard.retrieval.hybrid import HybridRetriever, expand_query

OUT_OF_SCOPE = [
    (r"\bcommercial\b|\bbusiness\s+(lease|premises)|\boffice\s+space\b|\bretail\s+space\b",
     "This tool only covers residential (home) tenancies, not commercial leases."),
    (r"\b(british\s+columbia|\bb\.?c\.?\b|alberta|quebec|québec|manitoba|saskatchewan|nova\s+scotia|new\s+brunswick|"
     r"newfoundland|prince\s+edward\s+island|yukon|nunavut|northwest\s+territories|united\s+states|california|new\s+york)\b",
     "This tool only covers Ontario. Rules are different in other provinces and countries."),
]

SYSTEM_QA = (
    "You answer a tenant's question about Ontario residential tenancy rules. Use ONLY the numbered sources. "
    "After each fact, cite its source id in square brackets, like [R05] or [clause 3]. If the sources do not "
    "answer the question, say you are not sure and suggest a legal clinic. Use short, simple sentences for a "
    "reader whose first language may not be English. Do not give legal advice; explain the rule. Plain text."
)


class Assistant:
    def __init__(self, retriever: HybridRetriever, llm=None, min_keyword_score: float = 1.0):
        self.retriever, self.llm, self.min_keyword_score = retriever, llm, min_keyword_score

    # ---------- shared steps ----------
    def _scope(self, question: str) -> str | None:
        for pattern, msg in OUT_OF_SCOPE:
            if re.search(pattern, question, re.I):
                return msg
        return None

    def _sources(self, question: str, report: LeaseReport | None, k: int = 3):
        hits = self.retriever.search(question, k=k, mode="hybrid_rerank")
        best_kw = max(self.retriever.bm25.scores(expand_query(question)) or [0])
        rules = [self.retriever.by_id[h.rule_id] for h in hits]
        clauses = []
        if report:  # the user's own clauses that touch the same rules
            ids = {r.id for r in rules}
            clauses = [f for f in report.findings if f.rule_id in ids][:2]
        return rules, clauses, best_kw

    def _abstain(self, question: str, msg: str) -> Answer:
        return Answer(question=question, text=f"{msg} Please contact a community legal clinic or the Landlord and "
                                              "Tenant Board for help.", source="abstained", abstained=True)

    @staticmethod
    def _context(rules, clauses) -> tuple[str, list[Citation]]:
        cites, parts = [], []
        for r in rules:
            parts.append(f"[{r.id}] {r.title} ({r.section}): {r.summary} Tip: {r.tenant_tip}")
            cites.append(Citation(ref=r.id, title=r.title, section=r.section, text=r.summary))
        for f in clauses:
            ref = f"clause {f.clause.number}"
            parts.append(f"[{ref}] (from the user's lease, our check: {f.verdict.label}): {f.clause.text}")
            cites.append(Citation(ref=ref, title=f"Your lease, clause {f.clause.number}", text=f.clause.text))
        return "\n".join(parts), cites

    def _rules_answer(self, question: str, rules, clauses, cites) -> Answer:
        top = rules[0]
        text = f"{top.summary} [{top.id}]\n\nTip: {top.tenant_tip} [{top.id}]"
        if len(rules) > 1 and rules[1].clause_verdict != Verdict.OK and set(tokenize(question)) & set(tokenize(rules[1].title)):
            text += f"\n\nRelated: {rules[1].title} [{rules[1].id}]"
        for f in clauses:
            if f.verdict != Verdict.OK:
                text += f"\n\nIn your lease, clause {f.clause.number} may be affected ({f.verdict.label}). [clause {f.clause.number}]"
        return Answer(question=question, text=text, citations=cites, source="rules")

    def _validate(self, text: str, context: str, cites: list[Citation]) -> dict:
        ok_cites, unknown = citations_valid(text, {c.ref for c in cites})
        bad_numbers = numbers_not_in(text, context)
        return {"citations_ok": ok_cites, "unknown_citations": unknown, "ungrounded_numbers": bad_numbers,
                "passed": ok_cites and not bad_numbers}

    # ---------- public API ----------
    def ask(self, question: str, report: LeaseReport | None = None) -> Answer:
        msg = self._scope(question)
        if msg:
            return self._abstain(question, msg)
        rules, clauses, best_kw = self._sources(question, report)
        if not rules or best_kw < self.min_keyword_score:
            return self._abstain(question, "I could not find a rule about that in my knowledge base.")
        context, cites = self._context(rules, clauses)
        fallback = self._rules_answer(question, rules, clauses, cites)
        if self.llm is None:
            return fallback
        try:
            resp = self.llm.complete(SYSTEM_QA, f"Question: {question}\n\nSources:\n{context}")
        except Exception as exc:
            fallback.source, fallback.trace = "llm_error", {"error": str(exc)}
            return fallback
        check = self._validate(resp.text, context, cites)
        if not check["passed"]:
            fallback.source, fallback.trace = "llm_rejected", check
            fallback.text = "(The AI answer did not pass our citation and number checks, so here is the rule itself.)\n\n" + fallback.text
            return fallback
        return Answer(question=question, text=resp.text, citations=cites, source="llm", trace=check)

    def stream(self, question: str, report: LeaseReport | None = None) -> Iterator[str | Answer]:
        """Yield text pieces while the LLM writes, then yield the final checked Answer.

        If the streamed text fails the checks, the final Answer is the rules answer
        (source="llm_rejected"), and the app replaces the streamed text with it.
        """
        msg = self._scope(question)
        rules, clauses, best_kw = ([], [], 0) if msg else self._sources(question, report)
        if msg or not rules or best_kw < self.min_keyword_score:
            yield self.ask(question, report)
            return
        if self.llm is None:
            ans = self.ask(question, report)
            yield ans.text
            yield ans
            return
        context, cites = self._context(rules, clauses)
        pieces = []
        try:
            for piece in self.llm.stream(SYSTEM_QA, f"Question: {question}\n\nSources:\n{context}"):
                pieces.append(piece)
                yield piece
        except Exception as exc:
            ans = self._rules_answer(question, rules, clauses, cites)
            ans.source, ans.trace = "llm_error", {"error": str(exc)}
            yield ans
            return
        text = "".join(pieces).strip()
        check = self._validate(text, context, cites)
        if check["passed"]:
            yield Answer(question=question, text=text, citations=cites, source="llm", trace=check)
        else:
            ans = self._rules_answer(question, rules, clauses, cites)
            ans.source, ans.trace = "llm_rejected", check
            yield ans


def format_answer(ans: Answer) -> str:
    lines = [ans.text, ""]
    if ans.citations:
        lines.append("Sources:")
        lines += [f"  [{c.ref}] {c.title}" + (f" - {c.section}" if c.section else "") for c in ans.citations]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)
