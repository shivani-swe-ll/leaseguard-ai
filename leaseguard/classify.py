"""Decide, for each clause, if it is OK, worth checking, or likely not allowed.

Order of evidence, strongest first:
1. trigger  - a rule's legal pattern matches the clause words (deterministic, explainable)
   + checks  - small numeric checks (NSF admin fee above $20, deposit above one month's rent)
2. llm      - (optional) the LLM picks one of the top retrieved rules and must quote the clause;
              the quote, the rule id and the verdict are all validated by code
3. retrieval- no trigger and no LLM: if keyword AND vector search both point at a
              "not allowed" rule, ask the user to check (REVIEW); otherwise OK
The checker prefers saying "check this" over guessing.
"""

from __future__ import annotations

import json
import re

from leaseguard.guardrails import quote_in
from leaseguard.llm import extract_json
from leaseguard.models import Clause, ClauseFinding, LeaseTerms, RetrievalHit, Rule, Verdict
from leaseguard.retrieval.hybrid import HybridRetriever

SYSTEM_CLASSIFY = (
    "You check one clause of an Ontario residential lease against candidate rules. "
    "Pick the ONE rule that applies, or \"none\". Decide the verdict: LIKELY_VOID (the clause breaks the rule), "
    "REVIEW (it may break the rule), or OK. Copy the exact words from the clause that matter. Reply with JSON only: "
    '{"rule_id": "R05 or none", "verdict": "LIKELY_VOID|REVIEW|OK", "evidence": "exact words from the clause", '
    '"reason": "one simple sentence"}'
)


def _amounts(text: str) -> list[tuple[float, int]]:
    return [(float(m.group(1).replace(",", "")), m.start()) for m in re.finditer(r"\$\s?([\d,]+(?:\.\d{2})?)", text)]


class ClauseChecker:
    def __init__(self, retriever: HybridRetriever, llm=None, k: int = 4):
        self.retriever, self.llm, self.k = retriever, llm, k
        self.rules = retriever.rules
        self.by_id = retriever.by_id

    def _finding(self, clause, rule: Rule | None, verdict: Verdict, confidence: float, method: str,
                 evidence: str, explanation: str, hits: list[RetrievalHit]) -> ClauseFinding:
        return ClauseFinding(
            clause=clause, verdict=verdict, rule_id=rule.id if rule else None, rule_title=rule.title if rule else None,
            section=rule.section if rule else None, confidence=round(confidence, 2), method=method, evidence=evidence,
            explanation=explanation, tenant_tip=rule.tenant_tip if rule else "", candidates=hits)

    def check(self, clause: Clause, terms: LeaseTerms | None = None) -> ClauseFinding:
        text = clause.text
        hits = self.retriever.search(text, k=self.k, mode="hybrid_rerank")
        order = {h.rule_id: i for i, h in enumerate(hits)}

        # 1. triggers (all rules, not only the top hits)
        matched = [(r, w) for r in self.rules if (w := r.matches(text))]
        if matched:
            matched.sort(key=lambda rw: (-rw[0].clause_verdict.severity, order.get(rw[0].id, 99)))
            rule, words = matched[0]
            special = self._special_checks(rule, clause, terms, hits)
            if special:
                return special
            conf = 0.95 if len(matched) == 1 else 0.9
            return self._finding(clause, rule, rule.clause_verdict, conf, "trigger", words,
                                 f"{rule.title} ({rule.section}). {rule.summary}", hits)

        # 2. LLM over the retrieved candidates
        if self.llm is not None:
            return self._llm_check(clause, hits)

        # 3. retrieval only
        top = self.by_id[hits[0].rule_id] if hits else None
        if (top and top.clause_verdict != Verdict.OK and hits[0].bm25_rank == 1
                and (hits[0].vector_rank or 99) <= 2):
            return self._finding(clause, top, Verdict.REVIEW, 0.4, "retrieval", "",
                                 f"This clause looks related to \"{top.title}\" ({top.section}). "
                                 "No exact pattern matched, so please read the rule and check.", hits)
        return self._finding(clause, None, Verdict.OK, 0.6, "none", "",
                             "No rule in the knowledge base matched this clause.", hits)

    # ---------- numeric checks ----------
    def _special_checks(self, rule: Rule, clause: Clause, terms: LeaseTerms | None, hits) -> ClauseFinding | None:
        text = clause.text
        if rule.id == "R09":   # NSF: bank charge + at most $20 admin
            for amount, pos in _amounts(text):
                near = text[max(0, pos - 12):pos + 25].lower()   # words right around this amount
                if "bank" not in near and amount > 20:
                    return self._finding(clause, rule, Verdict.LIKELY_VOID, 0.9, "check", f"${amount:,.2f}",
                                         f"The NSF admin charge of ${amount:,.2f} is more than the $20 allowed "
                                         f"({rule.section}).", hits)
        if rule.id == "R06" and terms and terms.monthly_rent:
            for amount, _ in _amounts(text):
                if amount > terms.monthly_rent + 0.01 and "deposit" in text.lower():
                    big = self.by_id.get("R06X", rule)
                    return self._finding(clause, big, Verdict.LIKELY_VOID, 0.85, "check", f"${amount:,.2f}",
                                         f"The rent deposit ${amount:,.2f} is more than one month's rent "
                                         f"(${terms.monthly_rent:,.2f}). {big.summary}", hits)
        return None

    # ---------- LLM ----------
    def _llm_check(self, clause: Clause, hits: list[RetrievalHit]) -> ClauseFinding:
        cands = [self.by_id[h.rule_id] for h in hits]
        payload = {"clause": clause.text,
                   "candidate_rules": [{"id": r.id, "title": r.title, "section": r.section, "rule": r.summary,
                                        "verdict_if_clause_matches": r.clause_verdict.value} for r in cands]}
        try:
            data = extract_json(self.llm.complete(SYSTEM_CLASSIFY, json.dumps(payload, indent=2), max_tokens=300).text)
            rid = str(data.get("rule_id", "none")).upper()
            verdict = Verdict(str(data.get("verdict", "")).upper())
            evidence = str(data.get("evidence", ""))
            reason = str(data.get("reason", "")).strip()
        except Exception:
            return self._finding(clause, None, Verdict.REVIEW, 0.3, "llm_rejected", "",
                                 "The AI answer could not be read, so please check this clause yourself.", hits)
        if rid == "NONE":
            return self._finding(clause, None, Verdict.OK, 0.6, "llm", "", reason or "No rule applies.", hits)
        rule = self.by_id.get(rid)
        valid = (rule is not None and rule in cands and quote_in(evidence, clause.text)
                 and verdict.severity <= rule.clause_verdict.severity)   # cannot call an "allowed" rule void
        if not valid:
            return self._finding(clause, None, Verdict.REVIEW, 0.3, "llm_rejected", "",
                                 "The AI answer did not pass our checks (wrong rule or a quote not in the clause), "
                                 "so please check this clause yourself.", hits)
        return self._finding(clause, rule, verdict, 0.7, "llm", evidence,
                             f"{reason} ({rule.title}, {rule.section})", hits)
