"""Check a whole lease and build a report."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from leaseguard.classify import ClauseChecker
from leaseguard.documents import load_pages, pages_from_text, split_clauses
from leaseguard.extract import extract_terms
from leaseguard.knowledge import DISCLAIMER, load_rules, rules_by_id
from leaseguard.models import Clause, ClauseFinding, LeaseReport, Verdict
from leaseguard.retrieval.hybrid import HybridRetriever

QUESTIONS = {
    "R03": "Your lease bans pets. Under Ontario law that clause is void; do you still plan to enforce it?",
    "R05": "Can you refund the damage/security deposit? Ontario only allows a rent deposit and a refundable key deposit.",
    "R06X": "Can we lower the deposit to one month's rent, as Ontario law allows?",
    "R08": "Which fees in the lease are you planning to charge? Late fees and admin fees are not allowed in Ontario.",
    "R10": "Can I pay rent each month by e-transfer instead of post-dated cheques?",
    "R12": "Can we remove the clause that makes me responsible for all repairs? Repairs are the landlord's duty in Ontario.",
    "R14": "Will you give 24 hours written notice before entering, as the law requires?",
    "R16": "Can we confirm the tenancy continues month to month after the first year?",
    "R18": "Can we confirm rent increases follow the 12-month rule with 90 days written notice?",
    "R19": "If I need to move, can I ask for your consent to sublet or assign?",
}


@lru_cache(maxsize=2)
def default_retriever() -> HybridRetriever:
    return HybridRetriever(load_rules())


def check_clauses(clauses: list[Clause], full_text: str, source_name: str, pages: int, llm=None,
                  retriever: HybridRetriever | None = None) -> LeaseReport:
    retriever = retriever or default_retriever()
    terms = extract_terms(full_text, llm=llm)
    checker = ClauseChecker(retriever, llm=llm)
    findings = [checker.check(c, terms) for c in clauses]

    term_checks: list[ClauseFinding] = []
    rules = rules_by_id()
    if terms.rent_deposit and terms.monthly_rent and terms.rent_deposit > terms.monthly_rent + 0.01:
        r = rules["R06X"]
        term_checks.append(ClauseFinding(
            clause=Clause(number="terms", text=terms.evidence.get("rent_deposit", "")), verdict=Verdict.LIKELY_VOID,
            rule_id=r.id, rule_title=r.title, section=r.section, confidence=0.85, method="check",
            evidence=terms.evidence.get("rent_deposit", ""),
            explanation=f"Rent deposit ${terms.rent_deposit:,.2f} is more than one month's rent "
                        f"${terms.monthly_rent:,.2f}.", tenant_tip=r.tenant_tip))

    asked = []
    for f in sorted([*findings, *term_checks], key=lambda f: -f.verdict.severity):
        if f.verdict != Verdict.OK and f.rule_id in QUESTIONS and QUESTIONS[f.rule_id] not in asked:
            asked.append(QUESTIONS[f.rule_id])
    return LeaseReport(source_name=source_name, pages=pages, terms=terms, findings=findings,
                       term_checks=term_checks, questions_for_landlord=asked)


def check_lease_file(path: str | Path, llm=None) -> LeaseReport:
    pages = load_pages(path)
    text = "\n".join(p.text for p in pages)
    return check_clauses(split_clauses(pages), text, Path(path).name, len(pages), llm=llm)


def check_lease_text(text: str, name: str = "pasted lease", llm=None) -> LeaseReport:
    pages = pages_from_text(text)
    return check_clauses(split_clauses(pages), text, name, 1, llm=llm)


def to_markdown(rep: LeaseReport) -> str:
    c = rep.counts
    t = rep.terms
    lines = [f"# LeaseGuard AI: {rep.source_name}", "",
             f"**{c['LIKELY_VOID']} likely not allowed · {c['REVIEW']} to check · {c['OK']} OK**", "",
             "## Key terms", ""]
    for label, val in [("Landlord", t.landlord), ("Tenant", t.tenant), ("Address", t.address),
                       ("Monthly rent", f"${t.monthly_rent:,.2f}" if t.monthly_rent else None),
                       ("Rent deposit", f"${t.rent_deposit:,.2f}" if t.rent_deposit else None),
                       ("Key deposit", f"${t.key_deposit:,.2f}" if t.key_deposit else None),
                       ("Start date", t.start_date), ("Term", t.term),
                       ("Utilities you pay", ", ".join(t.utilities_tenant_pays) or None)]:
        lines.append(f"- {label}: {val or 'not found'}")
    lines += ["", "## Clauses to look at", ""]
    if not rep.flagged:
        lines.append("Nothing flagged.")
    for f in rep.flagged:
        lines.append(f"- **{f.verdict.label}** · clause {f.clause.number} · {f.rule_title or ''} ({f.section or ''})")
        lines.append(f"  - Clause: \"{f.clause.text}\"")
        lines.append(f"  - Why: {f.explanation}")
        if f.tenant_tip:
            lines.append(f"  - Tip: {f.tenant_tip}")
    if rep.questions_for_landlord:
        lines += ["", "## Questions to ask your landlord", ""]
        lines += [f"{i}. {q}" for i, q in enumerate(rep.questions_for_landlord, 1)]
    lines += ["", f"_{DISCLAIMER}_"]
    return "\n".join(lines) + "\n"


def to_json(rep: LeaseReport) -> str:
    data = rep.model_dump(mode="json")
    data["counts"] = rep.counts
    return json.dumps(data, indent=2)
