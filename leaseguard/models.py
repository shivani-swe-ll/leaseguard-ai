"""Typed data models used across the project (Pydantic v2)."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Verdict(str, Enum):
    LIKELY_VOID = "LIKELY_VOID"   # very likely not enforceable in Ontario
    REVIEW = "REVIEW"             # may be a problem: read the rule and check
    OK = "OK"                     # follows the rules we know about

    @property
    def label(self) -> str:
        return {"LIKELY_VOID": "Likely not allowed", "REVIEW": "Check this", "OK": "OK"}[self.value]

    @property
    def severity(self) -> int:
        return {"LIKELY_VOID": 2, "REVIEW": 1, "OK": 0}[self.value]


class Rule(BaseModel):
    """One plain-language rule from the knowledge base (knowledge/rules.json)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    title: str
    section: str
    category: str
    clause_verdict: Verdict
    summary: str
    tenant_tip: str
    keywords: list[str]
    triggers: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)

    def index_text(self) -> str:
        """Text used to index this rule for search."""
        return " ".join([self.title, self.summary, " ".join(self.keywords), " ".join(self.examples)])

    def matches(self, text: str) -> str | None:
        """Return the matched words if a trigger fires and no exclude pattern applies."""
        if any(re.search(p, text, re.I) for p in self.exclude):
            return None
        for p in self.triggers:
            m = re.search(p, text, re.I)
            if m:
                return m.group(0)
        return None


class Clause(BaseModel):
    number: str           # "7" or "p2-3" when the lease has no numbers
    text: str
    page: int = 1


class RetrievalHit(BaseModel):
    rule_id: str
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    trigger: str | None = None   # words that fired a trigger, if any


class ClauseFinding(BaseModel):
    clause: Clause
    verdict: Verdict
    rule_id: str | None = None
    rule_title: str | None = None
    section: str | None = None
    confidence: float = Field(ge=0, le=1)
    method: str                       # "trigger" | "check" | "llm" | "retrieval" | "none"
    evidence: str = ""                # exact words from the clause that support the finding
    explanation: str
    tenant_tip: str = ""
    candidates: list[RetrievalHit] = Field(default_factory=list)


class LeaseTerms(BaseModel):
    """Key facts pulled out of the lease (structured extraction)."""

    landlord: str | None = None
    tenant: str | None = None
    address: str | None = None
    monthly_rent: float | None = None
    rent_deposit: float | None = None
    key_deposit: float | None = None
    start_date: str | None = None
    term: str | None = None
    utilities_tenant_pays: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)   # field -> exact text it came from
    source: str = "regex"                                    # "regex" | "llm" | "regex+llm"


class LeaseReport(BaseModel):
    source_name: str
    pages: int
    terms: LeaseTerms
    findings: list[ClauseFinding]
    term_checks: list[ClauseFinding] = Field(default_factory=list)  # cross-field checks, e.g. deposit > rent
    questions_for_landlord: list[str] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out = {v.value: 0 for v in Verdict}
        for f in [*self.findings, *self.term_checks]:
            out[f.verdict.value] += 1
        return out

    @property
    def flagged(self) -> list[ClauseFinding]:
        items = [f for f in [*self.findings, *self.term_checks] if f.verdict != Verdict.OK]
        return sorted(items, key=lambda f: (-f.verdict.severity, -f.confidence))


class Citation(BaseModel):
    ref: str            # "R05" or "clause 3"
    title: str
    section: str = ""
    text: str = ""


class Answer(BaseModel):
    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    source: str = "rules"         # "rules" | "llm" | "llm_rejected" | "llm_error" | "abstained"
    abstained: bool = False
    trace: dict = Field(default_factory=dict)
