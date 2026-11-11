"""Checks that stop the system from showing made-up facts.

1. numbers_not_in(answer, sources)   every number in an answer must appear in the sources
                                      (at the precision it was written: "$20" ok if the source
                                      says "$20.00", "$25" is not ok if the source says "$20")
2. quote_in(quote, text)              an "evidence" quote must really be in the clause
3. citations_valid(answer, allowed)   every [R05]-style citation must be a rule we retrieved,
                                      and the answer must cite at least one
"""

from __future__ import annotations

import re

NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
CITE = re.compile(r"\[(R\d{2}X?|A\d{2}|clause\s+[\w.-]+)\]", re.I)


def _numbers(text: str) -> list[tuple[float, int]]:
    out = []
    for tok in NUM.findall(text):
        clean = tok.replace("$", "").replace(",", "").replace("%", "")
        try:
            v = float(clean)
        except ValueError:
            continue
        out.append((v, len(clean.split(".")[1]) if "." in clean else 0))
    return out


def numbers_not_in(answer: str, sources: str, ignore_small: int = 3) -> list[float]:
    """Numbers in the answer that are not in the sources. Tiny counts (0-3) are ignored."""
    allowed = [v for v, _ in _numbers(sources)]
    bad = []
    for v, dec in _numbers(CITE.sub("", answer)):   # citation ids like [R05] are not facts
        if dec == 0 and v.is_integer() and v <= ignore_small:
            continue
        if not any(abs(round(a, dec) - v) < 1e-9 for a in allowed):
            bad.append(v)
    return bad


def _squash(s: str) -> str:
    return re.sub(r"[\s\"'`]+", " ", s).strip().lower()


def quote_in(quote: str, text: str) -> bool:
    return bool(quote.strip()) and _squash(quote) in _squash(text)


def citations(answer: str) -> list[str]:
    return [c.upper().replace("CLAUSE ", "clause ") for c in CITE.findall(answer)]


def citations_valid(answer: str, allowed: set[str]) -> tuple[bool, list[str]]:
    found = citations(answer)
    unknown = [c for c in found if c not in allowed]
    return bool(found) and not unknown, unknown
