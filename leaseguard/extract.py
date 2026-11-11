"""Structured extraction: pull key facts (rent, deposits, parties, dates) out of a lease.

Regex first (free, exact, explainable). If an LLM is configured, it fills the fields
regex missed, but every value must come with an exact quote from the lease, and the
code checks that the quote is really there and contains the value.
"""

from __future__ import annotations

import json
import re

from leaseguard.guardrails import quote_in
from leaseguard.llm import extract_json
from leaseguard.models import LeaseTerms

MONEY = r"\$\s?([\d,]+(?:\.\d{2})?)"
MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
UTILITIES = ["hydro", "electricity", "gas", "water", "heat", "internet", "cable", "parking"]


def _money(s: str) -> float:
    return float(s.replace(",", ""))


def _line(pattern: str, text: str) -> tuple[str, str] | None:
    m = re.search(pattern, text, re.I | re.M)
    return (m.group(m.lastindex or 0).strip(), m.group(0).strip()) if m else None


def regex_terms(text: str) -> LeaseTerms:
    t = LeaseTerms()
    ev: dict[str, str] = {}

    for field, pat in [
        ("landlord", r"^\s*landlord\s*:\s*(.+)$"),
        ("tenant", r"^\s*tenants?\s*:\s*(.+)$"),
        ("address", r"^\s*(?:rental\s+unit|premises|unit\s+address|address|property)\s*:\s*(.+)$"),
        ("term", r"^\s*term\s*:\s*(.+)$"),
    ]:
        hit = _line(pat, text)
        if hit:
            setattr(t, field, hit[0])
            ev[field] = hit[1]

    m = re.search(rf"(?:start\w*|from|beginning|commencing)\s+(?:on\s+)?({MONTHS}\s+\d{{1,2}},\s+\d{{4}})", text, re.I)
    if m:
        t.start_date, ev["start_date"] = m.group(1), m.group(0)

    for pat in [rf"(?:monthly\s+)?rent\s+(?:is|of|will\s+be|shall\s+be)\s+{MONEY}",
                rf"{MONEY}\s+(?:per|a|each|/)\s*month"]:
        m = re.search(pat, text, re.I)
        if m:
            t.monthly_rent, ev["monthly_rent"] = _money(m.group(1)), m.group(0)
            break

    m = re.search(rf"rent\s+deposit\s+(?:of|is|equal\s+to)?\s*{MONEY}", text, re.I)
    if m:
        t.rent_deposit, ev["rent_deposit"] = _money(m.group(1)), m.group(0)
    else:
        m = re.search(r"(rent\s+deposit\s+equal\s+to\s+one\s+month'?s\s+rent|first\s+and\s+last\s+month'?s\s+rent)", text, re.I)
        if m and t.monthly_rent:
            t.rent_deposit, ev["rent_deposit"] = t.monthly_rent, m.group(0)

    m = re.search(rf"key\s+deposit\s+(?:of\s+)?{MONEY}", text, re.I)
    if m:
        t.key_deposit, ev["key_deposit"] = _money(m.group(1)), m.group(0)

    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        s = sentence.lower()
        if re.search(r"tenant\s+(pays|will\s+pay|shall\s+pay|is\s+responsible)|paid\s+by\s+the\s+tenant|not\s+included", s):
            for u in UTILITIES:
                if re.search(rf"\b{u}\b", s) and u not in t.utilities_tenant_pays:
                    t.utilities_tenant_pays.append(u)
                    ev.setdefault("utilities_tenant_pays", sentence.strip())
    t.evidence = ev
    return t


SYSTEM_EXTRACT = (
    "You read a residential lease and return JSON only. For each field give the value and an EXACT quote "
    "copied from the lease that contains it. Use null when the lease does not say. Fields: landlord, tenant, "
    "address, monthly_rent (number), rent_deposit (number), key_deposit (number), start_date, term. "
    'Format: {"monthly_rent": {"value": 1750, "quote": "The monthly rent is $1,750"}, ...}'
)


def extract_terms(text: str, llm=None) -> LeaseTerms:
    terms = regex_terms(text)
    if llm is None:
        return terms
    missing = [f for f in ("landlord", "tenant", "address", "monthly_rent", "rent_deposit", "key_deposit",
                           "start_date", "term") if getattr(terms, f) in (None, "")]
    if not missing:
        return terms
    try:
        data = extract_json(llm.complete(SYSTEM_EXTRACT, text[:12000], max_tokens=600).text)
    except Exception:
        return terms
    added = False
    for f in missing:
        item = data.get(f) or {}
        value, quote = item.get("value"), str(item.get("quote") or "")
        if value in (None, "") or not quote_in(quote, text):
            continue
        if f in ("monthly_rent", "rent_deposit", "key_deposit"):
            try:
                value = float(str(value).replace(",", "").replace("$", ""))
            except ValueError:
                continue
            nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", quote)]
            if value not in nums:          # the number must really be in the quote
                continue
        elif str(value).lower() not in quote.lower():
            continue
        setattr(terms, f, value)
        terms.evidence[f] = quote
        added = True
    if added:
        terms.source = "regex+llm"
    return terms


def terms_json(terms: LeaseTerms) -> str:
    return json.dumps(terms.model_dump(exclude={"evidence"}), indent=2)
