"""The law knowledge base: plain-language rules with section citations."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from leaseguard.models import Rule

RULES_PATH = Path(__file__).with_name("rules.json")
DISCLAIMER = (
    "Educational project, not legal advice. Rules are plain-language summaries of the Residential "
    "Tenancies Act, 2006 (Ontario). Check the official law at ontario.ca/laws or ask a legal clinic."
)


@lru_cache(maxsize=4)
def load_rules(path: str | None = None) -> tuple[Rule, ...]:
    raw = json.loads(Path(path or RULES_PATH).read_text(encoding="utf-8"))
    rules = tuple(Rule(**r) for r in raw["rules"])
    for r in rules:  # fail fast on a broken pattern
        for p in [*r.triggers, *r.exclude]:
            re.compile(p, re.I)
    ids = [r.id for r in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate rule ids in rules.json")
    return rules


def rules_by_id(path: str | None = None) -> dict[str, Rule]:
    return {r.id: r for r in load_rules(path)}
