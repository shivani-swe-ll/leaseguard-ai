"""BM25 keyword search (pure Python)."""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are", "be", "by", "with", "at", "as",
        "it", "this", "that", "will", "shall", "may", "any", "all", "if", "from", "than", "can", "has", "have", "per",
        # question words: they carry no legal meaning
        "what", "how", "do", "does", "did", "my", "i", "me", "am", "much", "many", "when", "where", "who", "why",
        "should", "which", "there", "about", "they", "their", "you", "your", "best", "get", "need", "want", "wants"}


def stem(t: str) -> str:
    """A tiny stemmer: 'deposits' -> 'deposit', 'subletting' -> 'sublet', 'entered' -> 'enter'."""
    for suf, min_len in (("ting", 7), ("ing", 6), ("ies", 5), ("ed", 5), ("es", 5), ("s", 4)):
        if t.endswith(suf) and len(t) >= min_len:
            base = t[: -len(suf)]
            return base + "y" if suf == "ies" else base
    return t


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.7):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, length in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out
