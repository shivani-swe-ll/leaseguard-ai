"""Hybrid retrieval: BM25 + vectors, fused with Reciprocal Rank Fusion, then reranked.

Modes (the eval compares all four on the same questions):
  "bm25"          keyword search only
  "vector"        embedding search only
  "hybrid"        both lists fused with RRF: score = sum(1 / (60 + rank))
  "hybrid_rerank" hybrid, then a reranker that boosts rules whose trigger
                  patterns match the text and rules whose keywords appear in it
"""

from __future__ import annotations

from leaseguard.models import RetrievalHit, Rule
from leaseguard.retrieval.bm25 import BM25
from leaseguard.retrieval.vectors import LocalEmbedder, VectorIndex

RRF_K = 60

# Everyday words people use -> words used in the rules (helps keyword search only).
SYNONYMS = {
    "comes in": "enter entry", "come in": "enter entry", "walk in": "enter entry", "show up": "enter entry",
    "kick me out": "terminate vacate", "kick out": "terminate vacate", "evict": "terminate eviction",
    "late": "late fee penalty", "bounced": "NSF returned cheque", "pet": "pets animals",
    "sublease": "sublet", "roommate": "sublet assign", "raise": "increase", "hike": "increase",
    "fix": "repairs", "broken": "repairs maintenance", "heat": "vital services heat",
    "deposit": "deposit", "extra money": "fee charge", "fees": "fee charge",
}


def expand_query(query: str) -> str:
    q = query.lower()
    extra = [v for k, v in SYNONYMS.items() if k in q]
    return query + (" " + " ".join(extra) if extra else "")
MODES = ("bm25", "vector", "hybrid", "hybrid_rerank")


class HybridRetriever:
    def __init__(self, rules: tuple[Rule, ...] | list[Rule], embedder=None):
        self.rules = list(rules)
        self.by_id = {r.id: r for r in self.rules}
        texts = [r.index_text() for r in self.rules]
        self.bm25 = BM25(texts)
        self.embedder = embedder or LocalEmbedder()
        self.index = VectorIndex.build([r.id for r in self.rules], texts, self.embedder)

    def _bm25_ranked(self, query: str) -> list[tuple[str, float]]:
        scores = self.bm25.scores(query)
        pairs = sorted(zip([r.id for r in self.rules], scores), key=lambda x: -x[1])
        return [(i, s) for i, s in pairs if s > 0]

    def _vector_ranked(self, query: str) -> list[tuple[str, float]]:
        return self.index.search(self.embedder.embed([query])[0], k=len(self.rules))

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank") -> list[RetrievalHit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        bm = self._bm25_ranked(expand_query(query))
        vec = self._vector_ranked(query)
        bm_rank = {rid: i + 1 for i, (rid, _) in enumerate(bm)}
        vec_rank = {rid: i + 1 for i, (rid, _) in enumerate(vec)}

        if mode == "bm25":
            scored = {rid: s for rid, s in bm}
        elif mode == "vector":
            scored = {rid: s for rid, s in vec}
        else:
            scored = {}
            for rid in self.by_id:
                s = 0.0
                if rid in bm_rank:
                    s += 1 / (RRF_K + bm_rank[rid])
                if rid in vec_rank:
                    s += 1 / (RRF_K + vec_rank[rid])
                scored[rid] = s

        triggers = {}
        if mode == "hybrid_rerank":
            text = query.lower()
            for rid in list(scored):
                rule = self.by_id[rid]
                hit = rule.matches(query)
                if hit:
                    triggers[rid] = hit
                    scored[rid] += 1.0          # a matching legal pattern is very strong evidence
                # small boost per exact multi-word keyword phrase ("damage deposit", not just "rent")
                kw = sum(1 for w in rule.keywords if " " in w and w.lower() in text)
                scored[rid] += 0.004 * kw

        ranked = sorted(scored.items(), key=lambda x: -x[1])[:k]
        return [RetrievalHit(rule_id=rid, score=round(s, 5), bm25_rank=bm_rank.get(rid),
                             vector_rank=vec_rank.get(rid), trigger=triggers.get(rid)) for rid, s in ranked]
