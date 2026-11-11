"""Vector (embedding) search.

Two embedders, same interface (`embed(texts) -> np.ndarray`, rows L2-normalised):

* LocalEmbedder (default) - character n-gram TF-IDF vectors hashed into a fixed size.
  No model download, no API key, deterministic. It catches spelling variants and word
  forms ("sublet" / "subletting" / "sub-let") that exact keyword search misses.
* APIEmbedder - real neural embeddings from any OpenAI-compatible `/embeddings`
  endpoint (OpenAI, or Ollama with an embedding model such as nomic-embed-text).

VectorIndex stores the vectors in a numpy matrix and searches by cosine similarity.
It can be saved to / loaded from a .npz file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np


class LocalEmbedder:
    name = "local-char-ngram"

    def __init__(self, dim: int = 4096, ngrams: tuple[int, ...] = (3, 4, 5)):
        self.dim, self.ngrams = dim, ngrams
        self.idf = np.ones(dim, dtype=np.float32)

    def _grams(self, text: str) -> Counter:
        words = re.findall(r"[a-z0-9]+", text.lower())
        c: Counter = Counter()
        for w in words:
            w = f" {w} "
            for n in self.ngrams:
                for i in range(max(len(w) - n + 1, 1)):
                    c[self._bucket(w[i:i + n])] += 1
        return c

    def _bucket(self, gram: str) -> int:
        return int(hashlib.md5(gram.encode()).hexdigest()[:8], 16) % self.dim

    def fit(self, corpus: list[str]) -> "LocalEmbedder":
        df = np.zeros(self.dim, dtype=np.float32)
        for doc in corpus:
            for b in self._grams(doc):
                df[b] += 1
        n = len(corpus)
        self.idf = np.log((1 + n) / (1 + df)).astype(np.float32) + 1.0
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for b, f in self._grams(t).items():
                out[i, b] = (1 + math.log(f)) * self.idf[b]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class APIEmbedder:
    """Neural embeddings from an OpenAI-compatible /embeddings endpoint (stdlib only)."""

    def __init__(self, model: str, base_url: str, api_key: str = "", timeout: int = 60):
        self.name = f"api:{model}"
        self.model, self.base_url, self.api_key, self.timeout = model, base_url.rstrip("/"), api_key, timeout

    @classmethod
    def from_env(cls) -> "APIEmbedder | None":
        model = os.getenv("EMBED_MODEL", "").strip()
        if not model:
            return None
        base = os.getenv("EMBED_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        return cls(model, base, os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY", ""))

    def fit(self, corpus: list[str]) -> "APIEmbedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(f"{self.base_url}/embeddings", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        vecs = np.array([row["embedding"] for row in sorted(data["data"], key=lambda r: r["index"])], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)


class VectorIndex:
    def __init__(self, ids: list[str], vectors: np.ndarray, embedder_name: str):
        self.ids, self.vectors, self.embedder_name = ids, vectors, embedder_name

    @classmethod
    def build(cls, ids: list[str], texts: list[str], embedder) -> "VectorIndex":
        embedder.fit(texts)
        return cls(ids, embedder.embed(texts), embedder.name)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        sims = self.vectors @ query_vec
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, ids=np.array(self.ids), vectors=self.vectors, embedder=np.array(self.embedder_name))

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        data = np.load(path, allow_pickle=False)
        return cls([str(x) for x in data["ids"]], data["vectors"], str(data["embedder"]))
