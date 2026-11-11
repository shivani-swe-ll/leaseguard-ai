"""Small LLM client using only the Python standard library.

* LLM_PROVIDER=anthropic -> Anthropic Messages API
* LLM_PROVIDER=openai    -> any OpenAI-compatible API (OpenAI, Groq, Ollama at http://localhost:11434/v1, ...)

complete() returns the whole answer; stream() yields text pieces as they arrive
(server-sent events), so the web app can show the answer while it is written.
No LLM configured -> the app runs in rules-only mode.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@dataclass
class LLMClient:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    timeout: int = 60
    usage: dict = field(default_factory=lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})

    @classmethod
    def from_env(cls) -> "LLMClient | None":
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip()
        if not provider or not model:
            return None
        if provider == "anthropic":
            return cls(provider, model, os.getenv("ANTHROPIC_API_KEY", ""), "https://api.anthropic.com/v1")
        if provider == "openai":
            return cls(provider, model, os.getenv("OPENAI_API_KEY", ""),
                       os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"))
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'openai' or 'anthropic'.")

    # ---------- request building ----------
    def _request(self, system: str, user: str, max_tokens: int, stream: bool) -> urllib.request.Request:
        if self.provider == "anthropic":
            url = f"{self.base_url}/messages"
            headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            body = {"model": self.model, "max_tokens": max_tokens, "system": system, "temperature": 0,
                    "messages": [{"role": "user", "content": user}]}
        else:
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if stream:
            body["stream"] = True
        return urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")

    def _open(self, req: urllib.request.Request):
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM request failed ({exc.code}): {exc.read().decode(errors='replace')[:400]}") from exc

    # ---------- whole answer ----------
    def complete(self, system: str, user: str, max_tokens: int = 700) -> LLMResponse:
        start = time.perf_counter()
        with self._open(self._request(system, user, max_tokens, stream=False)) as resp:
            data = json.loads(resp.read().decode())
        if self.provider == "anthropic":
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            u = data.get("usage", {})
            tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
        else:
            text = data["choices"][0]["message"].get("content") or ""
            u = data.get("usage") or {}
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        self._count(tin, tout)
        return LLMResponse(text.strip(), tin, tout, int((time.perf_counter() - start) * 1000))

    # ---------- streaming ----------
    def stream(self, system: str, user: str, max_tokens: int = 700) -> Iterator[str]:
        """Yield text pieces as the model writes them (server-sent events)."""
        with self._open(self._request(system, user, max_tokens, stream=True)) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if self.provider == "anthropic":
                    if event.get("type") == "content_block_delta":
                        piece = event.get("delta", {}).get("text", "")
                        if piece:
                            yield piece
                else:
                    for choice in event.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            yield piece
        self._count(0, 0)

    def _count(self, tin: int, tout: int) -> None:
        self.usage["calls"] += 1
        self.usage["input_tokens"] += tin
        self.usage["output_tokens"] += tout


def extract_json(text: str) -> dict:
    """First JSON object in a model reply (handles ```json fences)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object in model reply")
    return json.loads(m.group(0))
