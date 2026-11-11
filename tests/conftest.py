import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from leaseguard.knowledge import load_rules
from leaseguard.llm import LLMResponse
from leaseguard.retrieval.hybrid import HybridRetriever


@pytest.fixture(scope="session")
def rules():
    return load_rules()


@pytest.fixture(scope="session")
def retriever(rules):
    return HybridRetriever(rules)


class FakeLLM:
    """Pretend LLM: returns a fixed reply (or raises), streams it in pieces, counts calls."""

    provider, model = "fake", "fake-model"

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def complete(self, system, user, max_tokens=700):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(self.reply, 10, 5, 1)

    def stream(self, system, user, max_tokens=700):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        for i in range(0, len(self.reply), 7):
            yield self.reply[i:i + 7]


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def fake_server():
    """A local HTTP server that answers like Anthropic or OpenAI, including streaming (SSE)."""
    state = {"requests": [], "reply": "Hello from the fake model [R05].", "stream": False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "body": body,
                                      "headers": {k.lower(): v for k, v in self.headers.items()}})
            anthropic = self.path.endswith("/messages")
            if self.path.endswith("/embeddings"):
                data = {"data": [{"index": i, "embedding": [1.0, float(i), 0.5]} for i in range(len(body["input"]))]}
                return self._json(data)
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in [state["reply"][:10], state["reply"][10:]]:
                    if anthropic:
                        ev = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": piece}}
                        self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(ev)}\n\n".encode())
                    else:
                        ev = {"choices": [{"delta": {"content": piece}}]}
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                if not anthropic:
                    self.wfile.write(b"data: [DONE]\n\n")
                return
            if anthropic:
                data = {"content": [{"type": "text", "text": state["reply"]}], "usage": {"input_tokens": 7, "output_tokens": 3}}
            else:
                data = {"choices": [{"message": {"content": state["reply"]}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
            self._json(data)

        def _json(self, data):
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", state
    httpd.shutdown()
