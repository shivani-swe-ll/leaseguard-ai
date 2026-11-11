"""Service layer: all feature logic for the API, testable without a web server."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from leaseguard import __version__
from leaseguard.knowledge import DISCLAIMER, load_rules
from leaseguard.llm import LLMClient
from leaseguard.qa import Assistant
from leaseguard.report import check_lease_file, check_lease_text, default_retriever, to_markdown


class CheckTextIn(BaseModel):
    text: str = Field(..., min_length=20, max_length=200_000, examples=["1. No pets are allowed.\n2. A damage deposit of $500 is required."])
    name: str = "pasted lease"


class CheckFileIn(BaseModel):
    filename: str = Field(..., examples=["lease.pdf"])
    content_base64: str = Field(..., description="the file's bytes, base64-encoded (PDF, TXT or MD)")


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["Can my landlord ask for a damage deposit?"])
    lease_text: str | None = Field(None, description="optional: a lease to consider in the answer")


class LeaseService:
    def __init__(self, llm: LLMClient | None = None, use_env_llm: bool = True):
        self.llm = llm if llm is not None else (LLMClient.from_env() if use_env_llm else None)
        self.retriever = default_retriever()
        self.assistant = Assistant(self.retriever, llm=self.llm)

    def health(self) -> dict:
        return {"status": "ok", "version": __version__, "rules": len(load_rules()),
                "llm": f"{self.llm.provider}/{self.llm.model}" if self.llm else None}

    def rules(self) -> list[dict]:
        return [r.model_dump(mode="json", exclude={"triggers", "exclude"}) for r in load_rules()]

    @staticmethod
    def _report_dict(rep) -> dict:
        data = rep.model_dump(mode="json")
        data["counts"] = rep.counts
        data["markdown"] = to_markdown(rep)
        data["disclaimer"] = DISCLAIMER
        return data

    def check_text(self, req: CheckTextIn) -> dict:
        return self._report_dict(check_lease_text(req.text, req.name, llm=self.llm))

    def check_file(self, req: CheckFileIn) -> dict:
        suffix = Path(req.filename).suffix.lower()
        if suffix not in (".pdf", ".txt", ".md"):
            raise ValueError("Use a PDF, TXT or MD file")
        try:
            raw = base64.b64decode(req.content_base64, validate=True)
        except Exception as exc:
            raise ValueError("content_base64 is not valid base64") from exc
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / f"upload{suffix}"
            path.write_bytes(raw)
            rep = check_lease_file(path, llm=self.llm)
        rep.source_name = Path(req.filename).name
        return self._report_dict(rep)

    def ask(self, req: AskIn) -> dict:
        report = check_lease_text(req.lease_text) if req.lease_text else None
        return self.assistant.ask(req.question, report).model_dump(mode="json")
