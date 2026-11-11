"""REST API (FastAPI).

Run:  uvicorn leaseguard.api:app --reload     Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from leaseguard import __version__
from leaseguard.service import AskIn, CheckFileIn, CheckTextIn, LeaseService

load_dotenv()
app = FastAPI(title="LeaseGuard AI API", version=__version__,
              description="Check Ontario residential leases against tenant law. Educational project, not legal advice.")
service = LeaseService()


def _run(fn, req):
    try:
        return fn(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return service.health()


@app.get("/rules")
def rules() -> list[dict]:
    return service.rules()


@app.post("/check")
def check(req: CheckTextIn) -> dict:
    return _run(service.check_text, req)


@app.post("/check/file")
def check_file(req: CheckFileIn) -> dict:
    return _run(service.check_file, req)


@app.post("/ask")
def ask(req: AskIn) -> dict:
    return _run(service.ask, req)
