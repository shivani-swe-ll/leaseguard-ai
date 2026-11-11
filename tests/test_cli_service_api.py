import base64
from pathlib import Path

import pytest

from leaseguard.cli import main
from leaseguard.service import AskIn, CheckFileIn, CheckTextIn, LeaseService

PROBLEM = "evals/sample_leases/problem_lease.txt"


def test_cli_check_and_reports(tmp_path, capsys):
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    assert main(["check", PROBLEM, "--llm", "off", "--md", str(md), "--json", str(js)]) == 0
    out = capsys.readouterr().out
    assert "likely not allowed" in out and "Questions to ask your landlord" in out
    assert md.exists() and js.exists()


def test_cli_ask_and_stream(capsys):
    assert main(["ask", "Is a key deposit allowed?", "--llm", "off"]) == 0
    assert "[R07]" in capsys.readouterr().out
    assert main(["ask", "Is a key deposit allowed?", "--llm", "off", "--stream"]) == 0
    assert "source: rules" in capsys.readouterr().out


def test_cli_rules_and_search(capsys):
    assert main(["rules"]) == 0 and "R05" in capsys.readouterr().out
    assert main(["search", "damage deposit"]) == 0 and "hybrid_rerank" in capsys.readouterr().out


@pytest.fixture(scope="module")
def service():
    return LeaseService(use_env_llm=False)


def test_service_check_text_and_file(service):
    rep = service.check_text(CheckTextIn(text="1. No pets are allowed.\n2. A damage deposit of $500 is required."))
    assert rep["counts"]["LIKELY_VOID"] == 2 and "markdown" in rep
    raw = base64.b64encode(Path("evals/sample_leases/mixed_lease.pdf").read_bytes()).decode()
    rep2 = service.check_file(CheckFileIn(filename="mixed.pdf", content_base64=raw))
    assert rep2["pages"] == 2 and rep2["source_name"] == "mixed.pdf"


def test_service_validation(service):
    with pytest.raises(ValueError):
        service.check_file(CheckFileIn(filename="lease.exe", content_base64="aGk="))
    with pytest.raises(ValueError):
        service.check_file(CheckFileIn(filename="lease.txt", content_base64="not base64!!"))


def test_service_ask_with_lease(service):
    ans = service.ask(AskIn(question="Is a no pets clause allowed?", lease_text="1. No pets are allowed in the unit."))
    assert ans["citations"][0]["ref"] == "R03"
    assert any(c["ref"] == "clause 1" for c in ans["citations"])


def test_http_endpoints(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from fastapi.testclient import TestClient

    from leaseguard.api import app

    client = TestClient(app)
    assert client.get("/health").json()["rules"] == 32
    assert len(client.get("/rules").json()) == 32
    r = client.post("/check", json={"text": "1. Subletting is strictly prohibited.\n2. Rent is $1,500 per month."})
    assert r.status_code == 200 and r.json()["counts"]["LIKELY_VOID"] == 1
    assert client.post("/check", json={"text": "short"}).status_code == 422
    raw = base64.b64encode(b"1. A late fee of $30 applies after the 5th day of the month.").decode()
    assert client.post("/check/file", json={"filename": "l.txt", "content_base64": raw}).json()["counts"]["LIKELY_VOID"] == 1
    assert client.post("/check/file", json={"filename": "l.exe", "content_base64": raw}).status_code == 422
    assert client.post("/ask", json={"question": "Can my landlord charge a late fee?"}).json()["citations"][0]["ref"] == "R08"
