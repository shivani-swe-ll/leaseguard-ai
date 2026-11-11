import pytest

from leaseguard.guardrails import citations, citations_valid, numbers_not_in, quote_in
from leaseguard.llm import LLMClient, extract_json
from leaseguard.models import Answer
from leaseguard.qa import Assistant, format_answer
from leaseguard.report import check_lease_file


def test_numbers_not_in():
    src = "NSF admin charge of no more than $20. Notice 24 hours."
    assert numbers_not_in("The admin fee is at most $20 [R09].", src) == []
    assert numbers_not_in("The admin fee is at most $25 [R09].", src) == [25.0]
    assert numbers_not_in("There are 2 rules.", src) == []            # tiny counts ignored


def test_quote_and_citations():
    assert quote_in("late  fee", "A Late fee applies")
    assert not quote_in("", "anything")
    assert citations("Yes [R05] and [clause 3].") == ["R05", "clause 3"]
    assert citations_valid("Yes [R05].", {"R05"}) == (True, [])
    assert citations_valid("Yes [R99].", {"R05"}) == (False, ["R99"])
    assert citations_valid("Yes.", {"R05"})[0] is False                # no citation at all


def test_rules_mode_answer_cites_the_rule(retriever):
    ans = Assistant(retriever).ask("Can my landlord ask for a damage deposit?")
    assert ans.source == "rules" and ans.citations[0].ref == "R05" and "[R05]" in ans.text


@pytest.mark.parametrize("q", ["What are the rules for commercial leases?", "How do I evict a tenant in Alberta?",
                               "What is the best pizza in Toronto?"])
def test_abstains(retriever, q):
    ans = Assistant(retriever).ask(q)
    assert ans.abstained and ans.source == "abstained"


def test_answer_mentions_users_lease(retriever):
    rep = check_lease_file("evals/sample_leases/problem_lease.txt")
    ans = Assistant(retriever).ask("Is a no pets clause allowed?", report=rep)
    assert "[clause 6]" in ans.text


def test_llm_answer_accepted_when_cited_and_grounded(retriever, fake_llm):
    llm = fake_llm("No. Damage deposits are not allowed in Ontario [R05].")
    ans = Assistant(retriever, llm=llm).ask("Can my landlord ask for a damage deposit?")
    assert ans.source == "llm" and ans.trace["passed"]


def test_llm_answer_rejected_without_citation(retriever, fake_llm):
    ans = Assistant(retriever, llm=fake_llm("No, they cannot.")).ask("Can my landlord ask for a damage deposit?")
    assert ans.source == "llm_rejected" and "[R05]" in ans.text


def test_llm_answer_rejected_with_invented_number(retriever, fake_llm):
    llm = fake_llm("They can charge up to $500 [R05].")
    ans = Assistant(retriever, llm=llm).ask("Can my landlord ask for a damage deposit?")
    assert ans.source == "llm_rejected" and 500.0 in ans.trace["ungrounded_numbers"]


def test_llm_answer_rejected_with_unknown_citation(retriever, fake_llm):
    ans = Assistant(retriever, llm=fake_llm("Not allowed [R13].")).ask("Can my landlord ask for a damage deposit?")
    assert ans.source == "llm_rejected"


def test_llm_error_falls_back(retriever, fake_llm):
    ans = Assistant(retriever, llm=fake_llm(RuntimeError("down"))).ask("Can my landlord ask for a damage deposit?")
    assert ans.source == "llm_error" and ans.citations


def test_streaming_yields_pieces_then_checked_answer(retriever, fake_llm):
    llm = fake_llm("No. Damage deposits are not allowed [R05].")
    out = list(Assistant(retriever, llm=llm).stream("Can my landlord ask for a damage deposit?"))
    pieces, final = out[:-1], out[-1]
    assert all(isinstance(p, str) for p in pieces) and len(pieces) > 1
    assert isinstance(final, Answer) and final.source == "llm"


def test_streaming_bad_answer_is_replaced(retriever, fake_llm):
    out = list(Assistant(retriever, llm=fake_llm("Yes, $900 is fine.")).stream("Can my landlord ask for a damage deposit?"))
    assert out[-1].source == "llm_rejected"


def test_streaming_without_llm(retriever):
    out = list(Assistant(retriever).stream("Can my landlord ask for a damage deposit?"))
    assert isinstance(out[0], str) and out[-1].source == "rules"


def test_format_answer(retriever):
    text = format_answer(Assistant(retriever).ask("Is a key deposit allowed?"))
    assert "Sources:" in text and "not legal advice" in text


# ---------- real HTTP client against a local fake server ----------
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_complete_formats(fake_server, provider):
    url, state = fake_server
    client = LLMClient(provider, "m", "key", url)
    assert client.complete("sys", "hi").text == "Hello from the fake model [R05]."
    req = state["requests"][0]
    if provider == "anthropic":
        assert req["path"] == "/messages" and req["body"]["system"] == "sys" and req["headers"]["x-api-key"] == "key"
    else:
        assert req["path"] == "/chat/completions" and req["headers"]["authorization"] == "Bearer key"
    assert client.usage["calls"] == 1 and client.usage["input_tokens"] == 7


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_stream_formats(fake_server, provider):
    url, state = fake_server
    pieces = list(LLMClient(provider, "m", "key", url).stream("sys", "hi"))
    assert "".join(pieces) == "Hello from the fake model [R05]." and len(pieces) == 2
    assert state["requests"][0]["body"]["stream"] is True


def test_from_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert LLMClient.from_env() is None
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1/")
    assert LLMClient.from_env().base_url == "http://localhost:11434/v1"


def test_extract_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
