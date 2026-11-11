import json

from leaseguard.evals import run_all
from leaseguard.models import Verdict
from leaseguard.report import check_lease_file, check_lease_text, to_json, to_markdown

SAMPLES = "evals/sample_leases"


def test_fair_lease_has_no_flags():
    rep = check_lease_file(f"{SAMPLES}/fair_lease.txt")
    assert rep.counts["LIKELY_VOID"] == 0 and rep.counts["REVIEW"] == 0
    assert rep.terms.monthly_rent == 2350.0


def test_problem_lease_flags_and_questions():
    rep = check_lease_file(f"{SAMPLES}/problem_lease.txt")
    assert rep.counts["LIKELY_VOID"] >= 12
    assert rep.flagged[0].verdict == Verdict.LIKELY_VOID
    assert any("pets" in q for q in rep.questions_for_landlord)


def test_pdf_lease_works():
    rep = check_lease_file(f"{SAMPLES}/mixed_lease.pdf")
    assert rep.pages == 2 and rep.terms.monthly_rent == 1750.0
    assert {f.clause.number for f in rep.flagged} == {"5", "6", "7", "8", "10", "11"}


def test_term_check_deposit_above_rent():
    rep = check_lease_text("Rent is $1,000 per month.\n\nThe tenant will pay a rent deposit of $2,000.")
    assert any(f.rule_id == "R06X" for f in rep.flagged)


def test_markdown_and_json_export():
    rep = check_lease_file(f"{SAMPLES}/problem_lease.txt")
    md = to_markdown(rep)
    assert "Questions to ask your landlord" in md and "not legal advice" in md
    data = json.loads(to_json(rep))
    assert data["counts"]["LIKELY_VOID"] == rep.counts["LIKELY_VOID"]


def test_evals_meet_thresholds(tmp_path):
    s = run_all(out_dir=tmp_path, verbose=False)
    assert s["retrieval_clauses"]["hybrid_rerank"]["hit@1"] >= 0.9
    assert s["retrieval_clauses"]["hybrid_rerank"]["mrr"] >= s["retrieval_clauses"]["bm25"]["mrr"]
    assert s["clauses"]["flagged"]["recall"] >= 0.95
    assert s["qa"]["faithfulness"] == 1.0 and s["qa"]["correct_refusals"] == 1.0
    assert s["leases"]["recall"] == 1.0
    assert (tmp_path / "latest.md").exists()
