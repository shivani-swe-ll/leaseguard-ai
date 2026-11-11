import json

from leaseguard.classify import ClauseChecker
from leaseguard.extract import extract_terms, regex_terms
from leaseguard.models import Clause, LeaseTerms, Verdict

LEASE = """Landlord: Maple Grove Properties Inc.
Tenant: Alex Sample
Rental unit: Unit 504, 88 Example Avenue, Toronto
Term: 12 months, starting September 1, 2026
1. The rent is $2,350 per month.
2. The tenant will pay a rent deposit of $2,350.
3. A refundable key deposit of $60 is required.
4. Heat is included. The tenant pays for hydro and internet."""


def check(retriever, text, llm=None, terms=None):
    return ClauseChecker(retriever, llm=llm).check(Clause(number="1", text=text), terms)


def test_regex_terms():
    t = regex_terms(LEASE)
    assert (t.landlord, t.tenant) == ("Maple Grove Properties Inc.", "Alex Sample")
    assert (t.monthly_rent, t.rent_deposit, t.key_deposit) == (2350.0, 2350.0, 60.0)
    assert t.start_date == "September 1, 2026"
    assert t.utilities_tenant_pays == ["hydro", "internet"]
    assert "$2,350" in t.evidence["monthly_rent"]


def test_llm_fills_missing_terms_only_with_real_quotes(fake_llm):
    text = "Monthly charge for the apartment: 1,800 dollars.\nLandlord: Jo"
    good = fake_llm(json.dumps({"monthly_rent": {"value": 1800, "quote": "Monthly charge for the apartment: 1,800 dollars"}}))
    assert extract_terms(text, llm=good).monthly_rent == 1800.0
    fake = fake_llm(json.dumps({"monthly_rent": {"value": 1900, "quote": "rent is 1,900"}}))
    assert extract_terms(text, llm=fake).monthly_rent is None          # quote not in the lease
    wrong = fake_llm(json.dumps({"monthly_rent": {"value": 2000, "quote": "Monthly charge for the apartment: 1,800 dollars"}}))
    assert extract_terms(text, llm=wrong).monthly_rent is None         # value not in the quote


def test_trigger_verdicts(retriever):
    assert check(retriever, "A damage deposit of $900 is due at signing.").verdict == Verdict.LIKELY_VOID
    assert check(retriever, "The tenant must have tenant insurance.").verdict == Verdict.OK
    assert check(retriever, "Tenant must give 90 days notice to end the tenancy.").verdict == Verdict.REVIEW


def test_most_severe_rule_wins(retriever):
    f = check(retriever, "Rent is $1,500 per month. A late fee of $25 applies after the 3rd.")
    assert f.verdict == Verdict.LIKELY_VOID and f.rule_id == "R08"


def test_nsf_admin_fee_limit(retriever):
    assert check(retriever, "NSF cheques cost the bank fee plus a $20 admin charge.").verdict == Verdict.OK
    f = check(retriever, "NSF cheques cost the bank fee plus a $45 admin charge.")
    assert f.verdict == Verdict.LIKELY_VOID and f.method == "check"


def test_rent_deposit_above_one_month(retriever):
    terms = LeaseTerms(monthly_rent=1500.0)
    f = check(retriever, "The tenant will pay a rent deposit of $3,000.", terms=terms)
    assert f.verdict == Verdict.LIKELY_VOID and f.rule_id == "R06X"


def test_emergency_entry_is_not_void(retriever):
    assert check(retriever, "The landlord may enter without notice in an emergency.").verdict != Verdict.LIKELY_VOID


def test_unrelated_clause_is_ok(retriever):
    f = check(retriever, "The building has a bicycle room on the ground floor.")
    assert f.verdict == Verdict.OK and f.rule_id is None


def test_llm_path_accepts_a_valid_answer(retriever, fake_llm):
    clause = "Tenant is responsible for fixing all appliances and plumbing problems."
    llm = fake_llm(json.dumps({"rule_id": "R12", "verdict": "LIKELY_VOID",
                               "evidence": "responsible for fixing all appliances", "reason": "Repairs are the landlord's job."}))
    f = check(retriever, clause, llm=llm)
    assert f.verdict == Verdict.LIKELY_VOID and f.method == "llm" and f.rule_id == "R12"


def test_llm_path_rejects_quote_not_in_clause(retriever, fake_llm):
    llm = fake_llm(json.dumps({"rule_id": "R12", "verdict": "LIKELY_VOID", "evidence": "tenant pays all repairs", "reason": "x"}))
    f = check(retriever, "Tenant is responsible for fixing all appliances and plumbing problems.", llm=llm)
    assert f.verdict == Verdict.REVIEW and f.method == "llm_rejected"


def test_llm_cannot_call_an_allowed_rule_void(retriever, fake_llm):
    llm = fake_llm(json.dumps({"rule_id": "A03", "verdict": "LIKELY_VOID", "evidence": "pays for hydro", "reason": "x"}))
    f = check(retriever, "Occupant pays for hydro monthly.", llm=llm)
    assert f.method in ("llm_rejected", "trigger")
    assert f.verdict != Verdict.LIKELY_VOID


def test_llm_rule_must_be_a_retrieved_candidate(retriever, fake_llm):
    llm = fake_llm(json.dumps({"rule_id": "R20", "verdict": "LIKELY_VOID", "evidence": "laundry", "reason": "x"}))
    f = check(retriever, "The laundry room is open from 7 to 11.", llm=llm)
    assert f.method == "llm_rejected"


def test_llm_errors_become_review(retriever, fake_llm):
    f = check(retriever, "The laundry room is open from 7 to 11.", llm=fake_llm("not json at all"))
    assert f.verdict == Verdict.REVIEW and f.method == "llm_rejected"


def test_llm_none_means_ok(retriever, fake_llm):
    llm = fake_llm(json.dumps({"rule_id": "none", "verdict": "OK", "evidence": "", "reason": "No rule applies."}))
    assert check(retriever, "The laundry room is open from 7 to 11.", llm=llm).verdict == Verdict.OK
