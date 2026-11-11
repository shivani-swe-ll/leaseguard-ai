import pytest

from leaseguard.documents import load_pages, pages_from_text, split_clauses
from leaseguard.knowledge import rules_by_id
from leaseguard.models import Verdict

SAMPLES = "evals/sample_leases"


def test_rules_load_and_are_unique(rules):
    assert len(rules) == 32
    assert len({r.id for r in rules}) == 32
    assert all(r.section and r.summary and r.tenant_tip for r in rules)


def test_every_example_triggers_its_own_rule(rules):
    for r in rules:
        for ex in r.examples:
            assert r.matches(ex), f"{r.id} does not match its own example: {ex}"


def test_exclude_patterns_stop_a_trigger(rules):
    by = rules_by_id()
    assert by["R08"].matches("A late fee of $50 applies.")
    assert not by["R08"].matches("If a cheque is returned NSF, a $20 administration fee applies.")
    assert not by["R14"].matches("The landlord may enter without notice in an emergency.")
    assert not by["R19"].matches("The tenant shall not sublet without the landlord's written consent.")


def test_allowed_and_void_rules_have_the_right_verdicts():
    by = rules_by_id()
    assert by["R05"].clause_verdict == Verdict.LIKELY_VOID
    assert by["R07"].clause_verdict == Verdict.OK
    assert by["R17"].clause_verdict == Verdict.REVIEW


def test_split_numbered_clauses_txt():
    clauses = split_clauses(load_pages(f"{SAMPLES}/problem_lease.txt"))
    assert [c.number for c in clauses] == [str(i) for i in range(1, 15)]
    assert clauses[2].text.startswith("At signing")


def test_split_pdf_keeps_page_numbers():
    clauses = split_clauses(load_pages(f"{SAMPLES}/mixed_lease.pdf"))
    assert len(clauses) == 12
    assert {c.page for c in clauses} == {1, 2}


def test_split_paragraphs_when_no_numbers():
    text = "The rent is $1,000 per month.\n\nNo pets are allowed in the apartment.\n\nOk"
    clauses = split_clauses(pages_from_text(text))
    assert [c.number for c in clauses] == ["p1-1", "p1-2"]


def test_wrapped_lines_join_their_clause():
    text = "1. The tenant must give\nninety days notice.\n2. Rent is due on the first."
    clauses = split_clauses(pages_from_text(text))
    assert clauses[0].text == "The tenant must give ninety days notice."


def test_unsupported_file_type(tmp_path):
    f = tmp_path / "lease.docx"
    f.write_text("x")
    with pytest.raises(ValueError):
        load_pages(f)
