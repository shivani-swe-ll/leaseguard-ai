"""Evaluation harness. Four evals, all reproducible and free (no API key needed):

1. Retrieval ablation  - BM25 vs vector vs hybrid vs hybrid+rerank: hit@1, hit@3, MRR
2. Clause verdicts     - accuracy, precision/recall for "flagged" and "likely void", confusion matrix
3. Q&A                 - right source cited, correct refusals, faithfulness (citations + numbers)
4. Whole leases        - recall/precision of the planted problems in the sample leases
Results: evals/results/latest.md and latest.json. With an LLM set in .env, `--with-llm`
runs 2-3 with the LLM path too.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from leaseguard.classify import ClauseChecker
from leaseguard.guardrails import citations_valid, numbers_not_in
from leaseguard.knowledge import load_rules
from leaseguard.models import Clause, Verdict
from leaseguard.qa import Assistant
from leaseguard.report import check_lease_file
from leaseguard.retrieval.hybrid import MODES, HybridRetriever

EVALS = Path("evals")


def _load(name: str, key: str) -> list[dict]:
    return json.loads((EVALS / name).read_text(encoding="utf-8"))[key]


def retrieval_eval(retriever: HybridRetriever, items: list[tuple[str, set[str]]]) -> dict:
    out = {}
    for mode in MODES:
        h1 = h3 = rr = 0.0
        for text, gold in items:
            ids = [h.rule_id for h in retriever.search(text, k=10, mode=mode)]
            rank = next((i + 1 for i, rid in enumerate(ids) if rid in gold), None)
            h1 += rank == 1
            h3 += rank is not None and rank <= 3
            rr += 1 / rank if rank else 0
        n = len(items)
        out[mode] = {"hit@1": round(h1 / n, 3), "hit@3": round(h3 / n, 3), "mrr": round(rr / n, 3)}
    return out


def clause_eval(checker: ClauseChecker, clauses: list[dict]) -> dict:
    rows, confusion = [], Counter()
    for c in clauses:
        f = checker.check(Clause(number=c["id"], text=c["text"]))
        exp, got = c["verdict"], f.verdict.value
        confusion[(exp, got)] += 1
        rows.append({"id": c["id"], "expected": exp, "got": got, "rule": f.rule_id, "expected_rule": c["rule"],
                     "method": f.method, "ok": exp == got})

    def prf(pred, gold):
        tp = sum(1 for r in rows if pred(r["got"]) and gold(r["expected"]))
        fp = sum(1 for r in rows if pred(r["got"]) and not gold(r["expected"]))
        fn = sum(1 for r in rows if not pred(r["got"]) and gold(r["expected"]))
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        return {"precision": round(p, 3), "recall": round(rc, 3), "f1": round(f1, 3)}

    flagged = lambda v: v != "OK"  # noqa: E731
    void = lambda v: v == "LIKELY_VOID"  # noqa: E731
    return {
        "n": len(rows),
        "accuracy": round(sum(r["ok"] for r in rows) / len(rows), 3),
        "flagged": prf(flagged, flagged),
        "likely_void": prf(void, void),
        "methods": dict(Counter(r["method"] for r in rows)),
        "confusion": {f"{e}->{g}": n for (e, g), n in sorted(confusion.items())},
        "misses": [r for r in rows if not r["ok"]],
    }


def qa_eval(assistant: Assistant, questions: list[dict]) -> dict:
    rows = []
    for q in questions:
        ans = assistant.ask(q["q"])
        if q.get("abstain"):
            ok = ans.abstained
            rows.append({"id": q["id"], "kind": "abstain", "ok": ok, "source": ans.source})
            continue
        top = ans.citations[0].ref if ans.citations else None
        context = "\n".join(c.text for c in ans.citations)
        cites_ok, _ = citations_valid(ans.text, {c.ref for c in ans.citations})
        faithful = cites_ok and not numbers_not_in(ans.text, context + " " + " ".join(c.title for c in ans.citations))
        rows.append({"id": q["id"], "kind": "answer", "ok": (not ans.abstained) and top in q["rules"],
                     "top_source": top, "expected": q["rules"], "faithful": faithful, "source": ans.source})
    answers = [r for r in rows if r["kind"] == "answer"]
    refusals = [r for r in rows if r["kind"] == "abstain"]
    return {
        "answers": len(answers),
        "right_source": round(sum(r["ok"] for r in answers) / len(answers), 3),
        "faithfulness": round(sum(r["faithful"] for r in answers) / len(answers), 3),
        "refusals": len(refusals),
        "correct_refusals": round(sum(r["ok"] for r in refusals) / len(refusals), 3) if refusals else None,
        "misses": [r for r in rows if not r["ok"] or (r["kind"] == "answer" and not r["faithful"])],
    }


def lease_eval(llm=None) -> dict:
    expected = json.loads((EVALS / "sample_leases" / "expected.json").read_text())["leases"]
    out, tp_all, fp_all, fn_all = {}, 0, 0, 0
    for name, exp in expected.items():
        rep = check_lease_file(EVALS / "sample_leases" / name, llm=llm)
        got = {f.clause.number for f in rep.findings if f.verdict != Verdict.OK}
        want = set(exp["flag"])
        tp, fp, fn = len(got & want), len(got - want), len(want - got)
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        out[name] = {"flagged": sorted(got, key=lambda x: int(x) if x.isdigit() else 0), "missed": sorted(want - got),
                     "false_alarms": sorted(got - want)}
    out["recall"] = round(tp_all / (tp_all + fn_all), 3) if tp_all + fn_all else 1.0
    out["precision"] = round(tp_all / (tp_all + fp_all), 3) if tp_all + fp_all else 1.0
    return out


def run_all(llm=None, out_dir: str | Path | None = "evals/results", verbose: bool = True) -> dict:
    rules = load_rules()
    retriever = HybridRetriever(rules)
    clauses = _load("clauses.json", "clauses")
    questions = _load("questions.json", "questions")
    clause_items = [(c["text"], {c["rule"]}) for c in clauses if c["rule"]]
    question_items = [(q["q"], set(q["rules"])) for q in questions if not q.get("abstain")]

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "llm": f"{llm.provider}/{llm.model}" if llm else "off (rules + retrieval only)",
        "rules": len(rules),
        "retrieval_clauses": retrieval_eval(retriever, clause_items),
        "retrieval_questions": retrieval_eval(retriever, question_items),
        "clauses": clause_eval(ClauseChecker(retriever, llm=llm), clauses),
        "clauses_holdout": clause_eval(ClauseChecker(retriever, llm=llm), _load("clauses_holdout.json", "clauses")),
        "qa": qa_eval(Assistant(retriever, llm=llm), questions),
        "leases": lease_eval(llm=llm),
    }
    md = to_markdown(summary)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (Path(out_dir) / "latest.md").write_text(md, encoding="utf-8")
    if verbose:
        print(md)
    return summary


def to_markdown(s: dict) -> str:
    c, q, le = s["clauses"], s["qa"], s["leases"]
    lines = ["# Eval results", "", f"- Run: {s['timestamp']}", f"- LLM: {s['llm']}", f"- Rules in knowledge base: {s['rules']}", ""]
    for title, key, n in [("Retrieval: clause -> rule", "retrieval_clauses", None),
                          ("Retrieval: question -> rule", "retrieval_questions", None)]:
        lines += [f"## {title}", "", "| Mode | hit@1 | hit@3 | MRR |", "|---|---|---|---|"]
        for mode, m in s[key].items():
            lines.append(f"| {mode} | {m['hit@1']:.0%} | {m['hit@3']:.0%} | {m['mrr']:.2f} |")
        lines.append("")
    lines += ["## Clause verdicts", "", f"{c['n']} clauses, accuracy **{c['accuracy']:.0%}**", "",
              "| Target | Precision | Recall | F1 |", "|---|---|---|---|",
              f"| Flagged (not OK) | {c['flagged']['precision']:.0%} | {c['flagged']['recall']:.0%} | {c['flagged']['f1']:.2f} |",
              f"| Likely void | {c['likely_void']['precision']:.0%} | {c['likely_void']['recall']:.0%} | {c['likely_void']['f1']:.2f} |",
              "", f"Decided by: {c['methods']}", "", "Confusion (expected->got): " + ", ".join(f"{k}: {v}" for k, v in c["confusion"].items()), ""]
    if c["misses"]:
        lines.append("Misses:")
        lines += [f"- {m['id']}: expected {m['expected']} ({m['expected_rule']}), got {m['got']} ({m['rule']}, {m['method']})" for m in c["misses"]]
        lines.append("")
    h = s.get("clauses_holdout")
    if h:
        lines += ["## Clause verdicts: held-out set (new wording, never used for tuning)", "",
                  f"{h['n']} clauses, accuracy **{h['accuracy']:.0%}** · flagged recall {h['flagged']['recall']:.0%} · "
                  f"likely-void recall {h['likely_void']['recall']:.0%} · flagged precision {h['flagged']['precision']:.0%}", "",
                  "Misses: " + ", ".join(f"{m['id']} ({m['expected']}->{m['got']})" for m in h["misses"]), ""]
    lines += ["## Q&A", "", f"- Right source on top: **{q['right_source']:.0%}** of {q['answers']} questions",
              f"- Faithfulness (valid citations, no invented numbers): **{q['faithfulness']:.0%}**",
              f"- Correct refusals: **{q['correct_refusals']:.0%}** of {q['refusals']} out-of-scope questions", ""]
    if q["misses"]:
        lines += [f"- miss: {json.dumps(m)}" for m in q["misses"]]
        lines.append("")
    lines += ["## Whole sample leases", "", f"- Recall of planted problems: **{le['recall']:.0%}**",
              f"- Precision: **{le['precision']:.0%}**", ""]
    for name in ("fair_lease.txt", "problem_lease.txt", "mixed_lease.pdf"):
        if name in le:
            d = le[name]
            lines.append(f"- {name}: flagged {d['flagged']}; missed {d['missed']}; false alarms {d['false_alarms']}")
    return "\n".join(lines) + "\n"
