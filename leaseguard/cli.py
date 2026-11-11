"""Command line.

    python -m leaseguard rules
    python -m leaseguard check evals/sample_leases/problem_lease.txt
    python -m leaseguard ask "Can my landlord ask for a damage deposit?"
    python -m leaseguard search "landlord comes in whenever he wants"
    python -m leaseguard eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from leaseguard.knowledge import DISCLAIMER, load_rules
from leaseguard.llm import LLMClient
from leaseguard.models import Answer, Verdict

ICON = {"LIKELY_VOID": "[X]", "REVIEW": "[?]", "OK": "[ok]"}


def _llm(args):
    return None if getattr(args, "llm", "auto") == "off" else LLMClient.from_env()


def cmd_rules(args) -> int:
    for r in load_rules():
        print(f"{r.id:<5} {ICON[r.clause_verdict.value]:<5} {r.title}  ({r.section})")
    return 0


def cmd_check(args) -> int:
    from leaseguard.report import check_lease_file, to_json, to_markdown

    llm = _llm(args)
    rep = check_lease_file(args.file, llm=llm)
    c = rep.counts
    print(f"Lease: {rep.source_name}  ({rep.pages} page(s), {len(rep.findings)} clauses)")
    print(f"LLM: {'off (rules + retrieval)' if llm is None else f'{llm.provider}/{llm.model}'}")
    print(f"\n{c['LIKELY_VOID']} likely not allowed   {c['REVIEW']} to check   {c['OK']} OK\n")
    t = rep.terms
    print(f"Rent: {t.monthly_rent or '?'}  Rent deposit: {t.rent_deposit or '?'}  Key deposit: {t.key_deposit or '-'}  "
          f"Start: {t.start_date or '?'}  You pay: {', '.join(t.utilities_tenant_pays) or '-'}\n")
    items = rep.findings if args.all else rep.flagged
    for f in items:
        print(f"{ICON[f.verdict.value]} clause {f.clause.number}: {f.clause.text}")
        if f.verdict != Verdict.OK or args.all:
            print(f"     -> {f.rule_title or 'no rule'} {('(' + f.section + ')') if f.section else ''}"
                  f"  [{f.method}, confidence {f.confidence:.0%}]")
            if f.verdict != Verdict.OK:
                print(f"     why: {f.explanation}")
        print()
    if rep.questions_for_landlord:
        print("Questions to ask your landlord:")
        for i, q in enumerate(rep.questions_for_landlord, 1):
            print(f"  {i}. {q}")
    if args.md:
        Path(args.md).write_text(to_markdown(rep), encoding="utf-8")
        print(f"\nReport saved to {args.md}")
    if args.json:
        Path(args.json).write_text(to_json(rep), encoding="utf-8")
        print(f"JSON saved to {args.json}")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_ask(args) -> int:
    from leaseguard.qa import Assistant, format_answer
    from leaseguard.report import check_lease_file, default_retriever

    report = check_lease_file(args.lease) if args.lease else None
    assistant = Assistant(default_retriever(), llm=_llm(args))
    if args.stream:
        final = None
        for piece in assistant.stream(args.question, report):
            if isinstance(piece, Answer):
                final = piece
            else:
                print(piece, end="", flush=True)
        print("\n")
        if final is None:
            return 1
        if final.source in ("llm", "rules"):
            print("Sources:")
            for c in final.citations:
                print(f"  [{c.ref}] {c.title}" + (f" - {c.section}" if c.section else ""))
            print(f"\n{DISCLAIMER}\n[source: {final.source}]")
        else:   # the streamed text failed the checks (or the LLM failed): show the safe answer instead
            print(f"[{final.source}] The answer above did not pass the checks. Safe answer:\n")
            print(format_answer(final))
        return 0
    ans = assistant.ask(args.question, report)
    print(format_answer(ans))
    print(f"\n[source: {ans.source}]")
    return 0


def cmd_search(args) -> int:
    from leaseguard.report import default_retriever
    from leaseguard.retrieval.hybrid import MODES

    r = default_retriever()
    for mode in MODES:
        hits = r.search(args.text, k=3, mode=mode)
        print(f"{mode:<14} " + "   ".join(f"{h.rule_id} {r.by_id[h.rule_id].title[:28]}" for h in hits))
    return 0


def cmd_eval(args) -> int:
    from leaseguard.evals import run_all

    llm = LLMClient.from_env() if args.with_llm else None
    if args.with_llm and llm is None:
        print("--with-llm needs LLM_PROVIDER and LLM_MODEL in .env", file=sys.stderr)
        return 2
    s = run_all(llm=llm, out_dir=args.out_dir)
    return 0 if s["clauses"]["flagged"]["recall"] >= args.min_recall else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="leaseguard", description="Check Ontario rental leases against tenant law")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("rules", help="list the rules in the knowledge base").set_defaults(func=cmd_rules)

    c = sub.add_parser("check", help="check a lease file (PDF, TXT, MD)")
    c.add_argument("file")
    c.add_argument("--all", action="store_true", help="show every clause, not only flagged ones")
    c.add_argument("--md", help="save a Markdown report")
    c.add_argument("--json", help="save a JSON report")
    c.add_argument("--llm", default="auto", choices=["auto", "off"])
    c.set_defaults(func=cmd_check)

    a = sub.add_parser("ask", help="ask a question about Ontario tenant rules")
    a.add_argument("question")
    a.add_argument("--lease", help="also look at this lease")
    a.add_argument("--stream", action="store_true", help="show the LLM answer as it is written")
    a.add_argument("--llm", default="auto", choices=["auto", "off"])
    a.set_defaults(func=cmd_ask)

    s = sub.add_parser("search", help="compare the 4 search modes on any text")
    s.add_argument("text")
    s.set_defaults(func=cmd_search)

    e = sub.add_parser("eval", help="run all evals")
    e.add_argument("--with-llm", action="store_true")
    e.add_argument("--out-dir", default="evals/results")
    e.add_argument("--min-recall", type=float, default=0.0, help="exit 1 if flagged recall is lower")
    e.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
