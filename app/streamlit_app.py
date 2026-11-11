"""LeaseGuard AI web app.  Run:  pip install -r requirements.txt  &&  streamlit run app/streamlit_app.py"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

import theme as ui  # noqa: E402
from leaseguard.knowledge import DISCLAIMER, load_rules  # noqa: E402
from leaseguard.llm import LLMClient  # noqa: E402
from leaseguard.models import Answer  # noqa: E402
from leaseguard.qa import Assistant  # noqa: E402
from leaseguard.report import check_lease_file, check_lease_text, default_retriever, to_json, to_markdown  # noqa: E402
from leaseguard.retrieval.hybrid import MODES  # noqa: E402

load_dotenv()
st.set_page_config(page_title="LeaseGuard AI", page_icon="🛡️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

SAMPLES = ROOT / "evals" / "sample_leases"
retriever = default_retriever()
rules = load_rules()

st.session_state.setdefault("report", None)
st.session_state.setdefault("chat", [])
st.session_state.setdefault("pending_q", None)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🛡️ LeaseGuard AI")
    st.caption("AI lease review agent · Ontario")
    llm = LLMClient.from_env()
    use_llm = st.toggle("Use AI model (LLM)", value=llm is not None, disabled=llm is None,
                        help="Set LLM_PROVIDER and LLM_MODEL in .env or Streamlit secrets to turn this on.")
    st.caption("Model: " + (f"`{llm.provider} / {llm.model}`" if llm else "off: rules + hybrid search"))
    st.divider()
    st.caption(DISCLAIMER)
active_llm = llm if use_llm else None

# ---------------- Header + navigation ----------------
st.markdown(ui.hero(len(rules)), unsafe_allow_html=True)
PAGES = ["🔍 Check my lease", "💬 Ask a question", "📚 Rule library", "🧪 Search lab", "📊 Evals"]


def pills(label: str, options: list[str], key: str) -> str:
    """Round pill buttons on Streamlit 1.40+, a plain row of radio buttons on older versions."""
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options, default=options[0], key=key, label_visibility="collapsed") or options[0]
    return st.radio(label, options, horizontal=True, key=key, label_visibility="collapsed")


def card(html: str):
    st.markdown(html, unsafe_allow_html=True)


page = pills("Go to", PAGES, key="nav")
st.write("")

# ---------------- Check ----------------
if page == PAGES[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Add your lease")
        source = st.radio("Source", ["Upload a file", "Try a sample", "Paste text"], horizontal=True,
                          label_visibility="collapsed")
        upload = sample = None
        pasted = ""
        if source == "Upload a file":
            upload = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"])
        elif source == "Try a sample":
            sample = st.selectbox("Made-up sample lease", ["problem_lease.txt", "mixed_lease.pdf", "fair_lease.txt"])
        else:
            pasted = st.text_area("Lease text", height=150,
                                  placeholder="1. No pets are allowed.\n2. A damage deposit of $500 is required.")
        go = st.button("Check my lease  →", type="primary")
    with right:
        card(ui.steps([
            "Your lease is split into <b>numbered clauses</b>.",
            "<b>Hybrid search</b> (keywords + vectors) finds the related tenant rules.",
            "Rule patterns, number checks and an optional <b>AI model</b> decide each clause.",
            "Every finding <b>cites the rule</b> and RTA section.",
        ]))

    if go:
        with st.spinner("Reading your lease clause by clause..."):
            if upload is not None:
                with tempfile.TemporaryDirectory() as d:
                    p = Path(d) / Path(upload.name).name
                    p.write_bytes(upload.getvalue())
                    st.session_state.report = check_lease_file(p, llm=active_llm)
            elif sample:
                st.session_state.report = check_lease_file(SAMPLES / sample, llm=active_llm)
            elif pasted.strip():
                st.session_state.report = check_lease_text(pasted, llm=active_llm)
            else:
                st.warning("Please add a lease first.")

    rep = st.session_state.report
    if rep is not None:
        st.divider()
        card(ui.summary_card(rep.counts, len(rep.findings), rep.source_name))

        t = rep.terms
        money = lambda v: f"${v:,.2f}" if v else None  # noqa: E731
        st.markdown("#### Key terms")
        card(ui.terms_grid([
            ("Landlord", t.landlord), ("Tenant", t.tenant), ("Address", t.address),
            ("Monthly rent", money(t.monthly_rent)), ("Rent deposit", money(t.rent_deposit)),
            ("Key deposit", money(t.key_deposit)), ("Start date", t.start_date), ("Term", t.term),
            ("Utilities you pay", ", ".join(t.utilities_tenant_pays) or None),
        ]))

        st.write("")
        main, side = st.columns([3, 2], gap="large")
        with main:
            st.markdown("#### Clauses to look at")
            show = pills("Show", ["All flagged", "⛔ Likely not allowed", "⚠️ Check this"], key="show")
            wanted = {"All flagged": ("LIKELY_VOID", "REVIEW"), "⛔ Likely not allowed": ("LIKELY_VOID",),
                      "⚠️ Check this": ("REVIEW",)}[show]
            items = [f for f in rep.flagged if f.verdict.value in wanted]
            if not rep.flagged:
                st.success("Nothing flagged. Still read the whole lease before you sign.")
            for f in items:
                card(ui.finding_card(f))
        with side:
            if rep.questions_for_landlord:
                st.markdown("#### Questions for your landlord")
                st.caption("Tick them off as you ask.")
                for i, q in enumerate(rep.questions_for_landlord):
                    st.checkbox(q, key=f"q_{i}")
            st.markdown("#### Download")
            st.download_button("⬇️ Report (Markdown)", to_markdown(rep), file_name="leaseguard-report.md",
                               use_container_width=True)
            st.download_button("⬇️ Report (JSON)", to_json(rep), file_name="leaseguard-report.json",
                               mime="application/json", use_container_width=True)
            with st.expander("Every clause (table)"):
                st.dataframe(pd.DataFrame([{"#": f.clause.number, "verdict": f.verdict.label, "rule": f.rule_id,
                                            "text": f.clause.text} for f in rep.findings]),
                             hide_index=True, use_container_width=True)

# ---------------- Ask ----------------
elif page == PAGES[1]:
    st.markdown("### Ask about your rights")
    note = "Answers cite the rules they use" + (", and point to your own clauses." if st.session_state.report
                                                else ". Check a lease first to also get answers about your own clauses.")
    st.caption(note)
    examples = ["Can my landlord ask for a damage deposit?", "Can I be charged a late fee?",
                "How much notice before the landlord enters?"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, key=f"ex_{ex}"):
            st.session_state.pending_q = ex

    for role, text in st.session_state.chat:
        st.chat_message(role, avatar="🧑" if role == "user" else "🏠").markdown(text)
    question = st.chat_input("Type your question...") or st.session_state.pending_q
    st.session_state.pending_q = None
    if question:
        st.chat_message("user", avatar="🧑").markdown(question)
        assistant = Assistant(retriever, llm=active_llm)
        holder = {}

        def pieces():
            for item in assistant.stream(question, st.session_state.report):
                if isinstance(item, Answer):
                    holder["final"] = item
                else:
                    yield item

        with st.chat_message("assistant", avatar="🛡️"):
            streamed = st.write_stream(pieces()) if active_llm else None
            final: Answer = holder.get("final") or assistant.ask(question, st.session_state.report)
            if active_llm is None or final.source != "llm":
                if active_llm is not None:
                    st.warning("The AI answer did not pass the citation and number checks, so here is the rule itself.")
                st.markdown(final.text)
            sources = "\n".join(f"- **[{c.ref}]** {c.title}" + (f" · {c.section}" if c.section else "")
                                for c in final.citations)
            if sources:
                st.markdown("**Sources**\n" + sources)
        shown = streamed if (active_llm and final.source == "llm") else final.text
        st.session_state.chat += [("user", question),
                                  ("assistant", f"{shown}\n\n**Sources**\n{sources}" if sources else shown)]
    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()

# ---------------- Rule library ----------------
elif page == PAGES[2]:
    st.markdown("### Rule library")
    st.caption(f"{len(rules)} plain-language summaries of the Residential Tenancies Act.")
    a, b, c = st.columns([2, 1, 1])
    text = a.text_input("Search rules", placeholder="deposit, pets, entry...")
    cat = b.selectbox("Category", ["All"] + sorted({r.category for r in rules}))
    kind = c.selectbox("Verdict", ["All", "Likely not allowed", "Check this", "Allowed"])
    kind_map = {"Likely not allowed": "LIKELY_VOID", "Check this": "REVIEW", "Allowed": "OK"}
    shown = [r for r in rules
             if (cat == "All" or r.category == cat)
             and (kind == "All" or r.clause_verdict.value == kind_map[kind])
             and (not text or text.lower() in (r.title + r.summary + " ".join(r.keywords)).lower())]
    st.caption(f"Showing {len(shown)} rules")
    cols = st.columns(2, gap="medium")
    for i, r in enumerate(shown):
        with cols[i % 2]:
            card(ui.rule_card(r))

# ---------------- Search lab ----------------
elif page == PAGES[3]:
    st.markdown("### Search lab")
    st.caption("See how the four search methods rank the rules for the same text. Hybrid + rerank is the one the app uses.")
    q = st.text_input("Clause or question", value="The landlord can come in whenever he wants")
    if q:
        cols = st.columns(len(MODES))
        names = {"bm25": "BM25 (keywords)", "vector": "Vectors", "hybrid": "Hybrid (RRF)",
                 "hybrid_rerank": "Hybrid + rerank ⭐"}
        for col, mode in zip(cols, MODES):
            hits = retriever.search(q, k=3, mode=mode)
            col.markdown(ui.ranking(names.get(mode, mode), [(h.rule_id, retriever.by_id[h.rule_id].title) for h in hits]),
                         unsafe_allow_html=True)

# ---------------- Evals ----------------
else:
    st.markdown("### How well does it work?")
    res_json = ROOT / "evals" / "results" / "latest.json"
    res_md = ROOT / "evals" / "results" / "latest.md"
    if res_json.exists():
        d = json.loads(res_json.read_text(encoding="utf-8"))
        st.caption(f"Last run {d['timestamp'][:16].replace('T', ' ')} · {d['llm']}")
        stats = [
            (f"{d['retrieval_clauses']['hybrid_rerank']['hit@1']:.0%}", "Right rule found first (hybrid + rerank)"),
            (f"{d['clauses']['flagged']['recall']:.0%}", "Problem clauses caught (dev set)"),
            (f"{d['clauses_holdout']['accuracy']:.0%}", "Accuracy on held-out new wording"),
            (f"{d['qa']['faithfulness']:.0%}", "Answers with valid citations"),
        ]
        for col, (v, label) in zip(st.columns(4), stats):
            col.markdown(ui.stat(v, label), unsafe_allow_html=True)
        st.write("")
        st.info("The held-out score is lower on purpose: it uses new wording the rules were never tuned on. "
                "That gap is why the optional AI model path exists.", icon="📝")
    if res_md.exists():
        with st.expander("Full results", expanded=False):
            st.markdown(res_md.read_text(encoding="utf-8"))
    else:
        st.write("Run `python3 -m leaseguard eval` to create the results.")

st.markdown('<div class="lc-foot">LeaseGuard AI · portfolio project · not legal advice</div>',
            unsafe_allow_html=True)
