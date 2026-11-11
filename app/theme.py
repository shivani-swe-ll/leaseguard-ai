"""Look and feel: colors, CSS and small HTML builders for the web app."""

from html import escape

FOREST = "#1F4E3D"
CREAM = "#FBF7F0"
SAND = "#F3ECE1"
LINE = "#EADFCF"
INK = "#2B2118"
MUTED = "#7A6A58"
CLAY = "#C2410C"

# verdict -> (text color, soft background, short label, icon)
VERDICT_STYLE = {
    "LIKELY_VOID": ("#B42318", "#FEE4E2", "Likely not allowed", "⛔"),
    "REVIEW": ("#B54708", "#FEF0C7", "Check this", "⚠️"),
    "OK": ("#067647", "#DCFAE6", "Looks OK", "✅"),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea {{ font-family: 'Inter', -apple-system, sans-serif; }}
h1, h2, h3, .lc-serif {{ font-family: 'Fraunces', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 1.6rem; max-width: 1150px; }}
section[data-testid="stSidebar"] {{ background: {SAND}; border-right: 1px solid {LINE}; }}
.stButton > button[kind="primary"] {{ background: {CLAY}; border: 1px solid {CLAY}; border-radius: 12px; font-weight: 600; padding: 0.5rem 1.4rem; }}
.stButton > button[kind="primary"]:hover {{ background: #9A3412; border-color: #9A3412; }}
.stButton > button[kind="secondary"] {{ border-radius: 999px; border: 1px solid {LINE}; background: #fff; font-size: 0.85rem; }}
div[data-testid="stSegmentedControl"] button {{ border-radius: 999px !important; }}
.lc-hero {{ background: radial-gradient(circle at 92% 10%, rgba(251,247,240,.18) 0 90px, transparent 91px), linear-gradient(135deg, {FOREST} 0%, #2F6B52 65%, #3E7F63 100%); color: {CREAM}; border-radius: 22px; padding: 30px 36px 26px; margin-bottom: 18px; }}
.lc-hero h1 {{ color: {CREAM} !important; font-size: 2.35rem; margin: 0 0 6px; padding: 0; }}
.lc-hero p {{ color: #E3EEE7; font-size: 1.05rem; margin: 0; max-width: 680px; }}
.lc-brand {{ color: #CFE3D7; font-size: .8rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 8px; }}
.lc-chip {{ display: inline-block; background: rgba(251,247,240,.14); border: 1px solid rgba(251,247,240,.35); border-radius: 999px; padding: 4px 12px; margin: 16px 8px 0 0; font-size: .82rem; color: {CREAM}; }}
.lc-card {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 18px 20px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(60,40,20,.05); }}
.lc-card h4 {{ margin: 0 0 4px; font-family: 'Fraunces', Georgia, serif; font-size: 1.08rem; color: {INK}; }}
.lc-find {{ border-left: 6px solid var(--c); }}
.lc-pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .76rem; font-weight: 600; background: var(--bg); color: var(--c); margin-right: 6px; }}
.lc-quote {{ font-family: Georgia, serif; font-style: italic; color: #3B3127; background: {CREAM}; border-radius: 10px; padding: 10px 14px; margin: 10px 0; }}
.lc-tip {{ background: #EEF6F1; border-radius: 10px; padding: 8px 12px; margin-top: 8px; color: {FOREST}; font-size: .92rem; }}
.lc-meta {{ color: {MUTED}; font-size: .78rem; margin-top: 8px; }}
.lc-bar {{ display: flex; height: 14px; border-radius: 999px; overflow: hidden; margin: 10px 0 8px; background: {SAND}; }}
.lc-legend span {{ margin-right: 14px; font-size: .85rem; color: {MUTED}; }}
.lc-big {{ font-family: 'Fraunces', Georgia, serif; font-size: 2.1rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
.lc-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(165px, 1fr)); gap: 10px; }}
.lc-term {{ background: {CREAM}; border: 1px solid {LINE}; border-radius: 12px; padding: 10px 12px; font-weight: 500; color: {INK}; }}
.lc-term b {{ display: block; font-size: .7rem; color: {MUTED}; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; margin-bottom: 2px; }}
.lc-step {{ display: flex; gap: 12px; align-items: flex-start; margin: 10px 0; color: {INK}; font-size: .93rem; }}
.lc-num {{ background: {FOREST}; color: {CREAM}; border-radius: 50%; min-width: 26px; height: 26px; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: .85rem; }}
.lc-stat {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 16px 18px; }}
.lc-stat .v {{ font-family: 'Fraunces', Georgia, serif; font-size: 1.9rem; font-weight: 700; color: {FOREST}; }}
.lc-stat .l {{ color: {MUTED}; font-size: .82rem; }}
.lc-rank {{ display: flex; gap: 8px; align-items: baseline; padding: 6px 0; border-bottom: 1px dashed {LINE}; font-size: .88rem; }}
.lc-rank i {{ color: {MUTED}; font-style: normal; min-width: 18px; }}
.lc-foot {{ color: {MUTED}; font-size: .8rem; text-align: center; margin-top: 30px; }}
</style>
"""


def e(text) -> str:
    return escape(str(text)) if text is not None else ""


def hero(n_rules: int) -> str:
    return (
        '<div class="lc-hero"><div class="lc-brand">🛡️ LeaseGuard AI · Ontario lease review agent</div>'
        '<h1>Know your lease before you sign.</h1>'
        "<p>Upload an Ontario rental lease. Every clause is checked against the Residential Tenancies Act, "
        "and every finding shows the rule and section it comes from.</p>"
        f'<span class="lc-chip">📚 {n_rules} tenant rules</span>'
        '<span class="lc-chip">🔎 Hybrid RAG search</span>'
        '<span class="lc-chip">🧾 Cites the RTA section</span>'
        '<span class="lc-chip">🔒 Works without an AI key</span></div>'
    )


def pill(verdict: str) -> str:
    c, bg, label, icon = VERDICT_STYLE[verdict]
    return f'<span class="lc-pill" style="--c:{c};--bg:{bg}">{icon} {label}</span>'


def summary_card(counts: dict, total: int, name: str) -> str:
    void, review, ok = counts["LIKELY_VOID"], counts["REVIEW"], counts["OK"]
    total = max(total, 1)
    seg = "".join(
        f'<div style="width:{n / total * 100:.1f}%;background:{VERDICT_STYLE[k][0]}"></div>'
        for k, n in (("LIKELY_VOID", void), ("REVIEW", review), ("OK", ok)) if n
    )
    need = void + review
    headline = "Nothing flagged 🎉" if need == 0 else f"{need} of {total} clauses need your attention"
    return (
        f'<div class="lc-card"><div class="lc-meta" style="margin:0">{e(name)}</div>'
        f'<div class="lc-big">{headline}</div><div class="lc-bar">{seg}</div>'
        f'<div class="lc-legend"><span>⛔ {void} likely not allowed</span><span>⚠️ {review} check this</span>'
        f"<span>✅ {ok} looks OK</span></div></div>"
    )


def terms_grid(rows) -> str:
    cells = "".join(f'<div class="lc-term"><b>{e(k)}</b>{e(v) if v else "—"}</div>' for k, v in rows)
    return f'<div class="lc-grid">{cells}</div>'


def finding_card(f) -> str:
    c = VERDICT_STYLE[f.verdict.value][0]
    tip = f'<div class="lc-tip">💡 {e(f.tenant_tip)}</div>' if f.tenant_tip else ""
    matched = f' · matched: “{e(f.evidence)}”' if f.evidence else ""
    why = f.explanation
    lead = f"{f.rule_title} ({f.section}). " if f.rule_title else None
    if lead and why.startswith(lead):
        why = why[len(lead):]  # the title and section are already shown above
    return (
        f'<div class="lc-card lc-find" style="--c:{c}">{pill(f.verdict.value)}'
        f'<span class="lc-meta">Clause {e(f.clause.number)} · page {e(f.clause.page)}</span>'
        f'<div class="lc-quote">“{e(f.clause.text)}”</div>'
        f"<h4>{e(f.rule_title or 'No matching rule')}</h4>"
        f'<div class="lc-meta" style="margin-top:0">{e(f.section or "")}</div>'
        f"<p style='margin:8px 0 0'>{e(why)}</p>{tip}"
        f'<div class="lc-meta">Decided by {e(f.method)} · confidence {f.confidence:.0%}{matched}</div></div>'
    )


def rule_card(r) -> str:
    c = VERDICT_STYLE[r.clause_verdict.value][0]
    example = f'<div class="lc-quote">e.g. “{e(r.examples[0])}”</div>' if r.examples else ""
    return (
        f'<div class="lc-card lc-find" style="--c:{c}">{pill(r.clause_verdict.value)}'
        f'<span class="lc-meta">{e(r.id)} · {e(r.category)}</span>'
        f'<h4 style="margin-top:8px">{e(r.title)}</h4><div class="lc-meta" style="margin-top:0">{e(r.section)}</div>'
        f"<p style='margin:8px 0 0'>{e(r.summary)}</p>{example}"
        f'<div class="lc-tip">💡 {e(r.tenant_tip)}</div></div>'
    )


def steps(items) -> str:
    rows = "".join(f'<div class="lc-step"><div class="lc-num">{i}</div><div>{text}</div></div>'
                   for i, text in enumerate(items, 1))
    return f'<div class="lc-card"><h4>How it works</h4>{rows}</div>'


def stat(value: str, label: str) -> str:
    return f'<div class="lc-stat"><div class="v">{e(value)}</div><div class="l">{e(label)}</div></div>'


def ranking(mode: str, items) -> str:
    rows = "".join(f'<div class="lc-rank"><i>{i}.</i><span><b>{e(rid)}</b> {e(title)}</span></div>'
                   for i, (rid, title) in enumerate(items, 1))
    return f'<div class="lc-card"><h4>{e(mode)}</h4>{rows}</div>'
