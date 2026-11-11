# 🛡️ LeaseGuard AI

[![CI](https://github.com/shivani-swe-ll/leaseguard-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/shivani-swe-ll/leaseguard-ai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**An AI agent that reads a rental lease and flags clauses that are likely not allowed under Ontario's Residential Tenancies Act (RTA). Every finding cites the rule and the section it comes from.**

**Live demo:** https://leaseguardai.streamlit.app/

> ⚠️ This is a portfolio project, not legal advice. The rules are plain-language summaries. For real problems, contact the Landlord and Tenant Board, a community legal clinic, or a lawyer.

---

## The problem

Many Ontario tenants sign leases with clauses the law says are **void**: "no pets", damage deposits, late fees, "the landlord may enter at any time", "the tenant pays for all repairs". Most tenants don't know these clauses can't be enforced. LeaseGuard AI reads the lease, checks each clause, and explains in plain English what the law says, with a citation.

## What it does

| Feature | How |
|---|---|
| **Check a lease** (PDF, TXT, paste) | Splits it into clauses and gives each one a verdict: 🔴 likely not allowed, 🟡 check this, 🟢 OK |
| **Key terms** | Pulls out rent, deposits, dates, parties and utilities (regex first, then an optional LLM with a verified quote) |
| **Number checks** | NSF fee above $20 → flagged · rent deposit above one month's rent → flagged |
| **Questions for your landlord** | A ready-made list based on what was flagged |
| **Ask a question** | A RAG chat about tenant rules. Answers cite [R05]-style rules and your own clauses, and are streamed |
| **Refuses out-of-scope questions** | e.g. commercial leases, other provinces, or "pizza recipe" |
| **Law library** | 32 plain-language rules with RTA section numbers |
| **Search lab** | Compare BM25, vector, hybrid and hybrid + rerank search side by side |
| **Evals** | Retrieval, classification, Q&A and whole-lease metrics, run in CI |
| **API + Docker** | FastAPI service, Docker image, health check |

Works **with no API key** (rules + hybrid search). Add a free LLM (Groq, Ollama) to make it smarter.

## Architecture

```
lease (PDF/TXT)
   │  documents.py  → pages → numbered clauses
   ▼
extract.py ───────────────► key terms (rent, deposits, dates)
   │
   ▼   for each clause
retrieval/hybrid.py
   BM25 (keywords) ─┐
   vectors (n-gram  ├─► Reciprocal Rank Fusion ─► rerank ─► top-k rules
   TF-IDF or API)  ─┘
   ▼
classify.py
   1. rule triggers (regex, with exclude patterns)
   2. number checks (NSF fee, deposit size)
   3. optional LLM — must pick a retrieved rule, quote the clause,
      and can't be more severe than the rule allows (checked in code)
   4. otherwise retrieval-only fallback
   ▼
report.py ─► findings + questions ─► Streamlit app / CLI / FastAPI
qa.py     ─► RAG answers + guardrails.py (citations valid? numbers grounded?)
```

**Why both rules and an LLM?** Rule patterns are precise and explainable, but they miss new wording (see the held-out results below). The LLM handles new wording, but it can make things up. So the LLM is only allowed to choose among retrieved rules, and its output is validated by code. If validation fails, the app falls back to the rules answer and says so.

## Evaluation results

Run them yourself: `python3 -m leaseguard eval` (results go to `evals/results/latest.md`). These numbers are **without an LLM**.

**Retrieval (clause → correct rule), 45 clauses**

| Mode | hit@1 | hit@3 | MRR |
|---|---|---|---|
| BM25 | 83% | 93% | 0.89 |
| Vector | 85% | 93% | 0.90 |
| Hybrid (RRF) | 85% | 93% | 0.91 |
| **Hybrid + rerank** | **100%** | **100%** | **1.00** |

**Clause verdicts**

| Set | Accuracy | Flagged recall | Flagged precision |
|---|---|---|---|
| Dev set (45 clauses) | 82% | 100% | 87% |
| **Held-out set** (15 clauses, new wording, never used for tuning) | **53%** | 60% | 100% |

**Q&A:** right source on top 100% · faithfulness 100% · correct refusals 3/3
**Whole sample leases:** 100% of planted problems found, 0 false alarms

### What the evals taught me (honest notes)

- The held-out score (53%) is much lower than the dev score (82%). **The rule patterns overfit to the wording I wrote.** That gap is the main reason the LLM path exists.
- Bugs the evals caught: the "emergency entry" clause was wrongly marked void (fixed with an exclude pattern); a $45 NSF fee was missed because the word "bank" was nearby (narrowed the search window); the keyword boost let generic words win (fixed); "pizza recipe" wasn't refused (question words added to stop words).
- Known miss: "Rent may be increased with sixty days' notice" is marked OK. In Ontario it needs 90 days' written notice. I left it unfixed so the held-out set stays honest.
- The CI fails if flagged recall on the dev set drops below 95%.

## Quick start (Mac / Linux)

```bash
git clone https://github.com/shivani-swe-ll/leaseguard-ai.git
cd leaseguard-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                     # run the tests
streamlit run app/streamlit_app.py            # open the web app
```

### Command line

```bash
python3 -m leaseguard check evals/sample_leases/problem_lease.txt
python3 -m leaseguard check evals/sample_leases/mixed_lease.pdf --md report.md
python3 -m leaseguard ask "Can my landlord charge a late fee?"
python3 -m leaseguard ask "Is a pet clause okay?" --lease evals/sample_leases/problem_lease.txt
python3 -m leaseguard search "landlord can come in anytime"
python3 -m leaseguard rules
python3 -m leaseguard eval
```

## Add an LLM (optional, free options)

Copy `.env.example` to `.env` and fill in one option:

- **Groq** (free tier): `LLM_PROVIDER=openai`, `OPENAI_BASE_URL=https://api.groq.com/openai/v1`
- **Ollama** (runs on your computer, free, no key): `OPENAI_BASE_URL=http://localhost:11434/v1`
- **Anthropic / OpenAI**: paid API keys

On Streamlit Cloud, put the same lines in **App settings → Secrets**. Never commit `.env`.

## REST API

```bash
pip install -r requirements-api.txt
uvicorn leaseguard.api:app --reload
# open http://localhost:8000/docs
```

| Method | Path | Body |
|---|---|---|
| GET | `/health` | none |
| GET | `/rules` | none |
| POST | `/check` | `{"text": "1. No pets are allowed. 2. A $500 damage deposit is required."}` |
| POST | `/check/file` | `{"filename": "lease.pdf", "content_base64": "..."}` |
| POST | `/ask` | `{"question": "Can I be charged a late fee?", "lease_text": null}` |

## Docker

```bash
docker build -t leaseguard .
docker run -p 8000:8000 leaseguard
# or both the API and the web app:
docker compose up
```

## Project structure

```
leaseguard/
  knowledge/rules.json   32 rules: summary, section, tip, keywords, regex triggers
  documents.py           PDF/TXT → pages → clauses
  retrieval/             bm25.py, vectors.py, hybrid.py (RRF + rerank)
  classify.py            clause verdicts (triggers, number checks, LLM, fallback)
  extract.py             key lease terms
  qa.py                  RAG question answering + streaming
  guardrails.py          citation and number checks
  llm.py                 Anthropic / OpenAI-compatible client with streaming
  report.py              full lease report, Markdown/JSON export
  evals.py               all evaluations
  service.py, api.py     FastAPI service
  cli.py                 command line
app/streamlit_app.py     web app
evals/                   labelled clauses, held-out set, questions, sample leases
tests/                   71 tests (a fake LLM and a fake HTTP server, no internet needed)
```

## Tech stack

Python 3.12 · NumPy · Pydantic v2 · pypdf · Streamlit · FastAPI · pytest · Docker · GitHub Actions. BM25 and the vector index are written from scratch (no vector database needed).

## Limitations

- Covers common residential tenancy rules only, not every section of the RTA.
- Scanned (image) PDFs aren't supported, because there's no OCR.
- Without an LLM, unusual wording is often missed (see the held-out score).
- Rules were last reviewed in November 2026. Laws change.

## License

MIT © 2026 Shivani Anghan
