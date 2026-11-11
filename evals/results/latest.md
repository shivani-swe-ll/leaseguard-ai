# Eval results

- Run: 2026-11-11T15:34:54
- LLM: off (rules + retrieval only)
- Rules in knowledge base: 32

## Retrieval: clause -> rule

| Mode | hit@1 | hit@3 | MRR |
|---|---|---|---|
| bm25 | 83% | 93% | 0.89 |
| vector | 85% | 93% | 0.90 |
| hybrid | 85% | 93% | 0.91 |
| hybrid_rerank | 100% | 100% | 1.00 |

## Retrieval: question -> rule

| Mode | hit@1 | hit@3 | MRR |
|---|---|---|---|
| bm25 | 87% | 100% | 0.93 |
| vector | 93% | 100% | 0.97 |
| hybrid | 100% | 100% | 1.00 |
| hybrid_rerank | 100% | 100% | 1.00 |

## Clause verdicts

45 clauses, accuracy **82%**

| Target | Precision | Recall | F1 |
|---|---|---|---|
| Flagged (not OK) | 87% | 100% | 0.93 |
| Likely void | 100% | 83% | 0.91 |

Decided by: {'trigger': 30, 'retrieval': 9, 'check': 1, 'none': 5}

Confusion (expected->got): LIKELY_VOID->LIKELY_VOID: 20, LIKELY_VOID->REVIEW: 4, OK->OK: 14, OK->REVIEW: 4, REVIEW->REVIEW: 3

Misses:
- c07: expected LIKELY_VOID (R16), got REVIEW (R16, retrieval)
- c10: expected LIKELY_VOID (R12), got REVIEW (R12, retrieval)
- c16: expected LIKELY_VOID (R01), got REVIEW (R01, retrieval)
- c19: expected LIKELY_VOID (R04), got REVIEW (R04, retrieval)
- c36: expected OK (None), got REVIEW (R11, retrieval)
- c38: expected OK (None), got REVIEW (R01, retrieval)
- c41: expected OK (R12), got REVIEW (R12, retrieval)
- c44: expected OK (R14), got REVIEW (R14, retrieval)

## Clause verdicts: held-out set (new wording, never used for tuning)

15 clauses, accuracy **53%** · flagged recall 60% · likely-void recall 22% · flagged precision 100%

Misses: h02 (LIKELY_VOID->OK), h03 (LIKELY_VOID->OK), h04 (LIKELY_VOID->REVIEW), h05 (LIKELY_VOID->REVIEW), h06 (LIKELY_VOID->OK), h07 (LIKELY_VOID->REVIEW), h14 (LIKELY_VOID->OK)

## Q&A

- Right source on top: **100%** of 15 questions
- Faithfulness (valid citations, no invented numbers): **100%**
- Correct refusals: **100%** of 3 out-of-scope questions

## Whole sample leases

- Recall of planted problems: **100%**
- Precision: **100%**

- fair_lease.txt: flagged []; missed []; false alarms []
- problem_lease.txt: flagged ['1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']; missed []; false alarms []
- mixed_lease.pdf: flagged ['5', '6', '7', '8', '10', '11']; missed []; false alarms []
