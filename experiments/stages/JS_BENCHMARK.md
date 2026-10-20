# JS chunk-level benchmark (stage 3)

This is the packing test that matters for SERP: BEIR labels are document-level,
but SERP selects chunks inside a token budget. The Python packer is a sanity check;
this benchmark measures whether packing preserves relevant documents when retrieval
already found them.

## Corpus

- Build `mcp_memory.sqlite` with `python/build_mcp_db.py` (SciFact, first 5 sentence
  chunks per document plus a `_full` chunk that retrieval must exclude).
- Backfill the gram index with `engine.buildGramIndex()` and the exact-token
  table with `engine.buildExactIndex()`.
- Relevance: a chunk is relevant when its `doc_id` is in the SciFact qrels for the
  query and the chunk id does not end in `_full`.
- Queries: SciFact test queries. This split was used in earlier research, so this
  is a regression/sanity check, not independent evidence.

## Systems (fixed)

| label | retrieval | packing |
|---|---|---|
| `word_truncate` | word BM25 | truncation |
| `word_truncate_nogate` | word BM25 | truncation, no coverage gate |
| `hybrid_truncate` | RRF(word BM25, char4 BM25) | truncation |
| `hybrid_truncate_nogate` | RRF(word BM25, char4 BM25) | truncation, no coverage gate |
| `hybrid_greedy` | RRF(word BM25, char4 BM25) | greedy coverage |
| `hybrid_residual` | RRF(word BM25, char4 BM25) | residual |
| `hybrid_p4` | RRF(word BM25, char4 BM25) | P4 marginal gain |
| `p4_current` | current `packContext` | existing P4 + residual |
| `hybrid_truncate_matched_residual` | RRF(word BM25, char4 BM25) | truncation at `hybrid_residual`'s used-token budget |
| `hybrid_truncate_matched_p4_current` | RRF(word BM25, char4 BM25) | truncation at `p4_current`'s used-token budget |

`hybrid_*` uses `packHybrid(query, { channels: ['word','gram'], packer, budget })`.
`p4_current` is the compatibility path. The `matched_*` rows separate budget
underfill from ranking quality: they re-run truncation at the token budget the
comparison system actually used.

## Metrics

- `chunkRecall`: relevant chunks selected / relevant chunks for the query.
- `chunkPrecision`: relevant chunks selected / chunks selected.
- `docRecall`: relevant docs with at least one selected chunk / relevant docs.
- `docPrecision`: relevant docs selected / docs selected.
- `coverage`: query tokens covered by selected chunks / query tokens.
- `tokens`: sum of `L` for selected chunks.
- `selected`: number of chunks selected.
- latency: wall-clock ms/query.
- exact-ID suite: `positives` built from `chunks_exact_terms` (letter + digit or
  underscore, length >= 4, df <= 20, present in a retrievable non-`_full` chunk;
  ordered by df descending); `exactFound` counts positives whose exact chunk
  appears in the fused top-100 (a miss is counted, never dropped);
  `exactTop1`/`exactTop10`; `nearMissAbove` over positives that have a
  same-prefix, edit-distance-1 exact term; `meanExactRank` over found positives.
- negative activation per group: single `zzq<digits>`, underscore `zzq_42`,
  mixed `zzq_42 p53`, and prose-like `figure_1`/`table_2` when those are not
  exact terms. Activation means an unknown identifier added a result beyond the
  known tokens' baseline (or returned any result when nothing is known).

## Budgets

`120`, `250`, `500` engine tokens. Report all three; do not choose the best budget
after seeing results.

## Results (SciFact test, 300 queries)

Latest frozen run, `results/stage-benchmark.json`. Means over the 300 queries that
have qrels; paired uncertainty is reported in the next section.

| budget | system | doc recall | chunk recall | tokens | picks |
|---:|---|---:|---:|---:|---:|
| 120 | `word_truncate` | 0.554 | 0.201 | 87 | 3.45 |
| 120 | `hybrid_truncate` | 0.563 | 0.210 | 91 | 3.53 |
| 120 | `hybrid_p4` | 0.530 | 0.185 | 91 | 3.34 |
| 120 | `p4_current` | 0.491 | 0.106 | 45 | 1.68 |
| 120 | `hybrid_residual` | 0.486 | 0.107 | 47 | 1.82 |
| 250 | `hybrid_truncate` | 0.601 | 0.257 | 166 | 6.04 |
| 250 | `hybrid_p4` | 0.590 | 0.234 | 166 | 5.82 |
| 250 | `p4_current` | 0.491 | 0.106 | 47 | 1.68 |
| 250 | `hybrid_residual` | 0.486 | 0.107 | 48 | 1.83 |
| 500 | `hybrid_truncate` | 0.632 | 0.278 | 257 | 9.07 |
| 500 | `hybrid_p4` | 0.632 | 0.271 | 257 | 8.97 |
| 500 | `p4_current` | 0.491 | 0.106 | 47 | 1.68 |
| 500 | `hybrid_residual` | 0.486 | 0.107 | 48 | 1.83 |

Removing the coverage gate lifts truncation substantially:

| budget | `hybrid_truncate` docR / docP | `hybrid_truncate_nogate` docR / docP |
|---:|---:|---:|
| 120 | 0.563 / **0.273** | **0.624** / 0.189 |
| 250 | 0.601 / **0.206** | **0.688** / 0.101 |
| 500 | 0.632 / **0.184** | **0.742** / 0.056 |

The gate trades recall for precision. Without it, truncation selects more
relevant documents but also many more irrelevant ones. For evidence-completeness
workloads (the deep mode), the no-gate variant is the better recall ceiling; for
ordinary search, the gate is a precision control. Both differences are
statistically significant in both directions (below).

## Paired bootstrap

`scripts/stage-benchmark.js` retains per-query `docRecall`, `docPrecision`,
`chunkRecall`, `chunkPrecision`, `tokens` and `selected` for every system and
budget, then runs a paired percentile bootstrap over queries: 1,000 resamples,
seed 1729, 95% interval, n = 300. Δ is A − B on the same queries; W/T/L counts
queries where A is higher/tied/lower. `results/stage-benchmark.json` stores the
summary under `paired` (33 comparisons; per-query arrays stay in memory).

Nominal-budget comparisons (`hybrid_truncate` at 120/250/500 tokens):

| A vs B | budget | metric | Δ | 95% CI | W/T/L | CI excludes 0 |
|---|---:|---|---:|---|---:|:---:|
| `hybrid_truncate` vs `p4_current` | 120 | docRecall | +0.072 | [+0.041, +0.107] | 28/270/2 | yes |
| `hybrid_truncate` vs `p4_current` | 120 | docPrecision | −0.101 | [−0.130, −0.070] | 30/178/92 | yes |
| `hybrid_truncate` vs `p4_current` | 250 | docRecall | +0.110 | [+0.077, +0.147] | 40/260/0 | yes |
| `hybrid_truncate` vs `p4_current` | 250 | docPrecision | −0.167 | [−0.202, −0.133] | 34/150/116 | yes |
| `hybrid_truncate` vs `p4_current` | 500 | docRecall | +0.141 | [+0.103, +0.182] | 48/252/0 | yes |
| `hybrid_truncate` vs `p4_current` | 500 | docPrecision | −0.189 | [−0.225, −0.154] | 42/139/119 | yes |
| `hybrid_truncate` vs `hybrid_residual` | 120 | docRecall | +0.078 | [+0.043, +0.115] | 31/265/4 | yes |
| `hybrid_truncate` vs `hybrid_residual` | 120 | docPrecision | −0.066 | [−0.095, −0.038] | 35/184/81 | yes |
| `hybrid_truncate` vs `hybrid_residual` | 250 | docRecall | +0.116 | [+0.078, +0.156] | 44/253/3 | yes |
| `hybrid_truncate` vs `hybrid_residual` | 250 | docPrecision | −0.133 | [−0.165, −0.103] | 37/153/110 | yes |
| `hybrid_truncate` vs `hybrid_residual` | 500 | docRecall | +0.146 | [+0.107, +0.188] | 50/250/0 | yes |
| `hybrid_truncate` vs `hybrid_residual` | 500 | docPrecision | −0.155 | [−0.188, −0.125] | 42/146/112 | yes |
| `hybrid_truncate` vs `hybrid_truncate_nogate` | 120 | docRecall | −0.061 | [−0.092, −0.032] | 2/275/23 | yes |
| `hybrid_truncate` vs `hybrid_truncate_nogate` | 120 | docPrecision | +0.084 | [+0.058, +0.111] | 106/167/27 | yes |
| `hybrid_truncate` vs `hybrid_truncate_nogate` | 250 | docRecall | −0.087 | [−0.122, −0.052] | 4/265/31 | yes |
| `hybrid_truncate` vs `hybrid_truncate_nogate` | 250 | docPrecision | +0.104 | [+0.078, +0.134] | 129/129/42 | yes |
| `hybrid_truncate` vs `hybrid_truncate_nogate` | 500 | docRecall | −0.110 | [−0.152, −0.072] | 6/254/40 | yes |
| `hybrid_truncate` vs `hybrid_truncate_nogate` | 500 | docPrecision | +0.127 | [+0.100, +0.159] | 154/97/49 | yes |
| `hybrid_truncate` vs `hybrid_p4` | 120 | docRecall | +0.033 | [+0.005, +0.062] | 19/274/7 | yes |
| `hybrid_truncate` vs `hybrid_p4` | 250 | docRecall | +0.012 | [−0.008, +0.030] | 9/287/4 | no |
| `hybrid_truncate` vs `hybrid_p4` | 500 | docRecall | +0.000 | [−0.011, +0.010] | 3/294/3 | no |

Matched-token comparisons. `p4_current` and `hybrid_residual` stop at about 46
tokens regardless of the budget, so these rows are identical across budgets;
they compare truncation at the tokens the other system actually spent.

| A vs B | budget | metric | Δ | 95% CI | W/T/L | CI excludes 0 |
|---|---:|---|---:|---|---:|:---:|
| `hybrid_truncate_matched_p4_current` vs `p4_current` | 120 | docRecall | +0.010 | [−0.022, +0.039] | 14/276/10 | no |
| `hybrid_truncate_matched_p4_current` vs `p4_current` | 120 | docPrecision | +0.018 | [−0.004, +0.040] | 26/259/15 | no |
| `hybrid_truncate_matched_p4_current` vs `p4_current` | 250 | docRecall | +0.010 | [−0.022, +0.039] | 14/276/10 | no |
| `hybrid_truncate_matched_p4_current` vs `p4_current` | 250 | docPrecision | +0.018 | [−0.004, +0.040] | 27/258/15 | no |
| `hybrid_truncate_matched_p4_current` vs `p4_current` | 500 | docRecall | +0.010 | [−0.022, +0.039] | 14/276/10 | no |
| `hybrid_truncate_matched_p4_current` vs `p4_current` | 500 | docPrecision | +0.018 | [−0.004, +0.040] | 27/258/15 | no |
| `hybrid_truncate_matched_residual` vs `hybrid_residual` | 120 | docRecall | −0.002 | [−0.037, +0.031] | 15/268/17 | no |
| `hybrid_truncate_matched_residual` vs `hybrid_residual` | 120 | docPrecision | +0.037 | [+0.010, +0.064] | 41/238/21 | yes |
| `hybrid_truncate_matched_residual` vs `hybrid_residual` | 250 | docRecall | −0.002 | [−0.037, +0.031] | 15/268/17 | no |
| `hybrid_truncate_matched_residual` vs `hybrid_residual` | 250 | docPrecision | +0.037 | [+0.010, +0.064] | 41/238/21 | yes |
| `hybrid_truncate_matched_residual` vs `hybrid_residual` | 500 | docRecall | −0.002 | [−0.037, +0.031] | 15/268/17 | no |
| `hybrid_truncate_matched_residual` vs `hybrid_residual` | 500 | docPrecision | +0.037 | [+0.010, +0.064] | 41/238/21 | yes |

**Honest reading.** The headline recall advantage is statistically real at the
nominal budget, but it is mostly budget utilization, not packing quality:
`p4_current` and `hybrid_residual` stop at ~46 tokens at every budget, while
truncation is handed 120/250/500. At matched token spend, truncation versus
`p4_current` is a tie on docRecall (+0.010 [−0.022, +0.039]) and docPrecision
(+0.018 [−0.004, +0.040]), and versus `hybrid_residual` it is a docRecall tie
(−0.002 [−0.037, +0.031]) with a small docPrecision gain (+0.037 [+0.010,
+0.064]). Against `hybrid_p4`, which uses the same RRF retrieval and the same
~257 tokens, truncation is a tie at 500 (0.000 [−0.011, +0.010]), not
significantly different at 250 (+0.012 [−0.008, +0.030]), and ahead only at 120
(+0.033 [+0.005, +0.062]). So truncation's case is not "more relevant documents
per token": it is that truncation spends the budget, is simpler, is never beaten
by P4 at equal tokens, and is slightly more precise than residual at matched
spend. The no-gate result is also confirmed in both directions at every budget:
it significantly raises docRecall and significantly lowers docPrecision. P4
should stay behind the flag because it does not beat truncation at equal tokens
and it underfills a larger budget, not because truncation dominates it on
recall per token.

Exact identifiers: 100 positives, `exactTop1 = 1.000`, `exactTop10 = 1.000`,
`nearMissAbove = 0`; negative activation `0` for single, underscore, mixed and
prose-like groups. This is a synthetic invariant, not a relevance result.

## Decision rule

Keep a packer only if it preserves at least as many relevant documents as
`hybrid_truncate` at the same budget and does not increase exact-ID near-miss
activation. P4 does not beat truncation at equal tokens, and `p4_current`
underfills a larger budget, so it remains behind a flag rather than the default.
The CI evidence does **not** support a claim that truncation preserves more
relevant documents per token; at matched tokens the recall difference against
`p4_current` is not significant. Market budgeted selection, not a recall-per-token
win over P4.
