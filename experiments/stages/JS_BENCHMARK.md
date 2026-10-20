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

Latest frozen run, `results/stage-benchmark.json`. Means; no confidence intervals
yet.

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
ordinary search, the gate is a precision control. Neither choice is free. There
are still no confidence intervals; a paired bootstrap is the next benchmark
improvement.

Exact identifiers: 100 positives, `exactTop1 = 1.000`, `exactTop10 = 1.000`,
`nearMissAbove = 0`; negative activation `0` for single, underscore, mixed and
prose-like groups. This is a synthetic invariant, not a relevance result.

## Decision rule

Keep a packer only if it preserves at least as many relevant documents as
`hybrid_truncate` at the same budget and does not increase exact-ID near-miss
activation. If `p4_current` does not beat `hybrid_truncate`, P4 is not the product
differentiator; keep it behind a flag and market the budgeted selection instead.
