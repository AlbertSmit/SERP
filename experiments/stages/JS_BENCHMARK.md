# JS chunk-level benchmark (stage 3)

This is the packing test that matters for SERP: BEIR labels are document-level,
but SERP selects chunks inside a token budget. The Python packer is a sanity check;
this benchmark measures whether packing preserves relevant documents when retrieval
already found them.

## Corpus

- Build `mcp_memory.sqlite` with `python/build_mcp_db.py` (SciFact, first 5 sentence
  chunks per document plus a `_full` chunk that retrieval must exclude).
- Backfill the gram index with `engine.buildGramIndex()`.
- Relevance: a chunk is relevant when its `doc_id` is in the SciFact qrels for the
  query and the chunk id does not end in `_full`.
- Queries: SciFact test queries. This split was used in earlier research, so this
  is a regression/sanity check, not independent evidence.

## Systems (fixed)

| label | retrieval | packing |
|---|---|---|
| `word_truncate` | word BM25 | truncation |
| `hybrid_truncate` | RRF(word BM25, char4 BM25) | truncation |
| `hybrid_greedy` | RRF(word BM25, char4 BM25) | greedy coverage |
| `hybrid_residual` | RRF(word BM25, char4 BM25) | residual |
| `hybrid_p4` | RRF(word BM25, char4 BM25) | P4 marginal gain |
| `p4_current` | current `packContext` | existing P4 + residual |

`hybrid_*` uses `packHybrid(query, { channels: ['word','gram'], packer, budget })`.
`p4_current` is the compatibility path.

## Metrics

- `chunk_recall@budget`: relevant chunks selected / relevant chunks for the query.
- `doc_recall@budget`: relevant docs with at least one selected chunk / relevant docs.
- `tokens_used`: sum of `L` for selected chunks.
- `selected`: number of chunks selected.
- latency: wall-clock ms/query.
- exact-ID suite: exact identifier in top-1/top-10; near-miss above exact; unknown
  identifier activation (any result).
- negative activation: unknown identifier returns any chunk.

## Budgets

`120`, `250`, `500` engine tokens. Report all three; do not choose the best budget
after seeing results.

## Decision rule

Keep a packer only if it preserves at least as many relevant documents as
`hybrid_truncate` at the same budget and does not increase exact-ID near-miss
activation. If `p4_current` does not beat `hybrid_truncate`, P4 is not the product
differentiator; keep it behind a flag and market the budgeted selection instead.
