# Practical guide: choosing a mode

[Back to the README](../README.md).

## Default: BM25 / FTS5

Use this for every query unless you know the query has a different shape.

```js
const chunkIds = engine.packHybrid(query, { budget: 250 }) // packer: 'truncate'
```

`packHybrid` runs word BM25 and the gated in-word 4-gram channel, fuses them
with RRF (`k = 60`), and truncates to the budget. Exact identifiers rank first,
and unknown identifiers never activate retrieval.

The MCP `pack_context` tool defaults to `packHybrid` + truncation
(`mode: "hybrid"`). Pass `mode: "legacy"` for the P4/SubConPack path. The library
method `packContext` remains the legacy P4 path for compatibility. On the chunk
benchmark that path underfills the budget and loses recall to truncation. Prefer
`packHybrid` in library code.

The default coverage gate trades recall for precision: at 250 tokens, gated
truncation is 0.601 document recall / 0.206 document precision, while no-gate
truncation is 0.688 / 0.101. Choose based on whether the caller needs evidence
completeness or precision.

## Deep: EBR, for multi-hop questions only

Use EBR when the caller knows that the answer needs two linked facts ("the
director of the film that …"). It improves complete-evidence recall on
HotpotQA, 2WikiMultiHopQA and MuSiQue. It hurts single-hop queries (SciFact
nDCG −0.044), so do not enable it by default. The text-only router did not
predict which queries benefit, so the caller must choose the mode explicitly.

EBR exists only in the Python lab (`experiments/stages/lab/hotpot.py`). It is
not in the JS engine or the MCP server.

## Structured: CESR-U, when constraints determine the evidence

Use CESR-U when the request names exact constraints that determine the evidence
set, for example "the definition of `X` and every caller of `X`" over a code
graph. It returns the smallest set that satisfies those constraints within the
budget. It is a packing objective, not a search engine:

- Its results rest on oracle constraints and synthetic graph qrels.
- On real PR-derived code search, with constraints extracted from text, it did
  not beat BM25.
- At small budgets it can exceed the budget to satisfy mandatory constraints.

It exists only in the Python lab (`experiments/stages/lab/cesr_u.py`).

## Optional profiles

- **RM3:** enable it per workload only after you measure it there. It helps on
  NFCorpus, SciDocs and TREC-COVID, and hurts on ArguAna and Quora.
- **Cross-encoder reranker:** rerank the fused pool when the latency is
  acceptable. A pool from EBR gives the reranker better candidates than a pool
  from BM25.

## Packing

Keep `packer: 'truncate'`. The `greedy`, `residual` and `p4` packers are
experimental. As configured, `residual` and `p4` stop near 47 tokens, whatever
the budget.

## Python and native tools

`python/sqlite_mcp_engine.py`, `python/benchmark_sqlite_e2e.py` and
`native/fts5_p4_scorer.c` belong to the historical P4 track. Their reported
numbers are in [algorithm.md](algorithm.md) and are superseded.
