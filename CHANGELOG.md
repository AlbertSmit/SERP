# Changelog

## 1.2.1 — 2026-10-20

Documentation-only correction from the paired EBR-vs-cross-encoder test. No
runtime code changed.

- EBR is statistically indistinguishable from a cross-encoder over BM25's top-100
  on both@10 (−0.010 [−0.037, +0.017]), but significantly behind it over EBR's
  own pool (−0.047 [−0.074, −0.021]) and on nDCG (−0.072 [−0.085, −0.058]).
- The like-for-like CPU cost is 69× (874 ms vs 12.7 ms per query), not the 22×
  mixed GPU/CPU figure. The 22× comparison remains as a GPU reference only.
- Update README, HOTPOT.md, NEURAL_REFERENCE.md and PLAN.md; add the paired-test
  runner and result JSON.

## 1.2.0 — 2026-10-20

Release of the frozen research state. MCP `pack_context` defaults to hybrid
BM25 + 4-gram RRF + truncation; the legacy P4 path is behind `mode: "legacy"`.
Research docs are reconciled with the result JSON, and the research program is
frozen.

## Research findings: repositioning

- Close the search for a universal no-embedding scorer: it was not achieved. P4
  failed its packing gate. A uniform lexical portfolio (macro nDCG 0.4615 vs
  BM25 0.4675) and a learned selector (0.4620) failed non-inferiority.
- BM25/FTS5 is the default ranker. Truncation is the default packer. P4,
  residual and greedy packing are experimental, because they underfill the
  budget and lose recall.
- EBR is a deep multi-hop mode in the Python lab. both@10 +0.105 on HotpotQA;
  all@10 +0.124 on 2Wiki and +0.065 on MuSiQue (paragraph identity). It hurts
  single-hop retrieval. Typed bridges and the text-only router are negative
  results.
- CESR-U is a structured evidence-set packing objective. A fixed text parser now
  plans 77.6% of the synthetic code queries and is oracle-identical on every
  query it can plan; `two_hop` is not plannable because the bridge name is not in
  the text. The results still use synthetic graph qrels and generator-template
  cues, so this validates the plumbing, not paraphrase robustness.
- The code graph and CESR do not beat BM25 on real PR-derived code search. The
  TS 7 AST graph improves correctness, not retrieval.
- RM3 is a per-workload option. The cross-encoder is an optional reranker and
  reference.
- Mark the P4 benchmark tables in `docs/algorithm.md` as historical.

## MCP default

- Change MCP `pack_context` to default to `packHybrid` (BM25 + 4-gram RRF) with
  truncation, and add `mode: "hybrid" | "legacy"`. The legacy P4 path remains
  available via `mode: "legacy"` or the bridge/temporal flags.
- Reinforce access counts for the hybrid path and expose `retrieval_mode` and
  `packer` in the response.
- Rename the MCP server to `serp-context-engine` and update the tool description.
- Keep `packContext` and `fullPipeline` unchanged for library compatibility.
- Add an MCP integration test for the hybrid default and the legacy flag.

## Staged retrieval

- Add `gramTokens` with word-boundary markers and a separate `chunks_grams`
  FTS5 index populated in the same transaction as the word index.
- Add `buildGramIndex` for explicit, batched backfills of existing databases.
- Add `rankWordBm25`, `rankGramBm25`, `fuseRrf`, `retrieveRanked`, `packRanked`,
  `packRankedDetailed`, and `packHybrid`.
- Add `chunks_exact_terms` with `buildExactIndex` and `exactIdentifierTerms`.
  Exact tokens come from `tokenize()` (underscores preserved) instead of unicode61
  phrase matching, so prose `figure 1` no longer makes `figure_1` look indexed.
- Gate the gram channel per token instead of all-or-nothing, so `zzq_42 p53`
  still retrieves via `p53` without activating `zzq_42` variants.
- Gate the word channel the same way: unknown identifiers no longer reach FTS5,
  where unicode61 would split `figure_1` into `figure` + `1` and match prose.
- Scope exact-first fusion to identifier tokens with an exact term; plain years
  and numbers are no longer exact markers.
- Give every packer the same `poolDepth` candidate pool.
- Add `chunkPrecision` and `docPrecision` to the stage benchmark.
- Keep `packContext` and `fullPipeline` behaviour unchanged.

## SubConPack 2.0

- Use positive RSJ information weights, an explicit unknown-token prior, and
  English/Dutch substantive-token filtering.
- Normalize by live mean chunk length and score header, prefix, stem, and
  compound matches explicitly.
- Raise candidate pooling to `max(floor(0.4M), 25)` and add an abstaining
  trigram fallback for vaults of at most 100 chunks. Larger vaults skip trigram
  comparisons entirely.
- Gate packing by weighted coverage with a shaped rare-identifier exception.
- Stop at 8% residual demand, reject zero-gain candidates, and apply a 0.26
  first-marginal dropoff.
- Reuse term-match results within each packing call across scoring, coverage,
  marginal gain, and residual decay.
- Expose `min_coverage` and `rel_dropoff` on `packContext` and `fullPipeline`
  for development evaluation.
