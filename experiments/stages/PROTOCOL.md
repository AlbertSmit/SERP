# Pre-registered protocol: SERP as a lexical recall layer

Frozen before any result is inspected. Deviations are recorded here, not silently applied.

## Hypothesis

BM25 remains the primary ranker. A separate lexical channel recovers relevant
documents BM25 misses. Fixed reciprocal-rank fusion combines them. Residual packing
runs only after retrieval, when a token budget requires selection.

## Systems (fixed)

Retrieval:

- `bm25`: word BM25, k1 1.2, b 0.75 (reference)
- `porter`: Porter-stemmed BM25 (fair baseline; primary comparison)
- `char4`: BM25 over in-word character 4-grams with word-boundary markers
- `vc`: corpus-vetted concept BM25 (prefix + document overlap >= 0.1)
- `vcgrams`: `VC + 0.25 * char4` (the frozen hybrid from `first_principles/hybrid.py`)
- `rrf_porter_char4`: RRF(Porter, char4), k = 60, depth = 1000
- `rrf_porter_vcgrams`: RRF(Porter, VC + grams)
- `rrf_porter_char4_vc`: RRF(Porter, char4, VC)
- `rrf_porter_char4_vcgrams`: RRF(Porter, char4, VC + grams)

Fusion: reciprocal-rank fusion, `k = 60`, no weight tuning.

Packing controls:

- `truncate`: take fused rank order, skip documents that do not fit
- `greedy`: max uncovered query weight per token
- `residual`: residual query-term demand with exponential decay

## Datasets

- `touche`: BEIR Webis-Touche2020 test, 382,545 docs, 49 queries. Never used in
  `first_principles/`. This is the independent labelled check.
- `scifact`: BEIR SciFact. Used before, so it is a regression/sanity check only,
  not evidence for the new corpus claim.

## Metrics

Retrieval:

- nDCG@10, R@100, MRR
- paired delta vs Porter BM25 with 95% bootstrap CI and W/T/L
- stage transition at depth 100: relevant docs found by the new channel but not
  by Porter (`new_relevant`), relevant docs lost (`lost_relevant`), and the union

Packing:

- `recall` = relevant selected / all relevant
- `retention` = relevant selected / relevant in fused top-100
- selected documents and selected relevant documents at the budget

Exact identifiers:

- exact hit rate and exact MRR on synthetic identifier queries
- `near_miss_before_exact`: a spelling variant outranks the exact identifier
- `near_miss_top10`: a spelling variant appears in the top 10
- `negative_activation`: an unknown identifier returns any result in the top 10

## Acceptance rules

The channel is kept only if, on `touche`:

1. `new_relevant > lost_relevant` at depth 100, and
2. the fused system's nDCG@10 is non-inferior to Porter within a pre-declared
   margin of `-0.005`, and
3. R@100 does not fall.

The packer is kept only if, on the chunked JS benchmark:

1. it preserves at least as many relevant documents as truncation at the same
   budget, and
2. it does not increase exact-identifier near-miss activation.

Exact identifiers are a hard invariant, not a trade-off: an exact identifier must
outrank any variant-only match. If a channel violates this, it is gated for tokens
containing digits or underscores.

## Stopping rule

If the channel adds noise rather than recall on `touche`, stop. Do not tune
against `scifact`, the SERP fixtures, or the codebase demo. One frozen run per
system; a bug-fix rerun is allowed and must be reported.

## What is not claimed

This does not test whether residual packing improves answer quality in an LLM.
It tests whether packing preserves retrieval evidence under a budget. That is a
necessary condition, not a sufficient one.
