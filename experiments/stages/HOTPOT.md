# Entity-Bridge Retrieval: a no-embedding multi-hop track

This is the first result of the revised research direction: don't try to out-score
BM25 on bag-of-words. Retrieve **evidence chains** instead.

## Why this direction

The earlier `semantic_lab` work killed corpus PPMI expansion, WordNet synonyms,
relevance feedback and compression as general BM25 replacements. It left one
direction open: a typed relation/evidence index, scored by constrained matching.
It never tested multi-hop retrieval.

HotpotQA is the right falsification target. Each question has **two supporting
paragraphs**. A single-document ranker can score high nDCG while still missing the
second fact. We therefore measure `both@k`: are both supporting paragraphs in the
top k? That is the metric a RAG-style system must beat, not just nDCG@10.

## Method: Entity-Bridge Retrieval (EBR)

No embeddings, no LLM calls, no neural index.

1. **First hop.** Porter BM25 retrieves the top 20 paragraphs.
2. **Bridge extraction.** From those seeds, extract capitalized entity spans and
   title mentions. Drop query entities. Score remaining entities by
   `idf(entity) / (seed_rank + 1)`, keep the top 20, and require `df <= 5%` of the
   corpus so hub entities cannot dominate.
3. **Second hop.** Retrieve every paragraph containing a bridge entity, scored by
   the summed bridge weight. This is a sparse entity-posting channel.
4. **Per-entity coverage.** For each query entity, retrieve paragraphs containing
   it, weighted by entity IDF. This is the comparison-question channel: it makes
   sure both sides of "Were X and Y..." are represented.
5. **Fusion.** Fixed reciprocal-rank fusion, `k = 60`, over Porter BM25, the
   bridge channel, and the per-entity channel. No weights are tuned.

The two added channels are deliberately different. Bridge handles the missing
second hop; per-entity handles the second comparison side. Either alone trades one
metric for another. The combined RRF is the system.

## Data and split

- BEIR-style HotpotQA dev distractor set: 7,405 questions, 66,581 unique
  paragraphs, 2 supporting paragraphs per question.
- 5,918 bridge questions, 1,487 comparison questions.
- Deterministic 20/80 split by MD5 of question id: 1,400 dev, 6,005 test.
- Dev is reported, not used for tuning. All constants were fixed before the test
  run: 20 seeds, 20 bridges, 5% df cap, RRF `k=60`.
- Two settings:
  - **global**: rank all 66,581 paragraphs.
  - **pool**: rerank only the question's 10 distractor paragraphs.

## Results: global pool, held-out test (6,005 questions)

| system | nDCG@10 | both@10 | both@5 | R@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.716 | 0.589 | 0.436 | 0.784 |
| Porter BM25 | 0.733 | 0.611 | 0.463 | 0.797 |
| bridge channel alone | 0.663 | 0.532 | 0.424 | 0.732 |
| per-entity channel alone | 0.552 | 0.340 | 0.241 | 0.599 |
| RRF(Porter, bridge) | 0.720 | 0.652 | 0.502 | 0.807 |
| RRF(Porter, entity) | 0.710 | 0.589 | 0.442 | 0.783 |
| **RRF(Porter, bridge, entity)** | **0.756** | **0.717** | **0.531** | **0.849** |

Paired bootstrap vs Porter BM25, 1,000 resamples:

| metric | Δ | 95% CI | W/T/L |
|---|---:|---|---:|
| nDCG@10 | **+0.0225** | [+0.0184, +0.0266] | 2146 / 2415 / 1444 |
| both@10 | **+0.1052** | [+0.0939, +0.1152] | 901 / 4835 / 269 |
| both@5 | **+0.0673** | [+0.0563, +0.0793] | 777 / 4855 / 373 |
| R@10 | **+0.0526** | [+0.0468, +0.0583] | 953 / 4735 / 317 |

Against unstemmed BM25 the combined system is **+0.1274 both@10** and
**+0.0391 nDCG@10**, both CIs well above zero.

The ablations matter:

- **Bridge alone**: both@10 +0.041, but nDCG **−0.013**. It finds second facts and
  adds noise.
- **Entity alone**: nDCG **−0.023**, both@10 **−0.023**. It is a comparison-channel
  specialist, not a ranker.
- **Combined**: nDCG +0.023 and both@10 +0.105. The channels fix different queries;
  RRF keeps what each gets right. This is the same lesson as the char4 work, but
  now the channel is a multi-hop evidence path, not a spelling variant.

## Results: standard 10-paragraph pool, held-out test

`both@10` is saturated (all 10 paragraphs are returned), so `both@5` is the useful
metric:

| system | nDCG@10 | both@5 |
|---|---:|---:|
| Porter BM25 | 0.823 | 0.533 |
| **RRF(Porter, bridge, entity)** | **0.836** | **0.649** |

Paired vs Porter: nDCG `+0.0135` [+0.0106, +0.0164], both@5 `+0.1152`
[+0.1041, +0.1266]. The gain is not an artifact of the global candidate pool.

## What this establishes

- A no-embedding method can improve **multi-hop supporting-fact retrieval** over a
  strong Porter BM25 baseline, not just nDCG on one document.
- The mechanism is interpretable: first-hop BM25 seeds, explicit bridge entities,
  second-hop entity postings, per-entity coverage, fixed RRF.
- It is not a new scoring formula. It is a new retrieval **methodology** for this
  project, and it targets a capability BM25 structurally lacks.

## Limits and next experiments

- **Entity extraction is untyped and noisy.** Capitalization is a proxy for
  entities; `figure 1`, sentence-initial words and generic noun phrases leak in.
  The next step is typed relations (directed by, born in, member of, located in)
  and bridge validation against the question's relation.
- **One dataset.** HotpotQA is multi-hop by construction. EBR should be gated by
  query type, not applied to every query. The next gate is single-hop regression on
  SciFact/Touche and a router that applies EBR only to bridge/comparison queries.
- **No neural reference yet.** A cross-encoder reranker and SPLADE on the same
  candidate pool would show how close this is to a neural RAG component. That
  comparison is the next experiment, not a claim.
- **The 10-paragraph pool is the standard distractor setting**, but the global
  pool is the stronger test. Both are reported.
- **Prior art**: entity-based multi-hop retrieval, graph-based IR and RRF are
  established families. This is not a field-level novelty claim; it is a measured
  project result and a direction worth pursuing.

## Reproduce

```sh
cd /Users/asmit/Personal/algo-serp-stages
.venv/bin/python experiments/stages/run_hotpot.py
```

Data: `experiments/stages/data/hotpot_dev.parquet`.
Results: `experiments/stages/results/hotpot.json`.
