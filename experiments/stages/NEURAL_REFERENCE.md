# Neural reference: how close is EBR to a cross-encoder reranker?

EBR is a **no-embedding** method: Porter BM25 first hop, explicit bridge entities,
per-entity coverage, fixed RRF. It contains no neural component.

This document adds exactly one neural component, purely as a **ruler**:

> a pretrained cross-encoder, reranking candidate paragraphs.

It is **not part of EBR**. It is not a proposed improvement, not a hybrid, and not
a claim that EBR needs a transformer. The only question it answers is:

> if a RAG-style system spends a neural forward pass per candidate, how much
> better is it than the sparse method, and does EBR's candidate pool help it?

Everything below is on the same deterministic subset, same qrels, same metrics.

## Setup

| item | value |
|---|---|
| data | HotpotQA dev-distractor parquet, 66,581 paragraphs, 7,405 questions |
| split | held-out **test** (`md5(qid) % 5 != 0`) |
| subset | **1,000 qids**, uniform without replacement, seed **1729**, sorted |
| metric cutoff | nDCG@10, both@10, both@5, R@10 (top-10 only, so 100- vs 1000-deep lists are comparable) |
| BM25 | Porter-stemmed, k1=1.2, b=0.75, over all paragraphs |
| EBR | `rrf_porter_bridge_entity`, RRF k=60, 20 seeds, 20 bridges, 5% df cap (pre-registered) |
| reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (BertForSequenceClassification, 6 layers, 384 hidden, 22.7M params) |
| reranker input | `query` vs `"{title}. {paragraph}"`, truncation `max_length=512`, `padding=True` |
| reranker score | `model(**inputs).logits.squeeze(-1)` under `torch.no_grad()` |
| device | **MPS (Apple GPU)**, `torch 2.13.0`, `transformers 5.16.1`; MPS available, **no CPU fallback** |
| batch size | 32 pairs; 147,932 unique pairs scored in total, 147.9 per query |
| pool construction | one union of (Porter top-100 ∪ EBR top-100) per query, so every pair is scored once and both rerankers see identical neural scores |
| model load | 0.39 s |
| wall time | 596 s for 1,000 queries end to end |

Reproduce:

```sh
cd /Users/asmit/Personal/algo-serp-stages
.venv/bin/python experiments/stages/run_hotpot_neural.py            # 1000 qids, seed 1729
.venv/bin/python experiments/stages/run_hotpot_neural.py --n 500    # smaller subset
```

Output: `experiments/stages/results/hotpot_neural_reference.json`.

## Results (1,000 test questions)

| system | nDCG@10 | both@10 | both@5 | R@10 |
|---|---:|---:|---:|---:|
| Porter BM25 | 0.719 | 0.599 | 0.450 | 0.788 |
| **EBR** (Porter + bridge + per-entity, RRF) | **0.746** | **0.707** | **0.519** | **0.844** |
| Cross-encoder over Porter top-100 | 0.818 | 0.717 | 0.608 | 0.856 |
| Cross-encoder over EBR top-100 | **0.828** | **0.754** | **0.624** | **0.875** |

The cross-encoder rows are neural and are **not** part of the method. The subset
reproduces the full-test ordering closely (full test: Porter 0.733/0.611, EBR
0.756/0.717), so the sample is representative.

## Paired bootstrap vs Porter BM25 (1,000 resamples, seed 1729)

| system | metric | Δ | 95% CI | W/T/L |
|---|---|---:|---|---:|
| EBR | nDCG@10 | **+0.0272** | [+0.0171, +0.0368] | 387 / 391 / 222 |
| EBR | both@10 | **+0.1080** | [+0.0820, +0.1370] | 149 / 810 / 41 |
| EBR | both@5 | **+0.0690** | [+0.0430, +0.0950] | 136 / 797 / 67 |
| EBR | R@10 | **+0.0555** | [+0.0425, +0.0700] | 159 / 792 / 49 |
| CE(Porter top-100) | nDCG@10 | **+0.0995** | [+0.0863, +0.1113] | 479 / 419 / 102 |
| CE(Porter top-100) | both@10 | **+0.1180** | [+0.0930, +0.1420] | 140 / 838 / 22 |
| CE(Porter top-100) | both@5 | **+0.1580** | [+0.1320, +0.1840] | 194 / 770 / 36 |
| CE(Porter top-100) | R@10 | **+0.0675** | [+0.0545, +0.0810] | 153 / 824 / 23 |
| CE(EBR top-100) | nDCG@10 | **+0.1090** | [+0.0949, +0.1206] | 507 / 394 / 99 |
| CE(EBR top-100) | both@10 | **+0.1550** | [+0.1310, +0.1800] | 172 / 811 / 17 |
| CE(EBR top-100) | both@5 | **+0.1740** | [+0.1470, +0.2010] | 208 / 758 / 34 |
| CE(EBR top-100) | R@10 | **+0.0865** | [+0.0730, +0.1000] | 185 / 798 / 17 |

Direct neural-vs-neural comparison, identical candidate scoring:

| comparison | metric | Δ | 95% CI | W/T/L |
|---|---|---:|---|---:|
| CE(EBR pool) − CE(Porter pool) | nDCG@10 | +0.0095 | [+0.0061, +0.0128] | 66 / 926 / 8 |
| CE(EBR pool) − CE(Porter pool) | both@10 | +0.0370 | [+0.0250, +0.0500] | 41 / 955 / 4 |
| CE(EBR pool) − CE(Porter pool) | both@5 | +0.0160 | [+0.0070, +0.0240] | 18 / 980 / 2 |
| CE(EBR pool) − CE(Porter pool) | R@10 | +0.0190 | [+0.0125, +0.0255] | 43 / 952 / 5 |

## How much of the neural gain does EBR already have?

Share of the cross-encoder's gain over Porter BM25 that EBR captures on its own:

| metric | EBR Δ | CE(Porter pool) Δ | EBR captures |
|---|---:|---:|---:|
| nDCG@10 | +0.0272 | +0.0995 | **27%** |
| both@10 | +0.1080 | +0.1180 | **92%** |
| both@5 | +0.0690 | +0.1580 | **44%** |
| R@10 | +0.0555 | +0.0675 | **82%** |

This is the central result of the document. On the metric EBR was designed for —
getting **both** supporting paragraphs into the top 10 — the no-embedding method
lands within 0.010 of a cross-encoder reranking BM25's own top-100
(0.707 vs 0.717). On ordinary relevance ordering (nDCG@10) it recovers only about
a quarter of the neural gain.

## Question type

| type | system | nDCG@10 | both@10 | both@5 |
|---|---|---:|---:|---:|
| bridge | Porter | 0.705 | 0.538 | 0.413 |
| bridge | EBR | 0.734 | 0.667 | 0.487 |
| bridge | CE(Porter pool) | 0.789 | 0.658 | 0.533 |
| bridge | CE(EBR pool) | 0.800 | 0.703 | 0.552 |
| comparison | Porter | 0.782 | 0.877 | 0.620 |
| comparison | EBR | 0.804 | 0.888 | 0.665 |
| comparison | CE(Porter pool) | 0.954 | 0.989 | 0.950 |
| comparison | CE(EBR pool) | 0.954 | 0.989 | 0.955 |

Descriptive, no per-type CIs (subsets are ~250 comparison questions). Two things
stand out. On bridge questions EBR's **both@10** (0.667) is above the neural
reranker over BM25's pool (0.658) — the sparse bridge channel finds second facts
that the cross-encoder never sees, because they are outside BM25's top-100. On
comparison questions the cross-encoder is very strong (nDCG 0.954 vs 0.804),
because "were X and Y..." is mostly a relevance-matching problem, not a
second-hop problem.

## Latency (ms/query, same device, single process)

| stage | mean | p50 | p95 |
|---|---:|---:|---:|
| Porter BM25 first hop | 2.5 | 1.7 | 7.4 |
| **EBR full pipeline** | **24.3** | 23.3 | 38.0 |
| EBR marginal over BM25 | 21.7 | 21.2 | 33.6 |
| Cross-encoder rerank (union, 147.9 pairs) | **535.8** | 528.9 | 699.6 |

The cross-encoder is **22× slower** than the entire EBR pipeline at 276 pairs/s,
and that ratio is on an Apple GPU with a 22.7M-parameter model. It also grows with
rerank depth, while EBR's cost is dominated by sparse posting traversal and is
CPU-only. No part of EBR touched the GPU.

## Verdict

**EBR does not match a neural reranker on relevance, but it does match it on the
multi-hop metric, and its pool makes the neural reranker better.**

1. **nDCG@10: the cross-encoder clearly wins.** CE(Porter pool) 0.818 and
   CE(EBR pool) 0.828 vs EBR 0.746. Paired Δ(CE − Porter) = +0.0995
   [+0.0863, +0.1113]; Δ(EBR − Porter) = +0.0272 [+0.0171, +0.0368]. EBR closes
   only 27% of that gap. If the deliverable were "rank paragraphs well", the
   no-embedding method would lose, and this reference says so.
2. **both@10: effectively parity.** EBR 0.707 vs CE(Porter pool) 0.717, with
   EBR's gain over Porter (+0.108) equal to 92% of the neural reranker's gain
   (+0.118). The capability HotpotQA is built to test — putting *both* evidence
   paragraphs in the top 10 — is largely obtained without embeddings, without a
   neural index, and at 1/22 the latency.
3. **The two are complementary, not redundant.** Reranking EBR's pool instead of
   BM25's pool is better for the *same* neural model: +0.0095 nDCG@10
   [+0.0061, +0.0128] and +0.0370 both@10 [+0.0250, +0.0500], on 926 and 955
   ties out of 1,000. EBR's bridge expansion puts second-hop paragraphs into the
   candidate set that BM25's top-100 lacks; a neural reranker can only rerank
   what it is given. This is the useful engineering result: entity bridging is a
   **recall/candidate-generation** mechanism that complements neural scoring
   rather than competing with it.
4. **Cost.** 24 ms vs 536 ms per query, CPU vs GPU, no model weights vs 22.7M
   parameters. On a cost-per-quality-point basis EBR is the better deal for the
   multi-hop metric and the worse deal for plain relevance ordering.

The honest one-line summary: **EBR buys the multi-hop capability that a neural
reranker buys, at about 4% of the latency, but it does not buy the neural
reranker's overall ranking quality, and the remaining nDCG gap is real.**

## Limits

- One dataset, one reranker, one depth (100). A stronger reranker (L-12, or a
  listwise LLM reranker) would widen the nDCG gap; a weaker one would narrow it.
- The cross-encoder sees paragraph text truncated at 512 tokens; HotpotQA
  paragraphs are shorter than that (mean pair length ~155 tokens), so truncation
  is not a factor here.
- Rerank metrics are computed on 100-long lists while BM25/EBR use 1,000-long
  lists. nDCG@10, both@k and R@10 depend only on the top 10, so the comparison is
  fair, but R@100 would not be.
- The reranker is scored on a per-query union of both pools (mean 148 pairs, ~52
  shared). Identical scores feed both rerankers, so the pool comparison is exact.
- Per-type numbers have no confidence intervals; treat them as descriptive.
