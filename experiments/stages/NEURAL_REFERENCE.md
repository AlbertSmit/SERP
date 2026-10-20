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
| device | **MPS (Apple GPU)** for all scored metrics, `torch 2.13.0`, `transformers 5.16.1`; the latency sample in `run_hotpot_neural_paired.py` loads a separate CPU copy |
| batch size | 32 pairs; 147,932 unique pairs scored in total, 147.9 per query |
| pool construction | one union of (Porter top-100 ∪ EBR top-100) per query, so every pair is scored once and both rerankers see identical neural scores |
| model load | 0.39 s |
| wall time | 596 s for 1,000 queries end to end |
| paired test | EBR vs CE directly: per-query metrics, 1,000 paired resamples, seed 1729 |
| CPU timing | separate 100-query sample, both systems on CPU, 8 torch threads, batch 32 |
| rerun check | the paired run reproduces every metric in this document to 7e-6 |

Reproduce:

```sh
cd /Users/asmit/Personal/algo-serp-stages
.venv/bin/python experiments/stages/run_hotpot_neural.py            # 1000 qids, seed 1729
.venv/bin/python experiments/stages/run_hotpot_neural.py --n 500    # smaller subset
.venv/bin/python experiments/stages/run_hotpot_neural_paired.py     # paired test + CPU latency
```

Outputs:

- `experiments/stages/results/hotpot_neural_reference.json` — the tables below.
- `experiments/stages/results/hotpot_neural_reference_paired.json` — per-query
  metrics for all four systems, the paired tests, and the CPU timing sample.

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

## Paired test: EBR against the cross-encoder

The share table further down is built from two separate gain-over-Porter estimates. It
never tested EBR and the cross-encoder against each other. Both answer the same 1,000
queries against the same baseline, so the missing test is a paired one on those
queries: Δ(EBR − CE) per query, 1,000 paired resamples, seed 1729, identical to the
methodology above.

| comparison | metric | Δ | 95% CI | W/T/L |
|---|---|---:|---|---:|
| EBR − CE(Porter top-100) | both@10 | −0.0100 | [−0.0370, +0.0170] | 86 / 818 / 96 |
| EBR − CE(Porter top-100) | nDCG@10 | **−0.0723** | [−0.0852, −0.0581] | 195 / 342 / 463 |
| EBR − CE(Porter top-100) | both@5 | **−0.0890** | [−0.1170, −0.0600] | 75 / 761 / 164 |
| EBR − CE(Porter top-100) | R@10 | −0.0120 | [−0.0265, +0.0025] | 87 / 805 / 108 |
| CE(EBR pool) − CE(Porter pool) | both@10 | **+0.0370** | [+0.0250, +0.0500] | 41 / 955 / 4 |
| CE(EBR pool) − CE(Porter pool) | nDCG@10 | **+0.0095** | [+0.0061, +0.0128] | 66 / 926 / 8 |
| EBR − CE(EBR pool) | both@10 | **−0.0470** | [−0.0740, −0.0210] | 63 / 827 / 110 |
| EBR − CE(EBR pool) | nDCG@10 | **−0.0817** | [−0.0950, −0.0680] | 181 / 336 / 483 |

W/T/L counts queries, so 818 ties on both@10 means the metric is coarse: most queries
put both facts in the top 10 under both systems or under neither, and the test is
decided by the ~180 queries where they differ.

Three readings:

1. **both@10 against CE(Porter top-100): statistically indistinguishable.** −0.0100
   [−0.0370, +0.0170], 86 wins and 96 losses. The CI is ±0.03 wide, so this says
   "the data does not resolve a difference", not "the systems are equal". Anything
   between EBR being 0.037 worse and 0.017 better is compatible.
2. **nDCG@10: the cross-encoder is significantly better.** −0.0723
   [−0.0852, −0.0581] against Porter's pool, −0.0817 against its own pool, 463 and
   483 losses against 195 and 181 wins. This is the same nDCG gap as before, now
   with a paired CI rather than a ratio of two point estimates.
3. **The both@10 parity belongs to the pool, not to the ranking.** Against a
   reranker over EBR's *own* top-100, EBR loses both@10 by 0.0470
   [−0.0740, −0.0210], CI clear of zero. Given the same candidates, the
   cross-encoder orders the two evidence paragraphs into the top 10 significantly
   more often than EBR does. EBR's 0.707 is high because its candidate set contains
   both facts, not because its ordering is as good as the neural model's. That is
   consistent with point 3 of the verdict: EBR is a recall/candidate-generation
   mechanism.

The corollary matters for how the 92% is quoted. It is a comparison of EBR against a
reranker that can only see BM25's top-100, which is the correct product comparison
for "what does the no-embedding stage buy before any reranker" and the wrong one for
"how good is EBR's ordering".

## How much of the neural gain does EBR already have?

Share of the cross-encoder's gain over Porter BM25 that EBR captures on its own,
with the paired test of the difference between the two gains:

| metric | EBR Δ | CE(Porter pool) Δ | EBR captures | Δ(EBR − CE) | 95% CI | share ≠ 100%? |
|---|---:|---:|---:|---:|---|---|
| nDCG@10 | +0.0272 | +0.0995 | **27%** | −0.0723 | [−0.0852, −0.0581] | yes, CE gain larger |
| both@10 | +0.1080 | +0.1180 | **92%** | −0.0100 | [−0.0370, +0.0170] | **no, CI includes 0** |
| both@5 | +0.0690 | +0.1580 | **44%** | −0.0890 | [−0.1170, −0.0600] | yes, CE gain larger |
| R@10 | +0.0555 | +0.0675 | **82%** | −0.0120 | [−0.0265, +0.0025] | **no, CI includes 0** |

The last column is the point of the exercise. A "share captured" number is only
meaningful if the two gains are distinguishable, and on the two metrics where the
share is high they are not. The 92% is compatible with 100% and with about 69%
(`1 + −0.037/0.118`); the 82% on R@10 is likewise not resolvable from 100%. On
nDCG@10 and both@5 the cross-encoder's gain is significantly larger than EBR's, and
the 27% and 44% shares are safe to quote.

On the metric EBR was designed for — getting **both** supporting paragraphs into the
top 10 — the no-embedding method is not distinguishable from a cross-encoder
reranking BM25's own top-100 (0.707 vs 0.717, CI on the difference includes zero). On
ordinary relevance ordering (nDCG@10) it recovers only about a quarter of the neural
gain, and that gap is significant. Both statements are now paired tests rather than
ratios of point estimates.

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

Descriptive, no per-type CIs (subsets are ~250 comparison questions). Three things
stand out. On bridge questions EBR's **both@10** (0.667) is above the neural
reranker over BM25's pool (0.658) — the sparse bridge channel finds second facts
that the cross-encoder never sees, because they are outside BM25's top-100. Give the
same reranker EBR's pool and it reaches 0.703, above EBR: the bridge advantage is
about which candidates exist, not about how EBR orders them, exactly as the paired
test says overall. On comparison questions the cross-encoder is very strong (nDCG
0.954 vs 0.804), because "were X and Y..." is mostly a relevance-matching problem,
not a second-hop problem.

## Latency

The published comparison mixed devices: EBR end to end on CPU against a GPU
rerank-only pass. It is replaced below by a same-device CPU sample, with the GPU
numbers kept only as a reference.

### CPU like-for-like (100-query sample, one process)

Both systems on CPU, 8 torch threads, batch 32, max length 512. The cross-encoder
time includes the candidate construction it needs — a reranker with no candidates
does nothing — and the EBR time includes everything up to the fused order a
truncation packer would consume. No packer is timed on either side.

| stage | p50 | p95 |
|---|---:|---:|
| **EBR end to end** (BM25 → bridges → per-entity → RRF → fused order) | **12.7** | **17.5** |
| BM25 first hop + top-100 construction | 5.1 | 5.8 |
| CE forward over Porter top-100 (100 pairs) | 868 | 1,196 |
| **CE over Porter pool, end to end** (candidates + forward) | **874** | **1,202** |
| CE forward over EBR top-100 (100 pairs) | 1,009 | 1,412 |
| **CE over EBR pool, end to end** | **1,023** | **1,425** |
| CE forward over the union (148.5 pairs) | 1,423 | 2,050 |
| **CE over the union, end to end** (the cost of both CE columns) | **1,435** | **2,063** |

p50 ratios against EBR end to end: **69×** for CE over the Porter pool, 81× over the
EBR pool, 113× for the union. The cross-encoder runs at 113.6 pairs/s on CPU against
276 pairs/s on MPS, so most of its cost is the forward pass; EBR's cost is sparse
posting traversal and does not grow with rerank depth.

Two honest caveats on the numbers:

- EBR measures 12.7 ms p50 here but 23.3 ms p50 in the original run, same code and
  same machine. Timing EBR alone against timing it while an MPS rerank loop runs
  moves it by 0.6 ms, so GPU contention is not the explanation; it is machine or
  thermal state. Every ratio above pairs measurements from the same pass, and the
  original 23.3 ms is not used in any ratio.
- The cross-encoder figure is device-dependent, and so is the ratio. On MPS it
  scored 276 pairs/s for the same batch and input lengths. A side check on this
  machine showed batch 8 running about 20% faster than batch 32 on CPU; the batch
  was left at 32 to match the published configuration rather than tuned.

### GPU reference (original run, not like for like)

| stage | mean | p50 | p95 |
|---|---:|---:|---:|
| Porter BM25 first hop | 2.5 | 1.7 | 7.4 |
| **EBR full pipeline** | **24.3** | 23.3 | 38.0 |
| EBR marginal over BM25 | 21.7 | 21.2 | 33.6 |
| Cross-encoder rerank (union, 147.9 pairs) | **535.8** | 528.9 | 699.6 |

That mixed-device pair gives 22×. It is not like for like, and making it like for
like moves the gap to 69×, not down: on the same device the cross-encoder is more
expensive relative to EBR than the published number suggested. The direction of the
correction is against EBR's convenience claim, which is why it is stated here rather
than dropped.

## Verdict

**EBR does not match a neural reranker on relevance. It is statistically
indistinguishable from one on both@10 when that reranker is limited to BM25's
top-100 — and behind it on both@10 when both use EBR's own pool. Its pool makes the
neural reranker better.**

1. **nDCG@10: the cross-encoder clearly wins.** CE(Porter pool) 0.818 and
   CE(EBR pool) 0.828 vs EBR 0.746. Paired Δ(CE − Porter) = +0.0995
   [+0.0863, +0.1113]; Δ(EBR − Porter) = +0.0272 [+0.0171, +0.0368]; direct
   Δ(EBR − CE) = −0.0723 [−0.0852, −0.0581]. EBR closes only 27% of that gap, and
   because the paired CI excludes zero that share is significantly below 100%. If
   the deliverable were "rank paragraphs well", the no-embedding method would
   lose, and this reference says so.
2. **both@10: statistically indistinguishable from CE over Porter's pool**, with
   one caveat that matters. EBR 0.707 vs CE(Porter pool) 0.717, Δ = −0.0100
   [−0.0370, +0.0170], 86 wins / 96 losses / 818 ties: the data does not resolve a
   difference, which is weaker than parity. Against a reranker over EBR's *own*
   top-100, EBR is significantly behind on the same metric (−0.0470
   [−0.0740, −0.0210]). Given the same candidates, the neural model orders the two
   evidence paragraphs into the top 10 more often. The 0.707 is a candidate-set
   result, not an ordering result.
3. **The two are complementary, not redundant.** Reranking EBR's pool instead of
   BM25's pool is better for the *same* neural model: +0.0095 nDCG@10
   [+0.0061, +0.0128] and +0.0370 both@10 [+0.0250, +0.0500], on 926 and 955
   ties out of 1,000. EBR's bridge expansion puts second-hop paragraphs into the
   candidate set that BM25's top-100 lacks; a neural reranker can only rerank
   what it is given. This is the useful engineering result: entity bridging is a
   **recall/candidate-generation** mechanism that complements neural scoring
   rather than competing with it.
4. **Cost, like for like on CPU.** 874 ms per query for CE(Porter pool) end to end
   (candidates + forward) against 12.7 ms for EBR end to end: **69×**, no model
   weights vs 22.7M parameters. The published 22× mixed a GPU rerank with a CPU
   pipeline and understated the cross-encoder's cost. On a cost-per-quality-point
   basis EBR is the better deal for the multi-hop metric and the worse deal for
   plain relevance ordering.

The honest one-line summary: **EBR buys most of the multi-hop capability that a
neural reranker buys, at roughly 1/70 of the CPU latency, but it does not buy the
reranker's ordering: rerank EBR's own candidates and the cross-encoder is better on
every metric, including both@10, and the nDCG gap is real.**

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
- The paired EBR-vs-CE test is only as informative as both@10's granularity: 818
  of 1,000 queries tie, so the CI spans ±0.03. A metric that resolves partial
  evidence sets (one of two facts, ordered by rank) would test the same question
  with more power.
- The CPU timing sample is 100 queries on one machine, one model, one batch size.
  The ratio is a property of that pairing, not a hardware-independent constant.
- EBR's bridge list is tie-broken by set iteration order, so a process with a
  different `PYTHONHASHSEED` can swap a paragraph or two at the top-100 pool
  boundary. Pools are therefore not byte-identical across processes, but rankings
  and metrics are: on the 10 of 1,000 queries where this happened between two
  processes, all four per-query metrics were identical, and the full rerun
  reproduced the original table to 7e-6.
- Per-type numbers have no confidence intervals; treat them as descriptive.
