# Gating Entity-Bridge Retrieval

EBR improves HotpotQA test `both@10` by **+0.105** and nDCG@10 by **+0.023** over
Porter BM25 (`HOTPOT.md`). It is a multi-hop method: the bridge channel spends
retrieval on entities the question does *not* mention, and the per-entity channel
retrieves "documents that name the same thing again". Neither is obviously safe for
a one-fact query. This experiment asks two questions:

1. Does EBR hurt single-hop retrieval? (SciFact, Touche, NFCorpus)
2. Can a fixed text-only router decide per query, and what does the gate cost?

The short answer: EBR does hurt single-hop retrieval, the router limits the damage
to a small statistically significant loss on SciFact, and — the uncomfortable part
— **the router's cues do not identify the queries where EBR helps**. On HotpotQA,
EBR improves the queries the router rejects by the same +0.101 `both@10` as the
ones it accepts (+0.108). The gate is a corpus-type prior, not a benefit selector.

## What was run

| set | docs | queries | split | role |
|---|---:|---:|---|---|
| SciFact | 5,183 | 809 / 300 | train / test | single-hop, dev sanity / evidence |
| Touche (Webis-Touche 2020) | 382,545 | 49 | test | single-hop, argument retrieval |
| NFCorpus | 3,633 | 324 | dev | single-hop, extra check |
| HotpotQA distractor | 66,581 | 1,400 / 6,005 | dev / test | multi-hop, dev sanity / evidence |

Systems: `porter` (Porter BM25), `entity` (per-entity channel alone), `bridge`
(channel alone), `rrf_porter_bridge`, `rrf_porter_entity`,
`rrf_porter_bridge_entity` (EBR), and `routed` (EBR where the router fires, Porter
otherwise). BEIR metrics are nDCG@10, R@100, MRR@10; HotpotQA metrics are
`both@10`, nDCG@10, R@10 (supporting-fact recall).

Constants are the HotpotQA ones, unchanged: 20 BM25 seeds, 20 bridge entities, 5%
document-frequency cap, RRF `k=60`, ranking depth 1,000. Paired 95% CIs are
2,000-resample bootstraps over query deltas. No constant was fitted here. SciFact
train and HotpotQA dev are reported for sanity; SciFact test, Touche test and
HotpotQA test are the evidence.

Router rules were written and frozen before the runs, after inspecting HotpotQA dev
questions and SciFact train claims only. There are no thresholds to tune: four
boolean rules in `lab/router.py`.

## 1. EBR on single-hop corpora: it hurts

SciFact test (300 queries), Δ vs Porter BM25:

| system | nDCG@10 | R@100 | MRR | ΔnDCG | 95% CI | W/T/L |
|---|---:|---:|---:|---:|---|---:|
| Porter BM25 | 0.6838 | 0.9280 | 0.6478 | | | |
| entity alone | 0.2019 | 0.3442 | 0.1810 | **−0.4819** | [−0.5266, −0.4370] | 6/85/209 |
| bridge alone | 0.6244 | 0.7804 | 0.5984 | **−0.0594** | [−0.0791, −0.0403] | 13/222/65 |
| RRF(Porter, bridge) | 0.6583 | 0.9270 | 0.6233 | −0.0255 | [−0.0407, −0.0107] | 24/232/44 |
| RRF(Porter, entity) | 0.5359 | 0.9252 | 0.4992 | −0.1478 | [−0.1823, −0.1157] | 14/169/117 |
| **RRF(Porter, bridge, entity)** | 0.6398 | 0.9237 | 0.6090 | **−0.0440** | [−0.0645, −0.0248] | 27/207/66 |

The same shape on the other two sets:

| set | system | nDCG@10 | R@100 | MRR | ΔnDCG | 95% CI |
|---|---|---:|---:|---:|---:|---|
| SciFact train (809) | Porter | 0.6976 | 0.9326 | 0.6687 | | |
| | entity | 0.2240 | 0.3652 | 0.2026 | −0.4736 | [−0.5027, −0.4415] |
| | bridge | 0.6430 | 0.7683 | 0.6289 | −0.0547 | [−0.0650, −0.0441] |
| | EBR | 0.6552 | 0.9156 | 0.6218 | −0.0425 | [−0.0555, −0.0296] |
| Touche test (49) | Porter | 0.3262 | 0.5647 | 0.5779 | | |
| | entity | 0.0298 | 0.0812 | 0.0751 | −0.2965 | [−0.3583, −0.2362] |
| | bridge | 0.2859 | 0.3319 | 0.6277 | −0.0404 | [−0.0782, −0.0021] |
| | EBR | 0.3479 | 0.5279 | 0.6412 | +0.0217 (ns) | [−0.0125, +0.0547] |
| NFCorpus dev (324) | Porter | 0.2918 | 0.2262 | 0.4931 | | |
| | entity | 0.0256 | 0.0172 | 0.0536 | −0.2662 | [−0.2991, −0.2339] |
| | bridge | 0.2386 | 0.1542 | 0.4792 | −0.0532 | [−0.0678, −0.0391] |
| | EBR | 0.2822 | 0.2427 | 0.4907 | −0.0096 (ns) | [−0.0213, +0.0021] |

What this says:

- **The per-entity channel is not a ranker outside multi-hop.** It drops nDCG by
  0.27–0.48. On a single-hop query the named entity is the *topic*; retrieving every
  document that repeats the name is a topical filter, not evidence. On HotpotQA the
  same channel was also weak alone (−0.023 nDCG) but it fixed comparison questions;
  there is no equivalent to fix here.
- **The bridge channel is mildly negative everywhere** (−0.04 to −0.06 nDCG). It is
  the same mechanism that gives HotpotQA its second facts, and here it mostly
  promotes co-mentioning documents.
- **Fusion inherits the damage.** EBR is −0.044 on SciFact test, and its R@100 change
  is −0.004 with a CI spanning zero — the loss is in the top-10 order, not in the
  candidate set.
- **Touche is the exception and not a win.** EBR's nDCG is +0.022 with a CI through
  zero, MRR +0.063 (ns), but **R@100 falls −0.037 [−0.0628, −0.0110]**: on
  argument retrieval the bridge channel buys shallow order at the cost of deep
  recall. RRF(Porter, bridge) alone is +0.023 nDCG (ns) there.
- **NFCorpus** is the same −0.010 (ns), with R@100 +0.017 [+0.005, +0.029] — the
  recall/order trade again, not a precision gain.

So EBR is not a universal replacement. On SciFact, where the channel damage is
largest, it is **significantly worse** than Porter BM25 (−0.044, CI excludes zero); on
NFCorpus the same direction is present but not significant.

## 2. The router

Four boolean rules over question text (`lab/router.py`), fire = OR:

| rule | fires when |
|---|---|
| `comparison` | a comparison cue ("both", "same", "which of", "versus", "compare", ...) and ≥1 entity |
| `anaphora` | "that/who/whom/whose/which/where/when" anywhere but the first token, and ≥1 entity |
| `whchain` | two or more distinct wh-words in the question |
| `pair_relation` | ≥2 entities and a relation word ("born", "located", "director", "member of", ...) |

Entity detection reuses `hotpot.CAP_RE` and `hotpot.STOP_ENTITIES`, splitting
conjuncts so "X and Y" counts as two entities. Headline-style queries (NFCorpus) are
lowercased first, because title case carries no entity signal.

Fire rates:

| set | split | queries | fire rate | fires per cue |
|---|---|---:|---:|---|
| HotpotQA | dev | 1,400 | **65.4%** | anaphora 599, pair_relation 445, whchain 278, comparison 116 |
| HotpotQA | test | 6,005 | **65.2%** | anaphora 2,548, pair_relation 1,830, whchain 1,055, comparison 546 |
| SciFact | train | 809 | 5.3% | anaphora 39, pair_relation 3, comparison 2 |
| SciFact | test | 300 | 7.0% | anaphora 17, pair_relation 4, comparison 1 |
| Touche | test | 49 | 0.0% | — |
| NFCorpus | dev | 324 | 0.0% | — |

The single-hop sets are sparse in the cues the rules look for: SciFact claims are
declarative science sentences and Touche/NFCorpus queries are short topical
questions without second proper nouns. That is the gate working. (A query can fire
several cues, so the per-cue counts exceed the fire count.)

## 3. Routed on HotpotQA: keeps two thirds of the gain

HotpotQA test (6,005 queries):

| system | nDCG@10 | both@10 | R@10 | both@5 | Δboth@10 vs Porter | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Porter BM25 | 0.7331 | 0.6113 | 0.7967 | 0.4633 | | |
| **routed** | 0.7479 | **0.6816** | 0.8321 | 0.5057 | **+0.0703** | [+0.0609, +0.0796] |
| always-EBR | 0.7556 | 0.7167 | 0.8494 | 0.5306 | +0.1054 | [+0.0943, +0.1161] |

- routed nDCG@10 +0.0148 [+0.0115, +0.0181], R@10 +0.0354 [+0.0308, +0.0403].
  W/T/L on `both@10`: 616 / 5,195 / 194.
- routed vs always-EBR: `both@10` **−0.0351** [−0.0415, −0.0291], nDCG@10 −0.0078.
  The gate throws away a third of the multi-hop gain because it rejects a third of
  the questions.
- Dev reproduces it (fire 65.4%, routed +0.0700, EBR +0.1014), so this is not a
  split artifact.

By question type (test): bridge n=4,780 Porter 0.5500 → routed 0.6358 → EBR 0.6728
`both@10`; comparison n=1,225 Porter 0.8506 → routed 0.8604 → EBR 0.8882. routed
vs Porter is +0.0858 [+0.0749, +0.0969] on bridge and +0.0098 [−0.0065, +0.0261] on
comparison — the gate helps where the multi-hop gain lives, and is neutral where
`both@10` is nearly saturated.

### The gate does not select the right queries

| HotpotQA test subset | queries | Porter both@10 | EBR both@10 | Δ |
|---|---:|---:|---:|---:|
| router fires | 3,918 | 0.6059 | 0.7136 | **+0.1077** [+0.0934, +0.1212] |
| router silent | 2,087 | 0.6215 | 0.7226 | **+0.1011** [+0.0834, +0.1193] |

The 35% of HotpotQA questions the router rejects benefit from EBR almost exactly as
much as the ones it accepts. Individually, every cue's fired subset gains:
`anaphora` +0.129, `whchain` +0.138, `pair_relation` +0.090, `comparison` +0.020
(both@10). The router is therefore **not** a specialist selector. It encodes "this
looks like a multi-hop question", which on HotpotQA is true of nearly every question
and false of nearly every SciFact claim. Its value is that the second condition
holds, not that the first one predicts benefit.

## 4. Routed on single-hop: small, significant loss on SciFact

| set | split | fire rate | Porter nDCG@10 | routed nDCG@10 | Δrouted vs Porter | 95% CI |
|---|---|---:|---:|---:|---:|---|
| SciFact | train | 5.3% | 0.6976 | 0.6911 | **−0.0065** | [−0.0105, −0.0030] |
| SciFact | test | 7.0% | 0.6838 | 0.6788 | **−0.0049** | [−0.0108, −0.0008] |
| Touche | test | 0.0% | 0.3262 | 0.3262 | 0.0000 | identical (0 fires) |
| NFCorpus | dev | 0.0% | 0.2918 | 0.2918 | 0.0000 | identical (0 fires) |

**Routed is not non-inferior to Porter on SciFact.** The CI excludes zero on both
splits: the gate removes ~89% of EBR's damage (−0.044 → −0.005) but not all of it.
Two details make this worse than the number suggests:

- **The false fires land on the easy queries.** Fired SciFact test queries have
  Porter nDCG 0.7702 versus 0.6773 for the silent ones. They are well-formed claims
  with relative clauses — exactly the text where Porter is already strong and where
  EBR has the most to lose: EBR on that subset is **−0.0705** [−0.1363, −0.0129]
  (train: −0.1221). Losses are 5 of 300 on test, 12 of 809 on train.
- **One rule carries it.** `anaphora` accounts for 17 of 21 SciFact test fires
  (−0.065 mean ΔnDCG, 4 losses / 12 ties / 1 win) and `pair_relation` for the other
  harmful one (−0.125, 2 losses / 2 ties). `comparison` fired once and changed
  nothing. Scientific prose uses "that/which/where" as ordinary relative clauses, so
  the anaphora rule is the weak point: it is the highest-yield rule on HotpotQA
  (2,548 fires) and the highest-harm rule on SciFact.

For a deployment on single-hop text this is a 0.7% relative nDCG loss for a 7% fire
rate. Small, but it is a real loss with a CI that excludes zero, and it is caused by
the gate firing, not by noise.

## Verdict

**EBR can be gated, but only in the weak sense: the gate limits exposure, it does
not aim the specialist.**

- **Not a universal replacement.** EBR is −0.044 nDCG@10 [−0.0645, −0.0248] on
  SciFact test and −0.059 for its bridge channel alone; the per-entity channel is
  −0.48 there. On Touche and NFCorpus the nDCG effect is not significant, and neither
  set shows a clean improvement: Touche's R@100 falls significantly (−0.037) while its
  nDCG rises by an amount the CI cannot separate from zero, and NFCorpus trades
  R@100 (+0.017) for an nDCG loss. Nothing in these three sets supports applying EBR
  to single-hop queries.
- **A gated specialist is supported for multi-hop.** Routed is +0.070 `both@10`
  [+0.061, +0.080] and +0.015 nDCG on HotpotQA test, with a router that fires on 65%
  of questions and needs no training.
- **Universal routed is not supported.** On SciFact it is significantly below Porter
  (−0.005), and on HotpotQA it gives back a third of the EBR gain (−0.035 `both@10`).
  Both losses are the price of the same 65%/7% fire split; there is no threshold that
  fixes both, because the cues do not measure benefit.
- **Do not claim query-type adaptation.** The fired/silent split on HotpotQA
  (+0.108 vs +0.101) shows the router separates corpora, not queries. A production
  system should choose the mode from a corpus/workload decision (multi-hop QA vs
  single-document search), not pretend a per-query cue model did it.

What would change the verdict: a second-hop channel that validates bridges against
the question's relation instead of trusting co-mention, so the channel stops
promoting topical neighbours on single-hop text. Then EBR might be safe without a
gate, and the router question disappears. The cheaper intermediate is a stricter
gate that requires an entity *pair* plus a relation or comparison cue, deliberately
giving up HotpotQA coverage (the silent subset shows that coverage is worth having);
it is untested here and would need the same fixed-split protocol.

## Limits

- **Touche is 49 queries.** Its CIs are wide enough that a ±0.02 nDCG effect cannot
  be resolved; only the R@100 loss is significant. Touche also has zero router fires,
  so the single-hop gate evidence is essentially SciFact (7%) plus NFCorpus (0%).
- **The single-hop corpora are biomedical and debate text**, not web queries.
  SciFact claims are full sentences with relative clauses, which is why `anaphora`
  misfires. Short web-style queries ("weather in Utrecht") would fire none of the
  rules, but that is an expectation, not a measurement.
- **No stopword list in the baseline**, as in all prior work here; absolute nDCG is
  below Anserini's Porter BM25.
- **Cost.** The Touche entity index is 1.0M entities over 382k documents, built in
  ~110 s with ~10 GB peak RSS (dominated by the existing harness index and its
  pickle). EBR is not free at this corpus size; it is a plausible offline index, not
  a per-query luxury.
- **Entity extraction is still untyped** (`HOTPOT.md` limit). Sentence-initial words
  and generic noun phrases are entities; the case-aware rules above only partly
  compensate.
- **EBR is not bit-reproducible across processes.** The bridge selection iterates
  Python sets and takes `Counter.most_common`, so ties at the 20-entity cutoff and
  float summation order follow per-process hash seeding. Across four runs the
  always-EBR HotpotQA `both@10` moved between 0.7166 and 0.7167 (+0.1052 / +0.1054)
  and SciFact bridge nDCG between −0.0545 and −0.0547. Single-query differences, not
  a direction change, but quote the third decimal with that in mind. This is
  pre-existing behaviour in `hotpot.py`, not introduced here; routed/system
  comparisons use the same run as their baseline.

## Reproduce

```sh
cd /Users/asmit/Personal/algo-serp-stages
.venv/bin/python experiments/stages/run_ebr_beir.py
```

New files only: `lab/ebr_beir.py` (adapter), `lab/router.py` (gate),
`run_ebr_beir.py`, `results/ebr_beir.json`. `lab/hotpot.py`, `run_hotpot.py` and the
engine are untouched; the always-EBR HotpotQA row is reproduced from them
(`both@10` +0.105 to +0.1054 depending on the run, nDCG +0.0225 — see the
reproducibility limit above; `HOTPOT.md` reports +0.1052). Data root
`/Users/asmit/Personal/algo`; Touche lives in
`experiments/stages/data/webis-touche2020`. Runtime is about 8 minutes, dominated by
loading the 382k-document Touche index for the entity pass.
