# E2 — Per-query selective lexical selector with a hard BM25 fallback

**Status: frozen run, gate FAILED.** One learned selector over three frozen E1
scorers (BM25, RM3, DPH), trained on the three screening splits only, tested once
on the seven BEIR sets. No test query used for training, tuning, feature choice or
margin choice.

Runner: `experiments/stages/run_selective_lexical.py`
Library: `experiments/stages/lab/selective_lexical.py`
Raw result: `experiments/stages/results/selective_lexical.json` (per-query metrics,
predictions, selection diagnostics, permutation importances)

```
PYTHONHASHSEED=0 ../../.venv/bin/python -u experiments/stages/run_selective_lexical.py
```

## Verdict

| system | macro nDCG@10 | vs BM25 | positive datasets |
|---|---|---|---|
| Porter BM25 | **0.4675** | — | — |
| **selector** (primary, margin 0.005) | **0.4620** | **−0.0055** | **3/7** |
| selector_safe (margin 0.020) | 0.4631 | −0.0044 | 2/7 |
| selector_clf (3-class, no margin) | 0.4675 | −0.00002 | 4/7 |
| always RM3 | 0.4666 | −0.0009 | 5/7 |
| always DPH | 0.4352 | −0.0323 | 0/7 |
| oracle (best per query, qrels) | 0.5013 | +0.0338 | 7/7 |
| portfolio_naf (E1 reference) | 0.4651 | −0.0024 | 2/7 |

The selector does not beat BM25 on macro, is positive on 3 of 7 sets, and loses
significantly on ArguAna (−0.0137) and Quora (−0.0225). It captures **none** of
the +0.0338 macro oracle headroom; the conservative margin reduces the damage
(−0.0055 → −0.0044) but never turns it positive.

The reason is visible in the diagnostics, not the aggregate: the regressor is
**confident and wrong**. Mean predicted gain on switched queries is +0.023 to
+0.054 nDCG per dataset, while the realised gain of the switched queries is
between −0.083 and +0.034. On the training holdout the argmax of the three
predicted nDCGs matches the true best candidate only **35%** of the time —
against **83%** for the constant rule "always BM25". nDCG@10 is not predictable
from these features well enough to out-rank a constant.

## Protocol

| dataset | split | role | docs | queries |
|---|---|---|---|---|
| NFCorpus | dev | train | 3,633 | 324 |
| SciFact | train | train | 5,183 | 809 |
| FiQA | dev | train | 57,638 | 500 |
| SciFact | test | test | 5,183 | 300 |
| NFCorpus | test | test | 3,633 | 323 |
| FiQA | test | test | 57,638 | 648 |
| ArguAna | test | test | 8,674 | 1,406 |
| SciDocs | test | test | 25,657 | 1,000 |
| TREC-COVID | test | test | 171,332 | 50 |
| Quora | test | test | 522,931 | 10,000 |

* Same Porter-stemmed index, same `lp.Model` impact matrices, same RM3 branch,
  same metrics and bootstrap as E1 (`lexical_portfolio.py` imported read-only).
* **Baseline reproduction.** Per dataset, our BM25 impact matrix matches
  `harness.bm25_fast` on 20 queries with `max|Δscore| = 0.0` and 0/20 top-100
  mismatches, and the seven BM25 nDCG@10 values equal E1 / `FINDINGS.md` exactly
  (0.6838 / 0.3241 / 0.2477 / 0.4720 / 0.1545 / 0.5887 / 0.8017). Beyond the row
  averages, **per-query nDCG@10 for BM25, RM3 and DPH is identical to E1 on all
  13,727 query rows (max |Δ| = 0)**, so the selector's candidate pool is bit-for-bit
  the frozen E1 pool.
* Candidates: `bm25`, `rm3`, `dph`. PL2 was left out (the brief made it optional):
  it is below DPH on five of seven sets in E1 and below BM25 on all seven, so a
  fourth regressor would add cost without a plausible direction. Recorded as a
  choice, not a result.
* Self-documents ignored exactly as in E1 (ArguAna 1,298, FiQA 55).

## Features (36, no qrels, no qid, no dataset name)

| block | features |
|---|---|
| query | length, unique terms, stopword ratio, OOV rate, mean/max/sum IDF, query clarity (KL of query term distribution vs collection, terms with `cf = 0` skipped), wh / boolean / comparison flags, digit ratio, identifier flag (alnum token), length / corpus avgdl |
| BM25 list | top-1 score, top-10 mean, top1−top2 gap (absolute and relative), normalised score entropy over top 100, documents with a positive score (count and rate) |
| RM3 | added terms, added mass, mean/max IDF of added terms, share of original terms kept, drift (`1 − Jaccard` of original vs expanded term sets), rank overlap with BM25 top-100 |
| DPH list | top-1 score, top-10 mean, score entropy, rank overlap with BM25 top-100 |
| corpus | log N, log avgdl, log vocabulary, vocabulary/N |

No `qid`, dataset identity, or qrel-derived quantity. The corpus block is
legitimate at inference (you know the index you search) but it is also a corpus
fingerprint: because training corpora and test corpora differ, those four columns
mostly encode "this is not one of the training sets" and cannot extrapolate
(tree models saturate outside the training range). This is a known limitation of
the design, recorded before the run.

## Model

* One `HistGradientBoostingRegressor` per candidate, target = per-query nDCG@10,
  fixed `max_iter=200, max_depth=3, learning_rate=0.1, l2_regularization=1.0,
  min_samples_leaf=20, random_state=1729`. 1,633 training queries.
* Inference: predict all three, clip to [0, 1], take the argmax; switch away from
  BM25 only when `pred_best − pred_bm25 > margin`. Margins 0.005 (primary) and
  0.020 (safe), both frozen before the run.
* `selector_clf`: `HistGradientBoostingClassifier` (same hyperparameters) on the
  3-class "who wins" target, plain argmax, no margin. Secondary.
* 80/20 check inside the training rows only: policy ΔnDCG **+0.0009**, argmax
  accuracy 0.349, RMSE 0.285–0.293 per candidate. That check already predicted the
  test outcome: a +0.001 holdout gain is noise, and the argmax was far worse than
  the constant. The final selectors are refit on all 1,633 rows.
* Training-set true best: BM25 1,357 / RM3 145 / DPH 131 — i.e. 83% of a
  representative query population is already won by BM25.

### Top permutation importances (training rows only, R²)

| candidate | features (mean importance) |
|---|---|
| BM25 | BM25 entropy +0.29, DPH entropy +0.12, DPH top-1 +0.10, BM25 relative gap +0.08, RM3 added mass +0.07 |
| RM3 | BM25 entropy +0.30, DPH top-1 +0.11, DPH entropy +0.10, BM25 relative gap +0.06, query IDF sum +0.06 |
| DPH | DPH entropy +0.26, BM25 entropy +0.18, BM25 relative gap +0.06, query max IDF +0.06 |

The models are dominated by "how peaked is the BM25 list", which is a query
difficulty signal, not a candidate-difference signal. Nothing in the feature set
separates *which* scorer will win for a query.

## Results

### nDCG@10

| system | SciFact | NFC | FiQA | Arg | SciDocs | COVID | Quora | macro |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.6838 | 0.3241 | 0.2477 | 0.4720 | 0.1545 | 0.5887 | 0.8017 | 0.4675 |
| always_rm3 | 0.6867 | 0.3369 | 0.2494 | 0.4318 | 0.1600 | 0.6168 | 0.7848 | 0.4666 |
| always_dph | 0.6732 | 0.3167 | 0.2371 | 0.4671 | 0.1485 | 0.5387 | 0.6652 | 0.4352 |
| **selector** | 0.6775 | 0.3325 | 0.2483 | 0.4582 | 0.1548 | 0.5832 | 0.7792 | **0.4620** |
| selector_safe | 0.6778 | 0.3324 | 0.2488 | 0.4605 | 0.1543 | 0.5856 | 0.7825 | 0.4631 |
| selector_clf | 0.6839 | 0.3255 | 0.2475 | 0.4713 | 0.1548 | 0.5892 | 0.8002 | 0.4675 |
| oracle | 0.7138 | 0.3454 | 0.2742 | 0.5342 | 0.1718 | 0.6411 | 0.8286 | 0.5013 |

### Recall@100

| system | SciFact | NFC | FiQA | Arg | SciDocs | COVID | Quora | macro |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.9280 | 0.2475 | 0.5584 | 0.9523 | 0.3577 | 0.1123 | 0.9751 | 0.5902 |
| always_rm3 | 0.9313 | 0.2935 | 0.5664 | 0.9246 | 0.3668 | 0.1200 | 0.9705 | 0.5962 |
| always_dph | 0.9086 | 0.2395 | 0.5258 | 0.9580 | 0.3455 | 0.0948 | 0.9312 | 0.5719 |
| selector | 0.9336 | 0.2734 | 0.5494 | 0.9516 | 0.3582 | 0.1093 | 0.9686 | 0.5920 |
| selector_safe | 0.9336 | 0.2730 | 0.5502 | 0.9502 | 0.3580 | 0.1089 | 0.9699 | 0.5920 |
| selector_clf | 0.9280 | 0.2481 | 0.5574 | 0.9523 | 0.3575 | 0.1107 | 0.9748 | 0.5898 |

### MRR

| system | SciFact | NFC | FiQA | Arg | SciDocs | COVID | Quora | macro |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.6478 | 0.5221 | 0.3052 | 0.3846 | 0.2687 | 0.8399 | 0.7952 | 0.5376 |
| always_rm3 | 0.6504 | 0.5230 | 0.3045 | 0.3593 | 0.2722 | 0.8569 | 0.7755 | 0.5345 |
| always_dph | 0.6417 | 0.5122 | 0.2940 | 0.3807 | 0.2603 | 0.8062 | 0.6486 | 0.5062 |
| selector | 0.6406 | 0.5265 | 0.3027 | 0.3770 | 0.2659 | 0.8352 | 0.7700 | 0.5311 |
| selector_safe | 0.6385 | 0.5240 | 0.3048 | 0.3786 | 0.2660 | 0.8472 | 0.7735 | 0.5332 |
| selector_clf | 0.6478 | 0.5230 | 0.3046 | 0.3841 | 0.2694 | 0.8399 | 0.7934 | 0.5375 |

### Paired ΔnDCG@10 vs Porter BM25 (1,000 resamples, seed 1729)

| system | SciFact | NFC | FiQA | Arg | SciDocs | COVID | Quora |
|---|---|---|---|---|---|---|---|
| **selector** | −0.0062 [−0.0136,+0.0015] | **+0.0084 [+0.0027,+0.0153]** | +0.0006 [−0.0036,+0.0045] | −0.0137 [−0.0207,−0.0065] | +0.0003 [−0.0019,+0.0026] | −0.0055 [−0.0259,+0.0171] | −0.0225 [−0.0252,−0.0200] |
| selector_safe | −0.0060 [−0.0127,+0.0005] | **+0.0083 [+0.0026,+0.0153]** | +0.0011 [−0.0027,+0.0045] | −0.0115 [−0.0178,−0.0048] | −0.0002 [−0.0022,+0.0019] | −0.0030 [−0.0242,+0.0194] | −0.0193 [−0.0216,−0.0170] |
| selector_clf | +0.0001 [−0.0036,+0.0038] | +0.0014 [−0.0004,+0.0036] | −0.0002 [−0.0014,+0.0011] | −0.0007 [−0.0024,+0.0011] | +0.0002 [+0.0000,+0.0006] | +0.0006 [−0.0003,+0.0017] | −0.0016 [−0.0022,−0.0010] |
| always_rm3 | +0.0029 [−0.0034,+0.0097] | **+0.0128 [+0.0063,+0.0210]** | +0.0016 [−0.0021,+0.0054] | −0.0401 [−0.0511,−0.0298] | **+0.0054 [+0.0032,+0.0077]** | **+0.0281 [+0.0104,+0.0481]** | −0.0170 [−0.0189,−0.0150] |
| always_dph | −0.0106 [−0.0251,+0.0046] | −0.0074 [−0.0137,−0.0019] | −0.0106 [−0.0191,−0.0024] | −0.0049 [−0.0132,+0.0025] | −0.0060 [−0.0094,−0.0027] | −0.0500 [−0.0845,−0.0154] | −0.1366 [−0.1425,−0.1310] |
| oracle | +0.0300 [+0.0214,+0.0397] | +0.0213 [+0.0145,+0.0290] | +0.0265 [+0.0211,+0.0323] | +0.0622 [+0.0554,+0.0691] | +0.0172 [+0.0150,+0.0196] | +0.0524 [+0.0351,+0.0699] | +0.0268 [+0.0251,+0.0287] |

selector W/T/L: SciFact 11/274/15, NFC 51/245/27, FiQA 56/550/42, ArguAna
167/1018/221, SciDocs 107/798/95, COVID 17/18/15, Quora 508/8388/1104.

## Selection distribution and actual outcome

Counts per dataset (BM25 / RM3 / DPH) and, for the queries actually switched
away from BM25, how often that candidate really beat BM25 and its mean ΔnDCG.

| dataset | margin 0.005 | RM3 win-rate / mean Δ | DPH win-rate / mean Δ | true best (b/r/d) |
|---|---|---|---|---|
| SciFact | 146 / 95 / 59 | 0.07 / −0.0016 | 0.07 / −0.0292 | 254 / 17 / 29 |
| NFCorpus | 108 / 129 / 86 | 0.26 / +0.0235 | 0.21 / −0.0037 | 226 / 58 / 39 |
| FiQA | 285 / 207 / 156 | 0.14 / +0.0025 | 0.17 / −0.0009 | 505 / 64 / 79 |
| ArguAna | 410 / 373 / 623 | 0.15 / −0.0473 | 0.18 / −0.0027 | 989 / 218 / 199 |
| SciDocs | 363 / 338 / 299 | 0.17 / +0.0053 | 0.17 / −0.0051 | 737 / 144 / 119 |
| TREC-COVID | 14 / 17 / 19 | **0.71 / +0.0336** | 0.26 / −0.0445 | 15 / 24 / 11 |
| Quora | 2597 / 5442 / 1961 | 0.07 / −0.0114 | 0.07 / −0.0832 | 8666 / 661 / 673 |

Reading the columns together:

* Where the switch happens often it is usually wrong. Quora switches 74% of
  queries and gets a win-rate of 7% on either alternative; ArguAna switches 71%
  and loses 0.047 on its RM3 picks (RM3 is the one scorer that is genuinely bad
  on ArguAna, and the selector cannot see it).
* The one place it works is TREC-COVID: 17 RM3 picks with a 71% win-rate and
  +0.034 mean gain. Those 17 picks are then cancelled by 19 DPH picks that lose
  0.045 each, turning a +0.028 RM3 opportunity into −0.0055.
* `selector_safe` behaves the same way, just with fewer switches (e.g. Quora
  66%, COVID 62%); the DPH errors scale down with the count, so the margin buys
  safety but not direction.
* `selector_clf` collapses to "always BM25" (9,761 / 10,000 on Quora). A
  classifier trained on these features cannot separate the winners, so it
  returns the majority class — which is a correct conclusion about the features.
  It is a numeric tie with BM25 (macro −0.00002, i.e. three queries' worth of
  nDCG in total), so it fails only the "macro above BM25" check while passing
  "no dataset below the margin" and "positive on 4/7". It is not evidence of a
  selector: it is evidence that abstaining is the best available policy.

## Gate

Rule (frozen, same as E1): macro nDCG@10 above Porter BM25; no dataset with
ΔnDCG < −0.005 **and** a CI upper bound below −0.005; positive point estimate on
at least 4 of 7. Primary system `selector`.

| dataset | Δ nDCG | 95% CI | point est. < −0.005 | CI excludes −0.005 | verdict |
|---|---|---|---|---|---|
| SciFact | −0.0062 | [−0.0136,+0.0015] | yes | no | pass |
| NFCorpus | +0.0084 | [+0.0027,+0.0153] | no | no | pass |
| FiQA | +0.0006 | [−0.0036,+0.0045] | no | no | pass |
| ArguAna | −0.0137 | [−0.0207,−0.0065] | yes | **yes** | **fail** |
| SciDocs | +0.0003 | [−0.0019,+0.0026] | no | no | pass |
| TREC-COVID | −0.0055 | [−0.0259,+0.0171] | yes | no | pass (not CI-confirmed) |
| Quora | −0.0225 | [−0.0252,−0.0200] | yes | **yes** | **fail** |

| check | result |
|---|---|
| macro nDCG above BM25 (0.4620 vs 0.4675) | **fail** |
| no dataset below the margin with CI excluding it | **fail** (ArguAna, Quora) |
| positive on ≥ 4/7 datasets | **fail** (3/7: NFCorpus, FiQA, SciDocs) |
| **overall** | **FAIL** |

`selector_safe` fails the macro check (0.4631) and the positive-count check (2/7).
Its only dataset below the margin with a CI excluding it is Quora (−0.0193, CI
upper −0.0170); ArguAna is −0.0115 with CI upper −0.0048, which is above the
−0.005 margin. The verdict does not depend on the margin: the safe selector still
does not beat BM25.

## Cost

| | value |
|---|---|
| total wall time | 8.6 min (training construction ≈ 1 min + seven test sets) |
| peak RSS | 5.19 GB (TREC-COVID and Quora, both reach it) |
| training | 1,633 queries × 36 features, 3 regressors + 1 classifier, < 2 s |
| feature extraction | 0.26 s per 1,000 queries (Quora); < 0.1 s on the small sets |
| selector inference | 3 model `predict` calls per query, microseconds |
| ms/query | BM25 0.21 (NFC) – 6.43 (Quora); RM3 0.86 – 13.26; DPH 0.21 – 6.54 |

**Cost of the policy, not the model.** A switch means running RM3 (2.0–2.3×
BM25) or a second scorer; with the observed switch rates a selector that worked
would roughly double serving cost. This one pays that cost on Quora for −0.0225.

## Limitations

1. **QPP signal, not model capacity.** These 36 features carry query difficulty
   and candidate-list shape, not candidate *difference*. The holdout argmax
   accuracy (35% vs 83% for the constant) says the failure is at the feature
   level, so a larger model would overfit faster rather than fix it. A different
   feature family (score-distribution divergence between candidates, per-query
   relevance-model agreement, corpus-relative expansion statistics) is untested.
2. **Corpus fingerprinting.** `N`, `avgdl`, vocabulary and vocabulary/N let the
   trees identify the training corpora; on unseen corpora they saturate and add
   no signal. This cannot help the test sets and may hurt calibration.
3. **Training-set scale.** 1,633 labelled queries, 83% of them BM25-won; the
   regressors see only 145 RM3 and 131 DPH wins. A positive-class imbalance this
   severe makes per-query benefit prediction a minority-class problem.
4. **Three candidates only.** PL2 was not included, so the oracle is over three
   scorers, not seven; the E1 oracle over all eight would be larger. This does
   not affect the gate (a smaller pool is friendlier to the selector).
5. **No probability calibration study.** The regressor's predicted gains are
   3–10× the realised gains; a per-dataset or global recalibration of predicted
   nDCG slopes was not attempted because it would require test data.

## Files

* `experiments/stages/lab/selective_lexical.py` — features, feature names,
  `Selector`, `ClassifierSelector`, margin policy, selection diagnostics.
* `experiments/stages/run_selective_lexical.py` — training-set construction,
  80/20 check, permutation importance, seven-set evaluation, JSON writer.
* `experiments/stages/results/selective_lexical.json` — all numbers above plus
  per-query metrics for every system and the full importance tables.
* `experiments/stages/SELECTIVE_LEXICAL.md` — this report.

## Conclusion

Per-query selection among BM25, RM3 and DPH, trained on screening qrels and gated
by a hard BM25 fallback, does not beat BM25 on any macro view and loses
significantly where it switches most (ArguAna, Quora). Together with E1 (no
unconditional fusion wins) this closes the "universal no-embedding scorer" line
for the current feature and scorer set: the component evidence that motivated E2
— RM3 +0.028 on TREC-COVID, DPH best on ArguAna — is real but only visible with
qrels, and nothing measurable at inference time predicts which query gets it.

The product conclusion stands: BM25 stays the default ranker, and RM3-style
expansion is a workload-scoped option (it is the only component with a
CI-positive gain on three of seven sets), not an automatically selected mode.
