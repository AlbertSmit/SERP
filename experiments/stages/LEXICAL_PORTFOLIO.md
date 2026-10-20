# E1 — Non-neural heterogeneous lexical portfolio

**Status: frozen run, gate FAILED.** Seven classical lexical scorers over one
Porter-stemmed index, one fixed RM3-style expansion, two fixed fusions, seven BEIR
test sets. No tuning on test, no router, no weights fitted.

Runner: `experiments/stages/run_lexical_portfolio.py`
Library: `experiments/stages/lab/lexical_portfolio.py`
Raw result: `experiments/stages/results/lexical_portfolio.json` (per-query metrics included)

```
PYTHONHASHSEED=0 ../../.venv/bin/python -u experiments/stages/run_lexical_portfolio.py
```

## Verdict

| | macro nDCG@10 | vs Porter BM25 | positive point estimates |
|---|---|---|---|
| Porter BM25 | 0.4675 | — | — |
| **portfolio_rrf** (primary, declared before the run) | **0.4615** | **−0.0060** | **0/7** |
| portfolio_naf | 0.4651 | −0.0024 | 2/7 |
| portfolio_rrf_no_rm3 | 0.4571 | −0.0104 | 0/7 |
| RM3 alone | 0.4666 | −0.0009 | 5/7 |

The equal-weight portfolio is **not above Porter BM25 on any of the seven sets**, and
the macro average is 0.0060 nDCG below it. Recall is the one place it is not worse:
`portfolio_rrf` macro R@100 is **0.5934 vs 0.5902** for BM25 (+0.0032), carried by
NFCorpus (+0.029) and ArguAna (+0.006).

Consequence, as pre-registered in `ARGON_PLAN.md`: this is a stop for "average many
lexical scorers and win". The next step is the selective fallback / QPP gate, not
another fusion-weight search. The heterogeneous scorers disagree in useful places —
RM3 is +0.0281 on TREC-COVID and +0.0128 on NFCorpus but −0.0401 on ArguAna — but an
unconditional average cannot capture that without a per-query decision.

## Protocol

| dataset | split | docs | queries | qrels | self-docs ignored |
|---|---|---|---|---|---|
| SciFact | test | 5,183 | 300 | 339 | 0 |
| NFCorpus | test | 3,633 | 323 | 12,334 | 0 |
| FiQA | test | 57,638 | 648 | 1,706 | 55 |
| ArguAna | test | 8,674 | 1,406 | 1,406 | 1,298 |
| SciDocs | test | 25,657 | 1,000 | 4,928 | 0 |
| TREC-COVID | test | 171,332 | 50 | 24,673 | 0 |
| Quora | test | 522,931 | 10,000 | 15,675 | 0 |

* **One Porter index per dataset**, `title + body`, same doc ids, both document and
  query tokens stemmed with the Snowball Porter stemmer (`channels.porter`).
* **Self-documents ignored.** A query whose id is also a corpus document is pushed to
  −inf before ranking, matching the frozen `first_principles/confirm_fresh.py` convention
  ("BEIR ignores the query's own document"). It affects ArguAna (1,298 queries) and FiQA
  (55); SciFact, NFCorpus, SciDocs, TREC-COVID and Quora have none. Without exclusion
  ArguAna BM25 is 0.3485; with it 0.4720.
* **Scoring is `s(q,d) = Σ_t w_t · impact[t,d]`** over a per-model sparse impact matrix
  built once and freed before the next model.
* **Baseline check** (every dataset, first 20 queries): our BM25 impact matrix equals
  `harness.bm25_fast` on the same Porter index with `max|Δscore| = 0.0` and 0/20 top-100
  mismatches. Absolute BM25 values reproduce `first_principles/FINDINGS.md` exactly
  (0.6838 / 0.3241 / 0.2477 / 0.4720 / 0.1545 / 0.5887 / 0.8017).
* **Statistics:** paired bootstrap over queries, 1,000 resamples, seed 1729, 95% CI of
  the mean ΔnDCG@10 vs Porter BM25; W/T/L counts.
* Metrics: nDCG@10 (graded, `(2^g−1)/log2(i+1)`), Recall@100 (per query, by |relevant|
  hit), MRR (first relevant in top 10), identical to `harness.metrics`.

## Formulas and constants

`N` documents, `df`/`cf` document/collection frequency of term `t`, `tf` in-document
frequency, `dl` document length, `avgdl` mean length, `T = Σ dl`, `qtf` query frequency.

**BM25** — `k1 = 1.2`, `b = 0.75`

```
idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
s(q,d) = Σ_t qtf_t · idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))
```

**BM25+** — same `k1`, `b`; `δ = 0.5` added inside the saturation bracket

```
s(q,d) = Σ_t qtf_t · idf(t) · [ tf·(k1+1)/(tf + k1·(1 − b + b·dl/avgdl)) + δ ]
```

**BM25L** — same `k1`, `b`, `δ = 0.5`

```
ctd = tf / (1 − b + b·dl/avgdl)
s(q,d) = Σ_t qtf_t · idf(t) · [ ctd·(k1+1)/(ctd + k1) + δ ]
```

With the idf fixed to the BM25 form above, BM25L is **algebraically identical** to
BM25+: substituting `ctd = tf/X` with `X = 1 − b + b·dl/avgdl` gives
`ctd(k1+1)/(ctd+k1) = tf(k1+1)/(tf + k1·X)`. The two models only separate when BM25L
uses its paper's idf `ln((N+1)/(df+0.5))`. The run therefore reports two identical
rankers (`bm25plus` = `bm25l` to 4 decimals on every set); they contribute the same
ranking twice to the portfolio. This is a property of the specified constants, not a bug.

**PL2** — `c = 1.0`, matches Terrier `matching.models.PL2`

```
tfn = tf · log2(1 + c·avgdl/dl),  λ = cf/N
s(q,d) = Σ_t qtf_t · (1/(tfn+1)) · [ tfn·log2(tfn/λ) + (λ − tfn)·log2(e) + 0.5·log2(2π·tfn) ]
```

(Verified term-by-term against `PL2.java`: `TF·log2(1/f) + f·log2 e + 0.5·log2(2πTF) +
TF·(log2 TF − log2 e)` is the same expression after collecting `TF`.)

**DPH** — parameter-free, matches Terrier `matching.models.DPH`

```
f = tf / dl
s(q,d) = Σ_t qtf_t · ((1−f)²/(tf+1)) · [ tf·log2( (tf·avgdl/dl)·(N/cf) ) + 0.5·log2(2π·tf·(1−f)) ]
```

*Deviation from the brief, documented on purpose.* The brief's variant
(`tfn = tf·avgdl/dl`, `f = tfn/(tfn+1)`, norm `(1−f)²/(tfn+1)`) makes the term weight
`≈ 1/(tfn+1)³·(tfn·log tfn)`, i.e. it **decreases** with term frequency. Reproduced in
the smoke run it scored 0.1773 nDCG on a 40-query SciFact slice, below every other
model. The brief instructs checking Terrier when a formula is uncertain, so the Terrier
source (fetched from `terrier-core/5.x`, class `DPH`) is used. The decision is
source-driven, was made on the smoke slice before the frozen seven-set run, and no
test-set number was used to choose it. `1−f = 0` (tf = dl, observed on TREC-COVID) is
clipped to `1e-12` to avoid `log(0)`; that document scores strongly negative, as
Java's `log(0) = −∞` would.

**DirichletLM** — `μ = 1500`

```
p_c = cf/T
s(q,d) = Σ_t qtf_t · log( (tf + μ·p_c) / (dl + μ) )
```

Absent query terms are included: implemented as a per-document base
`log(μ·p_c) − log(dl+μ)` for every query term, plus `log(1 + tf/(μ·p_c))` on documents
where the term occurs. Terms unseen in the corpus contribute nothing (Terrier ignores
them); they would otherwise be `log 0`.

**HiemstraLM** — `λ = 0.7`

```
s(q,d) = Σ_t qtf_t · log( λ·tf/dl + (1−λ)·cf/T )
```

Absent terms included, implemented as base `log((1−λ)·p_c)` plus
`log(1 + λ·tf / (dl·(1−λ)·p_c))` on matching documents.

## RM3 branch

One pass, no EM — an **RM3-style relevance model**, not the exact RM1/RM3 mixture:

1. BM25 top 10 documents (self-document excluded from the seed set).
2. `p(w|d) = tf/dl` over each document's stemmed terms.
3. `p(q|d) = softmax(BM25 score)` over those 10.
4. `p(w|R) = Σ_d p(w|d)·p(q|d)`, normalized to sum 1.
5. `q' = 0.6·p(w|q) + 0.4·p(w|R)`, `p(w|q) = qtf/|q|`; keep the top 20 terms.
6. Score with BM25 using the expanded weights: `s = Σ_t w_t · idf·tf(k1+1)/(tf+norm)`.

What is *not* exact RM3: no EM re-estimation of `p(q|d)` from the feedback set, and the
document weight is a softmax over BM25 scores rather than a query-likelihood. This is
stated so the row is not read as a Lavrenko–Croft reproduction.

## Fusion

* **RRF**, `k = 60`, over each model's ranked list truncated at 1,000:
  `score(d) = Σ_m 1/(60 + rank_m(d))`.
* **NAF**: per query, each model's scores are min-max normalized over that model's
  retrieved list (documents not retrieved by a model contribute 0, which is also where
  that model's own minimum maps), then averaged over models; top 1,000.
* Ties broken by document id ascending, matching `harness.rank`.

Systems: `portfolio_rrf`, `portfolio_naf` = BM25, BM25+, BM25L, PL2, DPH, DirichletLM,
HiemstraLM, RM3. `portfolio_rrf_no_rm3` drops RM3. Ablations `bm25_plus_pl2`,
`bm25_plus_dph`, `bm25_plus_lm` use RRF.

## Results

### nDCG@10

| system | SciFact | NFC | FiQA | Arg | SciDocs | COVID | Quora | macro |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.6838 | 0.3241 | 0.2477 | 0.4720 | 0.1545 | 0.5887 | 0.8017 | 0.4675 |
| bm25plus | 0.6835 | 0.3189 | 0.2457 | 0.4331 | 0.1510 | 0.5475 | 0.7978 | 0.4540 |
| bm25l | 0.6835 | 0.3189 | 0.2457 | 0.4331 | 0.1510 | 0.5475 | 0.7978 | 0.4540 |
| pl2 | 0.6644 | 0.3181 | 0.2267 | 0.3883 | 0.1512 | 0.4294 | 0.7995 | 0.4254 |
| dph | 0.6732 | 0.3167 | 0.2371 | 0.4671 | 0.1485 | 0.5387 | 0.6652 | 0.4352 |
| dirichletlm | 0.6746 | 0.3136 | 0.2298 | 0.4677 | 0.1483 | 0.5213 | 0.6495 | 0.4293 |
| hiemstralm | 0.6620 | 0.3026 | 0.2337 | 0.4402 | 0.1420 | 0.4506 | 0.7740 | 0.4293 |
| rm3 | 0.6867 | 0.3369 | 0.2494 | 0.4318 | 0.1600 | 0.6168 | 0.7848 | 0.4666 |
| **portfolio_rrf** | 0.6823 | 0.3224 | 0.2437 | 0.4687 | 0.1530 | 0.5767 | 0.7841 | 0.4615 |
| portfolio_naf | 0.6855 | 0.3305 | 0.2472 | 0.4715 | 0.1527 | 0.5746 | 0.7939 | 0.4651 |
| portfolio_rrf_no_rm3 | 0.6794 | 0.3193 | 0.2431 | 0.4658 | 0.1513 | 0.5612 | 0.7793 | 0.4571 |
| bm25_plus_pl2 | 0.6761 | 0.3226 | 0.2443 | 0.4634 | 0.1547 | 0.5773 | 0.8006 | 0.4627 |
| bm25_plus_dph | 0.6832 | 0.3222 | 0.2439 | 0.4780 | 0.1524 | 0.5724 | 0.7366 | 0.4555 |
| bm25_plus_lm | 0.6807 | 0.3210 | 0.2419 | 0.4717 | 0.1544 | 0.5781 | 0.7636 | 0.4588 |

### Recall@100

| system | SciFact | NFC | FiQA | Arg | SciDocs | COVID | Quora | macro |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.9280 | 0.2475 | 0.5584 | 0.9523 | 0.3577 | 0.1123 | 0.9751 | 0.5902 |
| bm25plus | 0.9297 | 0.2454 | 0.5426 | 0.9331 | 0.3498 | 0.1067 | 0.9734 | 0.5830 |
| bm25l | 0.9297 | 0.2454 | 0.5426 | 0.9331 | 0.3498 | 0.1067 | 0.9734 | 0.5830 |
| pl2 | 0.9019 | 0.2380 | 0.5257 | 0.8898 | 0.3476 | 0.0900 | 0.9739 | 0.5667 |
| dph | 0.9086 | 0.2395 | 0.5258 | 0.9580 | 0.3455 | 0.0948 | 0.9312 | 0.5719 |
| dirichletlm | 0.9269 | 0.2507 | 0.5318 | 0.9787 | 0.3522 | 0.1051 | 0.9423 | 0.5839 |
| hiemstralm | 0.9133 | 0.2365 | 0.5169 | 0.9196 | 0.3351 | 0.0852 | 0.9631 | 0.5671 |
| rm3 | 0.9313 | 0.2935 | 0.5664 | 0.9246 | 0.3668 | 0.1200 | 0.9705 | 0.5962 |
| **portfolio_rrf** | 0.9280 | 0.2764 | 0.5504 | 0.9587 | 0.3579 | 0.1079 | 0.9743 | 0.5934 |
| portfolio_naf | 0.9280 | 0.2742 | 0.5494 | 0.9566 | 0.3575 | 0.1078 | 0.9745 | 0.5926 |
| portfolio_rrf_no_rm3 | 0.9269 | 0.2483 | 0.5481 | 0.9538 | 0.3561 | 0.1057 | 0.9741 | 0.5876 |
| bm25_plus_pl2 | 0.9242 | 0.2422 | 0.5525 | 0.9488 | 0.3585 | 0.1045 | 0.9747 | 0.5865 |
| bm25_plus_dph | 0.9186 | 0.2440 | 0.5478 | 0.9587 | 0.3549 | 0.1050 | 0.9677 | 0.5853 |
| bm25_plus_lm | 0.9269 | 0.2535 | 0.5552 | 0.9716 | 0.3582 | 0.1102 | 0.9758 | 0.5930 |

### MRR

| system | SciFact | NFC | FiQA | Arg | SciDocs | COVID | Quora | macro |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.6478 | 0.5221 | 0.3052 | 0.3846 | 0.2687 | 0.8399 | 0.7952 | 0.5376 |
| bm25plus | 0.6508 | 0.5174 | 0.3041 | 0.3493 | 0.2642 | 0.8158 | 0.7915 | 0.5276 |
| bm25l | 0.6508 | 0.5174 | 0.3041 | 0.3493 | 0.2642 | 0.8158 | 0.7915 | 0.5276 |
| pl2 | 0.6308 | 0.5157 | 0.2737 | 0.3184 | 0.2655 | 0.7020 | 0.7939 | 0.5000 |
| dph | 0.6417 | 0.5122 | 0.2940 | 0.3807 | 0.2603 | 0.8062 | 0.6486 | 0.5062 |
| dirichletlm | 0.6353 | 0.5140 | 0.2854 | 0.3744 | 0.2535 | 0.8185 | 0.6219 | 0.5004 |
| hiemstralm | 0.6316 | 0.4896 | 0.2910 | 0.3597 | 0.2564 | 0.7555 | 0.7684 | 0.5074 |
| rm3 | 0.6504 | 0.5230 | 0.3045 | 0.3593 | 0.2722 | 0.8569 | 0.7755 | 0.5345 |
| **portfolio_rrf** | 0.6469 | 0.5158 | 0.3008 | 0.3801 | 0.2664 | 0.8365 | 0.7739 | 0.5315 |
| portfolio_naf | 0.6516 | 0.5224 | 0.3064 | 0.3834 | 0.2653 | 0.8292 | 0.7859 | 0.5349 |
| portfolio_rrf_no_rm3 | 0.6430 | 0.5144 | 0.3017 | 0.3770 | 0.2650 | 0.8325 | 0.7680 | 0.5288 |
| bm25_plus_pl2 | 0.6386 | 0.5161 | 0.2985 | 0.3788 | 0.2718 | 0.8195 | 0.7943 | 0.5311 |
| bm25_plus_dph | 0.6504 | 0.5201 | 0.2997 | 0.3903 | 0.2672 | 0.8300 | 0.7254 | 0.5262 |
| bm25_plus_lm | 0.6429 | 0.5246 | 0.2989 | 0.3822 | 0.2659 | 0.8374 | 0.7480 | 0.5286 |

### Paired ΔnDCG@10 vs Porter BM25 (1,000 resamples, seed 1729)

| system | SciFact | NFC | FiQA | Arg | SciDocs | COVID | Quora |
|---|---|---|---|---|---|---|---|
| portfolio_rrf | −0.0015 [−0.0079,+0.0059] | −0.0017 [−0.0045,+0.0012] | −0.0040 [−0.0095,+0.0015] | −0.0033 [−0.0076,+0.0012] | −0.0016 [−0.0034,+0.0005] | −0.0120 [−0.0285,+0.0034] | −0.0176 [−0.0201,−0.0152] |
| portfolio_naf | +0.0017 [−0.0051,+0.0093] | **+0.0064 [+0.0006,+0.0136]** | −0.0005 [−0.0059,+0.0050] | −0.0005 [−0.0046,+0.0040] | −0.0018 [−0.0037,+0.0001] | −0.0141 [−0.0287,+0.0007] | −0.0079 [−0.0097,−0.0060] |
| portfolio_rrf_no_rm3 | −0.0043 [−0.0117,+0.0037] | −0.0048 [−0.0078,−0.0020] | −0.0046 [−0.0105,+0.0018] | −0.0061 [−0.0105,−0.0013] | −0.0032 [−0.0052,−0.0011] | −0.0275 [−0.0488,−0.0083] | −0.0225 [−0.0253,−0.0198] |
| rm3 | +0.0029 [−0.0034,+0.0097] | **+0.0128 [+0.0063,+0.0210]** | +0.0016 [−0.0021,+0.0054] | −0.0401 [−0.0511,−0.0298] | **+0.0054 [+0.0032,+0.0077]** | **+0.0281 [+0.0104,+0.0481]** | −0.0170 [−0.0189,−0.0150] |

Portfolio W/T/L (win/tie/loss at query level): SciFact 16/265/19, NFC 38/232/53,
FiQA 76/480/92, ArguAna 150/1071/185, SciDocs 95/773/132, COVID 19/8/23, Quora
906/7455/1639. Ties dominate on SciFact/NFC/Quora because most queries have the same
nDCG under both systems.

Notable single-model results: `bm25_plus_dph` is the only system that beats BM25 on
ArguAna (+0.0060 [+0.0004,+0.0117]); RM3 has a positive point estimate on 5/7
(SciFact, NFCorpus, FiQA, SciDocs, TREC-COVID) and loses catastrophically on ArguAna
(−0.0401); the language models collapse
on Quora (DirichletLM −0.1522 [−0.1585,−0.1464], DPH −0.1366 [−0.1425,−0.1310]), where
documents average 11.7 tokens, so length-based Bayesian smoothing has almost no signal.

## Gate

Rule (frozen before the run): portfolio macro nDCG@10 above Porter BM25; no dataset
with ΔnDCG < −0.005 **and** a CI upper bound below −0.005; positive point estimate on
at least 4 of 7 datasets. Primary system `portfolio_rrf`.

| dataset | Δ nDCG | 95% CI | point est. < −0.005 | CI excludes −0.005 | verdict |
|---|---|---|---|---|---|
| SciFact | −0.0015 | [−0.0079,+0.0059] | no | no | pass |
| NFCorpus | −0.0017 | [−0.0045,+0.0012] | no | no | pass |
| FiQA | −0.0040 | [−0.0095,+0.0015] | no | no | pass |
| ArguAna | −0.0033 | [−0.0076,+0.0012] | no | no | pass |
| SciDocs | −0.0016 | [−0.0034,+0.0005] | no | no | pass |
| TREC-COVID | −0.0120 | [−0.0285,+0.0034] | yes | no | pass (not CI-confirmed) |
| Quora | −0.0176 | [−0.0201,−0.0152] | yes | **yes** | **fail** |

| check | result |
|---|---|
| macro nDCG above BM25 (0.4615 vs 0.4675) | **fail** |
| no dataset below the margin with CI excluding it | **fail** (Quora) |
| positive on ≥ 4/7 datasets | **fail** (0/7) |
| **overall** | **FAIL** |

`portfolio_naf` also fails all three checks (macro 0.4651; positive on 2/7; Quora
−0.0079 with CI upper −0.0060 below the margin), so the verdict does not depend on the
fusion rule.

**Gate interpretation.** Not a single dataset shows a positive point estimate under
either fusion. The loss is not one outlier: it is a uniform small dilution (−0.0015 to
−0.0040) plus two real losses (TREC-COVID −0.012, Quora −0.018). Quora is the
significant one: short paraphrase duplicates are already solved by exact stemmed
overlap, and averaging five weaker rankers into that list only moves correct duplicates
down. TREC-COVID has 50 queries and long documents; the individual models there range
from 0.429 (PL2) to 0.618 (RM3), so equal-weight fusion is dominated by the weakest
members.

## Cost

| | value |
|---|---|
| total wall time | 12.3 min (sum of per-dataset wall time) |
| peak RSS | 6.16 GB (Quora) |
| index build | 0.4–13.3 s per dataset (Porter transform), plus `harness.load` decode |
| impact matrix | 0.45–15.4 M non-zeros per model; built and freed one model at a time |
| ms/query (BM25) | 0.16 (NFC) … 5.02 (Quora) |
| ms/query (RM3) | 0.68 (NFC) … 10.23 (Quora) |
| ms/query (LM) | 0.25–5.7 |

Fused systems' latency was not itemized per system: the vectorized fusion pass is
included in each dataset's wall time but no per-system `ms/query` was recorded, so the
table above deliberately leaves those rows out rather than reporting a fabricated number.

**Models skipped:** none in the final run. An earlier attempt skipped DPH on TREC-COVID
because the brief's formula produced `log(0)` on documents where `tf = dl`; the Terrier
formulation with the `1−f` clip removed the failure. Every model was checked for
NaN/inf score vectors on SciFact during the smoke run, and no system reports a
non-finite metric in the final JSON.

## Files

* `experiments/stages/lab/lexical_portfolio.py` — impact matrices, scorers, RM3, fusion,
  bootstrap, gate rule.
* `experiments/stages/run_lexical_portfolio.py` — frozen runner, per-dataset tables,
  JSON writer.
* `experiments/stages/results/lexical_portfolio.json` — all numbers above plus rounded
  per-query `{ndcg, r100, mrr}` for every system and dataset (13,727 query rows).
* `experiments/stages/LEXICAL_PORTFOLIO.md` — this report.

## What this rules out, and what it leaves open

Ruled out for this configuration: an unconditional average (RRF or NAF) over BM25,
BM25+, BM25L, PL2, DPH, DirichletLM, HiemstraLM and RM3 beating Porter BM25 on BEIR
relevance. Fusion does not fix the fact that every component except RM3 is ≤ BM25 on
macro average.

Left open and *not* attempted here: per-query selection among the components. The
component evidence suggests where to look — RM3 carries TREC-COVID +0.028 and NFCorpus
+0.013 with CI above zero, DPH is the only ArguAna winner, and BM25 must be kept as the
fallback for Quora, where every expansion and every smoothing model loses. A QPP gate
must decide "expand / switch" per query without qrels; that is the next experiment, and
it should be judged on the same seven sets with the same bootstrap.
