# CESR-1: constrained evidence-set retrieval (oracle constraints)

This is the first **evidence-set** experiment. The objective is not document relevance:
each query carries a set of gold constraints (entities and relation cues), and the
retriever must choose the paragraphs that cover them under a token budget. BM25 is a
candidate generator here, not the thing being optimised.

The constraint extractor is assumed perfect. This is **oracle-constraint CESR**: an
upper bound on a deployable system, not a deployable system itself. If the objective
fails with gold constraints, it is not worth automating them.

Frozen before the run, not tuned: budgets `500/1000/2000` engine tokens; redundancy
`mu=0.2`; pair bonus `0.25`; cost exponent `0.35`; RRF `k=60`; pool depth 100 per
channel. No parameter search, no router, no typed filtering.

## Method

**Candidate pool.** Union of Porter BM25 top-100, the untyped bridge channel top-100,
and the per-entity channel top-100, built with the unmodified HotpotQA EBR constants
(20 first-hop seeds, 20 bridge entities, 5% entity df cap) imported read-only from
`lab/hotpot.py` via `lab/multihop_transfer.py`. Pool membership is budget-independent.

**Constraints.**

- 2Wiki: each row's `evidences` is a list of `[subject, relation, object]` triples.
  Subject and object become entity constraints; each distinct relation string becomes a
  relation constraint; each triple's endpoints become an evidence pair.
- MuSiQue: each `question_decomposition` step contributes entities from the
  sub-question (`hotpot.query_entities`) plus its answer as an entity constraint, and
  its endpoints as a pair. `#k` references are rewritten to the answer of step `k`
  before entity extraction. `paragraph_support_idx` is never read — that field names the
  gold paragraph and would turn the selector into a lookup.

**Coverage.** A paragraph covers an entity constraint when the normalised entity string
is an exact substring of normalised title+text (no fuzzy matching, v1). A relation
constraint is covered by the normalised cue or by a pre-declared synonym from a frozen
table of 15 entries (one per frequent relation; `RELATION_SYNONYMS` in `lab/cesr.py`).

**Weights.** Entity constraints weigh the EBR entity IDF where the normalised string is
in the entity index, else 1.0. Relation constraints weigh 1.0. Weights are normalised to
sum to one, so "marginal weight", "coverage" and "pair weight" share units.

**Selector.** Deterministic greedy submodular maximisation. At each step, among
candidates that fit the remaining budget:

```
score = marginal_constraint_weight / max(dl + 10, 1)^0.35
        - 0.2 * max_jaccard(candidate, selected)          # entity-set Jaccard
        + 0.25 * pair_weight                              # both endpoints of one triple
```

Stop when every constraint is covered, no candidate fits, or the best score is
non-positive. Output is the selection in order, with covered/uncovered constraints and
per-chunk provenance (channels that found it, tokens, marginal gain, redundancy at
selection time).

**`cesr_coverage_first`.** The same objective with the cost exponent set to 0 until every
constraint is covered, then EBR rank order fills the remaining budget. It ignores cost, so
it can overshoot the budget — the overspend rate is reported (0.211/0.025/0.002 on 2Wiki,
0.235/0.071/0.013 on MuSiQue).

**Baselines at the same token budget.** Porter BM25 truncated in rank order; EBR RRF
(Porter + bridge + entity, `k=60`) truncated in rank order. "EBR RRF truncated" and
"truncation of the EBR fused list" are the same operation, so they are one row
(`ebr_trunc`). Truncation uses the repo's skip-and-continue convention and `ix.dl` for
length.

## Data

| dataset | identity | questions | paragraphs | gold paragraphs | constraint source |
|---|---|---:|---:|---:|---|
| 2Wiki dev | title | 12,576 | 54,957 | 30,654 | `evidences` triples |
| MuSiQue dev | title+paragraph_text | 2,417 | 21,100 | 6,404 | `question_decomposition` + answers |

Dev only; both datasets are labelled. All constants were fixed before the run. A
200-question wiring slice was used to validate the harness, not to choose anything.

## Candidate-pool ceiling (the first hard limit)

| | 2Wiki | MuSiQue |
|---|---:|---:|
| gold paragraphs in pool | 0.875 | 0.777 |
| queries with **all** gold in pool | 0.732 | 0.576 |
| mean achievable constraint coverage from pool | 0.975 | 0.924 |
| queries where every constraint is coverable | 0.692 | 0.501 |

The pool, not the selector, sets the `all@10` ceiling: 0.732 / 0.576. This is the same
pool EBR uses; it is not a CESR handicap, but every number below must be read against it.

## Results

`cover` = normalised constraint weight covered; `redun` = mean pairwise entity Jaccard of
the selected set; `n` = paragraphs selected. `all@5`/`all@10` are prefix-completeness over
the **selection order** at the matched budget, so a system that returns fewer paragraphs
must still contain every gold paragraph inside them.

### 2Wiki dev (n=12,576)

| budget | system | all@5 | all@10 | recall | cover | redun | tokens | n | nDCG@10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | Porter trunc | 0.272 | 0.295 | 0.633 | 0.838 | 0.022 | 492 | 8.1 | 0.665 |
| 500 | EBR trunc | 0.335 | 0.366 | 0.687 | 0.870 | 0.037 | 489 | 7.4 | 0.712 |
| 500 | **CESR** | **0.627** | **0.628** | **0.843** | **0.965** | 0.030 | **246** | 2.9 | **0.867** |
| 500 | CESR coverage-first | 0.703 | 0.711 | 0.888 | 0.975 | 0.030 | 549 | 6.9 | 0.901 |
| 1000 | Porter trunc | 0.295 | 0.352 | 0.681 | 0.878 | 0.017 | 991 | 14.0 | 0.683 |
| 1000 | EBR trunc | 0.373 | 0.452 | 0.750 | 0.910 | 0.032 | 985 | 11.5 | 0.736 |
| 1000 | **CESR** | **0.690** | **0.694** | **0.878** | **0.974** | 0.031 | **294** | 3.1 | **0.891** |
| 1000 | CESR coverage-first | 0.710 | 0.720 | 0.893 | 0.975 | 0.029 | 1006 | 11.8 | 0.904 |
| 2000 | Porter trunc | 0.297 | 0.364 | 0.712 | 0.905 | 0.012 | 1997 | 26.4 | 0.686 |
| 2000 | EBR trunc | 0.377 | 0.486 | 0.803 | 0.936 | 0.025 | 1997 | 20.0 | 0.743 |
| 2000 | **CESR** | **0.693** | **0.699** | **0.881** | **0.975** | 0.032 | **303** | 3.1 | **0.893** |
| 2000 | CESR coverage-first | 0.710 | 0.723 | 0.895 | 0.975 | 0.024 | 1998 | 20.0 | 0.905 |

Paired bootstrap CESR − EBR, 1,000 resamples, seed 1729:

| budget | metric | Δ | 95% CI | W/T/L |
|---:|---|---:|---|---:|
| 500 | all@5 | +0.2917 | [+0.2836,+0.2999] | 3816/8612/148 |
| 500 | all@10 | **+0.2624** | [+0.2545,+0.2705] | 3469/8938/169 |
| 500 | recall | +0.1561 | [+0.1521,+0.1600] | 4804/7535/237 |
| 500 | constraint coverage | +0.0942 | [+0.0917,+0.0967] | 8242/4288/46 |
| 500 | nDCG@10 | +0.1548 | [+0.1513,+0.1585] | 7519/4454/603 |
| 1000 | all@10 | +0.2419 | [+0.2342,+0.2502] | 3265/9088/223 |
| 1000 | recall | +0.1277 | [+0.1239,+0.1318] | 4077/8232/267 |
| 2000 | all@10 | +0.2130 | [+0.2052,+0.2210] | 2924/9407/245 |
| 2000 | recall | +0.0780 | [+0.0745,+0.0817] | 2761/9494/321 |

Token ratio CESR/EBR: 0.50 / 0.30 / 0.15 at 500/1000/2000.

### MuSiQue dev, paragraph identity (n=2,417)

| budget | system | all@5 | all@10 | recall | cover | redun | tokens | n | nDCG@10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | Porter trunc | 0.089 | 0.096 | 0.424 | 0.665 | 0.030 | 487 | 6.1 | 0.441 |
| 500 | EBR trunc | 0.118 | 0.139 | 0.467 | 0.706 | 0.059 | 486 | 5.8 | 0.468 |
| 500 | **CESR** | **0.336** | **0.341** | **0.647** | **0.913** | 0.028 | **272** | 3.1 | **0.680** |
| 500 | CESR coverage-first | 0.399 | 0.429 | 0.716 | 0.924 | 0.037 | 601 | 6.6 | 0.722 |
| 1000 | Porter trunc | 0.096 | 0.139 | 0.490 | 0.734 | 0.023 | 985 | 11.6 | 0.466 |
| 1000 | EBR trunc | 0.125 | 0.197 | 0.544 | 0.776 | 0.050 | 985 | 10.8 | 0.497 |
| 1000 | **CESR** | **0.351** | **0.372** | **0.666** | **0.919** | 0.028 | **346** | 3.6 | **0.690** |
| 1000 | CESR coverage-first | 0.414 | 0.470 | 0.740 | 0.924 | 0.039 | 1033 | 11.2 | 0.733 |
| 2000 | Porter trunc | 0.096 | 0.145 | 0.556 | 0.792 | 0.016 | 1993 | 22.9 | 0.469 |
| 2000 | EBR trunc | 0.125 | 0.210 | 0.624 | 0.831 | 0.037 | 1992 | 21.0 | 0.503 |
| 2000 | **CESR** | **0.351** | **0.381** | **0.678** | **0.923** | 0.028 | **381** | 3.9 | **0.693** |
| 2000 | CESR coverage-first | 0.414 | 0.480 | 0.761 | 0.924 | 0.034 | 2003 | 21.0 | 0.735 |

| budget | metric | Δ | 95% CI | W/T/L |
|---:|---|---:|---|---:|
| 500 | all@5 | +0.2180 | [+0.1982,+0.2371] | 593/1758/66 |
| 500 | all@10 | **+0.2011** | [+0.1816,+0.2201] | 557/1789/71 |
| 500 | recall | +0.1793 | [+0.1663,+0.1918] | 1114/1106/197 |
| 500 | constraint coverage | +0.2072 | [+0.1974,+0.2169] | 1715/662/40 |
| 500 | nDCG@10 | +0.2113 | [+0.1996,+0.2230] | 1600/525/292 |
| 1000 | all@10 | +0.1754 | [+0.1552,+0.1949] | 549/1743/125 |
| 1000 | recall | +0.1217 | [+0.1102,+0.1334] | 912/1251/254 |
| 2000 | all@10 | +0.1709 | [+0.1502,+0.1907] | 557/1716/144 |
| 2000 | recall | +0.0541 | [+0.0427,+0.0664] | 696/1365/356 |

Token ratio CESR/EBR: 0.56 / 0.35 / 0.19.

Redundancy is lower than EBR at 500 and 1000 on both datasets (Δ −0.007…−0.031, CIs
excluding zero). At 2000 on 2Wiki it is slightly higher (+0.0066) because CESR selects only
~3 paragraphs and they can share an article. The penalty works, but redundancy was never
the main gap.

## Constraint coverage against its own ceiling

| | 2Wiki | MuSiQue |
|---|---:|---:|
| mean achievable coverage from pool | 0.975 | 0.924 |
| CESR coverage, budget 2000 | 0.975 | 0.923 |
| CESR covers **all** constraints @500 / @1000 / @2000 | 0.601 / 0.668 / 0.674 | 0.413 / 0.456 / 0.484 |
| `all@10` when all constraints are covered (budget 2000) | **0.919** | **0.521** |
| `all@10` overall, budget 2000 | 0.699 | 0.381 |
| `all@10` as a fraction of the per-query pool ceiling | 0.954 | 0.662 |

This is the most important table in the experiment.

On 2Wiki, covering every gold constraint is nearly equivalent to retrieving the full
evidence set: 0.919 `all@10` conditional on full coverage. The objective is well posed and
the selector reaches 95% of the pool ceiling. What is left is pool recall.

On MuSiQue, covering every constraint implies complete evidence only 52% of the time. The
constraint set is under-specified: intermediate answers and generic relation cues
("country", "member", "located in") are satisfied by non-gold paragraphs. The selector is
already at the coverage ceiling (0.923 of a maximum 0.924) and still only reaches 66% of
the pool ceiling. **More constraint coverage does not buy more evidence completeness on
MuSiQue, because the constraints do not identify the evidence.**

The coverage-first variant measures the same gap from the other side. After all
constraints are covered it keeps spending the budget, and `all@10` rises: 0.699 → 0.723 on
2Wiki, 0.381 → 0.480 on MuSiQue at 2000 tokens. Those extra paragraphs are not needed to
satisfy any constraint; they are needed because the constraint set does not pin the gold
set.

## Slices (all@10 / recall / constraint coverage)

2Wiki, budget 500 → 2000:

| slice | n | Porter @500 | EBR @500 | CESR @500 | CESR @2000 |
|---|---:|---:|---:|---:|---:|
| bridge_comparison | 2,751 | 0.004 / 0.479 | 0.014 / 0.554 / 0.895 | 0.180 / 0.734 / 0.953 | 0.221 / 0.760 / 0.958 |
| comparison | 3,040 | 0.845 / 0.922 | 0.923 / 0.961 / 0.948 | 0.960 / 0.979 / 0.990 | 0.990 / 0.995 / 0.997 |
| compositional | 5,236 | 0.149 / 0.559 | 0.244 / 0.613 / 0.831 | 0.662 / 0.828 / 0.956 | 0.765 / 0.881 / 0.969 |
| inference | 1,549 | 0.229 / 0.590 | 0.307 / 0.633 / 0.808 | 0.662 / 0.819 / 0.964 | 0.757 / 0.872 / 0.977 |

MuSiQue, budget 500 → 2000:

| slice | n | Porter @500 | EBR @500 | CESR @500 | CESR @2000 |
|---|---:|---:|---:|---:|---:|
| support_2 | 1,252 | 0.173 / 0.503 | 0.252 / 0.553 / 0.769 | 0.489 / 0.709 / 0.931 | 0.503 / 0.720 / 0.934 |
| support_3 | 760 | 0.021 / 0.389 | 0.026 / 0.427 / 0.696 | 0.236 / 0.626 / 0.919 | 0.299 / 0.679 / 0.934 |
| support_4 | 405 | 0.000 / 0.244 | 0.005 / 0.278 / 0.532 | 0.079 / 0.493 / 0.848 | 0.156 / 0.548 / 0.866 |

The `all@10` gain at matched budget (2000 tokens) is largest exactly where fusion is
weakest — 4-gold MuSiQue (0.010 → 0.156) and 2Wiki `bridge_comparison` (0.033 → 0.221). It
is smallest where fusion already saturates (`comparison`, 0.988 → 0.990). This is the
expected shape for a coverage objective, and it is also why aggregate numbers flatter it.

## Provenance (budget 500)

Share of selected chunks by the channels that found them:

| pool membership | 2Wiki | MuSiQue |
|---|---:|---:|
| porter only | 0.204 | 0.277 |
| porter + bridge | 0.141 | 0.180 |
| porter + bridge + entity | 0.402 | 0.227 |
| porter + entity | 0.074 | 0.099 |
| bridge only | 0.160 | 0.165 |
| entity only | 0.017 | 0.047 |
| bridge + entity | 0.002 | 0.006 |
| **outside Porter top-100** | **0.179** | **0.218** |

18–22% of the evidence CESR selects is not in Porter's top-100. Expansion still matters
under a constraint objective; the constraint just decides which expanded candidate is
worth a slot.

## Latency (CPU, full dev)

| | 2Wiki | MuSiQue |
|---|---:|---:|
| retrieval (Porter + bridge + entity + RRF + pool) | 13.8 ms/q | 6.2 ms/q |
| constraint construction + coverage matrix | 1.4 ms/q | 1.1 ms/q |
| CESR selection (per budget) | 0.28 ms/q | 0.31 ms/q |
| coverage-first selection (once per query) | 0.38 ms/q | 0.40 ms/q |
| coverage-first budget fill (per budget) | 0.14 ms/q | 0.13 ms/q |
| full evaluation | 305 s | 33 s |

Selection is 2–5% of end-to-end cost. The expensive part is still the EBR channels.

## Gate

Pre-registered rule: *CESR beats EBR on `all@10` at the same budget on at least one budget
for both datasets (Δ > 0, CI95 low > 0), or matches `all@10` (CI spans zero) with ≥20%
fewer tokens on both.*

Applied mechanically: all@10 beats EBR on **all three budgets on both datasets**, with CIs
well above zero (500 and 2000: +0.2624 [+0.2545,+0.2705] and +0.2130 [+0.2052,+0.2210] on
2Wiki; +0.2011 [+0.1816,+0.2201] and +0.1709 [+0.1502,+0.1907] on MuSiQue). CESR also
uses 15–56% of EBR's tokens, so the fewer-tokens branch would hold too if the deltas were
ties.

**Gate PASSED.** Evidence-set selection beats fusion on completeness at matched budget,
with fewer tokens, and the effect is not a metric artifact: recall, constraint coverage
and nDCG move the same way.

## Verdict

**The objective is real; the oracle test localises the bottleneck outside the selector.**

1. **The selector is not the problem.** CESR reaches 0.975 / 0.923 mean constraint
   coverage against a pool maximum of 0.975 / 0.924. There is nothing left for a better
   optimiser to win: exact submodular maximisation could add at most ~0.001 coverage.
2. **Candidate recall is the binding constraint on 2Wiki.** Only 73.2% of queries have all
   gold paragraphs anywhere in the pool, and CESR already achieves 95% of that ceiling.
   No selection objective can exceed it. Improving this direction on 2Wiki means widening
   the pool, not improving the objective.
3. **Constraint adequacy is the binding constraint on MuSiQue.** Full constraint coverage
   yields complete evidence only 52% of the time. The decomposition sub-questions plus
   answers do not pin the evidence set; generic relation cues are satisfied by
   distractors. Any automatic constraint extractor will be *worse* than this oracle, so
   MuSiQue is where the direction dies unless constraints become more specific (typed
   relations, answer-anchored pairs), not merely more numerous.
4. **The trade-off is genuine and worth reporting.** CESR returns 3–4 paragraphs where
   truncation returns 6–26 and still wins `all@10`: fewer tokens, more complete evidence.
   Extra paragraphs do still add raw evidence recall (coverage-first: 0.895 vs 0.881 on
   2Wiki, 0.761 vs 0.678 on MuSiQue at 2000 tokens), which is exactly the measure of how
   much the constraint set under-specifies the gold set.

**Is CESR a real algorithm direction?** Yes as a *selection/packing* objective, no as a
retrieval objective on its own. It is best read as: **retrieve broadly (EBR), then choose
the evidence set by coverage instead of by rank.** That composition is the natural next
experiment, and the oracle test says the interesting question is no longer "can coverage
beat relevance as a selector" (it can) but "can an automatic constraint extractor reach
enough constraint specificity that coverage implies completeness".

Do not ship CESR. Do keep it as the packing objective inside deep mode, gated on a
constraint extractor that is measured separately.

## Honest limits

- **Oracle constraints.** Gold triples and decomposition answers are not available at
  query time. This is an upper bound; a deployable extractor's error rate is not modelled
  here at all.
- **Answers as constraints.** MuSiQue entity constraints include the decomposition
  answers, including intermediate answers. On this dataset that is close to handing the
  selector the gold paragraph's most distinctive token.
- **`all@10` prefix semantics.** With 3–4 selected paragraphs, `all@10` means "the whole
  returned set contains all gold", which is a matched-token comparison, not a top-10
  ranking comparison. Selected count and tokens are reported on every row for that reason.
- **Candidate-recall ceiling is shared.** 0.732 / 0.576 per-query ceilings bound every
  system in the table, including EBR.
- **Entity weighting artifact.** `CAP_RE` is ASCII-only, so non-ASCII names inside
  paragraph bodies are not in the entity index and fall back to weight 1.0. The coverage
  test itself is unaffected (substring on normalised text), but weights for such entities
  are flat.
- **Two datasets, one corpus size each.** MuSiQue's paragraph-identity corpus is small
  (21,100 paragraphs); 2Wiki's pool is denser. The pool ceilings differ by 0.10 and this
  drives much of the between-dataset difference.
- **Fixed constants, no search.** `mu`, the bonus, and the exponent were not tuned; a
  search would move CESR's numbers and would also invalidate the comparison.

## Files

- `experiments/stages/lab/cesr.py` — constraints, corpus view, selector, metrics, gate
- `experiments/stages/run_cesr.py` — runner
- `experiments/stages/results/cesr.json` — all numbers, paired CIs, slices, provenance
  samples (25 queries per budget per dataset, with covered/uncovered constraints)

Reproduce (bit-identical):

```sh
cd /Users/asmit/Personal/algo-serp-stages
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_cesr.py
```

`--limit 200` runs the wiring slice. `PYTHONHASHSEED=0` is required: the imported EBR
channels accumulate bridge weights in set-iteration order.
