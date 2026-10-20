# CESR-code: the evidence-set objective with exact constraints

CESR-1 (`CESR.md`) asked whether a coverage objective beats rank truncation when the
constraints are *fuzzy*: capitalised prose entities matched by substring. It found that
the selector saturates the coverage ceiling and still misses evidence, because
constraint coverage does not imply evidence completeness.

This is the same objective with the opposite kind of constraint. A code graph has
typed, exact relations — `CALLS`, `IMPORTS`, `EXPORTS`, `DEFINED_IN`. A constraint is
not "the chunk contains the string `performer`" but "this chunk is in
`callers[tokenStem]`". There is no matching ambiguity left in the predicate. If the
objective still fails here, it fails because of the objective, not because of the
constraint extractor's noise.

## Read this before the numbers

This is a **capability test with oracle constraints**, twice over.

1. **The qrels are graph-derived.** Same 250 synthetic queries and same qrels as
   `CODE_GRAPH.md` (`generate_queries(graph, per_type=50, seed=1729)`). The relations
   the qrels were built from are the relations the constraints test. A positive result
   means retrieval can *execute* the relation, not that the graph is semantically right
   and not that a human would call those chunks relevant.
2. **The constraints are oracle.** They are built from query metadata (`type`,
   `symbols`, `module`, `chain`) and never from `qrels`, but nothing here parses a
   natural-language question. This is the upper bound of a deployable system.
   Label: **oracle-constraint CESR-code**.

Constants were fixed before the run and not tuned: budgets `500/1000/2000`, pool depth
100 per channel, `mu=0.2`, cost exponent `0.35`, path bonus `0.25`, weight `1.0` per
constraint, bootstrap 1,000 resamples, seed 1729.

One diagnostic (the text-ambiguity slice in §"The benchmark is ambiguous too") was
added after the first look at the per-type table. It is post-hoc and labelled as such
everywhere it appears.

## Method

**Candidate pool.** Identical to `CODE_GRAPH.md`: Porter BM25 top-100, entity channel
top-100, graph channel top-100, unioned with membership provenance. 150.6 chunks on
average (min 136, max 172) out of 289. Pool membership is budget-independent. A second
variant uses all 289 chunks, to separate pool recall from selector quality.

**Constraints.** One coverage set per atomic constraint, as a set of chunk ids the
graph says satisfy it, plus one joint relation that ties the set together. `IMPORTS` is
a file-level predicate: the chunks whose file imports the module.

| type | constraints (weight 1.0 each) | joint |
|---|---|---|
| `same_file` | `CALLS(symbols[0])`, `DEFINES_SAME_FILE(symbols[1])` | same file |
| `import_call` | `CALLS(symbols[0])`, `DEFINES(symbols[0])`, `IMPORTS(module)` | caller file imports module |
| `two_hop` | `CALLS(chain[0])`, `CALLS(chain[1])` | middle defines `chain[1]` and upper calls it |
| `export_import` | `EXPORTS(file)`, `DEFINES(symbols[0])`, `IMPORTS(module)` | all three in one file |
| `calls_two` | `CALLS(symbols[0])`, `CALLS(symbols[1])`, `DEFINES` of both | one chunk calls both |

`DEFINES_SAME_FILE` is `DEFINES(symbols[1])` plus the same-file joint with the calling
chunk: the file equality is what pins the anchor, so it is expressed as the joint
rather than folded into the predicate. Likewise `import_call`'s "on the caller's file"
is the same-file joint between the `CALLS` chunk and the `IMPORTS` chunks.

**Selector.** Deterministic greedy coverage. At each step, among candidates that fit the
remaining budget:

```
score = (new constraint weight + 0.25 * newly satisfied joints)
        / max(dl + 10, 1)^0.35
        - 0.2 * max_symbol_jaccard(candidate, selected)
```

Weights are `1.0` per constraint, normalised to sum to one, so coverage is a fraction.
The path bonus is fixed at `0.25` per joint, added outside the cost division, following
CESR-1's pair bonus. Redundancy is the maximum pairwise Jaccard of symbol sets
(defined + called + mentioned symbols). Stop on budget exhaustion, when nothing fits, or
when the best score is non-positive. Output is the selection in order with per-step
gain, bonus, score, redundancy and covered constraints.

**Baselines at the same token budget.** Porter BM25, entity channel, graph channel and
RRF(Porter, entity, graph) — the `CODE_GRAPH.md` primary — each truncated in rank order
with the repo's skip-and-continue convention using `ix.dl`.

**Metrics.** `all@5` / `all@10` are prefix-completeness over the selection order (all
qrel chunks inside the first k), `recall` over the whole selection, `coverage` is the
normalised constraint weight covered, plus tokens, selected count, redundancy and
`nDCG@10`. Paired bootstrap 95% CIs vs RRF at the matched budget, 1,000 resamples,
seed 1729.

## Corpus and whether the budget binds

| | value |
|---|---:|
| files / chunks | 14 / 289 |
| corpus tokens | 28,659 |
| mean / median / max chunk `dl` | 99.2 / 66 / 850 |
| mean gold evidence per query | **309 tokens** |
| gold evidence fits in 500 / 1000 / 2000 | 0.820 / 0.976 / 1.000 |
| budget as share of corpus (500 / 1000 / 2000) | 1.7% / 3.5% / 7.0% |

The budgets bind on *candidate lists*, not on *evidence*. Gold sets are 2–4 chunks
totalling ~309 tokens, so at 1000 and 2000 tokens every gold set fits with room to
spare, and at 500 four queries in five do. This is why `all@5` equals `all@10` on every
**CESR** row below: the selector never returns more than 3 chunks (histogram at budget
500: 1 chunk ×17, 2 ×184, 3 ×49), and the largest selection at a 2000-token budget is
1,013 tokens. The baselines do spend their budget (5.3 to 15.1 chunks), so their `all@5`
and `all@10` differ, and CESR's token column is a *selection-size* measure rather than a
truncation-pressure one. The secondary top-k view is in §"Top-k view".

## Candidate-pool ceiling

| | pool (3×100 union) | all 289 chunks |
|---|---:|---:|
| pool size | 150.6 | 289 |
| gold chunks in pool | 0.995 | 1.000 |
| queries with **all** gold in pool | 0.988 | 1.000 |
| mean achievable constraint coverage | 0.996 | 1.000 |
| queries where every constraint is coverable | 0.992 | 1.000 |

Unlike CESR-1 (0.732 / 0.576 all-gold-in-pool), the pool is not the limiting factor
here: three channels over 289 chunks cover almost everything. That makes this a clean
test of the selector. It also means the `all_chunks` variant is a small correction
(+0.008 `all@10`, see below), not a different experiment.

## Results (250 queries, pool variant)

`cover` = normalised constraint weight covered; `redun` = mean pairwise symbol Jaccard
of the selected set; `n` = chunks selected.

| budget | system | all@5 | all@10 | recall | cover | redun | tokens | n | nDCG@10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | Porter trunc | 0.132 | 0.200 | 0.460 | 0.701 | 0.120 | 484 | 5.3 | 0.376 |
| 500 | entity trunc | 0.220 | 0.228 | 0.481 | 0.565 | 0.140 | 295 | 2.5 | 0.458 |
| 500 | graph trunc | 0.152 | 0.216 | 0.468 | 0.656 | 0.088 | 483 | 5.3 | 0.416 |
| 500 | RRF trunc | 0.332 | 0.372 | 0.597 | 0.751 | 0.116 | 483 | 4.9 | 0.512 |
| 500 | **CESR-code** | **0.592** | **0.592** | **0.803** | **0.951** | 0.100 | **215** | 2.1 | **0.777** |
| 1000 | Porter trunc | 0.144 | 0.260 | 0.546 | 0.754 | 0.115 | 973 | 8.6 | 0.407 |
| 1000 | entity trunc | 0.264 | 0.296 | 0.544 | 0.612 | 0.155 | 448 | 3.4 | 0.493 |
| 1000 | graph trunc | 0.164 | 0.340 | 0.589 | 0.736 | 0.092 | 971 | 8.4 | 0.469 |
| 1000 | RRF trunc | 0.376 | 0.536 | 0.695 | 0.818 | 0.116 | 971 | 8.0 | 0.561 |
| 1000 | **CESR-code** | **0.672** | **0.672** | **0.841** | **0.978** | 0.104 | **246** | 2.2 | **0.805** |
| 2000 | Porter trunc | 0.152 | 0.264 | 0.653 | 0.803 | 0.102 | 1995 | 15.1 | 0.412 |
| 2000 | entity trunc | 0.272 | 0.340 | 0.572 | 0.627 | 0.155 | 569 | 3.9 | 0.505 |
| 2000 | graph trunc | 0.176 | 0.332 | 0.678 | 0.808 | 0.077 | 1997 | 14.8 | 0.474 |
| 2000 | RRF trunc | 0.364 | 0.600 | 0.765 | 0.856 | 0.095 | 1997 | 13.7 | 0.579 |
| 2000 | **CESR-code** | **0.676** | **0.676** | **0.843** | **0.980** | 0.104 | **249** | 2.2 | **0.807** |

Paired bootstrap CESR-code − RRF, 1,000 resamples, seed 1729:

| budget | metric | Δ | 95% CI | W/T/L |
|---:|---|---:|---|---:|
| 500 | all@5 | +0.2600 | [+0.1920, +0.3240] | 73/169/8 |
| 500 | all@10 | **+0.2200** | **[+0.1520, +0.2840]** | 65/175/10 |
| 500 | recall | +0.2067 | [+0.1687, +0.2443] | 117/118/15 |
| 500 | coverage | +0.2000 | [+0.1660, +0.2337] | 112/130/8 |
| 500 | nDCG@10 | +0.2654 | [+0.2300, +0.2985] | 197/29/24 |
| 1000 | all@10 | **+0.1360** | **[+0.0720, +0.2000]** | 50/184/16 |
| 1000 | recall | +0.1460 | [+0.1053, +0.1820] | 93/139/18 |
| 1000 | coverage | +0.1597 | [+0.1280, +0.1897] | 95/148/7 |
| 1000 | nDCG@10 | +0.2444 | [+0.2079, +0.2767] | 203/21/26 |
| 2000 | all@10 | **+0.0760** | **[+0.0160, +0.1360]** | 42/185/23 |
| 2000 | recall | +0.0783 | [+0.0413, +0.1117] | 68/155/27 |
| 2000 | coverage | +0.1243 | [+0.0937, +0.1540] | 78/162/10 |
| 2000 | nDCG@10 | +0.2280 | [+0.1924, +0.2593] | 197/20/33 |

Token ratio CESR/RRF: **0.45 / 0.25 / 0.12** at 500/1000/2000. Stated the other way
round: CESR at 500 tokens (0.592, 215 tokens) is within 0.008 of RRF at 2000 tokens
(0.600, 1,997 tokens), i.e. it recovers 99% of a nine-times-larger budget's completeness.

The gain shrinks as the budget grows because RRF's advantage is exactly the budget:
truncating a fused list buys more evidence with more tokens, while CESR stops when the
constraints are covered (mean 2.2 chunks, 249 tokens at a 2000-token budget). This is
the same shape as CESR-1 and it is the central diagnostic, not a footnote.

For reference, RRF truncation over Porter at budget 2000: `all@10` +0.336
[+0.272, +0.396], `recall` +0.112 [+0.081, +0.143] — the baseline is the strong
`CODE_GRAPH.md` primary, not a straw man.

### all_chunks variant

Selecting from all 289 chunks instead of the 150.6-chunk union:

| budget | all@5 | all@10 | recall | cover | tokens | n | Δ all@10 vs RRF | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 500 | 0.596 | 0.596 | 0.809 | 0.955 | 216 | 2.1 | +0.224 | [+0.160, +0.288] |
| 1000 | 0.680 | 0.680 | 0.849 | 0.982 | 248 | 2.2 | +0.144 | [+0.080, +0.208] |
| 2000 | 0.684 | 0.684 | 0.851 | 0.984 | 252 | 2.2 | +0.084 | [+0.020, +0.144] |

Pool recall costs 0.8 points of `all@10` at 2000 (`0.676` → `0.684`) and 0.4 at 500.
**The selector is not pool-limited and not optimiser-limited; it is constraint-limited.**
That is the whole result in one line.

## Constraint coverage against its own ceiling

| budget | coverage | ceiling | all constraints covered | all@10 given full coverage | all@10 overall |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.951 | 0.996 | 0.848 | 0.698 | 0.592 |
| 1000 | 0.978 | 0.996 | 0.928 | 0.724 | 0.672 |
| 2000 | 0.980 | 0.996 | 0.932 | **0.725** | 0.676 |

CESR-code reaches 98.4% of the achievable coverage (0.980 of 0.996). There is nothing
for a better optimiser to win. And yet, when every constraint is covered, the returned
set contains the full evidence set only **72.5%** of the time.

Why: **the objective maximises existence, the qrels demand the universal closure.**

- `two_hop` qrels are the middle chunk **plus every caller** of the middle. The
  constraint set has one `CALLS(chain[1])` predicate, satisfied by *one* caller. The
  selector stops as soon as one is selected; the other callers are gold and stay out.
- `import_call` qrels name *one* caller, but 17 chunks on average satisfy "calls W in a
  file that imports M". The selector picks one of the 17 and coverage is complete.

Classifying every missing gold chunk at budget 2000 (95 chunks over 250 queries):

| cause | share of missing gold | share, unique-text queries only |
|---|---:|---:|
| not in the candidate pool | 0.032 | 0.000 |
| **in the pool and satisfies an already-covered constraint** | **0.968** | **1.000** |
| in the pool but satisfies no constraint | 0.000 | 0.000 |

Every miss that is not a pool miss is a *substitution*: another chunk already filled
that constraint's role. Adding coverage cannot fix it — coverage is already 1.0 on those
queries — and neither can a better pool. Only a constraint that is universal
("all callers of X") or discriminating ("this caller, not that one") would.

This is the exact, measurable form of CESR-1's MuSiQue finding. There, generic relation
cues were satisfied by distractors; here, exact relations are satisfied by *equally
correct* alternatives. The failure mode survived the constraint extractor being made
perfect.

## The benchmark is ambiguous too

`generate_queries` dedupes on `(type, qrels)`, not on question text. The 250-query set
contains repeated question texts with **different** qrels:

| type | queries | distinct texts | texts with conflicting qrels | queries affected |
|---|---:|---:|---:|---:|
| same_file | 50 | 50 | 0 | 0 |
| import_call | 50 | 23 | 8 | **35** |
| two_hop | 50 | 35 | 10 | **25** |
| export_import | 50 | 50 | 0 | 0 |
| calls_two | 50 | 47 | 2 | 5 |

Two queries with identical text, e.g. *"Which chunk calls storeMemory in a file that
imports ../src/engine/engine.js?"*, have qrels `top@L26` in one and `top@L67` in the
other. No retriever can satisfy both. This is a defect of the query generator, not of
CESR; it applies equally to every system in the table and to `CODE_GRAPH.md`.

The honest split (`all@10`, budget 2000):

| question text | n | Porter | RRF | CESR-code | CESR coverage |
|---|---:|---:|---:|---:|---:|
| unique in the 250-set | 185 | 0.319 | 0.768 | **0.854** | 0.985 |
| repeated with conflicting qrels | 65 | 0.108 | 0.123 | 0.169 | 0.967 |

On the 65 ambiguous queries every system collapses to 0.11–0.17, as expected. On the
185 unambiguous ones CESR-code is +0.086 over RRF. Decomposing the 0.312 gap between
0.676 and the 0.988 pool ceiling: **0.178 is the text-ambiguity split** (giving the 65
ambiguous queries the unambiguous rate would put CESR at 0.854), and **the remaining
0.134 is the existence/universality gap** — on unambiguous queries the pool ceiling is
1.000 and CESR reaches 0.854. The pool itself costs 0.012.

Both halves matter, and neither is fixable by a better selector.

## Per-type and per-support

`all@10` by type (`unique` = share of queries where exactly one chunk satisfies the full
answer conjunction; `ceiling` = share with all gold in the pool):

| type | n | Porter | RRF | CESR@500 | CESR@1000 | CESR@2000 | ceiling | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| same_file | 50 | 0.580 | **0.980** | 0.920 | 0.960 | 0.980 | 1.000 | 0.980 |
| import_call | 50 | 0.240 | **0.360** | 0.160 | 0.200 | 0.200 | 1.000 | 0.100 |
| two_hop | 50 | 0.020 | 0.300 | 0.540 | 0.600 | **0.600** | 0.940 | 0.900 |
| export_import | 50 | 0.000 | 0.820 | 0.940 | 0.980 | **0.980** | 1.000 | 0.020 |
| calls_two | 50 | 0.480 | 0.540 | 0.400 | 0.620 | **0.620** | 1.000 | 0.640 |

`unique` counts chunks satisfying the full answer conjunction, so it reads as ambiguity
for four types but not for `export_import`: that answer set is deliberately *all* export
chunks of the pinned file (1–2), which is why its `unique` is 0.02 while it scores 0.98.

Restricted to the 185 unique-text queries, budget 2000:

| type | n | RRF | CESR |
|---|---:|---:|---:|
| same_file | 50 | 0.980 | 0.980 |
| import_call | 15 | **0.800** | 0.600 |
| two_hop | 25 | 0.600 | **0.800** |
| export_import | 50 | 0.820 | **0.980** |
| calls_two | 45 | 0.556 | **0.689** |

Three readings:

- **`two_hop` looks like the clean win, but it is the leakiest row.** The answer chunk
  contains neither query symbol; only a caller-of-a-caller can produce it. 0.800 vs
  0.600 on unambiguous queries, 0.600 vs 0.300 overall. However, the oracle constraint
  metadata contains `chain[1]`, the hidden bridge name that also defines the qrels; the
  natural-language query does not contain it. Treat this row as an oracle upper bound,
  not a deployable signal. It reproduces `CODE_GRAPH.md`'s finding that the graph
  channel is a bridge specialist, but the constraint used here is stronger than the
  query text.
- **`import_call` is a loss** (0.200 vs 0.360 overall, 0.600 vs 0.800 on the 15
  unambiguous queries). This is the row that keeps the result honest. With 17 chunks
  satisfying the constraint set, coverage has no tie-breaker; RRF's lexical surface
  happens to prefer the chunk the generator drew as gold. A coverage objective needs a
  *unique* satisfying set to beat relevance, and here there isn't one.
- **`same_file` and `export_import` are the circular rows.** Both are satisfied by a
  small closed set of chunks the graph channel also traverses; CESR ties RRF at 0.98 on
  `same_file` and wins `export_import` 0.98 vs 0.82, but neither should be read as a
  general retrieval gain.

`all@10` by support size (budget 2000):

| support | n | Porter | RRF | CESR |
|---|---:|---:|---:|---:|
| 2 | 182 | 0.231 | 0.665 | **0.758** |
| 3 | 64 | 0.375 | 0.453 | **0.484** |
| 4 | 4 | 0.000 | 0.000 | 0.000 |

The gain is largest at support 2 (`+0.093`) and nearly gone at support 3 (`+0.031`),
which is consistent with the existence/universality gap: every extra required chunk is
another chance to substitute.

## Top-k view

Budgets do not bind the evidence here, so the comparison is also stated as plain top-k,
which is what `CODE_GRAPH.md` measures:

| system | set size | `all@10` |
|---|---:|---:|
| RRF(Porter, entity, graph), rank-prefix top-10 (`CODE_GRAPH.md`) | 10 | 0.608 |
| RRF(Porter, entity, graph), 2000-token truncation (this run) | 13.7 | 0.600 |
| Porter BM25, 2000-token truncation (this run) | 15.1 | 0.264 |
| **CESR-code, 2000-token budget** | **2.2** | **0.676** |

The first two rows agree to 0.008, so token truncation and rank-prefix top-10 are the
same comparison at this corpus size. CESR returns 2.2 chunks where the baseline returns
10–15 and is complete on 6.8 more points of queries. The comparison is between returned
sets, not between rankings.

## Provenance (budget 500)

| pool membership of selected chunks | share |
|---|---:|
| porter + entity + graph | 0.575 |
| porter + graph | 0.256 |
| porter only | 0.126 |
| graph only | 0.032 |
| porter + entity | 0.011 |
| **outside Porter top-100** | **0.032** |

Only 3.2% of what CESR selects is outside Porter's top-100, against 18–22% in CESR-1.
Constraint-based selection on code is much more lexically reachable than on prose
paragraphs: the constraint names a symbol, and the symbol is in the chunk.

## Latency (CPU, 250 queries, one run)

| | ms/query |
|---|---:|
| retrieval (3 channels + RRF + pool) | 2.5 |
| constraint construction | 0.01 |
| CESR selection, 150-chunk pool (per budget) | 1.8 |
| CESR selection, all 289 chunks (per budget) | 3.5 |
| full evaluation (all systems, 3 budgets, 2 variants) | 4.7 s |

Latency varies by ~±30% between runs on this machine; every other number in this
document is byte-identical across runs. Selection is ~40% of the measured per-query cost
here, unlike CESR-1's 2–5%. The constant is the Python-level joint check per candidate
per step; with ≤3 steps and ≤4 constraints there is no algorithmic weight to it, and
vectorising it is a mechanical change. At 289 chunks this is not a scaling claim.

## Gate

Pre-registered rule: *CESR-code must beat RRF(Porter, entity, graph) on `all@10` at the
same budget on at least one budget, or match it with ≥20% fewer tokens.*

| budget | Δ all@10 vs RRF | 95% CI | token ratio | beats | matches with fewer tokens |
|---:|---:|---|---:|---|---|
| 500 | +0.2200 | [+0.1520, +0.2840] | 0.45 | yes | n/a (beat) |
| 1000 | +0.1360 | [+0.0720, +0.2000] | 0.25 | yes | n/a (beat) |
| 2000 | +0.0760 | [+0.0160, +0.1360] | 0.12 | yes | n/a (beat) |

**Gate PASSED** on all three budgets by the beat branch (CI lower bounds above zero).
The fewer-tokens branch was not triggered: `budgets_matching_with_fewer_tokens` is `[]`
in the JSON. The token ratios are 0.45 / 0.25 / 0.12. The 2000 budget is the thin one:
+0.076 with a CI lower bound of +0.016. The 500 and 1000 results are nowhere near the
boundary, so the pass does not depend on that margin.

Mechanically applying the rule answers the pre-registered question. It does not answer
the question the rule was a proxy for, which is whether coverage *implies* completeness.
It does not: 0.725.

## Verdict

**CESR-code is a real packing objective with a measured gain over truncation, and still
not a retrieval algorithm.**

1. **Against rank truncation, it wins.** `all@10` +0.220 / +0.136 / +0.076 at
   matched budgets, all CIs above zero, with 12–45% of the tokens, and it wins on
   `recall`, `coverage` and `nDCG@10` in the same direction. It is not a metric artifact.
2. **The mechanism is constraint execution, not ranking.** The gain is concentrated in
   `two_hop` (+0.300 overall, +0.200 on unambiguous queries), the only type whose gold
   contains neither query symbol. On the lexically reachable types it is a tie
   (`same_file` 0.98 vs 0.98) or a loss (`import_call` 0.20 vs 0.36).
3. **Coverage does not imply completeness, even when the constraints are exact.**
   Full coverage yields the full evidence set 72.5% of the time; 96.8% of the missing
   gold chunks are alternative satisfiers of an already-covered constraint. This is the
   MuSiQue result reproduced with a perfect extractor: the problem was never the
   entities, it is that *existential coverage cannot express a universally quantified
   evidence set*.
4. **Therefore the deployable form is unchanged from CESR-1:** retrieve broadly with
   RRF, then pack by exact constraints when constraints exist and are near-unique. As a
   ranker it has nothing to offer — it returns 2.2 chunks in gain-per-token order and
   stops, and its `nDCG@10` lead is the precision of a two-chunk set, not a ranking of
   the collection.
5. **What would change the verdict.** A constraint language with universal quantifiers
   ("every caller of X") and identity ("this caller, chosen by a second constraint"),
   plus a constraint extractor that produces them from text. `import_call` is the
   evidence that identity is the hard part: exact `CALLS` + `IMPORTS` still leaves 17
   candidates. Do not build the extractor before the constraint language can express the
   answer.

## Honest limits

- **Oracle constraints.** Built from `type`, `symbols`, `module`, `chain`; nothing parses
  the question. An automatic extractor will be worse, and `import_call` shows that even
  perfect extraction does not make the answer set unique.
- **Graph-derived qrels.** The relations that define the constraints also generated the
  labels, and the graph channel traverses them. The circularity is strongest on
  `export_import` (which is near-tautological). `two_hop` is also oracle-leaky: its
  constraint uses `chain[1]`, the hidden bridge name that defines the qrels, even though
  the natural-language query does not contain it.
- **Benchmark ambiguity.** 65/250 queries share their exact text with a query that has
  different gold. This caps every system and is disclosed as a post-hoc slice, not
  corrected — correcting it would be changing the benchmark after seeing results.
- **Budgets do not bind the evidence.** Mean gold is 309 tokens; 82% of gold sets fit in
  500 tokens. CESR's `all@5` = `all@10` at every budget, and its token column is a
  selection-size comparison, not a truncation-pressure one. The binding constraint is the
  number of chunks, not the token count.
- **Small corpus.** 14 files, 289 chunks, one repository, no human labels. A top-10 set
  is 3.5% of the corpus.
- **`IMPORTS` is a weak predicate.** At file level it covers 85 chunks on average (max
  195), so that constraint's coverage is nearly free; the joint, not the constraint, does
  the work. Mean constraint set size is 23 chunks, median 2.
- **Redundancy is measured on symbol sets**, not on text. CESR's mean pairwise Jaccard is
  0.100 / 0.104 / 0.104 against RRF's 0.116 / 0.116 / 0.095: lower at 500 and 1000,
  slightly higher at 2000 where it returns only 2.2 chunks. As in CESR-1, redundancy was
  never the main gap.
- **Fixed constants, no search.** `mu`, the exponent, the bonus and the weights were not
  tuned. A search would move the numbers and invalidate the comparison.
- **Determinism.** Every metric is byte-identical across runs; only the latency fields
  differ.

## Files

- `experiments/stages/lab/cesr_code.py` — constraints, coverage sets, joints, selector,
  truncation, metrics, gate
- `experiments/stages/run_cesr_code.py` — runner
- `experiments/stages/results/cesr_code.json` — all numbers, CIs, slices, ceilings,
  missing-gold causes, 60 provenance samples

Reproduce:

```sh
cd /Users/asmit/Personal/algo-serp-stages
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_cesr_code.py
```

~6 s. No data downloads; the corpus is the repo itself.
