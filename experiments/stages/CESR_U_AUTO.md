# CESR-U-auto: constraints parsed from the question, and what that costs

CESR-U (`CESR_U.md`) executed universal/identity constraints perfectly and left one question
explicitly open: every constraint came from the generator's metadata, so the result was an
upper bound, not a system. This is the follow-up that decides whether a deployable
"structured mode" exists. The metadata is deleted: the only query input is `query["text"]`
plus the graph.

The answer in one line: **structured mode survives, because wherever the text names the
entities the parsed constraint set is *identical* to the oracle one (194/194) and the output
is identical too (max absolute difference 0.000 on 149 text-unique queries). It is not a
general extractor: the text supplies the names on 77.6% of queries, `two_hop` (20% of the
set) cannot be parsed at all, and the cue phrases are the query generator's own templates.**

## Read this before the numbers

1. **The qrels are graph-derived.** Same 250 synthetic queries as `CODE_GRAPH.md` /
   `CESR_U.md` (`generate_queries(graph, per_type=50, seed=1729)`). A positive result means
   retrieval can execute a text-derived relation, not that a human would call those chunks
   relevant.
2. **The cue phrases are the benchmark's templates.** That is unavoidable: the questions
   were generated from the same five sentence patterns, so "same file", "calls both",
   "imports"+"calls", "exports"+"imports"+"defines" and "calls a function that calls" cover
   250/250. Intent classification is therefore a **template match, not language
   understanding**, and the 100% intent/symbol/module agreement below is near-tautological.
   What this experiment *does* test is the rest of the pipeline: whether a text-derived plan
   is the same object as a metadata-derived plan, what it costs when it is not, and whether
   the selector cares where the names came from. It does not test paraphrase robustness.
3. **The corpus moved since `CESR_U.md`.** The repository is the corpus, and the working
   tree carries an uncommitted edit to `scripts/stage-benchmark.js` made after that run:
   303 chunks and 143 symbols now against 289/132, 188 text-unique queries against 185. Every system here is measured on the
   current graph, so the oracle reference numbers differ slightly from the published CESR-U
   table (e.g. text-unique `all@10` 0.957, not 0.924). Within this document all comparisons
   are on one corpus.
4. **Primary view is pre-registered, not chosen.** Parsed text-unique: text-unique queries
   (dedupe by normalised question text, as in CESR-U) intersected with "the parser produced a
   constraint set". The 39 text-unique queries it cannot plan are reported as their own slice
   and are also included in the end-to-end numbers. No constant, cue or rule was tuned on the
   250 queries; `qrels` is never read by the parser, the plan builder or the selector.
5. **Two parser decisions were fixed before any retrieval number** and are disclosed because
   they are judgement calls, not deductions:
   - path spans are excised before symbol matching, otherwise `./tokenize.js` yields the
     defined symbol `tokenize` and 49/50 `export_import` questions become unsupported for a
     tokenisation reason rather than an information reason;
   - a module mention that occurs verbatim in a file's import list uses that literal relation;
     `resolve_module` is the fallback. A relative specifier only means something relative to
     its importer, and `IMPORTS(file, module)` in CESR-U is literally
     `module in file_imports[file]`.

## Method

### Parser v1 (frozen)

| step | rule |
|---|---|
| symbols | tokens exactly matching a key of `graph.defined`. `code_graph.query_symbols` supplies the candidate set (read-only), filtered to exact-case token matches and ordered by first appearance |
| modules | spans matching `./…`, `../…` or `*.js/.ts/...`, one mention; literal `file_imports` match first, `code_graph.resolve_module` fallback |
| intent | fixed cue phrases, checked two_hop → export_import → calls_two → import_call → same_file |
| unsupported | no guessing. The query falls back to RRF and is reported separately |
| `two_hop` | **always unsupported**: its text names only the bottom callee, the middle bridge chunk is the hidden `chain[1]` that also defines the qrels, and nothing here reconstructs it |

Cue table, exactly as pre-registered: `same_file` = "same file"; `calls_two` = "calls both";
`import_call` = "imports" + "calls"; `export_import` = "exports" + "imports" + "defines";
`two_hop` = "calls a function that calls" (detected, then declared unsupported). No cue was
added, and no stop-word list, arity heuristic or symbol-vocabulary widening was applied.

### Constraint plans

Built from parsed slots with CESR-U's operators, weights and plan objects, unchanged:

| intent | parsed slots | witness | mandatory | predicate |
|---|---|---|---|---|
| `same_file` | callee, anchor symbol | `CALLS_FILE_OF(a,b)` | `ALL_DEFINES_IN_FILE(b)` | `SAME_FILE` |
| `import_call` | symbol, module | `CALLS_IMPORTING(X,M)` | `EXACT/ALL_DEFINES(X)` | `IMPORTS(file,M)` |
| `export_import` | symbol, module | — (file must be **pinned uniquely** by `imports M ∧ defines X`, else unsupported) | `ALL_EXPORTS(file)`, `EXACT/ALL_DEFINES_IN_FILE` | `IMPORTS(file,M)` |
| `calls_two` | two symbols | `CALLS_BOTH(X,Y)` | `EXACT/ALL_DEFINES` of both | — |
| `two_hop` | — | — | — | unsupported |

Unsupported at plan level: symbol arity wrong, module arity wrong, module unresolved, no
importing callers, no common caller, no export units, file not uniquely pinned.

### Selector and evaluation

CESR-U's closure-first selector verbatim (`cu.select`), same constants: budgets
`500/1000/2000`, pool depth 100, `mu=0.2`, cost exponent `0.35`, path bonus `0.25`,
`LEX_TIEBREAK=0.02`, RRF `k=60`, 1,000 bootstrap resamples, seed 1729.

Systems: Porter BM25, entity, graph, RRF(Porter, entity, graph), **CESR-U-auto**,
CESR-U-oracle (reference, metadata constraints) and CESR-code (reference). Metrics:
`all@5`/`all@10`, recall, coverage, tokens, selected, redundancy, nDCG@10. Coverage is
reported under the auto constraint set (and, for the reference lanes, under the oracle set)
with the count of rows it is available for.

CESR-U-auto's policy on an unplannable query: return the RRF truncation, unchanged. It is
therefore identical to RRF on the 39 text-unique unsupported queries **by construction** —
that is the honest cost of not guessing, not a result.

## Parser coverage

| intent | queries | planned | planned % | unsupported reason | unique answer (planned) | mean options |
|---|---:|---:|---:|---|---:|---:|
| `same_file` | 50 | 44 | 0.880 | `symbol_arity` ×6 | 0.955 | 1.05 |
| `import_call` | 50 | 50 | 1.000 | — | 0.140 | **15.30** |
| `two_hop` | 50 | **0** | 0.000 | `bridge_name_not_in_text` ×50 | — | — |
| `export_import` | 50 | 50 | 1.000 | — | 1.000 | 1.00 |
| `calls_two` | 50 | 50 | 1.000 | — | 0.660 | 3.24 |
| **all** | **250** | **194** | **0.776** | | **0.680** | |

Text-unique view: **149/188 planned (0.793)**, 39 unsupported. Of the 149 planned,
**132 are constraint-determined (0.886)**; over all 188 text-unique queries the parsed
constraint set determines the answer on 132 (0.702). By intent the text-unique split is
`same_file` 50/44, `import_call` 14/14, `two_hop` 33/0, `export_import` 50/50,
`calls_two` 41/41.

**Agreement with the oracle metadata (diagnostic only, never an input):** on the 194 planned
queries the parser reproduces the generator's intent 194/194, its symbol list 194/194 and its
module string 100/100. The plans are not merely equivalent — they are the same objects
(labels, kinds, member sets, groups, weights, derived anchors, predicates and witness index
all equal, 194/194).

### Parser errors on a hand-checked sample

15 rows read by hand: the first two planned per intent, all six `symbol_arity` failures, and
one `two_hop` (`results/cesr_u_auto.json` → `hand_check`, each with `reviewed: true`).

| class | n | what it is | verdict |
|---|---:|---|---|
| planned | 8 sampled (194 total) | symbols and module occur verbatim; every label names a relation in the text; nothing invented | **0 errors** |
| `symbol_arity` | 6 (all text-unique, all `same_file`) | the callee is an **imported** name — `Database`, `fileURLToPath` — so it is in `graph.callers` but has no `graph.defined` key. One symbol is found instead of two | parser refuses; fallback. Honest failure, not a guess |
| `bridge_name_not_in_text` | 50 (`two_hop`) | the middle bridge chunk is not in the question | unsupported by design |

The six `symbol_arity` rows are the only *mechanical* parser error class, and it is a
vocabulary-boundary error: the frozen rule is `graph.defined`, the callee is defined in
`node:url`/`better-sqlite3`. Widening the vocabulary to `graph.callers` keys would recover
those six; it is not done here because the rule was frozen, and it would be worth little
(RRF already scores 0.900 on `same_file`, so the ceiling on that repair is about +0.01
`all@10`).

## Results: parsed text-unique (primary, n=149)

`cover` = constraint weight satisfied under the **auto** constraint set; `n` = chunks
selected. The CESR-U-auto and CESR-U-oracle rows are identical on this subset (measured:
maximum absolute difference 0.000); both are shown because that identity is the result.

| budget | system | all@5 | all@10 | recall | cover | redun | tokens | n | nDCG@10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | Porter | 0.201 | 0.282 | 0.591 | 0.595 | 0.088 | 487 | 5.4 | 0.488 |
| 500 | entity | 0.302 | 0.309 | 0.604 | 0.617 | 0.154 | 353 | 2.9 | 0.574 |
| 500 | graph | 0.128 | 0.221 | 0.530 | 0.547 | 0.083 | 486 | 5.6 | 0.501 |
| 500 | RRF | 0.490 | 0.503 | 0.735 | 0.738 | 0.094 | 489 | 5.2 | 0.654 |
| 500 | CESR-code | 0.705 | 0.705 | 0.852 | 0.883 | 0.044 | 207 | 2.0 | 0.862 |
| 500 | **CESR-U-auto** | **0.940** | **0.940** | **0.970** | **1.000** | 0.063 | 296 | 2.3 | **0.973** |
| 500 | CESR-U-oracle | 0.940 | 0.940 | 0.970 | 1.000 | 0.063 | 296 | 2.3 | 0.973 |
| 1000 | Porter | 0.262 | 0.396 | 0.702 | 0.715 | 0.085 | 980 | 8.8 | 0.530 |
| 1000 | entity | 0.416 | 0.443 | 0.700 | 0.705 | 0.158 | 542 | 3.9 | 0.633 |
| 1000 | graph | 0.121 | 0.362 | 0.651 | 0.654 | 0.076 | 976 | 9.2 | 0.546 |
| 1000 | RRF | 0.523 | 0.732 | 0.865 | 0.865 | 0.089 | 980 | 8.7 | 0.715 |
| 1000 | CESR-code | 0.832 | 0.832 | 0.918 | 0.949 | 0.047 | 265 | 2.2 | 0.915 |
| 1000 | **CESR-U-auto** | **0.946** | **0.946** | **0.973** | **1.000** | 0.062 | **297** | 2.3 | **0.975** |
| 1000 | CESR-U-oracle | 0.946 | 0.946 | 0.973 | 1.000 | 0.062 | 297 | 2.3 | 0.975 |
| 2000 | Porter | 0.255 | 0.376 | 0.829 | 0.841 | 0.084 | 1,996 | 16.3 | 0.529 |
| 2000 | entity | 0.450 | 0.510 | 0.743 | 0.745 | 0.162 | 677 | 4.5 | 0.654 |
| 2000 | graph | 0.121 | 0.362 | 0.755 | 0.758 | 0.063 | 1,997 | 16.7 | 0.552 |
| 2000 | RRF | 0.537 | 0.812 | 0.944 | 0.944 | 0.070 | 1,997 | 15.8 | 0.737 |
| 2000 | CESR-code | 0.852 | 0.852 | 0.927 | 0.958 | 0.049 | 273 | 2.2 | 0.921 |
| 2000 | **CESR-U-auto** | **0.946** | **0.946** | **0.973** | **1.000** | 0.062 | **297** | 2.3 | **0.975** |
| 2000 | CESR-U-oracle | 0.946 | 0.946 | 0.973 | 1.000 | 0.062 | 297 | 2.3 | 0.975 |

Paired bootstrap, 1,000 resamples, seed 1729:

| budget | metric | vs | Δ | 95% CI | W/T/L |
|---:|---|---|---:|---|---:|
| 500 | all@10 | RRF | +0.4362 | [+0.3557, +0.5168] | 68/78/3 |
| 500 | all@10 | CESR-code | +0.2349 | [+0.1678, +0.3154] | 36/112/1 |
| 500 | all@10 | CESR-U-oracle | 0.0000 | [0, 0] | 0/149/0 |
| 1000 | all@10 | RRF | +0.2148 | [+0.1477, +0.2886] | 35/111/3 |
| 1000 | all@10 | CESR-code | +0.1141 | [+0.0671, +0.1678] | 18/130/1 |
| 1000 | all@10 | CESR-U-oracle | 0.0000 | [0, 0] | 0/149/0 |
| 2000 | all@10 | RRF | **+0.1342** | **[+0.0738, +0.1946]** | 23/123/3 |
| 2000 | all@10 | CESR-code | +0.0940 | [+0.0535, +0.1411] | 15/133/1 |
| 2000 | all@10 | CESR-U-oracle | 0.0000 | [0, 0] | 0/149/0 |
| 2000 | recall | RRF | +0.0291 | [+0.0033, +0.0570] | 15/131/3 |
| 2000 | coverage | RRF | +0.0559 | [+0.0324, +0.0828] | 19/130/0 |
| 2000 | nDCG@10 | RRF | +0.2376 | [+0.1996, +0.2733] | 129/17/3 |

Token ratio CESR-U-auto / RRF on this subset: **0.60 / 0.30 / 0.15**. The equality with the
oracle lane is measured, not assumed: the maximum absolute difference over `all@5`, `all@10`,
recall, nDCG@10, tokens and selected, across all three budgets, is **0.000**.

### Full coverage against completeness

| subset | system | full coverage | all@10 given full | all@10 |
|---|---|---:|---:|---:|
| parsed text-unique | CESR-U-auto | 1.000 | 0.946 | 0.946 |
| parsed text-unique, **determined (132)** | CESR-U-auto | 1.000 | **1.000** | **1.000** |
| parsed text-unique, ambiguous (17) | CESR-U-auto | 1.000 | 0.529 | 0.529 |
| parsed text-unique, determined (132) | CESR-code | 0.917 | 1.000 | 0.917 |
| parsed text-unique, ambiguous (17) | CESR-code | 0.765 | 0.462 | 0.353 |

The CESR-U result reproduces exactly on the parsed slice: **132/132 determined queries
return the complete evidence set**. Missing gold is 97.9% wrong-alternative, 2.1% universal
incomplete (one chunk: an ambiguous `same_file` query where the anchor group follows the
lexically chosen caller into another file), 0% pool, 0% unconstrained — the same
diagnosis as `CESR_U.md`, now reached through text.

### Candidate-pool ceiling

| | value |
|---|---:|
| mean pool size | 151.2 |
| gold chunks in pool (parsed) | 0.998 |
| parsed queries with all gold in pool | 0.995 |
| mean achievable auto coverage | **1.000** |
| parsed queries where every auto constraint is coverable | **1.000** |

As in CESR-code, the pool is not the limit. What limits the system is whether the text names
the entities, not whether the graph can reach them.

## End to end: the deployable number

Parsed text-unique is the pre-registered subset, but a deployed system cannot know which
queries those are. The honest system number includes the RRF fallback.

| subset | budget | system | all@5 | all@10 | recall | tokens | n |
|---|---|---:|---|---:|---:|---:|---:|
| text-unique (188) | 500 | RRF | 0.431 | 0.452 | 0.707 | 485 | 5.2 |
| text-unique (188) | 500 | **auto** | **0.787** | **0.798** | **0.894** | **332** | 2.9 |
| text-unique (188) | 500 | oracle | 0.952 | 0.952 | 0.976 | 306 | 2.3 |
| text-unique (188) | 1000 | RRF | 0.463 | 0.644 | 0.820 | 977 | 8.4 |
| text-unique (188) | 1000 | **auto** | **0.798** | **0.814** | **0.906** | **437** | 3.3 |
| text-unique (188) | 1000 | oracle | 0.957 | 0.957 | 0.979 | 307 | 2.3 |
| text-unique (188) | 2000 | RRF | 0.484 | 0.739 | 0.902 | 1,997 | 14.9 |
| text-unique (188) | 2000 | CESR-code | 0.824 | 0.824 | 0.918 | 273 | 2.2 |
| text-unique (188) | 2000 | **auto** | **0.809** | **0.846** | **0.925** | **650** | 4.2 |
| text-unique (188) | 2000 | oracle | 0.957 | 0.957 | 0.979 | 307 | 2.3 |
| all 250 | 2000 | RRF | 0.384 | 0.616 | 0.775 | 1,998 | 14.1 |
| all 250 | 2000 | **auto** | **0.636** | **0.668** | **0.832** | **707** | 4.3 |
| all 250 | 2000 | oracle | 0.808 | 0.808 | 0.909 | 332 | 2.3 |
| conflicting (62) | 2000 | RRF | 0.081 | 0.242 | 0.390 | 1,998 | 11.8 |
| conflicting (62) | 2000 | **auto** | 0.113 | **0.129** | 0.551 | 879 | 4.8 |
| conflicting (62) | 2000 | oracle | 0.355 | 0.355 | 0.699 | 410 | 2.3 |

Paired bootstrap, CESR-U-auto vs baseline, `all@10`:

| subset | budget | vs RRF | vs CESR-U-oracle | vs CESR-code |
|---|---:|---|---|---|
| text-unique | 500 | +0.3457 [+0.2713, +0.4149] | −0.1543 [−0.2076, −0.1063] | +0.1064 [+0.0319, +0.1809] |
| text-unique | 1000 | +0.1702 [+0.1117, +0.2287] | −0.1436 [−0.1968, −0.0957] | +0.0053 [−0.0638, +0.0586] |
| text-unique | 2000 | **+0.1064 [+0.0585, +0.1596]** | −0.1117 [−0.1597, −0.0691] | +0.0213 [−0.0426, +0.0798] |
| all 250 | 2000 | **+0.0520 [+0.0040, +0.0961]** | −0.1400 [−0.1840, −0.1000] | −0.0120 [−0.0640, +0.0440] |
| conflicting | 2000 | **−0.1129 [−0.2258, −0.0161]** | −0.2258 [−0.3387, −0.1286] | −0.1129 [−0.2419, +0.0161] |

Three readings:

1. **End to end it still beats fusion**, at every budget, with 68%/45%/33% of the tokens
   (332 vs 485, 437 vs 977, 650 vs 1,997). The gain shrinks with budget for the same reason
   as in CESR-U: the closure is a fixed 2.3 chunks on the parsed 79% and the fallback is RRF
   on the rest.
2. **It does not beat the oracle-constraint systems end to end.** `−0.112` against
   CESR-U-oracle and a tie (`+0.021`, CI spanning zero) against CESR-code at 2000. The
   metadata is worth 11 points of `all@10` on text-unique queries, and that is the measured
   price of reading the question instead of the answer key.
3. **It is worse than RRF on the 62 conflicting-text queries** (−0.113, CI below zero).
   Those are questions whose *identical text* has two different gold sets; the structured
   path commits to one constraint set and therefore to one gold, while RRF's 11.8 chunks
   accidentally cover both. On a production corpus one question has one answer, so this
   slice is a generator artifact — but the mechanism is real: **committing to a witness is
   riskier than returning a wide list when the question is under-specified.**

## Per type (text-unique, parser intent)

`unique` and `cover` are over planned queries; `auto` includes the RRF fallback, so the
`two_hop` row is RRF.

| intent | n | planned | RRF | CESR-code | oracle | auto@500 | auto@1000 | auto@2000 | unique | cover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `same_file` | 50 | 44 | 0.900 | 0.980 | 0.980 | 0.880 | 0.880 | 0.900 | 0.955 | 1.000 |
| `import_call` | 14 | 14 | **0.786** | 0.500 | 0.714 | 0.643 | 0.714 | 0.714 | 0.500 | 1.000 |
| `two_hop` | 33 | **0** | 0.485 | 0.667 | **1.000** | 0.273 | 0.333 | 0.485 | — | — |
| `export_import` | 50 | 50 | 0.800 | 0.980 | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | 1.000 |
| `calls_two` | 41 | 41 | 0.659 | 0.683 | 0.927 | 0.927 | 0.927 | **0.927** | 0.805 | 1.000 |

Decomposition of the 0.111 end-to-end gap to the oracle at 2000:

- **`two_hop`: 0.091.** 33/188 queries × (1.000 − 0.485). Almost the whole gap, and entirely
  information that is not in the text.
- **`same_file`: 0.021.** 50/188 × (0.980 − 0.900), of which the six imported-callee
  fallbacks are the cause of the 0.080 within-type shortfall.
- **`import_call`, `export_import`, `calls_two`: 0.000.** The parsed plan is the oracle plan,
  so there is nothing for the parser to lose.

`import_call` is the row that keeps the whole exercise honest: **the parser is perfect there
(14/14 planned, oracle-identical) and the system still loses to RRF, 0.714 vs 0.786.** A mean
15.3 caller alternatives, 14% unique. Parsing was never the bottleneck for that type; the
query is.

## The null slice: `two_hop`

| | auto (=RRF) | RRF | CESR-code | CESR-U-oracle |
|---|---:|---:|---:|---:|
| all@10 @500 | 0.256 | 0.256 | 0.641 | **1.000** |
| all@10 @1000 | 0.308 | 0.308 | 0.718 | **1.000** |
| all@10 @2000 | 0.462 | 0.462 | 0.718 | **1.000** |

39 text-unique queries, 33 of them `two_hop`. The oracle gets 1.000 because `chain[1]` is
handed to it; the text does not contain `chain[1]`, and the hypothesis that a bridge could be
inferred from "a function that calls X" alone would require choosing among all callers of X,
which is exactly the ambiguity the qrels hide. This is a finding, not a bug to route around:
**one fifth of the benchmark is unanswerable from its text under any constraint extractor
that reads only the text.**

## Gate

Pre-registered rule:

> CESR-U-auto must beat RRF on `all@10` for the parsed text-unique subset with a CI above
> zero, or match it with ≥20% fewer tokens. (Gated at the budgets where gold evidence fits
> in ≥90% of queries: 1000 and 2000. It fits in 81% at 500.)

| budget | Δ all@10 vs RRF | 95% CI | tokens vs RRF | beat | match with ≥20% fewer tokens |
|---:|---:|---|---:|---|---|
| 500 | +0.4362 | [+0.3557, +0.5168] | 0.60 | yes | n/a (beat) |
| 1000 | +0.2148 | [+0.1477, +0.2886] | 0.30 | **yes** | n/a (beat) |
| 2000 | +0.1342 | [+0.0738, +0.1946] | 0.15 | **yes** | n/a (beat) |

**Gate PASSED** by the beat branch at both gated budgets, and at 500 as well. The
fewer-tokens branch was not needed: the token ratio is 0.15 at 2000, so the rule would also
pass on the tie branch.

Two checks that matter more than the mechanical pass:

- **End-to-end, not just on the pre-registered subset:** text-unique `all@10` at 2000 is
  0.846 vs RRF 0.739, `+0.1064 [+0.0585, +0.1596]`; all 250 is `+0.0520 [+0.0040, +0.0961]`.
  The deployable system beats fusion even with the 20.7% fallback.
- **The planned slice is the oracle system**, measured at a maximum absolute difference of
  0.000 across every metric and budget. There is no "auto is a bit worse than oracle" effect
  to explain: parsing either reproduces the constraint set or refuses.

## Verdict

**Structured mode survives automatic constraints — as an executor that is oracle-equivalent
wherever the question names its entities, and as a system that beats fusion end to end. It
does not survive as a general constraint extractor, and this benchmark cannot show that it
would.**

1. **The deployable claim is narrow and clean.** Text → constraints → closure loses *nothing*
   relative to metadata on the queries it can plan: 194/194 identical plans, 149/149 identical
   outputs, 132/132 determined queries complete. The binding constraint is no longer the
   objective, the pool or the optimiser; it is **coverage of the names in the question**,
   which is 77.6% overall and 79.3% on text-unique queries.
2. **The 20.7% it cannot plan is mostly information that is not in the text.** `two_hop` is
   17.6 points of the text-unique set and cannot be parsed by anything that reads only the
   question: the bridge chunk is the hidden answer key. The remaining 3.2 points are a real
   but small vocabulary bug (`Database`, `fileURLToPath` are imported, not defined), worth
   about +0.01 `all@10`.
3. **End to end the deployable system is the best non-oracle system on this benchmark and
   still 11 points behind the oracle.** text-unique `all@10` 0.846 vs 0.957 oracle, +0.106
   over RRF, a tie with oracle-constraint CESR-code. The 11 points are the measured price of
   not having the answer key, and 9.1 of them are `two_hop` alone.
4. **The honest cap on the parser claim is the template.** Intent, symbols and module agree
   100% because the cue list is the generator's own phrasing. This experiment shows the
   *plumbing* is sound and the *selector does not care* where the names came from. It does
   not show that a parser survives paraphrase, and the 6/194 vocabulary failures show where
   a real one would start to break: names that are imported or aliased, not defined.
5. **What would change the verdict.** (a) A bridge extractor that can name `chain[1]` from
   the text — 0.091 of the 0.111 gap, and the only path back to oracle parity. (b) A symbol
   vocabulary that includes imported names — 0.019. (c) A rephrased corpus that keeps the
   answers and changes the sentences — the only way to measure the parser rather than the
   templates. Without (c), "structured mode is deployable" remains a statement about this
   generator's five sentence patterns.

## Honest limits

- **Template parser on a template benchmark.** The cue phrases are the generator's phrasing.
  The 100% intent agreement is not evidence of language understanding, and no paraphrase was
  tested. The six vocabulary failures are the visible edge of that.
- **`two_hop` is unsupported by construction.** 50/250 queries, 33/188 text-unique. It is
  reported as a null slice and is the majority of the gap to the oracle. Reconstructing it
  from metadata would have made the experiment look better and meant nothing.
- **Oracle equality is partly definitional.** The parser reads the same slots the metadata
  encodes, so identical plans on planned queries are expected, not surprising. What is
  informative is the *refusal set* and its measured cost.
- **Graph-derived qrels, one repository, 303 chunks.** No human labels. A 10-chunk set is
  3.3% of the corpus. The corpus is the live repository and has drifted since `CESR_U.md`
  (289 → 303 chunks, 185 → 188 text-unique queries), so cross-document comparisons are
  approximate; all rows here share one graph.
- **Conflicting-text queries get worse, not better** (−0.113 vs RRF at 2000). Committing to a
  witness is a liability when the text is shared by two gold sets. The slice is a generator
  artifact, but the mechanism is not.
- **`all@10` rewards supersets.** CESR-U-auto returns 2.3 chunks on planned queries and 4.2
  with the fallback; the token and selected columns are there to keep that visible.
- **Budget violations at 500.** 18.8% of parsed text-unique queries exceed the budget by a
  mean 208 tokens; 3.4% at 1000; none at 2000. Mandatory evidence is never dropped. The 1000
  and 2000 rows are the matched-budget ones.
- **Two disclosed parser decisions** (path-span excision, literal-first module resolution)
  were fixed before any retrieval number but are judgement calls, and a different choice
  would change the `export_import` slice's module set.
- **Determinism.** Every metric and table is byte-identical across runs (`PYTHONHASHSEED=0`,
  two runs diffed with only the latency block removed); only latency fields vary.
- **Latency** (CPU, 250 queries): parse + plan 0.12 ms/query, retrieval 5.6 ms, CESR-U-auto
  selection 0.05 ms, CESR-code selection 2.9 ms, full evaluation ~4 s.

## Files

- `experiments/stages/lab/cesr_u_auto.py` — frozen parser v1, module resolution, plan
  builders per intent, unsupported policy, pool-ceiling helpers
- `experiments/stages/run_cesr_u_auto.py` — runner, subsets, baselines, paired bootstrap,
  parser diagnostics, hand-check sample, gate
- `experiments/stages/results/cesr_u_auto.json` — every table, CI, per-intent slice,
  determinacy split, ceiling, missing-gold causes, 15-row hand check, 24 samples
- `experiments/stages/lab/cesr_u.py` — the selector and plan objects reused unchanged

Reproduce:

```sh
cd /Users/asmit/Personal/algo-serp-stages
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_cesr_u_auto.py
```

~4 s. No downloads; the corpus is the repository itself.
