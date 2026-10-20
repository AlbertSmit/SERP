# CESR-U: universal and identity constraints, and where the ceiling actually was

CESR-code (`CESR_CODE.md`) reached 98.4% of achievable constraint coverage and still
returned the complete evidence set on only **72.5%** of queries. Its diagnosis was
specific: the objective maximised *existence* — one caller satisfies `CALLS(X)` — while
the qrels demanded the *universal closure* — every caller of the middle chunk, every
export of the file. 96.8% of missing gold was another chunk satisfying an
already-covered constraint.

This is the test of that diagnosis. The constraint language can now say:

- **universal** — `ALL_CALLERS(X)`, `ALL_DEFINES(X)`, `ALL_EXPORTS(file)`
- **identity** — `EXACT_DEFINES(X, chunk_id)`, and `CALL_CHAIN(X, middle)` where the
  middle chunk is pinned by `CALLS(X) ∩ DEFINES(middle)`
- **file predicates** — `IMPORTS(file, module)` evaluated on a chunk's file, instead of
  packing a whole importer chunk to satisfy coverage
- **relations** — `SAME_FILE(a, b)`, `CALLS_BOTH(X, Y)`

and the benchmark is viewed correctly: dedupe queries by normalised question text.

The answer, in one line: **universal/identity constraints close the gap completely on
every query whose constraint set determines the answer.** On the 161 text-unique queries
where the graph pins one evidence set, full coverage implies complete evidence **100%** of
the time (CESR-code: 96.1%). The residual 7.6% is not selector error and is not coverage
error: 100% of the remaining misses are a wrong choice among equally valid alternatives,
i.e. the query does not carry enough information to name its gold.

## Read this before the numbers

This is a **capability test with oracle constraints**, three times over.

1. **The qrels are graph-derived.** Same 250 synthetic queries and qrels as
   `CODE_GRAPH.md` (`generate_queries(graph, per_type=50, seed=1729)`). A positive result
   means retrieval can execute the relation, not that a human would call those chunks
   relevant.
2. **The constraints are oracle.** Built from `type`, `symbols`, `module` and `chain` and
   nothing else. `qrels` is never read by the builder, and neither is `qid` — the generated
   qid embeds the answer chunk id (`two_hop:tokenize:src/...:gramTokens`), which would be a
   leak. Nothing here parses a natural-language question. Label: **oracle-constraint
   CESR-U**.
3. **The benchmark view is corrected, not the benchmark.** Queries are deduped by
   normalised text. 185 of 250 survive; 65 share their text with a query that has different
   gold and are reported separately. The primary evaluation is the text-unique 185. No
   constant and no rule in this document was chosen by looking at the 250.

Constants are inherited verbatim from CESR-code so the comparison is matched: budgets
`500/1000/2000`, pool depth 100 per channel, `mu=0.2`, cost exponent `0.35`, path bonus
`0.25`, bootstrap 1,000 resamples, seed 1729. One constant is new and was fixed before the
run: lexical tie-break weight `0.02`.

## Method

### Constraint language

| operator | kind | members | meaning |
|---|---|---|---|
| `EXISTS_CALLS(X)` | witness | `callers(X)` | choose one chunk that calls X |
| `ALL_CALLERS(X)` | universal | `callers(X)` | **every** chunk that calls X |
| `EXISTS_DEFINES(X)` | witness | `defined(X)` | choose one definition |
| `EXACT_DEFINES(X, chunk)` | identity | `{chunk}` | this definition chunk, by id |
| `ALL_DEFINES(X)` | universal | `defined(X)` | every chunk defining X |
| `ALL_EXPORTS(file)` | universal | export units of `file` | every exported chunk of the file |
| `IMPORTS(file, module)` | predicate | — | file-level import test, **not** a packable set |
| `SAME_FILE(a, b)` | relation | — | `file(a) == file(b)` |
| `CALL_CHAIN(X, middle)` | identity + universal | `callers(X) ∩ defined(middle)` | middle calls X, every caller of middle |
| `CALLS_BOTH(X, Y)` | witness | `callers(X) ∩ callers(Y)` | choose a chunk that calls both |

Every chunk id is a graph id. `ALL_EXPORTS` excludes class containers: the query
generator's export evidence excludes them, and so does CESR-code's export channel. Class
containers can still be answers for other query types — one `calls_two` answer in this
benchmark is a 1,088-line class (see §Budgets).

Per-type composition, from query metadata only:

| type | witness (one of) | mandatory | predicate / relation |
|---|---|---|---|
| `same_file` | `CALLS_FILE_OF(symbols[0], symbols[1])`: callers whose file defines `symbols[1]` | `ALL_DEFINES_IN_FILE(symbols[1])` **follows the chosen caller's file** | `SAME_FILE(caller, anchor)` |
| `import_call` | `CALLS_IMPORTING(symbols[0], module)`: callers in a file importing `module` | `EXACT_DEFINES(symbols[0])` | `IMPORTS(file(caller), module)` |
| `two_hop` | `CALL_CHAIN(chain[0], chain[1])`: the middle chunk | `ALL_CALLERS(chain[1])` | chain: middle calls `chain[0]`, uppers call middle |
| `export_import` | — (the file is pinned by the graph) | `ALL_EXPORTS(file)`, `EXACT_DEFINES_IN_FILE(symbols[0], file)` | `IMPORTS(file, module)` |
| `calls_two` | `CALLS_BOTH(symbols[0], symbols[1])` | `EXACT_DEFINES(symbols[0])`, `EXACT_DEFINES(symbols[1])` | caller relation |

`EXACT_DEFINES` is used when the graph pins exactly one definition and `ALL_DEFINES` when it
does not. That distinction is semantic: a singleton definition set is an exact identity and
the chunk is fetched by id. In this query set every definition requirement happens to be a
singleton (50/50 `import_call`, 50/50 `export_import`, 100/100 symbol slots in `calls_two`),
so the universal fallback never fires — stated because it would be a silent difference on
another repository.

### Selector

Five deterministic phases:

1. **Resolve witnesses.** Each alternative set has one choice. Among alternatives, prefer
   one that fits the residual budget (budget minus witness-independent universal sets),
   then the one that belongs to the most other constraints, then the highest BM25 score of
   the question, then the shorter chunk, then chunk id. Coverage first, lexical second.
2. **Build the mandatory closure.** Every universal set, every identity chunk, and the
   universal sets that follow a witness (`same_file` anchors).
3. **Insert the closure first, by identity.** A mandatory chunk is fetched from the graph
   even if no channel returned it. It is not a ranked candidate.
4. **Greedy coverage for anything still uncovered.** CESR-code's score
   `(gain + 0.25·joins)/max(dl+10,1)^0.35 − 0.2·redundancy + 0.02·lexical`, with the
   lexical term as a tie-break. On this benchmark this phase is a **no-op on 250/250
   queries**: the closure already satisfies every constraint, so coverage is 1.000 before
   the loop starts and the selection is a one-shot closure computation.
5. **Repair.** If a universal set is still incomplete, add its members and report. It never
   fired (`repaired_members = 0`), because phase 3 inserts the full set. Mandatory evidence
   is never dropped: if the closure exceeds the budget the system reports the violation and
   keeps the evidence.

The last two points matter for the verdict. CESR-U is not doing iterative packing; it is
computing the closure and, where the constraint set is a disjunction, returning one witness.
The greedy machinery is carried over and idles.

**Identity retrieval is nearly irrelevant for recall here.** 3 of 572 selected chunks
(0.5%) were outside all three channels. The closure is reachable lexically; the value of the
constraint is *which* chunk, not *finding* it.

### Query set and benchmark view

| | queries | distinct texts | conflicting-text queries |
|---|---:|---:|---:|
| `same_file` | 50 | 50 | 0 |
| `import_call` | 50 | 23 | **35** |
| `two_hop` | 50 | 35 | **25** |
| `export_import` | 50 | 50 | 0 |
| `calls_two` | 50 | 47 | **5** |
| **total** | **250** | **205** | **65** |

Text-unique by type: `same_file` 50, `import_call` 15, `two_hop` 25, `export_import` 50,
`calls_two` 45. The text-unique set is 185, not a small slice; the largest loss is
`import_call`, which loses 35 of its 50 queries to 8 repeated question texts.

### Baselines, metrics, CIs

Porter BM25, RRF(Porter, entity, graph), and CESR-code, all truncated at the matched token
budget with the repo's skip-and-continue convention and `ix.dl`. Metrics: `all@5`/`all@10`
(prefix completeness over the returned order), `recall` (whole selection), `coverage`
(weighted constraint satisfaction **under the CESR-U constraint set**, for every system),
tokens, selected count, redundancy (symbol-set Jaccard), `nDCG@10`. Paired bootstrap,
1,000 resamples, seed 1729, against both CESR-code and RRF.

## What the constraint set can determine

| type | n | mean options | median | max | unique answer | qrels expressible | chosen witness correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| `same_file` | 50 | 1.02 | 1 | 2 | 0.980 | 1.000 | 1.000 |
| `import_call` | 50 | **17.14** | 13 | 34 | **0.100** | 1.000 | 0.240 |
| `two_hop` | 50 | 1.10 | 1 | 2 | 0.900 | 1.000 | 0.940 |
| `export_import` | 50 | 1.00 | 1 | 1 | 1.000 | 1.000 | 1.000 |
| `calls_two` | 50 | 1.84 | 1 | 10 | 0.640 | 1.000 | 0.760 |
| **all** | **250** | **4.42** | **1** | **34** | **0.724** | **1.000** | **0.788** |

`expressible` = there exists a legal witness choice under which the closure contains the
qrels. It is **1.000 for every type**: the operator set can state every answer in the
benchmark. `unique answer` = every witness set is a singleton, computed from the graph
before any qrels comparison. On the text-unique 185 the split is **161 determined / 24
ambiguous**; on all 250 it is 181 / 69.

That table is the experiment's frame. The language is not the limit. For 27.6% of queries
the *query* is the limit: `import_call` has 17.14 equally valid caller chunks (median 13,
max 34) and `calls_two` has 1.84 (max 10). No selector can choose among them from the
metadata the question carries.

## Results: text-unique queries (primary)

`cover` = normalised constraint weight; `redun` = mean pairwise symbol Jaccard; `n` = chunks
selected.

| budget | system | all@5 | all@10 | recall | cover | redun | tokens | n | nDCG@10 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | Porter | 0.162 | 0.254 | 0.571 | 0.595 | 0.088 | 485 | 5.5 | 0.469 |
| 500 | RRF | 0.427 | 0.481 | 0.728 | 0.749 | 0.101 | 482 | 5.2 | 0.629 |
| 500 | CESR-code | 0.741 | 0.741 | 0.880 | 0.914 | 0.069 | 212 | 2.1 | 0.873 |
| 500 | **CESR-U** | **0.919** | **0.919** | **0.967** | **1.000** | 0.073 | **280** | 2.3 | **0.975** |
| 1000 | Porter | 0.178 | 0.314 | 0.659 | 0.672 | 0.087 | 976 | 8.7 | 0.498 |
| 1000 | RRF | 0.486 | 0.692 | 0.846 | 0.857 | 0.095 | 972 | 8.6 | 0.687 |
| 1000 | CESR-code | 0.849 | 0.849 | 0.932 | 0.967 | 0.075 | 254 | 2.2 | 0.911 |
| 1000 | **CESR-U** | **0.919** | **0.919** | **0.967** | **1.000** | 0.073 | **281** | 2.3 | **0.975** |
| 2000 | Porter | 0.189 | 0.319 | 0.790 | 0.798 | 0.084 | 1,995 | 15.5 | 0.501 |
| 2000 | RRF | 0.470 | 0.768 | 0.914 | 0.916 | 0.080 | 1,997 | 14.7 | 0.705 |
| 2000 | CESR-code | 0.854 | 0.854 | 0.934 | 0.969 | 0.075 | 258 | 2.3 | 0.913 |
| 2000 | **CESR-U** | **0.924** | **0.924** | **0.969** | **1.000** | 0.073 | **284** | 2.3 | **0.977** |

Paired bootstrap, 1,000 resamples, seed 1729, text-unique:

| budget | metric | vs | Δ | 95% CI | W/T/L |
|---:|---|---|---:|---|---:|
| 500 | all@10 | CESR-code | +0.1784 | [+0.1135, +0.2378] | 37/144/4 |
| 500 | all@10 | RRF | +0.4378 | [+0.3568, +0.5135] | 85/96/4 |
| 1000 | all@10 | CESR-code | +0.0703 | [+0.0216, +0.1243] | 17/164/4 |
| 1000 | all@10 | RRF | +0.2270 | [+0.1568, +0.3027] | 49/129/7 |
| 2000 | all@10 | CESR-code | **+0.0703** | **[+0.0216, +0.1189]** | 17/164/4 |
| 2000 | all@10 | RRF | **+0.1568** | **[+0.0865, +0.2216]** | 37/140/8 |
| 2000 | recall | CESR-code | +0.0351 | [+0.0144, +0.0568] | 20/161/4 |
| 2000 | recall | RRF | +0.0559 | [+0.0270, +0.0865] | 29/148/8 |
| 2000 | coverage | CESR-code | +0.0306 | [+0.0180, +0.0464] | 17/168/0 |
| 2000 | coverage | RRF | +0.0838 | [+0.0577, +0.1108] | 35/150/0 |
| 2000 | nDCG@10 | CESR-code | +0.0641 | [+0.0437, +0.0878] | 39/142/4 |
| 2000 | nDCG@10 | RRF | +0.2719 | [+0.2382, +0.3024] | 162/18/5 |

Token ratio CESR-U/RRF: **0.58 / 0.29 / 0.14**. CESR-U returns 2.3 chunks against
RRF's 5.2 / 8.6 / 14.7, so `all@5` = `all@10` by construction (selection sizes: 2 chunks
×182, 3 ×64, 4 ×4). It is a set comparison, not a ranking comparison, and the nDCG column
is the precision of a 2-chunk set.

**The 17 wins / 4 losses is the whole result.** Splitting by constraint determinacy:

- **determined (161 queries): 13 wins, 0 losses.** CESR-code 0.919 → CESR-U 1.000. Of the
  13, **6 were queries CESR-code covered completely and still answered with a substitute**
  (5 `two_hop`, 1 `export_import` — the existence/universality gap), and **7 were queries
  where its greedy loop never reached full coverage at all** (all `calls_two`, where it
  stopped after the caller and one definition).
- **ambiguous (24 queries): 4 wins, 4 losses.** Both 0.417. The lexical witness is a coin
  flip, exactly as expected when the options are equally valid.

None of the gain is lexical luck: every determined win is a structural fix, and the 4
ambiguous wins are cancelled by 4 ambiguous losses.

## Results: all 250 and the conflicting 65

| budget | system | all@10 all 250 | all@10 conflicting 65 |
|---:|---|---:|---:|
| 500 | RRF | 0.372 | 0.062 |
| 500 | CESR-code | 0.592 | 0.169 |
| 500 | CESR-U | **0.780** | **0.385** |
| 1000 | RRF | 0.536 | 0.092 |
| 1000 | CESR-code | 0.672 | 0.169 |
| 1000 | CESR-U | **0.784** | **0.400** |
| 2000 | RRF | 0.600 | 0.123 |
| 2000 | CESR-code | 0.676 | 0.169 |
| 2000 | CESR-U | **0.788** | **0.400** |

The conflicting set is not uniformly unanswerable. 20 of its 65 queries are `two_hop` pairs
whose *text* is shared but whose `chain[1]` metadata differs, so each is still determined by
its own constraint set: CESR-U gets 20/20, CESR-code 10/20. The other 45 (`import_call` 35,
`calls_two` 5, `same_file` 0, `two_hop` 5) are genuine conflicts and CESR-U gets 6/45
(0.133). That is why CESR-U's conflicting-set number is 0.400 and not 0.169 — a third of
that slice is recoverable once the constraint set is read instead of the text.

## The key diagnostic: full coverage against completeness

Each system is conditioned on its **own** coverage mask, so the 72.5% row is comparable.

| subset | system | full coverage | all@10 given full coverage | all@10 overall |
|---|---|---:|---:|---:|
| all 250 | CESR-code | 0.932 | **0.725** | 0.676 |
| all 250 | CESR-U | 1.000 | **0.788** | 0.788 |
| text-unique | CESR-code | 0.941 | 0.908 | 0.854 |
| text-unique | CESR-U | 1.000 | **0.924** | 0.924 |
| text-unique, determined | CESR-code | 0.957 | 0.961 | 0.919 |
| text-unique, determined | CESR-U | 1.000 | **1.000** | **1.000** |
| text-unique, ambiguous | CESR-code | 0.833 | 0.500 | 0.417 |
| text-unique, ambiguous | CESR-U | 1.000 | 0.417 | 0.417 |

Three readings, in order of importance:

1. **On constraint-determined queries, coverage now implies completeness, exactly.** 161/161.
   The 3.9-point gap CESR-code had *conditional on its own full coverage* (0.961) was the
   existence/universality gap and is closed completely, not approximately. There was a
   second gap on the same 161 queries: 7 where its greedy loop never reached full coverage
   at all. Both are objective failures; both are gone.
2. **The 72.5% mixed two different failures.** On the 161 determined text-unique queries
   CESR-code loses 8.1 points (0.919 → 1.000); that is the objective failure, and CESR-U
   removes it. On the 24 ambiguous ones it loses 58.3 points (0.417) and no objective can
   remove it. Weighted by query count, that is 7.0 points and 7.6 points of CESR-code's
   14.6-point shortfall on the text-unique set: CESR-U removes the first and leaves exactly
   the second. On all 250, the same mixture explains why CESR-U's number is 0.788 and not
   higher: the 69 ambiguous queries are 27.6 points of the 250-set before any guessing, and
   lexical guessing recovers 6.4 of them.
3. **Where the constraint set is ambiguous, the objective is not the problem and neither is
   the witness rule.** CESR-U and CESR-code are identical on the ambiguous subset (0.417
   each), with four wins and four losses each.

### Missing gold, classified (text-unique)

| budget | missing chunks | witness alternative | universal incomplete | unconstrained | not in pool |
|---:|---:|---:|---:|---:|---:|
| 500 | 15 | **1.000** | 0.000 | 0.000 | 0.000 |
| 1000 | 15 | **1.000** | 0.000 | 0.000 | 0.000 |
| 2000 | 14 | **1.000** | 0.000 | 0.000 | 0.000 |

Compare CESR-code at the same conditioning: 96.8% alternative satisfier, 3.2% not in pool.
Now 100% of misses are the wrong alternative among a disjunction. Not one gold chunk is
missing because the pool lacked it, because a universal set was incomplete, or because the
operator set could not express it. The constraint language and the pool are both exonerated;
what remains is information that is not in the query.

## Per type (text-unique)

| type | n | Porter | RRF | code@500 | code@2000 | U@500 | U@1000 | U@2000 | unique answer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `same_file` | 50 | 0.580 | 0.980 | 0.920 | 0.980 | **1.000** | **1.000** | **1.000** | 0.980 |
| `import_call` | 15 | 0.600 | 0.800 | 0.467 | 0.600 | 0.533 | 0.533 | **0.600** | 0.333 |
| `two_hop` | 25 | 0.040 | 0.600 | 0.680 | 0.800 | **1.000** | **1.000** | **1.000** | 1.000 |
| `export_import` | 50 | 0.000 | 0.820 | 0.940 | 0.980 | **1.000** | **1.000** | **1.000** | 1.000 |
| `calls_two` | 45 | 0.444 | 0.556 | 0.444 | 0.689 | **0.822** | **0.822** | **0.822** | 0.711 |

Determinacy split, text-unique, budget 2000:

| type | n | determined | ambiguous | U all@10 | U \| determined | U \| ambiguous | code \| determined |
|---|---:|---:|---:|---:|---:|---:|---:|
| `same_file` | 50 | 49 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| `import_call` | 15 | 5 | 10 | 0.600 | **1.000** | 0.400 | 1.000 |
| `two_hop` | 25 | 25 | 0 | 1.000 | **1.000** | — | 0.800 |
| `export_import` | 50 | 50 | 0 | 1.000 | **1.000** | — | 0.980 |
| `calls_two` | 45 | 32 | 13 | 0.822 | **1.000** | 0.385 | 0.781 |

- **`two_hop` is the clean confirmation.** `ALL_CALLERS(chain[1])` is literally the upper
  half of the qrels and `CALL_CHAIN` is literally the middle. 0.800 → 1.000, with no
  ambiguity to hide behind (25/25 determined).
- **`export_import` closes.** The file is pinned by `(imports module, defines symbol)`;
  `ALL_EXPORTS` then selects the answer set. 0.980 → 1.000.
- **`same_file` closes its one ambiguous query** (0.980 → 1.000); the 49 determined
  `same_file` queries were already complete in both systems. The anchor set now follows the
  caller's file instead of being merely "some definition of Y", but that changes the set,
  not this score.
- **`calls_two` gains 0.689 → 0.822, entirely from determined queries** (0.781 → 1.000).
  The 13 ambiguous ones stay at 0.385 (CESR-code: 0.462).
- **`import_call` is a flat loss and the honest row.** 0.600 → 0.600, and on the 10
  ambiguous queries both systems get 0.400. The universal part (`EXACT_DEFINES`) now always
  selects the definition; the caller is one of 17.14. Identity from `(symbol, module)` does
  not exist. On the 5 determined queries both systems are already at 1.000, so there is
  nothing for a better objective to win.

Decomposition of the +0.0703 (net +13 queries): `calls_two` +6 (7 determined fixes, 1
ambiguous loss), `two_hop` +5 (all determined), `export_import` +1 (determined),
`same_file` +1 (ambiguous), `import_call` 0 (1 ambiguous win, 1 ambiguous loss). Every
structural fix is on a determined query; the ambiguous wins and losses are the same coin
flipped differently.

## Budgets: the closure is the spend, and the budget stops binding

| budget | over-budget rate | overspend among violators | mean spend | max spend |
|---:|---:|---:|---:|---:|
| 500 | **0.162** | 177 | 280 | 1,307 |
| 1000 | 0.011 | 160 | 281 | 1,307 |
| 2000 | 0.000 | — | 284 | 1,307 |

34/250 queries exceed 500 tokens, 2/250 exceed 1,000, none exceed 2,000. Mandatory evidence
is never dropped: the system reports the violation and returns the closure. The 1,307-token
case is one `calls_two` query whose answer chunk is the 1,088-line `SQLiteContextEngine`
class container, plus two definitions; the gold set itself does not fit at 500.

Consequence for reading the 500-token row: CESR-U's 0.919 is bought with a mean 280 tokens
and 177 tokens of overspend on 16% of queries. The 1000 and 2000 rows are clean, which is
why they are the gated ones.

**Unbudgeted "all mandatory" view** (witnesses + universal closure, no budget):

| subset | n | all@10 | recall | tokens | chunks |
|---|---:|---:|---:|---:|---:|
| all 250 | 250 | 0.788 | 0.904 | 314 | 2.29 |
| text-unique | 185 | 0.924 | 0.969 | 284 | 2.28 |
| conflicting | 65 | 0.400 | 0.718 | 400 | 2.32 |

The unbudgeted view is identical to the 2,000-token view. The budget is not the binding
constraint at 2,000; at 500 it truncates nothing but it is exceeded by the closure. **The
binding constraint is the constraint set's information content.**

## Gate

Pre-registered rule:

> CESR-U beats CESR-code **and** RRF on `all@10` for text-unique queries at every budget
> where gold evidence fits in ≥90% of queries (1000 and 2000; it fits in 82% at 500), and
> full coverage implies complete evidence at ≥0.90 on the text-unique set.

| budget | Δ all@10 vs CESR-code | 95% CI | Δ all@10 vs RRF | 95% CI |
|---:|---:|---|---:|---|
| 1000 | +0.0703 | [+0.0216, +0.1243] | +0.2270 | [+0.1568, +0.3027] |
| 2000 | +0.0703 | [+0.0216, +0.1189] | +0.1568 | [+0.0865, +0.2216] |

Full-coverage completeness, text-unique: **0.924** (threshold 0.90). Both CI lower bounds
are above zero for both baselines; the 500-token budget also beats both (+0.1784 vs
CESR-code, +0.4378 vs RRF) even though the rule did not require it.

**Gate PASSED.** The mechanical rule is met on every count, and the component that matters
most — completeness given full coverage on determined queries — is 1.000, not 0.92.

## Verdict

**CESR-U is a retrieval algorithm on queries whose constraint set determines the answer, and
an ambiguity-limited guess everywhere else. It is no longer an existence-packer.**

1. **The universal constraint was the missing half of the objective.** `ALL_CALLERS`,
   `ALL_EXPORTS` and the identity-pinned chain convert "a caller" into "the caller set".
   On the 161 determined text-unique queries, completeness goes 0.919 → **1.000** with zero
   regressions: 6 queries that were covered-then-substituted, and 7 where the existential
   greedy stopped before covering everything. 100% of remaining misses are wrong
   alternatives rather than coverage or pool failures.
2. **Identity works when the graph pins it and fails when the query does not.** `CALL_CHAIN`
   (middle = calls X ∩ defines middle) and `EXACT_DEFINES` are exact and cost nothing: the
   `two_hop` row is 25/25. But `import_call`'s caller is not expressible from
   `(symbol, module)` at all — 17.14 options, median 13. The BM25 witness tie-break is
   worth exactly zero there (4 wins, 4 losses on the ambiguous subset). Identity must come
   from the constraint, not from a lexical guess.
3. **The benchmark ceiling is now measurable and it is the query, not the selector.**
   181/250 queries have a unique answer under a language that can express 250/250. The
   other 69 are a single arbitrary draw from a satisfying set. That is a query-generator
   defect of the same kind as the text collision, and it caps any system that reads only
   these constraints at 0.724 on the 250-set plus whatever it can guess. CESR-U reaches
   0.788 because lexical guessing beats chance on some of them.
4. **Against truncation it still wins, with 14–58% of the tokens.** +0.157 `all@10` over RRF
   at 2,000 tokens, +0.070 over the corrected CESR-code baseline, with recall, coverage and
   nDCG moving the same way. It returns 2.3 chunks where RRF returns 14.7.
5. **What would change the verdict.** A constraint extractor that produces these operators
   from text. The oracle test now says the bottleneck is no longer the objective, the pool,
   or the optimiser: it is (a) whether an extractor can name `chain[1]`, the file, and the
   exact definition, and (b) whether the query carries enough information to identify a
   witness. (b) is not a retrieval problem. On this benchmark, 27.6% of queries fail (b).

**Is CESR-U a retrieval algorithm or still only a packer?** It is an *executor*: given
constraints that determine the evidence, it returns the evidence, complete, at a third of
fusion's token cost. That is a retrieval algorithm. Given constraints that determine only a
set of alternatives, it is a packer that returns one alternative and is right about as often
as chance. The distinction is now a property of the constraint set, measurable before
retrieval, and that is the useful result: **the 72.5% was not a packing failure.** Measured
on the same queries, 8.1 of its points are objective failure on determined queries and the
rest is unanswerable ambiguity; universal and identity constraints remove the first
entirely, and nothing in the constraint language can remove the second.

## Honest limits

- **Oracle constraints.** Built from `type`, `symbols`, `module`, `chain`; nothing parses a
  question. An automatic extractor will be worse, and this experiment cannot say by how
  much. It is the upper bound of a deployable CESR-U.
- **Graph-derived qrels.** The relations that define the constraints generated the labels,
  and the graph channel traverses them. The circularity is strongest on `export_import`
  (near-tautological: the file is pinned by the same two facts the qrels use) and weakest on
  `two_hop`, whose answer chunk contains neither query symbol.
- **The ambiguity criterion is graph-only, but the decision to gate on it is a judgement.**
  `unique answer` is computed from the constraint set before qrels is read, and the text
  dedupe is mechanical. Still: conditioning the headline on "determined queries" is a choice
  a sceptic can call goalpost movement. The unconditioned number is on the same table
  (0.924 text-unique, 0.788 all 250), and the gate as written uses the unconditioned 0.924.
- **`import_call` is a benchmark defect, not a retrieval result.** Its qrels name one caller
  from a mean 17.14-element satisfying set, chosen by the generator's loop order. Its 0.600
  (and CESR-code's 0.600) should be read as "unanswerable by construction" for 10 of 15
  text-unique queries, not as a system quality difference.
- **`all@10` rewards supersets.** CESR-U adds mandatory chunks that the qrels may not
  require (all definitions of a symbol, all exports of a file). It does not dump
  alternatives — 2.3 chunks per query, and selection size never exceeds 4 — but a system
  that returned the union of all alternatives would also score well on this metric. The
  tokens, selected count and nDCG columns are there to make that visible.
- **Definition identity uses the graph's single definition.** When a symbol has several
  definitions, `ALL_DEFINES` selects all of them and the qrels naming one is a generator
  convention, not a semantic identity. On this query set every definition requirement is a
  singleton, so the issue never binds — it would on another repository.
- **Small corpus, one repository.** 14 files, 289 chunks, no human labels. A 10-chunk set is
  3.5% of the corpus.
- **Budget violations at 500.** 16% of queries exceed the budget by a mean 177 tokens. The
  comparison at 500 is therefore not strictly matched-budget; the 1000/2000 rows are.
- **Fixed constants, no search.** All inherited constants were fixed in `CESR_CODE.md`; the
  one new constant (`LEX_TIEBREAK = 0.02`) was fixed before the run and its effect is
  measurable as zero on ambiguous queries.
- **Determinism.** Every metric and every table is byte-identical across runs; only the
  latency fields differ.
- **Latency** (CPU, 250 queries): retrieval 2.65 ms/query, constraint construction 0.02 ms,
  CESR-U selection 0.05 ms (a closure computation), CESR-code selection 2.02 ms, full
  evaluation 2.4 s. Selection is ~2% of end-to-end cost.

## Files

- `experiments/stages/lab/cesr_u.py` — constraint language, plans, witness resolution,
  closure selector, repair, coverage/ambiguity diagnostics, gate
- `experiments/stages/run_cesr_u.py` — runner, text-unique benchmark view, baselines,
  bootstrap, slices
- `experiments/stages/results/cesr_u.json` — every table, CI, per-type and determinacy
  slice, budget violations, missing-gold causes, 40 samples

Reproduce:

```sh
cd /Users/asmit/Personal/algo-serp-stages
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_cesr_u.py
```

~3 s. No downloads; the corpus is the repository itself.
