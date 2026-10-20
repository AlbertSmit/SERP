# Code-graph EBR: entity bridges where the entities are exact

The HotpotQA track asked whether a no-embedding retriever can follow a two-fact
chain when the bridge is a noisy capitalised span. SERP's real domain is code and
agent memory, where the bridge is not a guess: `CALLS`, `IMPORTS`, `EXPORTS` and
`DEFINED_IN` are exact, typed relations over the system's own files.

This is the counterpart experiment. It asks one narrow question: **if retrieval
is given the exact graph, can it follow evidence chains that BM25 structurally
cannot, without losing what BM25 already does well?**

## Read this before the numbers

This is a **synthetic capability test, not a relevance benchmark.** Every query is
generated from the graph, and every qrel is a set of graph-derived evidence
chunks. The graph channel traverses the same relations the qrels were built from.
So the test measures whether retrieval *can execute the relation*, not whether the
graph is semantically right, and not whether a human would call those chunks
relevant. A positive result here is a necessary condition for code-graph EBR, not
evidence that it improves real code search.

## Method

No embeddings, no LLM calls, no external parser.

1. **Chunks.** Regex + brace tracking over `src/`, `scripts/`, `test/` `*.js`.
   A chunk is a function, a class container (header and fields only, methods are
   split out), a method, or a top-level block split at blank lines. Import
   preambles and module constants become their own chunks, so an exported constant
   does not hide inside the import block.
2. **Symbols.** Function/class/method names, imported names, and call-site names.
   Identifiers are tokenised both whole and as camel/snake subwords (the Python
   equivalent of `dualTokenizeCode` in `scripts/index-codebase.js`).
3. **Relations.** `DEFINED_IN(chunk→file)`, `CALLS(chunk→symbol)`,
   `IMPORTS(file→module)`, `EXPORTS(file→symbol)`, `CONTAINS(file→chunk)`.
   Relative module strings are resolved to repo files (`./tokenize.js` from
   `src/engine/engine.js` → `src/engine/tokenize.js`).
4. **Index.** BM25 over `title + body` in the existing harness, with Porter
   stemming (`snowballstemmer`) on the code token stream.
5. **Channels.**
   - `porter`: Porter BM25.
   - `entity`: chunks mentioning a *defined symbol that appears verbatim in the
     query*, weighted by symbol IDF, hub symbols dropped above 5% df. Exact
     identifier matching, no capitalisation heuristic.
   - `graph`: BM25 top-10 seeds, then two hops over `CALLS` (both directions,
     anchored on the definition), resolved `IMPORTS` (both directions), and
     `DEFINED_IN → EXPORTS` (the containing file's exported entry points). Hop 2
     is weighted 0.5, import edges 0.5, export edges 0.75. No seeds are rescored:
     this is an expansion channel.
   - Fusion is fixed RRF, `k=60`, unweighted.
6. All constants were fixed before the final run. One design change was made
   *after* a first look at the per-type table: `DEFINED_IN → EXPORTS` was added as
   an edge type, because without it an export question has no traversable path in
   the graph at all. That is disclosed here rather than presented as pre-registered.

## Graph statistics

| quantity | value |
|---|---:|
| files | 14 |
| chunks | 289 (`function` 75, `method` 60, `block` 153, `class` 1) |
| symbols | 132 |
| `CALLS` edges | 397 (236 cross-file) |
| `IMPORTS` edges | 60 module strings, 19 resolve to a repo file |
| `EXPORTS` edges | 11 |
| `CONTAINS` edges | 289 |
| `DEFINED_IN` edges | 289 |
| avg `CALLS` out-degree | 1.37 |
| avg out-degree incl. inherited file imports | 5.54 |
| chunks with at least one call | 163 / 289 |
| symbols defined in more than one chunk | 5 |

The corpus is small: 14 files, 289 chunks, one file (`src/engine/engine.js`)
contributing 80. A top-10 list is a 3.5% slice of the corpus, which makes
`both@10` easier than it would be on a real repository.

## Query set

250 queries, 50 per type, fixed seed 1729, generated from the graph and then
shuffled deterministically. qrels are the evidence chain, not just the answer:
182 queries have 2 evidence chunks, 64 have 3, 4 have 4.

| type | question shape | evidence |
|---|---|---|
| `same_file` | which chunk calls X and is defined in the same file as Y? | caller + definition of Y |
| `import_call` | which chunk calls W in a file that imports Z? | caller + definition of W |
| `two_hop` | which chunk calls a function that calls X? | the intermediate caller + its caller |
| `export_import` | which exports are defined in the file that imports Z and defines W? | export chunks + definition of W |
| `calls_two` | which chunk calls both X and Y? | caller + both definitions |

Examples:

```
same_file     Which chunk calls tokenStem and is defined in the same file as collectCoveredWords?
import_call   Which chunk calls storeMemory in a file that imports ../src/engine/engine.js?
two_hop       Which chunk calls a function that calls cachedTermMatch?
export_import Which exports are defined in the file that imports ./tokenize.js and defines chunkHeaderTokenCounts?
calls_two     Which chunk calls both exactIdentifierSuite and printExact?
```

`both@k` keeps the HotpotQA meaning: **all** qrel chunks in the top k. Queries
whose answer set could not be pinned uniquely were discarded at generation time.

## Results (250 queries)

| system | both@5 | both@10 | nDCG@10 | R@10 | MRR | answer@10 |
|---|---:|---:|---:|---:|---:|---:|
| Porter BM25 | 0.152 | 0.288 | 0.419 | 0.561 | 0.469 | 0.408 |
| entity channel alone | 0.272 | 0.344 | 0.506 | 0.573 | 0.609 | 0.428 |
| graph channel alone | 0.176 | 0.336 | 0.477 | 0.594 | 0.561 | 0.548 |
| RRF(Porter, graph) | 0.336 | 0.556 | 0.555 | 0.719 | 0.595 | 0.716 |
| RRF(Porter, entity) | 0.272 | 0.360 | 0.528 | 0.616 | 0.633 | 0.480 |
| **RRF(Porter, entity, graph)** | **0.364** | **0.608** | **0.581** | **0.745** | **0.619** | **0.736** |

Paired bootstrap vs Porter, 1,000 resamples:

| system | metric | Δ | 95% CI | W/T/L |
|---|---|---:|---|---:|
| graph | both@10 | +0.048 | [−0.016, +0.108] | 38 / 186 / 26 |
| graph | nDCG@10 | +0.058 | [+0.029, +0.086] | 108 / 55 / 87 |
| RRF(Porter, graph) | both@10 | +0.268 | [+0.208, +0.324] | 72 / 173 / 5 |
| RRF(Porter, graph) | nDCG@10 | +0.136 | [+0.113, +0.158] | 159 / 58 / 33 |
| RRF(Porter, entity) | both@10 | +0.072 | [+0.036, +0.108] | 20 / 228 / 2 |
| **RRF(Porter, entity, graph)** | both@10 | **+0.320** | [+0.260, +0.376] | 82 / 166 / 2 |
| **RRF(Porter, entity, graph)** | both@5 | **+0.212** | [+0.156, +0.264] | 55 / 193 / 2 |
| **RRF(Porter, entity, graph)** | nDCG@10 | **+0.162** | [+0.139, +0.186] | 170 / 59 / 21 |
| **RRF(Porter, entity, graph)** | R@10 | **+0.184** | [+0.148, +0.220] | 96 / 148 / 6 |

## Where the gain comes from

`both@10` by query type:

| type | Porter | entity | graph | RRF(P,e) | RRF(P,e,g) |
|---|---:|---:|---:|---:|---:|
| same_file | 0.580 | 0.800 | 0.260 | 0.800 | **0.980** |
| import_call | 0.240 | 0.360 | 0.280 | 0.360 | 0.360 |
| two_hop | 0.120 | 0.040 | **0.420** | 0.120 | 0.320 |
| export_import | 0.000 | 0.000 | 0.300 | 0.000 | **0.840** |
| calls_two | 0.500 | 0.520 | 0.420 | 0.520 | 0.540 |

Per-type deltas of the combined system vs Porter, both@10:

| type | Δ | 95% CI |
|---|---:|---|
| same_file | +0.400 | [+0.260, +0.540] |
| import_call | +0.120 | [+0.040, +0.220] |
| two_hop | +0.200 | [+0.080, +0.320] |
| export_import | +0.840 | [+0.740, +0.940] |
| calls_two | +0.040 | [−0.040, +0.140] |

The ablations are more informative than the totals:

- **The graph channel alone is not a ranker.** It is competitive overall
  (0.336 vs 0.288) only because it is fed BM25 seeds. On lexical types it loses
  badly (`same_file` 0.260 vs 0.580, `calls_two` 0.420 vs 0.500) because it never
  scores the seed itself and cannot match an identifier it was not pointed at.
- **It is a bridge specialist.** It nearly quadruples `two_hop` (0.420 vs 0.120)
  and is the only channel that can produce `export_import` answers at all.
- **Fusion is what makes it usable.** RRF(Porter, graph) beats Porter on every
  query type and RRF(Porter, entity, graph) beats RRF(Porter, entity) by +0.248
  both@10, 62 wins / 0 losses.
- **The marginal value of the graph over a pure symbol channel is +0.100 both@10
  [+0.060, +0.145] once `export_import` is excluded** — that type is where the
  circularity is strongest, so this is the more conservative reading.

## Is the query set circular?

Partly, and it matters.

`export_import` is near-tautological: the qrels are the exported chunks of a file,
and the graph channel reaches them via `DEFINED_IN → EXPORTS`. A channel that can
traverse the relation that generated the answer will find the answer. The honest
content of that row is narrow but real: **without the export edge the system scores
0.000, and with it 0.840** — a graph index makes an entire class of question
answerable, and an incomplete graph makes none of it answerable.

`same_file` and `calls_two` are weakly circular in the other direction: both query
symbols are named, and the answer chunk contains one of them, so a 289-chunk
corpus lets an exact-identifier channel (and often BM25) find the evidence
lexically. The `+0.040` on `calls_two` is basically the null result for those.

`two_hop` is the clean test. The answer chunk contains **neither** query symbol;
only a caller-of-a-caller can produce it. There, graph 0.420 vs Porter 0.120
(+0.300, CI [+0.180, +0.440]), and graph's marginal over the entity channel is
+0.200 [+0.080, +0.320]. That is the one row that cannot be explained by lexical
matching.

Leave-one-type-out, `both@10`:

| excluded | Porter | RRF(P,e) | RRF(P,e,g) |
|---|---:|---:|---:|
| same_file | 0.215 | 0.250 | 0.515 |
| import_call | 0.300 | 0.360 | 0.670 |
| two_hop | 0.330 | 0.420 | 0.680 |
| export_import | 0.360 | 0.450 | 0.550 |
| calls_two | 0.235 | 0.320 | 0.625 |

No single type carries the result, but removing `export_import` removes most of it.

## Parser error is the real threat

A regex graph is lossy: dynamic dispatch, callbacks passed by reference, aliased
imports and properties on objects are invisible. To bound that, the qrels were
kept clean while the *retrieval* graph lost 20/40/60% of its `CALLS`, `EXPORTS`
and `IMPORTS` edges (deterministic, seed 1729):

| edge drop | Porter both@10 | graph both@10 | RRF(P,e,g) both@10 |
|---:|---:|---:|---:|
| 0% | 0.288 | 0.336 | 0.608 |
| 20% | 0.288 | 0.232 | 0.388 |
| 40% | 0.288 | 0.232 | 0.376 |
| 60% | 0.288 | 0.176 | 0.384 |

Two thirds of the advantage disappears at a 20% edge-loss rate, then the curve
flattens: the cheap lexical types keep working, the export and 2-hop chains do
not. `two_hop both@10` falls from 0.320 to 0.280 at 20% and to 0.180 at 60%, still
above Porter's 0.120.

The practical reading: **the benefit of a code graph is a bet on edge
completeness.** An exact graph is worth a lot; a 60%-recall parser is worth
roughly +0.10 both@10 over BM25, which is not nothing but is far from the headline.

## Verdict

- **Yes, in the capability sense.** On exact structural evidence chains, graph
  expansion beats Porter BM25 on every metric, and the fusion never loses to BM25
  on any query type. The mechanism is the same as on HotpotQA: lexical first hop,
  structural second hop, fixed RRF.
- **No, not as a general retrieval win.** The graph channel on its own is worse
  than BM25 wherever the query is lexical (`same_file` −0.32, `calls_two` −0.08
  both@10), and it costs shallow precision on `same_file` in fusion (both@5 0.48
  vs 0.80 for the entity channel). Like EBR, it should be gated to relational
  queries, not applied to every query.
- **The headline +0.32 both@10 is an upper bound under a perfect parser.** On the
  one non-circular type the gain is +0.20 both@10; with 20% edge noise the overall
  gain falls to +0.10.
- **A null result was possible and did not occur**: the entity channel alone is
  not sufficient (+0.072 both@10), so the graph edges add information beyond
  symbol co-occurrence, concentrated in `two_hop` and `export_import`.
- **Scope.** 14 files, 289 chunks, one repository, no human labels. The next
  honest test is a real repo with parser-measured edge recall and human or
  issue-derived qrels — not more synthetic types.

## Reproduce

```sh
cd /Users/asmit/Personal/algo-serp-stages
.venv/bin/python experiments/stages/run_code_graph.py
```

Code: `experiments/stages/lab/code_graph.py`, runner
`experiments/stages/run_code_graph.py`. Results:
`experiments/stages/results/code_graph.json`. Runtime ~2.5 s. Every number is
deterministic (verified by byte-identical JSON across runs).
