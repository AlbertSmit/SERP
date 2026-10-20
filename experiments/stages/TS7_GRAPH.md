# A TS7-native code graph, and what it costs against regex

Question: can the TypeScript 7 native compiler be the parser behind a code graph,
and is the resulting graph better than a regex extractor in a way that matters?

Short answer: **yes for the graph, not yet for the API surface.** Use TS7's AST at
build time: it removes the regex extractor's 11% false call edges, gets every chunk
boundary right, and lifts the Porter BM25 baseline by +0.104 both@10 on identical
queries. It does **not** improve the graph channel (+0.012) and does not move the
fused system at all (0.606 vs 0.608), so the case is correctness, not retrieval.
Do not build a per-keystroke service on `unstable/*` yet.

Everything below is reproducible from `experiments/stages/ts7-graph/`.

## 1. API verdict

`typescript@7.0.2` (stable, npm) ships the same `dist/api/*` + `dist/ast/*` tree as
`@typescript/native-preview`, with the same `./unstable/sync` and `./unstable/ast`
exports. The probe never needed the dev preview package: the stable tarball has the
API, behind unstable paths.

| | |
|---|---|
| entry points | `typescript/unstable/sync` (`API`, `Snapshot`, `Project`, `Program`, `Checker`, `Symbol`, `NodeHandle`), `typescript/unstable/ast` (`SyntaxKind`, `is*`, `node.forEachChild`) |
| usable | yes, for batch extraction |
| stable enough to depend on | **no** — `unstable/*` is explicitly unversioned |

What worked on the first try: `updateSnapshot({openProjects:[tsconfig]})` honours
`allowJs`, `project.rootFiles` gives exactly the 14 include files,
`program.getSourceFile()` returns a real `SourceFile` with text and line maps,
`node.forEachChild()` walks it, and `checker.getSymbolAtLocation()` resolves call
targets across files. `api.getTimingInfo().totals` reports request counts and
server vs transport time, which made profiling the build trivial.

Friction, in order of how much it cost:

1. **No batch checker.** 2,213 `getSymbolAtLocation` calls became 2,245 IPC
   requests and ~170–200 ms of round trips. This is 65% of the build. There is no
   way to ask for 500 symbols at once.
2. **Handles, not nodes.** `symbol.declarations` are `NodeHandle`s; each needs
   `.resolve(project)` (another round trip). Imported symbols need
   `flags & SymbolFlags.Alias` + `getAliasedSymbol` before anything resolves.
3. **Node kinds are not keywords.** Class constructors are
   `ConstructorDeclaration`, not `MethodDeclaration` — `isMethodDeclaration`
   silently drops them, and the chunk was missing from the graph until the regex
   diff caught it. `new X()` is `NewExpression`, not `CallExpression`, so a naive
   CALLS walker sees no constructor calls at all.
4. **`CallExpression.getStart()` is the start of the chain**, so `a\n .b()\n .c()`
   reports the line of `a`. Site lines need the callee's name token instead.
5. **Silent failure on a bad tsconfig.** Include globs resolve relative to the
   tsconfig directory. One `../` too few gives an empty project, not an error.
6. `openProject` is deprecated (`openProjects` is current), there is no free
   `forEachChild`, and `api.close()` is mandatory or the tsgo child stays alive
   (~50 ms per instance to spawn).

Verdict: a batch extractor can rely on this now, with a version pin and a
readiness check. A language server or an editor plugin should not.

## 2. Graph

14 files, 3,656 lines of ESM JavaScript with JSDoc, 147 KB.

| | TS7 AST | regex + brace counting |
|---|---:|---:|
| chunks | 165 | 204 |
| — functions | 78 | 78 |
| — methods (incl. constructor) | 61 | 111 |
| — classes | 1 | 1 |
| — script bodies | 25 | 14 |
| symbols / DEFINES | 165 | 204 |
| CALLS edges | 2,099 | 2,353 |
| — resolved to a declaration | 461 | 0 |
| — symbol known, external (e.g. `push`) | 1,003 | 0 |
| — unresolved | 635 | — |
| IMPORTS edges | 63 | 60 |
| CONTAINS | 165 | 204 |
| exported chunks | 9 | 9 |

The TS7 graph is the reference: `DEFINES`, `CALLS`, `IMPORTS` and `CONTAINS` all
come out of the tree, and 22% of call sites land on a real declaration chunk.
The other 78% are library or dynamic calls and are labelled as such rather than
guessed at.

## 3. Timing

Node startup (~65 ms) is in every process number; the in-process row is the
extractor itself.

| | process, median of 7 | in-process build |
|---|---:|---:|
| regex | 83 ms | **19 ms** |
| TS7, AST only | 178 ms | **84 ms** |
| TS7, with checker | 359 ms | **248 ms** |

TS7 breakdown with the checker: spawn + snapshot 50 ms, fetch + parse 6 ms, walk
26 ms, **checker 167 ms**. The AST-only variant is the same minus the checker:
spawn + snapshot 51 ms, parse 6 ms, walk 26 ms. 2,245 requests, 12 MB received,
30,647 AST nodes materialised, 195 ms of round trips of which 149 ms is server
time. The checker pass is the noisy part of the run: 167–236 ms across runs, i.e.
75–105 µs per round trip.

So: the AST is free, the type checker is the cost. AST-only is 4× regex; adding
declaration resolution makes it 13×. Since resolution is what buys
`CALLS.target_chunk` (and nothing else — chunk and symbol sets are identical
either way), it is a switch, not a default. `TS7_NO_CHECKER=1` builds the graph
without it, into its own output file.

## 4. TS7 vs regex

**Chunks.** Both find the same named declarations: 154 match by file+name, 0
TS7-only, 3 regex-only — and all 3 are false (`if`, `for`, `while` matched as
method declarations because `if (x) {` looks like `name(args) {` at 2-space
indent). The 165-vs-154 gap is 11 script-body chunks that share a `file+<module>`
key: TS7 emits one body chunk per contiguous uncovered statement run (up to 3 per
script), which is a definition choice, not a disagreement. On the 140 matched
non-body chunks: 140/140 identical start lines, 120 identical end lines, 19 end
early (brace counting breaks on regex literals such as
`/^\s{0,3}#{1,6}\s/` and on template literals), mean line IoU 0.871.

**CALLS.** Compared as unique `(file, line, last segment of callee)` sites:

| | sites | shared with other | exclusive |
|---|---:|---:|---:|
| TS7 | 1,969 | 1,964 | 5 |
| regex | 2,218 | 1,964 | 254 |

The regex's 254 exclusive sites are, on classification: 46 inside comments, 68
inside string/SQL literals, 137 function or method **definitions**, 3 other. That
is a false-positive rate of ~11% on its own output. TS7's 5 exclusive sites are
all correct (`delete()` — a JS keyword that a regex keyword filter has to drop,
and `engine[entry.method](...)` dynamic dispatch).

Two asymmetries that no site-count captures:

- **Only 1,712 of 2,099 TS7 edges keep the receiver** (`engine.getContent`);
  regex keeps 0 — it reports `getContent` and cannot tell two receivers apart.
- **Only TS7 resolves an edge to a declaration.** 461 edges, which is what makes
  the graph channel in §5 possible at all.

**IMPORTS.** TS7 63, regex 60, TS7 ⊇ regex. The 3 misses are all in
`scripts/stress-test-compact.js`, a minified file with four `import` statements on
one line; a line-anchored regex sees the first. Precision on the 20-edge sample is
20/20 for both.

**What each one finds that the other does not.**

| TS7 only | regex only |
|---|---|
| 461 edges resolved to a declaration chunk | 137 function/method *definitions* reported as calls |
| receiver-qualified callees (`engine.getContent`, 1,712 edges) | 46 comment and 68 string/SQL literal matches |
| `delete()` calls (dropped by any keyword filter) | 3 phantom method chunks (`if`, `for`, `while`) |
| `engine[entry.method]()` dynamic dispatch | — |
| 3 imports in a minified one-line file | — |
| exact end lines (19 regex ranges truncate) | — |

One regex advantage is real and worth recording: `new X()` arrives as a
`CallExpression` to a regex, but as a `NewExpression` to the AST — the first TS7
walker silently dropped 138 constructor calls until the diff with the regex graph
exposed it. A regex baseline is a useful oracle precisely because it fails
differently.

**Hand-checked precision.** Deterministic stride sample, every n/20th edge, each
verdict assigned by reading the source line.

| | CALLS | IMPORTS |
|---|---:|---:|
| TS7 | **20/20** | **20/20** |
| regex | **18/20** | 20/20 |

The two regex failures: a match inside a `//` comment, and `packRankedP4(queryText,
{ candidates }, settings) {` counted as a call to itself. Note that the import
sample misses the minified-file failure — sampling by stride cannot find a defect
that lives in one file, which is why the full-set diff in §4 is the honest number.

## 5. Retrieval: the same queries, two graphs

`CODE_GRAPH.md` published a 250-query evidence-chain set generated from the regex
graph. That set now exists, so the question is no longer "can the graph retrieve"
but "does the parser change the answer". `run_shared_queries.py` runs **the same
queries, the same qrels and the same channel code** (`lab/code_graph.py`,
`run_code_graph.py`) against both graphs. Nothing is re-implemented: the regex row
reproduces the published table (`porter` 0.288 / 0.419, `RRF(P,e,g)` 0.608 / 0.581),
which is the control that the harness is wired up correctly.

Chunk shapes were matched first (class header vs methods split, top-level code
split at blank lines): 283 chunks vs 289, and **571 of 571 qrel chunks map onto a
TS7 chunk by (file, start line), with zero fallbacks and 1 query dropped**. So the
evidence documents are the same spans of source. The edge counts in this section
(357 calls ts7 / 397 regex) are lower than §2's 2,099 because the channel only
keeps edges to symbols the repo defines or imports — the same filter on both sides.

| system | graph | both@5 | both@10 | nDCG@10 | R@10 |
|---|---|---:|---:|---:|---:|
| porter | regex | 0.152 | 0.288 | 0.419 | 0.561 |
| porter | **ts7** | 0.173 | **0.390** | 0.453 | 0.624 |
| entity | regex | 0.272 | 0.344 | 0.506 | 0.573 |
| entity | ts7 | 0.253 | 0.329 | 0.473 | 0.569 |
| graph | regex | 0.176 | 0.336 | 0.477 | 0.594 |
| graph | ts7 | 0.185 | 0.349 | 0.472 | 0.586 |
| RRF(P,g) | regex | 0.336 | 0.556 | 0.555 | 0.719 |
| RRF(P,g) | ts7 | 0.341 | 0.570 | 0.555 | 0.727 |
| RRF(P,e,g) | regex | 0.364 | 0.608 | 0.581 | 0.745 |
| RRF(P,e,g) | ts7 | 0.369 | 0.606 | 0.579 | 0.745 |

Paired per-query deltas, TS7 minus regex, same 249 queries:

| | both@10 | nDCG@10 | R@10 |
|---|---:|---:|---:|
| porter | **+0.104** (54W/28L) | +0.035 | +0.065 |
| graph | +0.012 (42W/42L) | −0.005 | −0.008 |
| RRF(P,e,g) | **0.000** (42W/42L) | −0.001 | +0.001 |

Three things follow.

**A better parser mostly helps BM25, not the graph.** The AST chunk set is worth
+0.104 both@10 to the lexical baseline and +0.012 to the graph channel. The
combined system lands on exactly the same number (0.606 vs 0.608). The graph's
*incremental* lift over its own baseline therefore shrinks from +0.320 to +0.216.
Read plainly: a third of the EBR headline is the regex parser's damage to BM25, not
the graph's contribution.

**The gain is segmentation, not text.** Replacing every TS7 chunk's text with the
regex chunk's text (the ablation: 260 chunks, including the 19 that the brace
matcher truncated) lowers porter from 0.390 to 0.369. So only ~0.02 of the 0.104
advantage is text; the rest is chunk boundaries — where a function starts and,
especially, where the previous one ends.

**Query provenance does not explain it.** The 250 queries were generated from the
regex graph's own edges, so that graph is in-distribution by construction. Re-running
the same five query types generated from the TS7 graph gives:

| queries from | graph | porter both@10 | RRF both@10 |
|---|---|---:|---:|
| regex | regex | 0.288 | 0.608 |
| regex | ts7 | 0.390 | 0.606 |
| ts7 | regex | 0.288 | 0.620 |
| ts7 | ts7 | 0.380 | 0.636 |

Each graph scores about the same on the other's queries. The parser comparison is
stable, and the fusion gap is ~0 in all four cells.

Still a capability test: qrels are read off a graph, so the graph channel cannot
lose by construction, and 283 chunks means top-10 is 3.5% of the corpus. What this
section establishes is narrow and useful — **swapping the parser does not change
the retrieval conclusion, and it does change the baseline.**

### Secondary: queries generated from the TS7 graph only

`make_queries.py` builds 97 queries of its own (both endpoints of a resolved call
edge, phrased as name words) and `run_retrieval.py` scores them with Porter BM25,
a symbol channel, a one-hop graph channel and RRF. It is a smaller, blunter version
of the same test:

| system | both@5 | both@10 | nDCG@10 | R@10 |
|---|---:|---:|---:|---:|
| Porter BM25 | 0.082 | 0.175 | 0.294 | 0.418 |
| symbol channel | 0.000 | 0.000 | 0.088 | 0.072 |
| graph channel | 0.062 | 0.093 | 0.377 | 0.448 |
| RRF(BM25, symbol, graph) | 0.175 | 0.289 | 0.415 | 0.541 |

Same shape of result: the graph hop is the load-bearing channel, the symbol channel
alone can never satisfy `both@k` because it cannot reach the caller, and fusion
beats either. The absolute numbers are not comparable to the table above — different
corpus, different query construction — and the 97 are kept only because they do not
depend on the other experiment's code.

## 6. Recommendation

**Use the TS7 AST for the graph; do not expect it to change the retrieval story.**

On this corpus the parser buys three things: no false call edges, exact chunk
ranges, and a better lexical baseline (+0.104 both@10 for BM25 on the same
queries). It does not buy a better graph channel (+0.012 both@10) and it does not
move the fused system (0.606 vs 0.608). So the case for the AST is correctness and
maintenance — a graph you can trust to be complete — not a number.

Concretely:

1. Default build: `TS7_NO_CHECKER=1` — 84 ms, chunks + syntactic CALLS + IMPORTS.
   This is the right default because the checker's only product is
   `CALLS.target_chunk`, and nothing else in the graph changes.
2. Opt-in: checker on, when resolved call targets are needed — 248 ms. Pin
   `typescript` exactly; the API is `unstable/*` and will move.
3. Do not ship a watch-mode or per-keystroke service on this yet: 75–105 µs per
   checker query with no batching is the wrong shape for interactivity.
4. The regex graph stays useful as an oracle, not as a product: it fails
   differently, and the `new X()` bug in the first TS7 walker was only found by
   diffing against it.
5. If the TS7 API blocks something, `oxc-parser` is the fallback that keeps AST
   fidelity. Going back to regex costs ~11% call-edge precision and ~0.10 both@10
   on the lexical baseline, for ~65 ms.

Two things would change this. A batch symbol-resolution API
(`getSymbolsAtLocations`) would make the checker pass free and collapse the
AST-only/checked split. And a parser-error budget — drop 20% of edges, as
`CODE_GRAPH.md` does — is the experiment that would actually distinguish the two
graphs, because that is where a lossy parser shows up; here both graphs are exact
enough that the retrieval delta is a baseline effect.

## Reproduce

```sh
cd /Users/asmit/Personal/algo-serp-stages/experiments/stages/ts7-graph
npm install                      # typescript 7.0.2, pinned

node graph_ts7.mjs               # out/graph_ts7.json + out/chunk_text.json
TS7_NO_CHECKER=1 node graph_ts7.mjs
node graph_regex.mjs             # out/graph_regex.json
node bench.mjs                   # out/bench.json (7 reps each)
node compare.mjs                 # out/compare.json + audit samples

/Users/asmit/Personal/algo-serp-stages/.venv/bin/python make_queries.py
/Users/asmit/Personal/algo-serp-stages/.venv/bin/python run_retrieval.py
/Users/asmit/Personal/algo-serp-stages/.venv/bin/python run_shared_queries.py
/Users/asmit/Personal/algo-serp-stages/.venv/bin/python finalize.py
```

Results: `experiments/stages/results/ts7_graph.json`.
Source is read-only; the graph scripts only write under `ts7-graph/out/` and
`../results/`.
