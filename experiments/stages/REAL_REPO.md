# Real, PR-derived code retrieval: does the code-graph / CESR direction transfer?

`CODE_GRAPH.md` generated its queries *from* the graph, so the graph channel was correct
by construction. This is the non-circular test: 92 merged pull requests from
**pmndrs/zustand**, each description used as a query, and the files each pull request
actually changed used as qrels. Nothing about the labels comes from the graph.

**Verdict: no transfer.** On file-level retrieval the lexical baseline is already at
0.828 recall@10 and 0.685 nDCG@10; fusing the graph channel *lowers* both
(0.787 / 0.621, nDCG@10 CI [−0.123, −0.006]), and the constraint selector raises
recall@10 by +0.0145 with a CI whose lower bound is exactly 0.0000 because 90 of 92
queries are ties. The synthetic +0.32 both@10 does not survive contact with real
queries. Details, and the three reasons, below.

## Read this before the numbers

1. **The qrels are noisy in a known direction.** A PR touches its test file because the
   test had to change, not because that file answers the description. 45% of the qrel
   instances are test files (58% of PRs touch one), and test files are *easier* to
   retrieve (0.914 vs 0.803 recall@10), so the noise helps every system roughly equally
   — but "file relevant" here means "file edited", which is broader than "file
   wanted".
2. **Temporal leakage.** Corpus and graph are built at repo HEAD (`d7a5583cff`,
   2026-09-29); PRs go back to 2023. `file_drift` is the measured fraction of a file's
   lines that changed between the PR's base commit and HEAD: mean 0.231, median 0.180.
   The 2025–2026 subset has mean drift 0.068 against 0.375 for 2023–2024, and is
   reported separately.
3. **The corpus is small: 48 files.** A top-10 list is 21% of it, so absolute numbers
   are inflated and the room for improvement is narrow (the random floor is 0.228
   recall@10; a perfect reranker over the candidate pool is 0.953). This is the single
   biggest reason the test is *hard to win* rather than *rigged against the graph* —
   see §7 and §11.
4. **No tuning.** Every constant was fixed before the first run, copied from
   `code_graph.py` / `cesr_code.py` or listed in §3. Nothing that selects documents
   reads qrels. Two parser bugs were fixed after the first run (multi-line signatures
   were being truncated into block chunks; a brace-less declaration could swallow the
   next one). No filter, constant or query changed with them; both are described in §2.

## 1. Data: pull requests

`gh pr list --repo pmndrs/zustand --state merged --limit 1000 --json
number,title,body,mergedAt,files,author` returns all 931 merged PRs since 2019; the whole
list, with per-file change types, is cached in `data/zustand_prs.json` so the run is
offline and reproducible. The 92 selected PRs, their cleaned query text, qrel files and
drift are written to `data/zustand_selected_prs.json`.

Selection, fixed before any retrieval number existed:

| rule | dropped |
|---|---:|
| merged before 2023-01-01 (pre-v4 layout, drift makes qrels meaningless) | 395 |
| no code file changed (docs / CI / lockfile only) | 384 |
| code-file count outside 1–5 | 12 |
| code file outside `src/`, `tests/`, `examples/` (not in the corpus) | 24 |
| dependency bump (`chore(deps)`, `build(deps)`, …) | 10 |
| bot author | 2 |
| changed file no longer exists at HEAD | 7 |
| cleaned query shorter than 8 words | 5 |
| **kept** | **92** |

| | value |
|---|---:|
| PRs | 92 (2023: 22, 2024: 27, 2025: 28, 2026: 15) |
| changed code files | 1×42, 2×31, 3×12, 4×5, 5×2 → 170 relevant file instances |
| PRs changing ≥1 test file | 53 / 92 |
| drift, all | mean 0.231, median 0.180, 52.2% ≤ 0.2 |
| drift, 2023–2024 (49 PRs) | mean 0.375 |
| drift, 2025–2026 (43 PRs) | mean 0.068 |

Query text is `title + ". " + body`, with markdown checkboxes, template sections
(`## Checklist`, `Related issues`, `Screenshots`, …), HTML comments, images, URLs and
fenced blocks longer than eight lines removed, then truncated to 200 whitespace tokens.

## 2. Corpus and parser

zustand at HEAD is `src/` (16 files), `tests/` (15) and `examples/` (17) — there is no
`packages/` directory. `examples/` is included deliberately: without it the corpus is
31 files and a top-10 list is a third of the collection.

| | value |
|---|---:|
| files | 48 (src 16, tests 15, examples 17) |
| chunks | 1,111 |
| chunks by kind | `block` 976, `type` 70, `function` 56, `interface` 9 (no classes in this repo) |
| chunks per file | mean 22.9 |
| chunk length (`dl`) | mean 46.0, median 34, max 814 |
| symbols (`DEFINES`) | 123 |
| `CALLS` edges | 1,305 (1,207 cross-file) |
| `IMPORTS` module strings / resolved to a repo file | 126 / 30 |
| `EXPORTS` edges | 46 |
| `CONTAINS` / `DEFINED_IN` | 1,111 / 1,111 |
| chunks with at least one call | 722 / 1,111 |

**Parser.** A tolerant line/brace scanner (`lab/real_repo.py:scan_file`), reusing
`code_graph.depth_map` / `strip_literals` / `block_end` and the same dual-emit
tokenisation + Porter stemming. It recognises `class`, `function`, `const f = (...) =>`,
`interface`, `type`, `enum`, class methods and class arrow properties, with TS modifiers
(`public`, `private`, `protected`, `readonly`, `abstract`, `override`, `async`, `get`,
`set`). Multi-line signatures are joined by a bracket-balance lookahead that stops at a
blank line or at the next declaration head. Everything not inside a declaration becomes
a `block` chunk, split at blank lines and capped at 40 lines.

Limitations, measured rather than asserted:

- **Declaration recall on `src/` is 88 chunks against an 87-declaration census** — the
  scanner finds essentially every top-level declaration, with overload signatures
  merged into the implementation chunk (so `useStore`'s three overloads are one chunk).
- **No non-blank line is lost**: 8,694 non-blank lines, 13 not inside any chunk
  (blocks shorter than 20 characters are dropped). Nothing is unretrievable, but 976 of
  1,111 chunks are unnamed `block`s, 97% of `tests/`. Test files are where chunking is
  least semantic.
- **Multi-line type heads truncate**: `type Devtools = <` becomes a one-line chunk and
  its continuation a block. Harmless for file retrieval, wrong for symbol facts.
- **`CALLS` is a name match, not a resolution.** It keeps a call only when the callee is
  a defined symbol or an imported name, and drops keyword names, but it cannot see
  receivers, aliases, dynamic dispatch or `new`. Of the 40 relative `IMPORTS` strings,
  30 resolve to a repo file; the other 10 are asset imports (`.png`, `.css`) and the
  remaining 86 strings are bare packages (`react`, `vitest`, `zustand`).
- **The TS7 AST was not used.** `TS7_GRAPH.md` showed the AST is the better parser
  (+0.104 `both@10` for the *lexical baseline* through segmentation alone, +0.012 for
  the graph channel). If that held here it would raise BM25 more than the graph, i.e.
  this parser's error is conservative for the systems being tested, not for the
  baseline.

## 3. Systems

All systems produce a file ranking; the metric is computed on that.

| system | definition |
|---|---|
| `bm25_chunk` | Porter BM25 over chunk text (`file + name + kind` title, chunk body), files by first appearance |
| `bm25_file` | Porter BM25, chunk scores aggregated per file by **max** |
| `bm25_file_sum` | same with **sum** aggregation (diagnostic) |
| `entity_file` | `code_graph.entity_scores` — exact defined-symbol mentions in the query, IDF weighted, hub symbols > 5% df dropped — max per file |
| `graph_file` | `code_graph.graph_scores` — BM25 top-10 seeds, two hops over `CALLS` (both directions), resolved `IMPORTS`, `DEFINED_IN→EXPORTS` — max per file |
| `graph_rrf` | unweighted RRF(BM25, entity, graph), `k=60`, depth 1000 |
| `cesr` | constraints extracted from the PR text, greedy coverage, then BM25 backfill to 10 files |
| `cesr_plus_graph` | same selector over the union of the three channel top-100s, RRF backfill |

`bm25_chunk` and `bm25_file` are **identical by construction**: the first chunk of a
file to appear in a score-ordered chunk list is that file's highest-scoring chunk, so
"first appearance" is "max aggregation". The table shows this as 92/92 exact ties. Max
is the primary aggregation; `sum` is kept because it behaves very differently (it
rewards many weakly matching chunks, i.e. large test files) and shows that the choice
matters: `sum` loses 0.221 nDCG@10 with a CI below zero.

**CESR constraints** are extracted from the PR text only — no qrels, no metadata:

- **symbols**: identifiers in the text that the graph defines, rarest first, at most 4,
  hub symbols dropped → `DEFINES(s)`, plus `CALLS(s)` if the text says "call(s)",
  `IMPORTS_FROM(file_of(s))` if it says "import(s)", `EXPORTS(file_of(s))` if it says
  "export(s)".
- **files**: paths, basenames and module stems named in the text, at most 3 →
  `COVERS(file)`. A fixed stoplist (`index`, `types`, `test`, `setup`, `utils`, `app`,
  `main`, and the module words `react`, `vanilla`, `shallow`, `middleware`) is not
  matched as a stem: those words appear in most zustand PR text, so a bare stem hit would
  fire on almost every query rather than identify a file.
- **universal part**: a coverage set of at most 3 chunks is treated as an identity and
  all of it is taken, in BM25 order. This is the only universal quantifier the language
  has; it is what stops "the file that defines `persist`" from contributing one
  arbitrary chunk.

Selection is greedy: `gain/max(dl+10,1)^0.35 + 0.01·bm25_prior − 0.2·symbol_jaccard`,
budget 2,000 tokens. 31.5% of queries extract no constraint at all and fall through to
BM25 order.

## 4. Results (92 PRs, pre-registered query text)

| system | R@1 | R@5 | R@10 | all@10 | nDCG@10 | MRR | all@10 (≤3 files) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bm25_chunk` | 0.383 | 0.749 | **0.828** | 0.783 | **0.685** | **0.674** | 0.824 |
| `bm25_file` | 0.383 | 0.749 | **0.828** | 0.783 | **0.685** | **0.674** | 0.824 |
| `bm25_file_sum` | 0.115 | 0.493 | 0.765 | 0.685 | 0.464 | 0.402 | 0.729 |
| `entity_file` | 0.082 | 0.302 | 0.313 | 0.283 | 0.233 | 0.222 | 0.306 |
| `graph_file` | 0.278 | 0.536 | 0.607 | 0.489 | 0.500 | 0.526 | 0.506 |
| `graph_rrf` | 0.321 | 0.705 | 0.787 | 0.739 | 0.621 | 0.608 | 0.765 |
| `cesr` | 0.383 | **0.798** | **0.842** | **0.793** | **0.703** | **0.682** | **0.835** |
| `cesr_plus_graph` | 0.383 | 0.787 | 0.822 | 0.783 | 0.694 | 0.669 | 0.824 |
| *random permutation* | 0.027 | 0.075 | 0.228 | 0.120 | 0.117 | 0.105 | — |

Chunk-level diagnostics (`chunk_recall@10 / @100`: does any chunk of the file appear in
the top 10 / 100 chunks):

| system | chunk R@10 | chunk R@100 |
|---|---:|---:|
| `bm25_chunk` | 0.758 | 0.947 |
| `graph_rrf` | 0.614 | 0.908 |
| `cesr` | 0.668 | 0.703 |
| `cesr_plus_graph` | 0.668 | 0.703 |

`cesr`'s chunk recall is capped by its selection size (2–3 chunks when constraints fire),
so it is a selection-size artefact, not a ranking statement.

Paired bootstrap vs `bm25_file`, 1,000 resamples, seed 1729:

| system | metric | Δ | 95% CI | W/T/L |
|---|---|---:|---|---:|
| `graph_rrf` | recall@10 | −0.0408 | [−0.0970, +0.0127] | 5/78/9 |
| `graph_rrf` | nDCG@10 | **−0.0643** | **[−0.1228, −0.0060]** | 22/32/38 |
| `graph_rrf` | recall@5 | −0.0444 | [−0.1105, +0.0199] | 6/73/13 |
| `graph_rrf` | all@10 | −0.0435 | [−0.1087, +0.0217] | 2/84/6 |
| `cesr` | recall@10 | +0.0145 | [+0.0000, +0.0399] | 2/90/0 |
| `cesr` | nDCG@10 | +0.0178 | [−0.0282, +0.0663] | 16/60/16 |
| `cesr` | recall@5 | **+0.0489** | **[+0.0018, +0.1014]** | 8/83/1 |
| `cesr` | all@10 | +0.0109 | [+0.0000, +0.0326] | 1/91/0 |
| `cesr_plus_graph` | recall@10 | −0.0054 | [−0.0399, +0.0308] | 3/85/4 |
| `cesr_plus_graph` | nDCG@10 | +0.0081 | [−0.0432, +0.0595] | 18/55/19 |
| `graph_file` | recall@10 | −0.2208 | [−0.2975, −0.1465] | 4/57/31 |
| `entity_file` | recall@10 | −0.5147 | [−0.6160, −0.4159] | 1/36/55 |

## 5. Where the graph loses

`graph_rrf` is not a neutral addition. It changes file recall@10 on 14 of 92 queries:
better on 5, worse on 9. Every loss is a correct file pushed out of the top 10 by an
expansion neighbour.

`graph_file` alone is far worse than BM25 on source files (0.745 vs 0.803) and far worse
on the metric that matters most, nDCG@10 (0.500 vs 0.685). This reproduces
`CODE_GRAPH.md`'s own finding — *the graph channel is not a ranker* — and adds the part
the synthetic benchmark could not see: it is also not a useful *reranker* when the query
already names the thing it wants. On a synthetic query ("which chunk calls X and is
defined in the same file as Y?") the graph's second hop is the answer. On a real PR
("fix(devtools): support Firefox/Safari stack format in findCallerName") the answer is
the file that contains `findCallerName`, and BM25 already has it at rank 1. The second
hop can only add `devtools.test.tsx`-style neighbours that are sometimes right, plus
`examples/*` import neighbours that are noise.

## 6. Slices

File recall@10 by qrel-file group:

| system | test files (53 queries) | source files (69 queries) |
|---|---:|---:|
| `bm25_file` | 0.914 | 0.803 |
| `graph_rrf` | 0.813 | 0.807 |
| `cesr` | 0.920 | 0.818 |
| `cesr_plus_graph` | 0.860 | 0.818 |
| `bm25_file_sum` | 0.984 | 0.657 |

Test files are easier for every system; `sum` aggregation wins them outright (0.984) and
pays for it on source files (0.657), which is why max is the primary aggregation.

Recall@10 / nDCG@10 by PR era and by qrel-set size:

| slice | n | `bm25_file` | `graph_rrf` | `cesr` | `cesr_plus_graph` |
|---|---:|---|---|---|---|
| 2023–2024 (drift 0.375) | 49 | 0.772 / 0.615 | 0.719 / 0.608 | **0.799 / 0.689** | 0.755 / 0.667 |
| 2025–2026 (drift 0.068) | 43 | 0.891 / 0.766 | 0.864 / 0.636 | 0.891 / 0.719 | **0.899 / 0.724** |
| 1 changed file | 42 | 0.881 / 0.679 | 0.833 / 0.610 | **0.905 / 0.726** | **0.905 / 0.726** |
| 2 changed files | 31 | 0.839 / 0.723 | 0.790 / 0.630 | 0.839 / 0.698 | 0.806 / 0.689 |
| 3 changed files | 12 | 0.806 / 0.751 | 0.806 / **0.755** | **0.833 / 0.791** | 0.806 / 0.754 |
| 4–5 changed files | 7 | 0.500 / 0.444 | 0.464 / 0.419 | 0.500 / 0.444 | 0.429 / 0.416 |
| PR touches a test file | 53 | 0.882 / 0.739 | 0.807 / 0.602 | **0.888 / 0.703** | 0.854 / 0.683 |
| source files only | 39 | 0.754 / 0.613 | 0.761 / **0.647** | **0.780 / 0.703** | **0.780 / 0.708** |

The graph is not a loss on the `source_only` slice (0.761 vs 0.754 recall@10, 0.647 vs
0.613 nDCG@10) and it ties on 3-file PRs at recall@10 while gaining 0.004 nDCG@10. Those
are the two places the direction predicted it might help, and both are small (n=39 and
n=12). The 2025–2026 slice, which is the cleanest measurement in the table, shows the
graph losing 0.027 recall@10 and 0.130 nDCG@10.

## 7. Candidate ceiling and where the remaining errors are

| | value |
|---|---:|
| qrel files reachable in the candidate pool (top-100 of each channel) | **0.953** |
| PRs with *all* changed files in the pool | 0.924 |
| `bm25_file` recall@10 | 0.828 |
| missed qrel-file instances at k=10 | 37 of 170 (0.218) |
| — of which in the candidate pool (addressable by a reranker) | 26 |
| — of which outside the pool (unreachable) | 11 |
| — of which test files | 8 |

So there is **0.125 of addressable headroom** and no system in this table takes it:
`cesr` captures +0.0145 of it and `graph_rrf` gives back 0.041. A graph that could not
win with headroom of 0.125 and a near-perfect pool is not being blocked by the pool.

## 8. What CESR actually did

Constraint extraction, measured against the qrels (for diagnosis only):

| | value |
|---|---:|
| queries with ≥1 constraint | 0.685 |
| queries where a defined symbol was named | 0.424 |
| queries where a file was named | 0.598 |
| mean constraints per query | 1.95 |
| **precision of a named file being a changed file** | **0.612** |
| **precision of a symbol's defining file being a changed file** | **0.500** |
| mean forced (universal) chunks | 1.15 |
| share of selected chunks outside Porter's top-100 | 0.059 |
| share of selected chunks from the graph channel | 0.362 |

The extractor is roughly half right, which is what an automatic PR reader should be
expected to be: a title names a feature and often a file, the body names a behaviour,
and the test file is rarely named at all. That precision is the first reason the gains
are small.

The second reason is the one `CESR_CODE.md` predicted: **coverage is existential**. When
the constraint names `src/middleware/devtools.ts` and BM25 already has that file at rank
1, satisfying the constraint changes nothing. `cesr` changed file recall@10 on exactly
**two** of 92 queries:

| PR | query | qrels | bm25 | cesr | mechanism |
|---|---|---|---:|---:|---|
| 1972 | `fix(types): accept undefined equalityFn for the deprecated useStore…` | `src/react.ts` | 0.00 | 1.00 | `DEFINES(useStore)` has a 3-chunk coverage set, so it became universal and all three chunks were forced; BM25 had `src/react.ts` below rank 10 |
| 2244 | `chore(test): stop testing deprecated features and test subscribe…` | `tests/shallow.test.tsx`, `tests/subscribe.test.tsx`, `tests/vanilla/subscribe.test.tsx` | 0.33 | 0.67 | the title names two of the changed files verbatim (`COVERS(tests/subscribe.test.tsx)`, `COVERS(tests/vanilla/subscribe.test.tsx)`); BM25 had them at 12+ |

Both are genuine constraint executions, and both come from the two halves of the
constraint language that are *not* graph traversal: the universal part (a coverage set
small enough to be an identity) and exact file naming. Neither is a second hop. Two out
of 92 is the whole effect.

## 9. Robustness: is the baseline saturating on path names?

PR bodies often contain file paths (`src/middleware/devtools.ts`), which is the obvious
way a lexical system could be cheating. Post-hoc check: strip every path and file name
from the query and re-run.

| system | recall@10 | nDCG@10 | Δ recall@10 vs `bm25_file` | Δ nDCG@10 |
|---|---:|---:|---|---|
| `bm25_file` | 0.806 | 0.672 | — | — |
| `graph_rrf` | 0.776 | 0.610 | −0.0299 [−0.0924, +0.0290] | −0.0626 [−0.1235, −0.0019] |
| `cesr` | 0.821 | 0.678 | +0.0145 [+0.0000, +0.0399] | +0.0063 [−0.0364, +0.0516] |
| `cesr_plus_graph` | 0.790 | 0.663 | −0.0163 [−0.0598, +0.0236] | −0.0093 [−0.0634, +0.0421] |

Stripping paths costs the baseline only 0.022 recall@10: the queries do not depend on
path leakage, because they are full of identifiers and feature words. The ordering of
systems is unchanged, and `graph_rrf`'s nDCG@10 loss survives with a CI below zero. The
conclusion is not an artefact of path names.

## 10. Gate

Pre-registered: *`graph_rrf` or `cesr` improves file recall@10 or nDCG@10 over
`bm25_file` with a paired-bootstrap 95% CI above zero.*

| check | Δ | 95% CI | above zero |
|---|---:|---|---|
| `graph_rrf` recall@10 | −0.0408 | [−0.0970, +0.0127] | no |
| `graph_rrf` nDCG@10 | −0.0643 | [−0.1228, −0.0060] | no (significantly worse) |
| `cesr` recall@10 | +0.0145 | [+0.0000, +0.0399] | no |
| `cesr` nDCG@10 | +0.0178 | [−0.0282, +0.0663] | no |
| `cesr_plus_graph` recall@10 | −0.0054 | [−0.0399, +0.0308] | no |
| `cesr_plus_graph` nDCG@10 | +0.0081 | [−0.0432, +0.0595] | no |

**Gate FAILED.** Two readings must be kept apart:

- `cesr` is a *strictly non-negative* intervention: 2 wins, 0 losses, 90 ties on
  recall@10. Its only CI that excludes zero is file recall@5 (+0.0489,
  [+0.0018, +0.1014]), which is not a metric in the gate. The reason the recall@10 CI
  lower bound is exactly 0.0000 is arithmetic, not luck: with two non-tied queries out of
  92, ~13% of bootstrap resamples contain neither of them. A tie-heavy delta cannot
  produce a lower bound above zero, however favourable the point estimate.
- `graph_rrf` is *strictly harmful* on this benchmark: nDCG@10 −0.064 with a CI below
  zero, recall@5 −0.044, and 9 losses to 5 wins.

So this is not "no signal"; it is "a signal too small to clear the bar, and a graph
contribution that is negative".

## 11. Verdict: does the direction transfer to real code search?

**No.** The synthetic +0.32 both@10 has three supports, and none of them survives here.

1. **The synthetic queries needed structure; these do not.** `two_hop` and
   `export_import` asked questions whose answers contain neither query symbol. A PR
   description names the thing it changes, and the file that defines it is rank 1 for
   BM25. There is no second hop to take. The graph channel's only measurable effect is
   displacing correct files with import-neighbours.
2. **The synthetic qrels were graph-true; these are edit-true.** 45% of the 170
   changed-file instances are test files (58% of PRs touch at least one), and the
   description names the source file, not the test. A constraint set built from the text
   can identify the source file (which is already found) and almost never the test file
   (which is the actual miss). The 11 files no channel can reach (§7) are mostly the same
   problem.
3. **The CESR objective is existential and the evidence is not.** `CESR_CODE.md`
   measured this on synthetic data (72.5% of full-coverage queries still missed gold);
   here it shows up as +0.0145 from exactly 2 queries. Nothing changed except the
   extractor being automatic: file-hit precision 0.612, symbol-file precision 0.500.

What would change the verdict — none of it cheap:

- **A corpus where lexical matching is weak.** 48 files is too few for a top-10 file
  list to be informative (ceiling 0.953, baseline 0.828). A 1,000-file repo with the
  same PRs would leave the graph room to matter, and is the one experiment that could
  overturn this.
- **Qrels that are "the file a developer would open", not "the files the PR touched".**
  Dropping test files from the qrels puts the baseline at 0.803 recall@10 (0.754 on the
  39 source-only PRs instead of 0.828), which is both harder and closer to the question a
  code search engine is actually asked.
- **A universal constraint language.** "every caller of X", "the file that imports Y and
  exports Z" — the relations the graph actually knows. The two wins in §8 are exactly
  this shape. But `CESR_CODE.md` already showed that even oracle constraints only pay
  when the satisfying set is near-unique.

What would *not* change it: a better parser. `TS7_GRAPH.md` found the AST helps the
lexical baseline more than the graph (+0.104 vs +0.012 `both@10`), so parser
improvements widen this gap rather than close it.

## 12. Honest limits

- **92 PRs, 48 files, one repository, one language.** Nothing here is a population
  estimate; the CIs are over queries, not over repositories.
- **The corpus is small.** 46% of queries have exactly one relevant file, so `recall@10`
  is close to `hit@10` and the metric is coarse. The random floor (0.228) and the pool
  ceiling (0.953) bound it.
- **Temporal leakage is measured, not removed.** Half the PRs are from 2023–2024 with
  mean drift 0.375; those qrels describe code that has since been rewritten. The
  2025–2026 slice is the clean one and it is reported separately, with the same ordering.
- **Qrels are edit-derived.** A file changed for a refactor is not the best evidence for
  the description; 45% of the qrel instances are tests. This inflates every system and is
  disclosed rather than corrected.
- **The parser is lossy** in the ways listed in §2. Its 1,305 call edges come from the
  same family of extractor that `TS7_GRAPH.md` measured at ~11% false positives on
  JavaScript. The edge-noise experiment in `CODE_GRAPH.md` suggests two thirds of a graph
  advantage dies at 20% edge loss; the advantage here is already zero, so a better parser
  would raise the ceiling on the graph from zero, not raise the graph.
- **`cesr`'s 2-wins result is not a null result.** It is a positive point estimate that
  the pre-registered rule correctly refuses to accept; §4 and §10 report it both ways.
- **One post-hoc variant** (`no_paths`) was added after the primary run. It is labelled,
  it is symmetric across systems, and it does not change the ordering.

## 13. Files and reproduce

- `experiments/stages/lab/real_repo.py` — TS chunker, graph, channels, constraint
  extraction, CESR selector, file metrics, bootstrap, gate
- `experiments/stages/run_real_repo.py` — runner
- `experiments/stages/results/real_repo.json` — every number, CI, slice, the 92 PR
  titles, miss decomposition, changed queries, 3 worked samples
- `experiments/stages/data/zustand_prs.json` — cached `gh pr list` payload (931 PRs with
  file lists) + the HEAD commit it was fetched at
- `experiments/stages/data/zustand_selected_prs.json` — the 92 kept PRs with cleaned
  query text, qrel files, drift and the drop counts
- `experiments/stages/repos/zustand` — the clone, gitignored, never committed

```sh
cd /Users/asmit/Personal/algo-serp-stages
git clone --depth 1 https://github.com/pmndrs/zustand.git experiments/stages/repos/zustand
git -C experiments/stages/repos/zustand fetch --unshallow   # drift measurement
PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_real_repo.py
```

Runtime: 0.7–1.6 s corpus build (file-cache dependent), 7 ms/query, 8–9 s total for both
query variants. All metrics are byte-identical across runs (verified twice); only the
latency fields vary.
