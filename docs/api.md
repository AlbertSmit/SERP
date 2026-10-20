# Engine API

[Back to the README](../README.md) · [Mode guide](practical-guide.md) · [Research](research.md)

## Quick start

```js
import { SQLiteContextEngine } from './src/engine/engine.js'

const engine = new SQLiteContextEngine('./memory.sqlite')

try {
  engine.storeMemory(
    'The API validates JWT claims at the ingress gateway.',
    'architecture',
    'architecture_auth'
  )

  const chunkIds = engine.packHybrid('JWT validation gateway', { budget: 250 })
  const contents = chunkIds.map(x => engine.getContent(x))
  console.log(contents)
} finally {
  engine.db.close()
}
```

`packHybrid` is the default path, and it is what the MCP `pack_context` tool
uses. `packContext` and `fullPipeline` are the legacy P4 path, kept for
compatibility (see [Legacy P4 path](#legacy-p4-path)).

## Retrieval stages

`packHybrid` is three stages, and each one can be called on its own:

```js
const { word, gram, fused } = engine.retrieveRanked('JWT validation gateway')
const chunkIds = engine.packRanked('JWT validation gateway', fused, {
  budget: 250,
  packer: 'truncate',
})
// Equivalent convenience entry point:
const same = engine.packHybrid('JWT validation gateway', { budget: 250, packer: 'truncate' })
```

- `retrieveRanked(query, { channels: ['word', 'gram'], depth: 100 })` returns the
  word BM25 list, the in-word 4-gram BM25 list, and their fused list. Rank lists
  contain `{ chunk_id, score }`, with positive BM25 relevance as the score.
  `rankWordBm25` and `rankGramBm25` return the individual lists.
- `fuseRrf(lists, { k: 60, depth: 100, queryText })` applies reciprocal-rank
  fusion. Pass `queryText` (or `queryTokens`) so that chunks containing an exact
  identifier token sort first. This is a hard rule, not a weight.
- `packRanked(query, fused, options)` applies the coverage gate, then packs the
  budget and returns chunk IDs. `packer` is `truncate` (default), `greedy`,
  `residual` or `p4`; all but `truncate` are experimental. Every packer sees the
  same `poolDepth` (default 100) candidate pool. `packRankedDetailed` also returns
  `skippedCoverage`, `usedTokens`, the effective budget and the packer.
- `packHybrid(query, options)` runs retrieve, fuse and pack in one call.

Budgets count the engine's lexical tokens, not a model tokenizer's tokens.
Full-document chunk IDs ending in `_full` are excluded from packing.

## Gram and exact-identifier indexes

The gram channel is a separate FTS5 table (`chunks_grams`) of in-word character
4-grams with word-boundary markers. Exact identifiers are tracked in
`chunks_exact_terms`, populated from `tokenize()` with underscores preserved.
New memories are indexed in the word, gram and exact tables in the same
transaction. Constructors do not backfill existing databases, so do it
explicitly:

```js
engine.buildGramIndex()  // number of chunks indexed, or 0 if the table is not empty
engine.buildExactIndex() // number of chunks indexed, or 0 if the table is not empty
engine.exactIdentifierTerms({ maxDf: 20 }) // [{ term, df }]
```

Both channels filter identifier-like tokens (containing `_` or a digit) per
token. Such a token is used only when it exists in `chunks_exact_terms`. Without
the filter, unicode61 would split `figure_1` into `figure` + `1` and match prose
that never contained the identifier. A query can mix unknown identifiers with
known tokens (`zzq_42 p53`): the known tokens still retrieve, and the unknown
ones activate nothing. Fusion sorts chunks containing an exact identifier token
(a letter plus a digit or underscore) first, so an exact `error_42` outranks
`error_43`.

## Coverage gate

`packRanked` and `packContext` keep only chunks that cover enough of the
query's weighted terms (`min_coverage`, default `0.315`). A rare identifier, one
containing `_` or a digit that occurs in at most `max(2, floor(0.02 × corpus
size))` documents, bypasses the gate. The gate is a precision control. At 250
tokens, gated truncation scores 0.601 document recall and 0.206 precision; with
no gate it scores 0.688 and 0.101 ([research](research.md#packing)).

## Legacy P4 path

`packContext(query, options)` and `fullPipeline(query, options)` run the original
P4/SubConPack scorer and residual packer. In the MCP server this path is
`mode: "legacy"`, or `use_bridges` / `temporal_mode`. On the chunk benchmark it
fills only about 47 of 250 tokens: it has lower recall and higher precision than
truncation.

- Candidate pooling filters terms that occur in more than
  `max(floor(0.4 × corpus size), 25)` documents. Vaults of at most 100 chunks also
  use an abstaining trigram fallback for close lexical variants.
- `fullPipeline` adds bridge expansion and temporal weighting; both default to
  on there. Callers can pass explicit `bridgeCandidates` to `packContext`.
- Development evaluations can pass `min_coverage` and `rel_dropoff` (frozen
  defaults `0.315` and `0.26`).

Scoring rules are in [history-p4.md](history-p4.md).

## Scratch index for tool output

`src/engine/scratch.js` holds session-scoped output indexing for the MCP
`index_output` and `search_output` tools.

- `chunkText(text, maxTokens)` splits text on newlines (CRLF tolerated). A line
  over the token limit is split at single spaces. Lines are packed greedily, in
  order and joined with `\n`, until the next line would exceed the limit. Chunks
  with no tokens are dropped. `maxTokens` defaults to 64 engine tokens.
- `indexText(engine, text, { source, label })` stores the chunks and returns a
  receipt: `source`, `label`, `raw_tokens`, `raw_chars`, `chunks`, and `head`
  (the first non-empty lines, each truncated). The source defaults to `out_<n>`.
  A duplicate source or text with no indexable tokens throws.
- `searchText(engine, query, { budget, source })` returns `query`, `budget_limit`,
  `source`, `tokens_returned`, `chunks_count`, and `results` (each with `rank`,
  `chunk_id`, `source`, `tokens_L`, `content`). `budget` defaults to 250. `source`
  filters the fused top-100 pool, so a restricted search can return fewer
  results than a full one.

Use a separate engine instance for the scratch index
(`new SQLiteContextEngine(':memory:')`), so memory stays isolated. A shared
engine would mix IDF statistics and the candidate pool between memory and tool
output.

```js
import { readFileSync } from 'node:fs'
import { SQLiteContextEngine } from './src/engine/engine.js'
import { indexText, searchText } from './src/engine/scratch.js'
const scratch = new SQLiteContextEngine(':memory:')
const receipt = indexText(scratch, readFileSync('build.log', 'utf8'), { label: 'npm test' })
console.log(searchText(scratch, 'ERR_PAYMENT_DECLINED', { source: receipt.source }).results)
```

## Event log and usage stats

`src/events.js` appends one JSON line per MCP tool call. `src/stats.js`
summarizes those lines.

- `eventLogPath(dbPath, env)` returns `SQLITE_CONTEXT_LOG` when set, `null`
  when it is `off` or the database is `:memory:`, and otherwise
  `sqlite-context.events.jsonl` next to the database.
- `createEventLog(path)` returns `{ path, record(event) }`. Each record gets a
  timestamp and a per-process `session` id. A failed write is ignored, so
  logging never fails a tool call. `createEventLog(null)` records nothing.
- `readEvents(path)` parses the log and skips malformed lines.
- `summarize(events, { days })` returns `period` (from, to, sessions, calls),
  `memory` (stored, recalls, empty rate, tokens returned, latency p50/p95),
  `output` (indexed, raw tokens, searches, searches per index, empty rate,
  tokens received, token savings, latency) and `errors`.

Logged fields per tool:

| Tool | Fields |
| --- | --- |
| `store_memory` | `doc_id`, `tokens`, `error` |
| `pack_context` | `query`, `budget`, `mode`, `results`, `tokens_returned`, `latency_ms` |
| `index_output` | `source`, `label`, `from_path`, `raw_tokens`, `receipt_tokens`, `chunks`, `latency_ms`, `error` |
| `search_output` | `query`, `budget`, `source`, `results`, `tokens_returned`, `latency_ms` |

`createMcpServer(engine, scratch, log)` takes the log as its third argument;
`startMcpServer` builds it from `MCP_SQLITE_PATH`. `src/cli.js` exposes
`runCli(argv, env)` for `stats` and `doctor`, which return an exit code.

## Tokenization

Tokenization helpers are exported from `src/engine/tokenize.js`. The legacy
`mcp_server.js` entry point also re-exports the engine and the tokenization
helpers.
