# sqlite-context

Local retrieval for coding agents: an MCP server and a Node library on SQLite
FTS5. Every answer fits the token budget you ask for. It runs offline, with no
embedding model and no vector database.

## What you can do with it

- **Give an agent memory across sessions.** Store decisions, conventions and
  notes with `store_memory`. Recall the ones that fit a budget with
  `pack_context`.
- **Keep large tool output out of the context.** Index a test log, a build log,
  a diff or a lockfile with `index_output`, then fetch only the relevant lines
  with `search_output`.
- **Look up exact identifiers.** `error_42`, `ACME-412` or `useValidatedQuery`
  rank above any partial match, and an identifier that isn't indexed returns
  nothing instead of loosely related prose.
- **Search a codebase.** `npm run demo:codebase` chunks a repository by
  function and class and queries it. It's a demo script, not an MCP tool.
- **Build your own retrieval.** The engine is a plain ES module with a staged
  API: retrieve, fuse and pack ([docs/api.md](docs/api.md)).

## Quick start

Requires Node.js 24 or newer. `better-sqlite3` is a native module, so
`npm install` may need a compiler when no prebuilt binary is available.

```sh
git clone git@github.com:AlbertSmit/sqlite-context.git
cd sqlite-context
npm install
```

Add it to Claude Code:

```sh
claude mcp add sqlite-context \
  -e MCP_SQLITE_PATH="$HOME/.sqlite-context/memory.sqlite" \
  -- node "$PWD/src/mcp/server.js"
```

Other MCP clients use the same command over stdio:

```json
{
  "mcpServers": {
    "sqlite-context": {
      "command": "node",
      "args": ["/absolute/path/to/sqlite-context/src/mcp/server.js"],
      "env": { "MCP_SQLITE_PATH": "/absolute/path/to/memory.sqlite" }
    }
  }
}
```

Without `MCP_SQLITE_PATH`, memory goes to `mcp_memory.sqlite` in the repository
root. The schema is created on first use.

## Recipes

**Remember something, recall it later**

```text
store_memory  { "content": "Releases go out on Thursday after the smoke test passes.", "doc_id": "release-process" }
pack_context  { "query": "release day", "budget": 250 }
```

`pack_context` returns the matching chunks with their token cost, and never
more than `budget` tokens. Query with keywords, not a full sentence: a chunk
that covers too little of the query is left out, and filler words count. "when
do we release" returns nothing here, "release day" finds the note. An empty
result means nothing stored covers the query well enough.

**Read a long log without pasting it**

```sh
npm test > /tmp/test.log 2>&1
```

```text
index_output   { "path": "/tmp/test.log", "label": "npm test" }
search_output  { "query": "AssertionError payment", "budget": 250 }
```

`index_output` returns a short receipt (size, chunk count, first lines), not the
log. `search_output` returns the matching lines. Pass `path`, not `content`:
inline content has already cost the tokens it took to write out. Read the raw
output instead when you need all of it, such as a diff you're about to apply.

**Search a repository**

```sh
npm run demo:codebase -- /path/to/repository
```

This builds `codebase_graph.sqlite` in the current directory and prints
example queries.

## Tools

| Tool | Does | Options |
| --- | --- | --- |
| `store_memory` | Stores a chunk of text | `content` (required), `doc_id`, `chunk_id` |
| `pack_context` | Returns stored chunks within a budget | `query` (required), `budget` (default 250, 30–4,000), `reinforce_access` (default on), `mode`, `use_bridges`, `temporal_mode` |
| `index_output` | Indexes text or a file into a scratch index and returns a receipt | `path` (file up to 20 MB) or `content`; `source`, `label` |
| `search_output` | Returns scratch-index chunks within a budget | `query` (required), `budget` (default 250, 64–4,000), `source` |

Things to know:

- **Budgets** count the engine's tokens (words and identifiers), not your
  model's tokens.
- **The scratch index** lives in memory and is cleared when the server restarts.
  It never mixes with stored memory.
- **`path` reads any file** the server process can read. There is no tool that
  runs commands.
- **`reinforce_access`** counts how often a chunk is recalled.
- **`mode: "legacy"`, `use_bridges` and `temporal_mode`** switch to the older P4
  packer. It returns fewer tokens with lower recall; keep the default unless
  you're comparing.

## How well it works

| Use | Result |
| --- | --- |
| Large tool output, 250-token budget | 94.8% fewer engine tokens than the full output; the answer was found for 13 of 16 questions written before the test (`node scripts/output-benchmark.js`) |
| Exact identifiers | An exact identifier ranks above its spelling variants; 0 of 82 probes with unknown identifiers pulled in unrelated results |
| General retrieval | BM25 ranking; no lexical alternative beat it on all of 7 BEIR test sets |

The tool-output savings are measured against pasting the whole output. They
exclude the receipt and any follow-up searches. Three short lookups (a version
number, a value in a table, a test total) were missed at every budget.
[docs/research.md](docs/research.md) has every measurement, with confidence
intervals and the negative results.

## Use it as a library

```js
import { readFileSync } from 'node:fs'
import { SQLiteContextEngine } from './src/engine/engine.js'
import { indexText, searchText } from './src/engine/scratch.js'

const engine = new SQLiteContextEngine('./memory.sqlite')
engine.storeMemory('The API validates JWT claims at the ingress gateway.', 'architecture')
const chunkIds = engine.packHybrid('JWT validation gateway', { budget: 250 })
const contents = chunkIds.map(id => engine.getContent(id))

const scratch = new SQLiteContextEngine(':memory:')
indexText(scratch, readFileSync('build.log', 'utf8'), { label: 'build' })
const { results } = searchText(scratch, 'TS2345', { budget: 120 })
```

## Documentation

- [docs/api.md](docs/api.md): engine API, indexes, the coverage gate, the scratch index
- [docs/practical-guide.md](docs/practical-guide.md): which mode to use when
- [docs/research.md](docs/research.md): measurements, confidence intervals, limitations
- [docs/history-p4.md](docs/history-p4.md): the original P4 scorer

## Development

| Command | Runs |
| --- | --- |
| `npm test` | Engine, stage, scratch-index and MCP tests |
| `npm run lint` | ESLint |
| `npm run check` | ESLint and the tests |
| `npm run stress` | Throughput, spam, pooling and decay probes (needs `mcp_memory.sqlite`) |
| `node scripts/stage-benchmark.js` | Chunk-level packing benchmark |
| `node scripts/output-benchmark.js` | Tool-output savings next to answers found |

```text
src/engine/engine.js      Storage, retrieval stages, packers, legacy P4 path
src/engine/tokenize.js    Code-aware tokenization and token counting
src/engine/scratch.js     Scratch index for tool output
src/mcp/server.js         MCP server
scripts/                  Demos and benchmarks
test/                     Tests and fixtures
experiments/stages/       Research lab (Python) and result data
python/, native/          Legacy P4 Python engine and native FTS5 scorer
```

## Background

This project started as SERP, an attempt to find a no-embedding scorer that
beats BM25 everywhere. None of the candidates did, so it now ships BM25
retrieval with budgeted packing. The research, including multi-hop and
structured-evidence prototypes that aren't in this engine, is in
[docs/research.md](docs/research.md).
