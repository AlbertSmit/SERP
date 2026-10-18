# SERP

**Stretched-Exponential Residual Packing.**

A lexical retrieval engine that stores memory in SQLite FTS5 and selects chunks
within a token budget. It combines P4 stretched-exponential scoring, residual
query-term decay, optional lexical bridge expansion, and temporal memory weighting.
Retrieval runs locally without embedding models or a vector database.

The JavaScript engine can run as a library or an MCP server. The repository also
contains a Python implementation, SciFact ingestion and benchmark tools, and a
native C scorer.

## Setup

Use Node.js 24 or newer. `better-sqlite3` is a native dependency, so installing it
may require a compiler toolchain when a prebuilt binary is unavailable.

The development dependency `@kaliber/code-cli` links to a private, installed
`kaliber-code-intelligence` checkout. The current link points to
`/Volumes/Development/kaliber-code-intelligence/apps/code-cli`. Install that
checkout's workspace dependencies with `pnpm install` at its root first.

Clone the repository, then install dependencies:

```sh
git clone git@github.com:AlbertSmit/SERP.git
cd SERP
npm install
npm test
npm run check
```

If your KCI checkout lives elsewhere, update the local dependency:

```sh
npm install --save-dev /path/to/kaliber-code-intelligence/apps/code-cli
```

KCI is development tooling. Its embedding model is separate from the retrieval
engine and is not used to store or retrieve application memory.

## Run the MCP server

```sh
npm start
```

The server uses stdio. For an MCP client, invoke Node directly so npm's status
output does not share the protocol stream:

```json
{
  "mcpServers": {
    "sqlite-context": {
      "command": "node",
      "args": ["/absolute/path/to/SERP/src/mcp/server.js"],
      "env": {
        "MCP_SQLITE_PATH": "/absolute/path/to/memory.sqlite"
      }
    }
  }
}
```

`MCP_SQLITE_PATH` is optional. By default, the server opens `mcp_memory.sqlite`
at the repository root and creates the schema when needed. Existing client
configurations using `node mcp_server.js` still work.

| Tool | Purpose | Options |
| --- | --- | --- |
| `store_memory` | Store a memory chunk and its token statistics | Required `content`; optional `doc_id` and `chunk_id` |
| `pack_context` | Select chunks within a token budget | Required `query`; optional `budget`, `use_bridges`, `temporal_mode`, and `reinforce_access` |

For `pack_context`, the default budget is 250, with an accepted range of
30–4,000. Bridges and temporal weighting default to off; access reinforcement
defaults to on. Budgets count the engine's lexical tokens, not a model tokenizer's
tokens. Full-document chunk IDs ending in `_full` are excluded from packing.

## Use the engine directly

```js
import { SQLiteContextEngine } from './src/engine/engine.js'

const engine = new SQLiteContextEngine('./memory.sqlite')

try {
  engine.storeMemory(
    'The API validates JWT claims at the ingress gateway.',
    'architecture',
    'architecture_auth'
  )

  const chunkIds = engine.packContext('JWT validation gateway', { budget: 250 })
  const contents = chunkIds.map(x => engine.getContent(x))
  console.log(contents)
} finally {
  engine.db.close()
}
```

Candidate pooling filters terms appearing in more than roughly 15% of the corpus,
so a tiny corpus can return no matches even for words it contains. Library callers
can provide explicit `bridgeCandidates` to `packContext` when needed. Use
`fullPipeline(query, options)` for bridge expansion and temporal weighting;
these default to on for the library's full pipeline.

Tokenization helpers are exported from `src/engine/tokenize.js`. The legacy
`mcp_server.js` entry point also re-exports the engine and tokenization helpers.

## Project layout

```text
src/
  engine/
    engine.js                  SQLite storage, scoring, packing, bridges and decay
    tokenize.js                Code-aware tokenization and token counting
  mcp/
    server.js                  MCP tools, server factory and stdio startup
scripts/
  index-codebase.js            Codebase indexing and retrieval demo
  stress-test.js               Corpus-based stress suite
  stress-test-compact.js       Compact variant of the stress suite
python/
  sqlite_mcp_engine.py         Python retrieval engine
  build_mcp_db.py              SciFact corpus and synthetic-noise database builder
  benchmark_sqlite_e2e.py      Python end-to-end benchmark
  schema_ingest.py             Legacy database schema
  ingest_corpus.py             Legacy corpus ingestion
native/
  fts5_p4_scorer.c             Native scorer source
  fts5_p4_scorer.dylib         Local compiled artifact, when built
test/
  engine.test.js              Engine regressions against captured original outputs
  mcp.test.js                 MCP integration and compatibility checks
  fixtures/                   Captured engine results
docs/
  algorithm.md                Algorithm notes and historical benchmark tables
  practical-guide.md          Python/native integration notes
mcp_server.js                 Compatibility entry point
```

The engine imports SQLite and tokenization code. The MCP adapter depends on the
engine. Scripts and tests import the modules they use; importing the engine does
not load the MCP server or transport.

Generated databases, SQLite sidecars, compiled extensions, dependencies and caches
are ignored. Existing databases remain at the repository root.

## Demos and benchmarks

Run these commands from the repository root:

```sh
npm run demo:codebase -- /path/to/repository
npm run stress
```

The codebase demo rebuilds `codebase_graph.sqlite` in the current working
directory. The stress suite copies the existing root `mcp_memory.sqlite` into a
temporary database and runs throughput, spam, candidate-pooling and decay probes.
These scripts print observations; they are separate from the automated test suite.

The Python tools need Python 3 with SQLite FTS5 support and the SciFact files at
`../datasets/scifact/`: `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`.

```sh
# Replaces the root mcp_memory.sqlite with the benchmark corpus.
python3 python/build_mcp_db.py
python3 python/benchmark_sqlite_e2e.py
```

The legacy schema and ingestion scripts use the separate root `mcp_memory.db`.
To build the optional native extension on macOS:

```sh
clang -shared -fPIC -O3 native/fts5_p4_scorer.c -o native/fts5_p4_scorer.dylib
```

The JavaScript engine does not require that extension. See the
[algorithm notes and reported benchmarks](docs/algorithm.md) and the
[practical integration guide](docs/practical-guide.md) for background. Historical
benchmark numbers have not been rerun as part of the code cleanup.

## Development checks

| Command | What it runs |
| --- | --- |
| `npm test` | Engine regressions and MCP integration tests |
| `npm run lint` | ESLint across JavaScript sources |
| `npm run lint:fix` | ESLint with automatic fixes |
| `npm run kci:scan` | Refresh the KCI index, then run its lint rule set |
| `npm run kci:check` | Run KCI lint against an existing, current index |
| `npm run check` | ESLint, then the KCI scan; run tests separately |
| `npm run kci -- scan .` | Scan all KCI rule sets |

The tests cover 120 captured pre-refactor cases for selection order, budgets,
temporal decay and access reinforcement, plus bridge weights, scores, empty
inputs, explicit candidates, rare-term pooling, and MCP storage/retrieval.

ESLint uses its recommended rules plus selected standalone settings from
`kaliber-code-intelligence`. It does not import Kaliber's ESLint config package,
plugin, or repository-local rules. Complexity, nesting, callback and parameter
limits are configured as warnings; the current sources have zero warnings.

KCI reads `code-vectors.config.json` and indexes JavaScript/TypeScript. Its index
and embedding model live in `~/.cache/kaliber-code-vectors/`; the first scan can
download the model. Python and C are outside these lint checks. KCI fails when it
finds violations and refuses a stale index, so run `kci:scan` after source or
configuration changes.
