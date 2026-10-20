# sqlite-context

Local memory retrieval for LLM agents: SQLite FTS5, BM25, a token budget, and an
MCP server. It needs no embedding model and no vector database.

You store text chunks. A query returns the best chunks that fit a token budget.
Retrieval is word BM25 fused (RRF, `k = 60`) with a gated in-word 4-gram channel
for spelling variants. Exact identifiers such as `error_42` always rank first,
and unknown identifiers never match prose. Ranked chunks are packed by
truncation.

> **Status.** This project used to be called SERP, after its original P4
> "Stretched-Exponential Residual Packing" scorer. A frozen research program
> tested whether a no-embedding scorer could beat BM25 everywhere. **None did.**
> BM25 is the default ranker, and P4 is a legacy option. Multi-hop retrieval
> (EBR) and constraint-driven evidence sets (CESR-U) were measured as research
> prototypes; they are not in this engine. See [docs/research.md](docs/research.md)
> for every result, including the negative ones.

## Setup

Use Node.js 24 or newer. `better-sqlite3` is a native dependency, so installing
it may need a compiler toolchain when no prebuilt binary is available.

```sh
git clone git@github.com:AlbertSmit/sqlite-context.git
cd sqlite-context
npm install
npm run check
```

## Run the MCP server

The server uses stdio. Point your MCP client at Node directly, so npm's output
does not share the protocol stream:

```json
{
  "mcpServers": {
    "sqlite-context": {
      "command": "node",
      "args": ["/absolute/path/to/sqlite-context/src/mcp/server.js"],
      "env": {
        "MCP_SQLITE_PATH": "/absolute/path/to/memory.sqlite"
      }
    }
  }
}
```

`MCP_SQLITE_PATH` is optional. By default the server opens `mcp_memory.sqlite`
at the repository root and creates the schema when needed. `npm start` runs the
same server, and existing configurations using `node mcp_server.js` still work.

| Tool | Purpose | Options |
| --- | --- | --- |
| `store_memory` | Store a chunk and its token statistics | Required `content`; optional `doc_id` and `chunk_id` |
| `pack_context` | Return chunks within a token budget | Required `query`; optional `budget` (default 250, range 30–4,000), `mode` (`hybrid` default, `legacy` P4), `use_bridges`, `temporal_mode`, `reinforce_access` |
| `index_output` | Index large text into a session-scoped scratch index and return a receipt | Exactly one of `content` or `path` (regular file of at most 20 MB); optional `source` (defaults to `out_<n>`), `label` |
| `search_output` | Return scratch chunks within a token budget | Required `query`; optional `budget` (default 250, range 64–4,000), `source` |

Budgets count the engine's lexical tokens, not a model tokenizer's. Bridges and
temporal weighting are legacy options and default to off. Access reinforcement
defaults to on.

## Use it as a library

```js
import { SQLiteContextEngine } from './src/engine/engine.js'

const engine = new SQLiteContextEngine('./memory.sqlite')
engine.storeMemory('The API validates JWT claims at the ingress gateway.', 'architecture')
const chunkIds = engine.packHybrid('JWT validation gateway', { budget: 250 })
```

[docs/api.md](docs/api.md) covers the staged API (`retrieveRanked`, `fuseRrf`,
`packRanked`), index backfills for existing databases, the coverage gate, and the
legacy `packContext` path.

## Keep large tool output out of the context

Output from a command or a file is split into chunks of whole lines, each at most 64 engine tokens, and stored in an in-memory scratch index owned by the server process. The scratch index is never mixed with `store_memory`/`pack_context` memory, and it disappears when the server restarts. `search_output` uses the same BM25 + 4-gram RRF + truncation as `pack_context`.

Flow: redirect the command's output to a file with your agent's own shell, call `index_output` with `path`, then call `search_output`. Passing `content` inline does not save tokens for text the model had to write out.

Security: `path` reads any regular file the server process can read. There is deliberately no command-execution tool.

Measured with `node scripts/output-benchmark.js` on four committed outputs from this repository (a `git log -p` commit, a source file, `package-lock.json`, a TAP test run) and 16 questions in `test/fixtures/outputs/questions.json`, fixed before the first run:

| budget | evidence_retained | returned_tokens | raw_tokens | token_savings | char_savings |
|---|---|---|---|---|---|
| 120 | 10/16 | 1483 | 56656 | 97.4% | 97.9% |
| 250 | 13/16 | 2930 | 56656 | 94.8% | 95.5% |
| 500 | 13/16 | 4251 | 56656 | 92.5% | 93.4% |

Savings count engine tokens and characters, not model tokens, against pasting the whole output into the context. They exclude the receipt (18–45 engine tokens per output) and any follow-up searches. Evidence retained is the share of questions whose answer string appears in the returned chunks; three questions miss at every budget (a lockfile version, a delta in a diff table, a test total), so a larger budget does not fix every lookup.

## What to expect

| Use | Status | Evidence |
| --- | --- | --- |
| General retrieval | **Default.** BM25 is the safest baseline measured | Every lexical alternative lost to BM25 on at least one of 7 BEIR sets |
| Spelling variants, identifiers | **Default.** 4-gram channel as a recall layer | Touche nDCG@10 +0.029 [+0.013, +0.046]; SciFact neutral. 0 of 82 unknown-ID probes activate |
| Packing | **Truncation default.** P4/residual are experimental | P4 fills about 47 of 250 tokens: doc recall 0.49 vs 0.60, precision 0.37 vs 0.21 |
| Multi-hop questions | Research prototype (Python) | EBR both@10 +0.105 on HotpotQA; transfers to 2Wiki and MuSiQue; hurts single-hop |
| Structured code evidence | Research prototype (Python) | CESR-U works when the query names its constraints; synthetic queries only |
| Code search over PRs | **Not supported** | Code graph and CESR did not beat BM25 on real PRs |

[docs/practical-guide.md](docs/practical-guide.md) explains when to use each mode.

## Documentation

- [docs/api.md](docs/api.md): engine API, indexes, the coverage gate, the legacy path
- [docs/practical-guide.md](docs/practical-guide.md): choosing a mode
- [docs/research.md](docs/research.md): findings, benchmarks with CIs, limitations, reproduction
- [docs/history-p4.md](docs/history-p4.md): the original P4 scorer and its superseded benchmarks
- [PLAN.md](PLAN.md) and [experiments/stages/](experiments/stages/): research plan, protocol, and per-experiment reports

## Project layout

```text
src/engine/engine.js         SQLite storage, retrieval stages, packers, legacy P4 path
src/engine/tokenize.js       Code-aware tokenization and token counting
src/engine/scratch.js        Scratch index: chunking, receipts, budgeted search over tool output
src/mcp/server.js            MCP tools, server factory and stdio startup
mcp_server.js                Compatibility entry point
scripts/                     Codebase demo, stress suite, chunk benchmark, legacy benchmarks
test/                        Engine regressions, stage checks, MCP integration
experiments/stages/          Research lab (Python), protocols, reports and result JSON
python/, native/             Legacy P4 Python engine and native FTS5 scorer
docs/                        API, mode guide, research findings, P4 history
```

The engine imports only SQLite and the tokenizer. The MCP adapter depends on the
engine, and importing the engine does not load the MCP server.

## Demos

```sh
npm run demo:codebase -- /path/to/repository   # rebuilds codebase_graph.sqlite in the cwd
npm run stress                                 # throughput, spam, pooling and decay probes
node scripts/stage-benchmark.js                # chunk-level packing benchmark (needs mcp_memory.sqlite)
node scripts/output-benchmark.js               # scratch-index savings next to evidence retained
```

These scripts print observations; they are separate from the test suite. The
Python tools under `python/` and the native scorer belong to the legacy P4
track. See [docs/history-p4.md](docs/history-p4.md).

## Development

| Command | What it runs |
| --- | --- |
| `npm test` | Engine regressions, stage checks and MCP integration tests |
| `npm run lint` | ESLint across JavaScript sources |
| `npm run lint:fix` | ESLint with automatic fixes |
| `npm run check` | ESLint and the test suite |

Structural lint limits are warnings. The legacy P4 path and the older benchmark
scripts still have complexity and nesting warnings, but there are no errors.
Python, C, cloned benchmark repositories and the TS 7 graph lab are not linted.
