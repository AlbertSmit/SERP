import { pathToFileURL } from 'node:url';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { SQLiteContextEngine, DEFAULT_DB_PATH } from '../engine/engine.js';

export function createMcpServer(engine) {
  const server = new McpServer({
    name: "sqlite-context",
    version: "1.2.0",
  });

  server.tool(
    "pack_context",
    "Retrieve memory chunks within a token budget using BM25 + 4-gram RRF and truncation by default. Use mode='legacy' for the experimental P4 packer, or enable bridges/temporal for the legacy full pipeline.",
    {
      query: z.string().describe("Search query or multi-aspect compound prompt"),
      budget: z.number().int().min(30).max(4000).default(250).describe("Maximum total token budget (B)"),
      mode: z.enum(["hybrid", "legacy"]).default("hybrid").describe("hybrid = BM25 + 4-gram RRF + truncation; legacy = experimental P4 packer"),
      use_bridges: z.boolean().default(false).describe("Legacy only: enable 1-hop speculative lexical bridge chaining"),
      temporal_mode: z.boolean().default(false).describe("Legacy only: enable entropy-regulated temporal memory decay"),
      reinforce_access: z.boolean().default(true).describe("Increment Hebbian access_count on retrieved chunks"),
    },
    async ({ query, budget, mode, use_bridges, temporal_mode, reinforce_access }) => {
      const t0 = performance.now();
      const legacy = mode === "legacy" || use_bridges || temporal_mode;
      const chunkIds = legacy
        ? engine.fullPipeline(query, { budget, use_bridges, temporal_mode, reinforce_access })
        : engine.packHybrid(query, {
            budget,
            channels: ["word", "gram"],
            packer: "truncate",
          });
      if (!legacy && reinforce_access) engine.reinforceSelected(chunkIds);
      const latencyMs = (performance.now() - t0).toFixed(2);

      let usedTokens = 0;
      const chunks = chunkIds.map((cid, idx) => {
        const meta = engine.docStats.get(cid) || { L: 0, U: 0, doc_id: "" };
        usedTokens += meta.L;

        return {
          rank: idx + 1,
          chunk_id: cid,
          doc_id: meta.doc_id,
          tokens_L: meta.L,
          unique_U: meta.U,
          content: engine.getContent(cid),
        };
      });

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                query,
                budget_limit: budget,
                retrieval_mode: legacy ? "legacy" : "hybrid",
                packer: legacy ? "p4" : "truncate",
                tokens_packed: usedTokens,
                chunks_count: chunks.length,
                latency_ms: Number(latencyMs),
                results: chunks,
              },
              null,
              2
            ),
          },
        ],
      };
    }
  );

  server.tool(
    "store_memory",
    "Store a new atomic memory chunk in SQLite FTS5, automatically computing L, U, L/U boilerplate ratio, and corpus-IDF intrinsic surprisal.",
    {
      content: z.string().min(5).describe("The atomic fact, architectural decision, or snippet to remember"),
      doc_id: z.string().default("mcp_session").describe("Logical topic, project, or document ID"),
      chunk_id: z.string().optional().describe("Optional custom chunk ID (must not end in _full)"),
    },
    async ({ content, doc_id, chunk_id }) => {
      const meta = engine.storeMemory(content, doc_id, chunk_id || null);
      if (!meta) {
        return {
          content: [{ type: "text", text: JSON.stringify({ error: "Empty token content" }) }],
          isError: true,
        };
      }

      return {
        content: [{ type: "text", text: JSON.stringify({ status: "stored", ...meta }, null, 2) }],
      };
    }
  );

  return server;
}

export async function startMcpServer() {
  const engine = new SQLiteContextEngine(process.env.MCP_SQLITE_PATH || DEFAULT_DB_PATH);
  const server = createMcpServer(engine);
  await server.connect(new StdioServerTransport());
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await startMcpServer();
}
