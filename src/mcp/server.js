import { pathToFileURL } from 'node:url';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { mkdirSync, readFileSync, statSync } from 'node:fs';
import { dirname } from 'node:path';
import { indexText, searchText } from '../engine/scratch.js';
import { SQLiteContextEngine, DEFAULT_DB_PATH } from '../engine/engine.js';
import { tokenize } from '../engine/tokenize.js';
import { createEventLog, eventLogPath, readEvents } from '../events.js';
import { summarize } from '../stats.js';

const MAX_INDEX_BYTES = 20 * 1024 * 1024;

export function createMcpServer(engine, scratch = new SQLiteContextEngine(":memory:"), log = createEventLog(null)) {
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
      log.record({
        tool: "pack_context", query, budget, mode: legacy ? "legacy" : "hybrid",
        results: chunks.length, tokens_returned: usedTokens, latency_ms: Number(latencyMs),
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
      log.record({ tool: "store_memory", doc_id, tokens: meta?.L ?? 0, error: meta ? undefined : "empty" });
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

  server.tool(
    "index_output",
    "Index large text (command output, file content, a fetched page) into a session-scoped scratch index instead of the context. Returns a receipt, not the content; query it with search_output. Separate from store_memory/pack_context. Prefer path: inline content has already cost the tokens it took to write.",
    {
      content: z.string().max(MAX_INDEX_BYTES).optional().describe("Text to index"),
      path: z.string().optional().describe("Index this regular file (at most 20 MB) instead of content"),
      source: z.string().regex(/^[A-Za-z0-9_.-]{1,64}$/).optional().describe("Source id; defaults to out_<n>"),
      label: z.string().max(200).default("").describe("Short description, e.g. the command that produced the output"),
    },
    async ({ content, path, source, label }) => {
      const t0 = performance.now();
      try {
        const receipt = { status: "indexed", ...indexText(scratch, outputText(content, path), { source, label }) };
        log.record({
          tool: "index_output", source: receipt.source, label, from_path: path !== undefined,
          raw_tokens: receipt.raw_tokens, receipt_tokens: tokenize(JSON.stringify(receipt)).length,
          chunks: receipt.chunks, latency_ms: elapsed(t0),
        });

        return jsonResult(receipt);
      } catch (error) {
        log.record({ tool: "index_output", error: error.message, latency_ms: elapsed(t0) });

        return jsonError(error.message);
      }
    }
  );

  server.tool(
    "search_output",
    "Search the scratch index filled by index_output and return matching chunks within a token budget (BM25 + 4-gram RRF + truncation). Does not search store_memory memory.",
    {
      query: z.string().min(1).describe("What to find, e.g. an error code, symbol or message"),
      budget: z.number().int().min(64).max(4000).default(250).describe("Maximum total engine tokens returned"),
      source: z.string().optional().describe("Restrict to one source id from an index_output receipt"),
    },
    async ({ query, budget, source }) => {
      const t0 = performance.now();
      const found = searchText(scratch, query, { budget, source });
      log.record({
        tool: "search_output", query, budget, source: source || null,
        results: found.chunks_count, tokens_returned: found.tokens_returned, latency_ms: elapsed(t0),
      });

      return jsonResult(found);
    }
  );

  server.tool(
    "usage_stats",
    "Summarize this server's logged usage: memory recalls and empty results, tool-output indexing and searches, token savings, latency.",
    {
      days: z.number().int().min(1).max(365).optional().describe("Only count the last N days"),
    },
    async ({ days }) => jsonResult({ log: log.path, ...summarize(readEvents(log.path), { days }) })
  );

  return server;
}

function outputText(content, path) {
  if ((content === undefined) === (path === undefined)) throw new Error("Pass exactly one of content or path");
  if (content !== undefined) return content;
  const stats = statSync(path);
  if (!stats.isFile() || stats.size > MAX_INDEX_BYTES) {
    throw new Error(`Not a regular file of at most ${MAX_INDEX_BYTES} bytes: ${path}`);
  }

  return readFileSync(path, "utf8");
}

function elapsed(start) {
  return Number((performance.now() - start).toFixed(2));
}

function jsonResult(value) {
  return { content: [{ type: "text", text: JSON.stringify(value, null, 2) }] };
}

function jsonError(message) {
  return { content: [{ type: "text", text: JSON.stringify({ error: message }) }], isError: true };
}

export async function startMcpServer() {
  const dbPath = process.env.MCP_SQLITE_PATH || DEFAULT_DB_PATH;
  mkdirSync(dirname(dbPath), { recursive: true });
  const engine = new SQLiteContextEngine(dbPath);
  const server = createMcpServer(engine, undefined, createEventLog(eventLogPath(dbPath)));
  await server.connect(new StdioServerTransport());
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await startMcpServer();
}
