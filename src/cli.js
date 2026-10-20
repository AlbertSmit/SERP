#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { accessSync, constants, existsSync } from "node:fs";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { SQLiteContextEngine, DEFAULT_DB_PATH } from "./engine/engine.js";
import { eventLogPath, readEvents } from "./events.js";
import { summarize } from "./stats.js";

const MIN_NODE_MAJOR = 24;
const USAGE = `Usage: sqlite-context <command> [options]

Commands:
  stats    Summarize logged usage (memory recalls, tool-output savings, latency)
  doctor   Check Node, the database, its indexes, the event log and Claude Code

Options:
  --db <path>   Database (default: $MCP_SQLITE_PATH, else ${DEFAULT_DB_PATH})
  --days <n>    stats: only the last n days
  --json        stats: print JSON
  --fix         doctor: backfill missing gram and exact-identifier indexes`;

export function runCli(argv, env = process.env) {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      db: { type: "string" },
      days: { type: "string" },
      json: { type: "boolean", default: false },
      fix: { type: "boolean", default: false },
    },
  });
  const dbPath = values.db || env.MCP_SQLITE_PATH || DEFAULT_DB_PATH;
  const logPath = eventLogPath(dbPath, env);

  if (positionals[0] === "stats") return printStats(logPath, values);
  if (positionals[0] === "doctor") return doctor(dbPath, logPath, values.fix);
  console.log(USAGE);

  return positionals[0] ? 1 : 0;
}

function printStats(logPath, { days, json }) {
  const summary = { log: logPath, ...summarize(readEvents(logPath), { days: days && Number(days) }) };
  if (json) {
    console.log(JSON.stringify(summary, null, 2));

    return 0;
  }
  const { period, memory, output } = summary;
  console.log(`log       ${logPath ?? "off"}`);
  console.log(`period    ${period.from ?? "-"} → ${period.to ?? "-"}  (${period.sessions} sessions, ${period.calls} calls, ${summary.errors} errors)`);
  console.log(`memory    ${memory.stored} stored, ${memory.recalls} recalls, ${percent(memory.empty_rate)} empty, ${memory.tokens_returned} tokens returned, p50 ${ms(memory.latency_ms.p50)}`);
  console.log(`output    ${output.indexed} indexed (${output.raw_tokens} tokens), ${output.searches} searches (${output.searches_per_index} per index), ${percent(output.empty_rate)} empty, p50 ${ms(output.latency_ms.p50)}`);
  console.log(`savings   ${output.tokens_received} tokens received for ${output.raw_tokens} indexed: ${percent(output.token_savings)} saved`);

  return 0;
}

function doctor(dbPath, logPath, fix) {
  const checks = [checkNode(), ...checkDatabase(dbPath, fix), checkLog(logPath), checkClaude()];
  for (const { level, name, detail } of checks) console.log(`${level.padEnd(5)} ${name.padEnd(12)} ${detail}`);

  return checks.some((x) => x.level === "fail") ? 1 : 0;
}

function checkNode() {
  const major = Number(process.versions.node.split(".")[0]);

  return check(major >= MIN_NODE_MAJOR, "node", `v${process.versions.node} (needs ${MIN_NODE_MAJOR}+)`);
}

function checkDatabase(dbPath, fix) {
  if (dbPath !== ":memory:" && !existsSync(dbPath)) {
    return [{ level: "warn", name: "database", detail: `${dbPath} does not exist yet; the server creates it on first use` }];
  }
  const engine = new SQLiteContextEngine(dbPath);
  try {
    if (fix) {
      engine.buildGramIndex();
      engine.buildExactIndex();
    }
    const chunks = engine.docStats.size;
    const grams = Number(engine.stmtGramCount.get().cnt);
    const exact = Number(engine.stmtExactTermCount.get().cnt);

    return [
      { level: "ok", name: "database", detail: `${dbPath} (${chunks} chunks)` },
      indexCheck("gram index", chunks, grams),
      indexCheck("exact index", chunks, exact),
    ];
  } finally {
    engine.db.close();
  }
}

function indexCheck(name, chunks, rows) {
  if (chunks === 0 || rows > 0) return { level: "ok", name, detail: `${rows} rows` };

  return { level: "warn", name, detail: "empty while chunks exist; run `sqlite-context doctor --fix`" };
}

function checkLog(logPath) {
  if (!logPath) return { level: "warn", name: "event log", detail: "off (SQLITE_CONTEXT_LOG=off); stats will be empty" };
  try {
    accessSync(existsSync(logPath) ? logPath : dirname(logPath), constants.W_OK);
  } catch {
    return { level: "fail", name: "event log", detail: `${logPath} is not writable` };
  }
  const events = readEvents(logPath);

  return { level: "ok", name: "event log", detail: `${logPath} (${events.length} events, last ${events.at(-1)?.ts ?? "never"})` };
}

function checkClaude() {
  const result = spawnSync("claude", ["mcp", "get", "sqlite-context"], { encoding: "utf8", timeout: 15000 });
  if (result.error) return { level: "warn", name: "claude code", detail: "claude CLI not found; skipped" };

  if (result.status !== 0) return { level: "warn", name: "claude code", detail: "not registered; see the README quick start" };

  return { level: "ok", name: "claude code", detail: "sqlite-context MCP server is registered" };
}

function check(passed, name, detail) {
  return { level: passed ? "ok" : "fail", name, detail };
}

function percent(value) {
  return value === null ? "-" : `${(value * 100).toFixed(1)}%`;
}

function ms(value) {
  return value === null ? "-" : `${value} ms`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exitCode = runCli(process.argv.slice(2));
}
