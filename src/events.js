import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

export const EVENT_LOG_NAME = "sqlite-context.events.jsonl";

export function eventLogPath(dbPath, env = process.env) {
  if (env.SQLITE_CONTEXT_LOG === "off") return null;
  if (env.SQLITE_CONTEXT_LOG) return env.SQLITE_CONTEXT_LOG;
  if (dbPath === ":memory:") return null;

  return join(dirname(resolve(dbPath)), EVENT_LOG_NAME);
}

export function createEventLog(path) {
  const session = `${process.pid}-${Date.now()}`;

  return {
    path,
    record(event) {
      if (!path) return;
      try {
        appendFileSync(path, `${JSON.stringify({ ts: new Date().toISOString(), session, ...event })}\n`);
      } catch {
        // A full disk or a read-only log must never fail the tool call itself.
      }
    },
  };
}

export function readEvents(path) {
  if (!path || !existsSync(path)) return [];

  return readFileSync(path, "utf8").split("\n").flatMap((line) => parseLine(line));
}

function parseLine(line) {
  if (!line.trim()) return [];
  try {
    return [JSON.parse(line)];
  } catch {
    return [];
  }
}
