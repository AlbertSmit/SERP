import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { SQLiteContextEngine } from '../src/engine/engine.js';
import { createEventLog, eventLogPath, readEvents } from '../src/events.js';
import { createMcpServer } from '../src/mcp/server.js';
import { summarize } from '../src/stats.js';
import { runCli } from '../src/cli.js';

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'sqlite-context-usage-'));
}

test('the event log sits next to the database and can be turned off', () => {
  assert.equal(eventLogPath('/data/memory.sqlite', {}), '/data/sqlite-context.events.jsonl');
  assert.equal(eventLogPath('/data/memory.sqlite', {SQLITE_CONTEXT_LOG: '/logs/x.jsonl'}), '/logs/x.jsonl');
  assert.equal(eventLogPath('/data/memory.sqlite', {SQLITE_CONTEXT_LOG: 'off'}), null);
  assert.equal(eventLogPath(':memory:', {}), null);
});

test('summarize counts empty recalls and computes savings from receipts and results', () => {
  const ts = new Date().toISOString();
  const events = [
    {ts, session: 'a', tool: 'store_memory', doc_id: 'x', tokens: 9},
    {ts, session: 'a', tool: 'pack_context', results: 2, tokens_returned: 40, latency_ms: 3},
    {ts, session: 'a', tool: 'pack_context', results: 0, tokens_returned: 0, latency_ms: 5},
    {ts, session: 'b', tool: 'index_output', raw_tokens: 1000, receipt_tokens: 30, chunks: 20},
    {ts, session: 'b', tool: 'search_output', results: 3, tokens_returned: 70, latency_ms: 2},
    {ts, session: 'b', tool: 'index_output', error: 'Pass exactly one of content or path'},
    {ts: '2020-01-01T00:00:00.000Z', session: 'old', tool: 'pack_context', results: 0},
  ];
  const summary = summarize(events, {days: 7});
  assert.equal(summary.period.sessions, 2);
  assert.equal(summary.errors, 1);
  assert.deepEqual([summary.memory.stored, summary.memory.recalls, summary.memory.empty_rate], [1, 2, 0.5]);
  assert.deepEqual([summary.output.indexed, summary.output.raw_tokens, summary.output.tokens_received], [1, 1000, 100]);
  assert.equal(summary.output.token_savings, 0.9);
  assert.equal(summarize(events).memory.recalls, 3);
});

test('MCP calls are logged and usage_stats summarizes them', async () => {
  const dir = tempDir();
  const logPath = path.join(dir, 'events.jsonl');
  const engine = new SQLiteContextEngine(':memory:');
  const server = createMcpServer(engine, undefined, createEventLog(logPath));
  const client = new Client({name: 'usage-test', version: '1.0.0'});
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  try {
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    await client.callTool({name: 'store_memory', arguments: {content: 'Releases go out on Thursday.', doc_id: 'release'}});
    await client.callTool({name: 'pack_context', arguments: {query: 'release thursday'}});
    await client.callTool({name: 'index_output', arguments: {content: 'FAIL src/a.test.js ERR_X_42 broke\nPASS src/b.test.js'}});
    await client.callTool({name: 'search_output', arguments: {query: 'ERR_X_42'}});

    const events = readEvents(logPath);
    assert.deepEqual(events.map((x) => x.tool), ['store_memory', 'pack_context', 'index_output', 'search_output']);
    assert.ok(events.every((x) => x.ts && x.session === events[0].session));
    assert.equal(events[1].query, 'release thursday');
    assert.ok(events[2].raw_tokens > 0 && events[2].receipt_tokens > 0);

    const stats = JSON.parse((await client.callTool({name: 'usage_stats', arguments: {}})).content[0].text);
    assert.equal(stats.log, logPath);
    assert.deepEqual([stats.memory.recalls, stats.output.indexed, stats.output.searches], [1, 1, 1]);
  } finally {
    await client.close();
    await server.close();
    engine.db.close();
    fs.rmSync(dir, {recursive: true, force: true});
  }
});

test('doctor reports a healthy database and --fix backfills missing indexes', () => {
  const dir = tempDir();
  const dbPath = path.join(dir, 'memory.sqlite');
  const env = {SQLITE_CONTEXT_LOG: path.join(dir, 'events.jsonl')};
  const engine = new SQLiteContextEngine(dbPath);
  engine.storeMemory('The gateway validates JWT claims.', 'architecture');
  engine.db.exec('DELETE FROM chunks_grams');
  engine.db.close();
  const lines = [];
  const log = console.log;
  console.log = (line) => lines.push(line);
  try {
    assert.equal(runCli(['doctor', '--db', dbPath], env), 0);
    assert.ok(lines.some((x) => x.startsWith('warn') && x.includes('gram index')));
    lines.length = 0;
    runCli(['doctor', '--db', dbPath, '--fix'], env);
    assert.ok(lines.some((x) => x.startsWith('ok') && x.includes('gram index')));
    lines.length = 0;
    assert.equal(runCli(['stats', '--db', dbPath, '--json'], env), 0);
    assert.equal(JSON.parse(lines.join('\n')).period.calls, 0);
  } finally {
    console.log = log;
    fs.rmSync(dir, {recursive: true, force: true});
  }
});
