import assert from 'node:assert/strict';
import test from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { SQLiteContextEngine, DEFAULT_DB_PATH } from '../src/engine/engine.js';
import { tokenize, countTokens } from '../src/engine/tokenize.js';
import { createMcpServer } from '../src/mcp/server.js';
import * as compatibility from '../mcp_server.js';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

test('compatibility exports and default database location are preserved', () => {
  assert.equal(compatibility.SQLiteContextEngine, SQLiteContextEngine);
  assert.equal(compatibility.tokenize, tokenize);
  assert.equal(compatibility.countTokens, countTokens);
  assert.equal(DEFAULT_DB_PATH, fileURLToPath(new URL('../mcp_memory.sqlite', import.meta.url)));
});

test('MCP transport lists tools, stores memory and retrieves packed context', async () => {
  const engine = new SQLiteContextEngine(':memory:');
  for (let i = 0; i < 10; i++) {
    engine.storeMemory(`Unrelated background document number ${i}.`, `background_${i}`, `background_${i}`);
  }
  const server = createMcpServer(engine);
  const client = new Client({name: 'integration-test', version: '1.0.0'});
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  try {
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    const {tools} = await client.listTools();
    assert.deepEqual(tools.map(x => x.name), ['pack_context', 'store_memory', 'index_output', 'search_output']);
    const stored = await client.callTool({name: 'store_memory', arguments: {
      content: 'SQLite context memory uses lexical retrieval.', doc_id: 'memory', chunk_id: 'memory_1',
    }});
    assert.equal(JSON.parse(stored.content[0].text).status, 'stored');
    const packed = await client.callTool({name: 'pack_context', arguments: {query: 'sqlite', budget: 100}});
    const result = JSON.parse(packed.content[0].text);
    assert.equal(result.budget_limit, 100);
    assert.equal(result.retrieval_mode, 'hybrid');
    assert.equal(result.packer, 'truncate');
    assert.equal(result.chunks_count, 1);
    assert.equal(result.results[0].chunk_id, 'memory_1');
    assert.equal(engine.docStats.get('memory_1').access_count, 2);

    const legacy = await client.callTool({name: 'pack_context', arguments: {
      query: 'sqlite', budget: 100, mode: 'legacy',
    }});
    const legacyResult = JSON.parse(legacy.content[0].text);
    assert.equal(legacyResult.retrieval_mode, 'legacy');
    assert.equal(legacyResult.packer, 'p4');
    assert.ok(legacyResult.tokens_packed <= 100);
    const empty = await client.callTool({name: 'store_memory', arguments: {content: '!!!!!'}});
    assert.equal(empty.isError, true);
  } finally {
    await client.close();
    await server.close();
    engine.db.close();
  }
});

test('MCP output tools index into a scratch index isolated from memory', async () => {
  const engine = new SQLiteContextEngine(':memory:');
  const scratch = new SQLiteContextEngine(':memory:');
  engine.storeMemory('Deployment notes mention the payment service.', 'memory', 'memory_1');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sqlite-context-'));
  const server = createMcpServer(engine, scratch);
  const client = new Client({name: 'integration-test', version: '1.0.0'});
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  try {
    await server.connect(serverTransport);
    await client.connect(clientTransport);
    const log = [
      ...Array.from({length: 200}, (_, i) => `PASS src/module_${i}.test.js ok`),
      'FAIL src/payment.test.js ERR_PAYMENT_DECLINED_4012 payment card rejected',
    ].join('\n');
    const indexed = await client.callTool({name: 'index_output', arguments: {content: log, label: 'npm test'}});
    const receipt = JSON.parse(indexed.content[0].text);
    assert.equal(receipt.status, 'indexed');
    assert.equal(receipt.source, 'out_1');
    assert.ok(receipt.chunks > 1);
    assert.ok(!indexed.content[0].text.includes('ERR_PAYMENT_DECLINED_4012'));

    const found = JSON.parse((await client.callTool({name: 'search_output', arguments: {
      query: 'ERR_PAYMENT_DECLINED_4012', budget: 120,
    }})).content[0].text);
    assert.ok(found.tokens_returned <= 120);
    assert.ok(found.results[0].content.includes('ERR_PAYMENT_DECLINED_4012'));

    const memory = JSON.parse((await client.callTool({name: 'pack_context', arguments: {
      query: 'payment ERR_PAYMENT_DECLINED_4012', budget: 500,
    }})).content[0].text);
    assert.ok(memory.results.every((x) => !x.content.includes('ERR_PAYMENT_DECLINED_4012')));
    assert.equal(engine.docStats.size, 1);
    const scratchOnly = JSON.parse((await client.callTool({name: 'search_output', arguments: {
      query: 'deployment payment', budget: 500,
    }})).content[0].text);
    assert.ok(scratchOnly.results.length > 0);
    assert.ok(scratchOnly.results.every((x) => !x.content.includes('Deployment notes')));

    const file = path.join(dir, 'build.log');
    fs.writeFileSync(file, 'webpack compiled with 2 warnings\nwarning TS2345 in src/app.ts');
    const fromFile = JSON.parse((await client.callTool({name: 'index_output', arguments: {
      path: file, source: 'build',
    }})).content[0].text);
    assert.equal(fromFile.source, 'build');
    const both = await client.callTool({name: 'index_output', arguments: {content: 'x y', path: file}});
    assert.equal(both.isError, true);
    const missing = await client.callTool({name: 'index_output', arguments: {path: path.join(dir, 'nope')}});
    assert.equal(missing.isError, true);
    const duplicate = await client.callTool({name: 'index_output', arguments: {content: 'again text', source: 'build'}});
    assert.equal(duplicate.isError, true);
  } finally {
    await client.close();
    await server.close();
    engine.db.close();
    scratch.db.close();
    fs.rmSync(dir, {recursive: true, force: true});
  }
});
