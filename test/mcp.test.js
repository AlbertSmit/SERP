import assert from 'node:assert/strict';
import test from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { SQLiteContextEngine, DEFAULT_DB_PATH } from '../src/engine/engine.js';
import { tokenize, countTokens } from '../src/engine/tokenize.js';
import { createMcpServer } from '../src/mcp/server.js';
import * as compatibility from '../mcp_server.js';
import { fileURLToPath } from 'node:url';

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
    assert.deepEqual(tools.map(x => x.name), ['pack_context', 'store_memory']);
    const stored = await client.callTool({name: 'store_memory', arguments: {
      content: 'SQLite context memory uses lexical retrieval.', doc_id: 'memory', chunk_id: 'memory_1',
    }});
    assert.equal(JSON.parse(stored.content[0].text).status, 'stored');
    const packed = await client.callTool({name: 'pack_context', arguments: {query: 'sqlite', budget: 100}});
    const result = JSON.parse(packed.content[0].text);
    assert.equal(result.budget_limit, 100);
    assert.equal(result.chunks_count, 1);
    assert.equal(result.results[0].chunk_id, 'memory_1');
    assert.equal(engine.docStats.get('memory_1').access_count, 2);
    const empty = await client.callTool({name: 'store_memory', arguments: {content: '!!!!!'}});
    assert.equal(empty.isError, true);
  } finally {
    await client.close();
    await server.close();
    engine.db.close();
  }
});
