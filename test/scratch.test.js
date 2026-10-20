import assert from 'node:assert/strict';
import test from 'node:test';
import { SQLiteContextEngine } from '../src/engine/engine.js';
import { tokenize } from '../src/engine/tokenize.js';
import { chunkText, indexText, searchText } from '../src/engine/scratch.js';

const ERROR_CODE = 'ERR_PAYMENT_DECLINED_4012';

function testLog() {
  const lines = Array.from({length: 300}, (_, i) => `PASS src/module_${i}.test.js ok assertion count 3`);
  lines[150] = `FAIL src/payment.test.js Error ${ERROR_CODE} card rejected`;

  return lines.join('\n');
}

test('chunkText packs whole lines up to 64 engine tokens and keeps every line', () => {
  const text = Array.from({length: 50}, () => 'alpha beta gamma delta').join('\n');
  const chunks = chunkText(text);
  assert.equal(chunks.length, 4);
  assert.ok(chunks.every((chunk) => tokenize(chunk).length <= 64));
  assert.equal(chunks.join('\n'), text);
});

test('chunkText splits a long line at spaces', () => {
  const line = Array.from({length: 150}, (_, i) => `word${i}`).join(' ');
  const chunks = chunkText(line);
  assert.deepEqual(chunks.map((chunk) => tokenize(chunk).length), [64, 64, 22]);
  assert.equal(chunks.join(' '), line);
});

test('indexText returns a receipt without the content and numbers sources', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    const text = testLog();
    const receipt = indexText(engine, text, {label: 'npm test'});
    assert.equal(receipt.source, 'out_1');
    assert.equal(receipt.label, 'npm test');
    assert.equal(receipt.raw_chars, text.length);
    assert.equal(receipt.raw_tokens, tokenize(text).length);
    assert.equal(receipt.chunks, engine.docStats.size);
    assert.equal(receipt.head.length, 5);
    assert.ok(!JSON.stringify(receipt).includes(ERROR_CODE));
    assert.equal(indexText(engine, 'second output text').source, 'out_2');
    assert.throws(() => indexText(engine, 'again', {source: 'out_1'}), /already indexed/);
    assert.throws(() => indexText(engine, '!!! ???'), /No indexable/);
  } finally {
    engine.db.close();
  }
});

test('searchText stays within budget and finds an exact error code', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    indexText(engine, testLog());
    for (const budget of [64, 120, 250, 500]) {
      const result = searchText(engine, 'payment error card', {budget});
      assert.ok(result.tokens_returned <= budget);
      assert.equal(result.tokens_returned, result.results.reduce((sum, x) => sum + x.tokens_L, 0));
    }
    const exact = searchText(engine, ERROR_CODE, {budget: 120});
    assert.ok(exact.results[0].content.includes(ERROR_CODE));
    assert.deepEqual(searchText(engine, 'ERR_PAYMENT_DECLINED_9999', {budget: 120}).results, []);
  } finally {
    engine.db.close();
  }
});

test('searchText can be restricted to one source', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    indexText(engine, 'build timeout after 30 seconds in step compile', {source: 'build'});
    indexText(engine, 'deploy timeout after 90 seconds in step upload', {source: 'deploy'});
    assert.equal(searchText(engine, 'timeout', {budget: 250}).results.length, 2);
    const deployOnly = searchText(engine, 'timeout', {budget: 250, source: 'deploy'});
    assert.deepEqual(deployOnly.results.map((x) => x.source), ['deploy']);
  } finally {
    engine.db.close();
  }
});
