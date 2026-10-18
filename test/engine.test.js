import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { SQLiteContextEngine } from '../src/engine/engine.js';
import { tokenize } from '../src/engine/tokenize.js';

// Captured from the original engine before the lint refactor.
const baseline = JSON.parse(fs.readFileSync(new URL('./fixtures/engine-baseline.json', import.meta.url), 'utf8'));

test('selection order, budgets, temporal decay and reinforcement match the original engine', () => {
  const engine = fixtureEngine();
  assert.ok(baseline.cases.filter(x => x.selected.length > 0).length > 50);
  try {
    for (const entry of baseline.cases) verifyCase(engine, entry);
    const bridges = engine.expandBridges('sqlite bridge', ['chunk_0', 'chunk_1', 'chunk_2']);
    assert.deepEqual(bridges.bridgeEvidence, baseline.bridges.evidence);
    assert.deepEqual([...bridges.psiMap], baseline.bridges.psi);
    for (const score of baseline.scores) verifyScore(engine, score);
  } finally {
    engine.db.close();
  }
});

function fixtureEngine() {
  const engine = new SQLiteContextEngine(':memory:');
  for (const doc of baseline.documents) {
    engine.storeMemory(doc.content, doc.id, doc.chunk, doc.age);
  }

  return engine;
}

function verifyCase(engine, entry) {
  const actual = engine[entry.method](entry.query, entry.options);
  assert.deepEqual(actual, entry.selected, JSON.stringify(entry.options));
  assert.deepEqual(actual.map(x => engine.docStats.get(x).access_count), entry.accesses);
  assert.ok(actual.reduce((total, id) => total + engine.docStats.get(id).L, 0) <= entry.options.budget);
  assert.ok(!actual.includes('excluded_full'));
}

function verifyScore(engine, expected) {
  assert.equal(engine.scoreChunkP4(expected.id, 'sqlite bridge'), expected.score);
  assert.equal(engine.temporalFactor(expected.id), expected.temporal);
}

test('empty corpus, explicit undefined defaults and bridge candidates', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    assert.deepEqual(engine.packContext('sqlite'), []);
    assert.deepEqual(engine.fullPipeline('sqlite'), []);
    assert.deepEqual(engine.expandBridges('sqlite', []), {bridgeEvidence: [], psiMap: new Map()});
    assert.equal(engine.storeMemory('!', 'empty'), null);
    engine.storeMemory('SQLite stores context memory.', 'doc', 'chunk');
    engine.storeMemory('SQLite stores full documents.', 'doc', 'doc_full');
    const options = {bridgeCandidates: ['chunk', 'missing', 'doc_full'], bridgeBoostMap: new Map([['chunk', 2]])};
    assert.deepEqual(engine.packContext('sqlite', options), ['chunk']);
    assert.deepEqual(engine.packContext('sqlite', {...options, budget: undefined}), ['chunk']);
    assert.deepEqual(engine.packContext('sqlite', {...options, budget: 0}), []);
    assert.deepEqual(tokenize('C++ C# .env R?'), ['cpp', 'csharp', 'dotenv_env', 'lang_r']);
  } finally {
    engine.db.close();
  }
});

test('per-term pooling finds a rare aspect behind 105 common matches', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    for (let i = 0; i < 105; i++) {
      engine.storeMemory(`quantumentanglement superconductor decoherence run ${i}`, `doc_${i}`, `decoy_${i}`);
    }
    engine.storeMemory('The raretarget stability threshold is 4.2 Kelvin.', 'rare', 'rare_chunk');
    const selected = engine.packContext('quantumentanglement superconductor decoherence raretarget', {budget: 150});
    assert.ok(selected.includes('rare_chunk'));
  } finally {
    engine.db.close();
  }
});
