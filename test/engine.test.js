import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { SQLiteContextEngine } from '../src/engine/engine.js';
import { isSubstantiveToken, tokenize } from '../src/engine/tokenize.js';

// Captured from the SubConPack 2.0 engine after the intentional ranking-rule update.
const baseline = JSON.parse(fs.readFileSync(new URL('./fixtures/engine-baseline.json', import.meta.url), 'utf8'));

test('selection order, budgets, temporal decay and reinforcement match the v2 snapshot', () => {
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

test('v2 information weights include unmatched substantive demand and stay positive', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('shared alpha', 'a', 'a');
    engine.storeMemory('shared beta beta beta beta beta', 'b', 'b');
    const focused = engine.scoreChunkP4('a', 'alpha');
    const compound = engine.scoreChunkP4('a', 'alpha neverindexed');
    assert.ok(compound < focused);
    assert.ok(engine.informationWeight('shared') > 0);
    assert.equal(engine.informationWeight('neverindexed'), Math.log1p(engine.M_ref / 2));
    assert.equal(engine.meanChunkLength, 4);
  } finally {
    engine.db.close();
  }
});

test('bilingual stopwords are excluded and small vaults retain common terms', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    for (let i = 0; i < 8; i++) engine.storeMemory(`common signal ${i}`, `doc_${i}`, `chunk_${i}`);
    assert.deepEqual([...engine.queryWeights('the and de het signal').keys()], ['signal']);
    assert.equal(isSubstantiveToken('voor'), false);
    assert.ok(engine.contextCandidates('common', []).length > 0);
  } finally {
    engine.db.close();
  }
});

test('prefix, stem and compound matches receive the discount while headers receive a boost', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('# Authorization\nauthorizationHandler validates tokens', 'auth', 'auth');
    const counts = engine.chunkTokenCounts('auth');
    const headers = engine.chunkHeaderTokenCounts('auth');
    assert.equal(engine.termMatch('authorization', counts, headers).count, 1.4);
    assert.equal(engine.termMatch('authoriz', counts, headers).count, 1.19);
    assert.equal(engine.termMatch('validate', counts, headers).count, 0.85);
    assert.equal(engine.termMatch('handler', counts, headers).count, 0.85);
  } finally {
    engine.db.close();
  }
});

test('small-vault trigram fallback retrieves close typos and abstains on unrelated text', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('PostgreSQL replication handbook', 'db', 'db');
    assert.deepEqual(engine.packContext('postgressql', {min_coverage: 0.9}), ['db']);
    assert.deepEqual(engine.packContext('xylophone', {min_coverage: 0}), []);
  } finally {
    engine.db.close();
  }
});

test('coverage gate admits shaped rare identifiers but not arbitrary unknown words', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('failure error_42 recovery', 'errors', 'error');
    engine.storeMemory('ordinary alpha text', 'ordinary', 'ordinary');
    assert.deepEqual(engine.packContext('error_42 missingone missingtwo', {min_coverage: 0.95}), ['error']);
    assert.deepEqual(engine.packContext('alpha missingone missingtwo', {min_coverage: 0.95}), []);
  } finally {
    engine.db.close();
  }
});

test('packer rejects zero-gain candidates and applies residual and relative dropoff stops', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('# Alpha\nalpha alpha alpha alpha', 'strong', 'strong');
    engine.storeMemory('alpha', 'weak', 'weak');
    engine.storeMemory('unrelated material', 'other', 'other');
    assert.deepEqual(engine.packContext('missing', {
      bridgeCandidates: ['other'], min_coverage: 0, rel_dropoff: 0,
    }), []);
    assert.equal(engine.packContext('alpha', {
      lambda_decay: 0, rel_dropoff: 0, min_coverage: 0,
    }).length, 2);
    assert.equal(engine.packContext('alpha', {
      lambda_decay: 0, rel_dropoff: 0.99, min_coverage: 0,
    }).length, 1);
    assert.equal(engine.packContext('alpha', {
      lambda_decay: 10, rel_dropoff: 0, min_coverage: 0,
    }).length, 1);
  } finally {
    engine.db.close();
  }
});
