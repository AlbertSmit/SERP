import assert from 'node:assert/strict';
import test from 'node:test';
import { SQLiteContextEngine } from '../src/engine/engine.js';
import { gramTokens } from '../src/engine/tokenize.js';

test('gramTokens marks short and long words without punctuation', () => {
  assert.deepEqual(gramTokens('ok'), ['1ok2']);
  assert.deepEqual(gramTokens('sqlite'), ['1sql', 'sqli', 'qlit', 'lite', 'ite2']);
  assert.deepEqual(gramTokens('SQLite cpp'), ['1sql', 'sqli', 'qlit', 'lite', 'ite2', '1cpp', 'cpp2']);
});

test('buildGramIndex backfills from the word index and is idempotent', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('PostgreSQL replication handbook', 'db', 'db');
    engine.storeMemory('sqlite lexical retrieval', 'sql', 'sql');
    engine.db.prepare('DELETE FROM chunks_grams').run();
    assert.equal(engine.buildGramIndex({batchSize: 1}), 2);
    assert.equal(engine.buildGramIndex(), 0);
    assert.deepEqual(engine.rankGramBm25('postgresql').map((x) => x.chunk_id), ['db']);
  } finally {
    engine.db.close();
  }
});

test('rankWordBm25 ranks substantive terms and excludes full documents', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('sqlite lexical context', 'doc', 'chunk');
    engine.storeMemory('sqlite full documents', 'doc', 'doc_full');
    engine.storeMemory('unrelated material', 'other', 'other');
    const ranked = engine.rankWordBm25('the sqlite');
    assert.deepEqual(ranked.map((x) => x.chunk_id), ['chunk']);
    assert.ok(ranked[0].score > 0);
    assert.deepEqual(engine.rankWordBm25('the and de het'), []);
  } finally {
    engine.db.close();
  }
});

test('rankGramBm25 recovers in-word variants and gates unknown identifiers', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('PostgreSQL replication handbook', 'db', 'db');
    engine.storeMemory('error_42 failure recovery', 'errors', 'exact');
    engine.storeMemory('error_43 failure recovery', 'errors', 'variant');
    assert.deepEqual(engine.rankGramBm25('postgresql').map((x) => x.chunk_id), ['db']);
    assert.ok(engine.rankGramBm25('error_42').length > 0);
    assert.deepEqual(engine.rankGramBm25('error_99'), []);
    assert.deepEqual(engine.rankGramBm25('zzq_42'), []);
    assert.ok(engine.rankGramBm25('postgresql error_42').length > 0);
    engine.loadStats();
    assert.equal(engine.N_i.has('error_42'), false);
    assert.ok(engine.rankGramBm25('error_42').length > 0);
    assert.deepEqual(engine.rankGramBm25('error_99'), []);
    assert.deepEqual(engine.rankGramBm25('zzq_42'), []);
  } finally {
    engine.db.close();
  }
});

test('fuseRrf replaces rank order with exact-first order and is deterministic', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('error_42 failure recovery', 'errors', 'exact');
    engine.storeMemory('error_43 failure recovery', 'errors', 'variant');
    const lists = [[
      {chunk_id: 'variant', score: 9},
      {chunk_id: 'exact', score: 1},
    ]];
    const fused = engine.fuseRrf(lists, {queryText: 'error_42'});
    assert.deepEqual(fused.map((x) => [x.chunk_id, x.exact]), [
      ['exact', true],
      ['variant', false],
    ]);
    assert.ok(fused[0].score < fused[1].score);
    const tied = engine.fuseRrf([
      [{chunk_id: 'b', score: 1}],
      [{chunk_id: 'a', score: 1}],
    ]);
    assert.deepEqual(tied.map((x) => x.chunk_id), ['a', 'b']);
  } finally {
    engine.db.close();
  }
});

test('packRanked respects the budget and reports coverage-gated skips', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('alpha one two three', 'd', 'one');
    engine.storeMemory('alpha alpha alpha alpha', 'd', 'two');
    engine.storeMemory('alpha one two three four five six seven eight nine', 'd', 'big');
    engine.storeMemory('unrelated material', 'd', 'unrelated');
    const fused = [
      {chunk_id: 'big', score: 4},
      {chunk_id: 'unrelated', score: 3},
      {chunk_id: 'one', score: 2},
      {chunk_id: 'two', score: 1},
    ];
    const detailed = engine.packRankedDetailed('alpha', fused, {budget: 5});
    assert.deepEqual(detailed.chunkIds, ['one']);
    assert.deepEqual(detailed.skippedCoverage, ['unrelated']);
    assert.ok(detailed.usedTokens <= detailed.budget);
    assert.deepEqual(engine.packRanked('alpha', fused, {budget: 5}), ['one']);
  } finally {
    engine.db.close();
  }
});

test('every packer stays within budget and only returns fused candidates', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('alpha alpha beta beta', 'd', 'alpha_beta');
    engine.storeMemory('beta beta beta', 'd', 'beta');
    engine.storeMemory('alpha gamma delta', 'd', 'alpha_gamma');
    const fused = [
      {chunk_id: 'alpha_beta', score: 3, exact: true},
      {chunk_id: 'beta', score: 2, exact: true},
      {chunk_id: 'alpha_gamma', score: 1, exact: true},
    ];
    const allowed = new Set(fused.map((x) => x.chunk_id));
    for (const packer of ['truncate', 'greedy', 'residual', 'p4']) {
      const selected = engine.packRanked('alpha beta', fused, {budget: 6, packer});
      const used = selected.reduce((total, id) => total + engine.docStats.get(id).L, 0);
      assert.ok(used <= 6, packer);
      assert.ok(selected.every((id) => allowed.has(id)), packer);
    }
  } finally {
    engine.db.close();
  }
});

test('retrieveRanked exposes channels and packHybrid composes the stages', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('sqlite lexical context', 'doc', 'chunk');
    engine.storeMemory('sqlite retrieval tokens', 'doc', 'tokens');
    engine.storeMemory('sqlite full documents', 'doc', 'doc_full');
    const {word, gram, fused} = engine.retrieveRanked('sqlite');
    assert.ok(word.some((x) => x.chunk_id === 'chunk'));
    assert.ok(word.every((x) => !x.chunk_id.endsWith('_full')));
    assert.ok(gram.length > 0);
    assert.ok(fused.every((x) => !x.chunk_id.endsWith('_full')));
    assert.deepEqual(engine.retrieveRanked('sqlite', {channels: ['word']}).gram, []);
    const packed = engine.packHybrid('sqlite', {budget: 5});
    assert.deepEqual(packed, engine.packRanked('sqlite', fused, {budget: 5}));
    const used = packed.reduce((total, id) => total + engine.docStats.get(id).L, 0);
    assert.ok(used <= 5);
    assert.ok(packed.every((id) => !id.endsWith('_full')));
  } finally {
    engine.db.close();
  }
});

test('buildExactIndex backfills exact tokens from the word index and is idempotent', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('PostgreSQL replication handbook', 'db', 'db');
    engine.storeMemory('failure error_42 recovery', 'errors', 'err');
    engine.storeMemory('only_in_full_99 marker', 'full', 'doc_full');
    assert.equal(engine.identifierHasExactMatch('error_42'), true);
    assert.equal(engine.identifierHasExactMatch('figure_1'), false);
    engine.db.prepare('DELETE FROM chunks_exact_terms').run();
    assert.equal(engine.identifierHasExactMatch('error_42'), false);
    assert.equal(engine.buildExactIndex({batchSize: 1}), 3);
    assert.equal(engine.buildExactIndex(), 0);
    assert.equal(engine.identifierHasExactMatch('error_42'), true);
    assert.deepEqual(engine.exactIdentifierTerms().map((x) => x.term), ['error_42', 'only_in_full_99']);
    assert.deepEqual(
      engine.exactIdentifierTerms({retrievableOnly: true}).map((x) => x.term),
      ['error_42'],
    );
  } finally {
    engine.db.close();
  }
});

test('rankGramBm25 filters unknown identifiers per token instead of all-or-nothing', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('p53 pathway regulation', 'bio', 'p53');
    engine.storeMemory('zzq_43 variant material', 'noise', 'variant');
    const mixed = engine.rankGramBm25('zzq_42 p53').map((x) => x.chunk_id);
    assert.ok(mixed.includes('p53'));
    assert.ok(!mixed.includes('variant'));
    assert.deepEqual(engine.rankGramBm25('zzq_42'), []);
    assert.deepEqual(engine.rankGramBm25('zzq_42 zzq_43x'), []);
  } finally {
    engine.db.close();
  }
});

test('rankGramBm25 does not treat prose "figure 1" as the identifier figure_1', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('see figure 1 for details', 'prose', 'prose');
    assert.equal(engine.identifierHasExactMatch('figure_1'), false);
    assert.deepEqual(engine.rankGramBm25('figure_1'), []);
    engine.storeMemory('figure_1 labels the diagram', 'real', 'real');
    assert.equal(engine.identifierHasExactMatch('figure_1'), true);
    assert.ok(engine.rankGramBm25('figure_1').length > 0);
  } finally {
    engine.db.close();
  }
});

test('retrieveRanked drops unknown identifiers from the word channel too', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('see figure 1 for details', 'prose', 'prose');
    engine.storeMemory('p53 pathway regulation', 'bio', 'p53');
    engine.storeMemory('error_42 failure recovery', 'errors', 'error');
    assert.deepEqual(engine.retrieveRanked('figure_1').fused, []);
    const mixed = engine.retrieveRanked('figure_1 p53');
    assert.ok(mixed.word.some((x) => x.chunk_id === 'p53'));
    assert.ok(mixed.fused.some((x) => x.chunk_id === 'p53'));
    assert.ok(mixed.fused.every((x) => !engine.chunkTokenCounts(x.chunk_id).has('figure_1')));
    const exact = engine.retrieveRanked('error_42');
    assert.equal(exact.fused[0].chunk_id, 'error');
    assert.equal(exact.fused[0].exact, true);
  } finally {
    engine.db.close();
  }
});

test('fuseRrf only marks identifier tokens with an exact term as exact', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('p53 brca1 pathway', 'bio', 'known');
    engine.storeMemory('2020 results only', 'year', 'year');
    engine.storeMemory('error_42 failure recovery', 'errors', 'error');
    const lists = [[
      {chunk_id: 'year', score: 3},
      {chunk_id: 'known', score: 2},
      {chunk_id: 'error', score: 1},
    ]];
    const fused = engine.fuseRrf(lists, {queryText: 'p53 brca1 2020'});
    const flags = Object.fromEntries(fused.map((x) => [x.chunk_id, x.exact]));
    assert.equal(flags.known, true);
    assert.equal(flags.year, false);
    const errorFused = engine.fuseRrf(lists, {queryText: 'error_42'});
    assert.equal(errorFused[0].chunk_id, 'error');
    assert.equal(errorFused[0].exact, true);
  } finally {
    engine.db.close();
  }
});

test('packRankedDetailed gives every packer the same pool depth', () => {
  const engine = new SQLiteContextEngine(':memory:');
  try {
    engine.storeMemory('alpha alpha beta beta', 'd', 'a');
    engine.storeMemory('alpha beta gamma', 'd', 'b');
    engine.storeMemory('alpha beta delta', 'd', 'c');
    const fused = [
      {chunk_id: 'c', score: 3, exact: true},
      {chunk_id: 'b', score: 2, exact: true},
      {chunk_id: 'a', score: 1, exact: true},
    ];
    for (const packer of ['truncate', 'greedy', 'residual', 'p4']) {
      const selected = engine.packRanked('alpha beta', fused, {budget: 100, packer, poolDepth: 2});
      assert.ok(selected.every((id) => id !== 'a'), packer);
    }
  } finally {
    engine.db.close();
  }
});
