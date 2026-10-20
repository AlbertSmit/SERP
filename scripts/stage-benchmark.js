#!/usr/bin/env node
/**
 * stage-benchmark.js — chunk-level retrieval + packing comparison for SERP.
 *
 * Build the corpus first:
 *   python3 python/build_mcp_db.py
 *
 * Then:
 *   node scripts/stage-benchmark.js
 *
 * This is a regression/sanity benchmark: SciFact test labels are document-level,
 * so a chunk is "relevant" when its parent document is relevant. It measures
 * whether packing preserves relevant documents, not sentence-level precision.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SQLiteContextEngine } from '../src/engine/engine.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const DB = process.env.SERP_DB || path.join(ROOT, 'mcp_memory.sqlite');
const DATA = process.env.SCIFACT_DIR || path.join(ROOT, 'datasets/scifact');
const BUDGETS = [120, 250, 500];
const DEPTH = 100;

function readJsonl(file) {
  return fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
}

function readQrels(file) {
  const qrels = new Map();
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim() || line.startsWith('query-id')) continue;
    const [qid, docId, score] = line.split('\t');
    if (Number(score) <= 0) continue;
    if (!qrels.has(qid)) qrels.set(qid, new Set());
    qrels.get(qid).add(String(docId));
  }
  return qrels;
}

function relevantChunks(engine, docs) {
  const chunks = new Set();
  for (const [chunkId, docId] of engine.chunkDocId) {
    if (!docId) continue;
    if (docs.has(docId) && !chunkId.endsWith('_full')) chunks.add(chunkId);
  }
  return chunks;
}

function select(engine, query, budget, system) {
  const options = { budget, channels: ['word', 'gram'] };
  switch (system) {
    case 'word_truncate':
      return engine.packHybrid(query, { ...options, channels: ['word'], packer: 'truncate' });
    case 'hybrid_truncate':
      return engine.packHybrid(query, { ...options, packer: 'truncate' });
    case 'hybrid_greedy':
      return engine.packHybrid(query, { ...options, packer: 'greedy' });
    case 'hybrid_residual':
      return engine.packHybrid(query, { ...options, packer: 'residual' });
    case 'hybrid_p4':
      return engine.packHybrid(query, { ...options, packer: 'p4' });
    case 'p4_current':
      return engine.packContext(query, { budget });
    default:
      throw new Error(`unknown system: ${system}`);
  }
}

function evaluate(engine, queries, qrels, budget) {
  const systems = [
    'word_truncate', 'hybrid_truncate', 'hybrid_greedy',
    'hybrid_residual', 'hybrid_p4', 'p4_current',
  ];
  const rows = Object.fromEntries(systems.map(s => [s, {
    chunkRecall: 0, docRecall: 0, tokens: 0, selected: 0, ms: 0, queries: 0,
  }]));

  for (const query of queries) {
    const relDocs = qrels.get(String(query._id));
    if (!relDocs || relDocs.size === 0) continue;
    const relChunks = relevantChunks(engine, relDocs);
    if (relChunks.size === 0) continue;

    for (const system of systems) {
      const start = performance.now();
      const selected = select(engine, query.text, budget, system);
      const elapsed = performance.now() - start;
      const row = rows[system];
      const selectedDocs = new Set();
      let chunkHits = 0;
      let tokens = 0;
      for (const cid of selected) {
        const meta = engine.docStats.get(cid);
        if (!meta) continue;
        tokens += meta.L;
        const docId = engine.chunkDocId.get(cid);
        if (docId) selectedDocs.add(docId);
        if (relChunks.has(cid)) chunkHits += 1;
      }
      row.chunkRecall += chunkHits / relChunks.size;
      let docHits = 0;
      for (const docId of relDocs) if (selectedDocs.has(docId)) docHits += 1;
      row.docRecall += docHits / relDocs.size;
      row.tokens += tokens;
      row.selected += selected.length;
      row.ms += elapsed;
      row.queries += 1;
    }
  }

  for (const row of Object.values(rows)) {
    for (const key of ['chunkRecall', 'docRecall', 'tokens', 'selected', 'ms']) {
      row[key] = row.queries ? row[key] / row.queries : 0;
    }
  }
  return rows;
}

function exactIdentifierSuite(engine) {
  const position = new Map();
  for (const [chunkId, meta] of engine.docStats) position.set(chunkId, meta);
  const identifiers = [...engine.N_i.entries()]
    .filter(([term, df]) => /[0-9]/.test(term) && term.length >= 4 && df >= 1 && df <= 20)
    .slice(0, 100);

  let exactTop1 = 0;
  let exactTop10 = 0;
  let nearMissAbove = 0;
  let positives = 0;
  for (const [term] of identifiers) {
    const result = engine.retrieveRanked(term, { channels: ['word', 'gram'], depth: DEPTH });
    const ranked = result.fused.map(x => x.chunk_id);
    const exactPosition = ranked.findIndex(cid => engine.chunkTokenCounts(cid).has(term));
    if (exactPosition < 0) continue;
    positives += 1;
    if (exactPosition === 0) exactTop1 += 1;
    if (exactPosition < 10) exactTop10 += 1;
    const near = ranked.findIndex((cid, index) => index < exactPosition
      && [...engine.chunkTokenCounts(cid).keys()].some(w => w !== term && w.startsWith(term.slice(0, 3))));
    if (near >= 0) nearMissAbove += 1;
  }

  let negativeActivation = 0;
  const negatives = Array.from({ length: 25 }, (_, i) => `zzq${String(100000 + i)}`);
  for (const term of negatives) {
    const result = engine.retrieveRanked(term, { channels: ['word', 'gram'], depth: 10 });
    if (result.fused.length > 0) negativeActivation += 1;
  }

  return {
    positives,
    exactTop1: positives ? exactTop1 / positives : 0,
    exactTop10: positives ? exactTop10 / positives : 0,
    nearMissAbove: positives ? nearMissAbove / positives : 0,
    negatives: negatives.length,
    negativeActivation: negativeActivation / negatives.length,
  };
}

function main() {
  if (!fs.existsSync(DB)) throw new Error(`missing database ${DB}; run python3 python/build_mcp_db.py`);

  const engine = new SQLiteContextEngine(DB);
  try {
    const indexed = engine.buildGramIndex();
    console.log(`[grams] ${indexed} rows indexed`);

    const queries = readJsonl(path.join(DATA, 'queries.jsonl'));
    const qrels = readQrels(path.join(DATA, 'qrels/test.tsv'));
    const report = { database: DB, budgets: {}, exactIdentifiers: exactIdentifierSuite(engine) };

    for (const budget of BUDGETS) {
      console.log(`\n== budget ${budget}`);
      const rows = evaluate(engine, queries, qrels, budget);
      report.budgets[budget] = rows;
      console.log(`${'system'.padEnd(20)} ${'chunkR'.padStart(8)} ${'docR'.padStart(8)} ${'tokens'.padStart(8)} ${'picks'.padStart(6)} ${'ms'.padStart(8)}`);
      for (const [name, row] of Object.entries(rows)) {
        console.log(`${name.padEnd(20)} ${row.chunkRecall.toFixed(3).padStart(8)} ${row.docRecall.toFixed(3).padStart(8)} ${row.tokens.toFixed(1).padStart(8)} ${row.selected.toFixed(2).padStart(6)} ${row.ms.toFixed(2).padStart(8)}`);
      }
    }

    console.log('\n== exact identifiers');
    console.log(JSON.stringify(report.exactIdentifiers, null, 2));
    const out = path.join(ROOT, 'experiments/stages/results/stage-benchmark.json');
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, JSON.stringify(report, null, 1));
    console.log(`[saved] ${out}`);
  } finally {
    engine.db.close();
  }
}

main();
