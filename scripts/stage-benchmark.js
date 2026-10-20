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
 *
 * The exact-identifier suite uses chunks_exact_terms (tokenize() output with
 * underscores preserved), not unicode61 phrase matching: "figure_1" is known
 * only when it was indexed as a token.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SQLiteContextEngine } from '../src/engine/engine.js';
import { tokenize } from '../src/engine/tokenize.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const DB = process.env.SERP_DB || path.join(ROOT, 'mcp_memory.sqlite');
const DATA = process.env.SCIFACT_DIR || path.join(ROOT, 'datasets/scifact');
const BUDGETS = [120, 250, 500];
const DEPTH = 100;
const POSITIVE_LIMIT = 100;
const POSITIVE_DEPTH = 100;
const NEGATIVE_COUNT = 25;
const SYSTEM_SPECS = {
  word_truncate: {channels: ['word'], packer: 'truncate'},
  word_truncate_nogate: {channels: ['word'], packer: 'truncate', min_coverage: 0},
  hybrid_truncate: {channels: ['word', 'gram'], packer: 'truncate'},
  hybrid_truncate_nogate: {channels: ['word', 'gram'], packer: 'truncate', min_coverage: 0},
  hybrid_greedy: {channels: ['word', 'gram'], packer: 'greedy'},
  hybrid_residual: {channels: ['word', 'gram'], packer: 'residual'},
  hybrid_p4: {channels: ['word', 'gram'], packer: 'p4'},
  p4_current: {legacy: true},
  hybrid_truncate_matched_residual: {channels: ['word', 'gram'], packer: 'truncate'},
  hybrid_truncate_matched_p4_current: {channels: ['word', 'gram'], packer: 'truncate'},
};
const SYSTEMS = Object.keys(SYSTEM_SPECS);

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
  const spec = SYSTEM_SPECS[system];
  if (spec.legacy) {
    const chunkIds = engine.packContext(query, {budget});
    return {chunkIds, usedTokens: usedTokens(engine, chunkIds)};
  }
  const {fused} = engine.retrieveRanked(query, {channels: spec.channels, depth: DEPTH});
  const options = {budget, packer: spec.packer};
  if (spec.min_coverage !== undefined) options.min_coverage = spec.min_coverage;
  const detailed = engine.packRankedDetailed(query, fused, options);

  return {chunkIds: detailed.chunkIds, usedTokens: detailed.usedTokens};
}

function usedTokens(engine, chunkIds) {
  return chunkIds.reduce((total, cid) => total + (engine.docStats.get(cid)?.L || 0), 0);
}

function evaluate(engine, queries, qrels, budget) {
  const rows = Object.fromEntries(SYSTEMS.map(s => [s, {
    chunkRecall: 0, docRecall: 0, chunkPrecision: 0, docPrecision: 0,
    coverage: 0, tokens: 0, selected: 0, ms: 0, queries: 0,
  }]));
  const cache = new Map();
  function selectCached(queryText, system, at) {
    const key = `${system}\u0000${at}\u0000${queryText}`;
    if (!cache.has(key)) {
      const start = performance.now();
      const selection = select(engine, queryText, at, system);
      selection.ms = performance.now() - start;
      cache.set(key, selection);
    }

    return cache.get(key);
  }

  for (const query of queries) {
    const relDocs = qrels.get(String(query._id));
    if (!relDocs || relDocs.size === 0) continue;
    const relChunks = relevantChunks(engine, relDocs);
    if (relChunks.size === 0) continue;
    const queryTokens = new Set(tokenize(query.text));
    const matchedBudget = {
      hybrid_truncate_matched_residual:
        selectCached(query.text, 'hybrid_residual', budget).usedTokens,
      hybrid_truncate_matched_p4_current:
        selectCached(query.text, 'p4_current', budget).usedTokens,
    };

    for (const system of SYSTEMS) {
      const at = matchedBudget[system] ?? budget;
      accumulate(rows[system], engine, selectCached(query.text, system, at),
        {relChunks, relDocs, queryTokens});
    }
  }

  for (const row of Object.values(rows)) {
    for (const key of ['chunkRecall', 'docRecall', 'chunkPrecision', 'docPrecision',
      'coverage', 'tokens', 'selected', 'ms']) {
      row[key] = row.queries ? row[key] / row.queries : 0;
    }
  }
  return rows;
}

function accumulate(row, engine, selection, context) {
  const {relChunks, relDocs, queryTokens} = context;
  const stats = selectionStats(engine, selection.chunkIds, relChunks, queryTokens);
  row.chunkRecall += stats.chunkHits / relChunks.size;
  let docHits = 0;
  for (const docId of relDocs) if (stats.selectedDocs.has(docId)) docHits += 1;
  row.docRecall += docHits / relDocs.size;
  row.chunkPrecision += selection.chunkIds.length
    ? stats.chunkHits / selection.chunkIds.length : 0;
  row.docPrecision += stats.selectedDocs.size
    ? docHits / stats.selectedDocs.size : 0;
  row.coverage += queryTokens.size ? stats.coveredTokens.size / queryTokens.size : 0;
  row.tokens += stats.tokens;
  row.selected += selection.chunkIds.length;
  row.ms += selection.ms;
  row.queries += 1;
}

function selectionStats(engine, chunkIds, relChunks, queryTokens) {
  const selectedDocs = new Set();
  const coveredTokens = new Set();
  let chunkHits = 0;
  let tokens = 0;
  for (const cid of chunkIds) {
    const meta = engine.docStats.get(cid);
    if (!meta) continue;
    tokens += meta.L;
    const docId = engine.chunkDocId.get(cid);
    if (docId) selectedDocs.add(docId);
    if (relChunks.has(cid)) chunkHits += 1;
    collectCoveredTokens(engine, cid, queryTokens, coveredTokens);
  }

  return {selectedDocs, coveredTokens, chunkHits, tokens};
}

function collectCoveredTokens(engine, cid, queryTokens, coveredTokens) {
  for (const token of engine.chunkTokenCounts(cid).keys()) {
    if (queryTokens.has(token)) coveredTokens.add(token);
  }
}

// ---------------------------------------------------------------------------
// Exact-identifier suite
// ---------------------------------------------------------------------------

function exactIdentifierSuite(engine) {
  // Retrieval excludes `_full` chunks, so a positive must be indexed in a chunk
  // that retrieval can return; otherwise it is a miss by construction.
  const allTerms = engine.exactIdentifierTerms({maxDf: 1e9, retrievableOnly: true});
  const termSet = new Set(allTerms.map(({term}) => term));
  const positives = [...engine.exactIdentifierTerms({maxDf: 20, retrievableOnly: true})]
    .sort((left, right) => right.df - left.df || compareTerms(left.term, right.term))
    .slice(0, POSITIVE_LIMIT);
  const nearMissIndex = buildNearMissIndex(termSet);
  const stats = {
    positives: positives.length,
    exactFound: 0,
    exactTop1: 0,
    exactTop10: 0,
    rankSum: 0,
    nearMissPositives: 0,
    nearMissEvaluated: 0,
    nearMissAbove: 0,
  };
  for (const {term} of positives) {
    const {fused} = engine.retrieveRanked(term, {channels: ['word', 'gram'], depth: DEPTH});
    const ranked = fused.slice(0, POSITIVE_DEPTH).map((entry) => entry.chunk_id);
    const position = ranked.findIndex((cid) => engine.chunkTokenCounts(cid).has(term));
    const variants = nearMissIndex.get(term);
    if (variants) stats.nearMissPositives += 1;
    if (position < 0) continue;
    stats.exactFound += 1;
    stats.rankSum += position + 1;
    if (position === 0) stats.exactTop1 += 1;
    if (position < 10) stats.exactTop10 += 1;
    if (!variants) continue;
    stats.nearMissEvaluated += 1;
    const above = ranked.slice(0, position).some(
      (cid) => variants.some((variant) => engine.chunkTokenCounts(cid).has(variant)),
    );
    if (above) stats.nearMissAbove += 1;
  }
  const total = positives.length;

  return {
    positives: total,
    exactFound: rate(stats.exactFound, total),
    exactTop1: rate(stats.exactTop1, total),
    exactTop10: rate(stats.exactTop10, total),
    misses: total - stats.exactFound,
    meanExactRank: stats.exactFound ? stats.rankSum / stats.exactFound : 0,
    nearMissAbove: rate(stats.nearMissAbove, stats.nearMissPositives),
    nearMissPositives: stats.nearMissPositives,
    nearMissEvaluated: stats.nearMissEvaluated,
    negatives: negativeSuite(engine, positives),
  };
}

function negativeSuite(engine, positives) {
  const knownTerm = positives.find(({term}) => engine.identifierHasExactMatch(term))?.term || 'p53';
  const prose = ['figure_1', 'table_2', 'stage_1', 'type_2', 'phase_2', 'factor_1', 'day_1']
    .filter((term) => !engine.identifierHasExactMatch(term));
  const groups = {
    single: Array.from({length: NEGATIVE_COUNT}, (_, i) => `zzq${100000 + i}`),
    underscore: Array.from({length: NEGATIVE_COUNT}, (_, i) => `zzq_${100000 + i}`),
    mixed: Array.from({length: NEGATIVE_COUNT}, (_, i) => `zzq_${100000 + i} ${knownTerm}`),
    prose,
  };
  const report = {known_term: knownTerm, prose_terms: prose};
  for (const [name, terms] of Object.entries(groups)) {
    report[name] = groupActivation(engine, terms);
  }

  return report;
}

function groupActivation(engine, terms) {
  let activated = 0;
  let nonEmpty = 0;
  for (const term of terms) {
    const outcome = negativeOutcome(engine, term);
    if (outcome.activated) activated += 1;
    if (outcome.nonEmpty) nonEmpty += 1;
  }

  return {
    queries: terms.length,
    activation: rate(activated, terms.length),
    non_empty: rate(nonEmpty, terms.length),
  };
}

function negativeOutcome(engine, query) {
  const tokens = [...new Set(tokenize(query))];
  const known = tokens.filter((token) => !engine.isUnknownIdentifier(token));
  const full = engine.retrieveRanked(query, {channels: ['word', 'gram'], depth: 10}).fused;
  if (known.length === 0) {
    return {activated: full.length > 0, nonEmpty: full.length > 0};
  }
  const baseIds = new Set(
    engine.retrieveRanked(known.join(' '), {channels: ['word', 'gram'], depth: 10})
      .fused.map((entry) => entry.chunk_id),
  );
  const extra = full.filter((entry) => !baseIds.has(entry.chunk_id));

  return {activated: extra.length > 0, nonEmpty: full.length > 0};
}

function buildNearMissIndex(terms) {
  const byPrefix = new Map();
  for (const term of terms) {
    const prefix = term.slice(0, 3);
    if (!byPrefix.has(prefix)) byPrefix.set(prefix, []);
    byPrefix.get(prefix).push(term);
  }
  const index = new Map();
  for (const term of terms) {
    const variants = byPrefix.get(term.slice(0, 3))
      .filter((candidate) => candidate !== term && editDistanceAtMostOne(term, candidate));
    if (variants.length > 0) index.set(term, variants);
  }

  return index;
}

function editDistanceAtMostOne(left, right) {
  if (left === right) return false;
  if (Math.abs(left.length - right.length) > 1) return false;
  if (left.length === right.length) return singleSubstitution(left, right);

  return singleInsertion(left, right);
}

function singleSubstitution(left, right) {
  let differences = 0;
  for (let i = 0; i < left.length; i++) {
    if (left[i] === right[i]) continue;
    differences += 1;
    if (differences > 1) return false;
  }

  return differences === 1;
}

function singleInsertion(left, right) {
  const [longer, shorter] = left.length > right.length ? [left, right] : [right, left];
  let i = 0;
  let j = 0;
  let skipped = false;
  while (i < longer.length && j < shorter.length) {
    if (longer[i] === shorter[j]) {
      i += 1;
      j += 1;
      continue;
    }
    if (skipped) return false;
    skipped = true;
    i += 1;
  }

  return true;
}

function rate(count, total) {
  return total ? count / total : 0;
}

function compareTerms(left, right) {
  if (left === right) return 0;

  return left < right ? -1 : 1;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

function printBudget(rows) {
  const header = `${'system'.padEnd(38)} ${'chunkR'.padStart(8)} ${'chunkP'.padStart(8)} ${'docR'.padStart(8)} ${'docP'.padStart(8)} ${'cover'.padStart(8)} ${'tokens'.padStart(8)} ${'picks'.padStart(6)} ${'ms'.padStart(8)}`;
  console.log(header);
  for (const [name, row] of Object.entries(rows)) {
    console.log(`${name.padEnd(38)} ${row.chunkRecall.toFixed(3).padStart(8)} ${row.chunkPrecision.toFixed(3).padStart(8)} ${row.docRecall.toFixed(3).padStart(8)} ${row.docPrecision.toFixed(3).padStart(8)} ${row.coverage.toFixed(3).padStart(8)} ${row.tokens.toFixed(1).padStart(8)} ${row.selected.toFixed(2).padStart(6)} ${row.ms.toFixed(2).padStart(8)}`);
  }
}

function printExact(suite) {
  console.log(`positives ${suite.positives}  found ${suite.exactFound.toFixed(3)}  top1 ${suite.exactTop1.toFixed(3)}  top10 ${suite.exactTop10.toFixed(3)}  misses ${suite.misses}  meanRank ${suite.meanExactRank.toFixed(2)}`);
  console.log(`near-miss above: ${suite.nearMissAbove.toFixed(3)} (${suite.nearMissEvaluated}/${suite.nearMissPositives} positives with variants evaluated)`);
  console.log(`negatives (known term ${suite.negatives.known_term}):`);
  for (const [name, group] of Object.entries(suite.negatives)) {
    if (name === 'known_term' || name === 'prose_terms') continue;
    console.log(`  ${name.padEnd(12)} n=${String(group.queries).padStart(3)}  activation ${group.activation.toFixed(3)}  non-empty ${group.non_empty.toFixed(3)}`);
  }
}

function main() {
  if (!fs.existsSync(DB)) throw new Error(`missing database ${DB}; run python3 python/build_mcp_db.py`);

  const engine = new SQLiteContextEngine(DB);
  try {
    const grams = engine.buildGramIndex();
    console.log(`[grams] ${grams} rows indexed`);
    const exact = engine.buildExactIndex();
    console.log(`[exact] ${exact} chunks indexed`);

    const queries = readJsonl(path.join(DATA, 'queries.jsonl'));
    const qrels = readQrels(path.join(DATA, 'qrels/test.tsv'));
    const report = { database: DB, budgets: {} };

    for (const budget of BUDGETS) {
      console.log(`\n== budget ${budget}`);
      const rows = evaluate(engine, queries, qrels, budget);
      report.budgets[budget] = rows;
      printBudget(rows);
    }

    report.exactIdentifiers = exactIdentifierSuite(engine);
    console.log('\n== exact identifiers');
    printExact(report.exactIdentifiers);
    const out = path.join(ROOT, 'experiments/stages/results/stage-benchmark.json');
    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, JSON.stringify(report, null, 1));
    console.log(`[saved] ${out}`);
  } finally {
    engine.db.close();
  }
}

main();
