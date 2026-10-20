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
const PER_QUERY_METRICS = [
  'chunkRecall', 'docRecall', 'chunkPrecision', 'docPrecision', 'tokens', 'selected',
];
// The paired comparisons the packing decision rests on. The matched rows are
// included because p4_current underfills the budget: comparing truncation at
// 120/250/500 tokens with P4 at ~46 tokens is not a packing comparison.
const PAIRED_COMPARISONS = [
  {a: 'hybrid_truncate', b: 'p4_current', metrics: ['docRecall', 'docPrecision']},
  {a: 'hybrid_truncate', b: 'hybrid_residual', metrics: ['docRecall', 'docPrecision']},
  {a: 'hybrid_truncate', b: 'hybrid_truncate_nogate', metrics: ['docRecall', 'docPrecision']},
  {a: 'hybrid_truncate', b: 'hybrid_p4', metrics: ['docRecall']},
  {a: 'hybrid_truncate_matched_p4_current', b: 'p4_current',
    metrics: ['docRecall', 'docPrecision']},
  {a: 'hybrid_truncate_matched_residual', b: 'hybrid_residual',
    metrics: ['docRecall', 'docPrecision']},
];
const BOOTSTRAP = {resamples: 1000, seed: 1729, level: 0.95};

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
  const perQuery = Object.fromEntries(SYSTEMS.map(s => [s, emptyQuerySeries()]));
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
      recordSelection({
        row: rows[system],
        series: perQuery[system],
        engine,
        selection: selectCached(query.text, system, at),
        context: {relChunks, relDocs, queryTokens},
      });
    }
  }

  for (const row of Object.values(rows)) {
    for (const key of ['chunkRecall', 'docRecall', 'chunkPrecision', 'docPrecision',
      'coverage', 'tokens', 'selected', 'ms']) {
      row[key] = row.queries ? row[key] / row.queries : 0;
    }
  }

  return {rows, perQuery};
}

function emptyQuerySeries() {
  return Object.fromEntries(PER_QUERY_METRICS.map(key => [key, []]));
}

function recordSelection({row, series, engine, selection, context}) {
  const metrics = queryMetrics(engine, selection, context);
  for (const key of [...PER_QUERY_METRICS, 'coverage']) row[key] += metrics[key];
  for (const key of PER_QUERY_METRICS) series[key].push(metrics[key]);
  row.ms += selection.ms;
  row.queries += 1;
}

function queryMetrics(engine, selection, context) {
  const {relChunks, relDocs, queryTokens} = context;
  const stats = selectionStats(engine, selection.chunkIds, relChunks, queryTokens);
  let docHits = 0;
  for (const docId of relDocs) if (stats.selectedDocs.has(docId)) docHits += 1;

  return {
    chunkRecall: stats.chunkHits / relChunks.size,
    docRecall: docHits / relDocs.size,
    chunkPrecision: selection.chunkIds.length
      ? stats.chunkHits / selection.chunkIds.length : 0,
    docPrecision: stats.selectedDocs.size
      ? docHits / stats.selectedDocs.size : 0,
    coverage: queryTokens.size ? stats.coveredTokens.size / queryTokens.size : 0,
    tokens: stats.tokens,
    selected: selection.chunkIds.length,
  };
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
// Paired bootstrap
// ---------------------------------------------------------------------------

function bootstrapComparisons(perQueryByBudget, budget) {
  const perQuery = perQueryByBudget[budget];
  const comparisons = [];
  for (const pair of PAIRED_COMPARISONS) {
    for (const metric of pair.metrics) {
      comparisons.push(comparePair({perQuery, pair, metric, budget}));
    }
  }

  return comparisons;
}

function comparePair({perQuery, pair, metric, budget}) {
  const a = perQuery[pair.a][metric];
  const b = perQuery[pair.b][metric];
  const deltas = bootstrapDeltas(a, b).sort((left, right) => left - right);
  const meanA = mean(a);
  const meanB = mean(b);
  const {wins, ties, losses} = outcomeCounts(a, b);
  const tail = (1 - BOOTSTRAP.level) / 2;
  const ciLow = quantile(deltas, tail);
  const ciHigh = quantile(deltas, 1 - tail);

  return {
    budget,
    a: pair.a,
    b: pair.b,
    metric,
    queries: a.length,
    mean_a: meanA,
    mean_b: meanB,
    delta: meanA - meanB,
    ci95: [ciLow, ciHigh],
    wins,
    ties,
    losses,
    ci_excludes_zero: ciLow > 0 || ciHigh < 0,
  };
}

function bootstrapDeltas(a, b) {
  const n = a.length;
  const random = mulberry32(BOOTSTRAP.seed);
  const deltas = [];
  for (let resample = 0; resample < BOOTSTRAP.resamples; resample++) {
    let sumA = 0;
    let sumB = 0;
    for (let i = 0; i < n; i++) {
      const index = Math.floor(random() * n);
      sumA += a[index];
      sumB += b[index];
    }
    deltas.push((sumA - sumB) / n);
  }

  return deltas;
}

// mulberry32: deterministic across platforms because Math.imul is exact 32-bit.
function mulberry32(seed) {
  let state = seed >>> 0;

  return function random() {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function outcomeCounts(a, b) {
  let wins = 0;
  let ties = 0;
  let losses = 0;
  for (let i = 0; i < a.length; i++) {
    const delta = a[i] - b[i];
    if (Math.abs(delta) < 1e-9) ties += 1;
    else if (delta > 0) wins += 1;
    else losses += 1;
  }

  return {wins, ties, losses};
}

function quantile(sorted, p) {
  if (sorted.length === 0) return 0;
  const position = (sorted.length - 1) * p;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  if (low === high) return sorted[low];

  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

function mean(values) {
  return values.length
    ? values.reduce((total, value) => total + value, 0) / values.length : 0;
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

function printPaired(comparisons) {
  const header = `${'budget'.padStart(6)} ${'comparison'.padEnd(58)} ${'metric'.padEnd(13)} ${'delta'.padStart(9)} ${'95% CI'.padStart(20)} ${'W/T/L'.padStart(12)}`;
  console.log(header);
  for (const comparison of comparisons) {
    const label = `${comparison.a} vs ${comparison.b}`;
    const ci = `[${formatSigned(comparison.ci95[0])}, ${formatSigned(comparison.ci95[1])}]`;
    const outcomes = `${comparison.wins}/${comparison.ties}/${comparison.losses}`;
    console.log(`${String(comparison.budget).padStart(6)} ${label.padEnd(58)} ${comparison.metric.padEnd(13)} ${formatSigned(comparison.delta).padStart(9)} ${ci.padStart(20)} ${outcomes.padStart(12)}`);
  }
}

function formatSigned(value) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(4)}`;
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
    const perQueryByBudget = {};

    for (const budget of BUDGETS) {
      console.log(`\n== budget ${budget}`);
      const {rows, perQuery} = evaluate(engine, queries, qrels, budget);
      report.budgets[budget] = rows;
      perQueryByBudget[budget] = perQuery;
      printBudget(rows);
    }

    const comparisons = BUDGETS.flatMap(
      (budget) => bootstrapComparisons(perQueryByBudget, budget),
    );
    report.paired = {
      method: 'paired percentile bootstrap over queries',
      resamples: BOOTSTRAP.resamples,
      seed: BOOTSTRAP.seed,
      level: BOOTSTRAP.level,
      comparisons,
    };
    console.log('\n== paired bootstrap (1000 resamples, seed 1729, 95% CI)');
    printPaired(comparisons);

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
