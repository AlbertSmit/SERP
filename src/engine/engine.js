import Database from 'better-sqlite3';
import { fileURLToPath } from 'node:url';
import { tokenize, gramTokens, wordGrams, countTokens, isSubstantiveToken } from './tokenize.js';

const RELATED_TOKEN_DISCOUNT = 0.85;
const HEADER_MATCH_MULTIPLIER = 1.4;
const SMALL_VAULT_LIMIT = 100;
const TRIGRAM_ABSTENTION_THRESHOLD = 0.45;
const RARE_IDENTIFIER_MAX_DOC_FRACTION = 0.02;
const RARE_IDENTIFIER_MAX_DOCS = 2;

export const DEFAULT_DB_PATH = fileURLToPath(new URL('../../mcp_memory.sqlite', import.meta.url));

export class SQLiteContextEngine {
  constructor(dbPath = DEFAULT_DB_PATH) {
    this.dbPath = dbPath;
    this.db = new Database(dbPath);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("synchronous = NORMAL");
    this.ensureSchema();
    this.loadStats();
  }

  ensureSchema() {
    const hasFts = this.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'").get();
    if (!hasFts) {
      this.db.exec("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, doc_id UNINDEXED, content, tokenize='unicode61'); CREATE VIRTUAL TABLE chunks_vocab USING fts5vocab(chunks_fts, 'row'); CREATE TABLE chunks_meta (rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL, doc_id TEXT, L INTEGER, U INTEGER, l_over_u REAL, intrinsic_surprisal REAL, age_days REAL, access_count INTEGER);");
    }
    this.db.exec("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_grams USING fts5(chunk_id UNINDEXED, doc_id UNINDEXED, content, tokenize='unicode61'); CREATE VIRTUAL TABLE IF NOT EXISTS chunks_grams_vocab USING fts5vocab(chunks_grams, 'row');");
    // Exact tokens as produced by tokenize() (underscores preserved). unicode61
    // splits "figure_1" into "figure" + "1", so a phrase query cannot tell an
    // indexed identifier from prose co-occurrence. This table can.
    this.db.exec("CREATE TABLE IF NOT EXISTS chunks_exact_terms (term TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY(term, chunk_id)) WITHOUT ROWID; CREATE INDEX IF NOT EXISTS chunks_exact_terms_chunk ON chunks_exact_terms(chunk_id);");
  }

  // ------------------------------------------------------------------
  // Fast Cold-Start: Load vocab & compact meta; lazy-load chunk text via LRU
  // ------------------------------------------------------------------
  loadStats() {
    this.N_i = new Map();
    this.T_i = new Map();

    const vocabRows = this.db.prepare("SELECT term, doc, cnt FROM chunks_vocab").all();
    for (const { term, doc, cnt } of vocabRows) {
      this.N_i.set(term, Number(doc));
      this.T_i.set(term, Number(cnt));
    }

    this.M_ref = Number(
      this.db.prepare("SELECT COUNT(*) AS cnt FROM chunks_fts").get()?.cnt || 0
    );

    this.docStats = new Map();
    this.chunkDocId = new Map();
    const metaRows = this.db
      .prepare(
        "SELECT rowid, chunk_id, doc_id, L, U, l_over_u, intrinsic_surprisal, age_days, access_count FROM chunks_meta"
      )
      .all();

    for (const row of metaRows) {
      const L = row.L ? Number(row.L) : 0;
      const U = Math.max(Number(row.U || 1), 1);
      this.docStats.set(row.chunk_id, {
        rowid: Number(row.rowid),
        doc_id: String(row.doc_id),
        L,
        U,
        l_over_u: Number(row.l_over_u),
        intrinsic: Number(row.intrinsic_surprisal),
        age_days: Number(row.age_days),
        access_count: Number(row.access_count),
      });
      this.chunkDocId.set(row.chunk_id, String(row.doc_id));
    }

    // Bounded LRU caches for chunk content & token counts (prevents OOM at 1M rows)
    this.chunkContent = new Map();
    this.chunkCounts = new Map();
    this.chunkHeaderCounts = new Map();
    this.MAX_CACHE_SIZE = 8000;
    this.meanChunkLength = metaRows.length > 0
      ? metaRows.reduce((sum, row) => sum + Number(row.L || 0), 0) / metaRows.length
      : 1;

    // Prepared statements
    this.stmtMatchCandidates = this.db.prepare(
      "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?"
    );
    this.stmtMatchRanked = this.db.prepare(
      "SELECT chunk_id, rank FROM chunks_fts WHERE chunks_fts MATCH ? AND substr(chunk_id, -5) <> '_full' ORDER BY rank LIMIT ?"
    );
    this.stmtGramRanked = this.db.prepare(
      "SELECT chunk_id, rank FROM chunks_grams WHERE chunks_grams MATCH ? AND substr(chunk_id, -5) <> '_full' ORDER BY rank LIMIT ?"
    );
    this.stmtGramCount = this.db.prepare("SELECT COUNT(*) AS cnt FROM chunks_grams");
    this.stmtContentBatch = this.db.prepare(
      "SELECT rowid, chunk_id, doc_id, content FROM chunks_fts WHERE rowid > ? ORDER BY rowid LIMIT ?"
    );
    this.stmtExactTermHit = this.db.prepare(
      "SELECT 1 AS hit FROM chunks_exact_terms WHERE term = ? LIMIT 1"
    );
    this.stmtExactTermCount = this.db.prepare("SELECT COUNT(*) AS cnt FROM chunks_exact_terms");
    this.stmtExactTermBatch = this.db.prepare(
      "SELECT term, COUNT(*) AS df FROM chunks_exact_terms GROUP BY term HAVING COUNT(*) <= ? ORDER BY df ASC, term ASC"
    );
    this.stmtExactTermBatchRetrievable = this.db.prepare(
      "SELECT term, COUNT(*) AS df FROM chunks_exact_terms WHERE substr(chunk_id, -5) <> '_full' GROUP BY term HAVING COUNT(*) <= ? ORDER BY df ASC, term ASC"
    );
    this.stmtInsertExactTerm = this.db.prepare(
      "INSERT OR IGNORE INTO chunks_exact_terms(term, chunk_id) VALUES (?, ?)"
    );
    this.stmtGetContentByRowid = this.db.prepare(
      "SELECT content FROM chunks_fts WHERE rowid = ?"
    );
    this.stmtGetContentByCid = this.db.prepare(
      "SELECT content FROM chunks_fts WHERE chunk_id = ?"
    );
    this.stmtVocabLookup = this.db.prepare(
      "SELECT doc, cnt FROM chunks_vocab WHERE term = ?"
    );
    this.stmtInsertFts = this.db.prepare(
      "INSERT INTO chunks_fts(chunk_id, doc_id, content) VALUES (?, ?, ?)"
    );
    this.stmtInsertGrams = this.db.prepare(
      "INSERT INTO chunks_grams(chunk_id, doc_id, content) VALUES (?, ?, ?)"
    );
    this.stmtInsertMeta = this.db.prepare(`
      INSERT OR REPLACE INTO chunks_meta
      (rowid, chunk_id, doc_id, L, U, l_over_u, intrinsic_surprisal, age_days, access_count)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    this.stmtBumpAccess = this.db.prepare(
      "UPDATE chunks_meta SET access_count = access_count + 1 WHERE chunk_id = ?"
    );
  }

  vocabStats(term) {
    if (this.N_i.has(term)) {
      return [this.N_i.get(term), this.T_i.get(term)];
    }
    const row = this.stmtVocabLookup.get(term);

    return row ? [Number(row.doc), Number(row.cnt)] : [Math.max(this.M_ref, 1), 1];
  }

  buildGramIndex({ batchSize = 500 } = {}) {
    if (Number(this.stmtGramCount.get().cnt) > 0) return 0;
    let lastRowid = 0;
    let indexed = 0;
    const insertBatch = this.db.transaction((rows) => {
      for (const row of rows) {
        this.stmtInsertGrams.run(
          row.chunk_id, row.doc_id, gramTokens(row.content).join(" "),
        );
        indexed += 1;
      }
    });
    while (true) {
      const rows = this.stmtContentBatch.all(lastRowid, batchSize);
      if (rows.length === 0) break;
      insertBatch(rows);
      lastRowid = Number(rows[rows.length - 1].rowid);
    }

    return indexed;
  }

  buildExactIndex({ batchSize = 500 } = {}) {
    if (Number(this.stmtExactTermCount.get().cnt) > 0) return 0;
    let lastRowid = 0;
    let indexed = 0;
    const insertBatch = this.db.transaction((rows) => {
      for (const row of rows) {
        for (const term of new Set(tokenize(row.content))) {
          this.stmtInsertExactTerm.run(term, row.chunk_id);
        }
        indexed += 1;
      }
    });
    while (true) {
      const rows = this.stmtContentBatch.all(lastRowid, batchSize);
      if (rows.length === 0) break;
      insertBatch(rows);
      lastRowid = Number(rows[rows.length - 1].rowid);
    }

    return indexed;
  }

  exactIdentifierTerms({ maxDf = 20, minLength = 4, retrievableOnly = false } = {}) {
    const rows = retrievableOnly
      ? this.stmtExactTermBatchRetrievable.all(maxDf)
      : this.stmtExactTermBatch.all(maxDf);

    return rows
      .filter(({ term }) => term.length >= minLength && isIdentifierToken(term))
      .map(({ term, df }) => ({ term, df: Number(df) }));
  }

  getContent(chunkId) {
    if (this.chunkContent.has(chunkId)) {
      return this.chunkContent.get(chunkId);
    }
    const meta = this.docStats.get(chunkId);
    const row = meta
      ? this.stmtGetContentByRowid.get(meta.rowid)
      : this.stmtGetContentByCid.get(chunkId);
    const content = row ? row.content : "";

    if (this.chunkContent.size >= this.MAX_CACHE_SIZE) {
      const oldestKey = this.chunkContent.keys().next().value;
      this.chunkContent.delete(oldestKey);
      this.chunkCounts.delete(oldestKey);
      this.chunkHeaderCounts.delete(oldestKey);
    }
    this.chunkContent.set(chunkId, content);

    return content;
  }

  chunkTokenCounts(chunkId) {
    if (this.chunkCounts.has(chunkId)) {
      return this.chunkCounts.get(chunkId);
    }
    const content = this.getContent(chunkId);
    const counts = countTokens(tokenize(content));
    this.chunkCounts.set(chunkId, counts);

    return counts;
  }

  chunkHeaderTokenCounts(chunkId) {
    if (this.chunkHeaderCounts.has(chunkId)) return this.chunkHeaderCounts.get(chunkId);
    const headings = this.getContent(chunkId)
      .split("\n")
      .filter((line) => /^\s{0,3}#{1,6}\s/.test(line))
      .join(" ");
    const counts = countTokens(tokenize(headings));
    this.chunkHeaderCounts.set(chunkId, counts);

    return counts;
  }

  // Compute True Corpus Information Density (mean IDF of unique tokens)
  computeCorpusSurprisal(counts) {
    if (!counts || counts.size === 0) return 0.1;
    let substantiveIdf = 0.0;
    let subCount = 0;
    const M = Math.max(this.M_ref, 100);
    for (const w of counts.keys()) {
      if (!isSubstantiveToken(w)) continue;
      const N_w = this.N_i.get(w) || 1;
      substantiveIdf += positiveRsjIdf(M, N_w);
      subCount++;
    }
    if (subCount === 0) return 0.5;

    // Reward both substantive rarity AND information breadth log(1 + subCount)
    return (substantiveIdf / subCount) * Math.log(1.0 + subCount);
  }

  // ------------------------------------------------------------------
  // 1. Unit-Anchored P4 Stretched-Exponential Base Scorer
  // ------------------------------------------------------------------
  scoreChunkP4(chunkId, queryText, params = {}) {
    const p = {
      beta_0: 1.5,
      gamma_b: 0.35,
      delta_l: 0.35,
      delta_c: 0.6,
      eta: 1.0,
      epsilon: 0.35,
      lambda_s: 0.5,
      L_mean: this.meanChunkLength,
      conc_mean: 1.5,
      conc_std: 0.8,
      p_stretch: 0.45,
      ...params,
    };

    const meta = this.docStats.get(chunkId);
    if (!meta) return 0.0;

    const { L, U } = meta;
    const conc = L / Math.max(U, 1);
    const qCounts = countTokens(tokenize(queryText));
    const cCounts = this.chunkTokenCounts(chunkId);

    const coreVals = [];
    let infoCover = 0.0;
    let totalInfo = 0.0;
    let nActive = 0;

    for (const [w, q] of qCounts.entries()) {
      if (!isSubstantiveToken(w)) continue;
      const infoW = this.informationWeight(w);
      totalInfo += infoW * Math.max(q, 1);
      const match = this.cachedTermMatch(chunkId, w, cCounts, p.matchCache);
      if (match.count <= 0) continue;
      nActive++;

      const [N_i, T_i] = this.vocabStats(w);
      const mu_i = T_i / Math.max(N_i, 1);
      const beta_i = p.beta_0 * Math.pow(mu_i, -p.gamma_b);

      const volScaleL = Math.pow(L / Math.max(p.L_mean, 1.0), p.delta_l);
      const cEff = Math.max(0.1, match.count / Math.max(volScaleL, 0.1));

      const pot =
        (1.0 - Math.exp(-beta_i * Math.pow(Math.max(cEff, 0.01), p.p_stretch))) /
        (1.0 - Math.exp(-beta_i + 1e-9));

      const deviation = (conc - p.conc_mean) / Math.max(p.conc_std, 1e-3);
      const entropyPen = p.delta_c * Math.tanh(p.eta * Math.max(0.0, deviation));
      const suppression = Math.exp(-entropyPen);

      coreVals.push(infoW * pot * suppression * q);
      infoCover += infoW * Math.max(q, 1);
    }

    if (nActive === 0) return 0.0;

    const sumPot = coreVals.reduce((a, b) => a + b, 0.0);
    const sublinear = Math.pow(1.0 + sumPot, 1.0 - p.epsilon / Math.max(nActive, 1));
    const diversityCoupling = 1.0 + p.lambda_s * Math.tanh(0.3 * nActive);
    let synergyCoverage = 1.0;
    if (totalInfo > 0) {
      const coverageFraction = Math.min(1.0, infoCover / totalInfo);
      synergyCoverage = 1.0 + p.lambda_s * Math.sqrt(coverageFraction) * Math.tanh(0.5 * nActive);
    }

    return sublinear * diversityCoupling * synergyCoverage * (sumPot + 0.01);
  }

  // ------------------------------------------------------------------
  // 2. Submodular Residual-Decay Packer (With Stratified Pooling)
  // ------------------------------------------------------------------
  packContext(queryText, options = {}) {
    const settings = packingOptions(options);
    const candidates = this.contextCandidates(queryText, settings.bridgeCandidates);
    const rWeights = this.queryWeights(queryText);
    const chunkScores = new Map();
    const matchCache = new Map();
    const p4Params = {
      delta_c: settings.alpha_l,
      alpha_l: settings.alpha_l,
      alpha_u: settings.alpha_u,
      lambda_decay: settings.lambda_decay,
      matchCache,
    };
    for (const cid of candidates) {
      chunkScores.set(cid,
        this.docStats.has(cid) ? this.scoreChunkP4(cid, queryText, p4Params) : 0.01);
    }

    const state = {
      selected: [],
      selectedSet: new Set(),
      coveredDocs: new Set(),
      totalL: 0,
      rWeights,
      queryWeights: new Map(rWeights),
      chunkScores,
      initRSum: Math.max(sumWeights(rWeights), 1e-6),
      firstMarginal: null,
      matchCache,
    };
    while (state.totalL < settings.budget && state.selected.length < 15) {
      if (sumWeights(rWeights) / state.initRSum < 0.08) break;
      const best = this.bestContextCandidate(candidates, state, settings);
      if (!best) break;
      const { cid: bestCid, gain } = best;
      if (state.firstMarginal === null) state.firstMarginal = gain;
      else if (gain / state.firstMarginal < settings.rel_dropoff) break;
      const meta = this.docStats.get(bestCid);
      if (!meta || state.totalL + meta.L > settings.budget) break;

      state.selected.push(bestCid);
      state.selectedSet.add(bestCid);
      state.totalL += meta.L;
      this.decayResidualWeights(bestCid, rWeights, settings.lambda_decay, matchCache);
      state.coveredDocs.add(this.parentDocument(bestCid));
    }
    if (settings.reinforce_access) {
      this.reinforceSelected(state.selected);
    }

    return state.selected;
  }

  // ------------------------------------------------------------------
  // 2b. Explicit stages: retrieve -> fuse -> pack
  // ------------------------------------------------------------------
  rankWordBm25(queryText, { depth = 100 } = {}) {
    // FTS5 unicode61 splits "figure_1" into "figure" + "1", so an unknown
    // identifier would match prose. Drop unknown identifiers per token; a mixed
    // query ("zzq_42 p53") still retrieves through its known tokens.
    const terms = [...new Set(tokenize(queryText))]
      .filter((token) => isSubstantiveToken(token) && !this.isUnknownIdentifier(token));
    if (terms.length === 0) return [];

    return this.stmtMatchRanked.all(ftsOrExpression(terms), depth)
      .map(({ chunk_id: chunkId, rank }) => ({ chunk_id: chunkId, score: -Number(rank) }));
  }

  rankGramBm25(queryText, { depth = 100 } = {}) {
    // Unknown identifiers must not activate gram variants, but a query may mix
    // them with known tokens ("zzq_42 p53"): drop the unknown tokens only.
    const tokens = [...new Set(tokenize(queryText))]
      .filter((token) => !this.isUnknownIdentifier(token));
    if (tokens.length === 0) return [];
    const grams = [...new Set(tokens.flatMap(wordGrams))];
    if (grams.length === 0) return [];

    return this.stmtGramRanked.all(ftsOrExpression(grams), depth)
      .map(({ chunk_id: chunkId, rank }) => ({ chunk_id: chunkId, score: -Number(rank) }));
  }

  isUnknownIdentifier(token) {
    return /[_\d]/.test(token) && !this.identifierHasExactMatch(token);
  }

  identifierHasExactMatch(token) {
    return Boolean(this.stmtExactTermHit.get(token));
  }

  fuseRrf(lists, options = {}) {
    const { k = 60, depth = 100 } = options;
    const exactTokens = this.exactQueryTokens(options);
    const scores = new Map();
    for (const list of lists) {
      const limit = Math.min(list.length, depth);
      for (let index = 0; index < limit; index++) {
        const chunkId = list[index].chunk_id;
        scores.set(chunkId, (scores.get(chunkId) || 0) + 1 / (k + index + 1));
      }
    }

    return [...scores]
      .map(([chunkId, score]) => ({
        chunk_id: chunkId,
        score,
        exact: this.isExactChunkMatch(chunkId, exactTokens),
      }))
      .sort(compareFused);
  }

  exactQueryTokens({ queryTokens, queryText } = {}) {
    const tokens = [...(queryTokens || tokenize(queryText || ''))];

    return new Set(tokens.filter(
      (token) => isIdentifierToken(token) && this.identifierHasExactMatch(token),
    ));
  }

  isExactChunkMatch(chunkId, exactTokens) {
    if (exactTokens.size === 0 || !this.docStats.has(chunkId)) return false;
    const counts = this.chunkTokenCounts(chunkId);
    for (const token of exactTokens) {
      if ((counts.get(token) || 0) > 0) return true;
    }

    return false;
  }

  retrieveRanked(queryText, { channels = ["word", "gram"], depth = 100 } = {}) {
    const word = channels.includes("word") ? this.rankWordBm25(queryText, { depth }) : [];
    const gram = channels.includes("gram") ? this.rankGramBm25(queryText, { depth }) : [];
    const fused = this.fuseRrf([word, gram], { depth, queryText });

    return { word, gram, fused };
  }

  packRanked(queryText, fused, options = {}) {
    return this.packRankedDetailed(queryText, fused, options).chunkIds;
  }

  packRankedDetailed(queryText, fused, options = {}) {
    const settings = rankedPackingOptions(options);
    // Every packer sees the same candidate pool so ranking quality is comparable.
    const limit = Math.min(fused.length, settings.poolDepth);
    const candidates = fused.slice(0, limit).map((entry) => entry.chunk_id);
    const gated = this.gatedRankedCandidates(queryText, candidates, settings);
    const selection = this.runRankedPacker(queryText, gated, settings);

    return {
      chunkIds: selection.chunkIds,
      skippedCoverage: gated.skippedCoverage,
      usedTokens: selection.usedTokens,
      budget: settings.budget,
      packer: settings.packer,
    };
  }

  packHybrid(queryText, options = {}) {
    const { channels, depth, ...packOptions } = options;
    const { fused } = this.retrieveRanked(queryText, { channels, depth });

    return this.packRanked(queryText, fused, packOptions);
  }

  gatedRankedCandidates(queryText, candidates, settings) {
    const queryWeights = this.queryWeights(queryText);
    const seen = new Set();
    const kept = [];
    const skippedCoverage = [];
    for (const chunkId of candidates) {
      if (seen.has(chunkId)) continue;
      seen.add(chunkId);
      const meta = this.docStats.get(chunkId);
      if (!meta || meta.L === 0) continue;
      if (!this.passesCoverageGate(
        chunkId, queryWeights, settings.min_coverage, settings.matchCache,
      )) {
        skippedCoverage.push(chunkId);
        continue;
      }
      kept.push(chunkId);
    }

    return { queryWeights, candidates: kept, skippedCoverage };
  }

  runRankedPacker(queryText, gated, settings) {
    if (settings.packer === "greedy") return this.packRankedGreedy(gated, settings);
    if (settings.packer === "residual") return this.packRankedResidual(gated, settings);
    if (settings.packer === "p4") return this.packRankedP4(queryText, gated, settings);

    return this.packRankedTruncate(gated, settings);
  }

  packRankedTruncate({ candidates }, settings) {
    const chunkIds = [];
    let usedTokens = 0;
    for (const chunkId of candidates) {
      const L = this.docStats.get(chunkId).L;
      if (usedTokens + L > settings.budget) continue;
      chunkIds.push(chunkId);
      usedTokens += L;
    }

    return { chunkIds, usedTokens };
  }

  packRankedGreedy({ candidates, queryWeights }, settings) {
    const remaining = new Map(queryWeights);
    const state = {
      selectedSet: new Set(),
      remaining,
      usedTokens: 0,
      budget: settings.budget,
    };
    while (state.selectedSet.size < candidates.length) {
      const pick = this.bestGreedyCandidate(candidates, state, settings.matchCache);
      if (!pick) break;
      state.selectedSet.add(pick.chunkId);
      state.usedTokens += this.docStats.get(pick.chunkId).L;
      this.coverQueryWords(pick.chunkId, remaining, settings.matchCache);
    }

    return { chunkIds: [...state.selectedSet], usedTokens: state.usedTokens };
  }

  bestGreedyCandidate(candidates, state, matchCache) {
    let best = null;
    for (const chunkId of candidates) {
      if (state.selectedSet.has(chunkId)) continue;
      const meta = this.docStats.get(chunkId);
      if (!meta || state.usedTokens + meta.L > state.budget) continue;
      const gain = this.greedyGain(chunkId, state.remaining, matchCache);
      if (gain <= 0) continue;
      const score = gain / Math.max(meta.L, 1);
      if (isBetterScored(score, chunkId, best)) best = { chunkId, score };
    }

    return best;
  }

  greedyGain(chunkId, weights, matchCache) {
    const counts = this.chunkTokenCounts(chunkId);
    let gain = 0;
    for (const [word, weight] of weights) {
      if (weight <= 0) continue;
      if (this.cachedTermMatch(chunkId, word, counts, matchCache).count > 0) gain += weight;
    }

    return gain;
  }

  coverQueryWords(chunkId, weights, matchCache) {
    const counts = this.chunkTokenCounts(chunkId);
    for (const [word, weight] of weights) {
      if (weight <= 0) continue;
      if (this.cachedTermMatch(chunkId, word, counts, matchCache).count > 0) {
        weights.set(word, 0);
      }
    }
  }

  packRankedResidual({ candidates, queryWeights }, settings) {
    const remaining = new Map(queryWeights);
    const initDemand = Math.max(sumWeights(remaining), 1e-6);
    const state = {
      selectedSet: new Set(),
      remaining,
      usedTokens: 0,
      budget: settings.budget,
      matchCache: settings.matchCache,
    };
    let firstGain = null;
    while (state.selectedSet.size < candidates.length) {
      if (sumWeights(remaining) / initDemand < 0.08) break;
      const pick = this.bestResidualCandidate(candidates, state);
      if (!pick) break;
      if (firstGain === null) firstGain = pick.score;
      else if (pick.score / firstGain < settings.rel_dropoff) break;
      state.selectedSet.add(pick.chunkId);
      state.usedTokens += this.docStats.get(pick.chunkId).L;
      this.decayResidualWeights(pick.chunkId, remaining, settings.lambda_decay, settings.matchCache);
    }

    return { chunkIds: [...state.selectedSet], usedTokens: state.usedTokens };
  }

  bestResidualCandidate(candidates, state) {
    let best = null;
    for (const chunkId of candidates) {
      if (state.selectedSet.has(chunkId)) continue;
      const meta = this.docStats.get(chunkId);
      if (!meta || state.usedTokens + meta.L > state.budget) continue;
      const gain = this.residualGain(chunkId, state.remaining, state.matchCache);
      if (gain <= 0) continue;
      const score = gain / effectiveTokenCost(meta);
      if (isBetterScored(score, chunkId, best)) best = { chunkId, score };
    }

    return best;
  }

  packRankedP4(queryText, { candidates }, settings) {
    const ranked = candidates
      .map((chunkId) => ({
        chunkId,
        score: this.scoreChunkP4(chunkId, queryText, { matchCache: settings.matchCache }),
      }))
      .filter((entry) => entry.score > 0)
      .sort(compareScoredCandidates);
    const chunkIds = [];
    let usedTokens = 0;
    for (const { chunkId } of ranked) {
      const L = this.docStats.get(chunkId).L;
      if (usedTokens + L > settings.budget) continue;
      chunkIds.push(chunkId);
      usedTokens += L;
    }

    return { chunkIds, usedTokens };
  }

  reinforceSelected(selected) {
    if (selected.length > 0) this.reinforceChunks(selected);
  }

  parentDocument(chunkId) {
    return this.chunkDocId.get(chunkId) || chunkId.split("_")[0];
  }

  contextCandidates(queryText, bridgeCandidates) {
    const stopThreshold = Math.max(Math.floor(0.4 * Math.max(this.M_ref, 1)), 25);
    const tokens = tokenize(queryText).filter(
      (x) => isSubstantiveToken(x) && (!this.N_i.has(x) || this.N_i.get(x) <= stopThreshold)
    );
    const candidateSet = new Set();
    if (tokens.length > 0) {
      const uniqueTokens = [...new Set(tokens)];
      const matchExpr = uniqueTokens.map((x) => `"${x}"`).join(" OR ");
      this.addMatchCandidates(candidateSet, matchExpr, 80);
      // Per-term pooling keeps rare query aspects behind common matches reachable.
      const rarestTokens = [...uniqueTokens]
        .sort((a, b) => (this.N_i.get(a) || 1) - (this.N_i.get(b) || 1))
        .slice(0, 4);
      for (const w of rarestTokens) {
        this.addMatchCandidates(candidateSet, `"${w}"`, 20);
      }
    }
    for (const cid of bridgeCandidates) {
      if (!cid.endsWith("_full")) candidateSet.add(cid);
    }
    if (this.M_ref <= SMALL_VAULT_LIMIT) this.addTrigramFallbackCandidates(candidateSet, tokens);

    return [...candidateSet];
  }

  addMatchCandidates(candidateSet, matchExpr, limit) {
    for (const row of this.stmtMatchCandidates.all(matchExpr, limit)) {
      if (!row.chunk_id.endsWith("_full")) candidateSet.add(row.chunk_id);
    }
  }

  queryWeights(queryText) {
    const weights = new Map();
    for (const [w, q] of countTokens(tokenize(queryText))) {
      if (!isSubstantiveToken(w)) continue;
      weights.set(w, this.informationWeight(w) * Math.max(q, 1));
    }

    return weights;
  }

  informationWeight(word) {
    const M = Math.max(this.M_ref, 1);
    if (!this.N_i.has(word)) return Math.log1p(M / 2);

    return positiveRsjIdf(M, this.N_i.get(word));
  }

  termMatch(word, counts, headerCounts = new Map()) {
    const exactCount = counts.get(word) || 0;
    if (exactCount > 0) {
      const multiplier = headerCounts.has(word) ? HEADER_MATCH_MULTIPLIER : 1;
      return { count: exactCount * multiplier, exact: true };
    }
    let relatedCount = 0;
    let isHeaderMatch = false;
    let trigramCount = 0;
    const usesTrigramFallback = this.M_ref <= SMALL_VAULT_LIMIT;
    for (const [candidate, count] of counts) {
      if (areRelatedTokens(word, candidate) && count > relatedCount) {
        relatedCount = count;
        isHeaderMatch = headerCounts.has(candidate);
      }
      if (!usesTrigramFallback) continue;
      const similarity = trigramJaccard(word, candidate);
      if (similarity >= TRIGRAM_ABSTENTION_THRESHOLD) {
        trigramCount = Math.max(trigramCount, count * similarity);
      }
    }
    if (relatedCount === 0 && trigramCount > 0) {
      return { count: trigramCount, exact: false };
    }
    const headerMultiplier = isHeaderMatch ? HEADER_MATCH_MULTIPLIER : 1;

    return { count: relatedCount * RELATED_TOKEN_DISCOUNT * headerMultiplier, exact: false };
  }

  cachedTermMatch(chunkId, word, counts, cache) {
    if (!cache) return this.termMatch(word, counts, this.chunkHeaderTokenCounts(chunkId));
    const key = `${chunkId}\0${word}`;
    if (!cache.has(key)) {
      cache.set(key, this.termMatch(word, counts, this.chunkHeaderTokenCounts(chunkId)));
    }

    return cache.get(key);
  }

  passesCoverageGate(chunkId, queryWeights, minCoverage, matchCache) {
    const totalWeight = sumWeights(queryWeights);
    if (totalWeight <= 0) return false;
    const counts = this.chunkTokenCounts(chunkId);
    let coveredWeight = 0;
    let hasRareIdentifier = false;
    for (const [word, weight] of queryWeights) {
      const match = this.cachedTermMatch(chunkId, word, counts, matchCache);
      if (match.count <= 0) continue;
      coveredWeight += weight;
      if (this.isRareIdentifier(word)) hasRareIdentifier = true;
    }

    return coveredWeight / totalWeight >= minCoverage || hasRareIdentifier;
  }

  isRareIdentifier(word) {
    if (!/[_\d]/.test(word) || !this.N_i.has(word)) return false;
    const threshold = Math.max(
      RARE_IDENTIFIER_MAX_DOCS,
      Math.floor(this.M_ref * RARE_IDENTIFIER_MAX_DOC_FRACTION),
    );

    return this.N_i.get(word) <= threshold;
  }

  addTrigramFallbackCandidates(candidateSet, queryTokens) {
    const substantive = [...new Set(queryTokens.filter(isSubstantiveToken))];
    if (substantive.length === 0) return;
    for (const [chunkId] of this.docStats) {
      if (chunkId.endsWith("_full") || candidateSet.has(chunkId)) continue;
      const chunkTokens = [...this.chunkTokenCounts(chunkId).keys()].filter(isSubstantiveToken);
      const bestSimilarity = maxTokenSimilarity(substantive, chunkTokens);
      if (bestSimilarity >= TRIGRAM_ABSTENTION_THRESHOLD) candidateSet.add(chunkId);
    }
  }

  termPotential(word, count) {
    const [N_i, T_i] = this.vocabStats(word);
    const mu_i = T_i / Math.max(N_i, 1);
    const beta_i = 1.5 * Math.pow(mu_i, -0.35);

    return (1.0 - Math.exp(-beta_i * Math.pow(Math.max(count, 0.01), 0.45))) /
      (1.0 - Math.exp(-beta_i + 1e-9));
  }

  residualGain(chunkId, weights, matchCache) {
    const counts = this.chunkTokenCounts(chunkId);
    let gain = 0.0;
    for (const [w, weight] of weights) {
      const count = this.cachedTermMatch(chunkId, w, counts, matchCache).count;
      if (count > 0) gain += weight * this.termPotential(w, count);
    }

    return gain;
  }

  decayResidualWeights(chunkId, weights, decay, matchCache) {
    const counts = this.chunkTokenCounts(chunkId);
    for (const [w, weight] of weights) {
      const count = this.cachedTermMatch(chunkId, w, counts, matchCache).count;
      if (count <= 0) continue;
      weights.set(w, weight * Math.exp(-decay * this.termPotential(w, count)));
    }
  }

  bestContextCandidate(candidates, state, settings) {
    const remRatio = sumWeights(state.rWeights) / state.initRSum;
    let bestCid = null;
    let bestMarginal = 0.0;
    for (const cid of candidates) {
      if (state.selectedSet.has(cid)) continue;
      const meta = this.docStats.get(cid);
      if (!meta || meta.L === 0 || state.totalL + meta.L > settings.budget) continue;
      const isBridge = state.selected.length > 0 && settings.bridgeBoostMap.has(cid);
      if (!isBridge && !this.passesCoverageGate(
        cid, state.queryWeights, settings.min_coverage, state.matchCache,
      )) continue;
      const gain = this.contextMarginalGain(cid, state, settings, remRatio);
      if (!(gain > bestMarginal)) continue;
      bestMarginal = gain;
      bestCid = cid;
    }

    return bestCid ? { cid: bestCid, gain: bestMarginal } : null;
  }

  contextMarginalGain(cid, state, settings, remRatio) {
    const meta = this.docStats.get(cid);
    const marginalSum = this.residualGain(cid, state.rWeights, state.matchCache);
    const costEff = Math.max(
      Math.pow(meta.L + 10, 0.35) * Math.pow(meta.L / Math.max(meta.U, 1), 0.15), 0.1);
    const parentDoc = this.parentDocument(cid);
    const synergy = !state.coveredDocs.has(parentDoc) ? 1.15 : 0.85;
    const baseScore = state.chunkScores.get(cid) || 0.0;
    const bridgeGate = state.selected.length >= 1 ? 1.0 : 0.0;
    const bridgeBoostVal = (settings.bridgeBoostMap.get(cid) || 0.0) * bridgeGate * 0.35;
    const temporalFactor = settings.temporal_mode ? this.temporalFactor(cid) : 1.0;
    const louSuppression = settings.temporal_mode
      ? Math.exp(-settings.alpha_u * Math.max(0.0, meta.l_over_u - 2.2)) : 1.0;

    return ((marginalSum + baseScore * 0.05 * remRatio + settings.w_bridge * bridgeBoostVal) *
      synergy * temporalFactor * louSuppression) / costEff;
  }

  // ------------------------------------------------------------------
  // 3. Zero-Embedding 1-Hop Lexical Bridge Expansion
  // ------------------------------------------------------------------
  expandBridges(queryText, seedChunks) {
    if (!seedChunks || seedChunks.length === 0) {
      return { bridgeEvidence: [], psiMap: new Map() };
    }
    const seeds = seedChunks.slice(0, 3);
    const qWords = new Set(tokenize(queryText));
    const bridgeCandidates = new Map();

    for (const cid of seeds) {
      this.collectBridgeTerms(cid, { seeds, qWords, bridgeCandidates });
    }

    const topBridges = [...bridgeCandidates.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    const psiMap = new Map(topBridges);
    const seedSet = new Set(seeds);
    const bridgeEvidence = [];
    const seenEvidence = new Set();
    for (const [word] of topBridges) {
      this.collectBridgeEvidence(word, { seedSet, seenEvidence, bridgeEvidence });
    }

    return { bridgeEvidence, psiMap };
  }

  collectBridgeTerms(chunkId, context) {
    const { seeds, qWords, bridgeCandidates } = context;
    for (const w of this.chunkTokenCounts(chunkId).keys()) {
      if (qWords.has(w) || w.length < 2 || bridgeCandidates.has(w)) continue;
      bridgeCandidates.set(w, this.bridgeStrength(w, seeds));
    }
  }

  collectBridgeEvidence(word, context) {
    const { seedSet, seenEvidence, bridgeEvidence } = context;
    for (const { chunk_id } of this.stmtMatchCandidates.all(`"${word}"`, 20)) {
      if (chunk_id.endsWith("_full") || seedSet.has(chunk_id) || seenEvidence.has(chunk_id)) continue;
      seenEvidence.add(chunk_id);
      bridgeEvidence.push(chunk_id);
    }
  }

  bridgeStrength(word, seeds) {
    const [N_i, T_i] = this.vocabStats(word);
    let seedHits = 0;
    for (const seed of seeds) {
      if (this.chunkTokenCounts(seed).has(word)) seedHits++;
    }

    return Math.pow(seedHits, 1.5) *
      Math.log(Math.max(this.M_ref, 1) / Math.max(N_i, 1)) *
      Math.pow(Math.max(T_i, 1) / Math.max(N_i, 1), -0.4);
  }

  // ------------------------------------------------------------------
  // 4. Corpus-IDF Regulated Temporal Memory Decay & Full Pipeline
  // ------------------------------------------------------------------
  temporalFactor(chunkId) {
    const meta = this.docStats.get(chunkId);
    if (!meta) return 1.0;
    // Use true corpus-level token rarity + structural density (fixes length bias!)
    const counts = this.chunkTokenCounts(chunkId);
    const corpusSurprisal = this.computeCorpusSurprisal(counts);
    const structuralDensity = Math.pow(Math.max(meta.U / Math.max(meta.L, 1), 0.05), 0.5);
    const decayRate =
      0.012 /
      (Math.max(corpusSurprisal ** 2, 0.1) *
        structuralDensity *
        Math.log(1 + meta.access_count));

    return Math.exp(-decayRate * meta.age_days);
  }

  fullPipeline(queryText, options = {}) {
    const {
      budget = 250,
      use_bridges = true,
      temporal_mode = true,
      lambda_decay = 2.2,
      alpha_l = 0.2,
      alpha_u = 0.05,
      w_bridge = 10.0,
      min_coverage = 0.315,
      rel_dropoff = 0.26,
      reinforce_access = false,
    } = options;

    const baseSelected = this.packContext(queryText, {
      budget,
      lambda_decay,
      alpha_l,
      alpha_u,
      w_bridge: 0.0,
      temporal_mode: false,
      reinforce_access: false,
      min_coverage,
      rel_dropoff,
    });

    const bridges = this.contextBridges(queryText, baseSelected, use_bridges);

    return this.packContext(queryText, {
      budget,
      bridgeCandidates: bridges.bridgeEvidence,
      bridgeBoostMap: bridges.bridgeBoostMap,
      lambda_decay,
      alpha_l,
      alpha_u,
      w_bridge,
      temporal_mode,
      reinforce_access,
      min_coverage,
      rel_dropoff,
    });
  }

  contextBridges(queryText, baseSelected, enabled) {
    const qCounts = countTokens(tokenize(queryText));
    const empty = { bridgeEvidence: [], bridgeBoostMap: new Map() };
    if (!enabled || baseSelected.length < 1 || qCounts.size === 0) return empty;
    const coveredWords = new Set();
    for (const cid of baseSelected) {
      this.collectCoveredWords(cid, qCounts, coveredWords);
    }
    const primaryCoverage = coveredWords.size / Math.max(qCounts.size, 1);
    if (primaryCoverage <= 0.3) return empty;
    const { bridgeEvidence, psiMap } = this.expandBridges(queryText, baseSelected.slice(0, 3));
    const bridgeBoostMap = new Map();
    for (const cid of bridgeEvidence) {
      bridgeBoostMap.set(cid, this.bridgeBoost(cid, psiMap));
    }

    return { bridgeEvidence, bridgeBoostMap };
  }

  collectCoveredWords(chunkId, queryCounts, coveredWords) {
    for (const word of this.chunkTokenCounts(chunkId).keys()) {
      if (queryCounts.has(word)) coveredWords.add(word);
    }
  }

  bridgeBoost(chunkId, psiMap) {
    const counts = this.chunkTokenCounts(chunkId);
    let maxPsi = 0.0;
    for (const [b, psi] of psiMap) {
      if ((counts.get(b) || 0) > 0) maxPsi = Math.max(maxPsi, psi);
    }

    return maxPsi;
  }

  // ------------------------------------------------------------------
  // 5. Real-Time Memory Ingestion & Hebbian Access Reinforcement
  // ------------------------------------------------------------------
  reinforceChunks(chunkIds) {
    const tx = this.db.transaction((cids) => {
      for (const cid of cids) {
        this.stmtBumpAccess.run(cid);
        const meta = this.docStats.get(cid);
        if (meta) meta.access_count += 1;
      }
    });
    tx(chunkIds);
  }

  storeMemory(content, docId = "agent_memory", chunkId = null, ageDays = 0.0) {
    const tokens = tokenize(content);
    const L = tokens.length;
    if (L === 0) return null;

    const counts = countTokens(tokens);
    const U = Math.max(counts.size, 1);
    const lOverU = L / U;

    for (const [term, c] of counts.entries()) {
      this.N_i.set(term, (this.N_i.get(term) || 0) + 1);
      this.T_i.set(term, (this.T_i.get(term) || 0) + c);
    }
    this.M_ref += 1;
    this.meanChunkLength = ((this.meanChunkLength * (this.M_ref - 1)) + L) / this.M_ref;

    // True corpus information density (mean IDF per unique token, length-invariant!)
    const intrinsic = this.computeCorpusSurprisal(counts);
    const cid = chunkId || `${docId}_sent_${Date.now()}_${Math.floor(Math.random() * 1000)}`;

    const insertTx = this.db.transaction(() => {
      const info = this.stmtInsertFts.run(cid, docId, content);
      const rowid = Number(info.lastInsertRowid);
      this.stmtInsertGrams.run(cid, docId, gramTokens(content).join(" "));
      for (const term of counts.keys()) this.stmtInsertExactTerm.run(term, cid);
      this.stmtInsertMeta.run(rowid, cid, docId, L, U, lOverU, intrinsic, ageDays, 1);

      return rowid;
    });

    const rowid = insertTx();

    this.docStats.set(cid, {
      rowid,
      doc_id: String(docId),
      L,
      U,
      l_over_u: lOverU,
      intrinsic,
      age_days: Number(ageDays),
      access_count: 1,
    });
    this.chunkDocId.set(cid, String(docId));
    this.chunkContent.set(cid, content);
    this.chunkCounts.set(cid, counts);
    this.chunkHeaderCounts.delete(cid);

    return {
      chunk_id: cid,
      doc_id: docId,
      L,
      U,
      l_over_u: Number(lOverU.toFixed(2)),
      intrinsic_surprisal: Number(intrinsic.toFixed(4)),
    };
  }
}

function packingOptions(options) {
  const defaults = {
    budget: 250,
    bridgeCandidates: [],
    bridgeBoostMap: new Map(),
    lambda_decay: 2.2,
    alpha_l: 0.2,
    alpha_u: 0.05,
    w_bridge: 1.0,
    temporal_mode: false,
    reinforce_access: false,
    min_coverage: 0.315,
    rel_dropoff: 0.26,
  };
  const settings = { ...options };
  for (const [key, value] of Object.entries(defaults)) {
    if (options[key] === undefined) settings[key] = value;
  }

  return settings;
}

function rankedPackingOptions(options = {}) {
  const defaults = {
    budget: 250,
    packer: "truncate",
    min_coverage: 0.315,
    rel_dropoff: 0.26,
    lambda_decay: 2.2,
    poolDepth: 100,
  };
  const settings = { ...options };
  for (const [key, value] of Object.entries(defaults)) {
    if (settings[key] === undefined) settings[key] = value;
  }
  settings.matchCache = new Map();

  return settings;
}

function ftsOrExpression(terms) {
  return terms.map((term) => `"${term.replaceAll('"', '""')}"`).join(" OR ");
}

function isIdentifierToken(token) {
  return /[a-z]/.test(token) && /[_\d]/.test(token);
}

function compareFused(left, right) {
  if (left.exact !== right.exact) return left.exact ? -1 : 1;
  if (right.score !== left.score) return right.score - left.score;

  return compareChunkIds(left.chunk_id, right.chunk_id);
}

function compareScoredCandidates(left, right) {
  if (right.score !== left.score) return right.score - left.score;

  return compareChunkIds(left.chunkId, right.chunkId);
}

function isBetterScored(score, chunkId, best) {
  if (!best) return true;
  if (score !== best.score) return score > best.score;

  return chunkId < best.chunkId;
}

function compareChunkIds(left, right) {
  if (left === right) return 0;

  return left < right ? -1 : 1;
}

function effectiveTokenCost(meta) {
  return Math.max(
    Math.pow(meta.L + 10, 0.35) * Math.pow(meta.L / Math.max(meta.U, 1), 0.15),
    0.1,
  );
}

function sumWeights(weights) {
  let sum = 0.0;
  for (const value of weights.values()) sum += value;

  return sum;
}

function positiveRsjIdf(corpusSize, documentFrequency) {
  const M = Math.max(corpusSize, 1);
  const N = Math.min(Math.max(documentFrequency, 0), M);

  return Math.log1p((M - N + 0.5) / (N + 0.5));
}

function areRelatedTokens(left, right) {
  if (left.length < 4 || right.length < 4) return false;
  if (left.startsWith(right) || right.startsWith(left)) return true;
  if (left.includes(right) || right.includes(left)) return true;
  const leftParts = left.split("_");
  const rightParts = right.split("_");
  if (leftParts.includes(right) || rightParts.includes(left)) return true;

  return tokenStem(left) === tokenStem(right);
}

function tokenStem(token) {
  return token.replace(/(?:heden|ingen|eren|lijk|baar|isch|ing|ers|en|es|s)$/u, "");
}

function maxTokenSimilarity(leftTokens, rightTokens) {
  let best = 0;
  for (const left of leftTokens) {
    for (const right of rightTokens) {
      best = Math.max(best, trigramJaccard(left, right));
    }
  }

  return best;
}

function trigramJaccard(left, right) {
  const leftTrigrams = trigrams(left);
  const rightTrigrams = trigrams(right);
  let intersection = 0;
  for (const trigram of leftTrigrams) {
    if (rightTrigrams.has(trigram)) intersection++;
  }
  const union = leftTrigrams.size + rightTrigrams.size - intersection;

  return union > 0 ? intersection / union : 0;
}

function trigrams(value) {
  if (value.length < 3) return new Set([value]);
  const result = new Set();
  for (let i = 0; i <= value.length - 3; i++) result.add(value.slice(i, i + 3));

  return result;
}
