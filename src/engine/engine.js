import Database from 'better-sqlite3';
import { fileURLToPath } from 'node:url';
import { tokenize, countTokens } from './tokenize.js';

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
    this.MAX_CACHE_SIZE = 8000;

    // Prepared statements
    this.stmtMatchCandidates = this.db.prepare(
      "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?"
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

  // Compute True Corpus Information Density (mean IDF of unique tokens)
  computeCorpusSurprisal(counts) {
    if (!counts || counts.size === 0) return 0.1;
    const CHAT_STOP = new Set(["ok","okay","thanks","thank","bro","hey","yes","no","good","sounds","cool","sure","great","lol","please","and","the","for","with","from","that","this","using","all","are","was","were"]);
    let substantiveIdf = 0.0;
    let subCount = 0;
    const M = Math.max(this.M_ref, 100);
    for (const w of counts.keys()) {
      if (w.length < 4 || CHAT_STOP.has(w)) continue;
      const N_w = this.N_i.get(w) || 1;
      substantiveIdf += Math.log(1.0 + (M - Math.min(N_w, M) + 0.5) / (N_w + 0.5));
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
      L_mean: 160.0,
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
      const cRaw = cCounts.get(w) || 0;
      if (cRaw <= 0) continue;
      nActive++;

      const [N_i, T_i] = this.vocabStats(w);
      const mu_i = T_i / Math.max(N_i, 1);
      const beta_i = p.beta_0 * Math.pow(mu_i, -p.gamma_b);

      const volScaleL = Math.pow(L / Math.max(p.L_mean, 1.0), p.delta_l);
      const cEff = Math.max(0.1, cRaw / Math.max(volScaleL, 0.1));

      const pot =
        (1.0 - Math.exp(-beta_i * Math.pow(Math.max(cEff, 0.01), p.p_stretch))) /
        (1.0 - Math.exp(-beta_i + 1e-9));

      const deviation = (conc - p.conc_mean) / Math.max(p.conc_std, 1e-3);
      const entropyPen = p.delta_c * Math.tanh(p.eta * Math.max(0.0, deviation));
      const suppression = Math.exp(-entropyPen);

      const infoW = Math.log(Math.max(this.M_ref, 1) / Math.max(N_i, 1));
      coreVals.push(infoW * pot * suppression * q);
      infoCover += infoW * Math.max(q, 1);
      totalInfo += infoW * Math.max(q, 1);
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
    const p4Params = {
      delta_c: settings.alpha_l,
      alpha_l: settings.alpha_l,
      alpha_u: settings.alpha_u,
      lambda_decay: settings.lambda_decay,
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
      chunkScores,
      initRSum: Math.max(sumWeights(rWeights), 1e-6),
    };
    while (state.totalL < settings.budget && state.selected.length < 15) {
      const bestCid = this.bestContextCandidate(candidates, state, settings);
      if (!bestCid) break;
      const meta = this.docStats.get(bestCid);
      if (!meta || state.totalL + meta.L > settings.budget) break;

      state.selected.push(bestCid);
      state.selectedSet.add(bestCid);
      state.totalL += meta.L;
      this.decayResidualWeights(bestCid, rWeights, settings.lambda_decay);
      state.coveredDocs.add(this.parentDocument(bestCid));
    }
    if (settings.reinforce_access) {
      this.reinforceSelected(state.selected);
    }

    return state.selected;
  }

  reinforceSelected(selected) {
    if (selected.length > 0) this.reinforceChunks(selected);
  }

  parentDocument(chunkId) {
    return this.chunkDocId.get(chunkId) || chunkId.split("_")[0];
  }

  contextCandidates(queryText, bridgeCandidates) {
    const stopThreshold = Math.floor(0.15 * Math.max(this.M_ref, 1));
    const tokens = tokenize(queryText).filter(
      (x) => x.length >= 2 && (!this.N_i.has(x) || this.N_i.get(x) <= stopThreshold)
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
      const [N_i] = this.vocabStats(w);
      weights.set(w,
        Math.log(1.0 + (Math.max(this.M_ref, 1) - N_i + 0.5) / (N_i + 0.5)) * Math.max(q, 1));
    }

    return weights;
  }

  termPotential(word, count) {
    const [N_i, T_i] = this.vocabStats(word);
    const mu_i = T_i / Math.max(N_i, 1);
    const beta_i = 1.5 * Math.pow(mu_i, -0.35);

    return (1.0 - Math.exp(-beta_i * Math.pow(Math.max(count, 0.01), 0.45))) /
      (1.0 - Math.exp(-beta_i + 1e-9));
  }

  residualGain(chunkId, weights) {
    const counts = this.chunkTokenCounts(chunkId);
    let gain = 0.0;
    for (const [w, weight] of weights) {
      const count = counts.get(w) || 0;
      if (count > 0) gain += weight * this.termPotential(w, count);
    }

    return gain;
  }

  decayResidualWeights(chunkId, weights, decay) {
    const counts = this.chunkTokenCounts(chunkId);
    for (const [w, weight] of weights) {
      const count = counts.get(w) || 0;
      if (count <= 0) continue;
      weights.set(w, weight * Math.exp(-decay * this.termPotential(w, count)));
    }
  }

  bestContextCandidate(candidates, state, settings) {
    const remRatio = sumWeights(state.rWeights) / state.initRSum;
    let bestCid = null;
    let bestMarginal = -1.0;
    for (const cid of candidates) {
      if (state.selectedSet.has(cid)) continue;
      const meta = this.docStats.get(cid);
      if (!meta || meta.L === 0 || state.totalL + meta.L > settings.budget) continue;
      const gain = this.contextMarginalGain(cid, state, settings, remRatio);
      if (!(gain > bestMarginal)) continue;
      bestMarginal = gain;
      bestCid = cid;
    }

    return bestCid;
  }

  contextMarginalGain(cid, state, settings, remRatio) {
    const meta = this.docStats.get(cid);
    const marginalSum = this.residualGain(cid, state.rWeights);
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

    // True corpus information density (mean IDF per unique token, length-invariant!)
    const intrinsic = this.computeCorpusSurprisal(counts);
    const cid = chunkId || `${docId}_sent_${Date.now()}_${Math.floor(Math.random() * 1000)}`;

    const insertTx = this.db.transaction(() => {
      const info = this.stmtInsertFts.run(cid, docId, content);
      const rowid = Number(info.lastInsertRowid);
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
  };
  const settings = { ...options };
  for (const [key, value] of Object.entries(defaults)) {
    if (options[key] === undefined) settings[key] = value;
  }

  return settings;
}

function sumWeights(weights) {
  let sum = 0.0;
  for (const value of weights.values()) sum += value;

  return sum;
}
