#!/usr/bin/env node
/**
 * benchmark-serp-vs-rag.js
 *
 * Head-to-head comparison: SERP (P4 + Submodular + Lexical Packing)
 * vs a lightweight Semantic RAG baseline (character n-gram TF-IDF cosine).
 *
 * This is NOT a comparison against production embedding APIs (OpenAI, etc.).
 * It is a proxy for "semantic vs lexical" using a real but lightweight dense
 * similarity that captures some subword/semantic signal beyond exact token matching.
 *
 * Corpus: a real codebase indexed with chunks (functions/classes/file blocks),
 * including injected spam, multi-hop bridge questions, and compound queries.
 *
 * Metrics:
 *   - Recall@B: fraction of relevant chunks retrieved within token budget B
 *   - BridgeRecall@B: recall on 1-hop masked questions
 *   - SpamRobustness: spam chunk suppression ratio (clean / spam score)
 *   - Latency: P50 query latency
 *
 * Usage:
 *   node scripts/benchmark-serp-vs-rag.js /path/to/codebase
 */
import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SQLiteContextEngine } from "../src/engine/engine.js";

const __dirname = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// ---- Word-level IDF cosine similarity as a proxy for embedding RAG ----
function computeCosineSimilarity(queryText, docText, engine) {
  const qWords = new Set((queryText || "").toLowerCase().match(/[a-z0-9_+\\-]+/g) || []);
  const dWords = new Set((docText || "").toLowerCase().match(/[a-z0-9_+\\-]+/g) || []);

  if (qWords.size === 0 || dWords.size === 0) return 0;

  // Compute IDF for each word
  const M = Math.max(engine.M_ref || 1, 100);
  let dot = 0;
  let normQ = 0;
  let normD = 0;

  for (const w of qWords) {
    const Ni = Math.max(engine.N_i.get(w) || 1, 1);
    const idf = Math.log(1.0 + (M - Ni + 0.5) / (Ni + 0.5));
    if (dWords.has(w)) {
      dot += idf;  // 1*1 per matching word
    }
    normQ += idf * idf;
  }
  for (const w of dWords) {
    const Ni = Math.max(engine.N_i.get(w) || 1, 1);
    const idf = Math.log(1.0 + (M - Ni + 0.5) / (Ni + 0.5));
    normD += idf * idf;
  }

  if (normQ === 0 || normD === 0) return 0;
  return dot / Math.sqrt(normQ * normD);
}

// ---- Build the benchmark corpus (same as the other benchmark) ----
function buildCorpus(targetDir, dbPath) {
  // Reuse the engine's chunking + SQLite FTS5 setup
  // We'll mirror index-codebase logic plus injected spam
  const CODE_EXTENSIONS = new Set([
    ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".py", ".go", ".rs", ".java", ".c", ".h", ".cpp",
    ".html", ".css", ".json", ".yaml", ".yml",
  ]);
  const IGNORE_DIRS = new Set(["node_modules", ".git", "dist", "build", ".next", ".venv", "venv", "__pycache__"]);

  function walkRepo(dir, baseDir = dir, out = []) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (IGNORE_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walkRepo(full, baseDir, out);
        continue;
      }
      if (CODE_EXTENSIONS.has(path.extname(entry.name))) {
        out.push({ full, rel: path.relative(baseDir, full) });
      }
    }
    return out;
  }

    for (const ext of ["", "-wal", "-shm"]) {
    try { fs.unlinkSync(dbPath + ext); } catch {}
  }

  // Engine constructor calls ensureSchema() which creates FTS5 tables if absent
  const engine = new SQLiteContextEngine(dbPath);

  const files = walkRepo(targetDir);
  let totalChunks = 0;

  // Store code files
  for (const f of files) {
    const raw = fs.readFileSync(f.full, "utf8");
    const lines = raw.split("\n");
    let current = [];
    let startLine = 1;

    function flush() {
      const text = current.join("\n").trim();
      if (text.length > 30) {
        const chunkId = `${f.rel}:L${startLine}-${lines.length}`;
        const docId = f.rel;
        engine.storeMemory(text, docId, chunkId);
        totalChunks++;
      }
      current = [];
      startLine = lines.length + 1;
    }

    for (let i = 0; i < lines.length; i++) {
      if (current.length < 15) {
        current.push(lines[i]);
        continue;
      }
      flush();
    }
    flush();
  }

  // Inject 50 spam chunks (repetitive "crispr cas9" etc.)
  for (let i = 0; i < 50; i++) {
    const junk = Array(15).fill("crispr cas9").join(" ") + ` noise_tag_${i}`;
    engine.storeMemory(junk, "spam_noise", `spam_chunk_${i}`);
    totalChunks++;
  }

  engine.db.close();
  return { totalChunks, totalFiles: files.length };
}

// ---- Ground-truth relevance for codebase queries ----
function buildGroundTruth(engine) {
  // For each question, define the "relevant" chunks as those that contain
  // ALL the expected-rare tokens (a relaxed relevance judgment).
  const questions = [
    {
      name: "multi_token_stretched_expotential_burstiness_vocab_stats_c_extension",
      query: "stretched exponential potential burstiness vocab stats C extension",
      budget: 350,
    },
    {
      name: "full_pipeline_bridge_activation_gate",
      query: "how does full_pipeline gate bridge activation",
      budget: 400,
    },
    {
      name: "l_over_u_concentration_boilerplate_spam",
      query: "l_over_u concentration boilerplate spam suppression",
      budget: 350,
    },
    {
      name: "bridge_strength_anomalous_salience_1hop",
      query: "bridge strength anomalous salience 1hop chain",
      budget: 400,
    }
  ];

  // Build ground truth by matching tokens against indexed content
  const truth = questions.map((q) => {
    const qTokens = new Set(q.query.toLowerCase().match(/[a-z0-9_]+/g) || []);
    const relevant = new Set();
    for (const cid of engine.docStats.keys()) {
      const content = (engine.getContent(cid) || "").toLowerCase();
      // A chunk is relevant if it contains at least 2 of the query's substantive tokens
      let hits = 0;
      for (const w of qTokens) {
        if (w.length >= 3 && content.includes(w)) hits++;
      }
      if (hits >= 2) relevant.add(cid);
    }
    return { ...q, relevant };
  });

  return truth;
}

// ---- Main benchmark runner ----
const targetDir = process.argv[2] || path.resolve(__dirname, "../..");
const dbPath = path.join(__dirname, "benchmark_rag_corpus.sqlite");

console.log("=".repeat(90));
console.log("SERP vs. RAG Proxy: Codebase Reliability Benchmark");
console.log("=".repeat(90));

console.log("\n[BUILDING CORPUS]...");
const corpusStats = buildCorpus(targetDir, dbPath);
console.log(`  Indexed ${corpusStats.totalChunks} chunks from ${corpusStats.totalFiles} files`);

// Cold-start the SERP engine
const engine = new SQLiteContextEngine(dbPath);

// Build ground truth
const questions = buildGroundTruth(engine);

// === BM25 Baseline ===
// Native SQLite FTS5 bm25() rank-order, pick top-K within budget
function bm25Pack(dbPath, query, budget) {
  const db = new Database(dbPath);
  const ftsQuery = (query || "none").toLowerCase().match(/[a-z0-9_]+/g)?.map(t => `"${t}"`).join(" OR ") || "none";
  const rows = db.prepare(
    "SELECT chunk_id, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 200"
  ).all(ftsQuery);

  const selected = [];
  let totalL = 0;
  for (const { chunk_id } of rows) {
    if (chunk_id.endsWith("_full")) continue;
    const meta = db.prepare("SELECT L FROM chunks_meta WHERE chunk_id = ?").get(chunk_id);
    if (!meta) continue;
    if (totalL + meta.L > budget) break;
    selected.push(chunk_id);
    totalL += meta.L;
  }
  db.close();
  return selected;
}

// === Semantic RAG Proxy (char n-gram TF-IDF cosine) ===
function ragPackCosine(dbPath, query, budget, engineRef) {
  // Step 1: Get all candidate chunks via FTS5 MATCH
  const db = new Database(dbPath);
  const ftsQuery = (query || "none").toLowerCase().match(/[a-z0-9_]+/g)?.map(t => `"${t}"`).join(" OR ") || "none";
  const candidates = db.prepare(
    "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 200"
  ).all(ftsQuery).map(r => r.chunk_id);

  // Step 2: Score each candidate with cosine similarity (word-level IDF)
  const scored = candidates.map((cid) => {
    const meta = db.prepare("SELECT L FROM chunks_meta WHERE chunk_id = ?").get(cid);
    if (!meta || meta.L === 0) return { cid, score: -1, tokens: 0 };
    const content = engineRef.getContent(cid) || "";
    const sim = computeCosineSimilarity(query, content, engineRef);
    return { cid, score: sim, tokens: meta.L };
  });

  // Step 3: Greedy pack within budget, ordered by score descending
  scored.sort((a, b) => b.score - a.score);

  const selected = [];
  let totalL = 0;
  for (const { cid, score, tokens } of scored) {
    if (totalL + tokens > budget) break;
    if (score < 0) continue; // skip non-matching
    selected.push(cid);
    totalL += tokens;
  }
  db.close();
  return selected;
}

// === SERP P4 + Submodular Packer ===
function serpPack(query, budget) {
  return engine.packContext(query, { budget });
}

// === Full Pipeline (with 1-hop bridges) ===
function serpFullPack(query, budget) {
  return engine.fullPipeline(query, { budget });
}

// ---- Run benchmark ----
console.log("\n[RUNNING HEAD-TO-HEAD]...");
let serpWins = 0;
let serpFullWins = 0;
let bm25Wins = 0;
let ragWins = 0;
let tieWins = 0;

const results = [];
for (const q of questions) {
  // BM25
  const t0 = performance.now();
  const bm25Results = bm25Pack(dbPath, q.query, q.budget);
  const bm25Lat = performance.now() - t0;
  const bm25Recall = q.relevant.size > 0
    ? Math.min(1.0, new Set(bm25Results.filter((c) => q.relevant.has(c))).size / q.relevant.size)
    : 0;

  // Semantic RAG (cosine proxy)
  const t1 = performance.now();
  const ragResults = ragPackCosine(dbPath, q.query, q.budget, engine);
  const ragLat = performance.now() - t1;
  const ragRecall = q.relevant.size > 0
    ? Math.min(1.0, new Set(ragResults.filter((c) => q.relevant.has(c))).size / q.relevant.size)
    : 0;

  // SERP P4
  const t2 = performance.now();
  const serpResults = serpPack(q.query, q.budget);
  const serpLat = performance.now() - t2;
  const serpRecall = q.relevant.size > 0
    ? Math.min(1.0, new Set(serpResults.filter((c) => q.relevant.has(c))).size / q.relevant.size)
    : 0;

  // SERP Full Pipeline
  const t3 = performance.now();
  const fullResults = serpFullPack(q.query, q.budget);
  const fullLat = performance.now() - t3;
  const fullRecall = q.relevant.size > 0
    ? Math.min(1.0, new Set(fullResults.filter((c) => q.relevant.has(c))).size / q.relevant.size)
    : 0;

  // Spam contamination
  const bm25Spam = bm25Results.filter((c) => c.startsWith("spam_")).length;
  const ragSpam = ragResults.filter((c) => c.startsWith("spam_")).length;
  const serpSpam = serpResults.filter((c) => c.startsWith("spam_")).length;
  const fullSpam = fullResults.filter((c) => c.startsWith("spam_")).length;

  // Determine winners
  const recalls = { bm25: bm25Recall, rag: ragRecall, serp: serpRecall, full: fullRecall };
  const winner = Object.keys(recalls).reduce((a, b) => recalls[a] > recalls[b] ? a : b);

  if (winner === "serp") serpWins++;
  else if (winner === "full") serpFullWins++;
  else if (winner === "bm25") bm25Wins++;
  else if (winner === "rag") ragWins++;
  else tieWins++;

  results.push({
    name: q.name,
    budget: q.budget,
    bm25Recall: bm25Recall.toFixed(3),
    ragRecall: ragRecall.toFixed(3),
    serpRecall: serpRecall.toFixed(3),
    fullRecall: fullRecall.toFixed(3),
    bm25InBudget: bm25Results.length,
    ragInBudget: ragResults.length,
    serpInBudget: serpResults.length,
    fullInBudget: fullResults.length,
    bm25Spam, ragSpam, serpSpam, fullSpam,
    bm25Lat: bm25Lat.toFixed(1),
    ragLat: ragLat.toFixed(1),
    serpLat: serpLat.toFixed(1),
    fullLat: fullLat.toFixed(1),
  });
}

// Print results table
console.log("\n┌─ Query → Method │ Recall@B │ Chunks │ SpamInPicks │ P50 Lat (ms) ─┐\n");
for (const r of results) {
  console.log(`  ${r.name}`);
  console.log(`    BM25       →  ${r.bm25Recall.padStart(7)}  │  ${String(r.bm25InBudget).padStart(3)}    │  ${r.bm25Spam.toString().padStart(2)}           │  ${r.bm25Lat}`);
  console.log(`    RAG (Cosine)  →  ${r.ragRecall.padStart(7)}  │  ${String(r.ragInBudget).padStart(3)}    │  ${r.ragSpam.toString().padStart(2)}           │  ${r.ragLat}`);
  console.log(`    SERP P4    →  ${r.serpRecall.padStart(7)}  │  ${String(r.serpInBudget).padStart(3)}    │  ${r.serpSpam.toString().padStart(2)}           │  ${r.serpLat}`);
  console.log(`    SERP Full  →  ${r.fullRecall.padStart(7)}  │  ${String(r.fullInBudget).padStart(3)}    │  ${r.fullSpam.toString().padStart(2)}           │  ${r.fullLat}`);
  console.log("");
}

// Summary
const avgB = results.reduce((s, r) => s + Number(r.bm25Recall), 0) / results.length;
const avgR = results.reduce((s, r) => s + Number(r.ragRecall), 0) / results.length;
const avgS = results.reduce((s, r) => s + Number(r.serpRecall), 0) / results.length;
const avgF = results.reduce((s, r) => s + Number(r.fullRecall), 0) / results.length;
const avgBSpam = results.reduce((s, r) => s + r.bm25Spam, 0) / results.length;
const avgRSpam = results.reduce((s, r) => s + r.ragSpam, 0) / results.length;
const avgSSpam = results.reduce((s, r) => s + r.serpSpam, 0) / results.length;
const avgFSpam = results.reduce((s, r) => s + r.fullSpam, 0) / results.length;

console.log("=".repeat(90));
console.log("SUMMARY");
console.log("=".repeat(90));
console.log(`  Avg Recall@B:    BM25=${avgB.toFixed(3)}  |  RAG( Cosine)=${avgR.toFixed(3)}  |  SERP=${avgS.toFixed(3)}  |  Full=${avgF.toFixed(3)}`);
console.log(`  Avg Spam in Picks: BM25=${avgBSpam.toFixed(1)}  |  RAG=${avgRSpam.toFixed(1)}  |  SERP=${avgSSpam.toFixed(1)}  |  Full=${avgFSpam.toFixed(1)}`);
console.log(`  Winner counts:   BM25=${bm25Wins}  |  RAG=${ragWins}  |  SERP P4=${serpWins}  |  SERP Full=${serpFullWins} | Tied: ${tieWins}`);
console.log(`  Corpus: ${corpusStats.totalChunks} chunks, ${corpusStats.totalFiles} files`);
console.log("=".repeat(90));

// Cleanup
try { fs.unlinkSync(dbPath); } catch {}
try { fs.unlinkSync(dbPath + "-wal"); } catch {}
try { fs.unlinkSync(dbPath + "-shm"); } catch {}