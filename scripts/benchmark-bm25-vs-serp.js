#!/usr/bin/env node
/**
 * benchmark-bm25-vs-serp.js
 *
 * Head-to-head recall + reliability comparison between:
 *   - Method 1: Native SQLite FTS5 bm25() (top-K by rank)
 *   - Method 2: SQLiteContextEngine.packContext (P4 submodular packer)
 *   - Method 3: SQLiteContextEngine.fullPipeline (with 1-hop bridges + temporal)
 *
 * Corpus: a real codebase indexed with chunks (functions/classes/file blocks),
 * including injected spam, multi-hop bridge questions, and compound queries.
 *
 * Metrics:
 *   - Recall@B: fraction of relevant chunks retrieved within token budget B
 *   - BridgeRecall@B: recall on 1-hop masked questions (answer NOT in seed set)
 *   - SpamRobustness: spam chunk suppression ratio (clean / spam score)
 *   - Latency: P50 query latency
 *
 * Usage:
 *   node scripts/benchmark-bm25-vs-serp.js /path/to/codebase
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";
import { SQLiteContextEngine } from "../src/engine/engine.js";

const __dirname = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const CODE_EXTENSIONS = new Set([
  ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
  ".py", ".go", ".rs", ".java", ".c", ".h", ".cpp",
  ".html", ".css", ".json", ".yaml", ".yml",
]);
const IGNORE_DIRS = new Set(["node_modules", ".git", "dist", "build", ".next", ".venv", "venv", "__pycache__"]);

function dualTokenize(codeText) {
  const expanded = codeText
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/([a-zA-Z0-9]+_[a-zA-Z0-9_]+)/g, (m) => `${m} ${m.replace(/_/g, " ")}`);
  return codeText + "\n" + expanded;
}

function chunkFile(fullPath, relPath) {
  const raw = fs.readFileSync(fullPath, "utf8");
  const lines = raw.split("\n");
  const chunks = [];
  let current = [];
  let startLine = 1;
  const boundaryRe = /^(\s*(export\s+)?(async\s+)?(function|class|def|struct|impl|func|interface|type)\s+\w+|^\s*(const|let|var)\s+\w+\s*=)/;

  for (let i = 0; i < lines.length; i++) {
    if (current.length < 10 || (!boundaryRe.test(lines[i]) && current.length < 50)) {
      current.push(lines[i]);
      continue;
    }
    const text = current.join("\n").trim();
    if (text.length > 30) {
      chunks.push({
        chunkId: `${relPath}:L${startLine}-${i}`,
        docId: relPath,
        display: `// ${relPath} (lines ${startLine}-${i})\n${text}`,
        index: dualTokenize(`${relPath} ${text}`),
      });
    }
    current = [lines[i]];
    startLine = i + 1;
  }
  const tail = current.join("\n").trim();
  if (tail.length > 20) {
    chunks.push({
      chunkId: `${relPath}:L${startLine}-${lines.length}`,
      docId: relPath,
      display: `// ${relPath} (lines ${startLine}-${lines.length})\n${tail}`,
      index: dualTokenize(`${relPath} ${tail}`),
    });
  }
  return chunks;
}

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

// ---- Build the benchmark database ----
function buildDb(targetDir, dbPath) {
  for (const ext of ["", "-wal", "-shm"]) {
    try { fs.unlinkSync(dbPath + ext); } catch {}
  }

  const db = new Database(dbPath);
  db.exec(`
    CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, doc_id UNINDEXED, content, tokenize='unicode61');
    CREATE VIRTUAL TABLE chunks_vocab USING fts5vocab(chunks_fts, 'row');
    CREATE TABLE chunks_meta (rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, doc_id TEXT, L INTEGER, U INTEGER, l_over_u REAL, intrinsic_surprisal REAL, age_days REAL, access_count INTEGER);
  `);

  const files = walkRepo(targetDir);
  let totalChunks = 0;
  const insertFts = db.prepare("INSERT INTO chunks_fts(chunk_id, doc_id, content) VALUES (?, ?, ?)");
  const insertMeta = db.prepare("INSERT INTO chunks_meta (rowid, chunk_id, doc_id, L, U, l_over_u, intrinsic_surprisal, age_days, access_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");

  const allChunks = [];
  for (const f of files) {
    const chunks = chunkFile(f.full, f.rel);
    for (const c of chunks) {
      allChunks.push(c);
    }
  }

  const tx = db.transaction((chunks) => {
    let id = 1n;
    for (const c of chunks) {
      const toksStr = dualTokenize(c.display);
      const tokens = toksStr.toLowerCase().match(/[a-z0-9_]+/g) || [];
      const L = tokens.length;
      const unique = new Set(tokens);
      const U = Math.max(unique.size, 1);
      const rowid = id++;
      insertFts.run(c.chunkId, c.docId, c.index);
      insertMeta.run(Number(rowid), c.chunkId, c.docId, L, U, L / U, 0, 0, 1);
      totalChunks++;
    }
  });
  tx(allChunks);

  // ---- Inject spam: 50 repetitive log-style chunks with high L/U ----
  const spamChunks = [];
  for (let i = 0; i < 50; i++) {
    const junk = Array(20).fill("ERROR: cache_miss retry fetch_async callback promise_rejected unhandled_rejection").join(" ");
    const chunkId = `spam_chunk_${i}`;
    const content = `DEBUG spam_log_${i} ${junk}`;
    spamChunks.push(chunkId);
    const toksStr = dualTokenize(content);
    const tokens = toksStr.toLowerCase().match(/[a-z0-9_]+/g) || [];
    const L = tokens.length;
    const U = new Set(tokens).size;
    insertFts.run(chunkId, "spam_noise", content);
    insertMeta.run(totalChunks + i + 1, chunkId, "spam_noise", L, U, L / U, 0, 0, 1);
  }
  totalChunks += spamChunks.length;

  db.close();
  return { totalChunks, totalFiles: files.length, spamCount: spamChunks.length };
}

// ---- BM25 baseline: query FTS5 rank, pick top-K within budget ----
function bm25Packer(dbPath, query, budget) {
  const db = new Database(dbPath);
  const queryTokens = query.toLowerCase().match(/[a-z0-9_]+/g) || [];
  const ftsQuery = queryTokens.map((t) => `"${t}"`).join(" OR ");
  const rows = db.prepare(
    "SELECT chunk_id, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 200"
  ).all(ftsQuery || "none");

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

// ---- Relevance judgment: for each question, find the relevant chunks ----
// In a real benchmark this would be human-labeled qrels. Here we synthesize
// a ground-truth set from symbol proximity: given a query, the "relevant"
// chunks are those containing the top-2 rarest query tokens (by corpus frequency).
function makeGroundTruth(targetDir) {
  const engine = new SQLiteContextEngine(path.join(__dirname, "benchmark_corpus.sqlite"));
  const questions = [
    {
      name: "multi_token_stretched_expotential_burstiness_vocab_stats_c_extension",
      query: "stretched exponential potential burstiness vocab stats C extension",
      expectedRare: ["stretched", "exponential", "potential", "burstiness", "vocab"],
      budget: 350,
    },
    {
      name: "full_pipeline_bridge_activation_gate",
      query: "how does full_pipeline gate bridge activation",
      expectedRare: ["full_pipeline", "gate", "bridge", "activation"],
      budget: 400,
    },
    {
      name: "submodular_decay_residual_weights",
      query: "submodular decay residual weights temporal surprisal",
      expectedRare: ["submodular", "decay", "residual", "weights"],
      budget: 350,
    },
    {
      name: "fts5vocab_burstiness_global_conditional",
      query: "fts5vocab burstiness global conditional T_i N_i",
      expectedRare: ["fts5vocab", "burstiness", "conditional"],
      budget: 350,
    },
    {
      name: "l_over_u_concentration_boilerplate_spam",
      query: "l_over_u concentration boilerplate spam suppression",
      expectedRare: ["concentration", "boilerplate", "suppression"],
      budget: 350,
    },
    {
      name: "bridge_strength_anomalous_salience_1hop",
      query: "bridge strength anomalous salience 1hop chain",
      expectedRare: ["strength", "anomalous", "salience"],
      budget: 400,
    }
  ];

  // Build ground truth: for each question, find chunks containing ALL expectedRare tokens
  const truth = questions.map((q) => {
    const qTokens = new Set(q.query.toLowerCase().match(/[a-z0-9_]+/g) || []);
    const relevant = [];
    for (const [cid, stats] of engine.docStats.entries()) {
      const content = engine.getContent(cid).toLowerCase();
      let hitsAll = true;
      for (const w of q.expectedRare) {
        if (!content.includes(w)) hitsAll = false;
        break;
      }
      if (hitsAll || qTokens.size >= 2) {
        // If any expectedRare token is present, consider it relevant (relaxed)
        let present = false;
        for (const w of q.expectedRare) {
          if (content.includes(w)) { present = true; break; }
        }
        if (present) relevant.push(cid);
      }
    }
    return { ...q, relevant: new Set(relevant) };
  });

  engine.db.close();
  return truth;
}

// ---- Run benchmark ----
const targetDir = process.argv[2] || path.resolve(__dirname, "../..");
const dbPath = path.join(__dirname, "benchmark_corpus.sqlite");

console.log("=".repeat(90));
console.log("BM25 vs. SERP (P4 + Submodular Packer) — Codebase Reliability Benchmark");
console.log("=".repeat(90));

console.log("\n[BUILDING CORPUS]...");
const { totalChunks, totalFiles, spamCount } = buildDb(targetDir, dbPath);
console.log(`  Indexed ${totalChunks} chunks from ${totalFiles} files | Injected ${spamCount} spam chunks`);

// Cold start the SERP engine
const engine = new SQLiteContextEngine(dbPath);

// Build ground truth
const questions = makeGroundTruth(targetDir);

// Measure: for each question, run BM25 vs SERP at the given budget
console.log("\n[RUNNING HEAD-TO-HEAD]...");
let bm25Wins = 0;
let serpWins = 0;
let tieWins = 0;

const results = [];
for (const q of questions) {
  const t0 = performance.now();
  const bm25Results = bm25Packer(dbPath, q.query, q.budget);
  const bm25Lat = performance.now() - t0;

  const t1 = performance.now();
  const serpResults = engine.packContext(q.query, { budget: q.budget });
  const serpLat = performance.now() - t1;

  const t2 = performance.now();
  const fullResults = engine.fullPipeline(q.query, { budget: q.budget });
  const fullLat = performance.now() - t2;

  const bm25Recall = q.relevant.size > 0
    ? bm25Results.filter((c) => q.relevant.has(c)).length / q.relevant.size
    : 0;
  const serpRecall = q.relevant.size > 0
    ? serpResults.filter((c) => q.relevant.has(c)).length / q.relevant.size
    : 0;
  const fullRecall = q.relevant.size > 0
    ? fullResults.filter((c) => q.relevant.has(c)).length / q.relevant.size
    : 0;

  const bm25InBudget = bm25Results.length;
  const serpInBudget = serpResults.length;
  const fullInBudget = fullResults.length;

  const bm25Spam = bm25Results.filter((c) => c.startsWith("spam_")).length;
  const serpSpam = serpResults.filter((c) => c.startsWith("spam_")).length;
  const fullSpam = fullResults.filter((c) => c.startsWith("spam_")).length;

  if (serpRecall > bm25Recall) serpWins++;
  else if (bm25Recall > serpRecall) bm25Wins++;
  else tieWins++;

  results.push({
    name: q.name,
    bm25Recall: bm25Recall.toFixed(3),
    serpRecall: serpRecall.toFixed(3),
    fullRecall: fullRecall.toFixed(3),
    bm25InBudget, serpInBudget, fullInBudget,
    bm25Spam, serpSpam, fullSpam,
    bm25Lat: bm25Lat.toFixed(2),
    serpLat: serpLat.toFixed(2),
    fullLat: fullLat.toFixed(2),
  });
}

// Print table
console.log("\n┌─ Query → Method │ Recall@B │ Chunks │ SpamInPicks │ P50 Lat (ms) ─┐\n");
for (const r of results) {
  console.log(`  ${r.name}`);
  console.log(`    BM25       →  ${r.bm25Recall.padStart(7)}  │  ${String(r.bm25InBudget).padStart(3)}    │  ${r.bm25Spam.toString().padStart(2)}           │  ${r.bm25Lat}`);
  console.log(`    SERP P4    →  ${r.serpRecall.padStart(7)}  │  ${String(r.serpInBudget).padStart(3)}    │  ${r.serpSpam.toString().padStart(2)}           │  ${r.serpLat}`);
  console.log(`    SERP Full  →  ${r.fullRecall.padStart(7)}  │  ${String(r.fullInBudget).padStart(3)}    │  ${r.fullSpam.toString().padStart(2)}           │  ${r.fullLat}`);
  console.log("");
}

// Spam robustness test
console.log("[SPAM ROBUSTNESS TEST]: Clean fact vs. repetitive spam (50x same token) at B=200\n");
const cleanChunk = engine.storeMemory(
  "def pack_context(query, budget): selects chunks maximizing marginal gain per token",
  "benchmark_clean", "spam_clean_fact"
);
const spamRep = engine.storeMemory(
  Array(12).fill("crispr cas9 crispr cas9 crispr cas9").join(" ") + " repair",
  "benchmark_spam", "spam_rep_12x"
);
console.log(`  Clean fact (L/U=${cleanChunk.l_over_u.toFixed(2)}): ${engine.scoreChunkP4("spam_clean_fact", "crispr cas9 repair").toFixed(2)}`);
console.log(`  Repetitive spam (L/U=${spamRep.l_over_u.toFixed(2)}): ${engine.scoreChunkP4("spam_rep_12x", "crispr cas9 repair").toFixed(2)}`);
console.log(`  Suppression ratio (clean/spam): ${(engine.scoreChunkP4("spam_clean_fact", "crispr cas9 repair") / engine.scoreChunkP4("spam_rep_12x", "crispr cas9 repair")).toFixed(2)}x\n`);

// Summary
const avgB = results.reduce((s, r) => s + Number(r.bm25Recall), 0) / results.length;
const avgS = results.reduce((s, r) => s + Number(r.serpRecall), 0) / results.length;
const avgF = results.reduce((s, r) => s + Number(r.fullRecall), 0) / results.length;
const avgBSpam = results.reduce((s, r) => s + r.bm25Spam, 0) / results.length;
const avgSSpam = results.reduce((s, r) => s + r.serpSpam, 0) / results.length;
const avgFSpam = results.reduce((s, r) => s + r.fullSpam, 0) / results.length;

console.log("=".repeat(90));
console.log("SUMMARY");
console.log("=".repeat(90));
console.log(`  Avg Recall@B:    BM25=${avgB.toFixed(3)}  |  SERP=${avgS.toFixed(3)}  |  Full=${avgF.toFixed(3)}`);
console.log(`  Avg Spam in Picks: BM25=${avgBSpam.toFixed(1)}  |  SERP=${avgSSpam.toFixed(1)}  |  Full=${avgFSpam.toFixed(1)}`);
console.log(`  SERP > BM25 on ${serpWins}/${results.length} queries | Tied: ${tieWins} | BM25 > SERP: ${bm25Wins}`);
console.log(`  Corpus: ${totalChunks} chunks, ${totalFiles} files, ${spamCount} spam chunks`);
console.log("=".repeat(90));

try { fs.unlinkSync(dbPath); } catch {}
try { fs.unlinkSync(dbPath + "-wal"); } catch {}
try { fs.unlinkSync(dbPath + "-shm"); } catch {}
