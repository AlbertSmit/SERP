#!/usr/bin/env node
/**
 * benchmark-react-corpus.js
 * Head-to-head on real TanStack Query codebase.
 */
import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SQLiteContextEngine } from "../src/engine/engine.js";

const __dirname = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const CODE_EXTENSIONS = new Set([".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"]);
const IGNORE_DIRS = new Set(["node_modules", ".git", "dist", "build", ".next", ".venv", "__pycache__", ".turbo", "coverage"]);

function walkRepo(dir, baseDir = dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (IGNORE_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) { walkRepo(full, baseDir, out); continue; }
    if (CODE_EXTENSIONS.has(path.extname(entry.name))) out.push({ full, rel: path.relative(baseDir, full) });
  }
  return out;
}

function buildCorpus(targetDir, dbPath) {
  for (const ext of ["", "-wal", "-shm"]) { try { fs.unlinkSync(dbPath + ext); } catch {} }
  const engine = new SQLiteContextEngine(dbPath);
  const files = walkRepo(targetDir);
  let total = 0;
  for (const f of files) {
    const raw = fs.readFileSync(f.full, "utf8");
    const lines = raw.split("\n");
    let cur = [], start = 1;
    const boundaryRe = /^\s*(export\s+)?(async\s+)?(function|class|def|const)\s+\w/;
    for (let i = 0; i < lines.length; i++) {
      if (cur.length < 15 || (!boundaryRe.test(lines[i]) && cur.length < 50)) { cur.push(lines[i]); continue; }
      const text = cur.join("\n").trim();
      if (text.length > 30) { engine.storeMemory(text, f.rel, f.rel + ":L" + start + "-" + i); total++; }
      cur = [lines[i]]; start = i + 1;
    }
    const tail = cur.join("\n").trim();
    if (tail.length > 30) { engine.storeMemory(tail, f.rel, f.rel + ":L" + start + "-" + lines.length); total++; }
  }
  for (let i = 0; i < 100; i++) {
    engine.storeMemory(Array(10).fill("useEffect useCallback useRef useState").join(" ") + " noise_" + i, "spam", "spam_chunk_" + i);
    total++;
  }
  engine.db.close();
  return { totalFiles: files.length, totalChunks: total };
}

function engStore(engine, docId, chunkId, content) {
  engine.storeMemory(content, docId, chunkId);
}

function bm25Pack(dbPath, query, budget) {
  const db = new Database(dbPath);
  const ftsQuery = (query || "none").toLowerCase().match(/[a-z0-9_]+/g)?.map(t => `"${t}"`).join(" OR ") || "none";
  const rows = db.prepare("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 200").all(ftsQuery);
  const selected = []; let totalL = 0;
  for (const { chunk_id } of rows) {
    if (chunk_id.endsWith("_full")) continue;
    const meta = db.prepare("SELECT L FROM chunks_meta WHERE chunk_id = ?").get(chunk_id);
    if (!meta || totalL + meta.L > budget) break;
    selected.push(chunk_id); totalL += meta.L;
  }
  db.close();
  return selected;
}

function computeCosineSimilarity(queryText, docText, engineRef) {
  const qWords = new Set((queryText || "").toLowerCase().match(/[a-z0-9_]+/g) || []);
  const dWords = new Set((docText || "").toLowerCase().match(/[a-z0-9_]+/g) || []);
  if (!qWords.size || !dWords.size) return 0;
  const M = Math.max(engineRef.M_ref || 1, 100);
  let dot = 0, nq = 0, nd = 0;
  for (const w of qWords) {
    const Ni = Math.max(engineRef.N_i.get(w) || 1, 1);
    const idf = Math.log(1.0 + (M - Ni + 0.5) / (Ni + 0.5));
    nq += idf * idf;
    if (dWords.has(w)) dot += idf;
  }
  for (const w of dWords) {
    const Ni = Math.max(engineRef.N_i.get(w) || 1, 1);
    const idf = Math.log(1.0 + (M - Ni + 0.5) / (Ni + 0.5));
    nd += idf * idf;
  }
  if (nq === 0 || nd === 0) return 0;
  return dot / Math.sqrt(nq * nd);
}

function ragPack(dbPath, query, budget, engineRef) {
  const db = new Database(dbPath);
  const ftsQuery = (query || "none").toLowerCase().match(/[a-z0-9_]+/g)?.map(t => `"${t}"`).join(" OR ") || "none";
  const candidates = db.prepare("SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 200").all(ftsQuery).map(r => r.chunk_id);
  const scored = candidates.map((cid) => {
    const meta = db.prepare("SELECT L FROM chunks_meta WHERE chunk_id = ?").get(cid);
    if (!meta || meta.L === 0) return { cid, score: -1 };
    const content = engineRef.getContent(cid) || "";
    return { cid, score: computeCosineSimilarity(query, content, engineRef), tokens: meta.L };
  });
  scored.sort((a, b) => b.score - a.score);
  const selected = []; let totalL = 0;
  for (const { cid, score, tokens } of scored) {
    if (totalL + tokens > budget) break;
    if (score < 0) continue;
    selected.push(cid); totalL += tokens;
  }
  db.close();
  return selected;
}

function buildGroundTruth(engine, targetDir, questions) {
  return questions.map((q) => {
    const qTokens = new Set(q.query.toLowerCase().match(/[a-z0-9_]+/g) || []);
    const relevant = new Set();
    for (const [cid, stats] of engine.docStats.entries()) {
      const content = (engine.getContent(cid) || "").toLowerCase();
      let hits = 0;
      for (const w of qTokens) { if (w.length >= 4 && content.includes(w)) hits++; }
      if (hits >= 3) relevant.add(cid);
    }
    return { ...q, relevant };
  });
}

const targetDir = process.argv[2] || path.resolve(__dirname, "..");
const dbPath = path.join(__dirname, "benchmark_react_corpus.sqlite");

console.log("=".repeat(90));
console.log("SERP vs BM25 vs RAG — REAL React Codebase Benchmark");
console.log("=".repeat(90));
console.log("\n[CORPUS]: TanStack Query packages directory");
const corpusStats = buildCorpus(targetDir, dbPath);
console.log("  " + corpusStats.totalChunks + " chunks from " + corpusStats.totalFiles + " files");

const engine = new SQLiteContextEngine(dbPath);

const questions = [
  { name: "useQuery_staleTime_cacheTime_refetchOnMount", query: "useQuery staleTime cacheTime refetchOnMount", budget: 300 },
  { name: "QueryClientProvider_defaultOptions_clientConfig", query: "QueryClientProvider defaultOptions clientConfig", budget: 350 },
  { name: "useMutation_onError_onSuccess_retry", query: "useMutation onError onSuccess retry", budget: 300 },
  { name: "infiniteQueryObserver_fetchNextPage_initialPageSize", query: "infiniteQueryObserver fetchNextPage initialPageSize", budget: 350 },
  { name: "useSuspense_refetchOnMount_queryKey_staleTime", query: "useSuspense refetchOnMount queryKey staleTime", budget: 300 },
  { name: "fallbackComponent_refreshRerenderRefreshRefetch", query: "fallbackComponent refresh rerender refreshRefetch", budget: 300 },
];

const truth = buildGroundTruth(engine, targetDir, questions);

console.log("\n[RUNNING HEAD-TO-HEAD]");
const results = [];
let serpWins = 0, fullWins = 0, bm25Wins = 0, ragWins = 0, ties = 0;

for (const q of truth) {
  const t0 = performance.now();
  const bm25R = bm25Pack(dbPath, q.query, q.budget) || [];
  const bm25Lat = performance.now() - t0;

  const t1 = performance.now();
  const ragR = ragPack(dbPath, q.query, q.budget, engine) || [];
  const ragLat = performance.now() - t1;

  const t2 = performance.now();
  const serpR = engine.packContext(q.query, { budget: q.budget }) || [];
  const serpLat = performance.now() - t2;

  const t3 = performance.now();
  const fullR = engine.fullPipeline(q.query, { budget: q.budget }) || [];
  const fullLat = performance.now() - t3;

  const bRecall = q.relevant.size > 0 ? Math.min(1, new Set(bm25R.filter(c => q.relevant.has(c))).size / q.relevant.size) : 0;
  const rRecall = q.relevant.size > 0 ? Math.min(1, new Set(ragR.filter(c => q.relevant.has(c))).size / q.relevant.size) : 0;
  const sRecall = q.relevant.size > 0 ? Math.min(1, new Set(serpR.filter(c => q.relevant.has(c))).size / q.relevant.size) : 0;
  const fRecall = q.relevant.size > 0 ? Math.min(1, new Set(fullR.filter(c => q.relevant.has(c))).size / q.relevant.size) : 0;

  const bSpam = bm25R.filter(c => c.startsWith("spam_")).length;
  const rSpam = ragR.filter(c => c.startsWith("spam_")).length;
  const sSpam = serpR.filter(c => c.startsWith("spam_")).length;
  const fSpam = fullR.filter(c => c.startsWith("spam_")).length;

  const winners = { bm25: bRecall, rag: rRecall, serp: sRecall, full: fRecall };
  const best = Object.keys(winners).reduce((a, b) => winners[a] > winners[b] ? a : b);
  if (best === "serp") serpWins++;
  else if (best === "full") fullWins++;
  else if (best === "bm25") bm25Wins++;
  else if (best === "rag") ragWins++;
  else ties++;

  results.push({
    name: q.name,
    bm25Recall: Number(bRecall.toFixed(3)),
    ragRecall: Number(rRecall.toFixed(3)),
    serpRecall: Number(sRecall.toFixed(3)),
    fullRecall: Number(fRecall.toFixed(3)),
    bInBudget: bm25R.length, ragInBudget: ragR.length, serpInBudget: serpR.length, fullInBudget: fullR.length,
    bSpam, rSpam, sSpam, fSpam,
    bLat: Number(bm25Lat.toFixed(1)), ragLat: Number(ragLat.toFixed(1)), serpLat: Number(serpLat.toFixed(1)), fullLat: Number(fullLat.toFixed(1))
  });
}

console.log("\n┌─ Query → Method │ Recall@B │ Chunks │ SpamInPicks ─┐\n");
for (const r of results) {
  console.log("  " + r.name);
  console.log("    BM25       → " + r.bm25Recall + " │ " + String(r.bInBudget) + "    │ " + r.bSpam.toString());
  console.log("    RAG(Cosine)→ " + r.ragRecall + " │ " + String(r.ragInBudget) + "    │ " + r.rSpam.toString());
  console.log("    SERP P4    → " + r.serpRecall + " │ " + String(r.serpInBudget) + "    │ " + r.sSpam.toString());
  console.log("    SERP Full  → " + r.fullRecall + " │ " + String(r.fullInBudget) + "    │ " + r.fSpam.toString());
}

const avgB = results.reduce((s, r) => s + r.bm25Recall, 0) / results.length;
const avgR = results.reduce((s, r) => s + r.ragRecall, 0) / results.length;
const avgS = results.reduce((s, r) => s + r.serpRecall, 0) / results.length;
const avgF = results.reduce((s, r) => s + r.fullRecall, 0) / results.length;
const avgBS = results.reduce((s, r) => s + r.bSpam, 0) / results.length;
const avgRS = results.reduce((s, r) => s + r.rSpam, 0) / results.length;
const avgSS = results.reduce((s, r) => s + r.sSpam, 0) / results.length;
const avgFS = results.reduce((s, r) => s + r.fSpam, 0) / results.length;

console.log("\n======================================================================");
console.log("SUMMARY");
console.log("  Avg Recall@B:     BM25=" + avgB.toFixed(3) + " | RAG=" + avgR.toFixed(3) + " | SERP=" + avgS.toFixed(3) + " | Full=" + avgF.toFixed(3));
console.log("  Avg Spam in Picks: BM25=" + avgBS.toFixed(1) + " | RAG=" + avgRS.toFixed(1) + " | SERP=" + avgSS.toFixed(1) + " | Full=" + avgFS.toFixed(1));
console.log("  Winners: BM25=" + bm25Wins + " | RAG=" + ragWins + " | SERP P4=" + serpWins + " | Full=" + fullWins + " | Ties=" + ties);
console.log("  Corpus: " + corpusStats.totalChunks + " chunks, " + corpusStats.totalFiles + " files");

try { fs.unlinkSync(dbPath); } catch {}
try { fs.unlinkSync(dbPath + "-wal"); } catch {}
try { fs.unlinkSync(dbPath + "-shm"); } catch {}
