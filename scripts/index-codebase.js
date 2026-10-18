#!/usr/bin/env node
/**
 * index_codebase.js — Zero-Embedding Codebase Graph-RAG on SQLite P4 Engine
 * Chunks a codebase by functions/classes, applies Dual-Emit Camel/Snake tokenization,
 * and demonstrates:
 *   1. Multi-File Feature Context Packing
 *   2. 1-Hop Call-Graph / Symbol Hopping (Bridge Expansion)
 *   3. Structural Pattern / Clone Recognition
 */
import fs from "node:fs";
import path from "node:path";
import { SQLiteContextEngine } from "../src/engine/engine.js";

const CODE_EXTENSIONS = new Set([".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h", ".go", ".rs", ".java"]);
const IGNORE_DIRS = new Set(["node_modules", ".git", "dist", "build", "__pycache__", ".next", "venv", ".venv"]);

// Dual-Emit Code Tokenizer: emits both exact symbol ('get_user_token') AND split subwords ('get', 'user', 'token')
export function dualTokenizeCode(codeText) {
  // Split camelCase (getUserToken -> get User Token) and snake_case alongside original identifiers
  const expanded = codeText.replace(/([a-z0-9])([A-Z])/g, "$1 $2").replace(/([a-zA-Z0-9]+_[a-zA-Z0-9_]+)/g, (m) => `${m} ${m.replace(/_/g, " ")}`);

  return codeText + "\n" + expanded;
}

// Split source file into logical function/class blocks (~15 to 120 tokens each)
function chunkSourceFile(filePath, relPath) {
  const raw = fs.readFileSync(filePath, "utf8");
  const lines = raw.split("\n");
  const chunks = [];
  let currentLines = [];
  let startLine = 1;

  // Boundary regex for functions, classes, methods, or top-level blocks
  const boundaryRe = /^(\s*(export\s+)?(async\s+)?(function|class|def|struct|impl|func|interface|type)\s+[a-zA-Z0-9_]+|static\s+[a-zA-Z0-9_*\s]+\([a-zA-Z0-9_*\s,]*\)\s*\{|#\s*---)/;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (currentLines.length < 12 || (!boundaryRe.test(line) && currentLines.length < 55)) {
      currentLines.push(line);
      continue;
    }
    const text = currentLines.join("\n").trim();
    if (text.length > 30) {
      chunks.push({
        chunkId: `${relPath}:L${startLine}-${i}`,
        docId: relPath,
        displayContent: `// File: ${relPath} (Lines ${startLine}-${i})\n${text}`,
        indexContent: dualTokenizeCode(`${relPath} ${text}`),
      });
    }
    currentLines = [];
    startLine = i + 1;
    currentLines.push(line);
  }
  if (currentLines.length > 0) {
    const text = currentLines.join("\n").trim();
    if (text.length > 20) {
      chunks.push({
        chunkId: `${relPath}:L${startLine}-${lines.length}`,
        docId: relPath,
        displayContent: `// File: ${relPath} (Lines ${startLine}-${lines.length})\n${text}`,
        indexContent: dualTokenizeCode(`${relPath} ${text}`),
      });
    }
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

// Run live demo on the current /Users/asmit/Personal/algo repo (or any passed directory)
const targetDir = process.argv[2] || path.resolve(import.meta.dirname, "../..");
const dbFile = path.resolve("./codebase_graph.sqlite");
for (const ext of ["", "-wal", "-shm"]) {
  try { fs.unlinkSync(dbFile + ext); } catch {}
}

const engine = new SQLiteContextEngine(dbFile);
const files = walkRepo(targetDir);
let totalChunks = 0;

for (const f of files) {
  const chunks = chunkSourceFile(f.full, f.rel);
  for (const c of chunks) {
    engine.storeMemory(c.indexContent, c.docId, c.chunkId);
    // Keep clean display content (without the dual-tokenized suffix) for readable output
    engine.chunkContent.set(c.chunkId, c.displayContent);
    totalChunks++;
  }
}

console.log("=".repeat(85));
console.log(`INDEXED CODEBASE: ${targetDir}`);
console.log(`Files: ${files.length} | Function/Logic Chunks: ${totalChunks} | Unique Symbols: ${engine.N_i.size}`);
console.log("=".repeat(85));

// 1. Multi-File Feature Discovery (Submodular Packing across files)
const q1 = "stretched exponential potential burstiness vocab stats C extension";
const t0 = performance.now();
const hits1 = engine.packContext(q1, { budget: 350 });
console.log(`\n[1. MULTI-FILE FEATURE TRACE] Query: "${q1}" (${(performance.now() - t0).toFixed(2)} ms)`);
for (const cid of hits1) {
  const m = engine.docStats.get(cid);
  console.log(`  -> ${cid.padEnd(42)} | L=${m.L} U=${m.U} (L/U=${m.l_over_u.toFixed(2)})`);
}

// 2. 1-Hop Call-Graph / Symbol Hopping (Bridge Expansion)
const q2 = "how does full_pipeline gate bridge activation";
const t1 = performance.now();
const hits2 = engine.fullPipeline(q2, { budget: 400, use_bridges: true });
console.log(`\n[2. 1-HOP CALL-GRAPH BRIDGE TRACE] Query: "${q2}" (${(performance.now() - t1).toFixed(2)} ms)`);
for (const cid of hits2) {
  console.log(`  -> ${cid}`);
}

// 3. Structural Pattern / Clone Recognition (Pass a code block to find parallel functions!)
const samplePattern = `
  def score_function(c_dict, q_dict, L, U, params):
      beta_i = beta_0 * (mu_i ** (-gamma_b))
      pot = 1.0 - math.exp(-beta_i * (c_eff ** 0.45))
`;
const t2 = performance.now();
const hits3 = engine.packContext(dualTokenizeCode(samplePattern), { budget: 400 });
console.log(`\n[3. STRUCTURAL PATTERN RECOGNITION] Finding parallel implementations across Python/JS/C (${(performance.now() - t2).toFixed(2)} ms):`);
for (const cid of hits3) {
  console.log(`  -> ${cid}`);
}
console.log("=".repeat(85));
