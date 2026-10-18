import { tokenize } from "../src/engine/tokenize.js";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SQLiteContextEngine } from "../src/engine/engine.js";

const __dirname = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SRC_DB = path.join(__dirname, "mcp_memory.sqlite");
const TEST_DB = path.join(__dirname, "stress_test_temp.sqlite");

for (const ext of ["", "-wal", "-shm"]) { try { fs.unlinkSync(TEST_DB + ext); } catch {} }
fs.copyFileSync(SRC_DB, TEST_DB);

console.log("=".repeat(80));
console.log("RED-TEAM & STRESS-TEST SUITE: SQLite P4 Submodular Context Engine");
console.log("=".repeat(80));

// TEST 1: Cold-Start & RAM
const memBefore = process.memoryUsage().heapUsed;
const tBoot0 = performance.now();
const engine = new SQLiteContextEngine(TEST_DB);
const bootMs = (performance.now() - tBoot0).toFixed(1);
const ramMB = ((process.memoryUsage().heapUsed - memBefore) / 1048576).toFixed(1);

console.log(`\n[TEST 1: COLD-START & MEMORY SCALING (${engine.M_ref} Chunks)]`);
console.log(`  Boot Time : ${bootMs} ms | V8 Heap: ${ramMB} MB`);
console.log(`  1M Chunks Projection: ~${((ramMB / engine.M_ref) * 1e6).toFixed(0)} MB RAM`);

// TEST 2: 200-Query Throughput
const queries = [
  "ADAR1 binds to Dicer to cleave pre-miRNA. AIRE is expressed in some skin tumors.",
  "myeloid-derived suppressor cells MDSC expansion S100A9 CD33 signaling",
  "CRISPR Cas9 genome editing electroporation rat zygotes knock-in",
  "chronic lymphocytic leukemia subclonal driver mutations SF3B1 TP53",
];
const lats = [];
for (let i = 0; i < 200; i++) {
  const t0 = performance.now();
  engine.packContext(queries[i % queries.length], { budget: 200 });
  lats.push(performance.now() - t0);
}
lats.sort((a, b) => a - b);
const qps = (1000 / (lats.reduce((a, b) => a + b, 0) / 200)).toFixed(0);
console.log(`\n[TEST 2: CPU THROUGHPUT (200 Queries @ B=200)]`);
console.log(`  P50: ${lats[100].toFixed(2)} ms | P95: ${lats[190].toFixed(2)} ms | Throughput: ${qps} QPS`);

// TEST 3: Word-Salad Spam Exploit
const clean = engine.storeMemory(
  "CRISPR and Cas9 ribonucleoproteins enable precise homology-directed repair in human T cells.",
  "clean_doc", "clean_s0"
);
const naive = engine.storeMemory(
  Array(15).fill("crispr cas9").join(" ") + " repair human cells",
  "naive_doc", "naive_s0"
);
const junk = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu amber bronze cobalt crimson emerald ivory jade onyx topaz";
const salad = engine.storeMemory(
  Array(5).fill("crispr cas9").join(" ") + " " + junk,
  "salad_doc", "salad_s0"
);

console.log(`\n[TEST 3: ADVERSARIAL SPAM INJECTION]`);
console.log(`  Clean Fact (L/U=${clean.l_over_u})      -> Score: ${engine.scoreChunkP4("clean_s0", "crispr cas9 repair").toFixed(3)}`);
console.log(`  Naive Spam (L/U=${naive.l_over_u})      -> Score: ${engine.scoreChunkP4("naive_s0", "crispr cas9 repair").toFixed(3)}`);
console.log(`  Word-Salad Spam (L/U=${salad.l_over_u}) -> Score: ${engine.scoreChunkP4("salad_s0", "crispr cas9 repair").toFixed(3)}`);

// TEST 4: FTS5 LIMIT-100 Starvation
for (let i = 0; i < 105; i++) {
  engine.storeMemory(`quantumentanglement2026 superconductor decoherence run ${i}`, `d_${i}`, `ds_${i}`);
}
engine.storeMemory("Critical threshold for raretargetbeta99 stability is 4.2 Kelvin.", "pb", "pb_s0");
const trapHits = engine.packContext("quantumentanglement2026 superconductor decoherence raretargetbeta99", { budget: 150 });
console.log(`\n[TEST 4: FTS5 'LIMIT 100' STARVATION TRAP]`);
console.log(`  Retrieved Part-B ('pb_s0') behind 105 Part-A decoys? -> ${trapHits.includes("pb_s0")}`);

// TEST 5: Synonym Wall
engine.storeMemory("Patients exhibited acute myocardial infarction following coronary occlusion.", "syn", "syn_s0");
const synHits = engine.packContext("heart attack blocked blood vessel", { budget: 120 });
console.log(`\n[TEST 5: LEXICAL SYNONYM WALL]`);
console.log(`  Query 'heart attack' found 'myocardial infarction'? -> ${synHits.includes("syn_s0")}`);

// TEST 6: Code Tokenizer
const codeQ = "How to fix C++ segfault in .env parser for R?";
console.log(`\n[TEST 6: CODE TOKENIZER BLINDNESS]`);
console.log(`  Input : "${codeQ}"`);
console.log(`  Tokens: ${JSON.stringify(tokenize(codeQ))}`);

// TEST 7: Temporal Surprisal Length Bias
const chat = engine.storeMemory("ok sounds good thanks bro", "chat", "chat_s0", 200);
const spec = engine.storeMemory(
  "Architectural Decision: Enforce mutual TLS authentication using rotating X509 certificates and validate JWT claims at ingress gateway.",
  "spec", "spec_s0", 200
);
console.log(`\n[TEST 7: TEMPORAL SURPRISAL LENGTH BIAS (Age = 200 Days)]`);
console.log(`  5-word trivial chat  -> surprisal=${chat.intrinsic_surprisal}, temporalFactor=${engine.temporalFactor("chat_s0").toFixed(4)}`);
console.log(`  18-word arch spec    -> surprisal=${spec.intrinsic_surprisal}, temporalFactor=${engine.temporalFactor("spec_s0").toFixed(4)}`);

fs.unlinkSync(TEST_DB);
console.log("\n" + "=".repeat(80));
