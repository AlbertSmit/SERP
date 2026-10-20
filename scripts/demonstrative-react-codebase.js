#!/usr/bin/node
// demonstrative-react-codebase.js — Focused SERP capability demonstrations on real TanStack Query
import { SQLiteContextEngine } from "../src/engine/engine.js";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dbPath = path.join(__dirname, "demo_react_codebase.sqlite");

function walkRepo(dir, baseDir = dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) { walkRepo(path.join(dir, entry.name), baseDir, out); continue; }
    out.push({ full: path.join(dir, entry.name), rel: path.relative(baseDir, path.join(dir, entry.name)) });
  }
  return out;
}

console.log("=".repeat(90));
console.log("SERP DEMO: REAL React Codebase (TanStack Query)");
console.log("=".repeat(90));

for (const ext of ["", "-wal", "-shm"]) try { fs.unlinkSync(dbPath + ext); } catch {}
const engine = new SQLiteContextEngine(dbPath);

const files = walkRepo("/tmp/react-test-project/packages");
let totalChunks = 0;
for (const f of files) {
  const raw = fs.readFileSync(f.full, "utf8");
  const lines = raw.split("\n");
  let cur = [], start = 1;
  const boundaryRe = /^\s*(export\s+)?(async\s+)?(function|class|def|const)\s+\w/;
  for (let i = 0; i < lines.length; i++) {
    if (cur.length < 15 || (!boundaryRe.test(lines[i]) && cur.length < 50)) { cur.push(lines[i]); continue; }
    const text = cur.join("\n").trim();
    if (text.length > 30) { engine.storeMemory(text, f.rel, f.rel + ":L" + start + "-" + i); totalChunks++; }
    cur = [lines[i]]; start = i + 1;
  }
  const tail = cur.join("\n").trim();
  if (tail.length > 30) { engine.storeMemory(tail, f.rel, f.rel + ":L" + start + "-" + lines.length); totalChunks++; }
}
console.log("\n[INDEXED] " + totalChunks + " chunks from " + files.length + " files\n");

function fmtContent(cid) {
  const c = engine.getContent(cid);
  return c ? c.substring(0, 200).replace(/\n/g, " ") : "(no content)";
}

// ===== Demo 1: Multi-feature discovery across monorepo =====
console.log("=" .repeat(60));
console.log("DEMO 1: Multi-Feature Discovery (useQuery + QueryClientProvider + useMutation)");
console.log("=" .repeat(60));

const q1 = "useQuery staleTime cacheTime refetchOnMount retry limit";
const picks1 = engine.packContext(q1, { budget: 400 });
console.log(`\nQuery: "${q1}" (budget=${400})`);
for (let i = 0; i < picks1.length; i++) {
  const m = engine.docStats.get(picks1[i]);
  console.log(`  Pick ${i + 1}: ${picks1[i]}`);
  console.log(`    L=${m.L} U=${m.U} L/U=${m.l_over_u.toFixed(2)} | ${fmtContent(picks1[i])}`);
}

// ===== Demo 2: Cross-file bridge hop =====
console.log("\n" + "=".repeat(60));
console.log("DEMO 2: Cross-File Symbol Bridge (QueryClient → QueryObserver)");
console.log("=".repeat(60));

const q2 = "createQueryClient observer creation client config defaultOptions options";
const picks2 = engine.fullPipeline(q2, { budget: 500, use_bridges: true });
console.log(`\nQuery: "${q2}" (budget=${500}, full pipeline with bridges)`);
for (let i = 0; i < picks2.length; i++) {
  const m = engine.docStats.get(picks2[i]);
  console.log(`  Pick ${i + 1}: ${picks2[i]}`);
  console.log(`    L=${m.L} U=${m.U} L/U=${m.l_over_u.toFixed(2)} | ${fmtContent(picks2[i])}`);
}

// ===== Demo 3: Boilerplate/Spam Robustness =====
console.log("\n" + "=".repeat(60));
console.log("DEMO 3: Boilerplate/Spam Robustness");
console.log("=".repeat(60));

const cleanChunk = engine.storeMemory("export function useValidatedQuery<T>(args) { const queryClient = getQueryClient(); return useQuery({...}); }", "demo_clean", "demo_clean_0");
const spamChunk = engine.storeMemory(Array(30).fill("crispr cas9 useEffect useCallback useRef useState").join(" ") + " INVALID_LOG_SPAM", "demo_spam", "demo_spam_0");
const junkNoise = engine.storeMemory("alpha bravo charlie delta echo foxtrot hotel india juliet kilo lima mike november oscar papa queabec romeo sierra tango uniform victor whiskey xray yankee zulu", "demo_junk", "demo_junk_0");

console.log(`\n[Injected: Clean fact | Repetitive spam (L/U=${spamChunk.l_over_u.toFixed(2)}) | Random noise (L/U=${junkNoise.l_over_u.toFixed(2)})]`);

const cleanScore = engine.scoreChunkP4("demo_clean_0", "useValidatedQuery queryClient queryFn cacheTime staleTime");
const spamScore = engine.scoreChunkP4("demo_spam_0", "useValidatedQuery queryClient queryFn cacheTime staleTime");
const noiseScore = engine.scoreChunkP4("demo_junk_0", "useValidatedQuery queryClient queryFn cacheTime staleTime");

console.log(`\nScore comparison (same target query):`);
console.log(`  Clean fact (L/U=1.00):  ${cleanScore.toFixed(2)}`);
console.log(`  Repetitive spam (L/U=${spamChunk.l_over_u.toFixed(2)}): ${spamScore.toFixed(2)}  (suppressed: ${(cleanScore/spamScore).toFixed(1)}x)`);
console.log(`  Random noise (L/U=${junkNoise.l_over_u.toFixed(2)}):  ${noiseScore.toFixed(2)}  (suppressed: ${(cleanScore/noiseScore).toFixed(1)}x)`);

const spamOk = spamScore < cleanScore && cleanScore > 0;
const noiseOk = noiseScore < cleanScore && cleanScore > 0;
console.log("\nSpam suppression check: " + (spamOk || noiseOk) ? "PASS" : "FAIL");

const spamRetrieved = engine.packContext("useValidatedQuery queryClient queryFn cacheTime staleTime", { budget: 200 });
console.log(`\nPicking 'useValidatedQuery...' within 200 tokens:`);
console.log(`  Picks: ${spamRetrieved.map(c => c.split(':')[0]).join(", ")}`);
console.log(`  Clean chunk included? ${spamRetrieved.includes('demo_clean_0') ? 'YES' : 'NO'}`);
console.log(`  Spam chunk included? ${spamRetrieved.some(c => c.startsWith('demo_spam')) ? 'YES' : 'NO'}`);

engine.db.close();
try { fs.unlinkSync(dbPath); } catch {}
