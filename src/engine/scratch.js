import { tokenize } from "./tokenize.js";

const MAX_CHUNK_TOKENS = 64;
const HEAD_LINES = 5;
const HEAD_LINE_CHARS = 160;

export function chunkText(text, maxTokens = MAX_CHUNK_TOKENS) {
  const lines = text.split(/\r?\n/).flatMap((line) => (
    tokenize(line).length > maxTokens ? joinWithin(line.split(" "), " ", maxTokens) : [line]
  ));

  return joinWithin(lines, "\n", maxTokens).filter((chunk) => tokenize(chunk).length > 0);
}

export function indexText(engine, text, { source, label = "" } = {}) {
  const id = source || nextSourceId(engine);
  if (engine.docStats.has(chunkIdFor(id, 0))) throw new Error(`Source already indexed: ${id}`);
  const chunks = chunkText(text);
  if (chunks.length === 0) throw new Error("No indexable tokens in content");
  chunks.forEach((chunk, index) => engine.storeMemory(chunk, id, chunkIdFor(id, index)));

  return {
    source: id,
    label,
    raw_tokens: tokenize(text).length,
    raw_chars: text.length,
    chunks: chunks.length,
    head: headLines(text),
  };
}

export function searchText(engine, query, { budget = 250, source } = {}) {
  const { fused } = engine.retrieveRanked(query);
  // ponytail: source filter runs after the top-100 fused pool; filter inside the FTS query if sources get crowded out
  const pool = source ? fused.filter(({ chunk_id: id }) => engine.chunkDocId.get(id) === source) : fused;
  const { chunkIds, usedTokens } = engine.packRankedDetailed(query, pool, { budget });

  return {
    query,
    budget_limit: budget,
    source: source || null,
    tokens_returned: usedTokens,
    chunks_count: chunkIds.length,
    results: chunkIds.map((chunkId, index) => ({
      rank: index + 1,
      chunk_id: chunkId,
      source: engine.chunkDocId.get(chunkId),
      tokens_L: engine.docStats.get(chunkId).L,
      content: engine.getContent(chunkId),
    })),
  };
}

// ponytail: a whitespace-free run above maxTokens stays one oversized chunk; the packer skips it when it exceeds the budget
function joinWithin(pieces, separator, maxTokens) {
  const chunks = [];
  let current = [];
  let tokens = 0;
  for (const piece of pieces) {
    const count = tokenize(piece).length;
    if (current.length > 0 && tokens + count > maxTokens) {
      chunks.push(current.join(separator));
      current = [];
      tokens = 0;
    }
    current.push(piece);
    tokens += count;
  }
  if (current.length > 0) chunks.push(current.join(separator));

  return chunks;
}

function headLines(text) {
  return text.split(/\r?\n/)
    .filter((line) => line.trim() !== "")
    .slice(0, HEAD_LINES)
    .map((line) => line.slice(0, HEAD_LINE_CHARS));
}

function nextSourceId(engine) {
  let n = 1;
  while (engine.docStats.has(chunkIdFor(`out_${n}`, 0))) n++;

  return `out_${n}`;
}

function chunkIdFor(source, index) {
  return `${source}#${String(index + 1).padStart(4, "0")}`;
}
