const DAY_MS = 24 * 60 * 60 * 1000;

export function summarize(events, { days, now = Date.now() } = {}) {
  const since = days ? now - days * DAY_MS : 0;
  const recent = events.filter((x) => Date.parse(x.ts) >= since);

  return {
    period: period(recent),
    memory: memoryStats(byTool(recent, "store_memory"), byTool(recent, "pack_context")),
    output: outputStats(byTool(recent, "index_output"), byTool(recent, "search_output")),
    errors: recent.filter((x) => x.error).length,
  };
}

function byTool(events, tool) {
  return events.filter((x) => x.tool === tool);
}

function period(events) {
  const times = events.map((x) => x.ts).sort();

  return {
    from: times[0] ?? null,
    to: times.at(-1) ?? null,
    sessions: new Set(events.map((x) => x.session)).size,
    calls: events.length,
  };
}

function memoryStats(stores, recalls) {
  return {
    stored: stores.filter((x) => !x.error).length,
    recalls: recalls.length,
    empty_recalls: recalls.filter((x) => x.results === 0).length,
    empty_rate: rate(recalls.filter((x) => x.results === 0).length, recalls.length),
    tokens_returned: sum(recalls, "tokens_returned"),
    latency_ms: percentiles(recalls),
  };
}

// Savings compare what the agent received from the scratch index (receipts
// plus search results) with the raw size of everything it indexed.
function outputStats(indexed, searches) {
  const ok = indexed.filter((x) => !x.error);
  const raw = sum(ok, "raw_tokens");
  const received = sum(ok, "receipt_tokens") + sum(searches, "tokens_returned");

  return {
    indexed: ok.length,
    raw_tokens: raw,
    searches: searches.length,
    searches_per_index: ok.length ? round(searches.length / ok.length) : 0,
    empty_searches: searches.filter((x) => x.results === 0).length,
    empty_rate: rate(searches.filter((x) => x.results === 0).length, searches.length),
    tokens_received: received,
    token_savings: raw ? rate(raw - received, raw) : null,
    latency_ms: percentiles(searches),
  };
}

function percentiles(events) {
  const values = events.map((x) => x.latency_ms).filter(Number.isFinite).sort((a, b) => a - b);
  if (values.length === 0) return { p50: null, p95: null };

  return {
    p50: values[Math.floor(values.length * 0.5)],
    p95: values[Math.min(values.length - 1, Math.floor(values.length * 0.95))],
  };
}

function sum(events, key) {
  return events.reduce((total, x) => total + (x[key] || 0), 0);
}

function rate(part, whole) {
  return whole ? round(part / whole) : null;
}

function round(value) {
  return Math.round(value * 1000) / 1000;
}
