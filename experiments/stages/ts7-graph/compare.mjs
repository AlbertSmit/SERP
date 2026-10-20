// compare.mjs — diff the TS7 AST graph against the regex graph, and emit the
// deterministic audit samples used for the hand-checked precision numbers.
//
// Writes out/compare.json and out/audit_sample.json.

import { readFileSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = `${HERE}/out`
const REPO = '/Users/asmit/Personal/algo-serp-stages'

const ts = JSON.parse(readFileSync(`${OUT}/graph_ts7.json`, 'utf8'))
const rg = JSON.parse(readFileSync(`${OUT}/graph_regex.json`, 'utf8'))

const sourceLines = new Map()
const linesOf = (file) => {
  if (!sourceLines.has(file)) {
    sourceLines.set(file, readFileSync(`${REPO}/${file}`, 'utf8').split('\n'))
  }
  return sourceLines.get(file)
}
const lineText = (file, line) => (linesOf(file)[line - 1] ?? '').trim()

const key = (...parts) => parts.join('|')
const setOf = (arr, fn) => new Set(arr.map(fn))
const only = (a, b) => [...a].filter((x) => !b.has(x))
const inter = (a, b) => [...a].filter((x) => b.has(x))

function chunkReport(tsg, rgg) {
  const norm = (name) => name.replace(/^<anon.*>$/, '<anon>')
  const tk = setOf(tsg.chunks, (c) => key(c.file, norm(c.name)))
  const rk = setOf(rgg.chunks, (c) => key(c.file, norm(c.name)))
  const both = inter(tk, rk)
  const byKeyTs = new Map(tsg.chunks.map((c) => [key(c.file, norm(c.name)), c]))
  const byKeyRg = new Map(rgg.chunks.map((c) => [key(c.file, norm(c.name)), c]))

  // Script-body chunks are excluded from range accuracy: "which top-level
  // statements form the module body" is a definition, not a parse fact, and the
  // two extractors define it differently on purpose.
  let exactRange = 0
  let startLineMatch = 0
  const ious = []
  const kindMismatch = []
  let considered = 0
  let regexTruncated = 0
  for (const k of both) {
    const a = byKeyTs.get(k)
    const b = byKeyRg.get(k)
    if (a.kind === 'body' || b.kind === 'body') continue
    considered++
    if (a.start_line === b.start_line && a.end_line === b.end_line) exactRange++
    if (a.start_line === b.start_line) startLineMatch++
    if (b.end_line < a.end_line) regexTruncated++
    const lo = Math.max(a.start_line, b.start_line)
    const hi = Math.min(a.end_line, b.end_line)
    const interLen = Math.max(0, hi - lo + 1)
    const union = Math.max(a.end_line, b.end_line) - Math.min(a.start_line, b.start_line) + 1
    ious.push(interLen / union)
    if (a.kind !== b.kind) kindMismatch.push({ key: k, ts: a.kind, regex: b.kind })
  }
  const mean = (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0)

  const byKind = (list) => {
    const o = {}
    for (const c of list) o[c.kind] = (o[c.kind] || 0) + 1
    return o
  }

  return {
    counts: {
      ts7: tsg.chunks.length,
      regex: rgg.chunks.length,
      matched_by_file_and_name: both.length,
      ts7_only: tk.size - both.length,
      regex_only: rk.size - both.length,
    },
    by_kind: { ts7: byKind(tsg.chunks), regex: byKind(rgg.chunks) },
    range_accuracy_on_matched: {
      chunks_considered: considered,
      exact_start_and_end: exactRange,
      start_line_match: startLineMatch,
      regex_ends_early: regexTruncated,
      mean_line_iou: +mean(ious).toFixed(3),
    },
    kind_mismatch: kindMismatch.slice(0, 20),
    kind_mismatch_count: kindMismatch.length,
    ts7_only_sample: only(tk, rk).slice(0, 25),
    regex_only_sample: only(rk, tk).slice(0, 25),
  }
}

function callReport(tsg, rgg) {
  // Compare at the level a consumer cares about: which caller calls which callee.
  // The regex graph only keeps the final name segment (`getContent`), the AST graph
  // keeps the receiver too (`engine.getContent`), so sites are compared on the last
  // segment and qualified-name availability is reported separately.
  const last = (callee) => (callee.includes('.') ? callee.split('.').pop() : callee).replace('[]', '')
  const pairOf = (e, chunkName) => key(chunkName, last(e.callee))
  const chunkName = (g) => new Map(g.chunks.map((c) => [c.id, c.name]))
  const tn = chunkName(tsg)
  const rn = chunkName(rgg)

  const tsPairs = setOf(tsg.relations.CALLS, (e) => pairOf(e, tn.get(e.chunk)))
  const rgPairs = setOf(rgg.relations.CALLS, (e) => pairOf(e, rn.get(e.chunk)))

  const tsCallees = setOf(tsg.relations.CALLS, (e) => last(e.callee))
  const rgCallees = setOf(rgg.relations.CALLS, (e) => last(e.callee))

  // Sites: same file+line+callee. Line numbers are directly comparable.
  const tsSites = setOf(tsg.relations.CALLS, (e) => key(e.file, e.line, last(e.callee)))
  const rgSites = setOf(rgg.relations.CALLS, (e) => key(e.file, e.line, last(e.callee)))
  const tsSitesQ = setOf(tsg.relations.CALLS, (e) => key(e.file, e.line, e.callee))

  return {
    edges: {
      ts7: tsg.relations.CALLS.length,
      regex: rgg.relations.CALLS.length,
    },
    distinct_callee_names: { ts7: tsCallees.size, regex: rgCallees.size },
    qualified_callee_names: {
      ts7_with_receiver: tsg.relations.CALLS.filter((e) => e.callee.includes('.')).length,
      regex_with_receiver: 0,
    },
    caller_callee_pairs: {
      ts7: tsPairs.size,
      regex: rgPairs.size,
      shared: inter(tsPairs, rgPairs).length,
    },
    sites_by_file_line_callee: {
      ts7: tsSites.size,
      regex: rgSites.size,
      shared: inter(tsSites, rgSites).length,
      ts7_only: only(tsSites, rgSites).length,
      regex_only: only(rgSites, tsSites).length,
    },
    resolution: {
      ts7_to_declaration: tsg.relations.CALLS.filter((e) => e.resolved === 'declaration').length,
      ts7_symbol_only: tsg.relations.CALLS.filter((e) => e.resolved === 'symbol_only').length,
      ts7_unresolved: tsg.relations.CALLS.filter((e) => e.resolved === 'unresolved').length,
      regex: 'none (syntactic only)',
    },
    regex_only_sample: only(rgSites, tsSites).slice(0, 20),
    ts7_only_sample: only(tsSites, rgSites).slice(0, 20),
    ts7_qualified_sites: tsSitesQ.size,
    regex_only_classification: classifySites(only(rgSites, tsSites)),
    ts7_only_classification: classifySites(only(tsSites, rgSites)),
  }
}

// Ranges of comments and string bodies, so a call site can be classified as
// "found inside prose or SQL" rather than guessed at from the line text.
const maskCache = new Map()
function nonCodeRanges(file) {
  if (maskCache.has(file)) return maskCache.get(file)
  const text = linesOf(file).join('\n')
  const ranges = []
  let i = 0
  while (i < text.length) {
    const ch = text[i]
    const next = text[i + 1]
    if (ch === '/' && next === '/') {
      const start = i
      while (i < text.length && text[i] !== '\n') i++
      ranges.push({ start, end: i, type: 'comment' })
      continue
    }
    if (ch === '/' && next === '*') {
      const start = i
      i += 2
      while (i < text.length && !(text[i] === '*' && text[i + 1] === '/')) i++
      i += 2
      ranges.push({ start, end: i, type: 'comment' })
      continue
    }
    if (ch === '"' || ch === "'" || ch === '`') {
      const quote = ch
      const start = i
      i++
      while (i < text.length && text[i] !== quote) {
        if (text[i] === '\\') i++
        i++
      }
      i++
      ranges.push({ start, end: i, type: 'string' })
      continue
    }
    i++
  }
  maskCache.set(file, { text, ranges })
  return { text, ranges }
}

function classifySites(sites) {
  const buckets = { comment: 0, string_literal: 0, declaration: 0, non_call_expression: 0, other: 0 }
  for (const site of sites) {
    const [file, lineNo, callee] = site.split('|')
    const line = Number(lineNo)
    const { text, ranges } = nonCodeRanges(file)
    const lineStart = linesOf(file).slice(0, line - 1).join('\n').length + (line > 1 ? 1 : 0)
    const lineTextRaw = linesOf(file)[line - 1] ?? ''
    const escaped = callee.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const m = new RegExp(`\\b${escaped}\\s*\\(`).exec(lineTextRaw)
    const pos = m ? lineStart + m.index : -1
    const inRange = pos === -1 ? null : ranges.find((r) => r.start <= pos && pos < r.end)
    if (inRange) {
      buckets[inRange.type === 'comment' ? 'comment' : 'string_literal']++
      continue
    }
    const trimmed = lineTextRaw.trim()
    if (/^(export\s+)?(async\s+)?function\b/.test(trimmed) || /^(static\s+)?(async\s+)?[A-Za-z_$][\w$]*\s*\([^)]*\)\s*\{$/.test(trimmed)) {
      buckets.declaration++
      continue
    }
    // a call the AST did not see at all: property/method name in a position that is
    // not a CallExpression callee (e.g. inside a template expression already counted)
    if (/\bnew\s/.test(trimmed) || trimmed.includes('](')) {
      buckets.non_call_expression++
      continue
    }
    buckets.other++
  }
  return buckets
}

function importReport(tsg, rgg) {
  const tsSet = setOf(tsg.relations.IMPORTS, (e) => key(e.file, e.module))
  const rgSet = setOf(rgg.relations.IMPORTS, (e) => key(e.file, e.module))
  return {
    edges: { ts7: tsg.relations.IMPORTS.length, regex: rgg.relations.IMPORTS.length },
    distinct_file_module: { ts7: tsSet.size, regex: rgSet.size, shared: inter(tsSet, rgSet).length },
    ts7_only: only(tsSet, rgSet),
    regex_only: only(rgSet, tsSet),
    named_binding_exact: {
      ts7: tsg.relations.IMPORTS.filter((e) => e.names.length > 0).length,
      regex: rgg.relations.IMPORTS.filter((e) => e.names.length > 0).length,
    },
  }
}

// ------------------------------------------------------------------ audit sample

function auditSample(graph, label) {
  const calls = [...graph.relations.CALLS].sort((a, b) =>
    a.file === b.file ? a.pos - b.pos : a.file < b.file ? -1 : 1
  )
  const picks = []
  if (calls.length) {
    const stride = calls.length / 20
    for (let i = 0; i < 20; i++) {
      const e = calls[Math.floor(i * stride + stride / 2) % calls.length]
      picks.push({ file: e.file, line: e.line, callee: e.callee, source: lineText(e.file, e.line) })
    }
  }
  const importsSorted = [...graph.relations.IMPORTS].sort((a, b) =>
    a.file === b.file ? a.line - b.line : a.file < b.file ? -1 : 1
  )
  const ipicks = []
  if (importsSorted.length) {
    const stride = importsSorted.length / 20
    for (let i = 0; i < 20; i++) {
      const e = importsSorted[Math.floor(i * stride + stride / 2) % importsSorted.length]
      ipicks.push({
        file: e.file,
        line: e.line,
        module: e.module,
        names: e.names,
        source: lineText(e.file, e.line),
      })
    }
  }
  return { label, calls: picks, imports: ipicks }
}

const result = {
  meta: {
    ts7: ts.meta,
    regex: rg.meta,
    compared_at: new Date().toISOString(),
  },
  chunks: chunkReport(ts, rg),
  calls: callReport(ts, rg),
  imports: importReport(ts, rg),
}

writeFileSync(`${OUT}/compare.json`, JSON.stringify(result, null, 1))
writeFileSync(
  `${OUT}/audit_sample_ts7.json`,
  JSON.stringify(auditSample(ts, 'ts7'), null, 1)
)
writeFileSync(
  `${OUT}/audit_sample_regex.json`,
  JSON.stringify(auditSample(rg, 'regex'), null, 1)
)

console.log(JSON.stringify(result, null, 1))
