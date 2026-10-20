// graph_regex.mjs — the reference "what a regex extractor gets you" graph.
//
// Deliberately competent rather than a strawman: it handles `export`, `async`,
// classes, methods, const-arrow functions, brace-matched line ranges, ESM imports
// and re-exports. It is still regex + brace counting, which is the thing under test.
//
// Writes out/graph_regex.json with the same schema as graph_ts7.json.

import { readFileSync, readdirSync, statSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = '/Users/asmit/Personal/algo-serp-stages'
const OUT = `${HERE}/out`
const ROOTS = ['src', 'scripts', 'test']

const KEYWORDS = new Set([
  'if', 'for', 'while', 'switch', 'catch', 'function', 'return', 'typeof', 'instanceof',
  'new', 'delete', 'void', 'await', 'else', 'do', 'try', 'finally', 'throw', 'case',
  'in', 'of', 'super', 'import', 'export', 'yield', 'constructor', 'async',
])

function listJs(root) {
  const out = []
  const walk = (dir) => {
    for (const entry of readdirSync(dir)) {
      const p = join(dir, entry)
      const st = statSync(p)
      if (st.isDirectory()) walk(p)
      else if (entry.endsWith('.js')) out.push(p)
    }
  }
  walk(join(REPO, root))
  return out
}

function lineOf(text, pos) {
  let line = 1
  for (let i = 0; i < pos && i < text.length; i++) if (text.charCodeAt(i) === 10) line++
  return line
}

function lineStarts(text) {
  const starts = [0]
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) starts.push(i + 1)
  return starts
}

// Naive brace matching, skipping strings and line comments well enough for JS.
function matchBrace(text, from) {
  let depth = 0
  let started = false
  for (let i = from; i < text.length; i++) {
    const ch = text[i]
    if (ch === '"' || ch === "'" || ch === '`') {
      const quote = ch
      i++
      while (i < text.length && text[i] !== quote) {
        if (text[i] === '\\') i++
        i++
      }
      continue
    }
    if (ch === '/' && text[i + 1] === '/') {
      while (i < text.length && text[i] !== '\n') i++
      continue
    }
    if (ch === '/' && text[i + 1] === '*') {
      i += 2
      while (i < text.length && !(text[i] === '*' && text[i + 1] === '/')) i++
      i++
      continue
    }
    if (ch === '{') {
      depth++
      started = true
    } else if (ch === '}') {
      depth--
      if (started && depth === 0) return i + 1
    }
  }
  return text.length
}

function build() {
  const t0 = performance.now()
  const files = ROOTS.flatMap(listJs).map((p) => relative(REPO, p)).sort()

  const chunks = []
  const symbols = []
  const defines = []
  const imports = []
  const contains = []
  const calls = []

  let bytesRead = 0

  for (const file of files) {
    const text = readFileSync(join(REPO, file), 'utf8')
    bytesRead += text.length
    const starts = lineStarts(text)
    const lineAt = (pos) => {
      let lo = 0
      let hi = starts.length - 1
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1
        if (starts[mid] <= pos) lo = mid
        else hi = mid - 1
      }
      return lo + 1
    }
    const fileChunks = []
    const add = (kind, name, start, end, exported) => {
      const chunk = {
        id: `${file}#${kind}:${name}@${lineAt(start)}`,
        kind,
        name,
        file,
        start_line: lineAt(start),
        end_line: lineAt(end),
        start_pos: start,
        end_pos: end,
        parent: null,
        exported: !!exported,
        bytes: end - start,
      }
      chunks.push(chunk)
      fileChunks.push(chunk)
      contains.push({ from: `file:${file}`, to: chunk.id, relation: 'CONTAINS' })
      symbols.push({ id: `sym:${chunk.id}`, name, kind, file, line: chunk.start_line })
      defines.push({ chunk: chunk.id, symbol: `sym:${chunk.id}`, name })
      return chunk
    }

    // classes first, so methods can be attributed to the enclosing class text range
    const classes = []
    for (const m of text.matchAll(/^[ \t]*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)/gm)) {
      const start = m.index + m[0].indexOf('class')
      const end = matchBrace(text, start)
      const c = add('class', m[1], start, end, /\bexport\b/.test(m[0]))
      classes.push({ name: m[1], start, end, chunk: c })
    }

    const fns = []
    for (const m of text.matchAll(
      /^[ \t]*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(/gm
    )) {
      const start = m.index + m[0].indexOf('function')
      const end = matchBrace(text, start)
      fns.push(add('function', m[1], start, end, /\bexport\b/.test(m[0])))
    }

    // const f = (...) => {      and      const f = function (...) {
    for (const m of text.matchAll(
      /^[ \t]*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\(?[\w$,\s{}[\]=]*\)?\s*=>)/gm
    )) {
      const start = m.index + m[0].indexOf('const')
      const end = matchBrace(text, start)
      fns.push(add('function', m[1], start, end, /\bexport\b/.test(m[0])))
    }

    // instance methods: 2-space indented, inside a class range, name(args) {
    for (const m of text.matchAll(/^[ \t]{2,}(?:static\s+)?(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^;]*\)\s*\{/gm)) {
      const start = m.index + m[0].indexOf(m[1])
      const owner = classes.find((c) => c.start < start && start < c.end)
      if (!owner) continue
      const end = matchBrace(text, start)
      add('method', `${owner.name}.${m[1]}`, start, end, false)
    }

    // script body: column-0 lines that no declaration chunk covers
    const covered = (pos) => fileChunks.some((c) => c.start_pos <= pos && pos < c.end_pos)
    const uncovered = starts.filter((s) => !covered(s) && /\S/.test(text.slice(s, s + 200).split('\n')[0] ?? ''))
    if (uncovered.length) {
      const first = uncovered[0]
      const lastLineStart = uncovered[uncovered.length - 1]
      let lastEnd = text.indexOf('\n', lastLineStart)
      if (lastEnd === -1) lastEnd = text.length
      add('body', '<module>', first, lastEnd, false)
    }

    // imports
    for (const m of text.matchAll(
      /^[ \t]*import\s+(?:([A-Za-z_$][\w$]*)\s*,?\s*)?(?:\{([^}]*)\}|\*\s+as\s+([A-Za-z_$][\w$]*))?\s*from\s*['"]([^'"]+)['"]/gm
    )) {
      const names = []
      if (m[1]) names.push(m[1])
      if (m[2]) for (const part of m[2].split(',')) {
        const n = part.trim().split(/\s+as\s+/).pop().trim()
        if (n) names.push(n)
      }
      if (m[3]) names.push(`* as ${m[3]}`)
      imports.push({ file, module: m[4], names, reexport: false, line: lineAt(m.index), pos: m.index })
    }
    for (const m of text.matchAll(/^[ \t]*export\s+\{[^}]*\}\s*from\s*['"]([^'"]+)['"]/gm)) {
      imports.push({ file, module: m[1], names: [], reexport: true, line: lineAt(m.index), pos: m.index })
    }

    // calls: name( or obj.name(  (the regex way: no scope, no types)
    for (const m of text.matchAll(/(?:\.\s*)?\b([A-Za-z_$][\w$]*)\s*\(/g)) {
      const name = m[1]
      if (KEYWORDS.has(name)) continue
      const pos = m.index + m[0].indexOf(name)
      calls.push({ file, pos, line: lineAt(pos), callee: name })
    }
  }

  // attribute calls to the innermost chunk that contains them
  for (const c of calls) {
    let best = null
    for (const chunk of chunks) {
      if (chunk.file !== c.file) continue
      if (chunk.start_pos <= c.pos && c.pos < chunk.end_pos) {
        if (!best || chunk.end_pos - chunk.start_pos < best.end_pos - best.start_pos) best = chunk
      }
    }
    if (!best) continue
    c.chunk = best.id
    c.target_chunk = null
    c.resolved = 'syntactic'
  }

  const elapsed = +(performance.now() - t0).toFixed(1)
  chunks.sort((a, b) => (a.file === b.file ? a.start_pos - b.start_pos : a.file < b.file ? -1 : 1))

  return {
    meta: {
      generator: 'graph_regex.mjs',
      engine: 'regex + brace counting (node 26)',
      repo: REPO,
      files: files.length,
      bytes_read: bytesRead,
      checker_used: false,
      call_sites: calls.length,
      nodes_walked: null,
      timings: { total_build_ms: elapsed },
    },
    files,
    chunks,
    symbols,
    relations: {
      DEFINES: defines,
      CALLS: calls.filter((c) => c.chunk),
      IMPORTS: imports,
      CONTAINS: contains,
    },
  }
}

mkdirSync(OUT, { recursive: true })
const g = build()
writeFileSync(`${OUT}/graph_regex.json`, JSON.stringify(g, null, 1))
console.log(JSON.stringify(g.meta, null, 1))
console.log(
  'chunks',
  g.chunks.length,
  '| symbols',
  g.symbols.length,
  '| CALLS',
  g.relations.CALLS.length,
  '| IMPORTS',
  g.relations.IMPORTS.length
)
