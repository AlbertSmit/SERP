// ts7_graph.mjs — build a code graph for the serp repo using the TypeScript 7
// native compiler's unstable sync API (typescript@7.0.2, unstable/sync + unstable/ast).
//
// Writes out/graph_ts7.json. The schema is shared with graph_regex.mjs so the two
// graphs can be diffed field by field.
//
// Read-only over ../../src, ../../scripts, ../../test. Writes only inside out/.

import { API, SymbolFlags } from 'typescript/unstable/sync'
import {
  SyntaxKind,
  isArrowFunction,
  isCallExpression,
  isClassDeclaration,
  isConstructorDeclaration,
  isExportDeclaration,
  isFunctionDeclaration,
  isFunctionExpression,
  isIdentifier,
  isImportDeclaration,
  isMethodDeclaration,
  isNewExpression,
  isPropertyAccessExpression,
  isVariableStatement,
} from 'typescript/unstable/ast'
import { writeFileSync, mkdirSync } from 'node:fs'
import { dirname, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = '/Users/asmit/Personal/algo-serp-stages'
const OUT = `${HERE}/out`

const rel = (p) => relative(REPO, p).split('\\').join('/')

function hasExportModifier(node) {
  const mods = node.modifiers
  if (!mods) return false
  for (const m of mods) if (m.kind === SyntaxKind.ExportKeyword) return true
  return false
}

// `a.b()` -> "a.b"; `f()` -> "f"; `this.m()` -> "this.m"; `a?.b.c()` -> "a.b.c"
function calleeName(call) {
  return exprName(call.expression)
}

function exprName(e) {
  if (!e) return null
  if (isIdentifier(e)) return e.text
  if (isPropertyAccessExpression(e)) {
    const obj = exprName(e.expression)
    const prop = e.name && e.name.text ? e.name.text : '?'
    return `${obj || '?'}.${prop}`
  }
  if (e.kind === SyntaxKind.ElementAccessExpression) return `${exprName(e.expression) || '?'}[]`
  if (e.kind === SyntaxKind.ThisKeyword) return 'this'
  return `<${SyntaxKind[e.kind]}>`
}

// Position of the member name token of a callee expression, falling back to the
// expression start. Makes site line numbers point at `bar` in `foo.bar()`.
function nameTokenPos(e, fallback) {
  if (!e) return fallback
  if (isPropertyAccessExpression(e) && e.name && typeof e.name.getStart === 'function') {
    return e.name.getStart()
  }
  return fallback
}

function buildGraph() {
  const timings = {}
  const t0 = performance.now()

  const chunks = []
  const symbols = []
  const defines = []
  const imports = []
  const contains = []
  const callEdges = []
  const orphanCalls = []
  let nodesWalked = 0
  let callSites = 0
  let checkerCalls = 0
  let checkerFailures = 0

  // TS7_NO_CHECKER=1 skips all checker round-trips (syntactic graph only), which
  // separates parser cost from type-checker cost.
  const useChecker = process.env.TS7_NO_CHECKER !== '1'

  const api = new API({ collectTiming: true })
  const snapshot = api.updateSnapshot({ openProjects: [`${HERE}/tsconfig.json`] })
  timings.spawn_and_snapshot_ms = +(performance.now() - t0).toFixed(1)

  const project = snapshot.getProjects()[0]
  if (!project) throw new Error('no project loaded')
  const program = project.program
  const checker = project.checker

  // Only repo files. The program also contains lib.d.ts and node_modules, which do
  // not belong in a code graph of this repo.
  const files = project.rootFiles.map(rel).sort()

  const chunkId = (file, kind, name, line) => `${file}#${kind}:${name}@${line}`

  const tParse0 = performance.now()
  const sourceFiles = new Map()
  const lineStartIndex = new Map()
  for (const f of files) {
    const sf = program.getSourceFile(`${REPO}/${f}`)
    if (!sf) throw new Error(`getSourceFile returned undefined for ${f}`)
    sourceFiles.set(f, sf)
    lineStartIndex.set(f, sf.getLineStarts())
  }
  timings.fetch_and_parse_ms = +(performance.now() - tParse0).toFixed(1)

  const lineOf = (file, pos) => {
    const lineStarts = lineStartIndex.get(file)
    let lo = 0
    let hi = lineStarts.length - 1
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if (lineStartIndex.get(file)[mid] <= pos) lo = mid
      else hi = mid - 1
    }
    return lo + 1
  }

  const byFile = new Map()
  for (const f of files) byFile.set(f, [])
  const fileIndex = new Map()

  const addChunk = (file, kind, name, start, end, parent, exported) => {
    const chunk = {
      id: chunkId(file, kind, name, lineOf(file, start)),
      kind,
      name,
      file,
      start_line: lineOf(file, start),
      end_line: lineOf(file, end),
      start_pos: start,
      end_pos: end,
      parent: parent ? parent.id : null,
      exported: !!exported,
      bytes: end - start,
    }
    chunks.push(chunk)
    byFile.get(file).push(chunk)
    contains.push({ from: `file:${file}`, to: chunk.id, relation: 'CONTAINS' })
    symbols.push({ id: `sym:${chunk.id}`, name, kind, file, line: chunk.start_line })
    defines.push({ chunk: chunk.id, symbol: `sym:${chunk.id}`, name })
    return chunk
  }

  // Innermost chunk containing a position.
  const enclosing = (pos, list) => {
    let best = null
    for (const c of list) {
      if (c.start_pos <= pos && pos < c.end_pos) {
        if (!best || c.end_pos - c.start_pos < best.end_pos - best.start_pos) best = c
      }
    }
    return best
  }

  const tWalk0 = performance.now()

  // ---------------------------------------------------------- phase A: declarations
  for (const [file, sf] of sourceFiles) {
    const declInfo = { citizenChunks: [] }
    fileIndex.set(file, declInfo)

    const declareTopLevel = (node, parent) => {
      if (isFunctionDeclaration(node)) {
        const name = node.name ? node.name.text : `<anon@${lineOf(file, node.getStart(sf))}>`
        const c = addChunk(
          file,
          'function',
          name,
          node.getStart(sf),
          node.getEnd(),
          parent,
          hasExportModifier(node)
        )
        declInfo.citizenChunks.push(c)
        return c
      }
      if (isClassDeclaration(node)) {
        const name = node.name ? node.name.text : `<anon@${lineOf(file, node.getStart(sf))}>`
        const c = addChunk(
          file,
          'class',
          name,
          node.getStart(sf),
          node.getEnd(),
          parent,
          hasExportModifier(node)
        )
        declInfo.citizenChunks.push(c)
        for (const m of node.members) {
          if (isMethodDeclaration(m) || isConstructorDeclaration(m)) {
            const mname =
              m.kind === SyntaxKind.Constructor
                ? 'constructor'
                : m.name && m.name.text
                  ? m.name.text
                  : '<computed>'
            const mc = addChunk(
              file,
              'method',
              `${name}.${mname}`,
              m.getStart(sf),
              m.getEnd(),
              c,
              hasExportModifier(m)
            )
            declInfo.citizenChunks.push(mc)
          }
        }
        return c
      }
      if (isVariableStatement(node)) {
        // `const f = () => {}` is a function to a human and to a regex extractor,
        // so it is one here too.
        for (const d of node.declarationList.declarations) {
          if (!d.initializer) continue
          if (isArrowFunction(d.initializer) || isFunctionExpression(d.initializer)) {
            const name = d.name && d.name.text ? d.name.text : '<anon>'
            const c = addChunk(
              file,
              'function',
              name,
              d.initializer.getStart(sf),
              d.initializer.getEnd(),
              parent,
              hasExportModifier(node)
            )
            declInfo.citizenChunks.push(c)
          }
        }
        return null
      }
      return null
    }

    // nested declarations (functions inside functions) are chunks too
    const walkDecls = (node, current) => {
      nodesWalked++
      const decl = declareTopLevel(node, current)
      const next = decl || current
      node.forEachChild((child) => walkDecls(child, next))
    }
    walkDecls(sf, null)
  }

  // ------------------------------------------------- phase B: calls and imports
  const tResolve0 = performance.now()
  let resolveMs = 0

  for (const [file, sf] of sourceFiles) {
    const declInfo = fileIndex.get(file)

    const visit = (node, current) => {
      const decl =
        isFunctionDeclaration(node) || isClassDeclaration(node) || isVariableStatement(node)
          ? enclosing(node.getStart(sf), byFile.get(file))
          : null
      const next = decl || current

      if (isCallExpression(node) || isNewExpression(node)) {
        callSites++
        const isNew = node.kind === SyntaxKind.NewExpression
        // Site position is the callee NAME token, not the start of the whole
        // expression: for `a.long().chain()` the latter is the line of `a`.
        const sitePos = isNew
          ? node.getStart(sf)
          : nameTokenPos(node.expression, node.getStart(sf))
        const callChunk = current || enclosing(sitePos, byFile.get(file))
        const callee = calleeName(node)
        if (callee) {
          const tR = performance.now()
          let targetChunk = null
          let targetSymbol = null
          let resolved = 'unresolved'
          try {
            const expr =
              isIdentifier(node.expression) || isPropertyAccessExpression(node.expression)
                ? node.expression
                : null
            if (expr && useChecker) {
              checkerCalls++
              let sym = checker.getSymbolAtLocation(expr)
              if (sym) {
                if (sym.flags & SymbolFlags.Alias) {
                  checkerCalls++
                  sym = checker.getAliasedSymbol(sym)
                }
                targetSymbol = sym.name
                const handle = sym.valueDeclaration || (sym.declarations && sym.declarations[0])
                if (handle) {
                  const dn = handle.resolve(project)
                  if (dn) {
                    const dfile = rel(dn.getSourceFile().fileName)
                    const list = byFile.get(dfile)
                    if (list) {
                      const c = enclosing(dn.getStart(), list)
                      if (c) targetChunk = c.id
                    }
                  }
                }
                resolved = targetChunk ? 'declaration' : 'symbol_only'
              }
            }
          } catch (e) {
            checkerFailures++
            resolved = 'checker_error'
          }
          resolveMs += performance.now() - tR
          const edge = {
            chunk: callChunk ? callChunk.id : null,
            file,
            pos: sitePos,
            line: lineOf(file, sitePos),
            kind: isNew ? 'new' : 'call',
            callee,
            callee_symbol: targetSymbol,
            target_chunk: targetChunk,
            resolved,
          }
          if (callChunk) callEdges.push(edge)
          else orphanCalls.push(edge)
        }
      }

      if (isImportDeclaration(node)) {
        const spec =
          node.moduleSpecifier && node.moduleSpecifier.text ? node.moduleSpecifier.text : null
        if (spec) {
          const names = []
          const clause = node.importClause
          if (clause) {
            if (clause.name) names.push(clause.name.text)
            const nb = clause.namedBindings
            if (nb) {
              if (nb.kind === SyntaxKind.NamespaceImport) names.push(`* as ${nb.name.text}`)
              else if (nb.elements) for (const e of nb.elements) names.push(e.name.text)
            }
          }
          imports.push({
            file,
            module: spec,
            names,
            reexport: false,
            line: lineOf(file, node.getStart(sf)),
          })
        }
      }

      if (isExportDeclaration(node)) {
        const spec = node.moduleSpecifier ? node.moduleSpecifier.text : null
        if (spec)
          imports.push({
            file,
            module: spec,
            names: [],
            reexport: true,
            line: lineOf(file, node.getStart(sf)),
          })
      }

      node.forEachChild((child) => visit(child, next))
    }
    // Calls are attributed to the innermost enclosing declaration chunk; when the
    // walk descends we recompute from position, so a nested arrow does not steal
    // its parent's calls.
    visit(sf, null)

    // ---------------------------------------------------- phase C: script body
    // Contiguous top-level statements that no declaration chunk covers become one
    // "body" chunk, so a script's main code is a node instead of being invisible.
    let runStart = null
    let runEnd = null
    const flush = () => {
      if (runStart === null) return
      const bodyChunk = addChunk(file, 'body', '<module>', runStart, runEnd, null, false)
      for (let i = orphanCalls.length - 1; i >= 0; i--) {
        const c = orphanCalls[i]
        if (c.file === file && c.pos >= bodyChunk.start_pos && c.pos < bodyChunk.end_pos) {
          c.chunk = bodyChunk.id
          callEdges.push(c)
          orphanCalls.splice(i, 1)
        }
      }
      runStart = runEnd = null
    }
    const covered = (pos) =>
      declInfo.citizenChunks.some((c) => c.start_pos <= pos && pos < c.end_pos)
    for (const stmt of sf.statements) {
      const s = stmt.getStart(sf)
      if (covered(s)) {
        flush()
        continue
      }
      if (runStart === null) runStart = s
      runEnd = stmt.getEnd()
    }
    flush()
  }

  timings.walk_and_resolve_ms = +(performance.now() - tWalk0).toFixed(1)
  timings.resolve_ms = +resolveMs.toFixed(1)
  timings.total_build_ms = +(performance.now() - t0).toFixed(1)

  const timingInfo = api.getTimingInfo()
  api.close()

  // Exact chunk text, sliced by the compiler's own source file, so downstream
  // consumers (retrieval) do not have to re-slice positions in another language
  // and risk UTF-16/codepoint drift.
  const chunkText = new Map()
  for (const [file, sf] of sourceFiles) {
    for (const c of byFile.get(file) || []) chunkText.set(c.id, sf.text.slice(c.start_pos, c.end_pos))
  }
  writeFileSync(`${OUT}/chunk_text.json`, JSON.stringify(Object.fromEntries(chunkText)))

  chunks.sort((a, b) => (a.file === b.file ? a.start_pos - b.start_pos : a.file < b.file ? -1 : 1))
  imports.sort((a, b) =>
    a.file === b.file ? (a.module < b.module ? -1 : 1) : a.file < b.file ? -1 : 1
  )

  return {
    meta: {
      generator: 'graph_ts7.mjs',
      typescript: '7.0.2',
      api: 'typescript/unstable/sync + typescript/unstable/ast (native tsgo)',
      repo: REPO,
      files: files.length,
      checker_used: useChecker,
      checker_calls: checkerCalls,
      checker_failures: checkerFailures,
      call_sites: callSites,
      calls_resolved_to_declaration: callEdges.filter((c) => c.resolved === 'declaration').length,
      nodes_walked: nodesWalked,
      transport: {
        enabled: timingInfo.enabled,
        requests: timingInfo.totals?.requestCount ?? null,
        round_trip_ms: timingInfo.totals ? +timingInfo.totals.roundTripMs.toFixed(1) : null,
        server_time_ms: timingInfo.totals ? +timingInfo.totals.serverTimeMs.toFixed(1) : null,
        transport_overhead_ms: timingInfo.totals
          ? +timingInfo.totals.transportOverheadMs.toFixed(1)
          : null,
        nodes_materialized: timingInfo.totals?.nodesMaterialized ?? null,
        bytes_received: timingInfo.totals?.bytesReceived ?? null,
      },
      timings,
    },
    files,
    chunks,
    symbols,
    relations: {
      DEFINES: defines,
      CALLS: callEdges,
      IMPORTS: imports,
      CONTAINS: contains,
    },
  }
}

mkdirSync(OUT, { recursive: true })
const OUT_FILE = process.env.TS7_NO_CHECKER === '1' ? 'graph_ts7_nochecker.json' : 'graph_ts7.json'
try {
  const graph = buildGraph()
  writeFileSync(`${OUT}/${OUT_FILE}`, JSON.stringify(graph, null, 1))
  console.log(JSON.stringify(graph.meta, null, 1))
  console.log(
    'chunks',
    graph.chunks.length,
    '| symbols',
    graph.symbols.length,
    '| DEFINES',
    graph.relations.DEFINES.length,
    '| CALLS',
    graph.relations.CALLS.length,
    '| IMPORTS',
    graph.relations.IMPORTS.length,
    '| CONTAINS',
    graph.relations.CONTAINS.length
  )
} catch (e) {
  console.error('FAILED:', e && e.stack ? e.stack : e)
  process.exit(1)
}
