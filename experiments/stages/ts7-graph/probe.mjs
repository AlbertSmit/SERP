// Probe: can we open a project and walk the AST with typescript@7.0.2's unstable sync API?
import { API } from 'typescript/unstable/sync'
import { SyntaxKind } from 'typescript/unstable/ast'

const t0 = performance.now()
const api = new API()
let snapshot
try {
  snapshot = api.updateSnapshot({ openProjects: ['./tsconfig.json'] })
} catch (e) {
  console.log('updateSnapshot failed:', e && e.message)
  process.exit(1)
}
const tOpen = performance.now()
console.log('snapshot in', (tOpen - t0).toFixed(1), 'ms')
const projects = snapshot.getProjects()
console.log('projects:', projects.length)
for (const p of projects) {
  console.log('  project', p.configFileName, 'rootFiles', p.rootFiles.length)
  console.log('  first roots', p.rootFiles.slice(0, 5))
  const program = p.program
  const names = program.getSourceFileNames()
  console.log('  program files:', names.length)
  const first = names.find((n) => n.includes('/src/'))
  console.log('  sample file:', first)
  const sf = program.getSourceFile(first)
  console.log('  sf?', !!sf, 'statements', sf && sf.statements.length)
  if (sf) {
    console.log('  first stmt kind', sf.statements[0].kind, SyntaxKind[sf.statements[0].kind])
    let n = 0
    const walk = (node) => {
      n++
      node.forEachChild(walk)
    }
    walk(sf)
    console.log('  nodes walked', n)
    const pos = sf.statements[0].getStart()
    console.log('  line/char', JSON.stringify(sf.getLineAndCharacterOfPosition(pos)))
  }
  console.log('  checker?', !!p.checker)
  const sf2 = p.program.getSourceFile(p.program.getSourceFileNames()[0])
  if (sf2) {
    const diags = p.program.getSyntacticDiagnostics(sf2.fileName)
    console.log('  syntactic diagnostics on', sf2.fileName, diags.length)
  }
}
const timing = api.getTimingInfo()
console.log('timing enabled', timing.enabled, 'requests', timing.totals && timing.totals.requests)
api.close()
