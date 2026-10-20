// bench.mjs — cold build timing for both graphs, repeated.
//
// Each TS7 run spawns a fresh tsgo process, which is what a CLI tool pays. A
// long-lived process pays the spawn once; that split is reported separately
// (spawn_and_snapshot_ms vs the rest) inside graph_ts7.json.
//
// Writes out/bench.json.

import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPS = 7

function run(script, env = {}) {
  const t0 = performance.now()
  execFileSync(process.execPath, [`${HERE}/${script}`], {
    stdio: 'ignore',
    env: { ...process.env, ...env },
  })
  return performance.now() - t0
}

const stats = (xs) => {
  const s = [...xs].sort((a, b) => a - b)
  return {
    runs_ms: xs.map((x) => +x.toFixed(1)),
    min: +s[0].toFixed(1),
    median: +s[Math.floor(s.length / 2)].toFixed(1),
    max: +s[s.length - 1].toFixed(1),
  }
}

const out = { reps: REPS, node: process.version }

// warm the fs cache first so the first run is not paying for cold file reads
run('graph_regex.mjs')
run('graph_ts7.mjs')
run('graph_ts7.mjs', { TS7_NO_CHECKER: '1' })

out.regex = stats(Array.from({ length: REPS }, () => run('graph_regex.mjs')))
// type checker disabled, to separate parser cost from checker cost.
out.ts7_no_checker_process = stats(
  Array.from({ length: REPS }, () => run('graph_ts7.mjs', { TS7_NO_CHECKER: '1' }))
)

// full build last, so out/graph_ts7.json holds the checked graph the other
// scripts consume.
out.ts7_total_process = stats(Array.from({ length: REPS }, () => run('graph_ts7.mjs')))

// in-process breakdowns, read from each variant's own output file
const meta = JSON.parse(readFileSync(`${HERE}/out/graph_ts7.json`, 'utf8')).meta
const metaNoChecker = JSON.parse(
  readFileSync(`${HERE}/out/graph_ts7_nochecker.json`, 'utf8')
).meta
out.ts7_breakdown = meta.timings
out.ts7_breakdown_no_checker = metaNoChecker.timings
out.ts7_transport = meta.transport
out.ts7_transport_no_checker = metaNoChecker.transport
out.regex_meta = JSON.parse(readFileSync(`${HERE}/out/graph_regex.json`, 'utf8')).meta

writeFileSync(`${HERE}/out/bench.json`, JSON.stringify(out, null, 1))
console.log(JSON.stringify(out, null, 1))
