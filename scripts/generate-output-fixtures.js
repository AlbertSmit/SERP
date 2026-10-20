#!/usr/bin/env node
// Writes the committed fixtures in test/fixtures/outputs from pinned git objects and one
// TAP run with durations zeroed. The measurement reads the committed files; do not regenerate.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'test/fixtures/outputs');
const RELEASE = '7441049083262a6477221c6b83280fd62ae0a2b0';
const FIXTURES = {
  'git-log-p.txt': ['git', ['log', '-p', '-1', '--no-color', '--no-ext-diff', '3faffe31d170bc1e9997a4ee58f1ab649ae2f023']],
  'engine-source.txt': ['git', ['show', `${RELEASE}:src/engine/engine.js`]],
  'package-lock.txt': ['git', ['show', `${RELEASE}:package-lock.json`]],
  'node-test-tap.txt': [process.execPath, [
    '--test', '--test-concurrency=1', '--test-reporter=tap', 'test/engine.test.js', 'test/stages.test.js',
  ]],
};

function run(command, args) {
  return execFileSync(command, args, {cwd: ROOT, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024});
}

function withoutDurations(text) {
  return text.replace(/(duration_ms:? )[\d.]+/g, (_, prefix) => `${prefix}0`);
}

fs.mkdirSync(OUT, {recursive: true});
for (const [name, [command, args]] of Object.entries(FIXTURES)) {
  const output = run(command, args);
  const file = path.join(OUT, name);
  fs.writeFileSync(file, name === 'node-test-tap.txt' ? withoutDurations(output) : output);
  console.log(`${name}: ${fs.statSync(file).size} bytes`);
}
