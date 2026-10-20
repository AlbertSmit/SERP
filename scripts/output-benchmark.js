#!/usr/bin/env node
// Scratch-index measurement: tokens returned vs raw, always next to whether the answer
// span was returned. Questions were fixed before the first run; do not tune on them.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SQLiteContextEngine } from '../src/engine/engine.js';
import { tokenize } from '../src/engine/tokenize.js';
import { indexText, searchText } from '../src/engine/scratch.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FIXTURES = path.join(ROOT, 'test/fixtures/outputs');
const BUDGETS = [120, 250, 500];

function loadSources(questions) {
  const names = [...new Set(questions.map((q) => q.source))];

  return new Map(names.map((name) => [name, fs.readFileSync(path.join(FIXTURES, name), 'utf8')]));
}

function assertAnswersPresent(questions, sources) {
  const missing = questions.filter((q) => !sources.get(q.source).includes(q.answer));
  if (missing.length > 0) throw new Error(`Answer not in raw source: ${missing.map((q) => q.id).join(', ')}`);
}

function measure(engine, question, budget, raw) {
  const contents = searchText(engine, question.question, {budget}).results.map((r) => r.content);

  return {
    id: question.id,
    budget,
    retained: contents.some((c) => c.includes(question.answer)),
    returned_tokens: tokenize(contents.join('\n')).length,
    raw_tokens: tokenize(raw).length,
    returned_chars: contents.reduce((sum, c) => sum + c.length, 0),
    raw_chars: raw.length,
  };
}

function total(rows, key) {
  return rows.reduce((sum, row) => sum + row[key], 0);
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function summarize(budget, rows) {
  return {
    budget,
    evidence_retained: `${rows.filter((r) => r.retained).length}/${rows.length}`,
    returned_tokens: total(rows, 'returned_tokens'),
    raw_tokens: total(rows, 'raw_tokens'),
    token_savings: percent(1 - total(rows, 'returned_tokens') / total(rows, 'raw_tokens')),
    char_savings: percent(1 - total(rows, 'returned_chars') / total(rows, 'raw_chars')),
  };
}

function printTable(rows) {
  const keys = Object.keys(rows[0]);
  console.log(`| ${keys.join(' | ')} |`);
  console.log(`|${keys.map(() => '---').join('|')}|`);
  for (const row of rows) console.log(`| ${keys.map((k) => row[k]).join(' | ')} |`);
  console.log('');
}

function main() {
  const questions = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'questions.json'), 'utf8'));
  const sources = loadSources(questions);
  assertAnswersPresent(questions, sources);
  const engine = new SQLiteContextEngine(':memory:');
  try {
    const receipts = [...sources].map(([name, text]) => indexText(engine, text, {source: name}));
    printTable(receipts.map((r) => ({
      source: r.source, raw_tokens: r.raw_tokens, raw_chars: r.raw_chars, chunks: r.chunks,
      receipt_tokens: tokenize(JSON.stringify(r)).length,
    })));
    const rows = BUDGETS.flatMap((budget) => questions.map((q) => measure(engine, q, budget, sources.get(q.source))));
    printTable(rows);
    printTable(BUDGETS.map((budget) => summarize(budget, rows.filter((r) => r.budget === budget))));
  } finally {
    engine.db.close();
  }
}

main();
