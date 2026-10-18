#!/usr/bin/env python3
"""
Ingestion: BEIR SciFact -> SQLite FTS5 (chunks_fts) + meta (chunks_meta)
Pre-computes L, U, L/U, intrinsic surprisal for sub-millisecond read-time scoring.
"""
import sqlite3, json, re, math, time
from collections import Counter
import schema_ingest

TOKEN_RE = re.compile(r"\b[a-z0-9]{2,}\b")

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

DATASET_DIR = schema_ingest.DATASET_DIR
DB_PATH = schema_ingest.DB_PATH

def ingest_corpus():
    conn = schema_ingest.build_database()
    # Load corpus
    with open(f"{DATASET_DIR}/corpus.jsonl") as f:
        for line in f:
            item = json.loads(line)
            doc_id = str(item['_id'])
            title = item.get('title', '')
            text = item.get('text', '')
            content = f"{title} {text}"
            tokens = tokenize(content)
            L = len(tokens)
            vocab = Counter(tokens)
            U = len(vocab)
            # Intrinsic surprisal: -sum(p(w) * log(p(w))) / L for this doc
            # This measures vocabulary diversity / entropy of token usage
            if L > 0:
                probs = [c / L for c in vocab.values()]
                # Shannon entropy (normalized by L to keep scale manageable)
                intrinsic = -sum(p * math.log(p + 1e-12) for p in probs) / max(L, 1)
            else:
                intrinsic = 0.0
            # Insert into FTS5
            conn.execute("INSERT INTO chunks_fts(rowid, content, title) VALUES (?, ?, ?)",
                         (int(doc_id) if doc_id.isdigit() else hash(doc_id), content, title))
            # Insert meta (if rowid mapping needed: for FTS5, rowid is the document ID)
            # Note: For simplicity, we use doc_id string as rowid reference; but FTS5 uses integer rowid.
            # We use the integer doc_id as the PRIMARY KEY reference.
            conn.execute(
                "INSERT OR IGNORE INTO chunks_meta(rowid, chunk_id, L, U, l_over_u, intrinsic_surprisal, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (int(doc_id) if doc_id.isdigit() else hash(doc_id) % (2**31), doc_id, L, U, L / max(U, 1), intrinsic, time.time())
            )
    conn.commit()
    print(f"Ingested {len([l for l in open(f'{DATASET_DIR}/corpus.jsonl')])} corpus docs.")
    conn.close()

if __name__ == '__main__':
    ingest_corpus()
