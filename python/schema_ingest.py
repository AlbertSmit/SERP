#!/usr/bin/env python3
"""
Schema design & ingestion helper for SQLite FTS5 + companion meta table.
Used by sqlite_mcp_engine.py and fts5_p4_scorer.c.
"""
import sqlite3, json, re, os, time
from collections import Counter

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'scifact')
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'mcp_memory.db')

def tokenize(text):
    return re.findall(r"\b[a-z0-9]{2,}\b", text.lower())

def build_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # Main FTS5 chunks table with content
    conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        content, title,
        tokenize='porter unicode61',
        columnsize=0
    );
    """)

    # Companion meta table (zero-overhead primitives: L, U, L/U, surprisal, access_count)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks_meta (
        rowid INTEGER PRIMARY KEY,
        chunk_id TEXT UNIQUE NOT NULL,
        L INTEGER DEFAULT 0,
        U INTEGER DEFAULT 0,
        l_over_u REAL DEFAULT 0.0,
        intrinsic_surprisal REAL DEFAULT 0.0,
        created_at REAL DEFAULT 0.0,
        access_count INTEGER DEFAULT 0
    );
    """)

    # Vocabulary frequency table (exposes N_i and T_i without scanning chunks_fts)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_rowid ON chunks_meta(rowid);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_access ON chunks_meta(access_count);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_lou ON chunks_meta(l_over_u);")
    conn.commit()
    return conn

if __name__ == '__main__':
    print("Schema helper loaded. DB:", DB_PATH)
