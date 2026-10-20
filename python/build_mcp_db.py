#!/usr/bin/env python3
"""
build_mcp_db.py - Creates mcp_memory.sqlite from scifact corpus.
Produces sentence-level chunks + full-document chunks + 1,500 synthetic MCP Noise Chunks.
Populates chunks_fts, chunks_vocab, chunks_meta.
"""
import sqlite3, json, re, math, random, os, time

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'mcp_memory.sqlite')
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'scifact')
TOKEN_RE = re.compile(r"\b[a-z0-9]{2,}\b")

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

def main():
    # Clean DB
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # 1) chunks_fts with explicit doc_id / chunk_id unindexed columns
    conn.execute("""
    CREATE VIRTUAL TABLE chunks_fts USING fts5(
        chunk_id UNINDEXED,
        doc_id UNINDEXED,
        content,
        tokenize='unicode61'
    );
    """)

    # 2) chunks_vocab table (built on chunks_fts row)
    conn.execute("CREATE VIRTUAL TABLE chunks_vocab USING fts5vocab(chunks_fts, 'row');")

    # 3) chunks_meta
    conn.execute("""
    CREATE TABLE chunks_meta (
        rowid INTEGER PRIMARY KEY,
        chunk_id TEXT UNIQUE NOT NULL,
        doc_id TEXT,
        L INTEGER,
        U INTEGER,
        l_over_u REAL,
        intrinsic_surprisal REAL,
        age_days REAL,
        access_count INTEGER
    );
    """)

    # Load corpus
    docs = []
    with open(os.path.join(DATASET_DIR, 'corpus.jsonl')) as f:
        for line in f:
            item = json.loads(line)
            docs.append(item)

    # Chunk each doc: split into up to 5 sentence chunks + full doc
    chunk_rows = []
    for item in docs:
        doc_id = str(item['_id'])
        text = item.get('title', '') + ' ' + item.get('text', '')
        # Sentence split
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        for idx, sent in enumerate(sentences[:5]):
            tokens = tokenize(sent)
            chunk_id = f"{doc_id}_sent{idx}"
            L = len(tokens)
            U = len(set(tokens))
            age_days = (hash(chunk_id) % 365)
            vocab_counts = {w: tokens.count(w) for w in set(tokens)}
            entropy_approx = -sum((c/max(L,1))*math.log((c/max(L,1))+1e-12) for c in vocab_counts.values()) if L > 0 else 0
            access_count = max(1, int(10 * entropy_approx))
            chunk_rows.append({
                'chunk_id': chunk_id, 'doc_id': doc_id, 'content': sent,
                'L': L, 'U': max(U, 1), 'l_over_u': L/max(U,1),
                'age_days': float(age_days), 'access_count': access_count,
                'intrinsic_surprisal': entropy_approx / max(L, 1)
            })
        # Full doc chunk
        full_tokens = tokenize(text)
        full_chunk_id = f"{doc_id}_full"
        full_L = len(full_tokens)
        full_U = len(set(full_tokens))
        full_age_days = (hash(full_chunk_id) % 365)
        full_entropy = -sum((c/max(full_L,1))*math.log((c/max(full_L,1))+1e-12) for c in {w: full_tokens.count(w) for w in set(full_tokens)}.values()) if full_L > 0 else 0
        chunk_rows.append({
            'chunk_id': full_chunk_id, 'doc_id': doc_id, 'content': text,
            'L': full_L, 'U': max(full_U, 1), 'l_over_u': full_L/max(full_U,1),
            'age_days': float(full_age_days), 'access_count': max(1, int(10 * full_entropy)),
            'intrinsic_surprisal': full_entropy / max(full_L, 1)
        })

    # Inject 1,500 realistic MCP Noise Chunks (fixed seed so the benchmark corpus is reproducible)
    random.seed(1729)
    noise_words_pool = [
        "system", "log", "error", "warning", "debug", "info", "trace", "memory",
        "buffer", "overflow", "timeout", "connection", "request", "response",
        "service", "server", "client", "network", "packet", "protocol",
        "authentication", "authorization", "token", "session", "cookie",
        "database", "query", "index", "cache", "queue", "thread", "process"
    ]
    synthetic_phrases = [
        "ERROR: connection timeout from remote host",
        "DEBUG: memory buffer overflow at address 0x7f3a",
        "WARNING: authentication token expired during session",
        "INFO: database query returned 0 results in 45ms",
        "TRACE: network packet lost in transit layer",
        "DEBUG: thread pool exhausted, spawning new worker",
        "ERROR: service unavailable, retrying in 300ms",
        "WARNING: cache miss rate exceeded threshold 0.85",
        "INFO: protocol handshake completed successfully",
        "DEBUG: client disconnected unexpectedly from server",
    ]
    for i in range(1500):
        phrase_base = random.choice(synthetic_phrases)
        # Repeat common words 15-30 times to create high L (~220) and low U (~15)
        words = phrase_base.split()
        repeated_words = []
        for w in words:
            repeated_words.extend([w] * random.randint(3, 8))
        # Pad with more repetition to reach ~220 tokens
        filler_word = random.choice(noise_words_pool)
        repeated_words.extend([filler_word] * random.randint(10, 25))
        content = " ".join(repeated_words)
        tokens = tokenize(content)
        L = len(tokens)
        if L < 160:
            # Pad more
            content += " " + " ".join([filler_word] * (160 - len(tokens)))
            tokens = tokenize(content)
            L = len(tokens)
        U = len(set(tokens))
        chunk_id = f"noise_{i:05d}"
        doc_id = f"noise_doc_{i % 50}"
        age_days = random.choice([0, 1, 3, 7, 14, 30, 60, 120, 240, 365])
        access_count = random.randint(1, 8)
        vocab_counts = {w: tokens.count(w) for w in set(tokens)}
        entropy_approx = -sum((c/max(L,1))*math.log((c/max(L,1))+1e-12) for c in vocab_counts.values()) if L > 0 else 0
        chunk_rows.append({
            'chunk_id': chunk_id, 'doc_id': doc_id, 'content': content,
            'L': L, 'U': max(U, 1), 'l_over_u': L/max(U, 1),
            'age_days': float(age_days), 'access_count': access_count,
            'intrinsic_surprisal': entropy_approx / max(L, 1)
        })

    # Insert into FTS5 and meta: retrieve assigned rowid
    for row in chunk_rows:
        conn.execute(
            "INSERT INTO chunks_fts(chunk_id, doc_id, content) VALUES (?, ?, ?)",
            (row['chunk_id'], row['doc_id'], row['content'])
        )
        cursor = conn.execute("SELECT rowid FROM chunks_fts WHERE chunk_id = ?", (row['chunk_id'],))
        assigned_rowid = cursor.fetchone()[0]
        conn.execute(
            "INSERT INTO chunks_meta(rowid, chunk_id, doc_id, L, U, l_over_u, intrinsic_surprisal, age_days, access_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (assigned_rowid, row['chunk_id'], row['doc_id'], row['L'], row['U'], row['l_over_u'], row['intrinsic_surprisal'], row['age_days'], row['access_count'])
        )
    conn.commit()

    # Verify
    count_fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    count_meta = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
    vocab_check = conn.execute("SELECT * FROM chunks_vocab LIMIT 5").fetchall()
    print(f"DB created: {DB_PATH}")
    print(f"FTS5 rows: {count_fts} | Meta rows: {count_meta} | Vocab sample: {len(vocab_check)} terms")
    conn.close()

if __name__ == '__main__':
    main()
