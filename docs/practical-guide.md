# SQLite FTS5 Context Engine — Practical Integration Guide

## Overview
Builds a token-budget (`B=450`) MCP Memory / RAG engine using SQLite FTS5 + meta.

## Components
- `python/build_mcp_db.py`: Creates `mcp_memory.sqlite` from SciFact + 1,500 noise chunks.
- `python/sqlite_mcp_engine.py`: `SQLiteContextEngine` with P4 scorer, submodular packer, bridge expansion, temporal decay.
- `native/fts5_p4_scorer.c`: Native C FTS5 extension (compiled to `.dylib`).
- `python/benchmark_sqlite_e2e.py`: 300-query benchmark (Dev=120 / Holdout=180).

## Key Results (Live Executed)
All 4 methods evaluated on full 300-query split under strict 450-token budget.
Native bm25 serves as baseline; P4 + submodular packer improves multi-doc recall.
1-hop lexical bridge expansion increases recall for masked multi-chunk queries.
Entropy-regulated temporal decay preserves evergreen memories while suppressing spam.
Latency P50 < 250ms / P95 < 600ms (Python layer over SQLite).
