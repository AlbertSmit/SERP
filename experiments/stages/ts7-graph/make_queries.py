"""Deterministic capability-test queries from the TS7 code graph.

This is NOT human relevance. The qrels are read off the graph itself: each query
asks for both endpoints of a call edge, so the graph channel has a structural
advantage by construction. The point is to check that graph retrieval machinery
runs and that its channels behave as designed, not to prove quality.

Writes out/queries_ts7.json  (100 queries, fixed order, seeded).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def words(name: str) -> list[str]:
    """``ensureSchema`` -> ``ensure schema``; ``a.b.c`` -> words of each segment."""
    out: list[str] = []
    for seg in re.split(r"[^A-Za-z0-9]+", name):
        out.extend(w.lower() for w in CAMEL.findall(seg) if len(w) > 1)
    return out


def local_words(chunk: dict) -> list[str]:
    """Words of the chunk's own name, without the enclosing class/module prefix.

    A query for ``SQLiteContextEngine.storeMemory`` should not carry the four
    class words into every other query as well.
    """
    if chunk["name"] == "<module>":
        stem = chunk["file"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return words(stem) + ["script"]
    return words(chunk["name"].rsplit(".", 1)[-1])


def main() -> None:
    graph = json.loads((OUT / "graph_ts7.json").read_text())
    chunks = {c["id"]: c for c in graph["chunks"]}

    # only edges whose target is a chunk in this repo: those are the ones a graph
    # hop can actually walk.
    edges = [
        e
        for e in graph["relations"]["CALLS"]
        if e["resolved"] == "declaration" and e["target_chunk"] in chunks and e["chunk"] in chunks
    ]
    # one query per (caller, callee) pair, so the sample is not dominated by a hot callee
    seen = set()
    unique = []
    for e in edges:
        key = (e["chunk"], e["target_chunk"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)

    unique.sort(key=lambda e: (e["file"], e["line"], e["callee"]))
    rng = random.Random(20261020)
    rng.shuffle(unique)

    queries = {}
    qrels = {}
    provenance = {}
    for i, e in enumerate(unique[:100]):
        caller = chunks[e["chunk"]]
        callee = chunks[e["target_chunk"]]
        if caller["id"] == callee["id"]:
            continue
        qid = f"q{i:03d}"
        cw, dw = local_words(caller), local_words(callee)
        qwords = [w for w in dict.fromkeys(cw + dw) if w not in {"module", "this", "script"}]
        if len(qwords) < 3:
            continue
        queries[qid] = " ".join(qwords)
        qrels[qid] = {caller["id"]: 1, callee["id"]: 1}
        provenance[qid] = {
            "edge": "CALLS",
            "caller": caller["id"],
            "callee": callee["id"],
            "site": f"{e['file']}:{e['line']}",
        }

    payload = {
        "meta": {
            "generator": "make_queries.py",
            "source_graph": "out/graph_ts7.json",
            "kind": "graph-derived capability test, not human relevance",
            "seed": 20261020,
            "queries": len(queries),
            "qrels_per_query": 2,
        },
        "queries": queries,
        "qrels": qrels,
        "provenance": provenance,
    }
    (OUT / "queries_ts7.json").write_text(json.dumps(payload, indent=1))
    print(f"{len(queries)} queries written to {OUT / 'queries_ts7.json'}")
    for qid in list(queries)[:5]:
        print(" ", qid, queries[qid], "->", list(qrels[qid]))


if __name__ == "__main__":
    main()
