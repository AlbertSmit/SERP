"""Code-graph retrieval over the TS7 graph: Porter BM25 vs symbol and graph channels.

Capability test, not human relevance. The 97 queries and their qrels are read off
the graph itself (see make_queries.py), so the graph channel has a structural
advantage by construction. Reported to check that the channels work and to give a
per-metric delta with a CI, not to claim retrieval quality.

Channels
  bm25     Porter-stemmed BM25 (k1=1.2, b=0.75) over chunk source text
  symbol   exact / camel-split identifier match against the chunk's declared name
  graph    one hop from the top-10 BM25 seeds over CALLS, IMPORTS and containment

Writes results/ts7_graph.json (metrics only; the doc set is in out/).
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import snowballstemmer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RESULTS = HERE.parent / "results"

PORTER = snowballstemmer.stemmer("porter")
_STEM: dict[str, str] = {}
K1, B = 1.2, 0.75
RRF_K = 60
SEEDS = 10

TOKEN_RE = __import__("re").compile(r"[^\W_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def stem(words: list[str]) -> list[str]:
    for w in words:
        if w not in _STEM:
            _STEM[w] = PORTER.stemWord(w)
    return [_STEM[w] for w in words]


class BM25:
    def __init__(self, docs: list[list[str]]):
        self.N = len(docs)
        self.tf = [Counter(d) for d in docs]
        self.dl = [len(d) for d in docs]
        self.avgdl = sum(self.dl) / self.N
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, counts in enumerate(self.tf):
            for term, n in counts.items():
                self.postings[term].append((i, n))
        self.df = {t: len(p) for t, p in self.postings.items()}

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5)) if df else 0.0

    def score(self, query: list[str]) -> list[float]:
        out = [0.0] * self.N
        for term, qtf in Counter(query).items():
            idf = self.idf(term)
            if not idf:
                continue
            for d, tf in self.postings.get(term, ()):
                norm = K1 * (1 - B + B * self.dl[d] / self.avgdl)
                out[d] += qtf * idf * tf * (K1 + 1) / (tf + norm)
        return out


def rank(scores: list[float], k: int = 100) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [i for i in order[:k] if scores[i] > 0]


def metrics(ranked: list[int], rel: dict[int, int]) -> dict[str, float]:
    gains = [rel.get(d, 0) for d in ranked[:10]]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    top10, top5 = set(ranked[:10]), set(ranked[:5])
    relset = set(rel)
    first = next((i + 1 for i, d in enumerate(ranked[:10]) if d in rel), None)
    return {
        "ndcg10": dcg / idcg if idcg else 0.0,
        "both@5": float(relset <= top5),
        "both@10": float(relset <= top10),
        "r10": len(relset & top10) / len(relset),
        "mrr": 1 / first if first else 0.0,
    }


def rrf(channels: list[list[float]]) -> list[float]:
    fused = [0.0] * len(channels[0])
    for scores in channels:
        for position, doc in enumerate(rank(scores), start=1):
            fused[doc] += 1.0 / (RRF_K + position)
    return fused


def load_graph():
    graph = json.loads((OUT / "graph_ts7.json").read_text())
    text = json.loads((OUT / "chunk_text.json").read_text())
    q = json.loads((OUT / "queries_ts7.json").read_text())
    return graph, text, q


def build_symbol_channel(chunks: list[dict]) -> tuple[list[set[str]], dict[str, float]]:
    name_tokens = []
    for c in chunks:
        toks = set(tokenize(c["name"])) | set(tokenize(c["name"].replace(".", " ")))
        for seg in c["name"].split("."):
            toks |= set(tokenize(seg))
        name_tokens.append(toks)
    df = Counter()
    for toks in name_tokens:
        for t in toks:
            df[t] += 1
    n = len(chunks)
    idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    return name_tokens, idf


def symbol_scores(query: str, chunks, name_tokens, idf) -> list[float]:
    q = tokenize(query)
    qset = set(q)
    out = []
    lowered = query.lower()
    for i, toks in enumerate(name_tokens):
        s = sum(idf.get(t, 0.0) for t in qset & toks)
        # whole-name mention is the strongest signal a developer can give
        flat = chunks[i]["name"].lower().replace(".", " ")
        if flat and flat in lowered:
            s += 5.0
        out.append(s)
    return out


def build_adjacency(graph, chunks, chunk_index) -> dict[int, dict[int, float]]:
    """One-hop weighted adjacency over chunk ids: CALLS both ways, IMPORTS file->file,
    and containment (parent/child)."""
    by_file: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        by_file[c["file"]].append(i)
    by_id = {c["id"]: i for i, c in enumerate(chunks)}

    adj: dict[int, dict[int, float]] = defaultdict(dict)

    def add(a: int, b: int, w: float):
        if a == b:
            return
        cur = adj[a].get(b, 0.0)
        if w > cur:
            adj[a][b] = w

    for e in graph["relations"]["CALLS"]:
        src, dst = by_id.get(e["chunk"]), by_id.get(e["target_chunk"] or "")
        if src is None or dst is None:
            continue
        add(src, dst, 1.0)
        add(dst, src, 0.6)

    # module specifier -> repo file, for relative imports only
    import re as _re

    for e in graph["relations"]["IMPORTS"]:
        spec = e["module"]
        if not spec.startswith("."):
            continue
        src_file = e["file"]
        parts = src_file.split("/")[:-1]
        for part in spec.split("/"):
            if part == "..":
                parts = parts[:-1]
            elif part not in (".", ""):
                parts.append(part)
        target = "/".join(parts)
        if target not in by_file:
            continue
        for a in by_file[src_file]:
            for b in by_file[target]:
                add(a, b, 0.3)
                add(b, a, 0.2)

    for i, c in enumerate(chunks):
        if c["parent"] in by_id:
            p = by_id[c["parent"]]
            add(p, i, 0.4)
            add(i, p, 0.4)
        # same-file sibling chunks under the same parent
        sibs = [j for j, o in enumerate(chunks) if o["parent"] == c["parent"] and j != i]
        for j in sibs:
            add(i, j, 0.15)

    return adj


def graph_scores(seed_scores: list[float], adj) -> list[float]:
    out = [0.0] * len(seed_scores)
    seeds = rank(seed_scores, SEEDS)
    for r, s in enumerate(seeds):
        weight = 1.0 / (r + 1)
        for dst, w in adj.get(s, {}).items():
            out[dst] += weight * w
    return out


def evaluate() -> dict:
    graph, text, q = load_graph()
    chunks = graph["chunks"]
    docs = [stem(tokenize(text[c["id"]])) for c in chunks]
    bm25 = BM25(docs)
    name_tokens, sym_idf = build_symbol_channel(chunks)
    adj = build_adjacency(graph, chunks, None)
    by_id = {c["id"]: i for i, c in enumerate(chunks)}

    queries = q["queries"]
    qrels = {qid: {by_id[d]: g for d, g in rel.items() if d in by_id} for qid, rel in q["qrels"].items()}

    per_query: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for qid, query in queries.items():
        b = bm25.score(stem(tokenize(query)))
        s = symbol_scores(query, chunks, name_tokens, sym_idf)
        g = graph_scores(b, adj)
        systems = {
            "bm25_porter": b,
            "symbol": s,
            "graph": g,
            "rrf_bm25_symbol": rrf([b, s]),
            "rrf_bm25_graph": rrf([b, g]),
            "rrf_bm25_symbol_graph": rrf([b, s, g]),
        }
        for label, scores in systems.items():
            per_query[label][qid] = metrics(rank(scores), qrels[qid])

    def mean(label, metric):
        return statistics.mean(v[metric] for v in per_query[label].values())

    def paired(base, cand, metric, reps=2000):
        deltas = [per_query[cand][q][metric] - per_query[base][q][metric] for q in queries]
        rng = random.Random(7)
        boots = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(reps))
        wins = sum(d > 1e-12 for d in deltas)
        losses = sum(d < -1e-12 for d in deltas)
        return {
            "delta": statistics.mean(deltas),
            "ci95": [boots[int(0.025 * reps)], boots[int(0.975 * reps)]],
            "wtl": [wins, len(deltas) - wins - losses, losses],
        }

    METRICS = ("both@5", "both@10", "ndcg10", "r10", "mrr")
    table = {}
    for label in per_query:
        row = {m: mean(label, m) for m in METRICS}
        if label != "bm25_porter":
            for m in METRICS:
                row[f"vs_bm25_{m}"] = paired("bm25_porter", label, m)
        table[label] = row

    return {
        "meta": {
            "kind": "capability test, graph-derived qrels (not human relevance)",
            "queries": len(queries),
            "docs": len(chunks),
            "doc_unit": "AST chunk (function/method/class/script body), full node text",
            "bm25": {"k1": K1, "b": B, "stemmer": "porter (snowballstemmer)"},
            "rrf_k": RRF_K,
            "graph_seeds": SEEDS,
            "warning": (
                "The graph channel is evaluated against qrels built from the same CALLS "
                "edges it walks. It cannot lose by design on both@k; treat the deltas as "
                "an upper bound on channel value, not as evidence of quality."
            ),
        },
        "systems": table,
    }


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    result = evaluate()
    (RESULTS / "ts7_graph.json").write_text(json.dumps(result, indent=1))

    print(f"{len(result['meta']) and result['meta']['queries']} queries, {result['meta']['docs']} docs")
    print(f"\n{'system':26} {'both@5':>7} {'both@10':>7} {'nDCG@10':>8} {'R@10':>7} {'MRR':>7}")
    for label, row in result["systems"].items():
        print(
            f"{label:26} {row['both@5']:7.3f} {row['both@10']:7.3f} "
            f"{row['ndcg10']:8.3f} {row['r10']:7.3f} {row['mrr']:7.3f}"
        )
    print(f"\n{'system vs BM25':26} {'metric':>8} {'delta':>8} {'95% CI':>20} {'W/T/L':>12}")
    for label, row in result["systems"].items():
        if label == "bm25_porter":
            continue
        for m in ("both@5", "both@10", "ndcg10", "r10"):
            p = row[f"vs_bm25_{m}"]
            lo, hi = p["ci95"]
            print(
                f"{label:26} {m:>8} {p['delta']:+8.3f} [{lo:+.3f},{hi:+.3f}] "
                f"{p['wtl'][0]:4}/{p['wtl'][1]:4}/{p['wtl'][2]:4}"
            )
