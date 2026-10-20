"""Shared harness for first-principles BM25 probes.

Screening splits: nfcorpus/dev, scifact/train, fiqa/dev. FiQA test is held out
for confirming the final synthesized algorithm only — never load it in a probe.

A scorer is `fn(ix, query_tokens) -> dict[doc_index, score]`. Missing docs score -inf.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# The original research data lives outside the SERP worktree. Override with ALGO_DATA_ROOT.
DATA = Path(os.environ.get("ALGO_DATA_ROOT", "/Users/asmit/Personal/algo"))
STAGE_DATA = Path(os.environ.get("STAGE_DATA", ROOT.parent / "data"))
DATASETS = {
    "nfcorpus": (DATA / "semantic_lab/data/nfcorpus", "dev"),
    "scifact": (DATA / "semantic_lab/data/scifact", "train"),
    "fiqa": (DATA / "bm25_lab/data/fiqa", "dev"),
}
# Fresh confirmation-only sets (never screened on): loaded explicitly by confirm_fresh.py.
FRESH = {name: (DATA / "first_principles/data_fresh" / f"{name}.dl/{name}", "test") for name in ("arguana", "scidocs", "trec-covid", "quora")}
# New, independently-labelled corpus for the stage experiment. Not used in first_principles.
EXTRA = {
    "touche": (STAGE_DATA / "webis-touche2020", "test"),
}
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Index:
    """Plain in-memory inverted index. Everything a probe might need, nothing precomputed for a specific idea."""

    def __init__(self, corpus: dict[str, tuple[str, str]] | None = None, *, doc_ids=None, title_tokens=None, body_tokens=None):
        self.doc_ids = doc_ids or sorted(corpus)
        self.N = len(self.doc_ids)
        self.title_tokens = title_tokens or [tokenize(corpus[d][0]) for d in self.doc_ids]
        self.body_tokens = body_tokens or [tokenize(corpus[d][1]) for d in self.doc_ids]
        self.tokens = [t + b for t, b in zip(self.title_tokens, self.body_tokens)]
        self.tf = [Counter(toks) for toks in self.tokens]
        self.dl = [len(toks) for toks in self.tokens]
        self.avgdl = sum(self.dl) / self.N
        self.df: Counter[str] = Counter()
        self.cf: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, counts in enumerate(self.tf):
            for term, n in counts.items():
                self.df[term] += 1
                self.cf[term] += n
                self.postings[term].append((i, n))
        self.postings = dict(self.postings)
        self.total_tokens = sum(self.dl)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5)) if df else 0.0


def bm25(ix: Index, q: list[str], k1: float = 1.2, b: float = 0.75) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for term, qtf in Counter(q).items():
        idf = ix.idf(term)
        for d, tf in ix.postings.get(term, ()):
            norm = k1 * (1 - b + b * ix.dl[d] / ix.avgdl)
            scores[d] += qtf * idf * tf * (k1 + 1) / (tf + norm)
    return scores


def bm25_fast(ix: Index, q: list[str], k1: float = 1.2, b: float = 0.75):
    """Same scores as bm25, via a cached sparse term x doc impact matrix. Returns a dense numpy array (0 = no match)."""
    import numpy as np
    from scipy.sparse import csr_matrix
    if getattr(ix, "_impacts", None) is None:
        terms = list(ix.postings)
        ix._row = {t: i for i, t in enumerate(terms)}
        rows, cols, vals = [], [], []
        norm = [k1 * (1 - b + b * dl / ix.avgdl) for dl in ix.dl]
        for i, t in enumerate(terms):
            idf = ix.idf(t)
            for d, tf in ix.postings[t]:
                rows.append(i); cols.append(d); vals.append(idf * tf * (k1 + 1) / (tf + norm[d]))
        ix._impacts = csr_matrix((vals, (rows, cols)), shape=(len(terms), ix.N))
    hits = [(ix._row[t], n) for t, n in Counter(q).items() if t in ix._row]
    if not hits:
        return np.zeros(ix.N)
    idx, w = zip(*hits)
    return np.asarray(ix._impacts[list(idx)].T @ np.array(w, dtype=float)).ravel()


def _read_corpus(path: Path) -> dict[str, tuple[str, str]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[str(r["_id"])] = (r.get("title") or "", r.get("text") or "")
    return rows


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        f = line.split("\t")
        if n == 0 and f[0].lower().startswith("query"):
            continue
        if len(f) == 3 and int(f[2]) > 0:
            out.setdefault(f[0], {})[f[1]] = int(f[2])
    return out


def load(name: str, confirm_split: str | None = None):
    """Returns (index, queries{qid: tokens}, qrels{qid: {docid: grade}}). Index is pickled to .cache.

    confirm_split loads a held-out split; only confirm.py may pass it, once, for a frozen candidate.
    """
    folder, split = DATASETS.get(name) or FRESH.get(name) or EXTRA[name]
    split = confirm_split or split
    cache = ROOT / ".cache" / f"{name}.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            ix = pickle.load(fh)
    else:
        ix = Index(_read_corpus(folder / "corpus.jsonl"))
        cache.parent.mkdir(exist_ok=True)
        # Stream the cache: dumps()/read_bytes() hold a second full copy of the index in memory,
        # which is several GB for a corpus the size of touche.
        with cache.open("wb") as fh:
            pickle.dump(ix, fh, protocol=pickle.HIGHEST_PROTOCOL)
    qrels = _read_qrels(folder / "qrels" / f"{split}.tsv")
    queries = {}
    with (folder / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if str(r["_id"]) in qrels:
                queries[str(r["_id"])] = tokenize(r["text"])
    return ix, queries, qrels


def transformed(name: str, transform):
    """(index, queries, qrels) with every doc/query word list passed through transform(list[str]) -> list[str].

    The transform sees the word tokens of one field or one query and returns the new unit stream
    (e.g. character n-grams, vowel-stripped words). Doc ids and qrels are unchanged.
    """
    ix, queries, qrels = load(name)
    new = Index(doc_ids=ix.doc_ids,
                title_tokens=[transform(t) for t in ix.title_tokens],
                body_tokens=[transform(b) for b in ix.body_tokens])
    return new, {qid: transform(q) for qid, q in queries.items()}, qrels


def compare_transforms(transforms: dict, experiment: str, datasets=tuple(DATASETS), scorer=bm25_fast) -> dict:
    """Plain BM25 on raw words vs `scorer` (default BM25) on each transformed representation."""
    report = {}
    for name in datasets:
        base, base_ms = evaluate(bm25, name)
        rows = {"baseline": {"ndcg": _mean(base, "ndcg"), "r100": _mean(base, "r100"), "mrr": _mean(base, "mrr"), "ms_per_query": base_ms}}
        print(f"\n== {name} ({len(base)} queries)\n{'representation':28} {'nDCG@10':>8} {'R@100':>7} {'MRR':>7} {'dNDCG':>8} {'95% CI':>19} {'W/T/L':>13} {'ms/q':>7} {'vocab':>8} {'postings':>10}")
        print(f"{'words (bm25)':28} {rows['baseline']['ndcg']:8.4f} {rows['baseline']['r100']:7.4f} {rows['baseline']['mrr']:7.4f} {'':>8} {'':>19} {'':>13} {base_ms:7.2f}")
        for label, transform in transforms.items():
            data = transformed(name, transform)
            cand, ms = evaluate(scorer, name, data)
            d, lo, hi, w, t, l = paired(base, cand)
            vocab, postings = len(data[0].df), sum(data[0].df.values())
            rows[label] = {"ndcg": _mean(cand, "ndcg"), "r100": _mean(cand, "r100"), "mrr": _mean(cand, "mrr"),
                           "delta_ndcg": d, "ci95": [lo, hi], "wtl": [w, t, l], "ms_per_query": ms, "vocab": vocab, "postings": postings}
            print(f"{label:28} {rows[label]['ndcg']:8.4f} {rows[label]['r100']:7.4f} {rows[label]['mrr']:7.4f} {d:+8.4f} [{lo:+.4f},{hi:+.4f}] {w:4}/{t:4}/{l:4} {ms:7.2f} {vocab:8} {postings:10}")
            del data
        report[name] = rows
    (ROOT / "results" / f"{experiment}.json").write_text(json.dumps(report, indent=1))
    return report


def rank(ix: Index, scores, k: int = 100) -> list[str]:
    import heapq
    if not isinstance(scores, dict):
        import numpy as np
        matched = np.flatnonzero(scores > 0)
        top = matched[np.lexsort((matched, -scores[matched]))[:k]]
        return [ix.doc_ids[d] for d in top]
    top = heapq.nsmallest(k, scores.items(), key=lambda kv: (-kv[1], ix.doc_ids[kv[0]]))
    return [ix.doc_ids[d] for d, _ in top]


def metrics(ranked: list[str], rel: dict[str, int]) -> dict[str, float]:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    first = next((i + 1 for i, d in enumerate(ranked[:10]) if d in rel), None)
    return {"ndcg": dcg / idcg, "r100": len(rel.keys() & set(ranked[:100])) / len(rel), "mrr": 1 / first if first else 0.0}


def evaluate(scorer, name: str, data=None) -> tuple[dict[str, dict[str, float]], float]:
    ix, queries, qrels = data or load(name)
    per_query, t0 = {}, time.perf_counter()
    for qid, q in queries.items():
        per_query[qid] = metrics(rank(ix, scorer(ix, q)), qrels[qid])
    return per_query, (time.perf_counter() - t0) * 1000 / len(queries)


def paired(base: dict, cand: dict, metric: str = "ndcg", reps: int = 2000) -> tuple[float, float, float, int, int, int]:
    deltas = [cand[q][metric] - base[q][metric] for q in base]
    rng = random.Random(7)
    boots = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(reps))
    wins = sum(x > 1e-12 for x in deltas)
    losses = sum(x < -1e-12 for x in deltas)
    return statistics.mean(deltas), boots[int(0.025 * reps)], boots[int(0.975 * reps)], wins, len(deltas) - wins - losses, losses


def compare(variants: dict, experiment: str, datasets=tuple(DATASETS), baseline=bm25) -> dict:
    """Run the baseline (default bm25) + variants on every screening dataset; print a table, save results/<experiment>.json."""
    report = {}
    for name in datasets:
        data = load(name)
        base, base_ms = evaluate(baseline, name, data)
        rows = {"baseline": {"ndcg": _mean(base, "ndcg"), "r100": _mean(base, "r100"), "mrr": _mean(base, "mrr"), "ms_per_query": base_ms}}
        print(f"\n== {name} ({len(base)} queries)\n{'variant':28} {'nDCG@10':>8} {'R@100':>7} {'MRR':>7} {'dNDCG':>8} {'95% CI':>19} {'W/T/L':>13} {'ms/q':>7}")
        print(f"{'base:' + baseline.__name__:28} {rows['baseline']['ndcg']:8.4f} {rows['baseline']['r100']:7.4f} {rows['baseline']['mrr']:7.4f} {'':>8} {'':>19} {'':>13} {base_ms:7.2f}")
        for label, fn in variants.items():
            cand, ms = evaluate(fn, name, data)
            d, lo, hi, w, t, l = paired(base, cand)
            rows[label] = {"ndcg": _mean(cand, "ndcg"), "r100": _mean(cand, "r100"), "mrr": _mean(cand, "mrr"),
                           "delta_ndcg": d, "ci95": [lo, hi], "wtl": [w, t, l], "ms_per_query": ms,
                           "delta_r100": _mean(cand, "r100") - rows["baseline"]["r100"]}
            print(f"{label:28} {rows[label]['ndcg']:8.4f} {rows[label]['r100']:7.4f} {rows[label]['mrr']:7.4f} {d:+8.4f} [{lo:+.4f},{hi:+.4f}] {w:4}/{t:4}/{l:4} {ms:7.2f}")
        report[name] = rows
    (ROOT / "results" / f"{experiment}.json").write_text(json.dumps(report, indent=1))
    return report


def _mean(per_query, metric):
    return statistics.mean(v[metric] for v in per_query.values())


if __name__ == "__main__":
    compare({"bm25_k1_0.9_b_0.4": lambda ix, q: bm25(ix, q, 0.9, 0.4)}, "_smoke", tuple(sys.argv[1:]) or tuple(DATASETS))
