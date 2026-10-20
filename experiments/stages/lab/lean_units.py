"""Memory-light equivalent of hybrid.py's gram channels, for corpora too large for harness.Index over grams.

Same BM25 (k1 1.2, b 0.75, same idf) over the same unit streams, but postings live in int32/float arrays
instead of Python tuples, and doc unit lists are never kept. Must rank identically to hybrid.py — see check().
"""
from __future__ import annotations

import math
from array import array
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

import harness
from hybrid import GRAM_WEIGHT, char4, words_and_char4
from vcbm25 import VCBM25

K1, B = 1.2, 0.75


class LeanUnitBM25:
    def __init__(self, ix: harness.Index, transform):
        self.transform, self.ids = transform, {}
        rows, cols, tfs, lengths = array("i"), array("i"), array("f"), []
        for d, (title, body) in enumerate(zip(ix.title_tokens, ix.body_tokens)):
            units = transform(title) + transform(body)
            lengths.append(len(units))
            for unit, n in Counter(units).items():
                rows.append(self.ids.setdefault(unit, len(self.ids)))
                cols.append(d)
                tfs.append(n)
        rows, cols, tf = np.frombuffer(rows, np.int32), np.frombuffer(cols, np.int32), np.frombuffer(tfs, np.float32).astype(float)
        dl = np.array(lengths, dtype=float)
        df = np.bincount(rows, minlength=len(self.ids)).astype(float)
        idf = np.log(1 + (ix.N - df + 0.5) / (df + 0.5))
        norm = K1 * (1 - B + B * dl / dl.mean())
        impacts = idf[rows] * tf * (K1 + 1) / (tf + norm[cols])
        self.impacts = csr_matrix((impacts, (rows, cols)), shape=(len(self.ids), ix.N))
        self.N = ix.N

    def score(self, words: list[str]) -> np.ndarray:
        hits = [(self.ids[u], n) for u, n in Counter(self.transform(words)).items() if u in self.ids]
        if not hits:
            return np.zeros(self.N)
        idx, w = zip(*hits)
        return np.asarray(self.impacts[list(idx)].T @ np.array(w, dtype=float)).ravel()


class LeanVCGrams:
    def __init__(self, ix):
        self.ix, self.vc, self.grams = ix, VCBM25(ix), LeanUnitBM25(ix, char4)

    def score(self, words):
        concepts = np.zeros(self.ix.N)
        for d, s in self.vc.score(words).items():
            concepts[d] = s
        return concepts + GRAM_WEIGHT * self.grams.score(words)


class LeanOneStream:
    def __init__(self, ix):
        self.units = LeanUnitBM25(ix, words_and_char4)

    def score(self, words):
        return self.units.score(words)


def check(name: str):
    from hybrid import OneStream, VCGrams
    ix, queries, _ = harness.load(name)
    for lean, frozen in ((LeanVCGrams(ix), VCGrams(ix)), (LeanOneStream(ix), OneStream(ix))):
        same = sum(harness.rank(ix, lean.score(q)) == harness.rank(ix, frozen.score(q)) for q in queries.values())
        print(f"{name} {type(lean).__name__}: identical top-100 {same}/{len(queries)}", flush=True)


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or ("nfcorpus", "scifact"):
        check(name)
