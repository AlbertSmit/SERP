"""Vetted-Concept BM25 (VC-BM25).

BM25 statistics, but the unit of evidence is a *concept*: a query word plus every
corpus surface form that (a) shares its stem-ish prefix and (b) is vouched for by the
corpus — the two words' document sets overlap. tf is summed over the concept's forms,
df is the size of their union, then ordinary BM25. Top-k runs exactly via MaxScore.
"""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

K1, B = 1.2, 0.75
MIN_OVERLAP = 0.1


class VCBM25:
    def __init__(self, ix):
        self.ix = ix
        self.norm = [K1 * (1 - B + B * dl / ix.avgdl) for dl in ix.dl]
        self.by_prefix = defaultdict(list)
        for term in ix.df:
            if len(term) >= 4:
                self.by_prefix[term[:4]].append(term)
        self._docs: dict[str, set[int]] = {}
        self._concepts: dict[str, tuple[dict[int, float], float]] = {}

    def docs(self, term: str) -> set[int]:
        if term not in self._docs:
            self._docs[term] = {d for d, _ in self.ix.postings.get(term, ())}
        return self._docs[term]

    def members(self, term: str) -> list[str]:
        if len(term) < 4:
            return [term]
        p = max(4, len(term) - 3)
        forms = [v for v in self.by_prefix[term[:4]] if v != term and v[:p] == term[:p] and len(v) <= len(term) + 4]
        if term not in self.ix.df:
            return [term] + forms
        own = self.docs(term)
        return [term] + [v for v in forms if len(own & self.docs(v)) / min(len(own), len(self.docs(v))) >= MIN_OVERLAP]

    def concept(self, term: str) -> tuple[dict[int, float], float]:
        """Per-document impact of the concept (idf · saturated tf) and its upper bound."""
        if term not in self._concepts:
            tf: dict[int, int] = defaultdict(int)
            for form in self.members(term):
                for d, n in self.ix.postings.get(form, ()):
                    tf[d] += n
            df = len(tf)
            idf = math.log(1 + (self.ix.N - df + 0.5) / (df + 0.5)) if df else 0.0
            impacts = {d: idf * n * (K1 + 1) / (n + self.norm[d]) for d, n in tf.items()}
            self._concepts[term] = (impacts, max(impacts.values(), default=0.0))
        return self._concepts[term]

    def score(self, query: list[str], k: int | None = None) -> dict[int, float]:
        """All matching docs when k is None; otherwise a candidate set guaranteed to contain the exact top-k."""
        weighted = [(qtf, *self.concept(t)) for t, qtf in Counter(query).items()]
        candidates = self._maxscore_candidates(weighted, k) if k else {d for _, impacts, _ in weighted for d in impacts}
        return {d: sum(qtf * impacts.get(d, 0.0) for qtf, impacts, _ in weighted) for d in candidates}

    @staticmethod
    def _maxscore_candidates(weighted, k: int) -> set[int]:
        """Walk concepts rarest-first; stop admitting new docs once the unwalked bounds can't reach the kth partial score."""
        order = sorted(weighted, key=lambda c: len(c[1]))
        remaining = sum(qtf * ub for qtf, _, ub in order)
        partial: dict[int, float] = defaultdict(float)
        for qtf, impacts, ub in order:
            if len(partial) >= k and remaining < heapq.nlargest(k, partial.values())[-1]:
                break
            for d, impact in impacts.items():
                partial[d] += qtf * impact
            remaining -= qtf * ub
        return set(partial)


def scorer(k: int | None = None):
    def run(ix, query):
        if not hasattr(ix, "_vcbm25"):
            ix._vcbm25 = VCBM25(ix)
        return ix._vcbm25.score(query, k)
    run.__name__ = f"vcbm25{'_top' + str(k) if k else ''}"
    return run
