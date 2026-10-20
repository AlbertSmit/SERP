"""Retrieval channels used by the SERP stage experiment.

Each channel is a dense scorer over the same document order. Keeping the channels
separate is the point: BM25 stays exact, and every additional unit of evidence can
be ablated, ranked, and measured on its own.
"""
from __future__ import annotations

import numpy as np
import snowballstemmer

import harness
from hybrid import char4, words_and_char4
from lean_units import LeanUnitBM25, LeanVCGrams
from vcbm25 import VCBM25

_PORTER = snowballstemmer.stemmer("porter")
_STEMS: dict[str, str] = {}


def porter(words: list[str]) -> list[str]:
    for word in words:
        if word not in _STEMS:
            _STEMS[word] = _PORTER.stemWord(word)
    return [_STEMS[word] for word in words]


def dense_from_dict(ix: harness.Index, scores: dict[int, float]) -> np.ndarray:
    dense = np.zeros(ix.N, dtype=float)
    for doc, score in scores.items():
        dense[doc] = score
    return dense


class WordBM25:
    name = "bm25"

    def __init__(self, ix):
        self.ix = ix

    def score(self, query):
        return harness.bm25_fast(self.ix, query)


class PorterBM25:
    name = "porter"

    def __init__(self, ix):
        self.ix = ix
        self.unit = LeanUnitBM25(ix, porter)

    def score(self, query):
        return self.unit.score(query)


class Char4BM25:
    name = "char4"

    def __init__(self, ix):
        self.ix = ix
        self.unit = LeanUnitBM25(ix, char4)

    def score(self, query):
        return self.unit.score(query)


class ConceptBM25:
    name = "vc"

    def __init__(self, ix):
        self.ix = ix
        self.vc = VCBM25(ix)

    def score(self, query):
        return dense_from_dict(self.ix, self.vc.score(query))


class ConceptGrams:
    name = "vcgrams"

    def __init__(self, ix):
        self.ix = ix
        self.system = LeanVCGrams(ix)

    def score(self, query):
        return self.system.score(query)


class OneStream:
    name = "onestream"

    def __init__(self, ix):
        self.ix = ix
        self.unit = LeanUnitBM25(ix, words_and_char4)

    def score(self, query):
        return self.unit.score(query)


SCORERS = {
    "bm25": WordBM25,
    "porter": PorterBM25,
    "char4": Char4BM25,
    "vc": ConceptBM25,
    "vcgrams": ConceptGrams,
    "onestream": OneStream,
}


def build(ix, names: list[str]) -> dict:
    """Build only the requested channels; Touche is large enough that this matters."""
    unknown = set(names) - set(SCORERS)
    if unknown:
        raise ValueError(f"unknown channels: {sorted(unknown)}")
    return {name: SCORERS[name](ix) for name in names}
