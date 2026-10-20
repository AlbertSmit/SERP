"""Round-4 candidates, frozen before the test run.

primary   (vc_grams):    VC-BM25 over corpus-vetted word concepts + 0.25 x BM25 over in-word char 4-grams.
secondary (one_stream):  plain BM25 over one index whose units are words + in-word char 4-grams.
"""
from __future__ import annotations

import numpy as np

import harness
from vcbm25 import VCBM25

GRAM_WEIGHT = 0.25


def char4(words: list[str]) -> list[str]:
    out = []
    for w in words:
        m = f"^{w}$"
        out += [m] if len(m) <= 4 else [m[i:i + 4] for i in range(len(m) - 3)]
    return out


def words_and_char4(words: list[str]) -> list[str]:
    return list(words) + char4(words)


def unit_index(ix: harness.Index, transform) -> harness.Index:
    return harness.Index(doc_ids=ix.doc_ids,
                         title_tokens=[transform(t) for t in ix.title_tokens],
                         body_tokens=[transform(b) for b in ix.body_tokens])


class VCGrams:
    def __init__(self, ix: harness.Index):
        self.ix, self.vc, self.grams = ix, VCBM25(ix), unit_index(ix, char4)

    def score(self, words: list[str]) -> np.ndarray:
        concepts = np.zeros(self.ix.N)
        for d, s in self.vc.score(words).items():
            concepts[d] = s
        return concepts + GRAM_WEIGHT * harness.bm25_fast(self.grams, char4(words))


class OneStream:
    def __init__(self, ix: harness.Index):
        self.units = unit_index(ix, words_and_char4)

    def score(self, words: list[str]) -> np.ndarray:
        return harness.bm25_fast(self.units, words_and_char4(words))
