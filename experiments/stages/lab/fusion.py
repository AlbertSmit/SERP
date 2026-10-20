"""Rank fusion. Fixed reciprocal-rank fusion, no tuning."""
from __future__ import annotations

import numpy as np

RRF_K = 60
DEPTH = 1000


def order(scores: np.ndarray, k: int = DEPTH) -> np.ndarray:
    matched = np.flatnonzero(scores > 0)
    if matched.size == 0:
        return matched
    return matched[np.lexsort((matched, -scores[matched]))[:k]]


def ranked_ids(ix, scores: np.ndarray, k: int = DEPTH) -> list[str]:
    return [ix.doc_ids[doc] for doc in order(scores, k)]


def rrf(ix, channels: list[np.ndarray], k: int = RRF_K, depth: int = DEPTH) -> np.ndarray:
    """Reciprocal-rank fusion over dense score arrays. Only positive scores enter a list."""
    fused = np.zeros(ix.N, dtype=float)
    for scores in channels:
        ranked = order(scores, depth)
        if ranked.size:
            fused[ranked] += 1.0 / (k + np.arange(1, ranked.size + 1))
    return fused


def rrf_lists(lists: list[list[str]], k: int = RRF_K, depth: int = DEPTH) -> list[tuple[str, float]]:
    """Fuse already-ranked doc-id lists; used by the JS-parity checks."""
    fused: dict[str, float] = {}
    for ranked in lists:
        for position, doc in enumerate(ranked[:depth], start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + position)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
