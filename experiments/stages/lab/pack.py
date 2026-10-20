"""Packing after retrieval. These are deliberately simple baselines for the SERP packer.

The question is not "can a clever packer win?" but "does residual packing preserve the
relevant documents that fusion retrieved, at a real budget?" Truncation is the control.
"""
from __future__ import annotations

import math
from collections import Counter

import harness


def truncate(ranked: list[str], lengths: dict[str, int], budget: int) -> list[str]:
    selected, used = [], 0
    for doc in ranked:
        length = lengths.get(doc, 0)
        if used + length > budget:
            continue
        selected.append(doc)
        used += length
    return selected


def _idf(ix, term):
    return ix.idf(term)


def _query_weights(ix, query):
    weights = {}
    for term, qtf in Counter(query).items():
        weights[term] = _idf(ix, term) * max(qtf, 1)
    return weights


def greedy_coverage(ranked, query, ix, budget, pool=100):
    """Pick the candidate with the most uncovered query weight per token."""
    weights = _query_weights(ix, query)
    uncovered = dict(weights)
    position = {doc_id: index for index, doc_id in enumerate(ix.doc_ids)}
    selected, used, chosen = [], 0, set()
    while used < budget:
        best, best_gain, best_new = None, 0.0, {}
        for doc_id in ranked[:pool]:
            if doc_id in chosen:
                continue
            doc = position[doc_id]
            length = ix.dl[doc]
            if used + length > budget:
                continue
            counts = ix.tf[doc]
            new = {t: w for t, w in uncovered.items() if counts.get(t, 0) > 0}
            gain = sum(new.values()) / max(length + 10, 1) ** 0.35
            if gain > best_gain:
                best, best_gain, best_new = doc_id, gain, new
        if best is None or best_gain <= 0:
            break
        selected.append(best)
        chosen.add(best)
        used += ix.dl[position[best]]
        for term in best_new:
            uncovered.pop(term, None)
    return selected


def residual(ranked, query, ix, budget, decay=2.2, pool=100):
    """Residual query-term demand with exponential decay after each pick."""
    weights = _query_weights(ix, query)
    residual = dict(weights)
    position = {doc_id: index for index, doc_id in enumerate(ix.doc_ids)}
    selected, used, chosen = [], 0, set()
    while used < budget:
        best, best_gain, best_matches = None, 0.0, []
        for doc_id in ranked[:pool]:
            if doc_id in chosen:
                continue
            doc = position[doc_id]
            length = ix.dl[doc]
            if used + length > budget:
                continue
            counts = ix.tf[doc]
            gain = 0.0
            matches = []
            for term, weight in residual.items():
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                potential = (1.0 - math.exp(-1.5 * tf ** 0.45)) / (1.0 - math.exp(-1.5))
                gain += weight * potential
                matches.append((term, potential))
            gain /= max(length + 10, 1) ** 0.35
            if gain > best_gain:
                best, best_gain, best_matches = doc_id, gain, matches
        if best is None or best_gain <= 0:
            break
        selected.append(best)
        chosen.add(best)
        used += ix.dl[position[best]]
        for term, potential in best_matches:
            residual[term] *= math.exp(-decay * potential)
    return selected


def metrics(selected, relevant: dict, fused_top: list[str], depth=100) -> dict:
    rel = set(relevant)
    selected_rel = rel & set(selected)
    fused_rel = rel & set(fused_top[:depth])
    return {
        "selected": len(selected),
        "selected_relevant": len(selected_rel),
        "recall": len(selected_rel) / max(len(rel), 1),
        "fused_relevant": len(fused_rel),
        "retention": len(selected_rel) / max(len(fused_rel), 1),
    }
