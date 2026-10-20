"""Synthetic exact-identifier and negative-query regression suite.

This is deliberately not a relevance benchmark. It tests one invariant: exact
identifiers must outrank spelling variants, and queries for unknown identifiers
must not activate a variant channel. Synthetic labels are allowed here because
the label is the identifier itself.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict

IDENT_RE = re.compile(r"^[a-z]{2,}[0-9][a-z0-9]{1,}$")


def _edit_distance_one(left: str, right: str) -> bool:
    if left == right:
        return False
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) == 1
    if len(left) > len(right):
        left, right = right, left
    i = j = 0
    skipped = False
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


def candidate_identifiers(ix, max_df=80, min_len=4, max_len=18):
    rows = []
    for term, df in ix.df.items():
        if not IDENT_RE.match(term):
            continue
        if not (min_len <= len(term) <= max_len):
            continue
        if df < 1 or df > max_df:
            continue
        rows.append((term, df))
    return sorted(rows, key=lambda item: (item[1], item[0]))


def near_misses(ix, term, candidates=None, limit=5):
    variants = []
    for other, _ in candidates if candidates is not None else candidate_identifiers(ix):
        if other == term:
            continue
        if other[:3] != term[:3]:
            continue
        if _edit_distance_one(term, other):
            variants.append(other)
            if len(variants) >= limit:
                break
    return variants


def doc_ids_for(ix, term):
    return {ix.doc_ids[doc] for doc, _ in ix.postings.get(term, ())}


def build_suite(ix, max_positive=60, max_negative=40, seed=1729):
    rng = random.Random(seed)
    candidates = candidate_identifiers(ix)
    positives = []
    for term, df in candidates:
        variants = near_misses(ix, term, candidates)
        if not variants:
            continue
        relevant = doc_ids_for(ix, term)
        near = set()
        for variant in variants:
            near |= doc_ids_for(ix, variant)
        near -= relevant
        if not relevant or not near:
            continue
        positives.append({
            "query": term,
            "relevant": sorted(relevant),
            "near_miss": sorted(near),
            "variants": variants,
        })
        if len(positives) >= max_positive:
            break

    vocab = set(ix.df)
    negatives = []
    while len(negatives) < max_negative:
        token = f"zzq{rng.randrange(10**6):06d}"
        if token in vocab:
            continue
        negatives.append({"query": token, "relevant": [], "near_miss": []})
    return {"positives": positives, "negatives": negatives}


def evaluate(ix, suite, score_fn, depth=100):
    position = {doc_id: index for index, doc_id in enumerate(ix.doc_ids)}
    exact_reciprocal = []
    exact_hits = 0
    near_miss_before = 0
    near_miss_top10 = 0
    for row in suite["positives"]:
        scores = score_fn(row["query"].split())
        ranked = sorted(
            (doc for doc in range(ix.N) if scores[doc] > 0),
            key=lambda doc: (-scores[doc], ix.doc_ids[doc]),
        )[:depth]
        ranked_ids = [ix.doc_ids[doc] for doc in ranked]
        rel = set(row["relevant"])
        near = set(row["near_miss"])
        first = next((i + 1 for i, doc in enumerate(ranked_ids) if doc in rel), None)
        if first:
            exact_hits += 1
            exact_reciprocal.append(1.0 / first)
        else:
            exact_reciprocal.append(0.0)
        near_positions = [i for i, doc in enumerate(ranked_ids) if doc in near]
        if first and any(pos < first - 1 for pos in near_positions):
            near_miss_before += 1
        if any(pos < 10 for pos in near_positions):
            near_miss_top10 += 1

    negative_activation = 0
    for row in suite["negatives"]:
        scores = score_fn(row["query"].split())
        ranked = sorted(
            (doc for doc in range(ix.N) if scores[doc] > 0),
            key=lambda doc: (-scores[doc], ix.doc_ids[doc]),
        )[:10]
        if ranked:
            negative_activation += 1

    n_pos = max(len(suite["positives"]), 1)
    n_neg = max(len(suite["negatives"]), 1)
    return {
        "positives": len(suite["positives"]),
        "negatives": len(suite["negatives"]),
        "exact_hit_rate": exact_hits / n_pos,
        "exact_mrr": sum(exact_reciprocal) / n_pos,
        "near_miss_before_exact": near_miss_before / n_pos,
        "near_miss_top10": near_miss_top10 / n_pos,
        "negative_activation": negative_activation / n_neg,
    }
