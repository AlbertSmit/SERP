"""Ship/kill mode boundary and safety matrix (ARGON A3).

This module contains no new scorer. It is a *product policy* over four frozen
retrieval modes, plus the safety invariants that decide whether any of them may
ship automatically:

  default  Porter BM25, k1=1.2, b=0.75.
  deep     EBR: Porter + untyped bridge + per-entity coverage, RRF k=60,
           20 seeds, 20 bridges, 5% df cap (HOTPOT.md constants, unchanged).
  routed   the frozen per-query router in lab/router.py: deep when its rules fire,
           default otherwise. Measured as an automatic baseline, never shipped.
  graph    Porter + code-graph expansion over the SERP chunk corpus, RRF k=60.
           Code corpus only; the router and the text corpora never see it.

The one policy mechanism added here is the mandatory exact-identifier gate: a
query that names a digit/underscore identifier never activates bridge, entity or
graph expansion, whatever mode was selected. The gate is a hard switch to the
lexical default, not a score adjustment.

Mode selection is deliberately *not* learned. `select()` maps a workload and a
mode to an actual mode; `is_automatic()` says which modes a workload may choose
without an explicit request. Everything else is a measurement.
"""
from __future__ import annotations

import re

import exact_ids
import router

# ---------------------------------------------------------------------------
# Frozen constants. Nothing below is fitted in this experiment.
# ---------------------------------------------------------------------------
MODES = ("default", "deep", "routed", "graph")

RRF_K = 60
SEEDS = 20
BRIDGES = 20
MAX_DF_FRAC = 0.05
MARGIN = -0.005          # single-hop non-inferiority margin, ARGON A3 gate 3

BEIR_METRICS = ("ndcg", "r100", "mrr")
HOTPOT_METRICS = ("ndcg@10", "both@10", "both@5", "recall@10")
CODE_METRICS = ("both@10", "both@5", "ndcg@10", "recall@10", "mrr@10")
EXACT_METRICS = ("exact_hit@1", "exact_hit@10", "exact_mrr",
                 "near_miss_before_exact", "negative_activation")

WORKLOADS = {
    "hotpot/test": {
        "kind": "multi_hop", "corpus": "hotpot", "split": "test",
        "declared_profile": "multi_hop_qa",
        "role": "held-out multi-hop evidence",
    },
    "scifact/test": {
        "kind": "single_hop", "corpus": "scifact", "split": "test",
        "declared_profile": None, "role": "held-out single-hop evidence",
    },
    "touche/test": {
        "kind": "single_hop", "corpus": "touche", "split": "test",
        "declared_profile": None, "role": "held-out argument retrieval",
    },
    "nfcorpus/dev": {
        "kind": "single_hop", "corpus": "nfcorpus", "split": "dev",
        "declared_profile": None, "role": "diagnostic only (+0.017 R@100 trade)",
    },
    "serp_code/250": {
        "kind": "code", "corpus": "code", "split": "synthetic",
        "declared_profile": "code_graph",
        "role": "synthetic capability test (graph-closure qrels)",
    },
}

# Which modes may be chosen by an automatic policy (= no per-query user intent).
AUTOMATIC = {
    "hotpot/test": ("default", "routed", "deep"),   # deep only via declared corpus profile
    "scifact/test": ("default", "routed"),
    "touche/test": ("default", "routed"),
    "nfcorpus/dev": ("default", "routed"),
    "serp_code/250": ("default", "graph"),
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def identifier_tokens(text: str) -> list[str]:
    """Digit/underscore identifiers, e.g. hba1c, batch2, sql_v2.

    A bare number ("1990") is not an identifier: it has no lexical variant to
    confuse and gating it would disable expansion on ordinary year questions.
    """
    found = []
    for token in TOKEN_RE.findall(text):
        if len(token) < 3:
            continue
        has_alpha = any(char.isalpha() for char in token)
        if not has_alpha:
            continue
        if "_" in token or any(char.isdigit() for char in token):
            found.append(token)
    return found


def identifier_gate(text: str) -> bool:
    """True => the exact-identifier gate blocks expansion for this query."""
    return bool(identifier_tokens(text))


# ---------------------------------------------------------------------------
# Mode boundary
# ---------------------------------------------------------------------------
def select(query: str, workload: str, mode: str) -> str:
    """The mode a request actually runs in, ignoring the safety gate.

    `routed` resolves per query; the other modes resolve per request/workload.
    `graph` is rejected off the code corpus instead of silently degrading.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "graph" and WORKLOADS[workload]["kind"] != "code":
        raise ValueError("graph mode is defined for the code corpus only")
    if mode == "routed":
        return "deep" if router.is_multihop(query) else "default"
    return mode


def ship_mode(query: str, workload: str, mode: str) -> str:
    """`select()` with the mandatory exact-identifier gate applied."""
    chosen = select(query, workload, mode)
    if chosen in ("deep", "graph") and identifier_gate(query):
        return "default"
    return chosen


def is_automatic(workload: str, mode: str) -> bool:
    return mode in AUTOMATIC[workload]


# ---------------------------------------------------------------------------
# Safety arithmetic
# ---------------------------------------------------------------------------
def safety_verdict(delta: float, ci: list[float], margin: float = MARGIN) -> str:
    """Non-inferiority at `margin`.

    pass          the whole 95% CI is above the margin.
    fail          the whole CI is below the margin (a significant loss beyond it).
    inconclusive  the CI straddles the margin: non-inferiority is *not* shown.
    """
    low, high = ci
    if high < margin:
        return "fail"
    if low > margin:
        return "pass"
    return "inconclusive"


def gate_row(delta: float, ci: list[float], margin: float = MARGIN) -> dict:
    return {
        "delta": delta,
        "ci95": list(ci),
        "margin": margin,
        "verdict": safety_verdict(delta, ci, margin),
        "significant_negative": ci[1] < 0.0,
        "significant_positive": ci[0] > 0.0,
    }


# ---------------------------------------------------------------------------
# Exact-identifier invariants
# ---------------------------------------------------------------------------
def ranked_ids(ix, scores, depth: int = 100) -> list[str]:
    import harness
    from fusion import ranked_ids as dense_ranked

    if isinstance(scores, dict):
        return harness.rank(ix, scores, depth)
    return dense_ranked(ix, scores, depth)


def evaluate_exact(ix, suite: dict, scorer, depth: int = 100) -> dict:
    """Exact hit@1/@10, MRR, near-miss-before-exact, unknown-ID activation.

    `scorer(query_text)` returns `(scores, expansion_active)`, where
    `expansion_active` is true when a non-lexical channel produced any candidate.
    That second value is the invariant exact_ids.evaluate cannot see.
    """
    hits1 = hits10 = near_before = near_top10 = 0
    reciprocal = 0.0
    positive_expansion = 0
    for row in suite["positives"]:
        scores, expanded = scorer(row["query"])
        positive_expansion += int(expanded)
        ranked = ranked_ids(ix, scores, depth)
        rel, near = set(row["relevant"]), set(row["near_miss"])
        first = next((i + 1 for i, doc in enumerate(ranked) if doc in rel), None)
        if first:
            reciprocal += 1.0 / first
        hits1 += int(first == 1)
        hits10 += int(bool(first and first <= 10))
        near_positions = [i for i, doc in enumerate(ranked) if doc in near]
        near_before += int(bool(first) and any(pos < first - 1 for pos in near_positions))
        near_top10 += int(any(pos < 10 for pos in near_positions))

    activated = unknown_expansion = 0
    for row in suite["negatives"]:
        scores, expanded = scorer(row["query"])
        activated += int(bool(ranked_ids(ix, scores, 10)))
        unknown_expansion += int(expanded)

    n_pos = max(len(suite["positives"]), 1)
    n_neg = max(len(suite["negatives"]), 1)
    return {
        "positives": len(suite["positives"]),
        "negatives": len(suite["negatives"]),
        "exact_hit@1": hits1 / n_pos,
        "exact_hit@10": hits10 / n_pos,
        "exact_mrr": reciprocal / n_pos,
        "near_miss_before_exact": near_before / n_pos,
        "near_miss_top10": near_top10 / n_pos,
        "negative_activation": activated / n_neg,
        "positive_expansion": positive_expansion / n_pos,
        "unknown_identifier_expansion": unknown_expansion / n_neg,
    }


def code_symbol_suite(ix, stem=None, max_positive: int = 60, max_negative: int = 40,
                      seed: int = 1729) -> dict:
    """Exact identifier vs near-miss over the code chunk index vocabulary.

    The same fixture logic as lab/exact_ids.py (edit-distance-one variants that
    share a prefix, random unknown identifiers as negatives), applied to the code
    chunk index instead of a text corpus. Relevant documents are the chunks that
    contain the exact term; near misses are the chunks that contain a variant but
    not the exact term. `stem` (the index's own stemmer) filters out terms the
    query pipeline would rewrite, so the fixture can never ask for a term the
    index does not store. This is a synthetic invariant suite, not relevance.
    """
    import random

    candidates = sorted(
        ((term, df) for term, df in ix.df.items()
         if 5 <= len(term) <= 40 and term.isalpha() and 1 <= df <= 10
         and (stem is None or stem(term) == term)),
        key=lambda item: (item[1], item[0]),
    )
    terms = [term for term, _ in candidates]

    def doc_ids_for(term: str) -> set[str]:
        return {ix.doc_ids[doc] for doc, _ in ix.postings.get(term, ())}

    positives = []
    for term, _ in candidates:
        variants = [
            other for other in terms
            if other != term and other[:3] == term[:3]
            and exact_ids._edit_distance_one(term, other)
        ][:5]
        if not variants:
            continue
        relevant = doc_ids_for(term)
        near = set()
        for variant in variants:
            near |= doc_ids_for(variant)
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

    rng = random.Random(seed)
    known = set(ix.df)
    negatives = []
    while len(negatives) < max_negative:
        token = f"zzq{rng.randrange(10**6):06d}"
        if token in known:
            continue
        negatives.append({"query": token, "relevant": [], "near_miss": []})
    return {"positives": positives, "negatives": negatives}
