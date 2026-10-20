"""Stage metrics. Retrieval quality and stage transitions are reported separately."""
from __future__ import annotations

from collections import defaultdict

import harness


def evaluate_ranked(ix, queries, qrels, score_fn, *, depth=1000, self_ids=None):
    """Returns per-query metrics plus the top-`depth` doc ids for stage analysis."""
    per_query = {}
    top_docs = {}
    for qid, query in queries.items():
        scores = score_fn(query)
        if self_ids and qid in self_ids:
            scores[self_ids[qid]] = 0.0
        ranked = harness.rank(ix, scores, depth)
        per_query[qid] = harness.metrics(ranked, qrels[qid])
        top_docs[qid] = ranked
    return per_query, top_docs


def mean(per_query, metric):
    return harness._mean(per_query, metric)


def paired(base, candidate, metric="ndcg"):
    return harness.paired(base, candidate, metric=metric)


def stage_transition(base_top: dict, candidate_top: dict, qrels: dict, depth: int = 100) -> dict:
    """Did the candidate retrieve relevant documents the baseline missed, and vice versa?"""
    rows = []
    for qid, rel in qrels.items():
        if not rel:
            continue
        base_rel = set(base_top.get(qid, [])[:depth]) & set(rel)
        cand_rel = set(candidate_top.get(qid, [])[:depth]) & set(rel)
        rows.append({
            "qid": qid,
            "base": len(base_rel),
            "candidate": len(cand_rel),
            "new": len(cand_rel - base_rel),
            "lost": len(base_rel - cand_rel),
            "union": len(base_rel | cand_rel),
            "unique_new": len(cand_rel - base_rel),
        })
    total = {
        "queries": len(rows),
        "base_relevant": sum(r["base"] for r in rows),
        "candidate_relevant": sum(r["candidate"] for r in rows),
        "new_relevant": sum(r["new"] for r in rows),
        "lost_relevant": sum(r["lost"] for r in rows),
        "union_relevant": sum(r["union"] for r in rows),
        "queries_with_new": sum(1 for r in rows if r["new"] > 0),
        "queries_with_loss": sum(1 for r in rows if r["lost"] > 0),
        "queries_with_net_gain": sum(1 for r in rows if r["new"] > r["lost"]),
        "queries_with_net_loss": sum(1 for r in rows if r["lost"] > r["new"]),
    }
    return {"depth": depth, "total": total, "rows": rows}


def summarize(runs: dict, base_label: str, metrics=("ndcg", "r100", "mrr")) -> dict:
    """Mean metrics plus paired deltas against a baseline label."""
    out = {}
    base = runs[base_label]
    for label, per_query in runs.items():
        row = {metric: mean(per_query, metric) for metric in metrics}
        if label != base_label:
            for metric in metrics:
                delta, low, high, win, tie, loss = paired(base, per_query, metric=metric)
                row[f"delta_{metric}"] = delta
                row[f"ci95_{metric}"] = [low, high]
                row[f"wtl_{metric}"] = [win, tie, loss]
        out[label] = row
    return out


def add_stage(summary: dict, label: str, stage: dict):
    summary.setdefault(label, {})["stage"] = stage["total"]
    return summary
