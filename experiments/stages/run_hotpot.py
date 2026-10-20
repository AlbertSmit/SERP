#!/usr/bin/env python3
"""Entity-Bridge Retrieval on HotpotQA.

Pre-registered constants: RRF k=60, 20 BM25 seeds, 20 bridge entities, bridge df
cap 5% of paragraphs. The dev split is not used for tuning; it is reported for
transparency. The test split is the held-out evidence.

Two retrieval settings:
  global  - rank all 66,581 paragraphs
  pool    - rerank only the question's 10 distractor paragraphs (standard setting)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lab"))

import fusion
import harness
import hotpot
from hotpot import dense
from lean_units import LeanUnitBM25
from channels import porter

SYSTEMS = (
    "bm25", "porter", "entity", "bridge",
    "rrf_porter_bridge", "rrf_porter_entity", "rrf_porter_bridge_entity",
)


def query_systems(index, qid, porter_system):
    ix = index["ix"]
    query = index["queries"][qid]
    word = harness.bm25_fast(ix, query)
    porter_scores = porter_system.score(query)
    entities = hotpot.query_entities(index["raw_queries"][qid], index["title_index"])
    entity_scores = hotpot.per_entity_scores(index, qid, entities)
    bridges, _ = hotpot.bridge_scores(index, qid, porter_scores)
    entity_dense = dense(ix, entity_scores, index["position"])
    bridge_dense = dense(ix, bridges, index["position"])
    return {
        "bm25": word,
        "porter": porter_scores,
        "entity": entity_dense,
        "bridge": bridge_dense,
        "rrf_porter_bridge": fusion.rrf(ix, [porter_scores, bridge_dense]),
        "rrf_porter_entity": fusion.rrf(ix, [porter_scores, entity_dense]),
        "rrf_porter_bridge_entity": fusion.rrf(
            ix, [porter_scores, bridge_dense, entity_dense],
        ),
    }


def ranked_from_scores(index, scores, pool=None):
    ix = index["ix"]
    if pool is None:
        return harness.rank(ix, scores, 1000)
    return sorted(
        pool,
        key=lambda doc: (-scores[index["position"][doc]], doc),
    )


def evaluate(index, split, porter_system):
    global_rows = {label: [] for label in SYSTEMS}
    pool_rows = {label: [] for label in SYSTEMS}
    for qid, qrels in index["qrels"].items():
        if not hotpot.in_split(qid, split):
            continue
        systems = query_systems(index, qid, porter_system)
        candidates = index["meta"][qid]["context"]
        for label, scores in systems.items():
            for target, pool in ((global_rows, None), (pool_rows, candidates)):
                ranked = ranked_from_scores(index, scores, pool)
                row = hotpot.metrics(ranked, qrels)
                row["qid"] = qid
                row["type"] = index["meta"][qid]["type"]
                target[label].append(row)
    return global_rows, pool_rows


def print_table(title, summary):
    print(f"\n== {title}")
    header = f"{'system':28} {'R@2':>6} {'R@5':>6} {'R@10':>6} {'both@2':>7} {'both@5':>7} {'both@10':>8} {'nDCG@10':>8} {'MRR@10':>7}"
    print(header)
    for label, row in summary.items():
        print(
            f"{label:28} {row['recall@2']:6.3f} {row['recall@5']:6.3f} {row['recall@10']:6.3f} "
            f"{row['both@2']:7.3f} {row['both@5']:7.3f} {row['both@10']:8.3f} "
            f"{row['ndcg@10']:8.3f} {row['mrr@10']:7.3f}"
        )


def print_paired(label, report):
    print(f"\n  paired vs {label}")
    for metric, row in report.items():
        lo, hi = row["ci95"]
        w, t, l = row["wtl"]
        print(f"    {metric:10} Δ={row['delta']:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  W/T/L={w}/{t}/{l}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(HERE / "data/hotpot_dev.parquet"))
    parser.add_argument("--out", default=str(HERE / "results/hotpot.json"))
    args = parser.parse_args()

    index = hotpot.load(args.data)
    print(f"[load] {index['ix'].N} paragraphs, {len(index['qrels'])} questions", flush=True)
    porter_system = LeanUnitBM25(index["ix"], porter)
    report = {}
    for split in ("dev", "test"):
        global_rows, pool_rows = evaluate(index, split, porter_system)
        for setting, rows in (("global", global_rows), ("pool", pool_rows)):
            summary = hotpot.summarize(rows)
            report[f"{setting}:{split}"] = {
                "summary": summary,
                "paired": {
                    "rrf_porter_bridge_entity_vs_porter": hotpot.paired_report(
                        rows["porter"], rows["rrf_porter_bridge_entity"],
                    ),
                    "rrf_porter_bridge_entity_vs_bm25": hotpot.paired_report(
                        rows["bm25"], rows["rrf_porter_bridge_entity"],
                    ),
                    "rrf_porter_bridge_vs_porter": hotpot.paired_report(
                        rows["porter"], rows["rrf_porter_bridge"],
                    ),
                    "rrf_porter_entity_vs_porter": hotpot.paired_report(
                        rows["porter"], rows["rrf_porter_entity"],
                    ),
                },
            }
            print_table(f"HotpotQA {split} ({setting})", summary)
            print_paired("porter", report[f"{setting}:{split}"]["paired"]["rrf_porter_bridge_entity_vs_porter"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
