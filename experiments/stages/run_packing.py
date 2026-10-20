#!/usr/bin/env python3
"""Stage 3: does packing preserve the relevant documents retrieval found?"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lab"))

import channels as channel_lib
import fusion
import harness
import pack


def self_ids_for(ix, queries):
    position = {doc: index for index, doc in enumerate(ix.doc_ids)}
    return {qid: position[qid] for qid in queries if qid in position}


def evaluate(name, budget, pool=100):
    ix, queries, qrels = harness.load(name)
    self_ids = self_ids_for(ix, queries)
    built = channel_lib.build(ix, ["porter", "char4"])
    lengths = {doc_id: ix.dl[index] for index, doc_id in enumerate(ix.doc_ids)}

    rows = {"truncate": [], "greedy": [], "residual": []}
    for qid, query in queries.items():
        fused = fusion.rrf(ix, [built["porter"].score(query), built["char4"].score(query)])
        if qid in self_ids:
            fused[self_ids[qid]] = 0.0
        ranked = harness.rank(ix, fused, 1000)
        rel = qrels[qid]
        tables = {
            "truncate": pack.truncate(ranked, lengths, budget),
            "greedy": pack.greedy_coverage(ranked, query, ix, budget, pool=pool),
            "residual": pack.residual(ranked, query, ix, budget, pool=pool),
        }
        for label, selected in tables.items():
            rows[label].append(pack.metrics(selected, rel, ranked, depth=100))

    summary = {}
    for label, results in rows.items():
        summary[label] = {
            "recall": sum(row["recall"] for row in results) / max(len(results), 1),
            "retention": sum(row["retention"] for row in results) / max(len(results), 1),
            "selected": sum(row["selected"] for row in results) / max(len(results), 1),
            "selected_relevant": sum(row["selected_relevant"] for row in results) / max(len(results), 1),
        }
    return {"dataset": name, "budget": budget, "queries": len(queries), "runs": summary}


def print_table(report):
    print(f"\n== {report['dataset']} budget={report['budget']} ({report['queries']} queries)")
    print(f"{'packer':12} {'recall':>8} {'retention':>10} {'selected':>9} {'relevant':>9}")
    for label, row in report["runs"].items():
        print(f"{label:12} {row['recall']:8.4f} {row['retention']:10.4f} {row['selected']:9.2f} {row['selected_relevant']:9.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="scifact,touche")
    parser.add_argument("--budgets", default="500,1000,2000")
    parser.add_argument("--out", default=str(HERE / "results/packing.json"))
    args = parser.parse_args()

    report = {}
    for name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        for budget in [int(item) for item in args.budgets.split(",") if item.strip()]:
            key = f"{name}:{budget}"
            report[key] = evaluate(name, budget)
            print_table(report[key])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
