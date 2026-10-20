#!/usr/bin/env python3
"""Exact-identifier and negative-query regression suite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lab"))

import channels as channel_lib
import exact_ids
import fusion
import harness
import metrics


def evaluate(name):
    ix, queries, qrels = harness.load(name)
    del queries, qrels
    suite = exact_ids.build_suite(ix)
    built = channel_lib.build(ix, ["bm25", "porter", "char4", "vcgrams"])

    systems = {label: channel.score for label, channel in built.items()}
    systems["rrf_porter_char4"] = lambda query: fusion.rrf(
        ix, [built["porter"].score(query), built["char4"].score(query)],
    )
    systems["rrf_porter_vcgrams"] = lambda query: fusion.rrf(
        ix, [built["porter"].score(query), built["vcgrams"].score(query)],
    )
    systems["rrf_porter_char4_vcgrams"] = lambda query: fusion.rrf(
        ix, [built["porter"].score(query), built["char4"].score(query), built["vcgrams"].score(query)],
    )

    report = {"dataset": name, "suite": {"positives": len(suite["positives"]), "negatives": len(suite["negatives"])}}
    for label, score_fn in systems.items():
        report[label] = exact_ids.evaluate(ix, suite, score_fn)
    return report


def print_table(report):
    print(f"\n== {report['dataset']} exact-ID suite ({report['suite']['positives']} positives, {report['suite']['negatives']} negatives)")
    print(f"{'system':28} {'exactHit':>9} {'exactMRR':>9} {'nearFirst':>10} {'near@10':>8} {'negAct':>7}")
    for label, row in report.items():
        if label in ("dataset", "suite"):
            continue
        print(f"{label:28} {row['exact_hit_rate']:9.3f} {row['exact_mrr']:9.3f} {row['near_miss_before_exact']:10.3f} {row['near_miss_top10']:8.3f} {row['negative_activation']:7.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="scifact")
    parser.add_argument("--out", default=str(HERE / "results/exact_ids.json"))
    args = parser.parse_args()

    report = {}
    for name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        report[name] = evaluate(name)
        print_table(report[name])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
