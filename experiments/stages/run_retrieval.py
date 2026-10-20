#!/usr/bin/env python3
"""Stage 1+2: retrieval channels and fixed RRF on a labelled corpus.

No tuning. Every channel is built once, scored greedily, and compared with the
Porter-stemmed BM25 baseline that the earlier research found fair.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lab"))

import channels as channel_lib
import fusion
import harness
import metrics

DEFAULT_CHANNELS = ("bm25", "porter", "char4", "vc", "vcgrams")


def self_ids_for(ix, queries):
    position = {doc: index for index, doc in enumerate(ix.doc_ids)}
    return {qid: position[qid] for qid in queries if qid in position}


def evaluate_dataset(name, channel_names):
    print(f"\n[load] {name}", flush=True)
    ix, queries, qrels = harness.load(name)
    print(f"[load] {name}: {ix.N} docs, {len(queries)} queries", flush=True)
    self_ids = self_ids_for(ix, queries)

    print(f"[channels] building {channel_names}", flush=True)
    built = channel_lib.build(ix, list(channel_names))
    print("[channels] ready", flush=True)

    runs = {}
    tops = {}
    for label, channel in built.items():
        start = time.perf_counter()
        per_query, top = metrics.evaluate_ranked(
            ix, queries, qrels, channel.score, depth=1000, self_ids=self_ids,
        )
        runs[label] = per_query
        tops[label] = top
        print(f"  {label:12} {time.perf_counter() - start:7.1f}s nDCG={metrics.mean(per_query, 'ndcg'):.4f}", flush=True)

    # Fixed fusion definitions, declared before looking at the results.
    if "porter" in runs:
        fusions = {
            "rrf_porter_char4": ["porter", "char4"],
            "rrf_porter_vc": ["porter", "vc"],
            "rrf_porter_vcgrams": ["porter", "vcgrams"],
            "rrf_porter_char4_vc": ["porter", "char4", "vc"],
            "rrf_porter_char4_vcgrams": ["porter", "char4", "vcgrams"],
        }
        for label, members in fusions.items():
            if not all(member in built for member in members):
                continue
            start = time.perf_counter()
            per_query, top = {}, {}
            for qid, query in queries.items():
                scores = fusion.rrf(ix, [built[member].score(query) for member in members])
                if qid in self_ids:
                    scores[self_ids[qid]] = 0.0
                ranked = harness.rank(ix, scores, 1000)
                per_query[qid] = harness.metrics(ranked, qrels[qid])
                top[qid] = ranked
            runs[label] = per_query
            tops[label] = top
            print(f"  {label:26} {time.perf_counter() - start:7.1f}s nDCG={metrics.mean(per_query, 'ndcg'):.4f}", flush=True)

    baseline = "porter" if "porter" in runs else next(iter(runs))
    summary = metrics.summarize(runs, baseline)
    for label in runs:
        if label == baseline:
            continue
        transition = metrics.stage_transition(tops[baseline], tops[label], qrels, depth=100)
        metrics.add_stage(summary, label, transition)

    return {
        "dataset": name,
        "documents": ix.N,
        "queries": len(queries),
        "baseline": baseline,
        "runs": summary,
        "config": {"channels": list(channel_names), "rrf_k": fusion.RRF_K, "depth": 1000},
    }


def print_table(report):
    print(f"\n== {report['dataset']} ({report['queries']} queries, baseline={report['baseline']})")
    header = f"{'run':28} {'nDCG@10':>8} {'dNDCG':>8} {'CI95':>18} {'R@100':>7} {'dR@100':>8} {'MRR':>7} {'new':>5} {'lost':>5}"
    print(header)
    for label, row in report["runs"].items():
        stage = row.get("stage", {})
        delta = row.get("delta_ndcg", 0.0)
        ci = row.get("ci95_ndcg", [0.0, 0.0])
        print(
            f"{label:28} {row['ndcg']:8.4f} {delta:+8.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
            f"{row['r100']:7.4f} {row.get('delta_r100', 0.0):+8.4f} {row['mrr']:7.4f} "
            f"{stage.get('new_relevant', 0):5} {stage.get('lost_relevant', 0):5}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="touche,scifact")
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    parser.add_argument("--out", default=str(HERE / "results/retrieval.json"))
    args = parser.parse_args()

    channel_names = [name.strip() for name in args.channels.split(",") if name.strip()]
    report = {}
    for name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        report[name] = evaluate_dataset(name, channel_names)
        print_table(report[name])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
