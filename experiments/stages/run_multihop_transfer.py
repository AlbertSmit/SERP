#!/usr/bin/env python3
"""A1: frozen multi-hop transfer test for Entity-Bridge Retrieval.

Runs the unchanged HotpotQA EBR composition on 2WikiMultiHopQA and MuSiQue dev.
No tuning, no router, no typed filtering. `all@k` is the primary metric; these are
evidence-completeness datasets, not ordinary relevance collections.

Usage:
  PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_multihop_transfer.py
  PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_multihop_transfer.py --limit 200

MuSiQue is evaluated twice: `musique` keeps the superseded title-only identity and
`musique_paragraph` uses (title, paragraph_text). Only the latter counts toward the
gate. PYTHONHASHSEED is pinned because the unmodified bridge/entity channels sum
weights in set-iteration order.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lab"))

import multihop_transfer as mt

LOADERS = {
    "2wiki": ("2wiki_dev.parquet", mt.load_2wiki),
    "musique": ("musique_dev.jsonl", mt.load_musique),
    "musique_paragraph": ("musique_dev.jsonl", mt.load_musique_paragraphs),
}
IDENTITY = {
    "2wiki": "title",
    "musique": "title (superseded; kept for comparison)",
    "musique_paragraph": "title+paragraph_text",
}
PRIMARY = ("2wiki", "musique_paragraph")
HEADER = (
    f"{'system':10} {'all@5':>6} {'all@10':>7} {'R@10':>6} {'R@100':>7} "
    f"{'nDCG@10':>8} {'MRR':>6}"
)


def dataset_block(index, stats, load_seconds, limit, name):
    started = time.perf_counter()
    rows, reach, latency = mt.evaluate(index, limit=limit)
    elapsed = time.perf_counter() - started

    summary = {system: mt.summarize(rows, system) for system in mt.SYSTEMS}
    paired = {
        "bridge": mt.paired_vs(rows, "porter", "bridge"),
        "entity": mt.paired_vs(rows, "porter", "entity"),
        "ebr": mt.paired_vs(rows, "porter", "ebr"),
    }
    slices = mt.all_slices(rows)
    latency = {
        **latency,
        "index_build_seconds": load_seconds,
        "evaluate_seconds": elapsed,
    }
    block = {
        "document_identity": IDENTITY[name],
        "stats": {**stats, "evaluated_questions": summary["porter"]["queries"]},
        "summary": summary,
        "paired_vs_porter": paired,
        "slices": slices,
        "bridge_reach": reach,
        "latency": latency,
    }
    return block, rows


def pooled_block(rows_by_dataset):
    rows = {system: [] for system in mt.SYSTEMS}
    for block in rows_by_dataset.values():
        for system in mt.SYSTEMS:
            rows[system].extend(block[system])
    return {
        "queries": len(rows["porter"]),
        "datasets": list(rows_by_dataset),
        "summary": {system: mt.summarize(rows, system) for system in mt.SYSTEMS},
        "paired_vs_porter": {
            "bridge": mt.paired_vs(rows, "porter", "bridge"),
            "entity": mt.paired_vs(rows, "porter", "entity"),
            "ebr": mt.paired_vs(rows, "porter", "ebr"),
        },
    }


def print_dataset(name, block):
    stats = block["stats"]
    print(f"\n== {name} [{block['document_identity']}]: {stats['evaluated_questions']} questions, "
          f"{stats['paragraphs']} paragraphs, {stats['title_collisions']} title collisions", flush=True)
    if "missing_support_titles" in stats:
        print(f"   missing support titles: {stats['missing_support_titles']}, "
              f"collapsed support titles: {stats['questions_with_collapsed_support_titles']}")
    else:
        print(f"   slots {stats['paragraph_slots']}, duplicate slots {stats['duplicate_paragraph_slots']}, "
              f"titles with multiple paragraphs {stats['titles_with_multiple_paragraphs']}, "
              f"missing support paragraphs {stats['missing_support_paragraphs']}, "
              f"questions with same-title confuser {stats['questions_with_same_title_confuser']}")
    print(HEADER)
    for system in mt.SYSTEMS:
        row = block["summary"][system]
        print(f"{system:10} {row['all@5']:6.3f} {row['all@10']:7.3f} {row['recall@10']:6.3f} "
              f"{row['recall@100']:7.3f} {row['ndcg@10']:8.3f} {row['mrr']:6.3f}")

    print(f"\n  paired vs porter")
    for candidate, report in block["paired_vs_porter"].items():
        for metric in ("all@5", "all@10", "recall@10", "recall@100", "ndcg@10", "mrr"):
            row = report[metric]
            lo, hi = row["ci95"]
            w, t, l = row["wtl"]
            print(f"    {candidate:7} {metric:10} Δ={row['delta']:+.4f} "
                  f"CI95=[{lo:+.4f},{hi:+.4f}] W/T/L={w}/{t}/{l}")

    print(f"\n  slices")
    for slice_name in block["slices"]["porter"]:
        n = block["slices"]["porter"][slice_name]["queries"]
        porter = block["slices"]["porter"][slice_name]
        bridge = block["slices"]["bridge"][slice_name]
        ebr = block["slices"]["ebr"][slice_name]
        print(f"    {slice_name} (n={n}): "
              f"porter all@10={porter['all@10']:.3f} R@10={porter['recall@10']:.3f} | "
              f"bridge all@10={bridge['all@10']:.3f} | "
              f"ebr all@10={ebr['all@10']:.3f} R@10={ebr['recall@10']:.3f} "
              f"nDCG={ebr['ndcg@10']:.3f}")

    reach = block["bridge_reach"]
    print(f"\n  bridge reach: {reach['bridge_reach_fraction']:.3f} of gold outside porter top-20 "
          f"retrieved at all, {reach['bridge_reach_top10_fraction']:.3f} in top-10; "
          f"gold in porter top-100 {reach['fraction_gold_in_porter_top100']:.3f}; "
          f"extra ebr candidates {reach['extra_ebr_candidates_mean']:.1f}/query")
    latency = block["latency"]
    print(f"  latency: porter {latency['porter_ms_per_query']:.2f} ms/query, "
          f"ebr {latency['ebr_ms_per_query']:.2f} ms/query, "
          f"index build {latency['index_build_seconds']:.1f} s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(HERE / "data"))
    parser.add_argument("--out", default=str(HERE / "results/multihop_transfer.json"))
    parser.add_argument("--datasets", default="2wiki,musique,musique_paragraph")
    parser.add_argument("--limit", type=int, default=0,
                        help="wiring slice only: evaluate the first N questions per dataset")
    args = parser.parse_args()

    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    report = {
        "experiment": "A1 multi-hop transfer, frozen EBR",
        "frozen_constants": {
            "bm25": {"k1": 1.2, "b": 0.75, "stemming": "porter"},
            "first_hop_seeds": 20,
            "bridge_entities": 20,
            "entity_df_cap_fraction": 0.05,
            "rrf_k": 60,
            "depth": 1000,
            "channels": ["porter", "bridge", "per_entity"],
            "typed_filtering": False,
            "router": False,
            "parameter_search": False,
        },
        "eval_split": "dev (labeled); test is unlabeled for 2Wiki and unused; no tuning split",
        "wiring_slice": args.limit or None,
        "environment": {
            "python": platform.python_version(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
            "note": (
                "The unmodified bridge/entity channels accumulate weights in set-iteration "
                "order, so results drift by <3e-4 unless PYTHONHASHSEED is pinned. Run with "
                "PYTHONHASHSEED=0 for a bit-identical reproduction."
            ),
        },
        "datasets": {},
    }

    rows_by_dataset = {}
    for name in names:
        filename, loader = LOADERS[name]
        started = time.perf_counter()
        index, stats = loader(str(Path(args.data_dir) / filename))
        load_seconds = time.perf_counter() - started
        print(f"[load] {name}: {stats['paragraphs']} paragraphs "
              f"({stats['title_collisions']} title collisions), "
              f"{stats['questions']} questions, {load_seconds:.1f}s", flush=True)
        block, rows = dataset_block(index, stats, load_seconds, args.limit or None, name)
        report["datasets"][name] = block
        rows_by_dataset[name] = rows
        print_dataset(name, block)

    gate_names = [name for name in PRIMARY if name in report["datasets"]]
    if not gate_names:
        gate_names = list(report["datasets"])
    report["acceptance_basis"] = gate_names
    report["pooled_basis"] = gate_names
    report["pooled"] = pooled_block({name: rows_by_dataset[name] for name in gate_names})
    report["acceptance"] = mt.acceptance({name: report["datasets"][name] for name in gate_names})

    pooled = report["pooled"]
    print(f"\n== pooled: {pooled['queries']} questions")
    print(HEADER)
    for system in mt.SYSTEMS:
        row = pooled["summary"][system]
        print(f"{system:10} {row['all@5']:6.3f} {row['all@10']:7.3f} {row['recall@10']:6.3f} "
              f"{row['recall@100']:7.3f} {row['ndcg@10']:8.3f} {row['mrr']:6.3f}")
    for candidate, metrics in pooled["paired_vs_porter"].items():
        for metric in ("all@10", "recall@10", "ndcg@10"):
            row = metrics[metric]
            lo, hi = row["ci95"]
            print(f"  {candidate:7} {metric:10} Δ={row['delta']:+.4f} CI95=[{lo:+.4f},{hi:+.4f}]")

    gate = report["acceptance"]
    print(f"\n[gate] passed={gate['passed']} verdict={gate['verdict']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
