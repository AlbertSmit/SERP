#!/usr/bin/env python3
"""CESR-1: constrained evidence-set retrieval, oracle-constraint ceiling test.

Loads 2WikiMultiHopQA dev and MuSiQue dev (paragraph identity), builds gold
constraints, and compares CESR against Porter BM25 truncation and EBR truncation at
500/1000/2000 engine tokens. No tuning on test: budgets, mu, pair bonus and the cost
exponent are frozen in `lab/cesr.py`.

Usage:
  PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_cesr.py
  PYTHONHASHSEED=0 .venv/bin/python experiments/stages/run_cesr.py --limit 100
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

import cesr
import multihop_transfer as mt

LOADERS = {
    "2wiki": ("2wiki_dev.parquet", mt.load_2wiki, cesr.load_2wiki_texts, cesr.load_2wiki_constraints),
    "musique_paragraph": (
        "musique_dev.jsonl", mt.load_musique_paragraphs,
        cesr.load_musique_texts, cesr.load_musique_constraints,
    ),
}
IDENTITY = {
    "2wiki": "title",
    "musique_paragraph": "title+paragraph_text",
}
CONSTRAINT_SOURCE = {
    "2wiki": "row.evidences triples [subject, relation, object]",
    "musique_paragraph": "row.question_decomposition sub-questions + answers (paragraph_support_idx ignored)",
}
HEADER = (
    f"{'system':22} {'all@5':>6} {'all@10':>7} {'recall':>7} {'cover':>6} "
    f"{'redun':>6} {'tokens':>7} {'n':>5} {'nDCG@10':>8}"
)


def dataset_block(name, index, texts, specs, limit):
    started = time.perf_counter()
    rows, reach, latency, provenance = cesr.evaluate(index, texts, specs, limit=limit)
    elapsed = time.perf_counter() - started

    summary = {
        system: {budget: cesr.summarize(rows[system][budget]) for budget in cesr.BUDGETS}
        for system in cesr.SYSTEMS
    }
    paired = {
        budget: cesr.paired(rows["ebr_trunc"][budget], rows["cesr"][budget])
        for budget in cesr.BUDGETS
    }
    token_ratio = {
        budget: summary["cesr"][budget]["tokens"] / max(summary["ebr_trunc"][budget]["tokens"], 1e-9)
        for budget in cesr.BUDGETS
    }
    slices = {
        system: {budget: cesr.slice_summary(rows[system][budget]) for budget in cesr.BUDGETS}
        for system in cesr.SYSTEMS
    }
    block = {
        "document_identity": IDENTITY[name],
        "constraint_source": CONSTRAINT_SOURCE[name],
        "stats": {**index["stats"], "evaluated_questions": summary["porter_trunc"][cesr.BUDGETS[0]]["queries"]},
        "summary": summary,
        "paired_vs_ebr": paired,
        "token_ratio_cesr_over_ebr": token_ratio,
        "slices": slices,
        "candidate_pool": reach,
        "latency": {**latency, "evaluate_seconds": elapsed},
        "provenance": {
            budget: {
                "channel_sources": dict(provenance[budget]["channel_sources"]),
                "constraints_covered_per_step": dict(provenance[budget]["steps"]),
                "samples": provenance[budget]["samples"],
            }
            for budget in cesr.BUDGETS
        },
    }
    return block, rows


def print_block(name, block):
    stats = block["stats"]
    print(f"\n== {name} [{block['document_identity']}]: {stats['evaluated_questions']} questions, "
          f"{stats['paragraphs']} paragraphs", flush=True)
    reach = block["candidate_pool"]
    print(f"   candidate pool: {reach['gold_in_candidate_pool']}/{reach['gold_paragraphs']} gold "
          f"paragraphs present (ceiling {reach['candidate_recall_ceiling']:.3f}); all gold present "
          f"for {reach['queries_with_all_gold_in_pool_rate']:.3f} of queries; mean achievable "
          f"constraint coverage {reach['mean_achievable_constraint_coverage']:.3f}; fully coverable "
          f"for {reach['queries_fully_coverable_from_pool_rate']:.3f}")
    for budget in cesr.BUDGETS:
        print(f"   budget {budget}: CESR covers all constraints in "
              f"{reach['cesr_full_coverage_rate'][budget]:.3f} of queries (coverage-first "
              f"{reach['coverage_first_full_coverage_rate'][budget]:.3f}, overspends the budget in "
              f"{reach['coverage_first_overspend_rate'][budget]:.3f})")
    for budget in cesr.BUDGETS:
        print(f"\n   budget {budget} tokens")
        print("   " + HEADER)
        for system in cesr.SYSTEMS:
            row = block["summary"][system][budget]
            print(f"   {system:22} {row['all@5']:6.3f} {row['all@10']:7.3f} {row['recall']:7.3f} "
                  f"{row['constraint_coverage']:6.3f} {row['redundancy']:6.3f} "
                  f"{row['tokens']:7.1f} {row['selected']:5.1f} {row['ndcg@10']:8.3f}")
        print(f"   paired CESR - EBR (tokens ratio {block['token_ratio_cesr_over_ebr'][budget]:.2f})")
        for metric in ("all@5", "all@10", "recall", "constraint_coverage", "redundancy", "ndcg@10"):
            row = block["paired_vs_ebr"][budget][metric]
            low, high = row["ci95"]
            wins, ties, losses = row["wtl"]
            print(f"     {metric:20} Δ={row['delta']:+.4f} CI95=[{low:+.4f},{high:+.4f}] "
                  f"W/T/L={wins}/{ties}/{losses}")
    latency = block["latency"]
    print(f"   latency: retrieval {latency['retrieval_ms_per_query']:.2f} ms/q, "
          f"constraints {latency['constraint_ms_per_query']:.2f} ms/q, "
          f"CESR select {latency['cesr_select_ms_per_query']:.2f} ms/q/budget, "
          f"coverage-first select {latency['coverage_first_ms_per_query']:.2f} ms/q + "
          f"fill {latency['coverage_fill_ms_per_query']:.2f} ms/q/budget, "
          f"evaluate {latency['evaluate_seconds']:.0f}s", flush=True)


def print_slices(name, block):
    print(f"\n   slices {name} (all@10 / recall / constraint coverage)")
    for slice_name, values in block["slices"]["cesr"][cesr.BUDGETS[0]].items():
        cells = []
        for budget in cesr.BUDGETS:
            ebr = block["slices"]["ebr_trunc"][budget][slice_name]
            ces = block["slices"]["cesr"][budget][slice_name]
            cells.append(f"{budget}: ebr {ebr['all@10']:.3f}/{ebr['recall']:.3f}/{ebr['constraint_coverage']:.3f} "
                         f"cesr {ces['all@10']:.3f}/{ces['recall']:.3f}/{ces['constraint_coverage']:.3f}")
        print(f"     {slice_name} (n={values['queries']}) " + " | ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(HERE / "data"))
    parser.add_argument("--out", default=str(HERE / "results/cesr.json"))
    parser.add_argument("--datasets", default="2wiki,musique_paragraph")
    parser.add_argument("--limit", type=int, default=0,
                        help="wiring slice only: evaluate the first N questions per dataset")
    args = parser.parse_args()

    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    report = {
        "experiment": "CESR-1: constrained evidence-set retrieval (oracle constraints)",
        "frozen_constants": {
            "budgets": list(cesr.BUDGETS),
            "candidate_pool": {
                "channels": ["porter_bm25", "untyped_bridge", "per_entity"],
                "depth_per_channel": cesr.POOL_DEPTH,
                "bm25": {"k1": 1.2, "b": 0.75, "stemming": "porter"},
                "bridge_entities": 20,
                "first_hop_seeds": 20,
                "entity_df_cap_fraction": 0.05,
                "rrf_k": 60,
            },
            "selector": {
                "score": "marginal_constraint_weight / max(dl+10,1)^0.35 - 0.2*max_jaccard + 0.25*pair_weight",
                "mu_redundancy": cesr.MU,
                "cost_exponent": cesr.COST_EXPONENT,
                "pair_bonus": cesr.PAIR_BONUS,
                "redundancy": "entity-set Jaccard against the selected set",
                "stop": "all constraints covered, no candidate fits the budget, or best score <= 0",
                "coverage_first": "same objective with cost exponent 0 until coverage is complete, then EBR rank fill",
            },
            "constraints": {
                "label": "oracle-constraint CESR (upper bound, not deployable)",
                "entity_weight": "EBR entity IDF where available, else 1.0, normalised to sum 1",
                "relation_weight": "1.0, normalised to sum 1",
                "coverage": "normalised entity string is an exact substring of normalised title+text",
                "relation_synonyms": cesr.RELATION_SYNONYMS,
                "musique_cue_vocabulary": list(cesr.MUSIQUE_CUE_VOCAB),
            },
            "bootstrap": {"resamples": cesr.RESAMPLES, "seed": cesr.BOOTSTRAP_SEED},
        },
        "eval_split": "dev (labeled); no tuning split; constants fixed before the run",
        "wiring_slice": args.limit or None,
        "environment": {
            "python": platform.python_version(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
            "device": "cpu",
        },
        "datasets": {},
    }

    token_ratio = {}
    for name in names:
        filename, loader, text_loader, constraint_loader = LOADERS[name]
        path = str(Path(args.data_dir) / filename)
        started = time.perf_counter()
        index, stats = loader(path)
        texts = text_loader(path)
        specs = constraint_loader(path, index["title_index"])
        index["stats"] = stats
        load_seconds = time.perf_counter() - started
        missing = set(specs) - set(index["qrels"])
        if missing:
            raise SystemExit(f"cesr: {len(missing)} constraints without a query ({name})")
        print(f"[load] {name}: {stats['paragraphs']} paragraphs, {stats['questions']} questions, "
              f"{load_seconds:.1f}s", flush=True)
        block, _ = dataset_block(name, index, texts, specs, args.limit or None)
        report["datasets"][name] = block
        for budget, ratio in block["token_ratio_cesr_over_ebr"].items():
            token_ratio[(name, budget)] = ratio
        print_block(name, block)
        print_slices(name, block)

    report["gate"] = cesr.acceptance(report["datasets"], token_ratio)
    gate = report["gate"]
    print(f"\n[gate] passed={gate['passed']}")
    print(f"       budgets beating EBR on all@10 for both datasets: {gate['budgets_beating_ebr']}")
    print(f"       budgets matching all@10 with <=80% tokens: {gate['budgets_matching_with_fewer_tokens']}")
    print(f"       verdict: {gate['verdict']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
