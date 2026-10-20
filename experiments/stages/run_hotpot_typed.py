#!/usr/bin/env python3
"""Relation-typed Entity-Bridge Retrieval on HotpotQA.

Compares the frozen untyped EBR system against two typed variants and a set of
ablations. Constants are inherited from hotpot.py and were fixed before the test
run: RRF k=60, 20 BM25 seeds, 20 bridges, 5% bridge df cap, 1000-deep channels.
The dev split is a sanity check, never a tuning signal.

    .venv/bin/python experiments/stages/run_hotpot_typed.py
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
import hotpot_typed
from hotpot import dense
from lean_units import LeanUnitBM25
from channels import porter

SYSTEMS = (
    "porter",
    "ebr_untyped",
    "rrf_porter_bridge_untyped",
    "untyped_bridge_only",
    "ebr_typed",
    "ebr_typed_link",
    "ebr_typed_plus_coverage",
    "ebr_typed_link_plus_coverage",
    "ebr_typed_anchor",
    "ebr_typed_anchor_plus_coverage",
    "typed_bridge_only",
)
METRIC_KEYS = (
    "ndcg@10", "both@10", "both@5", "recall@10",
    "recall@2", "recall@5", "both@2", "mrr@10",
)
REPORT_METRICS = ("ndcg@10", "both@10", "both@5", "recall@10")
MAIN_SYSTEMS = (
    "porter", "ebr_untyped", "rrf_porter_bridge_untyped", "ebr_typed",
    "ebr_typed_plus_coverage", "ebr_typed_anchor_plus_coverage",
)


def query_systems(data, qid, porter_system):
    ix = data["ix"]
    query = data["queries"][qid]
    porter_scores = porter_system.score(query)

    entities = hotpot.query_entities(data["raw_queries"][qid], data["title_index"])
    coverage = dense(ix, hotpot.per_entity_scores(data, qid, entities), data["position"])

    untyped_scores, untyped_top = hotpot.bridge_scores(data, qid, porter_scores)
    untyped_dense = dense(ix, untyped_scores, data["position"])

    typed, typed_top, stats = hotpot_typed.typed_bridge_scores(
        data, qid, porter_scores, untyped=(untyped_scores, untyped_top),
    )
    typed_dense = dense(ix, typed["union"], data["position"])
    link_dense = dense(ix, typed["link"], data["position"])
    anchor_dense = dense(ix, typed["anchor"], data["position"])

    # Second-hop reach: did the bridge channel retrieve a supporting paragraph
    # that first-hop BM25 did not already have among its seeds?
    seeds = set(harness.rank(ix, porter_scores, hotpot_typed.TOP_SEEDS))
    second_hop = [title for title in data["meta"][qid]["support"] if title not in seeds]
    stats["second_hop_n"] = len(second_hop)
    stats["second_hop_untyped"] = sum(1 for title in second_hop if untyped_scores.get(title, 0) > 0)
    stats["second_hop_typed"] = sum(1 for title in second_hop if typed["union"].get(title, 0) > 0)
    stats["second_hop_link"] = sum(1 for title in second_hop if typed["link"].get(title, 0) > 0)
    stats["second_hop_anchor"] = sum(1 for title in second_hop if typed["anchor"].get(title, 0) > 0)

    systems = {
        "porter": porter_scores,
        "ebr_untyped": fusion.rrf(ix, [porter_scores, untyped_dense, coverage]),
        "rrf_porter_bridge_untyped": fusion.rrf(ix, [porter_scores, untyped_dense]),
        "untyped_bridge_only": untyped_dense,
        "ebr_typed": fusion.rrf(ix, [porter_scores, typed_dense]),
        "ebr_typed_link": fusion.rrf(ix, [porter_scores, link_dense]),
        "ebr_typed_plus_coverage": fusion.rrf(ix, [porter_scores, typed_dense, coverage]),
        "ebr_typed_link_plus_coverage": fusion.rrf(ix, [porter_scores, link_dense, coverage]),
        "ebr_typed_anchor": fusion.rrf(ix, [porter_scores, anchor_dense]),
        "ebr_typed_anchor_plus_coverage": fusion.rrf(ix, [porter_scores, anchor_dense, coverage]),
        "typed_bridge_only": typed_dense,
    }
    return systems, stats


def ranked_from_scores(data, scores, pool=None):
    ix = data["ix"]
    if pool is None:
        return harness.rank(ix, scores, 1000)
    return sorted(pool, key=lambda doc: (-scores[data["position"][doc]], doc))


def evaluate(data, split, porter_system):
    global_rows = {label: [] for label in SYSTEMS}
    pool_rows = {label: [] for label in SYSTEMS}
    for qid, qrels in data["qrels"].items():
        if not hotpot.in_split(qid, split):
            continue
        systems, stats = query_systems(data, qid, porter_system)
        candidates = data["meta"][qid]["context"]
        for label, scores in systems.items():
            for target, pool in ((global_rows, None), (pool_rows, candidates)):
                ranked = ranked_from_scores(data, scores, pool)
                row = hotpot.metrics(ranked, qrels)
                row["qid"] = qid
                row["type"] = data["meta"][qid]["type"]
                row["relation"] = stats["types"][0] if stats["types"] else "unknown"
                row["detected"] = stats["detected"]
                row["bridges_untyped"] = stats["untyped_bridges"]
                row["bridges_typed"] = stats["typed_bridges"]
                row["bridges_link"] = stats["link_bridges"]
                row["second_hop_n"] = stats["second_hop_n"]
                row["second_hop_untyped"] = stats["second_hop_untyped"]
                row["second_hop_typed"] = stats["second_hop_typed"]
                row["second_hop_link"] = stats["second_hop_link"]
                row["second_hop_anchor"] = stats["second_hop_anchor"]
                target[label].append(row)
    return global_rows, pool_rows


def summarize(rows):
    out = {}
    for label, values in rows.items():
        if not values:
            continue
        entry = {key: sum(row[key] for row in values) / len(values) for key in METRIC_KEYS}
        for kind in ("bridge", "comparison"):
            subset = [row for row in values if row["type"] == kind]
            entry[kind] = {
                key: sum(row[key] for row in subset) / max(len(subset), 1) for key in METRIC_KEYS
            }
            entry[kind]["queries"] = len(subset)
        entry["queries"] = len(values)
        out[label] = entry
    return out


def slice_rows(rows, detected: bool):
    return {
        label: [row for row in values if row["detected"] is detected]
        for label, values in rows.items()
    }


def bridge_stats(rows):
    typed = rows["ebr_typed"]
    detected = [row for row in typed if row["detected"]]

    def mean(values):
        return sum(values) / max(len(values), 1)

    def reach(rows_subset, key):
        total = sum(row["second_hop_n"] for row in rows_subset)
        return sum(row[key] for row in rows_subset) / max(total, 1)

    return {
        "queries": len(typed),
        "relation_detected": len(detected),
        "relation_detected_frac": len(detected) / max(len(typed), 1),
        "mean_bridges_untyped": mean([row["bridges_untyped"] for row in typed]),
        "mean_bridges_typed_on_detected": mean([row["bridges_typed"] for row in detected]),
        "mean_bridges_link_on_detected": mean([row["bridges_link"] for row in detected]),
        "empty_typed_bridge_on_detected": mean(
            [1.0 if row["bridges_typed"] == 0 else 0.0 for row in detected]),
        "empty_link_bridge_on_detected": mean(
            [1.0 if row["bridges_link"] == 0 else 0.0 for row in detected]),
        "second_hop": {
            "gold_second_hop_docs": sum(row["second_hop_n"] for row in typed),
            "untyped_reach": reach(typed, "second_hop_untyped"),
            "typed_reach": reach(typed, "second_hop_typed"),
            "link_reach": reach(typed, "second_hop_link"),
            "anchor_reach": reach(typed, "second_hop_anchor"),
            "detected_untyped_reach": reach(detected, "second_hop_untyped"),
            "detected_typed_reach": reach(detected, "second_hop_typed"),
            "detected_link_reach": reach(detected, "second_hop_link"),
            "detected_anchor_reach": reach(detected, "second_hop_anchor"),
        },
    }


def print_table(title, summary):
    print(f"\n== {title}")
    header = (
        f"{'system':30} {'nDCG@10':>8} {'both@10':>8} {'both@5':>7} {'R@10':>7} "
        f"{'R@5':>7} {'both@2':>7} {'MRR@10':>7}"
    )
    print(header)
    for label, row in summary.items():
        print(
            f"{label:30} {row['ndcg@10']:8.3f} {row['both@10']:8.3f} {row['both@5']:7.3f} "
            f"{row['recall@10']:7.3f} {row['recall@5']:7.3f} {row['both@2']:7.3f} "
            f"{row['mrr@10']:7.3f}"
        )


def print_paired(title, report):
    print(f"\n  {title}")
    for metric, row in report.items():
        lo, hi = row["ci95"]
        w, t, l = row["wtl"]
        print(
            f"    {metric:10} Δ={row['delta']:+.4f}  "
            f"CI95=[{lo:+.4f},{hi:+.4f}]  W/T/L={w}/{t}/{l}"
        )


def paired_block(rows, base, candidate):
    return hotpot.paired_report(rows[base], rows[candidate], metrics=REPORT_METRICS)


def build_report(rows):
    return {
        "summary": summarize(rows),
        "bridge_stats": bridge_stats(rows),
        "paired": {
            f"{name}_vs_porter": paired_block(rows, "porter", name)
            for name in SYSTEMS if name != "porter"
        }
        | {
            "ebr_typed_vs_ebr_untyped": paired_block(rows, "ebr_untyped", "ebr_typed"),
            "ebr_typed_vs_rrf_porter_bridge_untyped": paired_block(
                rows, "rrf_porter_bridge_untyped", "ebr_typed"),
            "ebr_typed_link_vs_rrf_porter_bridge_untyped": paired_block(
                rows, "rrf_porter_bridge_untyped", "ebr_typed_link"),
            "ebr_typed_anchor_plus_coverage_vs_ebr_untyped": paired_block(
                rows, "ebr_untyped", "ebr_typed_anchor_plus_coverage"),
            "ebr_typed_anchor_vs_rrf_porter_bridge_untyped": paired_block(
                rows, "rrf_porter_bridge_untyped", "ebr_typed_anchor"),
        },
        "slices": {
            "relation_detected": {
                "summary": summarize(slice_rows(rows, True)),
                "paired": {
                    "ebr_untyped_vs_porter": paired_block(
                        slice_rows(rows, True), "porter", "ebr_untyped"),
                    "ebr_typed_vs_porter": paired_block(
                        slice_rows(rows, True), "porter", "ebr_typed"),
                    "ebr_typed_vs_rrf_porter_bridge_untyped": paired_block(
                        slice_rows(rows, True), "rrf_porter_bridge_untyped", "ebr_typed"),
                    "ebr_typed_link_vs_rrf_porter_bridge_untyped": paired_block(
                        slice_rows(rows, True), "rrf_porter_bridge_untyped", "ebr_typed_link"),
                    "ebr_typed_plus_coverage_vs_ebr_untyped": paired_block(
                        slice_rows(rows, True), "ebr_untyped", "ebr_typed_plus_coverage"),
                },
            },
            "relation_unknown": {
                "summary": summarize(slice_rows(rows, False)),
                "paired": {
                    "ebr_untyped_vs_porter": paired_block(
                        slice_rows(rows, False), "porter", "ebr_untyped"),
                },
            },
        },
        "per_relation": relation_slice(rows),
    }


def relation_slice(rows):
    out = {}
    names = sorted({row["relation"] for row in rows["ebr_typed"]})
    for name in names:
        selected = {
            label: [row for row in values if row["relation"] == name]
            for label, values in rows.items()
        }
        if not selected["ebr_typed"]:
            continue
        out[name] = {
            "queries": len(selected["ebr_typed"]),
            "summary": {
                label: {
                    key: sum(row[key] for row in selected[label]) / len(selected[label])
                    for key in REPORT_METRICS
                }
                for label in MAIN_SYSTEMS
            },
            "paired": {
                "ebr_typed_vs_rrf_porter_bridge_untyped": paired_block(
                    selected, "rrf_porter_bridge_untyped", "ebr_typed"),
                "ebr_typed_plus_coverage_vs_ebr_untyped": paired_block(
                    selected, "ebr_untyped", "ebr_typed_plus_coverage"),
            },
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(HERE / "data/hotpot_dev.parquet"))
    parser.add_argument("--out", default=str(HERE / "results/hotpot_typed.json"))
    parser.add_argument("--cache", default=str(HERE / "data/.cache_hotpot_typed.pkl"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--splits", default="dev,test")
    args = parser.parse_args()

    data = hotpot_typed.load(args.data, cache=None if args.no_cache else args.cache)
    print(
        f"[load] {data['ix'].N} paragraphs, {len(data['qrels'])} questions, "
        f"{data['relation_stats']['paragraphs_with_relations']} paragraphs with relations",
        flush=True,
    )
    porter_system = LeanUnitBM25(data["ix"], porter)

    report = {
        "meta": {
            "constants": {
                "rrf_k": fusion.RRF_K,
                "channel_depth": fusion.DEPTH,
                "seeds": hotpot_typed.TOP_SEEDS,
                "bridges": hotpot_typed.TOP_BRIDGES,
                "df_cap": hotpot_typed.MAX_DF_FRAC,
                "max_question_types": hotpot_typed.MAX_QUESTION_TYPES,
                "bootstrap_resamples": 1000,
            },
            "corpus": {
                "paragraphs": data["ix"].N,
                "questions": len(data["qrels"]),
                "paragraphs_with_relations": data["relation_stats"]["paragraphs_with_relations"],
            },
            "relation_lexicon": {
                "cues": len(hotpot_typed.CUE_TYPES),
                "types": sorted({t for types in hotpot_typed.CUE_TYPES.values() for t in types}
                                | {"nationality"}),
                "per_type_paragraphs": data["relation_stats"]["per_type_paragraphs"],
                "per_type_pairs": data["relation_stats"]["per_type_pairs"],
                "question_patterns": {
                    name: pattern.pattern for name, pattern in hotpot_typed.QUESTION_RES
                },
            },
        }
    }

    for split in args.splits.split(","):
        global_rows, pool_rows = evaluate(data, split, porter_system)
        for setting, rows in (("global", global_rows), ("pool", pool_rows)):
            key = f"{setting}:{split}"
            report[key] = build_report(rows)
            print_table(f"HotpotQA {split} ({setting})", report[key]["summary"])
            stats = report[key]["bridge_stats"]
            print(
                f"\n  relation detected {stats['relation_detected']}/{stats['queries']} "
                f"({stats['relation_detected_frac']:.3f}); "
                f"bridges untyped={stats['mean_bridges_untyped']:.1f} "
                f"typed={stats['mean_bridges_typed_on_detected']:.1f} "
                f"link={stats['mean_bridges_link_on_detected']:.1f}; "
                f"empty typed={stats['empty_typed_bridge_on_detected']:.3f}"
            )
            hop = stats["second_hop"]
            print(
                f"  second-hop reach (gold={hop['gold_second_hop_docs']}): "
                f"untyped={hop['untyped_reach']:.3f} typed={hop['typed_reach']:.3f} "
                f"link={hop['link_reach']:.3f} anchor={hop['anchor_reach']:.3f} | detected: "
                f"untyped={hop['detected_untyped_reach']:.3f} "
                f"typed={hop['detected_typed_reach']:.3f} "
                f"link={hop['detected_link_reach']:.3f} "
                f"anchor={hop['detected_anchor_reach']:.3f}"
            )
            for name, block in report[key]["paired"].items():
                print_paired(name, block)
            for name, block in report[key]["slices"].items():
                print_table(f"{split} {setting} slice: {name}", block["summary"])
                for label, ci in block["paired"].items():
                    print_paired(f"{name}: {label}", ci)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
