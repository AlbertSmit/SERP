#!/usr/bin/env python3
"""CESR-code: the CESR packing objective with exact code-graph constraints.

Same corpus, queries and candidate channels as run_code_graph.py. The difference is
the selection objective: instead of truncating a fused ranked list, choose a *set* of
chunks that covers the query's exact structural constraints (CALLS, DEFINES, IMPORTS,
EXPORTS, plus the path joint that ties them together) under a matched token budget.

Baselines at the same budget: Porter BM25, entity channel, graph channel and
RRF(Porter, entity, graph), all truncated in rank order with skip-and-continue.

The qrels are graph-derived. This is a capability test: does a coverage objective
execute the relation better than rank truncation on the same evidence? Not a
human-relevance result, and the constraints are oracle -- taken from query metadata,
never from qrels, but not producible from raw question text yet.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE / "lab"))

import numpy as np

import cesr_code as cc
import code_graph as cg

TYPE_ORDER = cc.TYPE_ORDER
BUDGETS = cc.BUDGETS
VARIANTS = ("pool", "all_chunks")
PAIR_METRICS = ("all@5", "all@10", "recall", "coverage", "ndcg@10")


# ------------------------------------------------------------------- evaluation


def evaluate(graph, ix, corpus, queries):
    """One pass over the query set; every system, budget and pool variant."""
    position = {doc: i for i, doc in enumerate(ix.doc_ids)}
    text_counts = Counter((query["type"], query["text"]) for query in queries)
    rows = {
        "pool": {system: {budget: [] for budget in BUDGETS} for system in cc.POOL_SYSTEMS},
        "all_chunks": {cc.CESR: {budget: [] for budget in BUDGETS}},
    }
    latency = Counter()
    reach = Counter()
    coverage = {variant: Counter() for variant in VARIANTS}
    provenance = {budget: Counter() for budget in BUDGETS}
    missing_gold = {budget: Counter() for budget in BUDGETS}
    samples = []
    ceiling_rows = []
    ambiguity = []

    for order, query in enumerate(queries):
        relevant = query["qrels"]
        gold = set(relevant)

        started = time.perf_counter()
        pool, channel_sources, channels = cc.build_pool(ix, graph, position, query)
        latency["retrieval_ms"] += (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        constraints = cc.query_constraints(corpus, query)
        latency["constraint_ms"] += (time.perf_counter() - started) * 1000

        answer_set = cc.answer_candidates(corpus, query, constraints)
        ambiguity.append({
            "qid": query["qid"],
            "type": query["type"],
            "answer_candidates": len(answer_set),
            "unique": int(len(answer_set) == 1),
        })

        baselines = {
            "porter_trunc": cc.truncate_all(channels["porter"], corpus),
            "entity_trunc": cc.truncate_all(channels["entity"], corpus),
            "graph_trunc": cc.truncate_all(channels["graph"], corpus),
            "rrf_trunc": cc.truncate_all(channels["rrf"], corpus),
        }

        candidates = {"pool": pool, "all_chunks": corpus.ids}
        gold_tokens = sum(corpus.dl[doc] for doc in gold)
        reach["gold_tokens"] += gold_tokens
        for budget in BUDGETS:
            reach[f"gold_fits@{budget}"] += int(gold_tokens <= budget)
        ceilings = {}
        for variant, candidate_ids in candidates.items():
            candidate_set = set(candidate_ids)
            achievable = np.array([
                bool(candidate_set & chunks) for chunks in constraints.covered_by
            ], dtype=bool)
            reach[f"{variant}_queries"] += 1
            reach[f"{variant}_gold"] += len(gold)
            reach[f"{variant}_gold_in_pool"] += len(gold & candidate_set)
            reach[f"{variant}_all_gold"] += int(gold <= candidate_set)
            reach[f"{variant}_achievable_weight"] += float(constraints.weights[achievable].sum())
            reach[f"{variant}_fully_coverable"] += int(achievable.all())
            ceilings[variant] = {
                "pool_size": len(candidate_ids),
                "gold_in_pool": bool(gold <= candidate_set),
                "achievable_coverage": float(constraints.weights[achievable].sum()),
            }
        ceiling_rows.append({
            "qid": query["qid"],
            "type": query["type"],
            "pool_size": ceilings["pool"]["pool_size"],
            "pool_all_gold": ceilings["pool"]["gold_in_pool"],
            "pool_achievable_coverage": ceilings["pool"]["achievable_coverage"],
        })

        for budget in BUDGETS:
            for system, docs in baselines.items():
                tokens = sum(corpus.dl[doc] for doc in docs[budget])
                row = cc.selection_metrics(corpus, docs[budget], relevant, constraints, tokens)
                row["qid"] = query["qid"]
                row["type"] = query["type"]
                row["support"] = len(relevant)
                row["text_ambiguous"] = int(text_counts[(query["type"], query["text"])] > 1)
                rows["pool"][system][budget].append(row)

            for variant, candidate_ids in candidates.items():
                started = time.perf_counter()
                selected, steps, covered = cc.greedy_select(
                    corpus, candidate_ids, constraints, budget)
                latency[f"select_ms_{variant}"] += (time.perf_counter() - started) * 1000
                tokens = sum(corpus.dl[doc] for doc in selected)
                row = cc.selection_metrics(corpus, selected, relevant, constraints, tokens)
                row["qid"] = query["qid"]
                row["type"] = query["type"]
                row["support"] = len(relevant)
                row["text_ambiguous"] = int(text_counts[(query["type"], query["text"])] > 1)
                rows[variant][cc.CESR][budget].append(row)

                full = bool(covered.all())
                coverage[variant][f"full@{budget}"] += int(full)
                coverage[variant][f"full_and_all10@{budget}"] += int(full and row["all@10"] == 1.0)
                coverage[variant][f"queries@{budget}"] += 1

                if variant == "pool":
                    for step in steps:
                        provenance[budget]["+".join(channel_sources.get(step["doc"], ["?"]))] += 1
                    pool_set = set(pool)
                    selected_set = set(selected)
                    unique_text = text_counts[(query["type"], query["text"])] == 1
                    prefix = "unique_" if unique_text else "ambiguous_"
                    for doc in gold - selected_set:
                        missing_gold[budget]["missing"] += 1
                        missing_gold[budget][prefix + "missing"] += 1
                        if doc not in pool_set:
                            cause = "not_in_pool"
                        elif any(doc in chunks for chunks in constraints.covered_by):
                            cause = "alternative_satisfier"
                        else:
                            cause = "unconstrained"
                        missing_gold[budget][cause] += 1
                        missing_gold[budget][prefix + cause] += 1
                    if order % max(len(queries) // 20, 1) == 0 and len(samples) < 60:
                        selected_sets = [set() for _ in constraints.labels]
                        for doc in selected:
                            for column, chunks in enumerate(constraints.covered_by):
                                if doc in chunks:
                                    selected_sets[column].add(doc)
                        samples.append({
                            "qid": query["qid"],
                            "type": query["type"],
                            "question": query["text"],
                            "budget": budget,
                            "qrels": relevant,
                            "constraints": [
                                {"label": label, "weight": round(float(constraints.weights[i]), 4),
                                 "covered": bool(covered[i]),
                                 "candidates": len(constraints.covered_by[i]),
                                 "in_pool": bool(set(pool) & constraints.covered_by[i])}
                                for i, label in enumerate(constraints.labels)
                            ],
                            "joints": [
                                {"label": joint["label"],
                                 "satisfied": cc.joint_satisfied(
                                     corpus, joint, selected_sets)}
                                for joint in constraints.joints
                            ],
                            "selected": [
                                {"doc": step["doc"], "tokens": step["tokens"],
                                 "gain": round(step["gain"], 4), "bonus": step["bonus"],
                                 "score": round(step["score"], 4),
                                 "redundancy": round(step["redundancy"], 3),
                                 "covers": step["covers"], "joins": step["joins"],
                                 "channels": channel_sources.get(step["doc"], ["!"])}
                                for step in steps
                            ],
                        })

    report = {
        "rows": rows,
        "latency": latency,
        "reach": reach,
        "coverage": coverage,
        "provenance": provenance,
        "missing_gold": missing_gold,
        "samples": samples,
        "ceiling_rows": ceiling_rows,
        "ambiguity": ambiguity,
        "queries": len(queries),
    }
    return report


# -------------------------------------------------------------------- reporting


def summarize_all(rows):
    return {
        variant: {
            system: {budget: cc.summarize(values) for budget, values in per_budget.items()}
            for system, per_budget in per_system.items()
        }
        for variant, per_system in rows.items()
    }


def paired_all(rows):
    baseline = rows["pool"]["rrf_trunc"]
    out = {}
    for variant, per_system in rows.items():
        out[variant] = {}
        for budget in BUDGETS:
            out[variant][str(budget)] = {
                cc.CESR: {metric: cc.paired(baseline[budget], per_system[cc.CESR][budget],
                                            metric)
                          for metric in PAIR_METRICS},
            }
            if variant == "pool":
                out[variant][str(budget)]["rrf_trunc_vs_porter"] = {
                    metric: cc.paired(per_system["porter_trunc"][budget],
                                      per_system["rrf_trunc"][budget], metric)
                    for metric in ("all@10", "recall")
                }
    return out


def slices_all(rows):
    out = {}
    for variant, per_system in rows.items():
        out[variant] = {}
        for system in (cc.CESR, "rrf_trunc", "porter_trunc"):
            if system not in per_system:
                continue
            out[variant][system] = {
                budget: {
                    "by_type": cc.slice_summary(per_system[system][budget], "type"),
                    "by_support": cc.slice_summary(per_system[system][budget], "support"),
                    "by_text_ambiguity": cc.slice_summary(
                        per_system[system][budget], "text_ambiguous"),
                }
                for budget in BUDGETS
            }
    return out


def ceiling_report(reach):
    n = max(reach["pool_queries"], 1)
    out = {}
    for variant in VARIANTS:
        out[variant] = {
            "pool_size_mean": None,
            "gold_chunks_in_pool": reach[f"{variant}_gold_in_pool"] / max(reach[f"{variant}_gold"], 1),
            "queries_with_all_gold_in_pool": reach[f"{variant}_all_gold"] / n,
            "mean_achievable_constraint_coverage": reach[f"{variant}_achievable_weight"] / n,
            "queries_fully_coverable": reach[f"{variant}_fully_coverable"] / n,
            "queries": int(reach[f"{variant}_queries"]),
            "gold_chunks": int(reach[f"{variant}_gold"]),
        }
    return out


def ambiguity_report(ambiguity):
    """How often the constraints identify a single answer chunk at all."""
    out = {"all": {}, "by_type": {}}
    for kind in TYPE_ORDER + ("all",):
        rows = ambiguity if kind == "all" else [r for r in ambiguity if r["type"] == kind]
        if not rows:
            continue
        block = {
            "queries": len(rows),
            "mean_answer_candidates": float(np.mean([r["answer_candidates"] for r in rows])),
            "median_answer_candidates": float(np.median([r["answer_candidates"] for r in rows])),
            "max_answer_candidates": int(max(r["answer_candidates"] for r in rows)),
            "unique_answer_rate": float(np.mean([r["unique"] for r in rows])),
        }
        if kind == "all":
            out["all"] = block
        else:
            out["by_type"][kind] = block
    return out


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def print_budget_table(summary, variant):
    systems = cc.POOL_SYSTEMS if variant == "pool" else (cc.CESR,)
    print(f"\n{variant}: matched-budget comparison (250 queries)")
    header = (f"{'budget':>6} {'system':16} " +
              " ".join(f"{metric:>9}" for metric in cc.METRICS))
    print(header)
    for budget in BUDGETS:
        for system in systems:
            row = summary[variant][system][budget]
            print(f"{budget:>6} {system:16} " + " ".join(
                f"{row[metric]:>9.3f}" for metric in cc.METRICS))
        print()


def print_paired(paired, variant):
    print(f"\npaired bootstrap vs RRF computed on the same channels, 1000 resamples, seed 1729"
          f" ({variant} candidate set)")
    print(f"{'budget':>6} {'metric':10} {'delta':>9} {'95% CI':>22} {'W/T/L':>12}")
    for budget in BUDGETS:
        for metric in PAIR_METRICS:
            entry = paired[variant][str(budget)][cc.CESR][metric]
            low, high = entry["ci95"]
            print(f"{budget:>6} {metric:10} {entry['delta']:+9.4f} "
                  f"[{low:+.4f},{high:+.4f}] "
                  f"{entry['wtl'][0]:>5}/{entry['wtl'][1]}/{entry['wtl'][2]}")
        print()


def main():
    graph = cg.build_graph(ROOT)
    ix = cg.build_index(graph)
    corpus = cc.build_corpus(graph, ix)
    queries = cg.generate_queries(graph, per_type=50, seed=cg.SEED)

    stats = graph.stats()
    total_tokens = int(sum(corpus.dl.values()))
    largest = max(corpus.dl.values())

    print("=" * 108)
    print("CESR-CODE: exact-constraint evidence-set retrieval over the SERP code graph")
    print("=" * 108)
    print(f"  chunks {len(corpus.ids)}   files {len(graph.files)}   "
          f"symbols {len(graph.defined)}   corpus tokens {total_tokens}")
    print(f"  mean dl {np.mean(list(corpus.dl.values())):.1f}   median "
          f"{int(np.median(list(corpus.dl.values())))}   max {largest}")
    print(f"  budgets {BUDGETS} bind at "
          f"{[round(b / total_tokens * 100, 1) for b in BUDGETS]}% of the corpus")
    print(f"  queries {len(queries)}  {dict(Counter(q['type'] for q in queries))}  "
          f"qrel sizes {dict(sorted(Counter(len(q['qrels']) for q in queries).items()))}")

    gold_tokens = float(np.mean([sum(corpus.dl[doc] for doc in q["qrels"])
                                 for q in queries]))
    print(f"  mean gold evidence = {gold_tokens:.0f} tokens per query "
          f"(the evidence set, not the candidate list, is what the budget must hold)")

    started = time.perf_counter()
    report = evaluate(graph, ix, corpus, queries)
    elapsed = time.perf_counter() - started
    rows = report["rows"]

    summary = summarize_all(rows)
    paired = paired_all(rows)
    slices = slices_all(rows)
    ceiling = ceiling_report(report["reach"])

    print(f"\ncandidate-pool ceiling")
    print(f"{'variant':12} {'pool':>6} {'gold in pool':>13} {'all gold':>9} "
          f"{'coverable':>10} {'all constraints':>16}")
    ceiling["pool"]["pool_size_mean"] = float(
        np.mean([row["pool_size"] for row in report["ceiling_rows"]]))
    ceiling["all_chunks"]["pool_size_mean"] = float(len(corpus.ids))
    for variant in VARIANTS:
        block = ceiling[variant]
        print(f"{variant:12} {block['pool_size_mean']:>6.1f} "
              f"{block['gold_chunks_in_pool']:>13.3f} "
              f"{block['queries_with_all_gold_in_pool']:>9.3f} "
              f"{block['mean_achievable_constraint_coverage']:>10.3f} "
              f"{block['queries_fully_coverable']:>16.3f}")

    for variant in VARIANTS:
        print_budget_table(summary, variant)
        print_paired(paired, variant)

    # CESR against its own constraint ceiling
    print("\nconstraint coverage and what full coverage buys (pool variant)")
    cover = report["coverage"]["pool"]
    n = report["queries"]
    print(f"{'budget':>6} {'coverage':>9} {'ceiling':>8} {'all covered':>12} "
          f"{'all@10 | full':>14} {'all@10':>8} {'tokens':>8} {'n':>5}")
    for budget in BUDGETS:
        row = summary["pool"][cc.CESR][budget]
        full_rate = cover[f"full@{budget}"] / max(cover[f"queries@{budget}"], 1)
        conditioned = (cover[f"full_and_all10@{budget}"] /
                       max(cover[f"full@{budget}"], 1))
        print(f"{budget:>6} {row['coverage']:>9.3f} "
              f"{ceiling['pool']['mean_achievable_constraint_coverage']:>8.3f} "
              f"{full_rate:>12.3f} {conditioned:>14.3f} {row['all@10']:>8.3f} "
              f"{row['tokens']:>8.0f} {row['selected']:>5.1f}")

    print("\nall@10 by query type (budget 2000 unless noted)")
    print(f"{'type':16} {'n':>4} {'porter':>8} {'rrf':>8} {'cesr@500':>9} "
          f"{'cesr@1000':>10} {'cesr@2000':>10} {'pool ceil':>10} {'unique':>7}")
    ambiguity = ambiguity_report(report["ambiguity"])
    for kind in TYPE_ORDER:
        base = [row for row in report["ceiling_rows"] if row["type"] == kind]
        subset = {"pool": {}}
        for system in ("porter_trunc", "rrf_trunc"):
            subset["pool"][system] = {
                budget: cc.slice_summary(rows["pool"][system][budget], "type")[kind]
                for budget in BUDGETS}
        subset["pool"][cc.CESR] = {
            budget: cc.slice_summary(rows["pool"][cc.CESR][budget], "type")[kind]
            for budget in BUDGETS}
        ceil = float(np.mean([1.0 if row["pool_all_gold"] else 0.0 for row in base]))
        unique = ambiguity["by_type"][kind]["unique_answer_rate"]
        print(f"{kind:16} {len(base):>4} "
              f"{subset['pool']['porter_trunc'][BUDGETS[-1]]['all@10']:>8.3f} "
              f"{subset['pool']['rrf_trunc'][BUDGETS[-1]]['all@10']:>8.3f} "
              f"{subset['pool'][cc.CESR][500]['all@10']:>9.3f} "
              f"{subset['pool'][cc.CESR][1000]['all@10']:>10.3f} "
              f"{subset['pool'][cc.CESR][2000]['all@10']:>10.3f} {ceil:>10.3f} "
              f"{unique:>7.3f}")

    print("\nconstraint ambiguity (chunks satisfying the full answer conjunction)")
    print(f"{'type':16} {'n':>4} {'mean':>7} {'median':>7} {'max':>5} {'unique':>8}")
    for kind in TYPE_ORDER + ("all",):
        block = ambiguity["all"] if kind == "all" else ambiguity["by_type"][kind]
        print(f"{kind:16} {block['queries']:>4} {block['mean_answer_candidates']:>7.2f} "
              f"{block['median_answer_candidates']:>7.1f} "
              f"{block['max_answer_candidates']:>5} {block['unique_answer_rate']:>8.3f}")

    print("\nall@10 by support size (budget 2000)")
    print(f"{'support':>8} {'n':>4} {'porter':>8} {'rrf':>8} {'cesr':>8}")
    for support in (2, 3, 4):
        base = [q for q in queries if len(q["qrels"]) == support]
        cells = []
        for system in ("porter_trunc", "rrf_trunc", cc.CESR):
            values = [row["all@10"] for row in rows["pool"][system][2000]
                      if row["support"] == support]
            cells.append(float(np.mean(values)) if values else float("nan"))
        print(f"{support:>8} {len(base):>4} " + " ".join(f"{c:>8.3f}" for c in cells))

    text_map = {}
    for query in queries:
        text_map.setdefault((query["type"], query["text"]), []).append(tuple(query["qrels"]))
    collisions = {}
    for kind in TYPE_ORDER:
        groups = [values for key, values in text_map.items() if key[0] == kind]
        repeated = [values for values in groups if len(values) > 1]
        conflicting = [values for values in repeated if len(set(values)) > 1]
        collisions[kind] = {
            "queries": sum(len(values) for values in groups),
            "distinct_texts": len(groups),
            "repeated_text_groups": len(repeated),
            "conflicting_qrel_groups": len(conflicting),
            "queries_in_conflicting_groups": sum(len(values) for values in conflicting),
        }

    print("\nbenchmark ambiguity: identical question text, different qrels (post-hoc slice)")
    print(f"{'type':14} {'n':>4} {'texts':>6} {'repeated':>9} {'conflicting':>12} "
          f"{'queries affected':>17}")
    for kind, block in collisions.items():
        print(f"{kind:14} {block['queries']:>4} {block['distinct_texts']:>6} "
              f"{block['repeated_text_groups']:>9} {block['conflicting_qrel_groups']:>12} "
              f"{block['queries_in_conflicting_groups']:>17}")

    print("\nall@10 by text ambiguity (budget 2000; 0 = unique question text)")
    print(f"{'text_amb':>9} {'n':>4} {'porter':>8} {'rrf':>8} {'cesr':>8} {'cover':>7}")
    ambiguity_slice = {}
    for flag in (0, 1):
        cells = []
        for system in ("porter_trunc", "rrf_trunc", cc.CESR):
            values = [row["all@10"] for row in rows["pool"][system][2000]
                      if row["text_ambiguous"] == flag]
            cells.append(float(np.mean(values)) if values else float("nan"))
        covered = [row["coverage"] for row in rows["pool"][cc.CESR][2000]
                   if row["text_ambiguous"] == flag]
        n = sum(1 for row in rows["pool"][cc.CESR][2000] if row["text_ambiguous"] == flag)
        ambiguity_slice[str(flag)] = {
            "queries": n,
            "porter_all@10": cells[0], "rrf_all@10": cells[1], "cesr_all@10": cells[2],
            "cesr_coverage": float(np.mean(covered)) if covered else float("nan"),
        }
        print(f"{flag:>9} {n:>4} " + " ".join(f"{c:>8.3f}" for c in cells) +
              f" {np.mean(covered) if covered else float('nan'):>7.3f}")

    print("\nall@10 by type, unique-text queries only (budget 2000)")
    print(f"{'type':16} {'n':>4} {'rrf':>8} {'cesr':>8}")
    unique_by_type = {}
    for kind in TYPE_ORDER:
        cells = []
        for system in ("rrf_trunc", cc.CESR):
            values = [row["all@10"] for row in rows["pool"][system][2000]
                      if row["text_ambiguous"] == 0 and row["type"] == kind]
            cells.append(float(np.mean(values)) if values else float("nan"))
        n = sum(1 for row in rows["pool"][cc.CESR][2000]
                if row["text_ambiguous"] == 0 and row["type"] == kind)
        unique_by_type[kind] = {"queries": n, "rrf_all@10": cells[0],
                                "cesr_all@10": cells[1]}
        print(f"{kind:16} {n:>4} {cells[0]:>8.3f} {cells[1]:>8.3f}")

    print("\nwhy gold chunks are missing (pool variant)")
    print(f"{'budget':>6} {'missing':>8} {'not in pool':>12} {'alternative':>12} "
          f"{'unconstrained':>14} {'missing(u)':>11} {'alt(u)':>8}")
    missing_report = {}
    for budget in BUDGETS:
        block = report["missing_gold"][budget]
        total = max(block["missing"], 1)
        unique_total = max(block["unique_missing"], 1)
        missing_report[str(budget)] = {
            key: block[key] / total for key in
            ("missing", "not_in_pool", "alternative_satisfier", "unconstrained")
        }
        missing_report[str(budget)]["unique_text"] = {
            key: block["unique_" + key] / unique_total for key in
            ("missing", "not_in_pool", "alternative_satisfier", "unconstrained")
        }
        print(f"{budget:>6} {block['missing']:>8} {block['not_in_pool'] / total:>12.3f} "
              f"{block['alternative_satisfier'] / total:>12.3f} "
              f"{block['unconstrained'] / total:>14.3f} {block['unique_missing']:>11} "
              f"{block['unique_alternative_satisfier'] / unique_total:>8.3f}")

    print("\nprovenance of CESR selections by channel (pool variant, budget 500)")
    total = sum(report["provenance"][500].values()) or 1
    for label, count in sorted(report["provenance"][500].items(),
                               key=lambda kv: -kv[1]):
        print(f"  {label:32} {count / total:>6.3f}")
    outside = sum(count for label, count in report["provenance"][500].items()
                  if "porter" not in label.split("+"))
    print(f"  outside Porter top-100{'':15} {outside / total:>6.3f}")

    latency = report["latency"]
    n = report["queries"]
    latency_report = {
        "retrieval_ms_per_query": latency["retrieval_ms"] / n,
        "constraint_ms_per_query": latency["constraint_ms"] / n,
        "cesr_select_ms_per_query": latency["select_ms_pool"] / n / len(BUDGETS),
        "cesr_all_chunks_select_ms_per_query": latency["select_ms_all_chunks"] / n / len(BUDGETS),
        "full_evaluation_s": elapsed,
    }
    print("\nlatency (CPU)")
    for key, value in latency_report.items():
        print(f"  {key:44} {value:>8.2f}")

    token_ratio = {
        budget: (summary["pool"][cc.CESR][budget]["tokens"] /
                 summary["pool"]["rrf_trunc"][budget]["tokens"])
        for budget in BUDGETS
    }
    gate = cc.gate(paired["pool"], token_ratio)
    print("\nGATE (pool variant, vs RRF at matched budget)")
    print(f"  rule: {gate['rule']}")
    for budget in BUDGETS:
        entry = gate["all_at_10_by_budget"][str(budget)]
        low, high = entry["ci95"]
        print(f"  budget {budget:>5}: all@10 delta {entry['delta']:+.4f} "
              f"[{low:+.4f},{high:+.4f}]  W/T/L "
              f"{entry['wtl'][0]}/{entry['wtl'][1]}/{entry['wtl'][2]}  "
              f"token ratio {gate['token_ratio_cesr_over_rrf'][str(budget)]:.2f}")
    print(f"  passed: {gate['passed']}  -> {gate['verdict']}")

    payload = {
        "meta": {
            "repo": str(ROOT),
            "scope": "src/, scripts/, test/ *.js",
            "seed": cg.SEED,
            "budgets": list(BUDGETS),
            "pool_depth": cc.POOL_DEPTH,
            "constants": {
                "mu": cc.MU,
                "cost_exponent": cc.COST_EXPONENT,
                "path_bonus": cc.PATH_BONUS,
                "constraint_weight": 1.0,
                "bootstrap_resamples": cc.RESAMPLES,
                "bootstrap_seed": cc.BOOTSTRAP_SEED,
            },
            "variant": "oracle-constraint CESR-code",
            "claim": (
                "Synthetic capability test. qrels are graph-derived, so this measures "
                "whether a coverage selector can execute the structural relation better "
                "than rank truncation, not human relevance. Constraints come from query "
                "metadata (type, symbols, module, chain), never from qrels; parsing them "
                "from raw question text is not implemented in this version."
            ),
            "systems": list(cc.POOL_SYSTEMS),
            "metrics": list(cc.METRICS),
        },
        "graph": stats,
        "corpus": {
            "chunks": len(corpus.ids),
            "corpus_tokens": total_tokens,
            "mean_dl": float(np.mean(list(corpus.dl.values()))),
            "max_dl": largest,
            "mean_gold_tokens": gold_tokens,
            "gold_evidence_fits_budget_rate": {
                str(b): report["reach"][f"gold_fits@{b}"] / report["queries"]
                for b in BUDGETS
            },
            "budget_pct_of_corpus": {
                str(b): b / total_tokens for b in BUDGETS
            },
            "budgets_bind": bool(max(BUDGETS) < total_tokens),
            "budget_binds_evidence": bool(gold_tokens > min(BUDGETS)),
        },
        "queries": {
            "total": len(queries),
            "by_type": dict(Counter(q["type"] for q in queries)),
            "qrel_sizes": {str(k): v for k, v in
                           sorted(Counter(len(q["qrels"]) for q in queries).items())},
            "text_collisions": collisions,
        },
        "ceiling": ceiling,
        "constraint_ambiguity": ambiguity,
        "systems": summary,
        "paired_vs_rrf": paired,
        "slices": slices,
        "text_ambiguity_slice": ambiguity_slice,
        "unique_text_by_type": unique_by_type,
        "coverage_vs_ceiling": {
            variant: {
                "constraint_coverage": {
                    str(budget): summary[variant][cc.CESR][budget]["coverage"]
                    for budget in BUDGETS
                },
                "full_coverage_rate": {
                    str(budget): report["coverage"][variant][f"full@{budget}"] /
                    max(report["coverage"][variant][f"queries@{budget}"], 1)
                    for budget in BUDGETS
                },
                "all_at_10_given_full_coverage": {
                    str(budget): report["coverage"][variant][f"full_and_all10@{budget}"] /
                    max(report["coverage"][variant][f"full@{budget}"], 1)
                    for budget in BUDGETS
                },
                "all_at_10_overall": {
                    str(budget): summary[variant][cc.CESR][budget]["all@10"]
                    for budget in BUDGETS
                },
            }
            for variant in VARIANTS
        },
        "provenance": {
            str(budget): {
                label: count for label, count in sorted(report["provenance"][budget].items())
            }
            for budget in BUDGETS
        },
        "missing_gold_causes": missing_report,
        "latency": latency_report,
        "gate": gate,
        "samples": report["samples"],
    }
    out = HERE / "results" / "cesr_code.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
