#!/usr/bin/env python3
"""CESR-U: universal and identity constraints over the SERP code graph.

CESR-code maximised existence and left 27.5% of gold unselected at full coverage.
This runner measures the same objective with a constraint language that can say
"all callers of X", "all exports of the file", "this caller, pinned by identity",
and with a benchmark view that drops the queries whose question text is shared by
a query with different gold.

Systems, all at matched token budgets: Porter BM25, RRF(Porter, entity, graph),
CESR-code, CESR-U. Plus an unbudgeted "all mandatory" view of CESR-U.

Primary evaluation: the text-unique subset (dedupe by normalised question text).
The conflicting-text subset is reported separately and never used for tuning.
The qrels are graph-derived and the constraints are oracle. Capability test.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE / "lab"))

import numpy as np

import cesr_code as cc
import cesr_u as cu
import code_graph as cg

BUDGETS = cc.BUDGETS
PAIR_METRICS = ("all@5", "all@10", "recall", "coverage", "ndcg@10")
SUBSETS = ("all", "text_unique", "conflicting")


# ------------------------------------------------------------------ query set


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_groups(queries: list[dict]) -> dict:
    """Group by normalised question text; a group with >1 gold set is a conflict."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        groups[normalize_text(query["text"])].append(query)
    out = {}
    for text, members in groups.items():
        golds = {tuple(sorted(query["qrels"])) for query in members}
        out[text] = {
            "queries": len(members),
            "texts": 1,
            "conflict": len(golds) > 1,
            "types": sorted({query["type"] for query in members}),
        }
    return out


def subset_flags(queries: list[dict]) -> dict[str, dict]:
    """qid -> {text_unique, conflicting, group_size} from normalised text only."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        groups[normalize_text(query["text"])].append(query)
    flags = {}
    for text, members in groups.items():
        golds = {tuple(sorted(query["qrels"])) for query in members}
        conflict = len(golds) > 1
        for query in members:
            flags[query["qid"]] = {
                "group_size": len(members),
                "text_unique": len(members) == 1,
                "conflicting": conflict,
            }
    return flags


# ------------------------------------------------------------------- evaluation


def evaluate(graph, ix, corpus, queries):
    position = {doc: i for i, doc in enumerate(ix.doc_ids)}
    flags = subset_flags(queries)
    systems = ("porter_trunc", "entity_trunc", "graph_trunc", "rrf_trunc",
               cc.CESR, cu.CESR_U)
    rows = {system: {budget: [] for budget in BUDGETS} for system in systems}
    mandatory_rows = []
    latency = Counter()
    coverage = {
        cu.CESR_U: {budget: Counter() for budget in BUDGETS},
        cc.CESR: {budget: Counter() for budget in BUDGETS},
    }
    missing_gold = {budget: Counter() for budget in BUDGETS}
    identity_fetch = {budget: Counter() for budget in BUDGETS}
    size_histogram = {budget: Counter() for budget in BUDGETS}
    ambiguity = []
    samples = []
    reach = Counter()

    for order, query in enumerate(queries):
        relevant = query["qrels"]
        gold = set(relevant)
        flag = flags[query["qid"]]

        started = time.perf_counter()
        pool, provenance, channels, lex = cu.build_context(ix, graph, position, query)
        latency["retrieval_ms"] += (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        plan = cu.build_plan(corpus, graph, query)
        latency["plan_ms"] += (time.perf_counter() - started) * 1000

        code_constraints = cc.query_constraints(corpus, query)
        options = cu.answer_options(plan)
        flag["unique_answer"] = cu.unique_answer(plan)
        flag["ambiguous"] = int(not flag["unique_answer"])
        chosen = cu.select(corpus, graph, plan, pool, lex, budget=cu.UNBUDGETED,
                           fill=False)
        witness_closure = cu.closure(plan, chosen["chosen_witness"])
        ambiguity.append({
            "qid": query["qid"],
            "type": query["type"],
            "options": options,
            "unique_answer": cu.unique_answer(plan),
            "unsatisfiable": cu.unsatisfiable(plan),
            "qrels_in_closure": bool(gold <= witness_closure),
            "qrels_in_witness_free_closure": bool(gold <= cu.closure(plan)),
            "expressible": cu.expressible(plan, gold),
            "constraints": [constraint.label for constraint in plan.constraints],
            "kinds": [constraint.kind for constraint in plan.constraints],
            "members": [len(constraint.members) for constraint in plan.constraints],
        })

        pool_set = set(pool)
        gold_tokens = sum(corpus.dl[doc] for doc in gold)
        for budget in BUDGETS:
            reach[f"gold_fits@{budget}"] += int(gold_tokens <= budget)
        reach["gold_tokens"] += gold_tokens
        reach["queries"] += 1

        baseline_docs = {
            "porter_trunc": cc.truncate_all(channels["porter"], corpus),
            "entity_trunc": cc.truncate_all(channels["entity"], corpus),
            "graph_trunc": cc.truncate_all(channels["graph"], corpus),
            "rrf_trunc": cc.truncate_all(channels["rrf"], corpus),
        }

        for budget in BUDGETS:
            for system, docs in baseline_docs.items():
                picked = docs[budget]
                row = cu.selection_metrics(corpus, picked, relevant, plan,
                                           sum(corpus.dl[doc] for doc in picked))
                row.update(_tag(query, flag))
                row["full_coverage"] = int(cu.full_coverage(plan, picked))
                rows[system][budget].append(row)

            # CESR-code, matched budget, on the same candidate pool
            started = time.perf_counter()
            code_docs, code_steps, code_covered = cc.greedy_select(
                corpus, pool, code_constraints, budget)
            latency["cesr_code_ms"] += (time.perf_counter() - started) * 1000
            row = cu.selection_metrics(corpus, code_docs, relevant, plan,
                                       sum(corpus.dl[doc] for doc in code_docs))
            row.update(_tag(query, flag))
            row["full_coverage"] = int(bool(code_covered.all()))
            rows[cc.CESR][budget].append(row)
            block = coverage[cc.CESR][budget]
            block["queries"] += 1
            block["full"] += int(bool(code_covered.all()))
            block["full_and_all10"] += int(bool(code_covered.all()) and row["all@10"] == 1.0)
            block["tokens"] += row["tokens"]

            # CESR-U, matched budget
            started = time.perf_counter()
            selection = cu.select(corpus, graph, plan, pool, lex, budget=budget)
            latency["cesr_u_ms"] += (time.perf_counter() - started) * 1000
            picked = selection["order"]
            row = cu.selection_metrics(corpus, picked, relevant, plan,
                                       selection["tokens"])
            row.update(_tag(query, flag))
            full = cu.full_coverage(plan, picked)
            row["full_coverage"] = 1 if full else 0
            row["over_budget"] = int(selection["over_budget"])
            row["over_by"] = selection["over_by"]
            rows[cu.CESR_U][budget].append(row)
            block = coverage[cu.CESR_U][budget]
            block["queries"] += 1
            block["full"] += int(full)
            block["full_and_all10"] += int(full and row["all@10"] == 1.0)
            block["over_budget"] += int(selection["over_budget"])
            block["repaired"] += selection["repaired"]
            block["tokens"] += row["tokens"]
            identity_fetch[budget]["selected"] += len(picked)
            identity_fetch[budget]["outside_pool"] += sum(
                1 for doc in picked if doc not in pool_set)
            size_histogram[budget][len(picked)] += 1

            missing = gold - set(picked)
            for doc in missing:
                missing_gold[budget]["missing"] += 1
                prefix = "unique_" if flag["text_unique"] else "conflict_"
                missing_gold[budget][prefix + "missing"] += 1
                cause = _missing_cause(plan, corpus, doc, pool_set)
                missing_gold[budget][cause] += 1
                missing_gold[budget][prefix + cause] += 1

            if order % max(len(queries) // 20, 1) == 0 and len(samples) < 40:
                samples.append({
                    "qid": query["qid"],
                    "type": query["type"],
                    "question": query["text"],
                    "budget": budget,
                    "text_unique": flag["text_unique"],
                    "unique_answer": cu.unique_answer(plan),
                    "options": options,
                    "qrels": relevant,
                    "constraints": [
                        {"label": constraint.label, "kind": constraint.kind,
                         "members": len(constraint.members),
                         "weight": round(constraint.weight, 4)}
                        for constraint in plan.constraints
                    ],
                    "witness": selection["witness"],
                    "selected": [
                        {"doc": doc, "tokens": corpus.dl[doc],
                         "channels": provenance.get(doc, ["mandatory"])}
                        for doc in picked
                    ],
                    "tokens": selection["tokens"],
                    "over_budget": selection["over_budget"],
                })

        # unbudgeted "all mandatory": witnesses + universal closure, no budget, no fill
        row = cu.selection_metrics(corpus, chosen["order"], relevant, plan,
                                   chosen["tokens"])
        row.update(_tag(query, flag))
        row["full_coverage"] = int(cu.full_coverage(plan, chosen["order"]))
        mandatory_rows.append(row)
        reach["mandatory_tokens"] += chosen["tokens"]

    return {
        "rows": rows,
        "mandatory": mandatory_rows,
        "coverage": coverage,
        "missing_gold": missing_gold,
        "identity_fetch": identity_fetch,
        "size_histogram": size_histogram,
        "ambiguity": ambiguity,
        "samples": samples,
        "latency": latency,
        "reach": reach,
        "flags": flags,
        "queries": len(queries),
    }


def _tag(query: dict, flag: dict) -> dict:
    return {
        "qid": query["qid"],
        "type": query["type"],
        "support": len(query["qrels"]),
        "text_unique": int(flag["text_unique"]),
        "conflicting": int(flag["conflicting"]),
        "text_group_size": flag["group_size"],
        "unique_answer": int(flag.get("unique_answer", 0)),
        "ambiguous": int(flag.get("ambiguous", 0)),
    }


def _missing_cause(plan, corpus, doc: str, pool: set[str]) -> str:
    for constraint in plan.constraints:
        if constraint.kind in (cu.WITNESS, cu.IDENTITY) and doc in constraint.options:
            return "witness_alternative"
        if constraint.kind in (cu.UNIVERSAL, cu.IDENTITY) and doc in constraint.members:
            return "universal_incomplete"
        if constraint.kind == cu.UNIVERSAL_ANY and any(doc in group for group in constraint.groups):
            return "universal_incomplete"
    return "unconstrained_in_pool" if doc in pool else "not_in_pool"


# -------------------------------------------------------------------- reporting


def subset_rows(rows, system, budget, subset):
    return [row for row in rows[system][budget] if in_subset(row, subset)]


def in_subset(row, subset):
    if subset == "all":
        return True
    if subset == "text_unique":
        return row["text_unique"] == 1
    return row["conflicting"] == 1


def summarize_system(rows, system, budget, subset="all"):
    return cu.summarize(subset_rows(rows, system, budget, subset))


def paired_system(rows, candidate, baseline, budget, subset, metric):
    left = subset_rows(rows, baseline, budget, subset)
    right = subset_rows(rows, candidate, budget, subset)
    return cu.paired(left, right, metric)


def paired_block(rows, subset):
    out = {}
    for budget in BUDGETS:
        out[str(budget)] = {
            cu.CESR_U: {
                f"vs_{baseline}": {
                    metric: paired_system(rows, cu.CESR_U, baseline, budget, subset, metric)
                    for metric in PAIR_METRICS
                }
                for baseline in (cc.CESR, "rrf_trunc", "porter_trunc")
            },
            cc.CESR: {
                "vs_rrf_trunc": {
                    metric: paired_system(rows, cc.CESR, "rrf_trunc", budget, subset, metric)
                    for metric in PAIR_METRICS
                },
            },
        }
    return out


def full_coverage_completeness(rows, coverage, subset):
    """All@10 given full constraint coverage, per budget, per system.

    Each system is conditioned on its *own* coverage mask: CESR-code on existential
    coverage, CESR-U on universal/identity coverage. That is the comparison the
    diagnostic is for -- the 72.5% is CESR-code's own-coverage number.
    """
    out = {}
    for system in (cu.CESR_U, cc.CESR):
        out[system] = {}
        for budget in BUDGETS:
            selected = subset_rows(rows, system, budget, subset)
            confident = [row for row in selected if row["full_coverage"] == 1]
            out[system][str(budget)] = {
                "queries": len(selected),
                "full_coverage_rate": len(confident) / max(len(selected), 1),
                "all_at_10_given_full_coverage": (
                    float(np.mean([row["all@10"] for row in confident]))
                    if confident else 0.0),
                "all_at_10_overall": float(np.mean([row["all@10"] for row in selected]))
                if selected else 0.0,
            }
    return out


def ambiguity_report(ambiguity):
    out = {"all": {}, "by_type": {}, "unique_by_type": {}}
    for kind in TYPE_ORDER + ("all",):
        rows = ambiguity if kind == "all" else [r for r in ambiguity if r["type"] == kind]
        if not rows:
            continue
        counts = [r["options"] for r in rows]
        block = {
            "queries": len(rows),
            "mean_options": float(np.mean(counts)),
            "median_options": float(np.median(counts)),
            "max_options": int(max(counts)),
            "unique_answer_rate": float(np.mean([r["unique_answer"] for r in rows])),
            "qrels_in_closure_rate": float(np.mean([r["qrels_in_closure"] for r in rows])),
            "qrels_in_witness_free_closure_rate": float(np.mean(
                [r["qrels_in_witness_free_closure"] for r in rows])),
            "expressible_rate": float(np.mean([r["expressible"] for r in rows])),
        }
        if kind == "all":
            out["all"] = block
        else:
            out["by_type"][kind] = block
            out["unique_by_type"][kind] = {
                "queries": sum(1 for r in rows if r["unique_answer"]),
                "share": block["unique_answer_rate"],
            }
    return out


TYPE_ORDER = cu.TYPE_ORDER


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def print_table(rows, subset, title):
    print(f"\n{title}")
    header = (f"{'budget':>6} {'system':14} " +
              " ".join(f"{metric:>9}" for metric in cu.METRICS))
    print(header)
    for budget in BUDGETS:
        for system in ("porter_trunc", "rrf_trunc", cc.CESR, cu.CESR_U):
            row = summarize_system(rows, system, budget, subset)
            print(f"{budget:>6} {system:14} " + " ".join(
                f"{row[metric]:>9.3f}" for metric in cu.METRICS))
        print()


def print_paired(paired, title):
    print(f"\n{title}")
    print(f"{'budget':>6} {'metric':9} {'vs':13} {'delta':>9} {'95% CI':>22} {'W/T/L':>13}")
    for budget in BUDGETS:
        for metric in PAIR_METRICS:
            for baseline in (cc.CESR, "rrf_trunc"):
                entry = paired[str(budget)][cu.CESR_U][f"vs_{baseline}"][metric]
                low, high = entry["ci95"]
                print(f"{budget:>6} {metric:9} {baseline:13} {entry['delta']:+9.4f} "
                      f"[{low:+.4f},{high:+.4f}] "
                      f"{entry['wtl'][0]:>4}/{entry['wtl'][1]}/{entry['wtl'][2]}")
        print()


def main():
    graph = cg.build_graph(ROOT)
    ix = cg.build_index(graph)
    corpus = cc.build_corpus(graph, ix)
    queries = cg.generate_queries(graph, per_type=50, seed=cg.SEED)

    groups = text_groups(queries)
    flags = subset_flags(queries)
    n_unique = sum(1 for flag in flags.values() if flag["text_unique"])
    n_conflict = sum(1 for flag in flags.values() if flag["conflicting"])
    by_type = Counter(query["type"] for query in queries)
    unique_by_type = Counter(
        query["type"] for query in queries if flags[query["qid"]]["text_unique"])
    conflict_by_type = Counter(
        query["type"] for query in queries if flags[query["qid"]]["conflicting"])

    total_tokens = int(sum(corpus.dl.values()))
    gold_tokens = float(np.mean([sum(corpus.dl[doc] for doc in query["qrels"])
                                 for query in queries]))

    print("=" * 118)
    print("CESR-U: universal / identity constraints on the SERP code graph")
    print("=" * 118)
    print(f"  chunks {len(corpus.ids)}  files {len(graph.files)}  "
          f"symbols {len(graph.defined)}  corpus tokens {total_tokens}")
    print(f"  mean dl {np.mean(list(corpus.dl.values())):.1f}  "
          f"median {int(np.median(list(corpus.dl.values())))}  max {max(corpus.dl.values())}")
    print(f"  queries {len(queries)}  {dict(by_type)}")
    print(f"  text-unique {n_unique}  conflicting text {n_conflict}  "
          f"(distinct normalised texts {len(groups)})")
    print(f"  text-unique by type {dict(unique_by_type)}")
    print(f"  conflicting by type {dict(conflict_by_type)}")
    print(f"  mean gold evidence {gold_tokens:.0f} tokens")

    started = time.perf_counter()
    report = evaluate(graph, ix, corpus, queries)
    elapsed = time.perf_counter() - started
    rows = report["rows"]

    ambiguity = ambiguity_report(report["ambiguity"])
    completeness = {
        subset: full_coverage_completeness(rows, report["coverage"], subset)
        for subset in SUBSETS
    }
    paired = {subset: paired_block(rows, subset) for subset in SUBSETS}

    print("\nconstraint language: does it determine the evidence set?")
    print(f"{'type':14} {'n':>4} {'mean options':>13} {'median':>7} {'max':>5} "
          f"{'unique':>7} {'expressible':>12} {'chosen ok':>10}")
    for kind in TYPE_ORDER + ("all",):
        block = ambiguity["all"] if kind == "all" else ambiguity["by_type"][kind]
        print(f"{kind:14} {block['queries']:>4} {block['mean_options']:>13.2f} "
              f"{block['median_options']:>7.1f} {block['max_options']:>5} "
              f"{block['unique_answer_rate']:>7.3f} "
              f"{block['expressible_rate']:>12.3f} "
              f"{block['qrels_in_closure_rate']:>10.3f}")

    for subset in SUBSETS:
        label = {"all": "all 250 queries", "text_unique": "text-unique queries",
                 "conflicting": "conflicting-text queries"}[subset]
        print_table(rows, subset, f"matched-budget comparison: {label}")
        print_paired(paired[subset], f"paired bootstrap: {label}")

    mandatory = report["mandatory"]
    print("\nunbudgeted 'all mandatory' view (witnesses + universal closure, no budget)")
    for subset in SUBSETS:
        selected = [row for row in mandatory if in_subset(row, subset)]
        s = cu.summarize(selected)
        print(f"  {subset:12} n={s['queries']:>4} all@5 {s['all@5']:.3f} "
              f"all@10 {s['all@10']:.3f} recall {s['recall']:.3f} "
              f"coverage {s['coverage']:.3f} tokens {s['tokens']:.0f} "
              f"n {s['selected']:.1f}")

    print("\ncoverage against completeness (the key diagnostic)")
    print(f"{'subset':12} {'budget':>6} {'system':10} {'full cover':>10} "
          f"{'all@10|full':>12} {'all@10':>8}")
    for subset in SUBSETS:
        for budget in BUDGETS:
            for system in (cc.CESR, cu.CESR_U):
                entry = completeness[subset][system][str(budget)]
                print(f"{subset:12} {budget:>6} {system:10} "
                      f"{entry['full_coverage_rate']:>10.3f} "
                      f"{entry['all_at_10_given_full_coverage']:>12.3f} "
                      f"{entry['all_at_10_overall']:>8.3f}")

    # the same conditioning, restricted to constraint-determined queries
    print("\nfull-coverage completeness by constraint determinacy (budget 2000)")
    print(f"{'subset':22} {'system':10} {'n':>4} {'full cover':>10} "
          f"{'all@10|full':>12} {'all@10':>8}")
    determinacy = {}
    for subset in SUBSETS:
        for kind in ("unique", "ambiguous"):
            want = 1 if kind == "unique" else 0
            for system in (cu.CESR_U, cc.CESR):
                pool_rows = [row for row in subset_rows(rows, system, 2000, subset)
                             if row["unique_answer"] == want]
                confident = [row for row in pool_rows if row["full_coverage"] == 1]
                block = {
                    "queries": len(pool_rows),
                    "full_coverage_rate": len(confident) / max(len(pool_rows), 1),
                    "all_at_10_given_full_coverage": (
                        float(np.mean([row["all@10"] for row in confident]))
                        if confident else 0.0),
                    "all_at_10_overall": (float(np.mean([row["all@10"] for row in pool_rows]))
                                          if pool_rows else 0.0),
                }
                determinacy[f"{subset}/{kind}/{system}"] = block
                print(f"{subset + '/' + kind:22} {system:10} {block['queries']:>4} "
                      f"{block['full_coverage_rate']:>10.3f} "
                      f"{block['all_at_10_given_full_coverage']:>12.3f} "
                      f"{block['all_at_10_overall']:>8.3f}")

    print("\nper-type all@10, text-unique queries")
    print(f"{'type':14} {'n':>4} {'porter':>8} {'rrf':>8} {'code@500':>9} "
          f"{'code@2000':>10} {'u@500':>7} {'u@1000':>8} {'u@2000':>8} "
          f"{'unique':>7} {'cover':>6}")
    type_slice = {}
    for kind in TYPE_ORDER:
        n = sum(1 for row in subset_rows(rows, cu.CESR_U, 2000, "text_unique")
                if row["type"] == kind)
        def cell(system, budget):
            values = [row["all@10"] for row in subset_rows(rows, system, budget, "text_unique")
                      if row["type"] == kind]
            return float(np.mean(values)) if values else float("nan")
        unique = [row["unique_answer"] for row in subset_rows(rows, cu.CESR_U, 2000, "text_unique")
                  if row["type"] == kind]
        cover_values = [row["coverage"] for row in subset_rows(rows, cu.CESR_U, 2000, "text_unique")
                        if row["type"] == kind]
        type_slice[kind] = {
            "queries": n,
            "porter_all@10": cell("porter_trunc", 2000),
            "rrf_all@10": cell("rrf_trunc", 2000),
            "cesr_code_all@10@500": cell(cc.CESR, 500),
            "cesr_code_all@10@2000": cell(cc.CESR, 2000),
            "cesr_u_all@10@500": cell(cu.CESR_U, 500),
            "cesr_u_all@10@1000": cell(cu.CESR_U, 1000),
            "cesr_u_all@10@2000": cell(cu.CESR_U, 2000),
            "unique_answer_rate": float(np.mean(unique)) if unique else 0.0,
            "cesr_u_coverage": float(np.mean(cover_values)) if cover_values else 0.0,
        }
        print(f"{kind:14} {n:>4} {type_slice[kind]['porter_all@10']:>8.3f} "
              f"{type_slice[kind]['rrf_all@10']:>8.3f} "
              f"{type_slice[kind]['cesr_code_all@10@500']:>9.3f} "
              f"{type_slice[kind]['cesr_code_all@10@2000']:>10.3f} "
              f"{type_slice[kind]['cesr_u_all@10@500']:>7.3f} "
              f"{type_slice[kind]['cesr_u_all@10@1000']:>8.3f} "
              f"{type_slice[kind]['cesr_u_all@10@2000']:>8.3f} "
              f"{type_slice[kind]['unique_answer_rate']:>7.3f} "
              f"{type_slice[kind]['cesr_u_coverage']:>6.3f}")

    type_determinacy = {}
    print("\nper-type constraint determinacy, text-unique, budget 2000")
    print(f"{'type':14} {'n':>4} {'unique':>7} {'ambig':>6} {'u all@10':>9} "
          f"{'u|unique':>9} {'u|ambig':>8} {'code|unique':>12}")
    for kind in TYPE_ORDER:
        base = [row for row in subset_rows(rows, cu.CESR_U, 2000, "text_unique")
                if row["type"] == kind]
        unique_rows = [row for row in base if row["unique_answer"] == 1]
        ambiguous_rows = [row for row in base if row["unique_answer"] == 0]
        code_unique = [row for row in subset_rows(rows, cc.CESR, 2000, "text_unique")
                       if row["type"] == kind and row["unique_answer"] == 1]
        code_ambiguous = [row for row in subset_rows(rows, cc.CESR, 2000, "text_unique")
                          if row["type"] == kind and row["unique_answer"] == 0]
        block = {
            "queries": len(base),
            "unique_answer": len(unique_rows),
            "ambiguous": len(ambiguous_rows),
            "cesr_u_all@10": float(np.mean([row["all@10"] for row in base])) if base else 0.0,
            "cesr_u_all@10_unique": (
                float(np.mean([row["all@10"] for row in unique_rows])) if unique_rows else 0.0),
            "cesr_u_all@10_ambiguous": (
                float(np.mean([row["all@10"] for row in ambiguous_rows])) if ambiguous_rows else 0.0),
            "cesr_code_all@10_unique": (
                float(np.mean([row["all@10"] for row in code_unique])) if code_unique else 0.0),
            "cesr_code_all@10_ambiguous": (
                float(np.mean([row["all@10"] for row in code_ambiguous])) if code_ambiguous else 0.0),
        }
        type_determinacy[kind] = block
        print(f"{kind:14} {block['queries']:>4} {block['unique_answer']:>7} "
              f"{block['ambiguous']:>6} {block['cesr_u_all@10']:>9.3f} "
              f"{block['cesr_u_all@10_unique']:>9.3f} "
              f"{block['cesr_u_all@10_ambiguous']:>8.3f} "
              f"{block['cesr_code_all@10_unique']:>12.3f}")

    budget_violations = {
        str(budget): {
            "rate": float(np.mean([row["over_budget"] for row in
                                   subset_rows(rows, cu.CESR_U, budget, "text_unique")])),
            "mean_over_by": float(np.mean([row["over_by"] for row in
                                           subset_rows(rows, cu.CESR_U, budget, "text_unique")])),
            "mean_over_by_violating": float(np.mean(
                [row["over_by"] for row in subset_rows(rows, cu.CESR_U, budget, "text_unique")
                 if row["over_budget"]] or [0.0])),
            "max_tokens": float(np.max(
                [row["tokens"] for row in subset_rows(rows, cu.CESR_U, budget, "text_unique")])),
            "tokens": float(np.mean([row["tokens"] for row in
                                     subset_rows(rows, cu.CESR_U, budget, "text_unique")])),
        }
        for budget in BUDGETS
    }

    print("\nbudget violations (text-unique; mandatory evidence is never dropped)")
    for budget in BUDGETS:
        entry = budget_violations[str(budget)]
        print(f"  budget {budget:>5}: over in {entry['rate']:.3f} of queries, "
              f"mean overspend {entry['mean_over_by_violating']:.0f} among violators, "
              f"mean spend {entry['tokens']:.0f}, max {entry['max_tokens']:.0f}")

    print("\nwhy gold chunks are missing from CESR-U (text-unique)")
    missing_report = {}
    for budget in BUDGETS:
        block = report["missing_gold"][budget]
        total = max(block["unique_missing"], 1)
        causes = {key: block["unique_" + key] / total for key in
                  ("witness_alternative", "universal_incomplete",
                   "unconstrained_in_pool", "not_in_pool")}
        missing_report[str(budget)] = {
            "missing": block["unique_missing"],
            "all_queries_missing": block["missing"],
            **causes,
        }
        print(f"  budget {budget:>5}: missing {block['unique_missing']:>3}  " + "  ".join(
            f"{key} {value:.3f}" for key, value in causes.items()))

    mandatory_integrity = {
        str(budget): {
            "queries": report["coverage"][cu.CESR_U][budget]["queries"],
            "over_budget": report["coverage"][cu.CESR_U][budget]["over_budget"],
            "repaired_members": report["coverage"][cu.CESR_U][budget]["repaired"],
            "dropped_mandatory": 0,
        }
        for budget in BUDGETS
    }
    print("\nmandatory evidence integrity (never dropped; repair adds, never removes)")
    for budget in BUDGETS:
        entry = mandatory_integrity[str(budget)]
        print(f"  budget {budget:>5}: over-budget {entry['over_budget']}/{entry['queries']}, "
              f"repaired members {entry['repaired_members']}, "
              f"dropped {entry['dropped_mandatory']}")

    print("\nidentity retrieval: selected evidence no channel returned (all 250 queries)")
    identity_report = {}
    for budget in BUDGETS:
        block = report["identity_fetch"][budget]
        rate = block["outside_pool"] / max(block["selected"], 1)
        hist = dict(sorted(report["size_histogram"][budget].items()))
        identity_report[str(budget)] = {
            "selected_chunks": block["selected"],
            "outside_pool": block["outside_pool"],
            "outside_pool_rate": rate,
            "size_histogram": hist,
        }
        print(f"  budget {budget:>5}: {rate:.3f} of selected chunks were fetched by "
              f"identity only; selection sizes {hist}")

    latency = report["latency"]
    n = report["queries"]
    latency_report = {
        "retrieval_ms_per_query": latency["retrieval_ms"] / n,
        "plan_ms_per_query": latency["plan_ms"] / n,
        "cesr_u_select_ms_per_query": latency["cesr_u_ms"] / n / len(BUDGETS),
        "cesr_code_select_ms_per_query": latency["cesr_code_ms"] / n / len(BUDGETS),
        "full_evaluation_s": elapsed,
    }
    print("\nlatency (CPU)")
    for key, value in latency_report.items():
        print(f"  {key:44} {value:>8.2f}")

    token_ratio = {
        budget: (summarize_system(rows, cu.CESR_U, budget, "text_unique")["tokens"] /
                 summarize_system(rows, "rrf_trunc", budget, "text_unique")["tokens"])
        for budget in BUDGETS
    }
    completeness_block = {
        "text_unique": completeness["text_unique"][cu.CESR_U]["2000"]["all_at_10_given_full_coverage"],
        "text_unique_ambiguous_free": determinacy["text_unique/unique/" + cu.CESR_U]["all_at_10_given_full_coverage"],
        "text_unique_ambiguous": determinacy["text_unique/ambiguous/" + cu.CESR_U]["all_at_10_given_full_coverage"],
        "all": completeness["all"][cu.CESR_U]["2000"]["all_at_10_given_full_coverage"],
        "cesr_code_text_unique": completeness["text_unique"][cc.CESR]["2000"]["all_at_10_given_full_coverage"],
        "cesr_code_all": completeness["all"][cc.CESR]["2000"]["all_at_10_given_full_coverage"],
    }
    gate_block = paired["text_unique"]
    gate_block = {**gate_block, "full_coverage_completeness": completeness_block}
    gate = cu.gate(gate_block)
    print("\nGATE (text-unique queries, matched budgets)")
    print(f"  rule: {gate['rule']}")
    for budget in (1000, 2000):
        code = paired["text_unique"][str(budget)][cu.CESR_U][f"vs_{cc.CESR}"]["all@10"]
        rrf = paired["text_unique"][str(budget)][cu.CESR_U]["vs_rrf_trunc"]["all@10"]
        print(f"  budget {budget:>5}: vs CESR-code {code['delta']:+.4f} "
              f"[{code['ci95'][0]:+.4f},{code['ci95'][1]:+.4f}]  "
              f"vs RRF {rrf['delta']:+.4f} [{rrf['ci95'][0]:+.4f},{rrf['ci95'][1]:+.4f}]")
    print(f"  full-coverage completeness text-unique {completeness_block['text_unique']:.3f} "
          f"(determined {completeness_block['text_unique_ambiguous_free']:.3f}, "
          f"ambiguous {completeness_block['text_unique_ambiguous']:.3f}; "
          f"CESR-code {completeness_block['cesr_code_text_unique']:.3f})")
    print(f"  passed: {gate['passed']} -> {gate['verdict']}")

    payload = {
        "meta": {
            "repo": str(ROOT),
            "seed": cg.SEED,
            "budgets": list(BUDGETS),
            "constants": {
                "mu": cu.MU, "cost_exponent": cu.COST_EXPONENT,
                "path_bonus": cu.PATH_BONUS, "lex_tiebreak": cu.LEX_TIEBREAK,
                "pool_depth": cu.POOL_DEPTH,
                "bootstrap_resamples": cu.RESAMPLES,
                "bootstrap_seed": cu.BOOTSTRAP_SEED,
            },
            "variant": "oracle-constraint CESR-U",
            "claim": (
                "Synthetic capability test. qrels are graph-derived, so this measures "
                "whether a criterion of universal quantification and exact identity can "
                "execute the evidence relation, not human relevance. Constraints come "
                "from query metadata (type, symbols, module, chain); qrels is never read "
                "by the builder."
            ),
            "systems": ["porter_trunc", "entity_trunc", "graph_trunc", "rrf_trunc",
                        cc.CESR, cu.CESR_U],
        },
        "graph": graph.stats(),
        "corpus": {
            "chunks": len(corpus.ids),
            "corpus_tokens": total_tokens,
            "mean_gold_tokens": gold_tokens,
            "gold_evidence_fits_budget_rate": {
                str(budget): report["reach"][f"gold_fits@{budget}"] / report["queries"]
                for budget in BUDGETS
            },
        },
        "queries": {
            "total": len(queries),
            "by_type": dict(by_type),
            "text_unique": n_unique,
            "conflicting": n_conflict,
            "distinct_texts": len(groups),
            "text_unique_by_type": dict(unique_by_type),
            "conflicting_by_type": dict(conflict_by_type),
            "conflicting_groups_by_type": {
                kind: {
                    "queries": sum(1 for query in queries if query["type"] == kind),
                    "distinct_texts": sum(1 for text, meta in groups.items()
                                          if kind in meta["types"]),
                    "queries_in_conflicting_groups": sum(
                        meta["queries"] for meta in groups.values()
                        if kind in meta["types"] and meta["conflict"]),
                }
                for kind in TYPE_ORDER
            },
        },
        "constraint_language": {
            "operators": [
                "EXISTS_CALLS(X)", "ALL_CALLERS(X)", "EXISTS_DEFINES(X)",
                "EXACT_DEFINES(X,chunk)", "ALL_DEFINES(X)", "ALL_EXPORTS(file)",
                "IMPORTS(file,module)", "SAME_FILE(a,b)", "CALL_CHAIN(X,middle)",
                "CALLS_BOTH(X,Y)",
            ],
            "ambiguity": ambiguity,
        },
        "systems": {
            subset: {
                system: {str(budget): summarize_system(rows, system, budget, subset)
                         for budget in BUDGETS}
                for system in ("porter_trunc", "entity_trunc", "graph_trunc",
                               "rrf_trunc", cc.CESR, cu.CESR_U)
            }
            for subset in SUBSETS
        },
        "mandatory_unbudgeted": {
            subset: cu.summarize([row for row in mandatory if in_subset(row, subset)])
            for subset in SUBSETS
        },
        "paired": paired,
        "coverage_vs_completeness": completeness,
        "determinacy": determinacy,
        "budget_violations": budget_violations,
        "mandatory_integrity": mandatory_integrity,
        "token_ratio_cesr_u_over_rrf": {str(k): v for k, v in token_ratio.items()},
        "missing_gold_causes": missing_report,
        "identity_retrieval": identity_report,
        "per_type_text_unique": type_slice,
        "per_type_determinacy": type_determinacy,
        "latency": latency_report,
        "gate": gate,
        "samples": report["samples"],
    }
    out = HERE / "results" / "cesr_u.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
