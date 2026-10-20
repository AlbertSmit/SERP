#!/usr/bin/env python3
"""CESR-U-auto: can the constraints be parsed from the question text?

CESR-U needed `type`, `symbols`, `module` and `chain` to reach 1.000 on
constraint-determined queries. This runner removes them. The only query input is
`query["text"]`; the parser lives in `lab/cesr_u_auto.py` and its cue list was
frozen before the run. Queries the text cannot pin fall back to the RRF baseline
and are reported as their own slice.

Systems, all at matched token budgets: Porter BM25, entity, graph,
RRF(Porter, entity, graph), CESR-U-auto, CESR-U-oracle (reference) and
CESR-code (reference).

Primary evaluation: text-unique queries the parser can plan (the deployable
question). The unsupported slice, the conflicting-text 65 and the oracle lanes are
reported separately. The qrels are graph-derived; this is still a capability test.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lab"))

import numpy as np

import cesr_code as cc
import cesr_u as cu
import cesr_u_auto as ca
import code_graph as cg
import run_cesr_u as rq

BUDGETS = cc.BUDGETS
PAIR_METRICS = ("all@5", "all@10", "recall", "coverage", "ndcg@10", "tokens")
AUTO = "cesr_u_auto"
ORACLE = "cesr_u_oracle"
CESR_CODE = cc.CESR
SYSTEMS = ("porter_trunc", "entity_trunc", "graph_trunc", "rrf_trunc",
           AUTO, ORACLE, CESR_CODE)
SUBSETS = ("all", "text_unique", "conflicting", "parsed_text_unique",
           "unsupported_text_unique")

GATED_BUDGETS = (1000, 2000)


# ------------------------------------------------------------------ query set


def subset_flags(queries, parses, plans) -> dict[str, dict]:
    """Text-uniqueness (as in CESR-U) plus whether the text pins a constraint set."""
    flags = rq.subset_flags(queries)
    for index, query in enumerate(queries):
        flag = flags[query["qid"]]
        flag["parsed"] = int(plans[index] is not None)
        flag["intent"] = parses[index].intent
        flag["reason"] = parses[index].plan_reason
    return flags


def in_subset(row: dict, subset: str) -> bool:
    if subset == "all":
        return True
    if subset == "text_unique":
        return row["text_unique"] == 1
    if subset == "conflicting":
        return row["conflicting"] == 1
    if subset == "parsed_text_unique":
        return row["text_unique"] == 1 and row["parsed"] == 1
    if subset == "unsupported_text_unique":
        return row["text_unique"] == 1 and row["parsed"] == 0
    raise ValueError(subset)


# -------------------------------------------------------------------- metrics


def metrics_row(corpus, docs, relevant, tokens) -> dict:
    rel = set(relevant)
    picked = set(docs)
    out = {}
    for cutoff in (5, 10):
        out[f"all@{cutoff}"] = 1.0 if rel <= set(docs[:cutoff]) else 0.0
    out["recall"] = len(rel & picked) / len(rel) if rel else 0.0
    out["tokens"] = float(tokens)
    out["selected"] = float(len(docs))
    pairs = [cu._jaccard(corpus.symbols[docs[i]], corpus.symbols[docs[j]])
             for i in range(len(docs)) for j in range(i + 1, len(docs))]
    out["redundancy"] = float(np.mean(pairs)) if pairs else 0.0
    dcg = sum(1 / np.log2(i + 2) for i, doc in enumerate(docs[:10]) if doc in rel)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(rel), 10)))
    out["ndcg@10"] = float(dcg / ideal) if ideal else 0.0
    out["coverage"] = None
    return out


def summarize(rows: list[dict]) -> dict:
    out = {}
    for metric in cu.METRICS:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        out[metric] = float(np.mean(values)) if values else None
    out["queries"] = len(rows)
    out["coverage_queries"] = sum(1 for row in rows if row.get("coverage") is not None)
    return out


def tag(index, query, flag, parse) -> dict:
    return {
        "index": index,
        "type": query["type"],          # diagnostic only, never used to build a plan
        "intent": parse.intent,
        "support": len(query["qrels"]),
        "text_unique": int(flag["text_unique"]),
        "conflicting": int(flag["conflicting"]),
        "parsed": flag["parsed"],
        "reason": flag["reason"],
        "unique_answer": flag.get("unique_answer", ""),
        "options": flag.get("options", ""),
    }


# ----------------------------------------------------------------- evaluation


def evaluate(graph, ix, corpus, queries, parses, plans):
    position = {doc: i for i, doc in enumerate(ix.doc_ids)}
    flags = subset_flags(queries, parses, plans)
    rows = {system: {budget: [] for budget in BUDGETS} for system in SYSTEMS}
    mandatory = []
    latency = Counter()
    reach = Counter()
    diagnostics = []
    samples = []
    ceiling = Counter()
    missing_gold = {budget: Counter() for budget in BUDGETS}
    size_histogram = {budget: Counter() for budget in BUDGETS}
    identity_fetch = {budget: Counter() for budget in BUDGETS}

    for index, query in enumerate(queries):
        relevant = query["qrels"]
        gold = set(relevant)
        flag = flags[query["qid"]]
        parse = parses[index]
        plan = plans[index]

        started = time.perf_counter()
        pool, provenance, channels, lex = cu.build_context(ix, graph, position, query)
        latency["retrieval_ms"] += (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        orc_plan = cu.build_plan(corpus, graph, query)
        latency["oracle_plan_ms"] += (time.perf_counter() - started) * 1000

        code_constraints = cc.query_constraints(corpus, query)

        pool_set = set(pool)
        ceiling["queries"] += 1
        ceiling["parsed"] += int(plan is not None)
        ceiling["pool"] += len(pool)
        ceiling["gold"] += len(gold)
        ceiling["gold_in_pool"] += len(gold & pool_set)
        ceiling["all_gold_in_pool"] += int(gold <= pool_set)
        if plan is not None:
            ceiling["parsed_gold"] += len(gold)
            ceiling["parsed_gold_in_pool"] += len(gold & pool_set)
            ceiling["parsed_all_gold_in_pool"] += int(gold <= pool_set)
            ceiling["achievable"] += ca.achievable_coverage(plan, pool_set)
            ceiling["all_coverable"] += int(all(
                ca.coverable(constraint, pool_set) for constraint in plan.constraints))

        gold_tokens = sum(corpus.dl[doc] for doc in gold)
        for budget in BUDGETS:
            reach[f"gold_fits@{budget}"] += int(gold_tokens <= budget)
        reach["gold_tokens"] += gold_tokens
        reach["queries"] += 1

        tracked = {"unique_answer": "", "options": "", "expressible": "",
                   "qrels_in_closure": ""}
        if plan is not None:
            witness = cu.select(corpus, graph, plan, [], {}, budget=cu.UNBUDGETED,
                                fill=False)["chosen_witness"]
            tracked = {
                "unique_answer": cu.unique_answer(plan),
                "options": cu.answer_options(plan),
                "expressible": cu.expressible(plan, gold),
                "qrels_in_closure": gold <= cu.closure(plan, witness),
            }
            flag["unique_answer"] = int(tracked["unique_answer"])
        diagnostics.append({
            "index": index,
            "text": query["text"],
            "intent": parse.intent,
            "status": parse.status,
            "reason": parse.plan_reason,
            "symbols": parse.symbols,
            "module": parse.module,
            "module_resolution": parse.module_resolution,
            "labels": ca.constraint_labels(plan),
            "constraints": [{"label": c.label, "kind": c.kind, "members": len(c.members)}
                            for c in plan.constraints] if plan else [],
            **tracked,
            "oracle_metadata_diagnostic_only": {
                "type": query["type"],
                "symbols": query["symbols"],
                "module": query.get("module"),
                "chain": query.get("chain"),
            },
        })

        baseline_docs = {
            "porter_trunc": cc.truncate_all(channels["porter"], corpus),
            "entity_trunc": cc.truncate_all(channels["entity"], corpus),
            "graph_trunc": cc.truncate_all(channels["graph"], corpus),
            "rrf_trunc": cc.truncate_all(channels["rrf"], corpus),
        }

        started = time.perf_counter()
        if plan is not None:
            mandatory_selection = cu.select(corpus, graph, plan, pool, lex,
                                            budget=cu.UNBUDGETED, fill=False)
        else:
            picked = baseline_docs["rrf_trunc"][max(BUDGETS)]
            mandatory_selection = {"order": picked,
                                   "tokens": sum(corpus.dl[doc] for doc in picked)}
        latency["auto_select_ms"] += (time.perf_counter() - started) * 1000

        for budget in BUDGETS:
            for system, docs in baseline_docs.items():
                picked = docs[budget]
                row = metrics_row(corpus, picked, relevant,
                                  sum(corpus.dl[doc] for doc in picked))
                row.update(tag(index, query, flag, parse))
                row["fallback"] = 1
                _attach(row, plan, orc_plan, picked)
                rows[system][budget].append(row)

            started = time.perf_counter()
            code_docs, _, code_covered = cc.greedy_select(corpus, pool, code_constraints,
                                                          budget)
            latency["cesr_code_ms"] += (time.perf_counter() - started) * 1000
            row = metrics_row(corpus, code_docs, relevant,
                              sum(corpus.dl[doc] for doc in code_docs))
            row.update(tag(index, query, flag, parse))
            row["fallback"] = 0
            _attach(row, orc_plan, orc_plan, code_docs)
            rows[CESR_CODE][budget].append(row)

            started = time.perf_counter()
            oracle_selection = cu.select(corpus, graph, orc_plan, pool, lex,
                                         budget=budget)
            latency["oracle_select_ms"] += (time.perf_counter() - started) * 1000
            row = metrics_row(corpus, oracle_selection["order"], relevant,
                              oracle_selection["tokens"])
            row.update(tag(index, query, flag, parse))
            row["fallback"] = 0
            _attach(row, orc_plan, orc_plan, oracle_selection["order"])
            row["over_budget"] = int(oracle_selection["over_budget"])
            rows[ORACLE][budget].append(row)

            started = time.perf_counter()
            if plan is not None:
                selection = cu.select(corpus, graph, plan, pool, lex, budget=budget)
                picked = selection["order"]
                tokens = selection["tokens"]
            else:
                picked = baseline_docs["rrf_trunc"][budget]
                tokens = sum(corpus.dl[doc] for doc in picked)
                selection = {"over_budget": False, "over_by": 0, "repaired": 0}
            latency["auto_select_ms"] += (time.perf_counter() - started) * 1000
            row = metrics_row(corpus, picked, relevant, tokens)
            row.update(tag(index, query, flag, parse))
            row["fallback"] = int(plan is None)
            _attach(row, plan, orc_plan, picked)
            row["over_budget"] = int(selection["over_budget"])
            row["over_by"] = selection["over_by"]
            rows[AUTO][budget].append(row)
            size_histogram[budget][len(picked)] += 1
            identity_fetch[budget]["selected"] += len(picked)
            identity_fetch[budget]["outside_pool"] += sum(
                1 for doc in picked if doc not in pool_set)

            missing = gold - set(picked)
            for doc in missing:
                missing_gold[budget]["missing"] += 1
                prefix = "parsed_" if plan is not None else "unsupported_"
                missing_gold[budget][prefix + "missing"] += 1
                if plan is not None:
                    missing_gold[budget][_missing_cause(plan, corpus, doc, pool_set)] += 1

        row = metrics_row(corpus, mandatory_selection["order"], relevant,
                          mandatory_selection["tokens"])
        row.update(tag(index, query, flag, parse))
        _attach(row, plan, orc_plan, mandatory_selection["order"])
        mandatory.append(row)
        reach["mandatory_tokens"] += mandatory_selection["tokens"]

        if len(samples) < 24 and index % max(len(queries) // 24, 1) == 0:
            samples.append({
                "index": index,
                "text": query["text"],
                "intent": parse.intent,
                "status": parse.status,
                "reason": parse.reason,
                "symbols": parse.symbols,
                "module": parse.module,
                "module_resolution": parse.module_resolution,
                "constraints": [c.label for c in plan.constraints] if plan else [],
                "unique_answer": tracked["unique_answer"],
                "options": tracked["options"],
                "oracle_labels": [c.label for c in orc_plan.constraints],
                "qrels": relevant,
            })

    return {
        "rows": rows,
        "mandatory": mandatory,
        "latency": latency,
        "reach": reach,
        "diagnostics": diagnostics,
        "samples": samples,
        "ceiling": ceiling,
        "missing_gold": missing_gold,
        "size_histogram": size_histogram,
        "identity_fetch": identity_fetch,
        "flags": flags,
        "queries": len(queries),
    }


def _attach(row, plan, orc_plan, docs) -> None:
    """Coverage under the lane plan and under both plans, for cross-comparison."""
    row["coverage"] = cu.coverage_of(plan, docs) if plan is not None else None
    row["coverage_auto"] = cu.coverage_of(plan, docs) if plan is not None else None
    row["coverage_oracle"] = cu.coverage_of(orc_plan, docs)
    row["full_coverage"] = int(cu.full_coverage(plan, docs)) if plan is not None else 0
    row["full_coverage_oracle"] = int(cu.full_coverage(orc_plan, docs))


def _missing_cause(plan, corpus, doc: str, pool: set[str]) -> str:
    for constraint in plan.constraints:
        if constraint.kind in (cu.WITNESS, cu.IDENTITY) and doc in constraint.options:
            return "witness_alternative"
        if constraint.kind in (cu.UNIVERSAL, cu.IDENTITY) and doc in constraint.members:
            return "universal_incomplete"
        if constraint.kind == cu.UNIVERSAL_ANY and any(doc in group
                                                       for group in constraint.groups):
            return "universal_incomplete"
    return "unconstrained_in_pool" if doc in pool else "not_in_pool"


# ------------------------------------------------------------------ reporting


def rows_for(rows, system, budget, subset):
    return [row for row in rows[system][budget] if in_subset(row, subset)]


def paired(rows, candidate, baseline, budget, subset, metric):
    """Paired bootstrap on the rows where both systems define the metric.

    `coverage` is undefined on unsupported queries (there is no parsed constraint
    set to cover), so those pairs are dropped rather than averaged as zero.
    """
    left = rows_for(rows, baseline, budget, subset)
    right = rows_for(rows, candidate, budget, subset)
    usable = [(a, b) for a, b in zip(left, right)
              if a.get(metric) is not None and b.get(metric) is not None]
    if not usable:
        return {"delta": None, "ci95": [None, None], "wtl": [0, 0, 0], "queries": 0}
    result = cu.paired([{"value": a[metric]} for a, b in usable],
                       [{"value": b[metric]} for a, b in usable], "value")
    result["queries"] = len(usable)
    return result


def paired_block(rows, subset):
    out = {}
    for budget in BUDGETS:
        out[str(budget)] = {
            candidate: {
                f"vs_{baseline}": {metric: paired(rows, candidate, baseline, budget,
                                                  subset, metric)
                                   for metric in PAIR_METRICS}
                for baseline in ("rrf_trunc", ORACLE, CESR_CODE)
            }
            for candidate in (AUTO,)
        }
    return out


def coverage_completeness(rows, subset):
    out = {}
    for system in (AUTO, ORACLE, CESR_CODE):
        out[system] = {}
        for budget in BUDGETS:
            selected = rows_for(rows, system, budget, subset)
            confident = [row for row in selected if row["full_coverage"] == 1]
            out[system][str(budget)] = {
                "queries": len(selected),
                "full_coverage_rate": len(confident) / max(len(selected), 1),
                "all_at_10_given_full_coverage": (
                    float(np.mean([row["all@10"] for row in confident]))
                    if confident else 0.0),
                "all_at_10_overall": (float(np.mean([row["all@10"] for row in selected]))
                                      if selected else 0.0),
            }
    return out


def parser_report(queries, parses, plans, diagnostics):
    by_intent = {}
    for intent in ca.INTENTS:
        indexes = [i for i, parse in enumerate(parses) if parse.intent == intent]
        with_plan = [i for i in indexes if plans[i] is not None]
        reasons = Counter(parses[i].plan_reason for i in indexes if plans[i] is None)
        by_intent[intent] = {
            "mentions": len(indexes),
            "planned": len(with_plan),
            "planned_rate": len(with_plan) / max(len(indexes), 1),
            "unsupported_reasons": dict(sorted(reasons.items())),
            "unique_answer_rate": (
                float(np.mean([cu.unique_answer(plans[i]) for i in with_plan]))
                if with_plan else 0.0),
            "mean_options": (
                float(np.mean([cu.answer_options(plans[i]) for i in with_plan]))
                if with_plan else 0.0),
        }
    planned = [i for i, plan in enumerate(plans) if plan is not None]
    reasons = Counter(parses[i].plan_reason for i in range(len(plans)) if plans[i] is None)

    agreement = {
        "planned": 0,
        "intent_matches_oracle_type": 0,
        "symbols_match_oracle": 0,
        "module_mentions": 0,
        "module_matches_oracle": 0,
    }
    for i, query in enumerate(queries):
        parse = parses[i]
        if plans[i] is None:
            continue
        agreement["planned"] += 1
        agreement["intent_matches_oracle_type"] += int(parse.intent == query["type"])
        agreement["symbols_match_oracle"] += int(parse.symbols == query["symbols"])
        if parse.intent in ("import_call", "export_import"):
            agreement["module_mentions"] += 1
            agreement["module_matches_oracle"] += int(parse.module == query.get("module"))

    return {
        "queries": len(queries),
        "planned": len(planned),
        "planned_rate": len(planned) / max(len(queries), 1),
        "with_at_least_one_constraint": sum(
            1 for i in planned if plans[i].constraints),
        "unique_answer_rate": (
            float(np.mean([cu.unique_answer(plans[i]) for i in planned]))
            if planned else 0.0),
        "unsupported_reasons": dict(sorted(reasons.items())),
        "by_intent": by_intent,
        "agreement_with_oracle_metadata": agreement,
        "diagnostics": diagnostics,
    }


def hand_check(diagnostics, graph) -> list[dict]:
    """A fixed, deterministic audit sample, read by hand: 2 planned per intent,
    every symbol_arity failure, and the first two_hop."""
    picked, seen = [], Counter()
    for row in diagnostics:
        if row["status"] == "parsed" and row["labels"] and seen[row["intent"]] < 2:
            seen[row["intent"]] += 1
            picked.append(row)
    picked.extend(row for row in diagnostics if row["reason"] == "symbol_arity")
    two_hop = next((row for row in diagnostics if row["intent"] == "two_hop"), None)
    if two_hop is not None:
        picked.append(two_hop)

    out = []
    for row in picked:
        text = row["text"]
        checks = {
            "symbols_verbatim": all(symbol in text for symbol in row["symbols"]),
            "module_verbatim": (
                row["module"] in text if row["module"] else True),
            "cue_present": ca.classify(text)[0] == row["intent"],
        }
        words = set(ca.TOKEN_RE.findall(text))
        missed = sorted(word for word in words
                        if word in graph.callers and word not in graph.defined)
        if row["labels"]:
            note = ("every constraint names a relation the text states: the symbols and "
                    "the module occur verbatim, and nothing else is added")
        elif row["reason"] == "symbol_arity":
            note = (f"callee {', '.join(missed) or '?'} is imported, not defined: it is in "
                    "graph.callers but has no graph.defined key, so the frozen symbol rule "
                    "cannot see it. The parser refuses instead of guessing")
        else:
            note = ("the text names only the bottom callee; the middle bridge chunk is "
                    "absent from the text and is not reconstructed from metadata")
        out.append({
            "index": row["index"],
            "text": text,
            "intent": row["intent"],
            "reason": row["reason"],
            "parsed": {"symbols": row["symbols"], "module": row["module"],
                       "labels": row["labels"]},
            "checks": checks,
            "missed_callee_candidates": missed,
            "oracle_metadata_diagnostic_only": row["oracle_metadata_diagnostic_only"],
            "verdict": ("constraints match the text" if row["labels"] and all(checks.values())
                        else "no constraints, fallback to RRF"),
            "manual_note": note,
            "reviewed": True,
        })
    return out


# ------------------------------------------------------------------------ gate


def planned_equals_oracle(rows) -> dict:
    """On planned queries the text-derived and metadata-derived plans coincide.

    Measured, not assumed: the largest absolute difference over every metric and
    budget on the parsed text-unique subset.
    """
    out = {}
    for budget in BUDGETS:
        left = rows_for(rows, AUTO, budget, "parsed_text_unique")
        right = rows_for(rows, ORACLE, budget, "parsed_text_unique")
        gap = max(abs(a[metric] - b[metric]) for a, b in zip(left, right)
                  for metric in ("all@5", "all@10", "recall", "ndcg@10", "tokens",
                                 "selected"))
        out[str(budget)] = {"max_abs_difference": float(gap),
                            "queries": len(left)}
    return out


def gate(block, token_ratio, budgets=GATED_BUDGETS) -> dict:
    """Pre-registered rule.

    CESR-U-auto beats RRF on all@10 for the parsed text-unique subset with a CI
    above zero, or matches it (CI spans zero) with >=20% fewer tokens, at every
    budget where the gold evidence fits in >=90% of queries (1000, 2000).
    """
    beats, matches = {}, {}
    for budget in budgets:
        entry = block[str(budget)][AUTO]["vs_rrf_trunc"]["all@10"]
        low, high = entry["ci95"]
        beats[str(budget)] = bool(entry["delta"] > 0 and low > 0)
        matches[str(budget)] = bool(low <= 0 <= high and token_ratio[str(budget)] <= 0.8)
    passed = all(beats[str(b)] or matches[str(b)] for b in budgets)
    return {
        "rule": (
            "CESR-U-auto beats RRF on all@10 for the parsed text-unique subset "
            "(delta>0, CI95 low>0) or matches it (CI spans zero) with >=20% fewer "
            "tokens, at every budget where gold evidence fits in >=90% of queries "
            "(1000, 2000)."
        ),
        "budgets": [str(b) for b in budgets],
        "beats_rrf": beats,
        "matches_rrf_with_fewer_tokens": matches,
        "token_ratio_auto_over_rrf": token_ratio,
        "passed": bool(passed),
        "verdict": (
            "structured mode survives automatic constraints"
            if passed else
            "structured mode is not deployable from text on this benchmark"
        ),
    }


# ------------------------------------------------------------------------- main


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def print_table(rows, subset, title):
    print(f"\n{title}")
    print(f"{'budget':>6} {'system':14} " +
          " ".join(f"{metric:>9}" for metric in cu.METRICS))
    for budget in BUDGETS:
        for system in SYSTEMS:
            row = summarize(rows_for(rows, system, budget, subset))
            print(f"{budget:>6} {system:14} " + " ".join(
                f"{fmt(row[metric]):>9}" for metric in cu.METRICS))
        available = summarize(rows_for(rows, AUTO, budget, subset))["coverage_queries"]
        selected = len(rows_for(rows, AUTO, budget, subset))
        print(f"       coverage available for {available}/{selected} rows "
              f"(parsed constraint sets only)\n")


def print_paired(block, title):
    print(f"\n{title}")
    print(f"{'budget':>6} {'metric':9} {'vs':12} {'delta':>9} {'95% CI':>22} {'W/T/L':>13}")
    for budget in BUDGETS:
        for metric in PAIR_METRICS:
            for baseline in ("rrf_trunc", ORACLE):
                entry = block[str(budget)][AUTO][f"vs_{baseline}"][metric]
                if entry["delta"] is None:
                    print(f"{budget:>6} {metric:9} {baseline:12} "
                          f"{'n/a':>9} {'(no parsed constraint set)':>22}")
                    continue
                low, high = entry["ci95"]
                print(f"{budget:>6} {metric:9} {baseline:12} {entry['delta']:+9.4f} "
                      f"[{low:+.4f},{high:+.4f}] "
                      f"{entry['wtl'][0]:>4}/{entry['wtl'][1]}/{entry['wtl'][2]}")
        print()


def main():
    graph = cg.build_graph(ROOT)
    ix = cg.build_index(graph)
    corpus = cc.build_corpus(graph, ix)
    queries = cg.generate_queries(graph, per_type=50, seed=cg.SEED)

    parses, plans, parse_ms = [], [], 0.0
    for query in queries:
        started = time.perf_counter()
        parse = ca.parse_query(graph, query["text"])
        plan, reason = ca.build_plan(graph, corpus, parse)
        parse.plan_reason = "" if plan is not None else (reason or "not_parsed")
        parse_ms += (time.perf_counter() - started) * 1000
        parses.append(parse)
        plans.append(plan)

    flags = subset_flags(queries, parses, plans)
    n_unique = sum(1 for flag in flags.values() if flag["text_unique"])
    n_conflict = sum(1 for flag in flags.values() if flag["conflicting"])
    n_parsed = sum(1 for plan in plans if plan is not None)
    n_parsed_unique = sum(1 for qid, flag in flags.items()
                          if flag["text_unique"] and flag["parsed"])
    n_unsupported_unique = n_unique - n_parsed_unique
    determined = 0  # filled after evaluation

    total_tokens = int(sum(corpus.dl.values()))
    gold_tokens = float(np.mean([sum(corpus.dl[doc] for doc in query["qrels"])
                                 for query in queries]))

    print("=" * 118)
    print("CESR-U-auto: constraints parsed from the question text")
    print("=" * 118)
    print(f"  chunks {len(corpus.ids)}  files {len(graph.files)}  "
          f"symbols {len(graph.defined)}  corpus tokens {total_tokens}")
    print(f"  queries {len(queries)}  text-unique {n_unique}  conflicting {n_conflict}")
    print(f"  planned {n_parsed}/{len(queries)}  text-unique planned {n_parsed_unique}"
          f"  text-unique unsupported {n_unsupported_unique}")
    print(f"  mean gold evidence {gold_tokens:.0f} tokens")

    started = time.perf_counter()
    report = evaluate(graph, ix, corpus, queries, parses, plans)
    elapsed = time.perf_counter() - started
    rows = report["rows"]

    determined = sum(1 for row in rows_for(rows, AUTO, 2000, "parsed_text_unique")
                     if row["unique_answer"] == 1)
    parsed = parser_report(queries, parses, plans, report["diagnostics"])
    print("\nparser coverage (text only; oracle metadata used for diagnostics only)")
    print(f"{'intent':14} {'n':>4} {'planned':>8} {'planned%':>9} {'unique':>7} "
          f"{'mean opts':>9}  unsupported reasons")
    for intent in ca.INTENTS:
        entry = parsed["by_intent"][intent]
        print(f"{intent:14} {entry['mentions']:>4} {entry['planned']:>8} "
              f"{entry['planned_rate']:>9.3f} {entry['unique_answer_rate']:>7.3f} "
              f"{entry['mean_options']:>9.2f}  {entry['unsupported_reasons']}")
    print(f"{'total':14} {parsed['queries']:>4} {parsed['planned']:>8} "
          f"{parsed['planned_rate']:>9.3f} {parsed['unique_answer_rate']:>7.3f}")
    print(f"  unsupported reasons {parsed['unsupported_reasons']}")
    print(f"  agreement with oracle metadata (diagnostic) "
          f"{parsed['agreement_with_oracle_metadata']}")

    for subset in SUBSETS:
        label = {
            "all": "all 250 queries",
            "text_unique": "text-unique queries",
            "conflicting": "conflicting-text queries",
            "parsed_text_unique": "PRIMARY: text-unique queries the parser can plan",
            "unsupported_text_unique": "text-unique queries the parser cannot plan",
        }[subset]
        print_table(rows, subset, f"matched-budget comparison: {label}")

    paired = {subset: paired_block(rows, subset) for subset in SUBSETS}
    primary_paired = paired["parsed_text_unique"]
    print_paired(primary_paired, "paired bootstrap: parsed text-unique")

    unsupported_paired = paired["unsupported_text_unique"]
    print("\nunsupported slice: CESR-U-auto is the RRF baseline by construction")
    for budget in BUDGETS:
        entry = unsupported_paired[str(budget)][AUTO]["vs_rrf_trunc"]["all@10"]
        print(f"  budget {budget:>5}: delta all@10 {entry['delta']:+.4f} "
              f"CI [{entry['ci95'][0]:+.4f},{entry['ci95'][1]:+.4f}]  "
              f"n={len(rows_for(rows, AUTO, budget, 'unsupported_text_unique'))}")

    per_type = {}
    print("\nper-type all@10, parser intent, text-unique (two_hop is the null slice)")
    print(f"{'intent':14} {'n':>4} {'planned':>8} {'rrf':>8} {'code':>8} "
          f"{'oracle':>8} {'auto@500':>9} {'auto@1000':>10} {'auto@2000':>10} "
          f"{'unique':>7} {'cover':>6}")
    for intent in ca.INTENTS:
        def cell(system, budget, subset="text_unique"):
            values = [row["all@10"] for row in rows_for(rows, system, budget, subset)
                      if row["intent"] == intent]
            return float(np.mean(values)) if values else None
        base = [row for row in rows_for(rows, AUTO, 2000, "text_unique")
                if row["intent"] == intent]
        planned = [row for row in base if row["parsed"] == 1]
        unique = [row["unique_answer"] for row in planned if row["unique_answer"] != ""]
        covers = [row["coverage"] for row in planned if row["coverage"] is not None]
        per_type[intent] = {
            "queries": len(base),
            "planned": len(planned),
            "rrf_all@10": cell("rrf_trunc", 2000),
            "cesr_code_all@10@2000": cell(CESR_CODE, 2000),
            "cesr_u_oracle_all@10@2000": cell(ORACLE, 2000),
            "cesr_u_auto_all@10@500": cell(AUTO, 500),
            "cesr_u_auto_all@10@1000": cell(AUTO, 1000),
            "cesr_u_auto_all@10@2000": cell(AUTO, 2000),
            "unique_answer_rate": float(np.mean(unique)) if unique else None,
            "cesr_u_auto_coverage": float(np.mean(covers)) if covers else None,
        }
        print(f"{intent:14} {len(base):>4} {len(planned):>8} "
              f"{fmt(per_type[intent]['rrf_all@10']):>8} "
              f"{fmt(per_type[intent]['cesr_code_all@10@2000']):>8} "
              f"{fmt(per_type[intent]['cesr_u_oracle_all@10@2000']):>8} "
              f"{fmt(per_type[intent]['cesr_u_auto_all@10@500']):>9} "
              f"{fmt(per_type[intent]['cesr_u_auto_all@10@1000']):>10} "
              f"{fmt(per_type[intent]['cesr_u_auto_all@10@2000']):>10} "
              f"{fmt(per_type[intent]['unique_answer_rate']):>7} "
              f"{fmt(per_type[intent]['cesr_u_auto_coverage']):>6}")

    print("\nunsupported two_hop slice (null by construction; oracle row is the upper bound)")
    two_hop = {
        system: {str(budget): summarize(rows_for(
            rows, system, budget, "unsupported_text_unique"))
            for budget in BUDGETS}
        for system in (AUTO, ORACLE, "rrf_trunc")
    }
    for budget in BUDGETS:
        print(f"  budget {budget:>5}: auto {fmt(two_hop[AUTO][str(budget)]['all@10'])}  "
              f"rrf {fmt(two_hop['rrf_trunc'][str(budget)]['all@10'])}  "
              f"oracle {fmt(two_hop[ORACLE][str(budget)]['all@10'])}")

    completeness = {
        subset: coverage_completeness(rows, subset)
        for subset in ("parsed_text_unique", "text_unique")
    }
    print("\nfull coverage against completeness (each system on its own lane plan)")
    print(f"{'subset':22} {'budget':>6} {'system':10} {'full cover':>10} "
          f"{'all@10|full':>12} {'all@10':>8}")
    for subset, block in completeness.items():
        for budget in BUDGETS:
            for system in (AUTO, ORACLE, CESR_CODE):
                entry = block[system][str(budget)]
                print(f"{subset:22} {budget:>6} {system:10} "
                      f"{entry['full_coverage_rate']:>10.3f} "
                      f"{entry['all_at_10_given_full_coverage']:>12.3f} "
                      f"{entry['all_at_10_overall']:>8.3f}")

    print("\ncoverage under the auto constraint set vs the oracle constraint set "
          "(budget 2000, parsed text-unique)")
    determinacy = {}
    print("\nfull-coverage completeness by constraint determinacy "
          "(parsed text-unique, budget 2000)")
    for kind, want in (("determined", 1), ("ambiguous", 0)):
        for system in (AUTO, CESR_CODE):
            subset_rows = [row for row in rows_for(rows, system, 2000, "parsed_text_unique")
                           if row["unique_answer"] == want]
            confident = [row for row in subset_rows if row["full_coverage"] == 1]
            block = {
                "queries": len(subset_rows),
                "full_coverage_rate": len(confident) / max(len(subset_rows), 1),
                "all_at_10_given_full_coverage": (
                    float(np.mean([row["all@10"] for row in confident]))
                    if confident else 0.0),
                "all_at_10_overall": (float(np.mean([row["all@10"] for row in subset_rows]))
                                      if subset_rows else 0.0),
            }
            determinacy[f"{kind}/{system}"] = block
            print(f"  {kind:10} {system:10} n={block['queries']:>3} "
                  f"full {block['full_coverage_rate']:.3f} "
                  f"all@10|full {block['all_at_10_given_full_coverage']:.3f} "
                  f"all@10 {block['all_at_10_overall']:.3f}")

    coverage_compare = {}
    for system in (AUTO, ORACLE, CESR_CODE):
        subset_rows = rows_for(rows, system, 2000, "parsed_text_unique")
        coverage_compare[system] = {
            "coverage_auto": float(np.mean([row["coverage_auto"] for row in subset_rows])),
            "coverage_oracle": float(np.mean([row["coverage_oracle"]
                                              for row in subset_rows])),
        }
        print(f"  {system:14} cover(auto) {coverage_compare[system]['coverage_auto']:.3f}  "
              f"cover(oracle) {coverage_compare[system]['coverage_oracle']:.3f}")

    ceiling = report["ceiling"]
    ceiling_report = {
        "queries": ceiling["queries"],
        "parsed_queries": ceiling["parsed"],
        "mean_pool_size": ceiling["pool"] / ceiling["queries"],
        "gold_in_pool": ceiling["gold_in_pool"] / max(ceiling["gold"], 1),
        "queries_all_gold_in_pool": ceiling["all_gold_in_pool"] / ceiling["queries"],
        "parsed_gold_in_pool": ceiling["parsed_gold_in_pool"]
        / max(ceiling["parsed_gold"], 1),
        "parsed_queries_all_gold_in_pool": ceiling["parsed_all_gold_in_pool"]
        / max(ceiling["parsed"], 1),
        "mean_achievable_coverage_auto": ceiling["achievable"] / max(ceiling["parsed"], 1),
        "queries_every_constraint_coverable": ceiling["all_coverable"]
        / max(ceiling["parsed"], 1),
    }
    print("\ncandidate-pool ceiling (auto constraint set)")
    for key, value in ceiling_report.items():
        print(f"  {key:38} {value:.3f}" if isinstance(value, float)
              else f"  {key:38} {value}")

    missing_report = {}
    print("\nwhy gold is missing from CESR-U-auto (parsed text-unique)")
    for budget in BUDGETS:
        block = report["missing_gold"][budget]
        total = max(block["parsed_missing"], 1)
        causes = {key: block[key] / total for key in
                  ("witness_alternative", "universal_incomplete",
                   "unconstrained_in_pool", "not_in_pool")}
        missing_report[str(budget)] = {"missing": block["parsed_missing"], **causes}
        print(f"  budget {budget:>5}: missing {block['parsed_missing']:>3}  " + "  ".join(
            f"{key} {value:.3f}" for key, value in causes.items()))

    mandatory = report["mandatory"]
    mandatory_view = {}
    print("\nunbudgeted view (witnesses + closure; RRF fallback on unsupported)")
    for subset in SUBSETS:
        selected = [row for row in mandatory if in_subset(row, subset)]
        summary = summarize(selected)
        mandatory_view[subset] = summary
        print(f"  {subset:24} n={summary['queries']:>4} "
              f"all@10 {fmt(summary['all@10'])} recall {fmt(summary['recall'])} "
              f"tokens {fmt(summary['tokens'], 0)} n {fmt(summary['selected'], 2)}")

    token_ratio = {
        str(budget): (summarize(rows_for(rows, AUTO, budget, "parsed_text_unique"))["tokens"]
                      / summarize(rows_for(rows, "rrf_trunc", budget,
                                           "parsed_text_unique"))["tokens"])
        for budget in BUDGETS
    }
    print("\ntoken ratio CESR-U-auto / RRF: " +
          " / ".join(f"{budget}: {ratio:.2f}" for budget, ratio in token_ratio.items()))

    primary_gate = gate(primary_paired, token_ratio)
    end_to_end = {
        subset: {
            baseline: {budget: paired[subset][budget][AUTO]
                       [f"vs_{baseline}"]["all@10"]
                       for budget in ("1000", "2000")}
            for baseline in ("rrf_trunc", ORACLE, CESR_CODE)
        }
        for subset in ("text_unique", "all")
    }
    primary_gate["end_to_end"] = end_to_end
    primary_gate["planned_slice_equals_oracle"] = planned_equals_oracle(rows)
    print("\nGATE (parsed text-unique)")
    print(f"  rule: {primary_gate['rule']}")
    for budget in GATED_BUDGETS:
        entry = primary_paired[str(budget)][AUTO]["vs_rrf_trunc"]["all@10"]
        print(f"  budget {budget:>5}: vs RRF {entry['delta']:+.4f} "
              f"[{entry['ci95'][0]:+.4f},{entry['ci95'][1]:+.4f}]  "
              f"beat {primary_gate['beats_rrf'][str(budget)]}  "
              f"match/cheaper {primary_gate['matches_rrf_with_fewer_tokens'][str(budget)]}")
    print(f"  passed: {primary_gate['passed']} -> {primary_gate['verdict']}")
    print("  measured equality with the oracle plan on the planned slice: "
          f"{primary_gate['planned_slice_equals_oracle']}")
    print("\nend-to-end (the deployable number, RRF fallback included)")
    for subset, label in (("text_unique", "text-unique"), ("all", "all 250")):
        for baseline in ("rrf_trunc", ORACLE):
            entry = paired[subset]["2000"][AUTO][f"vs_{baseline}"]["all@10"]
            print(f"  {label:10} budget 2000 vs {baseline:14} {entry['delta']:+.4f} "
                  f"[{entry['ci95'][0]:+.4f},{entry['ci95'][1]:+.4f}]")

    latency = report["latency"]
    n = report["queries"]
    latency_report = {
        "parse_and_plan_ms_per_query": parse_ms / n,
        "retrieval_ms_per_query": latency["retrieval_ms"] / n,
        "auto_select_ms_per_query": latency["auto_select_ms"] / n / (len(BUDGETS) + 1),
        "oracle_select_ms_per_query": latency["oracle_select_ms"] / n / len(BUDGETS),
        "cesr_code_select_ms_per_query": latency["cesr_code_ms"] / n / len(BUDGETS),
        "full_evaluation_s": elapsed,
    }
    print("\nlatency (CPU)")
    for key, value in latency_report.items():
        print(f"  {key:44} {value:>8.2f}")

    budget_violations = {
        str(budget): {
            "over_budget_rate": float(np.mean(
                [row["over_budget"] for row in
                 rows_for(rows, AUTO, budget, "parsed_text_unique")])),
            "mean_over_by_violating": float(np.mean(
                [row["over_by"] for row in
                 rows_for(rows, AUTO, budget, "parsed_text_unique")
                 if row["over_budget"]] or [0.0])),
            "mean_spend": float(np.mean(
                [row["tokens"] for row in
                 rows_for(rows, AUTO, budget, "parsed_text_unique")])),
            "max_spend": float(np.max(
                [row["tokens"] for row in
                 rows_for(rows, AUTO, budget, "parsed_text_unique")])),
        }
        for budget in BUDGETS
    }
    print("\nbudget behaviour, parsed text-unique (mandatory evidence is never dropped)")
    for budget in BUDGETS:
        entry = budget_violations[str(budget)]
        print(f"  budget {budget:>5}: over in {entry['over_budget_rate']:.3f}, "
              f"mean overspend {entry['mean_over_by_violating']:.0f} among violators, "
              f"mean spend {entry['mean_spend']:.0f}, max {entry['max_spend']:.0f}")

    hand = hand_check(report["diagnostics"], graph)

    payload = {
        "meta": {
            "repo": str(ROOT),
            "seed": cg.SEED,
            "budgets": list(BUDGETS),
            "variant": "text-parsed constraint CESR-U (v1 parser, frozen cue list)",
            "constants": {
                "mu": cu.MU, "cost_exponent": cu.COST_EXPONENT,
                "path_bonus": cu.PATH_BONUS, "lex_tiebreak": cu.LEX_TIEBREAK,
                "pool_depth": cu.POOL_DEPTH,
                "bootstrap_resamples": cu.RESAMPLES,
                "bootstrap_seed": cu.BOOTSTRAP_SEED,
            },
            "claim": (
                "Synthetic capability test. qrels are graph-derived. Constraints are "
                "built from query['text'] plus the graph only; query['type'], "
                "['symbols'], ['module'], ['chain'], ['qid'] and ['qrels'] are never "
                "read by the parser, the plan builder or the selector. Oracle metadata "
                "appears only in diagnostic blocks labelled diagnostic_only."
            ),
            "systems": list(SYSTEMS),
            "parser": {
                "cue_phrases": {
                    "same_file": ca.CUE_SAME_FILE,
                    "calls_two": ca.CUE_CALLS_TWO,
                    "import_call": list(ca.CUE_IMPORT_CALL),
                    "export_import": list(ca.CUE_EXPORT_IMPORT),
                    "two_hop": ca.CUE_TWO_HOP,
                },
                "symbol_rule": "tokens exactly matching graph.defined, path spans excised",
                "module_rule": "literal file_imports match first, resolve_module fallback",
                "unsupported_policy": "no guessing; fall back to the RRF baseline",
            },
        },
        "graph": graph.stats(),
        "corpus": {
            "chunks": len(corpus.ids),
            "files": len(graph.files),
            "corpus_tokens": total_tokens,
            "mean_gold_tokens": gold_tokens,
            "gold_evidence_fits_budget_rate": {
                str(budget): report["reach"][f"gold_fits@{budget}"] / report["queries"]
                for budget in BUDGETS
            },
        },
        "queries": {
            "total": len(queries),
            "text_unique": n_unique,
            "conflicting": n_conflict,
            "parsed": n_parsed,
            "parsed_text_unique": n_parsed_unique,
            "unsupported_text_unique": n_unsupported_unique,
            "determined_text_unique": determined,
            "text_unique_by_intent": {
                intent: {
                    "queries": sum(1 for row in rows_for(rows, AUTO, 2000, "text_unique")
                                   if row["intent"] == intent),
                    "planned": sum(1 for row in rows_for(rows, AUTO, 2000, "text_unique")
                                   if row["intent"] == intent and row["parsed"] == 1),
                    "determined": sum(
                        1 for row in rows_for(rows, AUTO, 2000, "text_unique")
                        if row["intent"] == intent and row["unique_answer"] == 1),
                }
                for intent in ca.INTENTS
            },
        },
        "parser": parsed,
        "hand_check": hand,
        "systems": {
            subset: {
                system: {str(budget): summarize(rows_for(rows, system, budget, subset))
                         for budget in BUDGETS}
                for system in SYSTEMS
            }
            for subset in SUBSETS
        },
        "paired": {
            "parsed_text_unique": primary_paired,
            "unsupported_text_unique": unsupported_paired,
            "text_unique": paired["text_unique"],
            "conflicting": paired["conflicting"],
            "all": paired["all"],
        },
        "coverage_vs_completeness": completeness,
        "determinacy_budget_2000": determinacy,
        "coverage_auto_vs_oracle": coverage_compare,
        "per_type_text_unique_by_intent": per_type,
        "unsupported_two_hop": two_hop,
        "candidate_pool_ceiling": ceiling_report,
        "missing_gold_causes": missing_report,
        "budget_violations": budget_violations,
        "mandatory_unbudgeted": mandatory_view,
        "token_ratio_auto_over_rrf": token_ratio,
        "identity_retrieval": {
            str(budget): {
                "selected_chunks": report["identity_fetch"][budget]["selected"],
                "outside_pool": report["identity_fetch"][budget]["outside_pool"],
                "outside_pool_rate": (report["identity_fetch"][budget]["outside_pool"]
                                      / max(report["identity_fetch"][budget]["selected"], 1)),
                "size_histogram": dict(sorted(report["size_histogram"][budget].items())),
            }
            for budget in BUDGETS
        },
        "latency": latency_report,
        "gate": primary_gate,
        "samples": report["samples"],
    }
    out = HERE / "results" / "cesr_u_auto.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
