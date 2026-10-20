#!/usr/bin/env python3
"""Code-graph EBR: does entity-bridge retrieval transfer from prose to code/memory?

Builds a graph over the SERP repo's own JavaScript (src/, scripts/, test/), turns
exact structural relations (CALLS, IMPORTS, EXPORTS, DEFINED_IN, CONTAINS) into a
deterministic evidence-chain query set, and measures whether graph expansion beats
Porter BM25 on those chains.

The qrels are derived from the graph. This is a capability test, not a
human-labelled relevance benchmark: it asks whether retrieval can *follow* exact
structural relations, not whether the graph is semantically right.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE / "lab"))

import numpy as np

import code_graph as cg
import fusion
import harness

SYSTEMS = ("porter", "entity", "graph", "rrf_porter_graph", "rrf_porter_entity",
           "rrf_porter_entity_graph")
METRICS = ("both@5", "both@10", "ndcg@10", "recall@10", "mrr@10")
TYPE_ORDER = ("same_file", "import_call", "two_hop", "export_import", "calls_two")


def score_systems(ix, graph, position, query):
    tokens = cg.porter(cg.code_tokens(query["text"]))
    bm25 = harness.bm25_fast(ix, tokens)
    entity = cg.entity_scores(graph, position, query["text"])
    seeds = [ix.doc_ids[i] for i in fusion.order(bm25, cg.SEED_CHUNKS)]
    graph_channel = cg.graph_scores(graph, position, seeds)
    return {
        "porter": bm25,
        "entity": entity,
        "graph": graph_channel,
        "rrf_porter_graph": fusion.rrf(ix, [bm25, graph_channel]),
        "rrf_porter_entity": fusion.rrf(ix, [bm25, entity]),
        "rrf_porter_entity_graph": fusion.rrf(ix, [bm25, entity, graph_channel]),
    }


def run(graph, queries, ix):
    position = {doc: i for i, doc in enumerate(ix.doc_ids)}
    rows = {label: [] for label in SYSTEMS}
    for query in queries:
        systems = score_systems(ix, graph, position, query)
        for label, scores in systems.items():
            ranked = harness.rank(ix, scores, 1000)
            row = cg.metrics(ranked, query["qrels"])
            row["answer@10"] = 1.0 if query["answer"] in ranked[:10] else 0.0
            row["type"] = query["type"]
            row["qid"] = query["qid"]
            rows[label].append(row)
    return rows


def run_noise(graph, queries, ix, drops=(0.0, 0.2, 0.4, 0.6)):
    """Same qrels, progressively damaged retrieval graph."""
    position = {doc: i for i, doc in enumerate(ix.doc_ids)}
    out = {}
    for drop in drops:
        noisy = graph if drop == 0 else cg.perturb(graph, drop)
        rows = {label: [] for label in ("porter", "graph", "rrf_porter_entity_graph")}
        for query in queries:
            systems = score_systems(ix, noisy, position, query)
            for label in rows:
                ranked = harness.rank(ix, systems[label], 1000)
                row = cg.metrics(ranked, query["qrels"])
                row["type"] = query["type"]
                rows[label].append(row)
        out[str(drop)] = {
            label: {
                "both@10": float(np.mean([row["both@10"] for row in values])),
                "both@5": float(np.mean([row["both@5"] for row in values])),
                "ndcg@10": float(np.mean([row["ndcg@10"] for row in values])),
                "by_type_both@10": {
                    kind: float(np.mean([row["both@10"] for row in values if row["type"] == kind]))
                    for kind in TYPE_ORDER
                },
            }
            for label, values in rows.items()
        }
    return out


def summarize(rows):
    out = {}
    for label, values in rows.items():
        out[label] = {metric: float(np.mean([row[metric] for row in values]))
                      for metric in METRICS + ("answer@10",)}
        out[label]["queries"] = len(values)
        out[label]["by_type"] = {}
        for kind in TYPE_ORDER:
            subset = [row for row in values if row["type"] == kind]
            if not subset:
                continue
            by_type = {metric: float(np.mean([row[metric] for row in subset]))
                       for metric in METRICS + ("answer@10",)}
            by_type["queries"] = len(subset)
            by_type["chain_gap"] = int(sum(
                1 for row in subset if row["answer@10"] > 0 and row["both@10"] == 0
            ))
            out[label]["by_type"][kind] = by_type
    return out


def paired_table(rows):
    out = {}
    for label in SYSTEMS[1:]:
        out[label] = {metric: cg.paired(rows["porter"], rows[label], metric) for metric in METRICS}
        out[label]["by_type"] = {}
        for kind in TYPE_ORDER:
            base = [row for row in rows["porter"] if row["type"] == kind]
            cand = [row for row in rows[label] if row["type"] == kind]
            if len(base) < 5:
                continue
            out[label]["by_type"][kind] = {
                metric: cg.paired(base, cand, metric) for metric in METRICS
            }
    # marginal value of the graph channel once the lexical+entity pair is already fused
    out["graph_marginal_over_rrf_porter_entity"] = {
        metric: cg.paired(rows["rrf_porter_entity"], rows["rrf_porter_entity_graph"], metric)
        for metric in METRICS
    }
    out["graph_marginal_over_rrf_porter_entity"]["by_type"] = {
        kind: {
            metric: cg.paired(
                [row for row in rows["rrf_porter_entity"] if row["type"] == kind],
                [row for row in rows["rrf_porter_entity_graph"] if row["type"] == kind],
                metric,
            )
            for metric in METRICS
        }
        for kind in TYPE_ORDER
    }
    out["graph_marginal_excluding_export_import"] = {
        metric: cg.paired(
            [row for row in rows["rrf_porter_entity"] if row["type"] != "export_import"],
            [row for row in rows["rrf_porter_entity_graph"] if row["type"] != "export_import"],
            metric,
        )
        for metric in METRICS
    }
    return out


def fmt(value: float) -> str:
    return f"{value:.3f}"


def print_table(summary):
    print(f"\n{'system':26} " + " ".join(f"{metric:>10}" for metric in METRICS + ("answer@10",)))
    for label in SYSTEMS:
        row = summary[label]
        print(f"{label:26} " + " ".join(f"{row[m]:>10.3f}" for m in METRICS + ("answer@10",)))


def print_paired(paired, label, metrics=("both@10", "both@5", "ndcg@10", "recall@10")):
    print(f"\npaired vs porter: {label}")
    for metric in metrics:
        entry = paired[label][metric]
        low, high = entry["ci95"]
        print(f"  {metric:10} delta {entry['delta']:+.4f}  CI [{low:+.4f}, {high:+.4f}]  "
              f"W/T/L {entry['wtl'][0]}/{entry['wtl'][1]}/{entry['wtl'][2]}")


def print_by_type(summary, metric="both@10"):
    print(f"\n{metric} by query type")
    print(f"{'type':16} {'n':>4} " + " ".join(f"{label:>22}" for label in SYSTEMS))
    for kind in TYPE_ORDER:
        counts = [summary[label]["by_type"].get(kind, {}).get("queries", 0) for label in SYSTEMS]
        cells = " ".join(
            f"{summary[label]['by_type'].get(kind, {}).get(metric, float('nan')):>22.3f}"
            for label in SYSTEMS
        )
        print(f"{kind:16} {max(counts):>4} {cells}")


def leave_one_out(rows, metric="both@10"):
    """Is the overall result one query type? Recompute means excluding each type."""
    out = {}
    for excluded in TYPE_ORDER:
        out[excluded] = {
            label: float(np.mean([row[metric] for row in values if row["type"] != excluded]))
            for label, values in rows.items()
        }
    return out


def main():
    graph = cg.build_graph(ROOT)
    ix = cg.build_index(graph)
    queries = cg.generate_queries(graph, per_type=50, seed=cg.SEED)
    stats = graph.stats()
    chunks_without_calls = sum(1 for token_set in graph.calls.values() if not token_set)
    stats["chunks_without_calls"] = chunks_without_calls
    stats["seed_chunks"] = cg.SEED_CHUNKS
    stats["hub_df_cap"] = max(2, cg.HUB_DF_FRAC * len(graph.chunks))

    print("=" * 100)
    print("CODE GRAPH (src/, scripts/, test/ JavaScript, regex parse)")
    print("=" * 100)
    for key, value in stats.items():
        print(f"  {key:32} {value}")

    print(f"\nchunk kinds: {dict(Counter(chunk['kind'] for chunk in graph.chunks))}")
    print(f"chunks per file: {dict(Counter(chunk['file'] for chunk in graph.chunks))}")
    print(f"\nqueries: {len(queries)}  {dict(Counter(q['type'] for q in queries))}")
    print(f"qrel size distribution: {dict(sorted(Counter(len(q['qrels']) for q in queries).items()))}")

    rows = run(graph, queries, ix)
    summary = summarize(rows)
    paired = paired_table(rows)

    print_table(summary)
    print_by_type(summary, "both@10")
    print_by_type(summary, "both@5")
    for label in SYSTEMS[1:]:
        print_paired(paired, label)
    print_paired(paired, "graph_marginal_over_rrf_porter_entity")
    print_paired(paired, "graph_marginal_excluding_export_import")

    noise = run_noise(graph, queries, ix)
    print("\nedge noise (qrels clean, retrieval graph damaged)")
    print(f"{'drop':>6} " + " ".join(f"{label:>24}" for label in
                                     ("porter", "graph", "rrf_porter_entity_graph")))
    for drop, results in noise.items():
        cells = " ".join(
            f"{results[label]['both@10']:>10.3f} both@10 {results[label]['ndcg@10']:>6.3f}"
            for label in ("porter", "graph", "rrf_porter_entity_graph")
        )
        print(f"{drop:>6} {cells}")
    for drop, results in noise.items():
        row = " ".join(f"{results[label]['by_type_both@10']['two_hop']:>8.3f}"
                       for label in ("porter", "graph", "rrf_porter_entity_graph"))
        print(f"  two_hop both@10 @ drop {drop}: {row}")

    loo = leave_one_out(rows)
    print("\nleave-one-type-out, both@10")
    print(f"{'excluded':16} " + " ".join(f"{label:>24}" for label in
                                         ("porter", "rrf_porter_entity", "rrf_porter_entity_graph")))
    for excluded, values in loo.items():
        print(f"{excluded:16} " + " ".join(f"{values[label]:>24.3f}" for label in
                                           ("porter", "rrf_porter_entity", "rrf_porter_entity_graph")))

    print("\nsample queries")
    for kind in TYPE_ORDER:
        sample = next(q for q in queries if q["type"] == kind)
        print(f"  [{kind}] {sample['text']}")
        print(f"      qrels={sample['qrels']}")

    payload = {
        "meta": {
            "repo": str(ROOT),
            "scope": "src/, scripts/, test/ *.js",
            "seed": cg.SEED,
            "rrf_k": cg.RRF_K,
            "seed_chunks": cg.SEED_CHUNKS,
            "hop2_weight": cg.HOP2_WEIGHT,
            "hub_df_frac": cg.HUB_DF_FRAC,
            "systems": list(SYSTEMS),
            "metrics": list(METRICS),
            "claim": (
                "Synthetic capability test. qrels are derived from the same graph the "
                "retrieval channels traverse, so this measures whether retrieval can follow "
                "exact structural relations -- not whether the graph is semantically correct, "
                "and not human-judged relevance."
            ),
        },
        "graph": stats,
        "chunk_kinds": dict(Counter(chunk["kind"] for chunk in graph.chunks)),
        "chunks_per_file": dict(Counter(chunk["file"] for chunk in graph.chunks)),
        "queries": {
            "total": len(queries),
            "by_type": dict(Counter(q["type"] for q in queries)),
            "qrel_sizes": {str(k): v for k, v in sorted(Counter(len(q["qrels"]) for q in queries).items())},
            "samples": {
                kind: next(q["text"] for q in queries if q["type"] == kind) for kind in TYPE_ORDER
            },
        },
        "systems": summary,
        "paired_vs_porter": paired,
        "edge_noise": noise,
        "leave_one_out_both@10": loo,
    }
    out = HERE / "results" / "code_graph.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
