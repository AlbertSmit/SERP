#!/usr/bin/env python3
"""Real, PR-derived code retrieval benchmark for the code-graph / CESR direction.

Queries are merged pull request descriptions from pmndrs/zustand; qrels are the code
files those pull requests changed. Unlike CODE_GRAPH.md nothing here is generated
from the graph, so a graph win cannot be an artefact of the labels.

    bm25_chunk        Porter BM25 over chunks, files by first appearance
    bm25_file         Porter BM25, chunk scores aggregated per file by max
    bm25_file_sum     same with sum aggregation (diagnostic)
    entity_file       exact-identifier channel, max per file (diagnostic)
    graph_file        graph expansion channel, max per file (diagnostic)
    graph_rrf         RRF(BM25, entity, graph), files by first appearance
    cesr              constraints from the PR text, greedy coverage, BM25 backfill
    cesr_plus_graph   same, over the union of the three candidate lists

Two query variants are scored. `full` is the pre-registered one. `no_paths` strips
file names and paths out of the PR text and is reported as a post-hoc robustness
check: path mentions are the single biggest reason a lexical baseline saturates on
PR-derived qrels, and removing them makes the query "where does this change belong"
instead of "which file does this text name". See experiments/stages/REAL_REPO.md.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE / "lab"))

import numpy as np

import real_repo as rr

REPO = "pmndrs/zustand"
CHECKOUT = HERE / "repos" / "zustand"
PR_CACHE = HERE / "data" / "zustand_prs.json"
OUT = HERE / "results" / "real_repo.json"

PATH_TOKEN = re.compile(r"\b[\w./-]*\.(?:[cm]?[jt]sx?)\b")
VARIANTS = {"full": lambda text: text,
            "no_paths": lambda text: PATH_TOKEN.sub(" ", text)}


def era_of(merged: str) -> str:
    return "2025-2026" if merged >= "2025-01-01" else "2023-2024"


def random_baseline(queries: list[dict], corpus) -> dict:
    """Deterministic shuffled file ranking: the floor any ranking must beat."""
    rows = []
    for query in queries:
        rng = np.random.default_rng(rr.SEED + int(query["qid"]))
        ranked = [corpus.files[i] for i in rng.permutation(len(corpus.files))]
        rows.append({**rr.file_metrics(ranked, query["relevant"]), "qid": query["qid"]})
    return rr.summarize(rows)


def evaluate(queries: list[dict], corpus, transform) -> tuple[dict, dict]:
    rows: dict[str, list[dict]] = {system: [] for system in rr.SYSTEMS}
    diagnostics: dict[str, list[dict]] = {
        "ceiling": [], "constraints": [], "provenance": [], "timing": [],
        "analysis": []}

    for query in queries:
        text = transform(query["text"])
        result, ms = rr.time_run(rr.run_query, corpus, text)
        relevant = query["relevant"]

        for system, output in result["systems"].items():
            row = {
                **rr.file_metrics(output["files"], relevant),
                "qid": query["qid"],
                "relevant": relevant,
                "top10": output["files"][:10],
                "groups": query["groups"],
            }
            if output["chunks"]:
                row.update(rr.chunk_recall(output["chunks"], corpus, relevant))
            rows[system].append(row)

        constraints = result["constraints"]
        named = set(constraints.file_hits)
        symbol_files = {corpus.file_of[chunk] for symbol in constraints.symbol_hits
                        for chunk in corpus.graph.defined.get(symbol, ())}
        diagnostics["constraints"].append({
            "file_precision": (len(named & set(relevant)) / len(named)) if named else None,
            "symbol_file_precision": (
                len(symbol_files & set(relevant)) / len(symbol_files)) if symbol_files else None,
            "qid": query["qid"],
            "labels": len(constraints.labels),
            "symbols": len(constraints.symbol_hits),
            "files": len(constraints.file_hits),
            "cues": len(constraints.cues),
            "mandatory": len(constraints.mandatory),
            "selected": len(result["systems"]["cesr"]["chunks"]),
        })
        diagnostics["ceiling"].append({
            "qid": query["qid"], **rr.candidate_ceiling(result, relevant, corpus)})
        chosen = result["systems"]["cesr"]["chunks"]
        diagnostics["provenance"].append({
            "qid": query["qid"],
            "porter_only": sum(
                1 for doc in chosen
                if result["pool_provenance"].get(doc) == ["porter"]) / max(len(chosen), 1),
            "graph": sum(
                1 for doc in chosen
                if "graph" in result["pool_provenance"].get(doc, [])) / max(len(chosen), 1),
            "outside_porter_top100": sum(
                1 for doc in chosen
                if doc not in result["systems"]["bm25_chunk"]["chunks"][:100]
            ) / max(len(chosen), 1),
            "pool_size": len(result["pool"]),
        })
        diagnostics["timing"].append(ms)
        diagnostics["analysis"].append({
            "qid": query["qid"],
            "relevant": relevant,
            "groups": query["groups"],
            "bm25_files": corpus.aggregate_files(
                result["channels"]["porter"], depth=len(corpus.files)),
            "pool_files": sorted({corpus.file_of[doc] for doc in result["pool"]}),
            "bm25_recall10": rows["bm25_file"][-1]["recall@10"],
            "cesr_recall10": rows["cesr"][-1]["recall@10"],
            "graph_recall10": rows["graph_rrf"][-1]["recall@10"],
        })
    return rows, diagnostics


def miss_decomposition(analysis: list[dict]) -> dict:
    """Where the lexical baseline's remaining errors live.

    A missed file that is outside the candidate pool cannot be recovered by any
    reranker; one that is inside the pool is addressable headroom.
    """
    missed = []
    for row in analysis:
        rank = {file: i + 1 for i, file in enumerate(row["bm25_files"])}
        pool = set(row["pool_files"])
        for file in row["relevant"]:
            if rank.get(file, 10 ** 6) > 10:
                missed.append({
                    "file": file, "in_pool": file in pool,
                    "group": row["groups"][file], "rank": rank.get(file),
                })
    total = sum(len(row["relevant"]) for row in analysis)
    return {
        "relevant_file_instances": total,
        "missed_at_10": len(missed),
        "missed_share": len(missed) / max(total, 1),
        "missed_in_pool": sum(1 for m in missed if m["in_pool"]),
        "missed_outside_pool": sum(1 for m in missed if not m["in_pool"]),
        "missed_test": sum(1 for m in missed if m["group"] == "test"),
        "missed_never_scored": sum(1 for m in missed if m["rank"] is None),
    }


def changed_queries(analysis: list[dict], key: str) -> list[dict]:
    """Queries whose file recall@10 the system changed, with the title for review."""
    return [{"qid": row["qid"],
             "bm25_file": row["bm25_recall10"],
             key: row[key],
             "relevant": row["relevant"]}
            for row in analysis if row["bm25_recall10"] != row[key]]


def slices(rows: dict[str, list[dict]], queries: list[dict]) -> dict:
    by_qid = {query["qid"]: query for query in queries}
    out: dict[str, dict] = {}

    def bucket(name: str, predicate) -> None:
        block = {}
        for system, values in rows.items():
            subset = [row for row in values if predicate(by_qid[row["qid"]])]
            if subset:
                block[system] = rr.summarize(subset)
        out[name] = block

    for era in ("2023-2024", "2025-2026"):
        bucket(f"era:{era}", lambda q, era=era: era_of(q["merged"]) == era)
    for size in (1, 2, 3):
        bucket(f"qrel_files:{size}", lambda q, size=size: len(q["relevant"]) == size)
    bucket("qrel_files:4-5", lambda q: len(q["relevant"]) >= 4)
    bucket("touches_test", lambda q: "test" in q["groups"].values())
    bucket("source_only", lambda q: "test" not in q["groups"].values())
    return out


def all_at_10(rows: dict[str, list[dict]], queries: list[dict], cap: int | None) -> dict:
    by_qid = {query["qid"]: query for query in queries}
    out = {}
    for system, values in rows.items():
        subset = [row for row in values
                  if cap is None or len(by_qid[row["qid"]]["relevant"]) <= cap]
        out[system] = float(np.mean([row["all@10"] for row in subset])) if subset else 0.0
    return out


def paired_table(rows: dict[str, list[dict]], base: str = "bm25_file") -> dict:
    return {system: {metric: rr.paired(rows[base], rows[system], metric)
                     for metric in rr.METRICS}
            for system in rr.SYSTEMS if system != base}


def blocked_report(rows, diagnostics, queries, corpus) -> dict:
    """Everything one query variant contributes to the JSON."""
    summary = {system: rr.summarize(values) for system, values in rows.items()}
    capped, uncapped = all_at_10(rows, queries, 3), all_at_10(rows, queries, None)
    paired = paired_table(rows)
    groups = {system: {group: rr.slice_recall(values, group)
                       for group in ("test", "source")}
              for system, values in rows.items()}
    constraints = diagnostics["constraints"]
    provenance = diagnostics["provenance"]
    return {
        "systems": summary,
        "all_at_10": {"capped_at_3_qrel_files": capped, "all": uncapped},
        "paired_vs_bm25_file": paired,
        "slices": slices(rows, queries),
        "file_group_recall": groups,
        "chunk_diagnostics": {
            system: {f"chunk_recall@{k}": float(np.mean(
                [row[f"chunk_recall@{k}"] for row in values
                 if f"chunk_recall@{k}" in row]))
                for k in (10, 100)}
            for system, values in rows.items()
            if any("chunk_recall@10" in row for row in values)
        },
        "ceiling": {
            "files_in_pool": float(np.mean(
                [row["files_in_pool"] for row in diagnostics["ceiling"]])),
            "all_files_in_pool": float(np.mean(
                [row["all_in_pool"] for row in diagnostics["ceiling"]])),
            "pool_depth": rr.POOL_DEPTH,
        },
        "constraints": {
            "queries_with_constraints": float(np.mean(
                [row["labels"] > 0 for row in constraints])),
            "mean_labels": float(np.mean([row["labels"] for row in constraints])),
            "mean_symbols": float(np.mean([row["symbols"] for row in constraints])),
            "mean_file_hits": float(np.mean([row["files"] for row in constraints])),
            "queries_with_symbol": float(np.mean([row["symbols"] > 0 for row in constraints])),
            "queries_with_file": float(np.mean([row["files"] > 0 for row in constraints])),
            "mean_mandatory": float(np.mean([row["mandatory"] for row in constraints])),
            "mean_selected_chunks": float(np.mean([row["selected"] for row in constraints])),
            "file_hit_precision": float(np.nanmean(
                [row["file_precision"] for row in constraints
                 if row["file_precision"] is not None])),
            "symbol_file_precision": float(np.nanmean(
                [row["symbol_file_precision"] for row in constraints
                 if row["symbol_file_precision"] is not None])),
        },
        "provenance": {
            "porter_only": float(np.mean([row["porter_only"] for row in provenance])),
            "graph": float(np.mean([row["graph"] for row in provenance])),
            "outside_porter_top100": float(np.mean(
                [row["outside_porter_top100"] for row in provenance])),
            "pool_size": float(np.mean([row["pool_size"] for row in provenance])),
        },
        "gate": rr.gate(paired),
        "latency_ms_per_query": float(np.mean(diagnostics["timing"])),
        "miss_decomposition": miss_decomposition(diagnostics["analysis"]),
        "queries_changed": {
            "cesr": changed_queries(diagnostics["analysis"], "cesr_recall10"),
            "graph_rrf": changed_queries(diagnostics["analysis"], "graph_recall10"),
        },
    }


def print_table(title: str, block: dict) -> None:
    print(f"\n{title}")
    print(f"{'system':18} " + " ".join(f"{metric:>10}" for metric in rr.METRICS)
          + f" {'all@10':>8} {'chR@10':>8} {'chR@100':>8}")
    for system in rr.SYSTEMS:
        row = block["systems"][system]
        chunk = block["chunk_diagnostics"].get(system, {})
        print(f"{system:18} " + " ".join(f"{row[metric]:>10.3f}" for metric in rr.METRICS)
              + f" {block['all_at_10']['capped_at_3_qrel_files'][system]:>8.3f}"
              + f" {chunk.get('chunk_recall@10', float('nan')):>8.3f}"
              + f" {chunk.get('chunk_recall@100', float('nan')):>8.3f}")


def print_paired(block: dict, systems=("graph_rrf", "cesr", "cesr_plus_graph")) -> None:
    print(f"\npaired vs bm25_file (1,000 resamples, seed {rr.SEED})")
    for system in systems:
        for metric in ("recall@10", "ndcg@10", "recall@5"):
            entry = block["paired_vs_bm25_file"][system][metric]
            low, high = entry["ci95"]
            print(f"  {system:16} {metric:10} {entry['delta']:+.4f} "
                  f"[{low:+.4f}, {high:+.4f}]  W/T/L "
                  f"{entry['wtl'][0]}/{entry['wtl'][1]}/{entry['wtl'][2]}")


def main() -> None:
    started = time.perf_counter()
    t0 = time.perf_counter()
    raw = rr.fetch_pr_list(CHECKOUT, PR_CACHE, REPO)
    corpus = rr.build_corpus(CHECKOUT)
    build_ms = (time.perf_counter() - t0) * 1000
    prs, drops = rr.select_prs(raw, CHECKOUT, set(corpus.files))
    queries = rr.build_queries(prs, corpus)
    (HERE / "data" / "zustand_selected_prs.json").write_text(json.dumps({
        "head": rr.git_head(CHECKOUT),
        "drops": drops,
        "prs": prs,
    }, indent=1))
    head = rr.git_head(CHECKOUT)

    blocks: dict[str, dict] = {}
    for name, transform in VARIANTS.items():
        rows, diagnostics = evaluate(queries, corpus, transform)
        blocks[name] = blocked_report(rows, diagnostics, queries, corpus)
        if name == "full":
            blocks[name]["random_baseline"] = random_baseline(queries, corpus)

    primary = blocks["full"]
    print("=" * 108)
    print(f"REAL PR BENCHMARK  {REPO} @ {head[:10]}  ({len(queries)} merged PRs, "
          f"{len(corpus.files)} files, {len(corpus.ix.doc_ids)} chunks)")
    print("=" * 108)
    print(f"  chunks by kind  {dict(Counter(c['kind'] for c in corpus.graph.chunks))}")
    print(f"  symbols         {len(corpus.graph.defined)}  "
          f"calls {corpus.graph.stats()['call_edges']}  "
          f"imports resolved {corpus.graph.stats()['import_edges_resolved']}/"
          f"{corpus.graph.stats()['import_edges']}")
    print(f"  PRs             {len(prs)}  "
          f"{dict(Counter(era_of(p['mergedAt']) for p in prs))}  "
          f"qrels sizes {dict(Counter(len(q['relevant']) for q in queries))}")
    print(f"  drift vs HEAD   mean {np.mean([p['drift'] for p in prs]):.3f}  "
          f"median {np.median([p['drift'] for p in prs]):.3f}  "
          f"share<=0.2 {np.mean([p['drift'] <= 0.2 for p in prs]):.3f}")
    print(f"  dropped         {drops}")
    print(f"  random floor    " + " ".join(
        f"{metric}={primary['random_baseline'][metric]:.3f}" for metric in rr.METRICS))

    print_table("VARIANT full (pre-registered: PR title + body)", primary)
    print_paired(primary)

    print("\nfile recall@10 by qrel-file group")
    print(f"{'system':18} {'test':>8} {'n':>4} {'source':>8} {'n':>4}")
    for system in rr.SYSTEMS:
        group = primary["file_group_recall"][system]
        print(f"{system:18} {group['test']['recall@10']:>8.3f} {group['test']['queries']:>4} "
              f"{group['source']['recall@10']:>8.3f} {group['source']['queries']:>4}")

    print("\nfile recall@10 by era")
    print(f"{'system':18} " + " ".join(f"{era:>12}" for era in ("2023-2024", "2025-2026")))
    for system in rr.SYSTEMS:
        print(f"{system:18} " + " ".join(
            f"{primary['slices'][f'era:{era}'][system]['recall@10']:>12.3f}"
            for era in ("2023-2024", "2025-2026")))

    print_table("VARIANT no_paths (post-hoc robustness: file names stripped)", blocks["no_paths"])
    print_paired(blocks["no_paths"])

    print("\nGATE (full variant)")
    for key, check in primary["gate"]["checks"].items():
        print(f"  {key:34} {check['delta']:+.4f} "
              f"[{check['ci95'][0]:+.4f}, {check['ci95'][1]:+.4f}]  "
              f"{'PASS' if check['ci_above_zero'] else 'fail'}")
    print(f"  => {'PASSED' if primary['gate']['passed'] else 'FAILED'}")

    miss = primary["miss_decomposition"]
    print(f"\nmiss decomposition (bm25_file @10): {miss['missed_at_10']} of "
          f"{miss['relevant_file_instances']} relevant file instances missed "
          f"({miss['missed_share']:.3f}); in pool {miss['missed_in_pool']}, "
          f"outside pool {miss['missed_outside_pool']}, test {miss['missed_test']}")
    print(f"queries changed by cesr: {len(primary['queries_changed']['cesr'])}, "
          f"by graph_rrf: {len(primary['queries_changed']['graph_rrf'])}")

    samples = []
    for query in (queries[0], queries[len(queries) // 2], queries[-1]):
        result = rr.run_query(corpus, query["text"])
        samples.append({
            "number": query["number"], "title": query["title"],
            "merged": query["merged"], "drift": query["drift"],
            "query": query["text"][:600], "qrels": query["relevant"],
            "constraints": {
                "labels": result["constraints"].labels,
                "symbols": result["constraints"].symbol_hits,
                "files": result["constraints"].file_hits,
                "cues": result["constraints"].cues,
            },
            "top10": {system: output["files"]
                      for system, output in result["systems"].items()},
        })

    payload = {
        "meta": {
            "repo": REPO,
            "head": head,
            "checkout": str(CHECKOUT.relative_to(ROOT)),
            "corpus_dirs": list(rr.SCAN_DIRS),
            "query_source": "merged PR title + body, truncated to 200 whitespace tokens",
            "qrels_source": "code files changed by the merged PR",
            "pr_window": f"merged >= {rr.MIN_MERGED}",
            "pr_limit": rr.PR_LIMIT,
            "seed": rr.SEED,
            "parser": "tolerant line/brace TypeScript scanner (see REAL_REPO.md)",
            "variants": {
                "full": "pre-registered query text",
                "no_paths": ("post-hoc: file names and paths removed from the query, "
                             "to test whether the lexical baseline is saturating on "
                             "path mentions"),
            },
            "constants": {
                "rrf_k": rr.RRF_K, "pool_depth": rr.POOL_DEPTH,
                "seed_chunks": rr.SEED_CHUNKS, "hub_df_frac": rr.HUB_DF_FRAC,
                "cesr_budget": rr.CESR_BUDGET, "mu": rr.MU,
                "cost_exponent": rr.COST_EXPONENT, "bm25_tie": rr.BM25_TIE,
                "small_set": rr.SMALL_SET,
                "max_symbol_constraints": rr.MAX_SYMBOL_CONSTRAINTS,
                "max_file_constraints": rr.MAX_FILE_CONSTRAINTS,
                "resamples": rr.RESAMPLES,
            },
            "claim": (
                "PR-derived, non-circular: neither queries nor labels come from the graph. "
                "Labels are 'files the author changed', which includes tests and refactors "
                "and is therefore noisy in a known direction. Corpus and graph are at HEAD "
                "while PRs merged earlier, so qrels for old PRs describe code that has since "
                "moved; per-PR drift is recorded."
            ),
        },
        "prs": {
            "count": len(prs),
            "numbers": [pr["number"] for pr in prs],
            "by_era": dict(Counter(era_of(pr["mergedAt"]) for pr in prs)),
            "changed_files": dict(Counter(len(pr["qrels_files"]) for pr in prs)),
            "drift": {
                "mean": float(np.mean([pr["drift"] for pr in prs])),
                "median": float(np.median([pr["drift"] for pr in prs])),
                "share_le_0.2": float(np.mean([pr["drift"] <= 0.2 for pr in prs])),
                "by_era": {
                    era: {"prs": len(subset),
                          "mean": float(np.mean([pr["drift"] for pr in subset]))}
                    for era in ("2023-2024", "2025-2026")
                    for subset in [[p for p in prs if era_of(p["mergedAt"]) == era]]
                },
            },
            "dropped": drops,
            "titles": [{"number": pr["number"], "title": pr["title"],
                        "merged": pr["mergedAt"][:10], "files": pr["qrels_files"]}
                       for pr in prs],
        },
        "parser": {
            "files": len(corpus.files),
            "chunks": len(corpus.ix.doc_ids),
            "chunk_kinds": dict(Counter(c["kind"] for c in corpus.graph.chunks)),
            "graph": corpus.graph.stats(),
            "chunks_per_file": dict(Counter(c["file"] for c in corpus.graph.chunks)),
            "mean_dl": float(np.mean(list(corpus.dl.values()))),
            "max_dl": int(max(corpus.dl.values())),
        },
        "queries": {
            "count": len(queries),
            "by_era": dict(Counter(era_of(q["merged"]) for q in queries)),
            "qrel_sizes": {str(k): v for k, v in
                           sorted(Counter(len(q["relevant"]) for q in queries).items())},
            "with_test_qrel": float(np.mean(["test" in q["groups"].values() for q in queries])),
        },
        "full": primary,
        "no_paths": blocks["no_paths"],
        "runtime": {
            "corpus_build_ms": build_ms,
            "total_s": float(time.perf_counter() - started),
        },
        "samples": samples,
    }
    payload["systems"] = primary["systems"]
    payload["paired_vs_bm25_file"] = primary["paired_vs_bm25_file"]
    payload["gate"] = primary["gate"]
    payload["ceiling"] = primary["ceiling"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT}")
    print(f"total {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
