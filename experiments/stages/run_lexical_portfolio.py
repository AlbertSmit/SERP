#!/usr/bin/env python3
"""ARGON E1: non-neural heterogeneous lexical portfolio, frozen, seven BEIR test sets.

One configuration, one run, no tuning on test. Seven classical scorers over a
single Porter-stemmed index, one fixed RM3-style expansion of BM25, and two fixed
fusions (RRF k=60; NAF = per-query min-max then mean).

Primary system: `portfolio_rrf` (declared before the run). `portfolio_naf` is
reported as the same portfolio under the alternative fixed fusion.

Usage:
  PYTHONHASHSEED=0 ../../.venv/bin/python -u run_lexical_portfolio.py
  PYTHONHASHSEED=0 ../../.venv/bin/python -u run_lexical_portfolio.py --datasets scifact nfcorpus
  PYTHONHASHSEED=0 ../../.venv/bin/python -u run_lexical_portfolio.py --limit 100
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lab"))

import numpy as np

import harness
import lexical_portfolio as lp
from channels import porter

RESULTS = HERE / "results"
ALL_SETS = (
    ("scifact", "test"),
    ("nfcorpus", "test"),
    ("fiqa", "test"),
    ("arguana", None),
    ("scidocs", None),
    ("trec-covid", None),
    ("quora", None),
)
SPLIT = dict(ALL_SETS)
RESAMPLES = 1000
CI_SEED = 1729


def log(message: str) -> None:
    print(message, flush=True)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def verify_bm25(pix: harness.Index, model: lp.Model, queries: dict, n: int = 20) -> dict:
    """Our BM25 impact matrix must reproduce `harness.bm25_fast` on the Porter index."""
    qids = list(queries)[:n]
    worst_score, worst_top, checked = 0.0, 0, 0
    for qid in qids:
        q = queries[qid]
        fast = np.asarray(harness.bm25_fast(pix, q), dtype=np.float64)
        ours = model.score(Counter(q))
        worst_score = max(worst_score, float(np.max(np.abs(fast - ours))))
        top_fast = harness.rank(pix, fast, 100)
        top_ours = [pix.doc_ids[d] for d in lp.topk(ours, 100)]
        if top_fast != top_ours:
            worst_top += 1
        checked += 1
    pix.__dict__.pop("_impacts", None)
    pix.__dict__.pop("_row", None)
    gc.collect()
    ok = worst_score < 1e-8 and worst_top == 0
    log(
        f"    baseline check: max|score diff|={worst_score:.3e}, "
        f"top-100 mismatches={worst_top}/{checked} -> {'MATCH' if ok else 'MISMATCH'}"
    )
    return {"queries": checked, "max_abs_score_diff": worst_score,
            "top100_mismatches": worst_top, "match": ok}


def run_dataset(name: str, split: str | None, limit: int | None, depth: int) -> dict:
    t_start = time.perf_counter()
    log(f"\n=== {name} ({split or 'default test'}) ===")
    t0 = time.perf_counter()
    raw, queries, qrels = harness.load(name, confirm_split=split)
    n_docs = int(raw.N)
    log(f"    loaded index: N={raw.N:,} docs, vocab={len(raw.df):,}, "
        f"avgdl={raw.avgdl:.1f}, {time.perf_counter() - t0:.1f}s")
    if limit:
        queries = dict(list(queries.items())[:limit])
    queries = {qid: porter(toks) for qid, toks in queries.items()}
    log(f"    queries={len(queries):,} (porter-stemmed), "
        f"qrels={sum(len(v) for v in qrels.values()):,}")
    t0 = time.perf_counter()
    pix = lp.porter_index(raw)
    build_ms = (time.perf_counter() - t0) * 1000.0
    position = {d: i for i, d in enumerate(pix.doc_ids)}
    self_ids = {qid: position[qid] for qid in queries if qid in position}
    del raw
    gc.collect()
    log(f"    porter index built in {build_ms / 1000:.1f}s, rss={rss_gb():.2f} GB, "
        f"self-documents ignored={len(self_ids)}")

    lists: dict[str, dict] = {}
    per_query: dict[str, dict] = {}
    timing: dict[str, float] = {}
    skipped: dict[str, str] = {}
    base_check = None

    # BM25 first: it is the baseline, the RM3 seed, and the formula sanity check.
    for model_name in lp.MODELS:
        model = None
        try:
            t0 = time.perf_counter()
            model = lp.Model(model_name, pix)
            log(f"    {model_name:11} impact matrix nnz={model.M.nnz:,} "
                f"({time.perf_counter() - t0:.1f}s)")
            if model_name == "bm25":
                base_check = verify_bm25(pix, model, queries)
            qlists, qmetrics, ms = lp.evaluate(model, queries, qrels, depth=depth,
                                               self_ids=self_ids)
            lists[model_name] = qlists
            per_query[model_name] = qmetrics
            timing[model_name] = ms
            log(f"    {model_name:11} nDCG@10={lp.mean_metric(qmetrics, 'ndcg'):.4f} "
                f"R@100={lp.mean_metric(qmetrics, 'r100'):.4f} {ms:.2f} ms/q")
            if model_name == "bm25":
                t0 = time.perf_counter()
                expanded, rm3_metrics, rm3_ms = lp.evaluate(
                    model, queries, qrels, depth=depth,
                    expand=lambda qid, q, m=model: lp.rm3_weights(
                        pix, m, q, exclude=self_ids.get(qid)
                    ),
                    self_ids=self_ids,
                )
                lists["rm3"] = expanded
                per_query["rm3"] = rm3_metrics
                timing["rm3"] = rm3_ms
                log(f"    {'rm3':11} nDCG@10={lp.mean_metric(rm3_metrics, 'ndcg'):.4f} "
                    f"R@100={lp.mean_metric(rm3_metrics, 'r100'):.4f} {rm3_ms:.2f} ms/q")
        except (MemoryError, ValueError) as exc:  # report, never silently skip
            skipped[model_name] = f"{type(exc).__name__}: {exc}"
            log(f"    {model_name:11} SKIPPED: {type(exc).__name__}: {exc}")
            continue
        finally:
            if model is not None:
                model.free()
            del model
            gc.collect()

    t0 = time.perf_counter()
    for system in lp.SYSTEMS:
        if system in lists:
            continue
        members = tuple(m for m in lp.SYSTEM_MEMBERS[system] if m in lists)
        if len(members) != len(lp.SYSTEM_MEMBERS[system]):
            skipped[system] = "member model unavailable"
            continue
        method = "naf" if system == "portfolio_naf" else "rrf"
        fused = lp.fuse(lists, members, method, lists[members[0]].keys())
        lists[system] = fused
        per_query[system] = lp.metrics_for(pix, fused, qrels)
        log(f"    {system:20} nDCG@10={lp.mean_metric(per_query[system], 'ndcg'):.4f} "
            f"R@100={lp.mean_metric(per_query[system], 'r100'):.4f}")
    timing["_fusion_all_ms"] = (time.perf_counter() - t0) * 1000.0

    systems = {}
    for system in lp.SYSTEMS:
        if system not in per_query:
            continue
        stats = lp.paired_bootstrap(per_query["bm25"], per_query[system],
                                    reps=RESAMPLES, seed=CI_SEED)
        systems[system] = {
            "ndcg10": lp.mean_metric(per_query[system], "ndcg"),
            "r100": lp.mean_metric(per_query[system], "r100"),
            "mrr": lp.mean_metric(per_query[system], "mrr"),
            "delta_ndcg": stats["delta"],
            "ci95": stats["ci95"],
            "wtl": stats["wtl"],
            "ms_per_query": timing.get(system),
            "per_query": {q: {k: round(v, 6) for k, v in m.items()}
                          for q, m in per_query[system].items()},
        }

    print(f"\n{'system':22} {'nDCG@10':>8} {'R@100':>7} {'MRR':>7} {'dNDCG':>8} "
          f"{'95% CI':>19} {'W/T/L':>13}")
    for system, row in systems.items():
        d, lo, hi = row["delta_ndcg"], row["ci95"][0], row["ci95"][1]
        w, t, l = row["wtl"]
        print(f"{system:22} {row['ndcg10']:8.4f} {row['r100']:7.4f} {row['mrr']:7.4f} "
              f"{d:+8.4f} [{lo:+.4f},{hi:+.4f}] {w:5}/{t:5}/{l:5}")

    del pix, lists
    gc.collect()
    return {
        "split": split or "test",
        "n_docs": n_docs,
        "n_queries": len(queries),
        "n_qrels": sum(len(v) for v in qrels.values()),
        "porter_index_build_ms": build_ms,
        "self_documents_ignored": len(self_ids),
        "baseline_check": base_check,
        "skipped": skipped,
        "systems": systems,
        "wall_s": time.perf_counter() - t_start,
        "max_rss_gb": rss_gb(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[n for n, _ in ALL_SETS])
    ap.add_argument("--limit", type=int, default=None, help="queries per dataset (smoke only)")
    ap.add_argument("--depth", type=int, default=lp.DEPTH)
    ap.add_argument("--out", default=str(RESULTS / "lexical_portfolio.json"))
    args = ap.parse_args()

    log("E1 lexical portfolio — frozen configuration")
    log(f"python {platform.python_version()} | numpy {np.__version__} | "
        f"host {platform.machine()} | data root {harness.DATA}")
    log(f"constants: k1={lp.K1} b={lp.B} delta={lp.DELTA} pl2_c={lp.PL2_C} dph_c={lp.DPH_C} "
        f"mu={lp.MU} hiem_lambda={lp.HIEM_LAMBDA} rm3=(lambda={lp.RM3_LAMBDA},"
        f"terms={lp.RM3_TERMS},docs={lp.RM3_DOCS}) rrf_k={lp.RRF_K} depth={args.depth}")

    report = {
        "experiment": "lexical_portfolio",
        "frozen": {
            "k1": lp.K1, "b": lp.B, "delta": lp.DELTA, "pl2_c": lp.PL2_C,
            "dph_c": lp.DPH_C, "mu": lp.MU, "hiemstra_lambda": lp.HIEM_LAMBDA,
            "rm3_lambda": lp.RM3_LAMBDA, "rm3_terms": lp.RM3_TERMS,
            "rm3_docs": lp.RM3_DOCS, "rrf_k": lp.RRF_K, "depth": args.depth,
            "resamples": RESAMPLES, "ci_seed": CI_SEED,
            "primary_portfolio": lp.PRIMARY_PORTFOLIO,
        },
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "platform": platform.platform(),
            "data_root": str(harness.DATA),
        },
        "datasets": {},
    }
    t_all = time.perf_counter()
    for name in args.datasets:
        report["datasets"][name] = run_dataset(name, SPLIT[name], args.limit, args.depth)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))

    systems = [s for s in lp.SYSTEMS if all(s in d["systems"] for d in report["datasets"].values())]
    macro = {}
    for system in systems:
        macro[system] = {
            metric: float(np.mean([report["datasets"][n]["systems"][system][metric]
                                   for n in report["datasets"]]))
            for metric in ("ndcg10", "r100", "mrr", "delta_ndcg")
        }
    report["macro"] = macro
    full = set(report["datasets"]) == {n for n, _ in ALL_SETS}
    if full and lp.PRIMARY_PORTFOLIO in systems:
        report["gate"] = lp.gate(report["datasets"], lp.PRIMARY_PORTFOLIO)
    else:
        report["gate"] = {"skipped": "partial run: gate requires all seven test sets"}
    report["runtime_s"] = time.perf_counter() - t_all
    report["max_rss_gb"] = rss_gb()
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\n{'macro-average':22} {'nDCG@10':>8} {'R@100':>7} {'MRR':>7} {'dNDCG':>8}")
    for system in systems:
        row = macro[system]
        print(f"{system:22} {row['ndcg10']:8.4f} {row['r100']:7.4f} {row['mrr']:7.4f} "
              f"{row['delta_ndcg']:+8.4f}")
    g = report["gate"]
    if "pass" not in g:
        print(f"\ngate: {g['skipped']}")
        print(f"\nwrote {args.out} in {report['runtime_s'] / 60:.1f} min, "
              f"max rss {report['max_rss_gb']:.2f} GB")
        return
    print(f"\ngate ({g['primary_system']}): {'PASS' if g['pass'] else 'FAIL'}")
    print(f"  macro nDCG {g['macro_ndcg']:.4f} vs bm25 {g['macro_bm25_ndcg']:.4f} "
          f"(delta {g['macro_delta']:+.4f})")
    print(f"  positive datasets {len(g['positive_datasets'])}/{len(report['datasets'])}: "
          f"{', '.join(g['positive_datasets'])}")
    print(f"  failing datasets (delta < {g['margin']}, CI upper < {g['margin']}): "
          f"{g['failing_datasets'] or 'none'}")
    print(f"  checks: {g['checks']}")
    print(f"\nwrote {args.out} in {report['runtime_s'] / 60:.1f} min, "
          f"max rss {report['max_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()
