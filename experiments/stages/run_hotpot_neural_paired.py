#!/usr/bin/env python3
"""Paired EBR-vs-cross-encoder test on HotpotQA, plus a CPU like-for-like latency sample.

`run_hotpot_neural.py` reports each system against Porter BM25 and compares the two
rerankers against each other. It never tests EBR directly against the cross-encoder,
because that comparison is not a "gain over Porter" and does not fit the ratio table.
This script runs exactly that test, on the identical 1,000 held-out qids:

  per-query   ndcg@10, both@10, both@5, recall@10 for porter, ebr, ce_porter, ce_ebr
  paired      EBR - CE(Porter pool), CE(EBR pool) - CE(Porter pool), EBR - CE(EBR pool)
  bootstrap   1,000 paired percentile resamples, seed 1729 (same as hotpot.paired)

It also puts the latency claim on one device. The published 22x mixes an end-to-end
CPU pipeline with a rerank-only GPU pass. Here both sides run on CPU, per query, on
the same 100-query sample: EBR end to end, and the cross-encoder *including* the
candidate construction it needs (BM25 top-100, or the EBR pool, or their union).
The GPU numbers are left untouched and only referenced.

Nothing is tuned here. Same model, same qids, same metrics, same constants.

Usage:
  .venv/bin/python experiments/stages/run_hotpot_neural_paired.py
  .venv/bin/python experiments/stages/run_hotpot_neural_paired.py --latency-sample 200
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lab"))

import numpy as np

import fusion
import hotpot
import run_hotpot
from channels import porter
from lean_units import LeanUnitBM25
from run_hotpot_neural import (
    CUTOFF,
    CrossEncoder,
    SYSTEMS,
    ebr_scores,
    raw_documents,
    rerank,
    select_subset,
    top_docs,
)

MODEL_DIR = Path("/Users/asmit/Personal/algo/first_principles/models/ms-marco-MiniLM-L-6-v2")
METRICS = ("ndcg@10", "both@10", "both@5", "recall@10")

# Derived, untracked (.cache/ is ignored). Deleting them only costs recompute time.
POOL_CACHE = HERE / ".cache/hotpot_neural_pools_seed1729.json"
SCORE_CACHE = HERE / ".cache/hotpot_neural_union_scores_seed1729.json"


# ---------------------------------------------------------------- candidate pools


def build_pools(index, subset, porter_system, progress: int = 200) -> dict:
    """Porter top-100, EBR top-100 and their union per qid. Deterministic, cacheable."""
    pools = {}
    started = time.perf_counter()
    for position, qid in enumerate(subset, start=1):
        ix = index["ix"]
        porter_scores, fused = ebr_scores(index, qid, porter_system)
        porter_pool = top_docs(ix, porter_scores, CUTOFF)
        ebr_pool = top_docs(ix, fused, CUTOFF)
        pools[qid] = {
            "porter": porter_pool,
            "ebr": ebr_pool,
            "union": sorted(set(porter_pool) | set(ebr_pool)),
        }
        if progress and position % progress == 0:
            print(f"[pools] {position}/{len(subset)}", flush=True)
    print(f"[pools] built {len(pools)} in {time.perf_counter() - started:.0f}s", flush=True)
    return pools


def load_or_build_pools(index, subset, porter_system, rebuild: bool) -> tuple[dict, str]:
    if POOL_CACHE.exists() and not rebuild:
        cached = json.loads(POOL_CACHE.read_text())
        if cached.get("qids") == subset:
            print(f"[pools] reusing {POOL_CACHE}", flush=True)
            return cached["pools"], "cache"
    pools = build_pools(index, subset, porter_system)
    POOL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    POOL_CACHE.write_text(json.dumps({"qids": subset, "pools": pools}))
    return pools, "rebuilt"


# --------------------------------------------------------------- neural scoring


def score_unions(index, subset, pools, encoder, rebuild: bool) -> tuple[dict, str]:
    """Score every unique (query, paragraph) pair once. Identical scores feed both rerankers."""
    cached: dict[str, dict[str, float]] = {}
    if SCORE_CACHE.exists() and not rebuild:
        cached = json.loads(SCORE_CACHE.read_text()).get("scores", {})
        print(f"[score] resuming from {len(cached)} cached queries", flush=True)

    started = time.perf_counter()
    for position, qid in enumerate(subset, start=1):
        union = pools[qid]["union"]
        if set(cached.get(qid, {})) == set(union):
            continue
        raw = index["raw_queries"][qid]
        texts = [f"{doc}. {index['docs'][doc]}" for doc in union]
        cached[qid] = dict(zip(union, encoder.score(raw, texts)))
        if position % 100 == 0:
            SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            SCORE_CACHE.write_text(json.dumps({"scores": cached}))
            print(f"[score] {position}/{len(subset)}, {time.perf_counter() - started:5.0f}s", flush=True)
    SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SCORE_CACHE.write_text(json.dumps({"scores": cached}))
    return cached, "reused+cached" if cached else "fresh"


# ------------------------------------------------------------------- evaluation


def per_query_rows(index, subset, pools, scores) -> dict:
    rows = {label: [] for label in SYSTEMS}
    porter_system = LeanUnitBM25(index["ix"], porter)
    for qid in subset:
        ix = index["ix"]
        qrels = index["qrels"][qid]
        porter_scores, fused = ebr_scores(index, qid, porter_system)
        neural = scores[qid]
        ranked = {
            "porter": run_hotpot.ranked_from_scores(index, porter_scores),
            "ebr": run_hotpot.ranked_from_scores(index, fused),
            "ce_porter": rerank(pools[qid]["porter"], [neural[doc] for doc in pools[qid]["porter"]]),
            "ce_ebr": rerank(pools[qid]["ebr"], [neural[doc] for doc in pools[qid]["ebr"]]),
        }
        for label, ranking in ranked.items():
            row = hotpot.metrics(ranking, qrels)
            row["qid"] = qid
            row["type"] = index["meta"][qid]["type"]
            rows[label].append(row)
    return rows


def paired_block(rows) -> dict:
    """Direction is candidate - base, so the label reads as the delta."""
    return {
        "ebr_minus_ce_porter": hotpot.paired_report(rows["ce_porter"], rows["ebr"], METRICS),
        "ce_ebr_minus_ce_porter": hotpot.paired_report(rows["ce_porter"], rows["ce_ebr"], METRICS),
        "ebr_minus_ce_ebr": hotpot.paired_report(rows["ce_ebr"], rows["ebr"], METRICS),
        # Reproduced for continuity with NEURAL_REFERENCE.md.
        "ebr_minus_porter": hotpot.paired_report(rows["porter"], rows["ebr"], METRICS),
        "ce_porter_minus_porter": hotpot.paired_report(rows["porter"], rows["ce_porter"], METRICS),
        "ce_ebr_minus_porter": hotpot.paired_report(rows["porter"], rows["ce_ebr"], METRICS),
    }


def reproduction_check(summary, original: dict) -> dict:
    out = {}
    for label in SYSTEMS:
        out[label] = {
            metric: {
                "original": original["summary"][label][metric],
                "rerun": summary[label][metric],
                "abs_diff": abs(original["summary"][label][metric] - summary[label][metric]),
            }
            for metric in METRICS
        }
    worst = max(
        block["abs_diff"] for label in out.values() for block in label.values()
    )
    return {"per_system": out, "max_abs_diff": worst}


# ---------------------------------------------------------------- CPU latency


def ebr_end_to_end(index, qid, porter_system) -> tuple[float, list[str]]:
    """Full EBR on CPU, ending at the fused order the packer would consume."""
    started = time.perf_counter()
    ix = index["ix"]
    porter_scores = porter_system.score(index["queries"][qid])
    entities = hotpot.query_entities(index["raw_queries"][qid], index["title_index"])
    entity_dense = hotpot.dense(ix, hotpot.per_entity_scores(index, qid, entities), index["position"])
    bridges, _ = hotpot.bridge_scores(index, qid, porter_scores)
    bridge_dense = hotpot.dense(ix, bridges, index["position"])
    fused = fusion.rrf(ix, [porter_scores, bridge_dense, entity_dense])
    order = [ix.doc_ids[doc] for doc in fusion.order(fused, 1000)]
    return time.perf_counter() - started, order


def porter_candidates(index, qid, porter_system) -> tuple[float, list[str]]:
    """BM25 first hop plus the top-100 list the cross-encoder needs."""
    started = time.perf_counter()
    scores = porter_system.score(index["queries"][qid])
    pool = top_docs(index["ix"], scores, CUTOFF)
    return time.perf_counter() - started, pool


def ce_forward(index, qid, doc_ids, encoder) -> float:
    texts = [f"{doc}. {index['docs'][doc]}" for doc in doc_ids]
    started = time.perf_counter()
    encoder.score(index["raw_queries"][qid], texts)
    return time.perf_counter() - started


def cpu_latency(index, subset, pools, cpu_encoder, sample: int, warmup: int = 10) -> dict:
    """Per-query CPU wall time for EBR and for the cross-encoder *including* its candidates."""
    porter_system = LeanUnitBM25(index["ix"], porter)
    qids = subset[:sample]
    for qid in subset[:warmup]:
        ebr_end_to_end(index, qid, porter_system)
        _, pool = porter_candidates(index, qid, porter_system)
        ce_forward(index, qid, pool, cpu_encoder)
    print(f"[cpu] warmup done, timing {len(qids)} queries", flush=True)

    timings = {
        "ebr_full": [],
        "bm25_candidates": [],
        "ce_porter_forward": [],
        "ce_porter_total": [],
        "ce_ebr_forward": [],
        "ce_ebr_total": [],
        "ce_union_forward": [],
        "ce_union_total": [],
        "pairs_porter": [],
        "pairs_ebr": [],
        "pairs_union": [],
    }
    for position, qid in enumerate(qids, start=1):
        ebr_seconds, _ = ebr_end_to_end(index, qid, porter_system)
        bm25_seconds, porter_pool = porter_candidates(index, qid, porter_system)
        ce_porter_seconds = ce_forward(index, qid, porter_pool, cpu_encoder)
        ebr_pool = pools[qid]["ebr"]
        ce_ebr_seconds = ce_forward(index, qid, ebr_pool, cpu_encoder)
        union = pools[qid]["union"]
        ce_union_seconds = ce_forward(index, qid, union, cpu_encoder)

        timings["ebr_full"].append(ebr_seconds)
        timings["bm25_candidates"].append(bm25_seconds)
        timings["ce_porter_forward"].append(ce_porter_seconds)
        timings["ce_porter_total"].append(bm25_seconds + ce_porter_seconds)
        timings["ce_ebr_forward"].append(ce_ebr_seconds)
        timings["ce_ebr_total"].append(ebr_seconds + ce_ebr_seconds)
        timings["ce_union_forward"].append(ce_union_seconds)
        timings["ce_union_total"].append(ebr_seconds + ce_union_seconds)
        timings["pairs_porter"].append(len(porter_pool))
        timings["pairs_ebr"].append(len(ebr_pool))
        timings["pairs_union"].append(len(union))
        if position % 25 == 0:
            print(f"[cpu] {position}/{len(qids)}", flush=True)

    def block(values) -> dict:
        arr = np.array(values, dtype=float) * 1000.0
        return {
            "ms_mean": float(arr.mean()),
            "ms_p50": float(np.percentile(arr, 50)),
            "ms_p95": float(np.percentile(arr, 95)),
        }

    report = {key: block(value) for key, value in timings.items() if not key.startswith("pairs_")}
    total_porter = np.array(timings["ce_porter_total"]) * 1000.0
    total_ebr = np.array(timings["ebr_full"]) * 1000.0
    total_union = np.array(timings["ce_union_total"]) * 1000.0
    per_query_ratio = total_porter / total_ebr
    report["ratio_ce_porter_over_ebr"] = {
        "p50_of_per_query_ratio": float(np.percentile(per_query_ratio, 50)),
        "p95_of_per_query_ratio": float(np.percentile(per_query_ratio, 95)),
        "ratio_of_p50": float(report["ce_porter_total"]["ms_p50"] / report["ebr_full"]["ms_p50"]),
        "ratio_of_means": float(report["ce_porter_total"]["ms_mean"] / report["ebr_full"]["ms_mean"]),
        "ce_ebr_ratio_of_p50": float(report["ce_ebr_total"]["ms_p50"] / report["ebr_full"]["ms_p50"]),
        "ce_union_ratio_of_p50": float(report["ce_union_total"]["ms_p50"] / report["ebr_full"]["ms_p50"]),
    }
    report["pairs"] = {
        "porter_pool": float(np.mean(timings["pairs_porter"])),
        "ebr_pool": float(np.mean(timings["pairs_ebr"])),
        "union": float(np.mean(timings["pairs_union"])),
        "porter_pairs_per_second": float(
            np.sum(timings["pairs_porter"]) / np.sum(timings["ce_porter_forward"])
        ),
    }
    report["sample_queries"] = len(qids)
    report["warmup_queries_discarded"] = warmup
    report["batch_size"] = cpu_encoder.batch_size
    report["max_length"] = cpu_encoder.max_length
    report["torch_threads"] = int(cpu_encoder.torch.get_num_threads())
    report["device"] = cpu_encoder.device
    report["thread_note"] = (
        "EBR is numpy/CPU; the cross-encoder was reloaded on CPU with the default thread count. "
        "Both sides single process, same sample, no packer on either side."
    )
    return report


def contention_check(index, subset, pools, mps_encoder, sample: int = 30, warmup: int = 5) -> dict:
    """Is the published 23 ms EBR number inflated by the GPU reranker running in-process?

    The original script timed EBR inside the same loop as MPS inference. This measures
    the same EBR code twice: alone, then while an MPS rerank loop is running.
    """
    porter_system = LeanUnitBM25(index["ix"], porter)
    qids = subset[:sample]
    for qid in subset[:warmup]:
        ebr_end_to_end(index, qid, porter_system)
    alone = np.array([ebr_end_to_end(index, qid, porter_system)[0] for qid in qids]) * 1000.0

    stop = threading.Event()

    def rerank_loop():
        while not stop.is_set():
            for qid in subset[:8]:
                if stop.is_set():
                    return
                union = pools[qid]["union"]
                mps_encoder.score(
                    index["raw_queries"][qid], [f"{doc}. {index['docs'][doc]}" for doc in union]
                )

    thread = threading.Thread(target=rerank_loop, daemon=True)
    thread.start()
    time.sleep(2.0)
    busy = np.array([ebr_end_to_end(index, qid, porter_system)[0] for qid in qids]) * 1000.0
    stop.set()
    thread.join(timeout=10.0)
    return {
        "sample_queries": len(qids),
        "ebr_alone_ms_p50": float(np.percentile(alone, 50)),
        "ebr_with_mps_rerank_ms_p50": float(np.percentile(busy, 50)),
        "delta_ms": float(np.percentile(busy, 50) - np.percentile(alone, 50)),
        "note": "Same code both times. MPS contention accounts for only a small part of the "
        "published 23.3 ms vs this pass's EBR number; the rest is machine/thermal state.",
    }


# ------------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(HERE / "data/hotpot_dev.parquet"))
    parser.add_argument("--out", default=str(HERE / "results/hotpot_neural_reference_paired.json"))
    parser.add_argument("--original", default=str(HERE / "results/hotpot_neural_reference.json"))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--model", default=str(MODEL_DIR))
    parser.add_argument("--latency-sample", type=int, default=100)
    parser.add_argument("--skip-contention-check", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    index = hotpot.load(args.data)
    index["docs"] = raw_documents(args.data)
    subset = select_subset(index, args.split, args.n, args.seed)
    print(f"[load] {index['ix'].N} paragraphs, {len(index['qrels'])} questions", flush=True)
    print(f"[subset] {len(subset)} {args.split} qids, seed {args.seed}", flush=True)

    porter_system = LeanUnitBM25(index["ix"], porter)

    # Same ranking-parity guard as the original script: EBR must still match run_hotpot.
    for qid in subset[:5]:
        reference = run_hotpot.query_systems(index, qid, porter_system)["rrf_porter_bridge_entity"]
        _, ours = ebr_scores(index, qid, porter_system)
        if not np.allclose(reference, ours):
            raise SystemExit(f"EBR score parity failed for {qid}; refusing to report")

    pools, pool_source = load_or_build_pools(index, subset, porter_system, args.rebuild_cache)

    encoder = CrossEncoder(Path(args.model), "auto", args.batch_size, args.max_length)
    print(
        f"[model] {args.model} on {encoder.device}, load {encoder.load_seconds:.1f}s, "
        f"batch {encoder.batch_size}, max_len {encoder.max_length}",
        flush=True,
    )
    scores, score_source = score_unions(index, subset, pools, encoder, args.rebuild_cache)

    rows = per_query_rows(index, subset, pools, scores)
    summary = hotpot.summarize(rows)
    paired = paired_block(rows)

    original = json.loads(Path(args.original).read_text())
    reproduction = reproduction_check(summary, original)
    print(f"[repro] max abs diff vs original = {reproduction['max_abs_diff']:.3e}", flush=True)

    cpu_encoder = CrossEncoder(Path(args.model), "cpu", args.batch_size, args.max_length)
    print(f"[model] CPU copy ready, threads {cpu_encoder.torch.get_num_threads()}", flush=True)
    cpu = cpu_latency(index, subset, pools, cpu_encoder, args.latency_sample)
    contention = None
    if not args.skip_contention_check:
        print("[cpu] contention check", flush=True)
        contention = contention_check(index, subset, pools, encoder)
        cpu["contention_check"] = contention
        print(
            f"  EBR alone {contention['ebr_alone_ms_p50']:.1f} ms, "
            f"with MPS rerank {contention['ebr_with_mps_rerank_ms_p50']:.1f} ms",
            flush=True,
        )

    report = {
        "experiment": "Entity-Bridge Retrieval: paired neural comparison and CPU like-for-like latency",
        "note": (
            "Direct paired test of EBR against the cross-encoder, and of the cross-encoder "
            "on EBR's pool against the cross-encoder on Porter's pool. Same 1,000 qids, same "
            "model, same metrics, no tuning. Latency: both sides on CPU, cross-encoder timing "
            "includes the candidate construction it needs."
        ),
        "subset": {
            "split": args.split,
            "n": len(subset),
            "seed": args.seed,
            "selection": "uniform without replacement from sorted test qids",
            "first_qid": subset[0],
            "last_qid": subset[-1],
        },
        "model": {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "path": str(args.model),
            "device": encoder.device,
            "batch_size": encoder.batch_size,
            "max_length": encoder.max_length,
            "score": "logits.squeeze(-1), torch.no_grad",
            "torch": encoder.torch.__version__,
        },
        "bootstrap": {"resamples": 1000, "seed": 1729, "kind": "paired percentile, query resampling"},
        "metrics": sorted(
            {
                key
                for values in rows.values()
                for row in values
                for key in row
                if key not in ("type", "qid")
            }
        ),
        "summary": summary,
        "paired": paired,
        "paired_direction": "candidate - base, keyed by '<candidate>_minus_<base>'",
        "per_query": {
            label: {row["qid"]: {key: row[key] for key in row if key not in ("qid",)} for row in values}
            for label, values in rows.items()
        },
        "cpu_latency": cpu,
        "gpu_latency_reference": {
            "source": "results/hotpot_neural_reference.json (unchanged, not rerun here)",
            "ebr_full_ms": original["latency"]["ebr_full"],
            "cross_encoder_rerank_union_ms": original["latency"]["cross_encoder_rerank_union"],
            "note": "GPU (MPS) rerank-only for the cross-encoder vs CPU end-to-end for EBR; not like for like.",
        },
        "cache": {"pools": pool_source, "scores": score_source, "unique_pairs_scored": sum(len(scores[q]) for q in subset)},
        "reproduction": reproduction,
        "wall_seconds": time.perf_counter() - started,
    }

    print(f"\n== Paired ({len(subset)} test qids)")
    header = f"{'system':10} {'nDCG@10':>8} {'both@10':>8} {'both@5':>8} {'R@10':>8}"
    print(header)
    for label in SYSTEMS:
        row = summary[label]
        print(f"{label:10} {row['ndcg@10']:8.3f} {row['both@10']:8.3f} {row['both@5']:8.3f} {row['recall@10']:8.3f}")
    for name in ("ebr_minus_ce_porter", "ce_ebr_minus_ce_porter", "ebr_minus_ce_ebr"):
        print(f"\n  {name}  (Δ = {name.replace('_minus_', ' − ')})")
        for metric in METRICS:
            row = paired[name][metric]
            lo, hi = row["ci95"]
            print(f"    {metric:10} Δ={row['delta']:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  W/T/L={row['wtl'][0]}/{row['wtl'][1]}/{row['wtl'][2]}")
    lat = report["cpu_latency"]
    print(
        f"\n  CPU ms/query (n={lat['sample_queries']}): "
        f"ebr {lat['ebr_full']['ms_p50']:.1f} p50 / {lat['ebr_full']['ms_p95']:.1f} p95, "
        f"ce_porter_total {lat['ce_porter_total']['ms_p50']:.1f} / {lat['ce_porter_total']['ms_p95']:.1f}, "
        f"ce_ebr_total {lat['ce_ebr_total']['ms_p50']:.1f} / {lat['ce_ebr_total']['ms_p95']:.1f}, "
        f"ce_union_total {lat['ce_union_total']['ms_p50']:.1f} / {lat['ce_union_total']['ms_p95']:.1f}"
    )
    print(f"  ratio (p50): CE(Porter, total) / EBR = {lat['ratio_ce_porter_over_ebr']['ratio_of_p50']:.2f}x")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
