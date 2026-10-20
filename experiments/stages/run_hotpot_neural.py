#!/usr/bin/env python3
"""Neural reference for Entity-Bridge Retrieval (EBR) on HotpotQA.

This is a COMPARISON, not part of the proposed no-embedding method. EBR uses no
embeddings, no neural index and no LLM. Here we add one neural component purely as
a reference point: a pretrained cross-encoder (ms-marco-MiniLM-L-6) that reranks
candidate paragraphs. The question it answers is "how close does the sparse,
no-embedding method get to a RAG-style neural reranker?", not "can we improve EBR
with a transformer?".

Systems (identical deterministic subset of the HotpotQA test split):
  porter                   Porter BM25 over all 66,581 paragraphs
  ebr                      RRF(Porter, bridge channel, per-entity channel), k=60
  ce_porter                cross-encoder rerank of the Porter BM25 top-100
  ce_ebr                   cross-encoder rerank of the EBR fused top-100

The two rerankers share one candidate union per query, so every candidate pair is
scored exactly once and both rerankers see the identical neural scores.

Usage:
  .venv/bin/python experiments/stages/run_hotpot_neural.py
  .venv/bin/python experiments/stages/run_hotpot_neural.py --n 500 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lab"))

import numpy as np
import pyarrow.parquet as pq

import fusion
import harness
import hotpot
import run_hotpot
from channels import porter
from hotpot import dense
from lean_units import LeanUnitBM25

MODEL_DIR = Path("/Users/asmit/Personal/algo/first_principles/models/ms-marco-MiniLM-L-6-v2")
SYSTEMS = ("porter", "ebr", "ce_porter", "ce_ebr")
CUTOFF = 100


def raw_documents(path: str) -> dict[str, str]:
    """Title -> raw paragraph text, first occurrence wins (same rule as hotpot.load)."""
    docs: dict[str, str] = {}
    for row in pq.read_table(path).to_pylist():
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            if title not in docs:
                docs[title] = " ".join(sentences)
    return docs


def select_subset(index, split: str, n: int, seed: int) -> list[str]:
    pool = sorted(qid for qid in index["qrels"] if hotpot.in_split(qid, split))
    if n >= len(pool):
        return pool
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(pool), size=n, replace=False)
    return sorted(pool[i] for i in picked)


def top_docs(ix, scores, k: int = CUTOFF) -> list[str]:
    return [ix.doc_ids[i] for i in fusion.order(scores, k)]


def ebr_scores(index, qid, porter_system):
    """Exactly the rrf_porter_bridge_entity channel set from run_hotpot.query_systems."""
    ix = index["ix"]
    query = index["queries"][qid]
    porter_scores = porter_system.score(query)
    entities = hotpot.query_entities(index["raw_queries"][qid], index["title_index"])
    entity_dense = dense(ix, hotpot.per_entity_scores(index, qid, entities), index["position"])
    bridges, _ = hotpot.bridge_scores(index, qid, porter_scores)
    bridge_dense = dense(ix, bridges, index["position"])
    return porter_scores, fusion.rrf(ix, [porter_scores, bridge_dense, entity_dense])


class CrossEncoder:
    """ms-marco cross-encoder. MPS when available, CPU otherwise; both reported."""

    def __init__(self, model_dir: Path, device: str = "auto", batch_size: int = 32, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            print("[model] MPS requested but unavailable, falling back to CPU", flush=True)
            device = "cpu"
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(device).eval()
        self.load_seconds = time.perf_counter() - started

    def score(self, query: str, documents: list[str]) -> list[float]:
        torch = self.torch
        if not documents:
            return []
        out: list[float] = []
        with torch.no_grad():
            for start in range(0, len(documents), self.batch_size):
                batch = documents[start:start + self.batch_size]
                inputs = self.tokenizer(
                    [query] * len(batch), batch,
                    padding=True, truncation=True, max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits.squeeze(-1)
                out.extend(logits.detach().float().cpu().tolist())
        if self.device == "mps":
            torch.mps.synchronize()
        return out


def rerank(ranked: list[str], scores: list[float]) -> list[str]:
    return [doc for doc, _ in sorted(zip(ranked, scores), key=lambda pair: (-pair[1], pair[0]))]


def evaluate(index, subset, porter_system, encoder, progress: int = 100):
    rows = {label: [] for label in SYSTEMS}
    latency = {"bm25": [], "ebr": [], "ebr_marginal": [], "rerank": []}
    pairs_total = 0
    truncated = 0
    started = time.perf_counter()
    for position, qid in enumerate(subset, start=1):
        ix = index["ix"]
        qrels = index["qrels"][qid]
        raw = index["raw_queries"][qid]

        t0 = time.perf_counter()
        porter_scores = porter_system.score(index["queries"][qid])
        t1 = time.perf_counter()
        entities = hotpot.query_entities(raw, index["title_index"])
        entity_dense = dense(ix, hotpot.per_entity_scores(index, qid, entities), index["position"])
        bridges, _ = hotpot.bridge_scores(index, qid, porter_scores)
        bridge_dense = dense(ix, bridges, index["position"])
        fused = fusion.rrf(ix, [porter_scores, bridge_dense, entity_dense])
        t2 = time.perf_counter()
        latency["bm25"].append(t1 - t0)
        latency["ebr"].append(t2 - t0)
        latency["ebr_marginal"].append(t2 - t1)

        porter_ranked = run_hotpot.ranked_from_scores(index, porter_scores)
        ebr_ranked = run_hotpot.ranked_from_scores(index, fused)
        porter_pool = top_docs(ix, porter_scores, CUTOFF)
        ebr_pool = top_docs(ix, fused, CUTOFF)
        if len(ebr_pool) < CUTOFF:
            truncated += 1
        union = sorted(set(porter_pool) | set(ebr_pool))

        t3 = time.perf_counter()
        neural = encoder.score(raw, [f"{doc}. {index['docs'][doc]}" for doc in union])
        latency["rerank"].append(time.perf_counter() - t3)
        neural_by_doc = dict(zip(union, neural))
        pairs_total += len(union)

        ranked = {
            "porter": porter_ranked,
            "ebr": ebr_ranked,
            "ce_porter": rerank(porter_pool, [neural_by_doc[doc] for doc in porter_pool]),
            "ce_ebr": rerank(ebr_pool, [neural_by_doc[doc] for doc in ebr_pool]),
        }
        for label, ranking in ranked.items():
            row = hotpot.metrics(ranking, qrels)
            row["qid"] = qid
            row["type"] = index["meta"][qid]["type"]
            rows[label].append(row)

        if progress and position % progress == 0:
            elapsed = time.perf_counter() - started
            print(f"[eval] {position}/{len(subset)} queries, {elapsed:5.0f}s", flush=True)
    return rows, latency, pairs_total, truncated


def latency_report(latency, pairs_total, queries):
    def block(values):
        arr = np.array(values, dtype=float) * 1000.0
        return {
            "ms_mean": float(arr.mean()),
            "ms_p50": float(np.percentile(arr, 50)),
            "ms_p95": float(np.percentile(arr, 95)),
        }

    rerank = block(latency["rerank"])
    return {
        "bm25_first_hop": block(latency["bm25"]),
        "ebr_full": block(latency["ebr"]),
        "ebr_marginal_over_bm25": block(latency["ebr_marginal"]),
        "cross_encoder_rerank_union": {
            **rerank,
            "note": "one union of both top-100 pools per query, batch 32, max_length 512",
            "pairs_total": pairs_total,
            "pairs_per_query": pairs_total / queries,
            "pairs_per_second": pairs_total / (sum(latency["rerank"]) or 1.0),
        },
        "queries": queries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(HERE / "data/hotpot_dev.parquet"))
    parser.add_argument("--out", default=str(HERE / "results/hotpot_neural_reference.json"))
    parser.add_argument("--n", type=int, default=1000, help="number of test questions")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--model", default=str(MODEL_DIR))
    args = parser.parse_args()

    started = time.perf_counter()
    index = hotpot.load(args.data)
    index["docs"] = raw_documents(args.data)
    subset = select_subset(index, args.split, args.n, args.seed)
    print(f"[load] {index['ix'].N} paragraphs, {len(index['qrels'])} questions", flush=True)
    print(f"[subset] {len(subset)} {args.split} qids, seed {args.seed}", flush=True)

    porter_system = LeanUnitBM25(index["ix"], porter)
    encoder = CrossEncoder(Path(args.model), args.device, args.batch_size, args.max_length)
    print(
        f"[model] {args.model} on {encoder.device} "
        f"(mps_available={encoder.torch.backends.mps.is_available()}), "
        f"load {encoder.load_seconds:.1f}s, batch {encoder.batch_size}, max_len {encoder.max_length}",
        flush=True,
    )

    # Ranking parity with the no-neural pipeline: EBR scores must match run_hotpot.
    for qid in subset[:5]:
        reference = run_hotpot.query_systems(index, qid, porter_system)["rrf_porter_bridge_entity"]
        _, ours = ebr_scores(index, qid, porter_system)
        if not np.allclose(reference, ours):
            raise SystemExit(f"EBR score parity failed for {qid}; refusing to report")

    rows, latency, pairs_total, truncated = evaluate(index, subset, porter_system, encoder)
    summary = hotpot.summarize(rows)
    paired = {
        "ebr_vs_porter": hotpot.paired_report(rows["porter"], rows["ebr"], ("ndcg@10", "both@10", "both@5", "recall@10")),
        "ce_porter_vs_porter": hotpot.paired_report(rows["porter"], rows["ce_porter"], ("ndcg@10", "both@10", "both@5", "recall@10")),
        "ce_ebr_vs_porter": hotpot.paired_report(rows["porter"], rows["ce_ebr"], ("ndcg@10", "both@10", "both@5", "recall@10")),
        "ce_ebr_vs_ce_porter": hotpot.paired_report(rows["ce_porter"], rows["ce_ebr"], ("ndcg@10", "both@10", "both@5", "recall@10")),
        "ce_ebr_vs_ebr": hotpot.paired_report(rows["ebr"], rows["ce_ebr"], ("ndcg@10", "both@10", "both@5", "recall@10")),
    }

    full = json.loads((HERE / "results/hotpot.json").read_text())["global:test"]["summary"]
    report = {
        "experiment": "Entity-Bridge Retrieval: neural cross-encoder reference",
        "note": (
            "The cross-encoder is a pretrained neural reranker. It is NOT part of EBR and "
            "not part of the proposed no-embedding method. It bounds how much of a "
            "RAG-style neural reranker's quality the sparse method already reaches."
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
            "architecture": "BertForSequenceClassification",
            "parameters_millions": 22.7,
            "device": encoder.device,
            "mps_available": bool(encoder.torch.backends.mps.is_available()),
            "fallback_to_cpu": encoder.device == "cpu" and encoder.torch.backends.mps.is_available(),
            "batch_size": encoder.batch_size,
            "max_length": encoder.max_length,
            "score": "logits.squeeze(-1), torch.no_grad",
            "load_seconds": encoder.load_seconds,
            "torch": encoder.torch.__version__,
        },
        "summary": summary,
        "paired_vs_porter": paired,
        "latency": latency_report(latency, pairs_total, len(subset)),
        "pool_notes": {
            "rerank_depth": CUTOFF,
            "ebr_pool_truncated_queries": truncated,
            "unique_pairs_scored": pairs_total,
        },
        "sanity": {
            "full_global_test_porter": {
                key: full["porter"][key] for key in ("ndcg@10", "both@10", "both@5", "recall@10")
            },
            "full_global_test_ebr": {
                key: full["rrf_porter_bridge_entity"][key] for key in ("ndcg@10", "both@10", "both@5", "recall@10")
            },
            "subset_porter": {
                key: summary["porter"][key] for key in ("ndcg@10", "both@10", "both@5", "recall@10")
            },
            "subset_ebr": {
                key: summary["ebr"][key] for key in ("ndcg@10", "both@10", "both@5", "recall@10")
            },
        },
        "wall_seconds": time.perf_counter() - started,
    }

    print(f"\n== HotpotQA neural reference ({len(subset)} test qids)")
    header = f"{'system':10} {'nDCG@10':>8} {'both@10':>8} {'both@5':>8} {'R@10':>8}"
    print(header)
    for label in SYSTEMS:
        row = summary[label]
        print(f"{label:10} {row['ndcg@10']:8.3f} {row['both@10']:8.3f} {row['both@5']:8.3f} {row['recall@10']:8.3f}")
    for label, block in paired.items():
        print(f"\n  {label}")
        for metric, row in block.items():
            lo, hi = row["ci95"]
            print(f"    {metric:10} Δ={row['delta']:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  W/T/L={row['wtl'][0]}/{row['wtl'][1]}/{row['wtl'][2]}")
    lat = report["latency"]
    print(
        f"\n  latency ms/query: bm25 {lat['bm25_first_hop']['ms_mean']:.1f}, "
        f"ebr {lat['ebr_full']['ms_mean']:.1f}, "
        f"rerank {lat['cross_encoder_rerank_union']['ms_mean']:.0f} "
        f"({lat['cross_encoder_rerank_union']['pairs_per_second']:.0f} pairs/s)"
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
