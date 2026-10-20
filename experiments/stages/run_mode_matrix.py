#!/usr/bin/env python3
"""ARGON A3: ship/kill the mode boundary and safety invariants.

One frozen configuration per mode, one runner, one result file. This experiment
does not look for a better scorer. It asks whether a product policy of
`default = Porter BM25`, `deep = EBR`, `routed = frozen per-query router`,
`graph = Porter + code graph` can ship without silently degrading ordinary search.

Workloads, fixed before the run:
  hotpot/test      6,005 held-out multi-hop questions
  scifact/test       300 held-out single-hop claims
  touche/test         49 held-out argument queries
  nfcorpus/dev       324 diagnostic single-hop queries
  serp_code/250      250 synthetic graph-closure code queries (capability test)
  exact-ID          JS engine suite (read-only) + Python fixture on SciFact + code symbols

Metrics: Hotpot both@10/both@5/nDCG@10/R@10; BEIR nDCG@10/R@100/MRR; code
both@10/R@10/nDCG@10; exact hit@1/@10, MRR, near-miss-before-exact, unknown-ID
activation; p50/p95 latency; build/storage cost; mode fire rate. Paired bootstrap
is 1,000 resamples, seed 1729, against Porter BM25.

Safety is decided by `mode_matrix.safety_verdict` at the pre-registered margin
-0.005, not by reading the sign of a delta.
"""
from __future__ import annotations

import copy
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE / "lab"))

import numpy as np

import channels
import code_graph as cg
import ebr_beir
import exact_ids
import fusion
import harness
import hotpot
import mode_matrix as mm
import router
from lean_units import LeanUnitBM25

RESAMPLES = 1000
CI_SEED = 1729
BEIR_SETS = (("scifact", "test"), ("touche", "test"), ("nfcorpus", "dev"))
BEIR_MODES = ("default", "deep", "routed", "deep_gated", "routed_gated")
HOTPOT_MODES = ("default", "deep", "routed", "deep_gated", "routed_gated")
CODE_MODES = ("default", "graph", "graph_gated")

RESULTS = HERE / "results"
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def peak_rss_mb() -> float:
    """Process peak RSS. Monotonic, so report it as a process peak, not a delta."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def p50_p95(samples: list[float]) -> dict:
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "queries": 0}
    values = np.array(samples, dtype=float)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(values.mean()),
        "queries": int(values.size),
    }


def mean(rows: list[dict], metric: str) -> float:
    return sum(row[metric] for row in rows) / max(len(rows), 1)


def summarize(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    return {metric: mean(rows, metric) for metric in metrics}


def paired(base: list[dict], candidate: list[dict], metrics: tuple[str, ...]) -> dict:
    return {
        metric: hotpot.paired(base, candidate, metric,
                              resamples=RESAMPLES, seed=CI_SEED)
        for metric in metrics
    }


def paired_with_verdict(base, candidate, metrics, margin_metrics=("ndcg", "ndcg@10")) -> dict:
    table = paired(base, candidate, metrics)
    for metric in metrics:
        if metric in margin_metrics:
            table[metric].update(mm.gate_row(table[metric]["delta"], table[metric]["ci95"]))
    return table


def latency(samples: list[float]) -> dict:
    return p50_p95(samples)


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# BEIR workloads
# ---------------------------------------------------------------------------
def beir_fixture_scorers(data, porter_system):
    """Exact-ID fixture scorers over a loaded BEIR corpus (SciFact)."""
    ix = data.ix

    def deep_scores(text: str):
        tokens = harness.tokenize(text)
        porter_scores = porter_system.score(tokens)
        local = copy.copy(data)
        local.raw_queries = {"__fixture__": text}
        entities = ebr_beir.query_entities(local, "__fixture__")
        entity = ebr_beir.dense(local, ebr_beir.per_entity_scores(local, entities))
        bridges, _ = ebr_beir.bridge_scores(local, "__fixture__", porter_scores)
        bridge = ebr_beir.dense(local, bridges)
        fused = fusion.rrf(ix, [porter_scores, bridge, entity])
        expanded = bool((bridge > 0).any() or (entity > 0).any())
        return fused, expanded

    def default_scores(text: str):
        return porter_system.score(harness.tokenize(text)), False

    def deep_gated(text: str):
        if mm.identifier_gate(text):
            return porter_system.score(harness.tokenize(text)), False
        return deep_scores(text)

    return {"default": default_scores, "deep": deep_scores, "deep_gated": deep_gated}


def run_beir(name: str, split: str) -> tuple[dict, dict]:
    workload = f"{name}/{split}"
    print(f"\n[load] {workload}", flush=True)
    started = time.perf_counter()
    data = ebr_beir.load(name, split)
    build_seconds = time.perf_counter() - started
    porter_system = LeanUnitBM25(data.ix, channels.porter)
    print(f"[load] {workload}: {data.stats()} in {build_seconds:.1f}s", flush=True)

    rows = {mode: [] for mode in BEIR_MODES}
    lat = {mode: [] for mode in ("default", "deep", "routed")}
    fired_count = gated_count = 0
    order = list(data.qrels)
    warm = order[0]
    porter_system.score(data.queries[warm])
    ebr_beir.channels(data, warm, porter_system.score(data.queries[warm]))

    for qid in order:
        qrels = data.qrels[qid]
        question = data.raw_queries[qid]
        t0 = time.perf_counter()
        porter_scores = porter_system.score(data.queries[qid])
        t_porter = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        chans = ebr_beir.channels(data, qid, porter_scores)
        t_deep = (time.perf_counter() - t0) * 1000
        deep_scores = chans["rrf_porter_bridge_entity"]

        t0 = time.perf_counter()
        fired = router.is_multihop(question)
        t_router = (time.perf_counter() - t0) * 1000
        gated = mm.identifier_gate(question)
        fired_count += int(fired)
        gated_count += int(gated)

        systems = {
            "default": porter_scores,
            "deep": deep_scores,
            "routed": deep_scores if fired else porter_scores,
            "deep_gated": porter_scores if gated else deep_scores,
            "routed_gated": porter_scores if (gated or not fired) else deep_scores,
        }
        lat["default"].append(t_porter)
        lat["deep"].append(t_porter + t_deep)
        lat["routed"].append(t_porter + t_router + (t_deep if fired else 0.0))

        for mode, scores in systems.items():
            row = harness.metrics(harness.rank(data.ix, scores, 1000), qrels)
            row["fired"] = fired
            row["gated"] = gated
            rows[mode].append(row)

    summary = {mode: summarize(rows[mode], mm.BEIR_METRICS) for mode in BEIR_MODES}
    deltas = {
        mode: paired_with_verdict(rows["default"], rows[mode], mm.BEIR_METRICS)
        for mode in BEIR_MODES if mode != "default"
    }
    report = {
        "workload": workload,
        "kind": "single_hop",
        "role": mm.WORKLOADS[workload]["role"],
        "diagnostic_only": workload == "nfcorpus/dev",
        "queries": len(order),
        "router_fire_rate": fired_count / max(len(order), 1),
        "identifier_gate_rate": gated_count / max(len(order), 1),
        "automatic_modes": list(mm.AUTOMATIC[workload]),
        "summary": summary,
        "paired_vs_porter": deltas,
        "latency_ms": {mode: latency(values) for mode, values in lat.items()},
        "cost": {
            "docs": data.ix.N,
            "entities": len(data.entity_docs),
            "entity_postings": int(sum(len(v) for v in data.entity_docs.values())),
            "build_seconds": build_seconds,
            "peak_rss_mb": peak_rss_mb(),
            "note": "build_seconds covers index load + entity-postings build; peak RSS is process-wide.",
        },
        "qids_sha256": digest(order),
    }
    print(
        f"[{workload}] porter ndcg={summary['default']['ndcg']:.4f} "
        f"deep={summary['deep']['ndcg']:.4f} routed={summary['routed']['ndcg']:.4f} "
        f"fire={report['router_fire_rate']:.1%} gate={report['identifier_gate_rate']:.1%}",
        flush=True,
    )
    return report, {"data": data, "porter_system": porter_system, "rows": rows}


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------
def run_hotpot() -> dict:
    workload = "hotpot/test"
    print(f"\n[load] {workload}", flush=True)
    started = time.perf_counter()
    index = hotpot.load(str(DATA / "hotpot_dev.parquet"))
    build_seconds = time.perf_counter() - started
    porter_system = LeanUnitBM25(index["ix"], channels.porter)
    keep = {qid: qrels for qid, qrels in index["qrels"].items()
            if hotpot.in_split(qid, "test")}
    print(f"[load] {workload}: {index['ix'].N} paragraphs, {len(keep)} questions "
          f"in {build_seconds:.1f}s", flush=True)

    rows = {mode: [] for mode in HOTPOT_MODES}
    lat = {mode: [] for mode in ("default", "deep", "routed")}
    fired_count = gated_count = 0
    order = list(keep)
    warm = order[0]
    porter_system.score(index["queries"][warm])

    for qid in order:
        qrels = keep[qid]
        question = index["raw_queries"][qid]
        t0 = time.perf_counter()
        porter_scores = porter_system.score(index["queries"][qid])
        t_porter = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        entities = hotpot.query_entities(question, index["title_index"])
        entity_scores = hotpot.dense(
            index["ix"], hotpot.per_entity_scores(index, qid, entities), index["position"])
        bridges, _ = hotpot.bridge_scores(index, qid, porter_scores)
        bridge_dense = hotpot.dense(index["ix"], bridges, index["position"])
        deep_scores = fusion.rrf(index["ix"], [porter_scores, bridge_dense, entity_scores])
        t_deep = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fired = router.is_multihop(question)
        t_router = (time.perf_counter() - t0) * 1000
        gated = mm.identifier_gate(question)
        fired_count += int(fired)
        gated_count += int(gated)

        systems = {
            "default": porter_scores,
            "deep": deep_scores,
            "routed": deep_scores if fired else porter_scores,
            "deep_gated": porter_scores if gated else deep_scores,
            "routed_gated": porter_scores if (gated or not fired) else deep_scores,
        }
        lat["default"].append(t_porter)
        lat["deep"].append(t_porter + t_deep)
        lat["routed"].append(t_porter + t_router + (t_deep if fired else 0.0))

        for mode, scores in systems.items():
            row = hotpot.metrics(harness.rank(index["ix"], scores, 1000), qrels)
            row["fired"] = fired
            row["gated"] = gated
            rows[mode].append(row)

    summary = {mode: summarize(rows[mode], mm.HOTPOT_METRICS) for mode in HOTPOT_MODES}
    deltas = {
        mode: paired_with_verdict(rows["default"], rows[mode], mm.HOTPOT_METRICS)
        for mode in HOTPOT_MODES if mode != "default"
    }
    return {
        "workload": workload,
        "kind": "multi_hop",
        "role": mm.WORKLOADS[workload]["role"],
        "declared_profile": mm.WORKLOADS[workload]["declared_profile"],
        "queries": len(order),
        "router_fire_rate": fired_count / max(len(order), 1),
        "identifier_gate_rate": gated_count / max(len(order), 1),
        "automatic_modes": list(mm.AUTOMATIC[workload]),
        "summary": summary,
        "paired_vs_porter": deltas,
        "latency_ms": {mode: latency(values) for mode, values in lat.items()},
        "cost": {
            "docs": index["ix"].N,
            "entities": len(index["entity_paragraphs"]),
            "entity_postings": int(sum(len(v) for v in index["entity_paragraphs"].values())),
            "build_seconds": build_seconds,
            "peak_rss_mb": peak_rss_mb(),
            "note": "build_seconds covers parquet parse + index + entity postings; peak RSS is process-wide.",
        },
        "qids_sha256": digest(order),
    }


# ---------------------------------------------------------------------------
# Code corpus
# ---------------------------------------------------------------------------
def run_code() -> tuple[dict, dict]:
    workload = "serp_code/250"
    print(f"\n[build] {workload}", flush=True)
    started = time.perf_counter()
    graph = cg.build_graph(ROOT)
    ix = cg.build_index(graph)
    queries = cg.generate_queries(graph, per_type=50, seed=cg.SEED)
    build_seconds = time.perf_counter() - started
    position = {doc: i for i, doc in enumerate(ix.doc_ids)}
    print(f"[build] {workload}: {graph.stats()} queries={len(queries)} "
          f"in {build_seconds:.2f}s", flush=True)

    rows = {mode: [] for mode in CODE_MODES}
    diagnostic = []
    lat = {mode: [] for mode in ("default", "graph")}
    gate_count = 0

    def score_query(text: str):
        tokens = cg.porter(cg.code_tokens(text))
        t0 = time.perf_counter()
        bm25 = harness.bm25_fast(ix, tokens)
        t_porter = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        seeds = [ix.doc_ids[i] for i in fusion.order(bm25, cg.SEED_CHUNKS)]
        graph_scores = cg.graph_scores(graph, position, seeds)
        t_graph = (time.perf_counter() - t0) * 1000
        fused = fusion.rrf(ix, [bm25, graph_scores])
        return bm25, graph_scores, fused, t_porter, t_graph

    score_query(queries[0]["text"])  # warm the sparse impacts cache

    for query in queries:
        text = query["text"]
        bm25, graph_scores, fused, t_porter, t_graph = score_query(text)
        gated = mm.identifier_gate(text)
        gate_count += int(gated)
        systems = {
            "default": bm25,
            "graph": fused,
            "graph_gated": bm25 if gated else fused,
        }
        lat["default"].append(t_porter)
        lat["graph"].append(t_porter + t_graph)
        for mode, scores in systems.items():
            ranked = harness.rank(ix, scores, 1000)
            row = cg.metrics(ranked, query["qrels"])
            row["answer@10"] = 1.0 if query["answer"] in ranked[:10] else 0.0
            row["type"] = query["type"]
            row["gated"] = gated
            rows[mode].append(row)
        entity = cg.entity_scores(graph, position, text)
        diagnostic.append(cg.metrics(
            harness.rank(ix, fusion.rrf(ix, [bm25, entity, graph_scores]), 1000),
            query["qrels"]))

    summary = {mode: summarize(rows[mode], mm.CODE_METRICS) for mode in CODE_MODES}
    deltas = {
        mode: paired_with_verdict(rows["default"], rows[mode], mm.CODE_METRICS,
                                  margin_metrics=("ndcg@10",))
        for mode in CODE_MODES if mode != "default"
    }
    by_type = {}
    for kind in ("same_file", "import_call", "two_hop", "export_import", "calls_two"):
        subset = {mode: [row for row in rows[mode] if row["type"] == kind] for mode in CODE_MODES}
        by_type[kind] = {
            "queries": len(subset["default"]),
            "both@10": {mode: mean(subset[mode], "both@10") for mode in CODE_MODES},
        }
    report = {
        "workload": workload,
        "kind": "code",
        "role": mm.WORKLOADS[workload]["role"],
        "claim_boundary": ("synthetic capability test; qrels are graph-closure from the same "
                           "graph the channel traverses, not human relevance"),
        "declared_profile": mm.WORKLOADS[workload]["declared_profile"],
        "queries": len(queries),
        "identifier_gate_rate": gate_count / max(len(queries), 1),
        "automatic_modes": list(mm.AUTOMATIC[workload]),
        "summary": summary,
        "paired_vs_porter": deltas,
        "by_type": by_type,
        "diagnostic_not_a_mode": {
            "label": "rrf_porter_entity_graph",
            "summary": summarize(diagnostic, mm.CODE_METRICS),
        },
        "latency_ms": {mode: latency(values) for mode, values in lat.items()},
        "cost": {
            "docs_chunks": len(graph.chunks),
            "graph": graph.stats(),
            "build_seconds": build_seconds,
            "peak_rss_mb": peak_rss_mb(),
            "note": "build_seconds covers repo parse + graph + index + query generation.",
        },
        "qids_sha256": digest([query["qid"] for query in queries]),
    }
    print(f"[{workload}] porter both@10={summary['default']['both@10']:.3f} "
          f"graph both@10={summary['graph']['both@10']:.3f}", flush=True)
    return report, {"graph": graph, "ix": ix, "queries": queries}


# ---------------------------------------------------------------------------
# Exact-identifier invariants
# ---------------------------------------------------------------------------
def run_scifact_exact(data, porter_system) -> dict:
    """Python fixture (lab/exact_ids.py) on SciFact for every text mode."""
    suite = exact_ids.build_suite(data.ix)
    scorers = beir_fixture_scorers(data, porter_system)
    out = {"suite": {"positives": len(suite["positives"]),
                     "negatives": len(suite["negatives"])},
           "systems": {}}
    for label, scorer in scorers.items():
        out["systems"][label] = mm.evaluate_exact(data.ix, suite, scorer)
    # the frozen router cannot fire on a bare identifier (no capitalised entity),
    # so routed is default by construction; assert it rather than assume it.
    fired = sum(router.is_multihop(row["query"]) for row in suite["positives"])
    out["router_fires_on_identifiers"] = fired
    return out


def run_code_exact(graph, ix) -> dict:
    """Exact symbol vs near-miss over the code chunk corpus, verbatim and unknown."""
    suite = mm.code_symbol_suite(ix, stem=lambda term: cg.porter([term])[0])
    position = {doc: i for i, doc in enumerate(ix.doc_ids)}

    def score(text: str):
        tokens = cg.porter(cg.code_tokens(text))
        bm25 = harness.bm25_fast(ix, tokens)
        seeds = [ix.doc_ids[i] for i in fusion.order(bm25, cg.SEED_CHUNKS)]
        graph_scores = cg.graph_scores(graph, position, seeds)
        fused = fusion.rrf(ix, [bm25, graph_scores])
        expanded = bool((graph_scores > 0).any())
        return bm25, fused, expanded

    def default_scorer(text: str):
        bm25, _, _ = score(text)
        return bm25, False

    def graph_scorer(text: str):
        _, fused, expanded = score(text)
        return fused, expanded

    def graph_gated(text: str):
        bm25, fused, expanded = score(text)
        if mm.identifier_gate(text):
            return bm25, False
        return fused, expanded

    out = {"suite": {"positives": len(suite["positives"]),
                     "negatives": len(suite["negatives"])},
           "systems": {}}
    for label, scorer in (("default", default_scorer), ("graph", graph_scorer),
                          ("graph_gated", graph_gated)):
        out["systems"][label] = mm.evaluate_exact(ix, suite, scorer)
    return out


def js_engine_exact() -> dict:
    """The JS engine's own exact-ID suite, read from its frozen benchmark result."""
    path = RESULTS / "stage-benchmark.json"
    if not path.exists():
        return {"available": False, "reason": f"{path.name} absent"}
    payload = json.loads(path.read_text())
    exact = payload.get("exactIdentifiers")
    if not exact:
        return {"available": False, "reason": "exactIdentifiers block absent"}
    negatives = exact.get("negatives", {})
    activation = {
        key: value.get("activation", 0.0)
        for key, value in negatives.items() if isinstance(value, dict)
    }
    return {
        "available": True,
        "source": "results/stage-benchmark.json (JS engine, read-only)",
        "positives": exact.get("positives"),
        "exact_hit@1": exact.get("exactTop1"),
        "exact_hit@10": exact.get("exactTop10"),
        "exact_found": exact.get("exactFound"),
        "mean_exact_rank": exact.get("meanExactRank"),
        "near_miss_before_exact": exact.get("nearMissAbove"),
        "near_miss_positives": exact.get("nearMissPositives"),
        "negative_activation": activation,
        "note": ("MRR is not emitted by the JS suite; mean exact rank is. "
                 "Activation is the JS engine's own non-empty-result counter."),
    }


# ---------------------------------------------------------------------------
# Gate 1: default reproduces Porter
# ---------------------------------------------------------------------------
def reproduction_checks(beir: dict, hotpot_report: dict, code: dict) -> dict:
    reference_path = RESULTS / "ebr_beir.json"
    code_path = RESULTS / "code_graph.json"
    checks = {"tolerance": 1e-4, "sources": [], "sets": {}, "passed": None}

    reference = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    if reference:
        checks["sources"].append("results/ebr_beir.json")
        for workload, metrics in (("scifact/test", mm.BEIR_METRICS),
                                  ("touche/test", mm.BEIR_METRICS),
                                  ("nfcorpus/dev", mm.BEIR_METRICS)):
            name, split = workload.split("/")
            stored = reference.get("beir", {}).get(name, {}).get(split, {})
            stored = stored.get("summary", {}).get("porter", {})
            observed = beir[workload]["summary"]["default"]
            checks["sets"][workload] = {
                metric: {"observed": observed[metric],
                         "reference": stored.get(metric),
                         "abs_diff": abs(observed[metric] - stored.get(metric, float("nan")))
                         if metric in stored else None}
                for metric in metrics
            }
        stored = reference.get("hotpot", {}).get("test", {}).get("summary", {}).get("porter", {})
        observed = hotpot_report["summary"]["default"]
        checks["sets"]["hotpot/test"] = {
            metric: {"observed": observed[metric],
                     "reference": stored.get(metric),
                     "abs_diff": abs(observed[metric] - stored.get(metric, float("nan")))
                     if metric in stored else None}
            for metric in ("ndcg@10", "both@10")
        }

    if code_path.exists():
        checks["sources"].append("results/code_graph.json")
        stored = json.loads(code_path.read_text()).get("systems", {}).get("porter", {})
        observed = code["summary"]["default"]
        checks["sets"]["serp_code/250"] = {
            metric: {"observed": observed[metric],
                     "reference": stored.get(metric),
                     "abs_diff": abs(observed[metric] - stored.get(metric, float("nan")))
                     if metric in stored else None}
            for metric in ("both@10", "ndcg@10")
        }

    diffs = [
        row["abs_diff"]
        for lines in checks["sets"].values()
        for row in lines.values()
        if row["abs_diff"] is not None and not np.isnan(row["abs_diff"])
    ]
    checks["max_abs_diff"] = max(diffs) if diffs else None
    checks["passed"] = bool(diffs) and max(diffs) <= checks["tolerance"]
    return checks


# ---------------------------------------------------------------------------
# Gate 3: single-hop non-inferiority of automatic policies
# ---------------------------------------------------------------------------
def single_hop_gate(beir: dict) -> dict:
    """Gate 3. Automatic policies must show non-inferiority at -0.005.

    Two readings are reported. `literal_passed` is the letter of the gate: no
    policy's CI lies entirely below the margin. `passed` is the stricter
    non-inferiority reading: every CI must clear the margin. A CI such as
    [-0.011, -0.0007] does not establish the -0.005 bound, so the stricter
    reading governs a ship decision.
    """
    out = {
        "margin": mm.MARGIN,
        "policies": {},
        "explicit_deep_informational": {},
        "literal_passed": None,
        "passed": None,
        "failing": [],
        "router_loses_on_scifact": False,
    }
    for workload in ("scifact/test", "touche/test", "nfcorpus/dev"):
        for mode in ("routed", "routed_gated"):
            entry = beir[workload]["paired_vs_porter"][mode]["ndcg"]
            out["policies"][f"{workload}::{mode}"] = {
                "delta": entry["delta"], "ci95": entry["ci95"],
                "verdict": entry["verdict"],
                "significant_negative": entry["significant_negative"],
            }
        deep = beir[workload]["paired_vs_porter"]["deep"]["ndcg"]
        out["explicit_deep_informational"][workload] = {
            "delta": deep["delta"], "ci95": deep["ci95"],
            "verdict": deep["verdict"],
            "note": "deep is explicit here, never automatic on a single-hop workload",
        }

    out["literal_passed"] = all(
        entry["verdict"] != "fail" for entry in out["policies"].values())
    out["passed"] = all(
        entry["verdict"] == "pass" for entry in out["policies"].values())
    out["failing"] = [key for key, entry in out["policies"].items()
                      if entry["verdict"] != "pass"]
    out["router_loses_on_scifact"] = bool(
        any("scifact" in key and entry["significant_negative"]
            for key, entry in out["policies"].items()))
    return out


def verdict_payload(report: dict) -> dict:
    gates = report["gates"]
    safety_failed = any(not value["passed"] for value in gates.values())
    single_hop = gates["single_hop_non_inferiority"]
    return {
        "bm25_only_default": bool(safety_failed),
        "deep_shipped_automatically": False,
        "deep_shipped_behind_explicit_flag": True,
        "graph_shipped_behind_corpus_flag": True,
        "router_shipped": False,
        "failing_gates": [name for name, value in gates.items() if not value["passed"]],
        "single_hop_key_result": {
            "router_loses_on_scifact": single_hop["router_loses_on_scifact"],
            "literal_gate_passed": single_hop["literal_passed"],
            "non_inferiority_passed": single_hop["passed"],
            "failing": single_hop["failing"],
        },
        "recommended_policy": (
            "Porter BM25 default with the exact-identifier gate; deep (EBR) only as an "
            "explicit mode or declared multi-hop profile; graph only behind a code-corpus "
            "profile; the per-query router is not shipped."
            if safety_failed else
            "Porter BM25 default; deep and graph remain explicit corpus-scoped flags; "
            "the per-query router is not shipped."
        ),
        "reason": (
            "At least one safety invariant failed. Default search stays lexical; "
            "specialists are opt-in flags, not automatic policy."
            if safety_failed else
            "Every safety invariant passed; specialists may ship, still behind flags "
            "for cost and scope rather than safety."
        ),
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def print_workload(report: dict, metrics: tuple[str, ...]) -> None:
    print(f"\n== {report['workload']}  n={report['queries']}  "
          f"router fire={report.get('router_fire_rate', 0.0):.1%}  "
          f"ID gate={report['identifier_gate_rate']:.1%}")
    header = " ".join(f"{metric:>9}" for metric in metrics)
    print(f"{'mode':14} {header}  {'Δndcg':>8} {'95% CI':>20} {'safety':>12}")
    for mode in report["summary"]:
        row = report["summary"][mode]
        cells = " ".join(f"{row[metric]:9.4f}" for metric in metrics)
        if mode == "default":
            print(f"{mode:14} {cells}")
            continue
        entry = report["paired_vs_porter"][mode].get(
            "ndcg", report["paired_vs_porter"][mode].get("ndcg@10"))
        lo, hi = entry["ci95"]
        print(f"{mode:14} {cells}  {entry['delta']:+8.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {entry['verdict']:>12}")


def print_exact(exact: dict) -> None:
    print("\n== exact-identifier invariants")
    for label, block in exact.items():
        if label in ("js_engine",):
            continue
        if "systems" not in block:
            continue
        print(f"  [{label}]  positives={block['suite']['positives']} "
              f"negatives={block['suite']['negatives']}")
        print(f"    {'system':14} {'hit@1':>7} {'hit@10':>7} {'MRR':>7} "
              f"{'nearFirst':>10} {'negAct':>7} {'expandAct':>10}")
        for system, row in block["systems"].items():
            print(f"    {system:14} {row['exact_hit@1']:7.3f} {row['exact_hit@10']:7.3f} "
                  f"{row['exact_mrr']:7.3f} {row['near_miss_before_exact']:10.3f} "
                  f"{row['negative_activation']:7.3f} "
                  f"{row['unknown_identifier_expansion']:10.3f}")
    js = exact.get("js_engine", {})
    if js.get("available"):
        print(f"  [js_engine] positives={js['positives']} hit@1={js['exact_hit@1']:.3f} "
              f"hit@10={js['exact_hit@10']:.3f} meanRank={js['mean_exact_rank']:.2f} "
              f"nearFirst={js['near_miss_before_exact']:.3f} "
              f"negAct={json.dumps(js['negative_activation'])}")


def main() -> None:
    started = time.perf_counter()
    if (ebr_beir.SEEDS, ebr_beir.BRIDGES, ebr_beir.MAX_DF_FRAC, ebr_beir.RRF_K) != (
            mm.SEEDS, mm.BRIDGES, mm.MAX_DF_FRAC, mm.RRF_K):
        raise SystemExit("frozen constants disagree between mode_matrix and ebr_beir")

    report = {
        "meta": {
            "experiment": "ARGON A3 — mode boundary and safety matrix",
            "protocol": "one frozen configuration per mode; no tuning on test queries",
            "seeds": mm.SEEDS, "bridges": mm.BRIDGES,
            "max_df_frac": mm.MAX_DF_FRAC, "rrf_k": mm.RRF_K,
            "resamples": RESAMPLES, "ci_seed": CI_SEED, "margin": mm.MARGIN,
            "router": "lab/router.py, frozen rules (EBR_ROUTER.md)",
            "modes": list(mm.MODES),
            "workloads": {name: {key: value for key, value in spec.items()}
                          for name, spec in mm.WORKLOADS.items()},
            "identifier_gate": ("digit/underscore identifiers force the lexical default; "
                                "bare numbers are not identifiers"),
        },
        "workloads": {},
        "exact_ids": {},
        "gates": {},
    }

    beir_reports = {}
    for name, split in BEIR_SETS:
        workload_report, _ = run_beir(name, split)
        beir_reports[f"{name}/{split}"] = workload_report
        report["workloads"][f"{name}/{split}"] = workload_report

    hotpot_report = run_hotpot()
    report["workloads"]["hotpot/test"] = hotpot_report

    code_report, code_state = run_code()
    report["workloads"]["serp_code/250"] = code_report

    # SciFact is the only text corpus small enough to evaluate the exact fixture
    # without a second 66k-document load, so re-load it (index is cached).
    data = ebr_beir.load("scifact", "test")
    porter_system = LeanUnitBM25(data.ix, channels.porter)
    report["exact_ids"]["scifact_python_fixture"] = run_scifact_exact(data, porter_system)
    report["exact_ids"]["code_symbols"] = run_code_exact(code_state["graph"], code_state["ix"])
    report["exact_ids"]["js_engine"] = js_engine_exact()

    exact_scifact = report["exact_ids"]["scifact_python_fixture"]["systems"]
    exact_code = report["exact_ids"]["code_symbols"]["systems"]
    exact_ok = True
    for system, row in exact_scifact.items():
        shipped = system in ("default", "deep_gated")
        if shipped and (row["exact_hit@1"] < 1.0 or row["near_miss_before_exact"] > 0.0):
            exact_ok = False
        if row["unknown_identifier_expansion"] > 0.0:
            exact_ok = False
    for system, row in exact_code.items():
        shipped = system in ("default", "graph_gated")
        if shipped and (row["exact_hit@1"] < 1.0 or row["near_miss_before_exact"] > 0.0):
            exact_ok = False
        if row["unknown_identifier_expansion"] > 0.0:
            exact_ok = False

    reproduction = reproduction_checks(beir_reports, hotpot_report, code_report)
    single_hop = single_hop_gate(beir_reports)
    automatic_deep = [name for name, modes in mm.AUTOMATIC.items() if "deep" in modes]
    selection_ok = automatic_deep == ["hotpot/test"] and not mm.is_automatic(
        "scifact/test", "deep")

    report["gates"] = {
        "default_reproduces_porter": reproduction,
        "deep_selection_boundary": {
            "passed": selection_ok,
            "declared_profiles": {"multi_hop_qa": "hotpot/test", "code_graph": "serp_code/250"},
            "automatic_deep_workloads": automatic_deep,
            "per_query_router_is_shipped": False,
            "note": ("deep is automatic only for the declared multi_hop_qa workload; "
                     "routed is measured as a baseline row and never shipped."),
        },
        "single_hop_non_inferiority": single_hop,
        "exact_identifiers": {
            "passed": exact_ok,
            "js_engine_available": report["exact_ids"]["js_engine"].get("available", False),
            "scifact_fixture": exact_scifact,
            "code_symbols": exact_code,
        },
    }
    report["verdict"] = verdict_payload(report)
    report["runtime_seconds"] = time.perf_counter() - started

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "mode_matrix.json"
    out.write_text(json.dumps(report, indent=1))

    manifest = DATA / "mode_queries.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for workload, spec in mm.WORKLOADS.items():
        entry = report["workloads"][workload]
        lines.append({
            "workload": workload, "corpus": spec["corpus"], "split": spec["split"],
            "kind": spec["kind"], "role": spec["role"],
            "queries": entry["queries"], "qids_sha256": entry["qids_sha256"],
            "automatic_modes": entry["automatic_modes"],
            "declared_profile": spec["declared_profile"],
        })
    for workload, block in (("scifact_exact_fixture",
                             report["exact_ids"]["scifact_python_fixture"]),
                            ("code_symbol_fixture", report["exact_ids"]["code_symbols"])):
        lines.append({
            "workload": workload, "kind": "exact_id_fixture",
            "positives": block["suite"]["positives"], "negatives": block["suite"]["negatives"],
            "modes": sorted(block["systems"]),
        })
    manifest.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    for workload in ("scifact/test", "touche/test", "nfcorpus/dev"):
        print_workload(report["workloads"][workload], mm.BEIR_METRICS)
    print_workload(report["workloads"]["hotpot/test"], mm.HOTPOT_METRICS)
    print_workload(report["workloads"]["serp_code/250"], mm.CODE_METRICS)
    print_exact(report["exact_ids"])

    print("\n== latency (ms/query) and cost")
    for workload, entry in report["workloads"].items():
        lat = entry["latency_ms"]
        cells = "  ".join(f"{mode} p50={row['p50']:.2f}/p95={row['p95']:.2f}"
                          for mode, row in lat.items())
        print(f"  {workload:16} {cells}  build={entry['cost']['build_seconds']:.1f}s")
        if "entities" in entry["cost"]:
            print(f"  {'':16} entities={entry['cost']['entities']} "
                  f"postings={entry['cost']['entity_postings']} "
                  f"peakRSS={entry['cost']['peak_rss_mb']:.0f}MB")

    print("\n== gates")
    for name, gate in report["gates"].items():
        state = "PASS" if gate["passed"] else "FAIL"
        print(f"  {name:30} {state}")
        if name == "single_hop_non_inferiority":
            for key, entry in gate["policies"].items():
                print(f"    {key:34} Δ={entry['delta']:+.4f} "
                      f"CI[{entry['ci95'][0]:+.4f},{entry['ci95'][1]:+.4f}] {entry['verdict']}")
        if name == "default_reproduces_porter":
            print(f"    max_abs_diff={gate['max_abs_diff']} tol={gate['tolerance']}")
    print(f"\n  verdict: {json.dumps(report['verdict'], indent=2)}")
    print(f"\n[saved] {out}")
    print(f"[saved] {manifest}")
    print(f"[runtime] {report['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
