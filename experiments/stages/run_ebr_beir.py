#!/usr/bin/env python3
"""Is EBR safe to apply everywhere, or does it need a gate?

EBR improves HotpotQA multi-hop retrieval (+0.105 both@10 over Porter BM25). A
multi-hop channel is not automatically safe on a single-hop corpus, so this run
measures three things:

  1. EBR channels on single-hop BEIR sets (SciFact, Touche, NFCorpus). If the
     bridge and per-entity channels cost nDCG here, EBR cannot be a universal
     replacement.
  2. A fixed text-only router (lab/router.py) on HotpotQA: how often does it fire,
     and is routed (EBR when it fires, Porter otherwise) close to always-EBR?
  3. The same router on the single-hop sets: fire rate and non-inferiority of
     routed against always-Porter.

Constants are the HotpotQA ones (HOTPOT.md): 20 seeds, 20 bridges, 5% df cap,
RRF k=60. Nothing is tuned here. SciFact train and HotpotQA dev are reported as
dev sanity; SciFact test, Touche test and HotpotQA test are the evidence.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lab"))

import ebr_beir
import fusion
import harness
import hotpot
import router
from channels import porter
from lean_units import LeanUnitBM25

BEIR_SETS = (("scifact", "train"), ("scifact", "test"), ("touche", "test"), ("nfcorpus", "dev"))
BEIR_SYSTEMS = ("porter", "entity", "bridge", "rrf_porter_bridge", "rrf_porter_entity",
                "rrf_porter_bridge_entity", "routed")
HOTPOT_SYSTEMS = ("porter", "routed", "rrf_porter_bridge_entity")
BEIR_METRICS = ("ndcg", "r100", "mrr")
HOTPOT_METRICS = ("ndcg@10", "both@10", "recall@10")
CI_KW = {"resamples": 2000}


def load_beir(name: str, split: str):
    seconds = time.perf_counter()
    data = ebr_beir.load(name, split)
    print(f"[load] {name}/{split}: {data.stats()} in {time.perf_counter() - seconds:.1f}s", flush=True)
    return data


def beir_rows(data: ebr_beir.Corpus, porter_system):
    """One row per query per system, plus the router decision and its cues."""
    rows = {label: [] for label in BEIR_SYSTEMS}
    fired_by_cue = Counter()
    for qid, qrels in data.qrels.items():
        porter_scores = porter_system.score(data.queries[qid])
        channels = ebr_beir.channels(data, qid, porter_scores)
        question = data.raw_queries[qid]
        fired = router.is_multihop(question)
        fired_cues = tuple(cue for cue, hit in router.cues(question).items() if hit) if fired else ()
        if fired:
            for cue in fired_cues:
                fired_by_cue[cue] += 1
        ebr_scores = channels["rrf_porter_bridge_entity"]
        systems = {"porter": porter_scores, "routed": ebr_scores if fired else porter_scores}
        systems.update(channels)
        for label, scores in systems.items():
            row = harness.metrics(harness.rank(data.ix, scores, 1000), qrels)
            row["fired"] = fired
            row["cues"] = fired_cues
            if label == "porter":
                row["qid"] = qid
                row["question"] = question
            rows[label].append(row)
    return rows, fired_by_cue


def hotpot_rows(index, porter_system):
    rows = {label: [] for label in HOTPOT_SYSTEMS}
    fired_by_cue = Counter()
    for qid, qrels in index["qrels"].items():
        query = index["queries"][qid]
        porter_scores = porter_system.score(query)
        entities = hotpot.query_entities(index["raw_queries"][qid], index["title_index"])
        entity_scores = hotpot.dense(index["ix"], hotpot.per_entity_scores(index, qid, entities), index["position"])
        bridges, _ = hotpot.bridge_scores(index, qid, porter_scores)
        bridge_dense = hotpot.dense(index["ix"], bridges, index["position"])
        ebr_scores = fusion.rrf(index["ix"], [porter_scores, bridge_dense, entity_scores])
        question = index["raw_queries"][qid]
        fired = router.is_multihop(question)
        fired_cues = tuple(cue for cue, hit in router.cues(question).items() if hit) if fired else ()
        if fired:
            for cue in fired_cues:
                fired_by_cue[cue] += 1
        systems = {"porter": porter_scores,
                   "routed": ebr_scores if fired else porter_scores,
                   "rrf_porter_bridge_entity": ebr_scores}
        for label, scores in systems.items():
            row = hotpot.metrics(harness.rank(index["ix"], scores, 1000), qrels)
            row["qid"] = qid
            row["fired"] = fired
            row["cues"] = fired_cues
            row["type"] = index["meta"][qid]["type"]
            rows[label].append(row)
    return rows, fired_by_cue


def mean(rows, metric):
    return sum(row[metric] for row in rows) / max(len(rows), 1)


def summary(rows, metrics, systems):
    return {label: {metric: mean(rows[label], metric) for metric in metrics}
            for label in systems}


def paired(base_rows, cand_rows, metrics):
    return {metric: hotpot.paired(base_rows, cand_rows, metric, **CI_KW) for metric in metrics}


def subset(rows, fired):
    return [row for row in rows if row["fired"] is fired]


def fired_delta_by_cue(rows, metrics, base="porter", cand="rrf_porter_bridge_entity"):
    """EBR minus Porter per rule, over the queries that rule fired on. Queries can fire several rules."""
    groups: dict[str, list[int]] = {}
    for position, row in enumerate(rows["porter"]):
        if not row["fired"]:
            continue
        for cue in row["cues"]:
            groups.setdefault(cue, []).append(position)
    report = {}
    for cue, positions in sorted(groups.items()):
        report[cue] = {"queries": len(positions)}
        for metric in metrics:
            diff = [rows[cand][i][metric] - rows[base][i][metric] for i in positions]
            report[cue][f"delta_{metric}"] = sum(diff) / len(diff)
            report[cue][f"wtl_{metric}"] = [sum(d > 1e-12 for d in diff),
                                            sum(abs(d) <= 1e-12 for d in diff),
                                            sum(d < -1e-12 for d in diff)]
    return report


def fired_examples(rows, base="porter", cand="rrf_porter_bridge_entity", limit=25):
    return [
        {
            "qid": row["qid"],
            "question": row["question"],
            "cues": list(row["cues"]),
            "porter_ndcg": row["ndcg"],
            "ebr_ndcg": rows[cand][position]["ndcg"],
        }
        for position, row in enumerate(rows[base]) if row["fired"]
    ][:limit]


def print_beir(name, split, rows, fire_rate, cues):
    print(f"\n== {name}/{split}  queries={len(rows['porter'])}  router fire={fire_rate:.1%}  cues={dict(cues)}")
    print(f"{'system':26} {'nDCG@10':>8} {'R@100':>7} {'MRR':>7}  {'ΔnDCG':>8} {'95% CI':>20} {'W/T/L':>14}")
    for label in BEIR_SYSTEMS:
        ndcg, r100, mrr = (mean(rows[label], m) for m in BEIR_METRICS)
        line = f"{label:26} {ndcg:8.4f} {r100:7.4f} {mrr:7.4f}"
        if label != "porter":
            d = paired(rows["porter"], rows[label], ("ndcg",))["ndcg"]
            lo, hi = d["ci95"]
            w, t, l = d["wtl"]
            line += f"  {d['delta']:+8.4f} [{lo:+.4f},{hi:+.4f}] {w:5}/{t:4}/{l:4}"
        print(line)
    cost = paired(rows["rrf_porter_bridge_entity"], rows["routed"], ("ndcg",))["ndcg"]
    lo, hi = cost["ci95"]
    print(f"  routed vs always-EBR: ΔnDCG={cost['delta']:+.4f} [{lo:+.4f},{hi:+.4f}]   "
          f"fired delta by cue: {({cue: round(v['delta_ndcg'], 3) for cue, v in fired_delta_by_cue(rows, ('ndcg',)).items()})}")


def print_hotpot(split, rows, fire_rate, cues):
    print(f"\n== hotpot/{split}  queries={len(rows['porter'])}  router fire={fire_rate:.1%}  cues={dict(cues)}")
    print(f"{'system':26} {'nDCG@10':>8} {'both@10':>8} {'R@10':>7}  {'Δboth@10':>9} {'95% CI':>20} {'ΔnDCG':>8} {'95% CI':>20}")
    for label in HOTPOT_SYSTEMS:
        ndcg, both, r10 = (mean(rows[label], m) for m in HOTPOT_METRICS)
        line = f"{label:26} {ndcg:8.4f} {both:8.4f} {r10:7.4f}"
        if label != "porter":
            b = paired(rows["porter"], rows[label], ("both@10", "ndcg@10"))
            blo, bhi = b["both@10"]["ci95"]
            nlo, nhi = b["ndcg@10"]["ci95"]
            line += (f"  {b['both@10']['delta']:+9.4f} [{blo:+.4f},{bhi:+.4f}]"
                     f" {b['ndcg@10']['delta']:+8.4f} [{nlo:+.4f},{nhi:+.4f}]")
        print(line)
    cost = paired(rows["rrf_porter_bridge_entity"], rows["routed"], ("both@10", "ndcg@10"))
    lo, hi = cost["both@10"]["ci95"]
    nlo, nhi = cost["ndcg@10"]["ci95"]
    print(f"  routed vs always-EBR: Δboth@10={cost['both@10']['delta']:+.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"ΔnDCG={cost['ndcg@10']['delta']:+.4f} [{nlo:+.4f},{nhi:+.4f}]")


def beir_report(rows, fire_rate, cues):
    report = {
        "queries": len(rows["porter"]),
        "router_fire_rate": fire_rate,
        "fired_cues": dict(cues),
        "summary": summary(rows, BEIR_METRICS, BEIR_SYSTEMS),
        "paired_vs_porter": {label: paired(rows["porter"], rows[label], BEIR_METRICS)
                             for label in BEIR_SYSTEMS if label != "porter"},
        "paired_routed_minus_ebr": paired(rows["rrf_porter_bridge_entity"], rows["routed"], ("ndcg",)),
        "fired_delta_by_cue": fired_delta_by_cue(rows, ("ndcg",)),
        "fired_examples": fired_examples(rows),
    }
    for fired in (True, False):
        block = {label: subset(rows[label], fired) for label in ("porter", "rrf_porter_bridge_entity")}
        if block["porter"]:
            report["fired_subset" if fired else "silent_subset"] = {
                "queries": len(block["porter"]),
                "porter": {m: mean(block["porter"], m) for m in BEIR_METRICS},
                "rrf_porter_bridge_entity": {m: mean(block["rrf_porter_bridge_entity"], m) for m in BEIR_METRICS},
                "paired_ebr_vs_porter": paired(block["porter"], block["rrf_porter_bridge_entity"], BEIR_METRICS),
            }
    return report


def hotpot_report(rows, fire_rate, cues):
    report = {
        "queries": len(rows["porter"]),
        "router_fire_rate": fire_rate,
        "fired_cues": dict(cues),
        "summary": summary(rows, HOTPOT_METRICS + ("both@5",), HOTPOT_SYSTEMS),
        "paired_vs_porter": {label: paired(rows["porter"], rows[label], HOTPOT_METRICS + ("both@5",))
                             for label in HOTPOT_SYSTEMS if label != "porter"},
        "paired_routed_minus_ebr": paired(rows["rrf_porter_bridge_entity"], rows["routed"], ("both@10", "ndcg@10")),
        "fired_delta_by_cue": fired_delta_by_cue(rows, ("both@10",)),
        "by_type": {},
    }
    for kind in ("bridge", "comparison"):
        mask = [index for index, row in enumerate(rows["porter"]) if row["type"] == kind]
        block = {label: [rows[label][i] for i in mask] for label in HOTPOT_SYSTEMS}
        report["by_type"][kind] = {
            "queries": len(mask),
            "summary": summary(block, HOTPOT_METRICS, HOTPOT_SYSTEMS),
            "paired_routed_vs_porter": paired(block["porter"], block["routed"], ("both@10", "ndcg@10")),
        }
    for fired in (True, False):
        block = {label: subset(rows[label], fired) for label in HOTPOT_SYSTEMS}
        if block["porter"]:
            report["fired_subset" if fired else "silent_subset"] = {
                "queries": len(block["porter"]),
                "summary": summary(block, ("both@10", "ndcg@10"), ("porter", "rrf_porter_bridge_entity")),
                "paired_ebr_vs_porter": paired(block["porter"], block["rrf_porter_bridge_entity"], ("both@10", "ndcg@10")),
            }
    return report


def main():
    report = {
        "config": {
            "beir_sets": [f"{name}/{split}" for name, split in BEIR_SETS],
            "hotpot_splits": ["dev", "test"],
            "seeds": ebr_beir.SEEDS,
            "bridges": ebr_beir.BRIDGES,
            "max_df_frac": ebr_beir.MAX_DF_FRAC,
            "rrf_k": ebr_beir.RRF_K,
            "router": "lab/router.py, rules frozen on HotpotQA dev + SciFact train",
            "data_root": str(harness.DATA),
        },
        "beir": {},
        "hotpot": {},
    }

    for name, split in BEIR_SETS:
        data = load_beir(name, split)
        porter_system = LeanUnitBM25(data.ix, porter)
        seconds = time.perf_counter()
        rows, cues = beir_rows(data, porter_system)
        fire_rate = sum(row["fired"] for row in rows["porter"]) / len(rows["porter"])
        print_beir(name, split, rows, fire_rate, cues)
        report["beir"].setdefault(name, {})[split] = beir_report(rows, fire_rate, cues)
        print(f"[done] {name}/{split} in {time.perf_counter() - seconds:.0f}s", flush=True)
        del data, porter_system

    index = hotpot.load(str(HERE / "data/hotpot_dev.parquet"))
    print(f"\n[load] hotpot: {index['ix'].N} paragraphs, {len(index['qrels'])} questions", flush=True)
    porter_system = LeanUnitBM25(index["ix"], porter)
    for split in ("dev", "test"):
        keep = {qid: qrels for qid, qrels in index["qrels"].items() if hotpot.in_split(qid, split)}
        split_index = dict(index, qrels=keep)
        seconds = time.perf_counter()
        rows, cues = hotpot_rows(split_index, porter_system)
        fire_rate = sum(row["fired"] for row in rows["porter"]) / len(rows["porter"])
        print_hotpot(split, rows, fire_rate, cues)
        report["hotpot"][split] = hotpot_report(rows, fire_rate, cues)
        print(f"[done] hotpot/{split} in {time.perf_counter() - seconds:.0f}s", flush=True)

    out = HERE / "results/ebr_beir.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
