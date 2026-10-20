"""Frozen multi-hop transfer test: EBR on 2WikiMultiHopQA and MuSiQue.

The HotpotQA EBR implementation (experiments/stages/lab/hotpot.py) is imported
read-only. Nothing in it is modified or re-tuned: Porter BM25 k1=1.2/b=0.75,
20 first-hop seeds, 20 bridge entities, 5% entity df cap, fixed RRF k=60, depth
1000, no typed filtering, no router, no parameter search.

Datasets are evidence-completeness benchmarks. `all@k` (every gold paragraph in
the top k) is the primary metric; nDCG is reported because it is the ordinary
ranking view, not because it is the target.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict

import pyarrow.parquet as pq

import fusion
import harness
import hotpot

SYSTEMS = ("porter", "bridge", "entity", "ebr")
METRICS = ("all@5", "all@10", "recall@10", "recall@100", "ndcg@10", "mrr")
DEPTH = fusion.DEPTH


# --------------------------------------------------------------------------- data


def build_corpus(pairs):
    """Global paragraph corpus, deduplicated by title (first occurrence wins).

    Returns (paragraphs{title: text}, collisions) where `collisions` counts repeat
    titles whose text differs from the first occurrence.
    """
    paragraphs: dict[str, str] = {}
    collisions = 0
    for title, text in pairs:
        if title in paragraphs:
            if paragraphs[title] != text:
                collisions += 1
            continue
        paragraphs[title] = text
    return paragraphs, collisions


def build_index(paragraphs: dict[str, str], titles: dict[str, str] | None = None):
    """Same index shape hotpot.load() produces, so hotpot's channels run unmodified.

    `paragraphs` is keyed by document identity. For 2Wiki the key is the title, so
    `titles` is optional. MuSiQue paragraph-level runs key by (title, text) and pass
    the display title separately, because entity extraction must see the title.
    """
    titles = titles if titles is not None else {key: key for key in paragraphs}
    ix = harness.Index({key: (titles[key], body) for key, body in paragraphs.items()})
    title_index = {}
    for key in ix.doc_ids:
        title_index[hotpot.normalize_entity(titles[key])] = key
    paragraph_entities: dict[str, set[str]] = {}
    entity_paragraphs: dict[str, set[str]] = defaultdict(set)
    for doc_id in ix.doc_ids:
        entities = hotpot.extract_entities(titles[doc_id] + " " + paragraphs[doc_id])
        entities.add(hotpot.normalize_entity(titles[doc_id]))
        paragraph_entities[doc_id] = entities
        for entity in entities:
            entity_paragraphs[entity].add(doc_id)
    return {
        "ix": ix,
        "position": {doc: i for i, doc in enumerate(ix.doc_ids)},
        "title_index": title_index,
        "paragraph_entities": paragraph_entities,
        "entity_paragraphs": dict(entity_paragraphs),
        "entity_df": {entity: len(docs) for entity, docs in entity_paragraphs.items()},
        "raw_queries": {},
        "queries": {},
        "qrels": {},
        "meta": {},
    }


def load_2wiki(path: str):
    rows = pq.read_table(path).to_pylist()
    pairs = []
    for row in rows:
        for title, sentences in json.loads(row["context"]):
            pairs.append((title, " ".join(sentences)))
    paragraphs, collisions = build_corpus(pairs)
    index = build_index(paragraphs)
    known = set(index["ix"].doc_ids)
    missing = 0
    collapsed = 0
    for row in rows:
        qid = str(row["_id"])
        facts = json.loads(row["supporting_facts"])
        supports = sorted({title for title, _ in facts})
        missing += sum(1 for title in supports if title not in known)
        collapsed += len(facts) > len(supports)
        index["raw_queries"][qid] = row["question"]
        index["queries"][qid] = harness.tokenize(row["question"])
        index["qrels"][qid] = {title: 1 for title in supports if title in known}
        index["meta"][qid] = {
            "type": row["type"],
            "slice": row["type"],
            "support_count": len(supports),
            "support": supports,
        }
    stats = {
        "questions": len(rows),
        "paragraphs": len(paragraphs),
        "title_collisions": collisions,
        "missing_support_titles": missing,
        "questions_with_collapsed_support_titles": collapsed,
        "support_count_histogram": dict(
            sorted(Counter(m["support_count"] for m in index["meta"].values()).items())
        ),
    }
    return index, stats


def load_musique(path: str):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    pairs = []
    for row in rows:
        for paragraph in row["paragraphs"]:
            pairs.append((paragraph["title"], paragraph["paragraph_text"]))
    paragraphs, collisions = build_corpus(pairs)
    index = build_index(paragraphs)
    known = set(index["ix"].doc_ids)
    missing = 0
    collapsed = 0
    leakage = 0
    for row in rows:
        qid = str(row["id"])
        supporting = [p for p in row["paragraphs"] if p["is_supporting"]]
        supports = sorted({p["title"] for p in supporting})
        missing += sum(1 for title in supports if title not in known)
        collapsed += len(supporting) > len(supports)
        leakage += any(
            not p["is_supporting"] and p["title"] in set(supports) for p in row["paragraphs"]
        )
        index["raw_queries"][qid] = row["question"]
        index["queries"][qid] = harness.tokenize(row["question"])
        index["qrels"][qid] = {title: 1 for title in supports if title in known}
        label = f"support_{len(supports)}"
        index["meta"][qid] = {
            "type": label,
            "slice": label,
            "support_count": len(supports),
            "support": supports,
        }
    stats = {
        "questions": len(rows),
        "paragraphs": len(paragraphs),
        "title_collisions": collisions,
        "missing_support_titles": missing,
        "questions_with_collapsed_support_titles": collapsed,
        "questions_with_title_leakage": leakage,
        "support_count_histogram": dict(
            sorted(Counter(m["support_count"] for m in index["meta"].values()).items())
        ),
    }
    return index, stats


MUSIQUE_KEY_SEP = "\x00"


def paragraph_key(title: str, text: str) -> str:
    """Stable MuSiQue paragraph identity: title plus exact paragraph text."""
    return f"{title}{MUSIQUE_KEY_SEP}{text}"


def load_musique_paragraphs(path: str):
    """MuSiQue with unique paragraph identity instead of title-only identity.

    The corpus key is (title, paragraph_text), so two different paragraphs of the
    same Wikipedia article stay separate documents, and the qrels are the supporting
    paragraphs themselves. A non-supporting paragraph that shares a gold title can no
    longer be credited as relevant; it remains in the corpus as a hard distractor.
    """
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    paragraphs: dict[str, str] = {}
    titles: dict[str, str] = {}
    per_title: dict[str, set[str]] = defaultdict(set)
    slots = 0
    for row in rows:
        for paragraph in row["paragraphs"]:
            slots += 1
            key = paragraph_key(paragraph["title"], paragraph["paragraph_text"])
            paragraphs[key] = paragraph["paragraph_text"]
            titles[key] = paragraph["title"]
            per_title[paragraph["title"]].add(paragraph["paragraph_text"])

    index = build_index(paragraphs, titles)
    known = set(index["ix"].doc_ids)
    missing = 0
    collapsed = 0
    confuser_questions = 0
    confuser_paragraphs = 0
    identical_slots = 0
    for row in rows:
        qid = str(row["id"])
        gold = {
            paragraph_key(p["title"], p["paragraph_text"])
            for p in row["paragraphs"] if p["is_supporting"]
        }
        gold_titles = {p["title"] for p in row["paragraphs"] if p["is_supporting"]}
        missing += sum(1 for key in gold if key not in known)
        collapsed += sum(1 for p in row["paragraphs"] if p["is_supporting"]) > len(gold)
        confusers = [
            p for p in row["paragraphs"]
            if not p["is_supporting"] and p["title"] in gold_titles
        ]
        if confusers:
            confuser_questions += 1
            confuser_paragraphs += len(confusers)
            identical_slots += sum(
                1 for p in confusers if paragraph_key(p["title"], p["paragraph_text"]) in gold
            )
        index["raw_queries"][qid] = row["question"]
        index["queries"][qid] = harness.tokenize(row["question"])
        index["qrels"][qid] = {key: 1 for key in gold if key in known}
        label = f"support_{len(gold)}"
        index["meta"][qid] = {
            "type": label,
            "slice": label,
            "support_count": len(gold),
            "support": sorted(gold),
        }
    stats = {
        "questions": len(rows),
        "paragraph_slots": slots,
        "paragraphs": len(paragraphs),
        "duplicate_paragraph_slots": slots - len(paragraphs),
        "titles": len(per_title),
        "titles_with_multiple_paragraphs": sum(1 for texts in per_title.values() if len(texts) > 1),
        "title_collisions": sum(len(texts) - 1 for texts in per_title.values()),
        "missing_support_paragraphs": missing,
        "questions_with_collapsed_support_paragraphs": collapsed,
        "questions_with_same_title_confuser": confuser_questions,
        "same_title_confuser_paragraphs": confuser_paragraphs,
        "same_title_confusers_identical_to_gold": identical_slots,
        "support_count_histogram": dict(
            sorted(Counter(m["support_count"] for m in index["meta"].values()).items())
        ),
    }
    return index, stats


# ----------------------------------------------------------------------- channels


def porter_scores_for(index, qid, porter_system):
    return porter_system.score(index["queries"][qid])


def expansion_channels(index, qid, porter_scores):
    """Bridge + per-entity sparse channels, frozen upstream constants."""
    ix = index["ix"]
    entities = hotpot.query_entities(index["raw_queries"][qid], index["title_index"])
    entity_scores = hotpot.per_entity_scores(index, qid, entities)
    bridges, _ = hotpot.bridge_scores(index, qid, porter_scores)
    return (
        hotpot.dense(ix, bridges, index["position"]),
        hotpot.dense(ix, entity_scores, index["position"]),
    )


def fuse(index, porter_scores, bridge_dense, entity_dense):
    return fusion.rrf(index["ix"], [porter_scores, bridge_dense, entity_dense])


# ------------------------------------------------------------------------ metrics


def metrics(ranked, relevant):
    rel = set(relevant)
    out = {}
    for cutoff in (5, 10):
        top = set(ranked[:cutoff])
        out[f"all@{cutoff}"] = 1.0 if rel <= top else 0.0
        out[f"recall@{cutoff}"] = len(rel & top) / len(rel)
    out["recall@100"] = len(rel & set(ranked[:100])) / len(rel)
    dcg = sum(1 / math.log2(i + 2) for i, doc in enumerate(ranked[:10]) if doc in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    out["ndcg@10"] = dcg / ideal if ideal else 0.0
    first = next((i + 1 for i, doc in enumerate(ranked) if doc in rel), None)
    out["mrr"] = 1 / first if first else 0.0
    return out


def evaluate(index, limit: int | None = None):
    from channels import porter as porter_transform
    from lean_units import LeanUnitBM25

    ix = index["ix"]
    porter_bm25 = LeanUnitBM25(ix, porter_transform)
    rows = {label: [] for label in SYSTEMS}
    reach = Counter()
    latency = Counter()
    qids = list(index["qrels"])
    if limit:
        qids = qids[:limit]

    for qid in qids:
        relevant = index["qrels"][qid]

        started = time.perf_counter()
        porter_scores = porter_scores_for(index, qid, porter_bm25)
        latency["porter_score_ms"] += (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        bridge_dense, entity_dense = expansion_channels(index, qid, porter_scores)
        latency["expansion_ms"] += (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        ebr_scores = fuse(index, porter_scores, bridge_dense, entity_dense)
        latency["fusion_ms"] += (time.perf_counter() - started) * 1000

        ranked = {}
        for label, scores in (
            ("porter", porter_scores), ("bridge", bridge_dense),
            ("entity", entity_dense), ("ebr", ebr_scores),
        ):
            started = time.perf_counter()
            ranked[label] = fusion.ranked_ids(ix, scores, DEPTH)
            latency[f"rank_{label}_ms"] += (time.perf_counter() - started) * 1000

        for label in SYSTEMS:
            row = metrics(ranked[label], relevant)
            row["qid"] = qid
            row["slice"] = index["meta"][qid]["slice"]
            rows[label].append(row)

        porter20 = set(ranked["porter"][:20])
        porter100 = set(ranked["porter"][:100])
        bridge_all = set(ranked["bridge"])
        bridge10 = set(ranked["bridge"][:10])
        ebr100 = set(ranked["ebr"][:100])
        for title in relevant:
            reach["gold"] += 1
            if title in porter100:
                reach["gold_in_porter100"] += 1
            if title in porter20:
                continue
            reach["gold_outside_porter20"] += 1
            if title in bridge_all:
                reach["bridge_reaches"] += 1
            if title in bridge10:
                reach["bridge_top10"] += 1
        reach["extra_candidates"] += len(ebr100 - porter100)
        reach["ebr_top100_total"] += len(ebr100)

    n = max(len(qids), 1)
    reach_report = {
        "gold_paragraphs": reach["gold"],
        "gold_in_porter_top100": reach["gold_in_porter100"],
        "fraction_gold_in_porter_top100": reach["gold_in_porter100"] / max(reach["gold"], 1),
        "gold_outside_porter_top20": reach["gold_outside_porter20"],
        "bridge_reaches_outside_top20": reach["bridge_reaches"],
        "bridge_reach_fraction": reach["bridge_reaches"] / max(reach["gold_outside_porter20"], 1),
        "bridge_reaches_in_top10": reach["bridge_top10"],
        "bridge_reach_top10_fraction": reach["bridge_top10"] / max(reach["gold_outside_porter20"], 1),
        "extra_ebr_candidates_mean": reach["extra_candidates"] / n,
        "extra_ebr_candidates_fraction_of_top100": reach["extra_candidates"]
        / max(reach["ebr_top100_total"], 1),
    }
    latency_report = {
        "queries_timed": n,
        "porter_ms_per_query": (latency["porter_score_ms"] + latency["rank_porter_ms"]) / n,
        "ebr_ms_per_query": (
            latency["porter_score_ms"] + latency["expansion_ms"] + latency["fusion_ms"]
            + latency["rank_ebr_ms"]
        ) / n,
        "expansion_ms_per_query": latency["expansion_ms"] / n,
        "fusion_ms_per_query": latency["fusion_ms"] / n,
        "rank_porter_ms_per_query": latency["rank_porter_ms"] / n,
        "rank_bridge_ms_per_query": latency["rank_bridge_ms"] / n,
        "rank_entity_ms_per_query": latency["rank_entity_ms"] / n,
        "rank_ebr_ms_per_query": latency["rank_ebr_ms"] / n,
    }
    return rows, reach_report, latency_report


def summarize(rows, system):
    values = rows[system]
    out = {metric: sum(row[metric] for row in values) / len(values) for metric in METRICS}
    out["queries"] = len(values)
    return out


def slice_summary(rows, system):
    subsets = defaultdict(list)
    for row in rows[system]:
        subsets[row["slice"]].append(row)
    out = {}
    for name, subset in sorted(subsets.items()):
        summary = {metric: sum(row[metric] for row in subset) / len(subset) for metric in METRICS}
        summary["queries"] = len(subset)
        out[name] = summary
    return out


def all_slices(rows):
    return {system: slice_summary(rows, system) for system in SYSTEMS}


def paired_vs(rows, base, candidate, metrics=METRICS):
    return {
        metric: hotpot.paired(rows[base], rows[candidate], metric, resamples=1000, seed=1729)
        for metric in metrics
    }


def acceptance(datasets: dict) -> dict:
    """Pre-registered gate, applied mechanically to the frozen result."""
    positive_evidence = []
    ndcg_failures = []
    for name, block in datasets.items():
        paired = block["paired_vs_porter"]["ebr"]
        for metric in ("all@10", "recall@10"):
            if paired[metric]["ci95"][0] > 0:
                positive_evidence.append({
                    "dataset": name, "metric": metric,
                    "delta": paired[metric]["delta"], "ci95": paired[metric]["ci95"],
                })
        if paired["ndcg@10"]["ci95"][1] < -0.005:
            ndcg_failures.append({"dataset": name, **paired["ndcg@10"]})
    passed = bool(positive_evidence) and not ndcg_failures
    return {
        "rule": (
            "Keep the direction only if EBR has a positive CI on all@10 or recall@10 on at least one "
            "external dataset AND no nDCG@10 loss larger than -0.005 with a CI excluding that margin on "
            "either dataset."
        ),
        "positive_evidence": positive_evidence,
        "ndcg_failures": ndcg_failures,
        "passed": passed,
        "verdict": (
            "EBR transfers at the pre-registered constants; keep the direction."
            if passed else "HotpotQA specialist; do not generalize EBR"
        ),
    }
