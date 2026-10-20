"""Entity-Bridge Retrieval (EBR) on HotpotQA.

No embeddings, no LLM calls. BM25 retrieves the first hop. High-surprisal entities
from those seeds become bridge terms for the second hop. Fixed RRF combines the
channels. Evaluation is supporting-fact recall, including the both-facts metric
that a single-document ranker can miss.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq

import harness
from fusion import rrf
from lean_units import LeanUnitBM25
from channels import porter

CAP_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&'’.\-]*)"
    r"(?:\s+(?:(?:of|the|and|de|van|von|del|la|le|da|di|du|for|in|on)\s+)?"
    r"[A-Z][A-Za-z0-9&'’.\-]*){0,5}"
)
STOP_ENTITIES = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "we", "i", "his", "her",
    "their", "our", "my", "you", "your", "in", "on", "at", "for", "from", "by",
    "with", "as", "during", "after", "before", "however", "although", "because",
    "while", "when", "where", "which", "who", "what", "how", "why", "was", "were",
    "is", "are", "be", "been", "being", "did", "does", "do", "has", "have", "had",
    "one", "two", "first", "second", "new", "also", "both", "same", "many", "most",
}


def normalize_entity(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    value = value.strip(" .,;:()[]\"'")
    return value


def extract_entities(text: str) -> set[str]:
    found = set()
    for match in CAP_RE.finditer(text):
        value = normalize_entity(match.group(0))
        if not value or value in STOP_ENTITIES:
            continue
        if len(value) < 2 or len(value.split()) > 6:
            continue
        if value.isdigit():
            continue
        found.add(value)
    return found


def load(path: str):
    rows = pq.read_table(path).to_pylist()
    paragraphs: dict[str, tuple[str, str]] = {}
    for row in rows:
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            if title not in paragraphs:
                paragraphs[title] = (title, " ".join(sentences))

    ix = harness.Index(paragraphs)
    raw_queries, queries, qrels, meta = {}, {}, {}, {}
    for row in rows:
        qid = str(row["id"])
        raw_queries[qid] = row["question"]
        queries[qid] = harness.tokenize(row["question"])
        supports = sorted(set(row["supporting_facts"]["title"]))
        qrels[qid] = {title: 1 for title in supports}
        meta[qid] = {
            "type": row["type"],
            "level": row["level"],
            "context": list(row["context"]["title"]),
            "support": supports,
        }

    title_index = {normalize_entity(title): title for title in ix.doc_ids}
    paragraph_entities: dict[str, set[str]] = {}
    entity_paragraphs: dict[str, set[str]] = defaultdict(set)
    for doc_id in ix.doc_ids:
        title, body = paragraphs[doc_id]
        entities = extract_entities(title + " " + body)
        entities.add(normalize_entity(title))
        paragraph_entities[doc_id] = entities
        for entity in entities:
            entity_paragraphs[entity].add(doc_id)

    return {
        "ix": ix,
        "position": {doc: i for i, doc in enumerate(ix.doc_ids)},
        "raw_queries": raw_queries,
        "queries": queries,
        "qrels": qrels,
        "meta": meta,
        "title_index": title_index,
        "paragraph_entities": paragraph_entities,
        "entity_paragraphs": dict(entity_paragraphs),
        "entity_df": {entity: len(docs) for entity, docs in entity_paragraphs.items()},
    }


def query_entities(question: str, title_index: dict[str, str]) -> set[str]:
    found = extract_entities(question)
    words = question.split()
    for size in range(1, 7):
        for start in range(0, len(words) - size + 1):
            key = normalize_entity(" ".join(words[start:start + size]))
            if key in title_index:
                found.add(key)
    return found


def entity_idf(n: int, df: int) -> float:
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def per_entity_scores(index, qid, entities):
    """A sparse channel that retrieves paragraphs containing each query entity."""
    scores = Counter()
    entity_df = index["entity_df"]
    for entity in entities:
        df = entity_df.get(entity, 0)
        if df < 1 or df > 0.05 * index["ix"].N:
            continue
        weight = entity_idf(index["ix"].N, df)
        for doc in index["entity_paragraphs"].get(entity, ()):
            scores[doc] += weight
    return scores


def bridge_scores(index, qid, bm25_scores, top_seeds=20, top_bridges=20, max_df_frac=0.05):
    ix = index["ix"]
    seeds = harness.rank(ix, bm25_scores, top_seeds)
    q_entities = query_entities(index["raw_queries"][qid], index["title_index"])
    bridges = Counter()
    for rank, doc in enumerate(seeds):
        for entity in index["paragraph_entities"].get(doc, ()):
            if entity in q_entities:
                continue
            df = index["entity_df"].get(entity, 0)
            if df < 1 or df > max_df_frac * ix.N:
                continue
            bridges[entity] += entity_idf(ix.N, df) / (rank + 1)

    top = bridges.most_common(top_bridges)
    paragraphs = Counter()
    for entity, weight in top:
        for doc in index["entity_paragraphs"].get(entity, ()):
            paragraphs[doc] += weight
    return paragraphs, top


def dense(ix, scores: Counter, position=None):
    import numpy as np
    out = np.zeros(ix.N, dtype=float)
    if position is None:
        position = {doc_id: i for i, doc_id in enumerate(ix.doc_ids)}
    for doc, score in scores.items():
        out[position[doc]] = score
    return out


def metrics(ranked, relevant, k=10):
    rel = set(relevant)
    out = {}
    for cutoff in (2, 5, 10):
        top = ranked[:cutoff]
        hits = rel & set(top)
        out[f"recall@{cutoff}"] = len(hits) / len(rel)
        out[f"both@{cutoff}"] = 1.0 if rel <= set(top) else 0.0
    top = ranked[:k]
    dcg = sum(1 / math.log2(i + 2) for i, doc in enumerate(top) if doc in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    first = next((i + 1 for i, doc in enumerate(top) if doc in rel), None)
    out[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    out["mrr@10"] = 1 / first if first else 0.0
    return out


def evaluate(index, split: str, systems: dict):
    rows = defaultdict(list)
    for qid, qrels in index["qrels"].items():
        if not in_split(qid, split):
            continue
        for label, scorer in systems.items():
            scores = scorer(qid)
            ranked = harness.rank(index["ix"], scores, 1000)
            row = metrics(ranked, qrels)
            row["type"] = index["meta"][qid]["type"]
            rows[label].append(row)
    return rows


def in_split(qid: str, split: str) -> bool:
    bucket = int(hashlib.md5(qid.encode()).hexdigest(), 16) % 5
    return bucket == 0 if split == "dev" else bucket != 0


def paired(base_rows, candidate_rows, metric: str, resamples: int = 1000, seed: int = 1729):
    import numpy as np
    base = np.array([row[metric] for row in base_rows], dtype=float)
    candidate = np.array([row[metric] for row in candidate_rows], dtype=float)
    diff = candidate - base
    if diff.size == 0:
        return {"delta": 0.0, "ci95": [0.0, 0.0], "wtl": [0, 0, 0]}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, diff.size, size=(resamples, diff.size))
    means = diff[samples].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "delta": float(diff.mean()),
        "ci95": [float(low), float(high)],
        "wtl": [int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())],
    }


def paired_report(base_rows, candidate_rows, metrics=("ndcg@10", "both@10", "both@5", "recall@10")):
    return {metric: paired(base_rows, candidate_rows, metric) for metric in metrics}


def summarize(rows):
    out = {}
    for label, values in rows.items():
        out[label] = {}
        for key in values[0]:
            if key in ("type", "qid"):
                continue
            out[label][key] = sum(row[key] for row in values) / len(values)
        for kind in ("bridge", "comparison"):
            subset = [row for row in values if row["type"] == kind]
            out[label][kind] = {
                key: sum(row[key] for row in subset) / max(len(subset), 1)
                for key in values[0] if key not in ("type", "qid")
            }
        out[label]["queries"] = len(values)
    return out
