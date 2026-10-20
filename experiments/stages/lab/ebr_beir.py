"""Entity-Bridge Retrieval over BEIR-style corpora.

Generic adapter for the EBR method in `hotpot.py`. The retrieval logic is the same
one that was measured on HotpotQA: Porter BM25 seeds the first hop, high-idf
entities from those seeds become bridge terms for the second hop, per-query-entity
postings form the comparison channel, and fixed RRF (k=60) fuses the three lists.
`hotpot.extract_entities`, `hotpot.normalize_entity`, `hotpot.entity_idf`,
`hotpot.STOP_ENTITIES` and `fusion.rrf` are reused unchanged.

What differs is the data layer:
  - documents and queries are read from BEIR `corpus.jsonl` / `queries.jsonl`,
    because the EBR channels need raw text (case carries the entity signal);
  - documents are addressed by integer position instead of doc-id string, because
    Touche has 382k documents and a dict of sets of strings is too large;
  - the pre-registered constants below are the HotpotQA ones, unchanged.

Constants were fixed on HotpotQA before this file existed (HOTPOT.md). Nothing in
this module is fitted to the single-hop sets it is now run on.
"""
from __future__ import annotations

import json
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

import harness
import hotpot
from fusion import order, rrf

SEEDS = 20          # first-hop BM25 seeds
BRIDGES = 20        # bridge entities kept per query
MAX_DF_FRAC = 0.05  # hub-entity cap, as a fraction of the corpus
RRF_K = 60          # fusion constant, from HOTPOT.md


class Corpus:
    """The index plus everything the EBR channels read: raw text and entity postings."""

    def __init__(self, name, split, ix, raw_queries, queries, qrels,
                 title_index, doc_entities, entity_docs, entity_df):
        self.name = name
        self.split = split
        self.ix = ix
        self.raw_queries = raw_queries
        self.queries = queries
        self.qrels = qrels
        self.title_index = title_index
        self.doc_entities = doc_entities      # position -> set[str]
        self.entity_docs = entity_docs        # entity -> array('i') of positions
        self.entity_df = entity_df            # entity -> document frequency

    def stats(self) -> dict:
        return {
            "docs": self.ix.N,
            "queries": len(self.qrels),
            "entities": len(self.entity_docs),
            "mean_entities_per_doc": sum(len(e) for e in self.doc_entities) / max(self.ix.N, 1),
        }


def source(name: str) -> tuple[Path, str]:
    """The dataset folder and its default split, from the harness registries."""
    for registry in (harness.DATASETS, harness.FRESH, harness.EXTRA):
        if name in registry:
            return registry[name]
    raise KeyError(f"unknown dataset {name!r}")


def read_raw_queries(path: Path) -> dict[str, str]:
    queries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        queries[str(row["_id"])] = row.get("text") or ""
    return queries


def build_entities(ix: harness.Index, corpus: dict[str, tuple[str, str]]):
    """Entity postings for the whole corpus. One set of entities per document."""
    doc_entities: list[set[str]] = []
    entity_docs: dict[str, array] = {}
    title_index: dict[str, str] = {}
    pool: dict[str, str] = {}          # string pool: entity names repeat across documents
    intern = pool.setdefault

    for position, doc_id in enumerate(ix.doc_ids):
        title, body = corpus[doc_id]
        entities = {intern(entity, entity) for entity in hotpot.extract_entities(title + " " + body)}
        title_key = hotpot.normalize_entity(title)
        if title_key:
            entities.add(intern(title_key, title_key))
            title_index[title_key] = doc_id
        doc_entities.append(entities)
        for entity in entities:
            entity_docs.setdefault(entity, array("i")).append(position)

    return title_index, doc_entities, entity_docs, {e: len(docs) for e, docs in entity_docs.items()}


def load(name: str, split: str | None = None) -> Corpus:
    folder, default_split = source(name)
    split = split or default_split

    ix, _, _ = harness.load(name)                      # cached tokenised index, split-independent
    corpus = harness._read_corpus(folder / "corpus.jsonl")
    qrels = harness._read_qrels(folder / "qrels" / f"{split}.tsv")
    raw = read_raw_queries(folder / "queries.jsonl")

    raw_queries = {qid: raw[qid] for qid in qrels if qid in raw}
    queries = {qid: harness.tokenize(text) for qid, text in raw_queries.items()}
    title_index, doc_entities, entity_docs, entity_df = build_entities(ix, corpus)

    return Corpus(name, split, ix, raw_queries, queries, qrels,
                  title_index, doc_entities, entity_docs, entity_df)


def query_entities(data: Corpus, qid: str) -> set[str]:
    """Capitalised spans plus any question substring that is a document title."""
    question = data.raw_queries[qid]
    found = set(hotpot.extract_entities(question))
    words = question.split()
    for size in range(1, 7):
        for start in range(0, len(words) - size + 1):
            key = hotpot.normalize_entity(" ".join(words[start:start + size]))
            if key in data.title_index:
                found.add(key)
    return found


def per_entity_scores(data: Corpus, entities: set[str]) -> dict[int, float]:
    """Sparse channel: documents that mention each query entity, weighted by entity idf."""
    scores: Counter = Counter()
    n = data.ix.N
    for entity in entities:
        df = data.entity_df.get(entity, 0)
        if df < 1 or df > MAX_DF_FRAC * n:
            continue
        weight = hotpot.entity_idf(n, df)
        for doc in data.entity_docs.get(entity, ()):
            scores[doc] += weight
    return scores


def bridge_scores(data: Corpus, qid: str, seed_scores: np.ndarray,
                  top_seeds: int = SEEDS, top_bridges: int = BRIDGES):
    """Second hop: entities in the first-hop seeds that the question does not name."""
    n = data.ix.N
    question_entities = query_entities(data, qid)
    bridges: Counter = Counter()
    for rank, doc in enumerate(order(seed_scores, top_seeds)):
        for entity in data.doc_entities[doc]:
            if entity in question_entities:
                continue
            df = data.entity_df.get(entity, 0)
            if df < 1 or df > MAX_DF_FRAC * n:
                continue
            bridges[entity] += hotpot.entity_idf(n, df) / (rank + 1)

    kept = bridges.most_common(top_bridges)
    scores: Counter = Counter()
    for entity, weight in kept:
        for doc in data.entity_docs.get(entity, ()):
            scores[doc] += weight
    return scores, kept


def dense(data: Corpus, scores: dict[int, float]) -> np.ndarray:
    out = np.zeros(data.ix.N, dtype=float)
    for doc, score in scores.items():
        out[doc] = score
    return out


def channels(data: Corpus, qid: str, porter_scores: np.ndarray) -> dict[str, np.ndarray]:
    """Every system that run_ebr_beir.py reports, from one set of channels."""
    entities = query_entities(data, qid)
    entity_scores = dense(data, per_entity_scores(data, entities))
    bridge, _ = bridge_scores(data, qid, porter_scores)
    bridge_dense = dense(data, bridge)
    return {
        "entity": entity_scores,
        "bridge": bridge_dense,
        "rrf_porter_bridge": rrf(data.ix, [porter_scores, bridge_dense]),
        "rrf_porter_entity": rrf(data.ix, [porter_scores, entity_scores]),
        "rrf_porter_bridge_entity": rrf(data.ix, [porter_scores, bridge_dense, entity_scores]),
    }
