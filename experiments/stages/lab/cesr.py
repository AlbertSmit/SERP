"""CESR-1: Constrained Evidence-Set Retrieval (oracle-constraint ceiling test).

The objective here is **not** document relevance. Every query has a set of gold
constraints (entities and relation cues). A candidate paragraph covers a constraint
by containing it. The retrieval problem is to choose the paragraphs that cover the
constraint set under a token budget:

    maximise sum(weight of covered constraints)
    subject to sum(dl) <= budget

BM25 is a candidate generator only. The selector is deterministic greedy submodular
maximisation over the candidate pool:

    marginal_constraint_weight / max(dl + 10, 1)**0.35
      - MU * max_jaccard(candidate, selected)      # entity-set Jaccard
      + PAIR_BONUS * pair_weight                   # both endpoints of one triple

Constraints are gold-derived: 2Wiki `evidences` triples, MuSiQue
`question_decomposition` (sub-question text plus its answer; `paragraph_support_idx`
is never read). That makes this an **upper bound**, not a deployable system: the
constraint extractor is assumed perfect.

The EBR composition (Porter BM25, untyped bridge channel, per-entity channel, fixed
RRF k=60) is imported read-only from `hotpot.py` / `multihop_transfer.py`.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

import fusion
import hotpot
from multihop_transfer import expansion_channels, fuse, paragraph_key

# ------------------------------------------------------------------ frozen constants

BUDGETS = (500, 1000, 2000)
POOL_DEPTH = 100
MU = 0.2
COST_EXPONENT = 0.35
PAIR_BONUS = 0.25
RESAMPLES = 1000
BOOTSTRAP_SEED = 1729

SYSTEMS = ("porter_trunc", "ebr_trunc", "cesr", "cesr_coverage_first")
METRICS = (
    "all@5", "all@10", "recall", "constraint_coverage", "redundancy",
    "tokens", "selected", "ndcg@10", "mrr",
)

# Pre-declared, frozen relation synonym expansion. One entry per frequent relation in
# the two datasets; nothing here was chosen after looking at retrieval results.
RELATION_SYNONYMS = {
    "date of birth": ("born",),
    "date of death": ("died",),
    "place of birth": ("born in", "born at"),
    "place of death": ("died in", "died at"),
    "publication date": ("published",),
    "country of citizenship": ("citizen of", "nationality"),
    "country of origin": ("origin",),
    "educated at": ("studied at", "graduated from"),
    "place of burial": ("buried",),
    "spouse": ("wife", "husband", "married"),
    "director": ("directed",),
    "performer": ("performed", "singer"),
    "founded by": ("founder",),
    "headquarters": ("headquartered",),
    "record label": ("label",),
}

# MuSiQue sub-questions are short ("Green >> performer"); when there is no explicit
# relation slot we look for these pre-declared cues in the sub-question text.
MUSIQUE_CUE_VOCAB = (
    "spouse", "father", "mother", "child", "sibling", "director", "performer",
    "composer", "producer", "publisher", "author", "country", "capital",
    "located in", "date of birth", "date of death", "founded by", "headquarters",
    "occupation", "genre", "record label", "educated at", "border", "followed by",
    "member",
)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_entity(value: str) -> str:
    """Same normalisation the EBR entity index uses, so constraints and docs agree."""
    return hotpot.normalize_entity(value)


# ------------------------------------------------------------------- raw text / specs


def load_2wiki_texts(path: str):
    """(doc_key -> (title, body)) with the same first-occurrence rule as load_2wiki."""
    rows = pq.read_table(path).to_pylist()
    texts: dict[str, tuple[str, str]] = {}
    for row in rows:
        for title, sentences in json.loads(row["context"]):
            if title in texts:
                continue
            texts[title] = (title, " ".join(sentences))
    return texts


def load_2wiki_constraints(path: str, title_index=None):
    """Gold constraints from `evidences`: entity constraints for subject/object,
    one relation constraint per distinct relation, and endpoint pairs per triple."""
    del title_index  # 2Wiki constraints are already named entities
    rows = pq.read_table(path).to_pylist()
    specs = {}
    for row in rows:
        qid = str(row["_id"])
        entities, relations, pairs = [], [], []
        for subject, relation, obj in json.loads(row["evidences"]):
            entities.extend((str(subject), str(obj)))
            relations.append(str(relation))
            pairs.append((str(subject), str(obj)))
        specs[qid] = {"entities": entities, "relations": relations, "pairs": pairs}
    return specs


def load_musique_texts(path: str):
    texts: dict[str, tuple[str, str]] = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        for paragraph in row["paragraphs"]:
            key = paragraph_key(paragraph["title"], paragraph["paragraph_text"])
            texts.setdefault(key, (paragraph["title"], paragraph["paragraph_text"]))
    return texts


def resolve_decomposition(parts):
    """Replace `#k` with the answer of step k. `paragraph_support_idx` is never read."""
    answers = [str(part["answer"]) for part in parts]
    resolved = []
    for text in (str(part["question"]) for part in parts):
        for step in range(1, len(parts) + 1):
            text = re.sub(rf"#{step}\b", answers[step - 1], text)
        resolved.append(text)
    return resolved


def decomposition_relations(question: str):
    text = normalize_ws(question.lower())
    if ">>" in text:
        cue = normalize_entity(text.split(">>", 1)[1])
        return [cue] if len(cue) >= 3 else []
    return [cue for cue in MUSIQUE_CUE_VOCAB if cue in text]


def load_musique_constraints(path: str, title_index: dict[str, str]):
    """Entity constraints from each sub-question (plus its answer) and relation cues.

    `paragraph_support_idx` is deliberately ignored: it names the gold paragraph
    directly and would turn the experiment into a lookup.
    """
    specs = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row["id"])
        parts = row["question_decomposition"]
        questions = resolve_decomposition(parts)
        entities, relations, pairs = [], [], []
        for part, question in zip(parts, questions):
            answer = str(part["answer"])
            step_entities = sorted(hotpot.query_entities(question, title_index))
            entities.extend(step_entities)
            entities.append(answer)
            for entity in step_entities:
                pairs.append((entity, answer))
            relations.extend(decomposition_relations(question))
        specs[qid] = {"entities": entities, "relations": relations, "pairs": pairs}
    return specs


# ----------------------------------------------------------------------- corpus view


class Corpus:
    """Normalised paragraph text and entity sets, indexed by document position."""

    def __init__(self, index, texts: dict[str, tuple[str, str]]):
        self.ix = index["ix"]
        self.position = index["position"]
        missing = [doc for doc in self.ix.doc_ids if doc not in texts]
        if missing:
            raise SystemExit(f"cesr: {len(missing)} documents missing raw text")
        self.norm = [
            normalize_ws((texts[doc][0] + " " + texts[doc][1]).lower()) for doc in self.ix.doc_ids
        ]
        self.entities = [index["paragraph_entities"][doc] for doc in self.ix.doc_ids]
        self.lengths = {doc: self.ix.dl[pos] for doc, pos in self.position.items()}


# ------------------------------------------------------------------------ constraints


def build_query_constraints(spec, index):
    """Weighted constraints for one query plus the evidence pairs used by the bonus.

    Entity weights are the EBR entity IDF where the normalised string is in the entity
    index, else 1.0. Relation constraints always weigh 1.0. All weights are normalised
    to sum to one, so `marginal weight` and `pair weight` share units.
    """
    entity_df = index["entity_df"]
    total_docs = index["ix"].N
    constraints, entity_index = [], {}
    for value in spec["entities"]:
        key = normalize_entity(value)
        if not key or key in entity_index:
            continue
        df = entity_df.get(key, 0)
        weight = hotpot.entity_idf(total_docs, df) if df else 1.0
        entity_index[key] = len(constraints)
        constraints.append({
            "kind": "entity", "text": key, "label": f"entity:{key}",
            "weight": weight, "phrases": (key,),
        })
    labels = {c["label"] for c in constraints}
    for value in spec["relations"]:
        key = normalize_entity(value)
        if len(key) < 3:
            continue
        label = f"relation:{key}"
        if label in labels:
            continue
        labels.add(label)
        phrases = (key,) + RELATION_SYNONYMS.get(key, ())
        constraints.append({
            "kind": "relation", "text": key, "label": label, "weight": 1.0,
            "phrases": tuple(dict.fromkeys(phrases)),
        })
    total = sum(c["weight"] for c in constraints)
    if total <= 0:
        return [], []
    for constraint in constraints:
        constraint["weight"] /= total
    pairs = []
    for subject, obj in spec["pairs"]:
        left = entity_index.get(normalize_entity(subject))
        right = entity_index.get(normalize_entity(obj))
        if left is not None and right is not None and left != right:
            pairs.append((left, right))
    return constraints, list(dict.fromkeys(pairs))


def constraint_covered(corpus, position: int, constraint) -> bool:
    if constraint["kind"] == "entity" and constraint["text"] in corpus.entities[position]:
        return True
    text = corpus.norm[position]
    return any(phrase in text for phrase in constraint["phrases"])


def coverage_matrix(corpus, positions, constraints):
    matrix = np.zeros((len(positions), len(constraints)), dtype=bool)
    for column, constraint in enumerate(constraints):
        for row, position in enumerate(positions):
            matrix[row, column] = constraint_covered(corpus, position, constraint)
    return matrix


# -------------------------------------------------------------------------- selection


def pair_bonus_vector(matrix, pairs, weights, n_candidates):
    bonus = np.zeros(n_candidates, dtype=float)
    for left, right in pairs:
        covered = matrix[:, left] & matrix[:, right]
        bonus[covered] += PAIR_BONUS * (weights[left] + weights[right])
    return bonus


def greedy_select(pool_positions, lengths, matrix, weights, bonus, entity_sets,
                  labels, budget, exponent=COST_EXPONENT):
    """Greedy submodular maximisation with a redundancy penalty and a pair bonus.

    Stops when the constraint set is fully covered, no remaining candidate fits the
    budget, or the best score is non-positive. Returns the selection and per-step
    provenance.
    """
    n, m = matrix.shape
    float_matrix = matrix.astype(float)
    taken = np.zeros(n, dtype=bool)
    covered = np.zeros(m, dtype=bool)
    residual = float_matrix @ weights
    redundancy = np.zeros(n, dtype=float)
    cost = np.power(np.maximum(lengths + 10.0, 1.0), exponent)
    sizes = [len(entities) for entities in entity_sets]
    used, selected, steps = 0, [], []

    while not covered.all():
        fits = ~taken
        if np.isfinite(budget):
            fits = fits & (used + lengths <= budget)
        if not fits.any():
            break
        score = np.full(n, -np.inf)
        score[fits] = residual[fits] / cost[fits] - MU * redundancy[fits] + bonus[fits]
        best = int(np.argmax(score))
        if not score[best] > 0:
            break

        newly = matrix[best] & ~covered
        covered |= matrix[best]
        residual = residual - float_matrix[:, newly] @ weights[newly]
        selected.append(best)
        steps.append({
            "index": best,
            "position": pool_positions[best],
            "gain": float(weights[newly].sum()),
            "score": float(score[best]),
            "tokens": int(lengths[best]),
            "redundancy": float(redundancy[best]),
            "covers": [labels[i] for i in np.flatnonzero(newly)],
        })
        used += int(lengths[best])
        taken[best] = True

        chosen_entities, chosen_size = entity_sets[best], sizes[best]
        for other in range(n):
            if taken[other]:
                continue
            shared = len(chosen_entities & entity_sets[other])
            if shared:
                jaccard = shared / (chosen_size + sizes[other] - shared)
                if jaccard > redundancy[other]:
                    redundancy[other] = jaccard

    return selected, steps, covered


def fill_by_rank(selected, ranked_ids, lengths_of, budget, used):
    """Truncate a fused ranked list into the remaining budget (skip-and-continue)."""
    chosen = set(selected)
    for doc in ranked_ids:
        if doc in chosen:
            continue
        length = lengths_of(doc)
        if used + length > budget:
            continue
        selected.append(doc)
        chosen.add(doc)
        used += length
    return used


def pool_from_channels(ix, channels, depth=POOL_DEPTH):
    """Union of the per-channel top-`depth`, with membership provenance."""
    pool, provenance = [], {}
    for label, scores in zip(("porter", "bridge", "entity"), channels):
        for doc in fusion.ranked_ids(ix, scores, depth):
            if doc not in provenance:
                provenance[doc] = [label]
                pool.append(doc)
            else:
                provenance[doc].append(label)
    return pool, provenance


# ---------------------------------------------------------------------------- metrics


def selection_metrics(selected_docs, relevant, covered, weights, entity_sets, tokens):
    rel = set(relevant)
    picked = set(selected_docs)
    out = {}
    for cutoff in (5, 10):
        out[f"all@{cutoff}"] = 1.0 if rel <= set(selected_docs[:cutoff]) else 0.0
    out["recall"] = len(rel & picked) / len(rel) if rel else 0.0
    out["constraint_coverage"] = float(weights[covered].sum())
    pairs = [
        len(entity_sets[i] & entity_sets[j]) / max(len(entity_sets[i] | entity_sets[j]), 1)
        for i in range(len(entity_sets)) for j in range(i + 1, len(entity_sets))
    ]
    out["redundancy"] = float(np.mean(pairs)) if pairs else 0.0
    dcg = sum(1 / np.log2(i + 2) for i, doc in enumerate(selected_docs[:10]) if doc in rel)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(rel), 10)))
    out["ndcg@10"] = float(dcg / ideal) if ideal else 0.0
    first = next((i + 1 for i, doc in enumerate(selected_docs) if doc in rel), None)
    out["mrr"] = 1 / first if first else 0.0
    out["tokens"] = float(tokens)
    out["selected"] = float(len(selected_docs))
    return out


# --------------------------------------------------------------------------- evaluate


def evaluate(index, texts, specs, limit=None, sample_queries=25):
    ix = index["ix"]
    corpus = Corpus(index, texts)
    from channels import porter as porter_transform
    from lean_units import LeanUnitBM25

    porter_bm25 = LeanUnitBM25(ix, porter_transform)
    rows = {system: {budget: [] for budget in BUDGETS} for system in SYSTEMS}
    latency = Counter()
    coverage_stats = Counter()
    provenance = {budget: {"channel_sources": Counter(), "steps": Counter(),
                           "samples": []} for budget in BUDGETS}
    qids = list(index["qrels"])
    if limit:
        qids = qids[:limit]
    sample_stride = max(len(qids) // sample_queries, 1)

    for order, qid in enumerate(qids):
        relevant = index["qrels"][qid]
        constraints, pairs = build_query_constraints(specs[qid], index)
        weights = np.array([c["weight"] for c in constraints], dtype=float)
        labels = [c["label"] for c in constraints]
        slice_name = index["meta"][qid]["slice"]

        started = time_start()
        porter_scores = porter_bm25.score(index["queries"][qid])
        bridge_dense, entity_dense = expansion_channels(index, qid, porter_scores)
        ebr_scores = fuse(index, porter_scores, bridge_dense, entity_dense)
        pool, pool_provenance = pool_from_channels(
            ix, (porter_scores, bridge_dense, entity_dense))
        latency["retrieval_ms"] += time_split(started)

        positions = [corpus.position[doc] for doc in pool]
        lengths = np.array([ix.dl[position] for position in positions], dtype=float)
        entity_sets = [corpus.entities[position] for position in positions]
        gold = set(relevant)
        coverage_stats["queries"] += 1
        coverage_stats["gold"] += len(gold)
        coverage_stats["gold_in_pool"] += len(gold & set(pool))

        started = time_start()
        matrix = coverage_matrix(corpus, positions, constraints)
        bonus = pair_bonus_vector(matrix, pairs, weights, len(pool))
        latency["constraint_ms"] += time_split(started)

        achievable = matrix.any(axis=0) if len(constraints) else np.zeros(0, dtype=bool)
        coverage_stats["achievable_weight"] += float(weights[achievable].sum()) if len(achievable) else 0.0
        coverage_stats["queries_fully_coverable"] += int(achievable.all()) if len(achievable) else 0
        coverage_stats["queries_all_gold_in_pool"] += int(gold <= set(pool))

        porter_ranked = fusion.ranked_ids(ix, porter_scores, fusion.DEPTH)
        ebr_ranked = fusion.ranked_ids(ix, ebr_scores, fusion.DEPTH)
        porter_trunc = truncate_all(porter_ranked, ix, corpus, BUDGETS)
        ebr_trunc = truncate_all(ebr_ranked, ix, corpus, BUDGETS)

        # The coverage-first selection ignores cost, so it does not depend on the budget.
        started = time_start()
        first, _, first_covered = greedy_select(
            positions, lengths, matrix, weights, bonus, entity_sets, labels,
            np.inf, exponent=0.0)
        first_base = [pool[i] for i in first]
        first_used_base = int(sum(lengths[i] for i in first))
        latency["coverage_first_ms"] += time_split(started)

        for budget in BUDGETS:
            started = time_start()
            selected, steps, covered = greedy_select(
                positions, lengths, matrix, weights, bonus, entity_sets, labels, budget)
            latency["cesr_select_ms"] += time_split(started)

            started = time_start()
            first_docs = list(first_base)
            fill_by_rank(first_docs, ebr_ranked, lambda d: ix.dl[corpus.position[d]],
                         budget, first_used_base)
            latency["coverage_fill_ms"] += time_split(started)

            cesr_docs = [pool[i] for i in selected]
            for system, docs, cover in (
                ("porter_trunc", porter_trunc[budget], None),
                ("ebr_trunc", ebr_trunc[budget], None),
                ("cesr", cesr_docs, covered),
                ("cesr_coverage_first", first_docs, first_covered),
            ):
                if cover is None:
                    cover = np.zeros(len(constraints), dtype=bool)
                    for doc in docs:
                        position = corpus.position[doc]
                        for column, constraint in enumerate(constraints):
                            cover[column] |= constraint_covered(corpus, position, constraint)
                tokens = sum(ix.dl[corpus.position[doc]] for doc in docs)
                row = selection_metrics(
                    docs, relevant, cover, weights,
                    [corpus.entities[corpus.position[doc]] for doc in docs], tokens)
                row["qid"] = qid
                row["slice"] = slice_name
                rows[system][budget].append(row)
                if system == "cesr":
                    fully_covered = bool(covered.all())
                    coverage_stats[f"full_coverage_{budget}"] += int(fully_covered)
                    coverage_stats[f"full_coverage_and_all10_{budget}"] += int(
                        fully_covered and row["all@10"] == 1.0)

            coverage_stats[f"cesr_full_coverage_{budget}"] += int(covered.all())
            coverage_stats[f"coverage_first_full_coverage_{budget}"] += int(first_covered.all())
            coverage_stats[f"coverage_first_overspend_{budget}"] += int(first_used_base > budget)

            if order % sample_stride == 0 and len(provenance[budget]["samples"]) < sample_queries:
                provenance[budget]["samples"].append({
                    "qid": qid,
                    "question": index["raw_queries"][qid],
                    "constraints": [
                        {"label": c["label"], "weight": round(c["weight"], 6),
                         "covered": bool(covered[i])}
                        for i, c in enumerate(constraints)
                    ],
                    "pairs": [[constraints[i]["label"], constraints[j]["label"]] for i, j in pairs],
                    "selected": [
                        {
                            "doc": pool[step["index"]][:180],
                            "tokens": step["tokens"],
                            "gain": round(step["gain"], 6),
                            "redundancy": round(step["redundancy"], 4),
                            "covers": step["covers"],
                            "channels": pool_provenance[pool[step["index"]]],
                        }
                        for step in steps
                    ],
                })
            for step in steps:
                channels = pool_provenance[pool[step["index"]]]
                provenance[budget]["channel_sources"]["+".join(channels)] += 1
                provenance[budget]["steps"][len(step["covers"])] += 1

    n = max(len(qids), 1)
    reach = {
        "queries": len(qids),
        "gold_paragraphs": coverage_stats["gold"],
        "gold_in_candidate_pool": coverage_stats["gold_in_pool"],
        "candidate_recall_ceiling": coverage_stats["gold_in_pool"] / max(coverage_stats["gold"], 1),
        "queries_with_all_gold_in_pool": coverage_stats["queries_all_gold_in_pool"],
        "queries_with_all_gold_in_pool_rate": coverage_stats["queries_all_gold_in_pool"] / n,
        "mean_achievable_constraint_coverage": coverage_stats["achievable_weight"] / n,
        "queries_fully_coverable_from_pool_rate": coverage_stats["queries_fully_coverable"] / n,
        "cesr_full_coverage_queries": {
            budget: coverage_stats[f"cesr_full_coverage_{budget}"] for budget in BUDGETS
        },
        "cesr_full_coverage_rate": {
            budget: coverage_stats[f"cesr_full_coverage_{budget}"] / n for budget in BUDGETS
        },
        "coverage_first_full_coverage_rate": {
            budget: coverage_stats[f"coverage_first_full_coverage_{budget}"] / n for budget in BUDGETS
        },
        "coverage_first_overspend_rate": {
            budget: coverage_stats[f"coverage_first_overspend_{budget}"] / n for budget in BUDGETS
        },
        "all_at_10_given_full_coverage": {
            budget: (
                coverage_stats[f"full_coverage_and_all10_{budget}"]
                / coverage_stats[f"full_coverage_{budget}"]
                if coverage_stats[f"full_coverage_{budget}"] else 0.0
            )
            for budget in BUDGETS
        },
    }
    latency_report = {
        "retrieval_ms_per_query": latency["retrieval_ms"] / n,
        "constraint_ms_per_query": latency["constraint_ms"] / n,
        "cesr_select_ms_per_query": latency["cesr_select_ms"] / n / len(BUDGETS),
        "coverage_first_ms_per_query": latency["coverage_first_ms"] / n,
        "coverage_fill_ms_per_query": latency["coverage_fill_ms"] / n / len(BUDGETS),
    }
    return rows, reach, latency_report, provenance


def time_start():
    return time.perf_counter()


def time_split(started):
    return (time.perf_counter() - started) * 1000


def truncate_docs(ranked, ix, corpus, budget):
    """Rank-order truncation with the repo's skip-and-continue convention."""
    selected, used = [], 0
    for doc in ranked:
        length = ix.dl[corpus.position[doc]]
        if used + length > budget:
            continue
        selected.append(doc)
        used += length
    return selected


def truncate_all(ranked, ix, corpus, budgets):
    """One pass over the ranked list producing every budget's rank-order truncation."""
    selected = {budget: [] for budget in budgets}
    used = {budget: 0 for budget in budgets}
    largest = max(budgets)
    for doc in ranked:
        length = ix.dl[corpus.position[doc]]
        if used[largest] + length > largest:
            continue
        for budget in budgets:
            if used[budget] + length <= budget:
                selected[budget].append(doc)
                used[budget] += length
    return selected


# ------------------------------------------------------------------------- summaries


def summarize(rows):
    out = {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
    out["queries"] = len(rows)
    return out


def slice_summary(rows):
    subsets = {}
    for row in rows:
        subsets.setdefault(row["slice"], []).append(row)
    return {name: summarize(values) for name, values in sorted(subsets.items())}


def paired(base_rows, candidate_rows, metrics=METRICS):
    return {
        metric: hotpot.paired(base_rows, candidate_rows, metric,
                              resamples=RESAMPLES, seed=BOOTSTRAP_SEED)
        for metric in metrics
    }


def acceptance(datasets, token_ratio):
    """Pre-registered gate.

    CESR must beat EBR on all@10 at a matched budget on at least one budget for both
    datasets (delta > 0 and CI95 low > 0), or match all@10 (CI spans zero) with at
    least 20% fewer tokens on both datasets at that budget.
    """
    beats, matches = {}, {}
    for name, block in datasets.items():
        for budget in BUDGETS:
            report = block["paired_vs_ebr"][budget]["all@10"]
            low, high = report["ci95"]
            beats[(name, budget)] = report["delta"] > 0 and low > 0
            matches[(name, budget)] = low <= 0 <= high
    budget_beat = [b for b in BUDGETS if all(beats[(name, b)] for name in datasets)]
    budget_match = [
        b for b in BUDGETS
        if all(matches[(name, b)] for name in datasets)
        and all(token_ratio[(name, b)] <= 0.8 for name in datasets)
    ]
    passed = bool(budget_beat) or bool(budget_match)
    return {
        "rule": (
            "CESR beats EBR on all@10 at the same budget for at least one budget on both "
            "datasets (delta>0, CI95 low>0), OR matches all@10 at the same budget (CI spans "
            "zero) with >=20% fewer tokens on both datasets."
        ),
        "budgets_beating_ebr": budget_beat,
        "budgets_matching_with_fewer_tokens": budget_match,
        "tokens_ratio_cesr_over_ebr": {
            f"{name}@{budget}": token_ratio[(name, budget)]
            for name in datasets for budget in BUDGETS
        },
        "passed": passed,
        "verdict": (
            "CESR beats fusion under oracle constraints; the evidence-set objective is a real direction."
            if passed else
            "evidence-set objective does not beat fusion even with gold constraints"
        ),
    }
