"""CESR-code: Constrained Evidence-Set Retrieval over exact code-graph constraints.

CESR-1 (lab/cesr.py) asked the evidence-set question with *fuzzy* constraints:
capitalised prose entities and relation cues, matched by substring. It found that
the selector reaches the coverage ceiling and still misses evidence, because the
constraint set does not identify the gold set.

This is the same packing objective with the opposite kind of constraint. A code
graph has typed, exact relations -- CALLS, IMPORTS, EXPORTS, DEFINED_IN -- so a
constraint is not "the paragraph contains the string `performer`" but "this chunk
calls `tokenStem`" or "this chunk's file imports `./tokenize.js`". There is no
matching ambiguity: a chunk either satisfies a graph predicate or it does not.

Per query type the constraint set is built from the *query metadata*
(`type`, `symbols`, `module`, `chain`), never from `qrels`:

    same_file      CALLS(symbols[0])            + DEFINES(symbols[1])      + same-file joint
    import_call    CALLS(symbols[0])            + DEFINES(symbols[0])
                   + IMPORTS(module)                                       + caller-file joint
    two_hop        CALLS(chain[0])              + CALLS(chain[1])          + chain joint
    export_import  EXPORTS(file)                + DEFINES(symbols[0])
                   + IMPORTS(module)                                       + same-file joint
    calls_two      CALLS(symbols[0]) + CALLS(symbols[1])
                   + DEFINES(symbols[0]) + DEFINES(symbols[1])             + same-caller joint

`IMPORTS` is a file-level predicate: the set of chunks whose file imports the
module. The joint is what makes it a *caller-file* constraint rather than an
over-permissive one, exactly as CESR-1's pair bonus pinned the two endpoints of a
triple.

The selector is deterministic greedy coverage under a token budget:

    score = (new constraint weight + PATH_BONUS * new joints satisfied)
            / max(dl + 10, 1)**0.35
            - MU * max_symbol_jaccard(candidate, selected)

Weights are 1.0 per constraint, normalised to sum to one; the path bonus is a
fixed 0.25 per joint. Stop when the budget is exhausted, no candidate fits, or
the best score is non-positive.

The qrels of the host benchmark are graph-derived, so this is a **capability
test**: it measures whether retrieval can execute the relation, not human
relevance. See experiments/stages/CESR_CODE.md.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import code_graph as cg
import fusion
import harness

# --------------------------------------------------------------- frozen constants

BUDGETS = (500, 1000, 2000)
POOL_DEPTH = 100
MU = 0.2
COST_EXPONENT = 0.35
PATH_BONUS = 0.25
RESAMPLES = 1000
BOOTSTRAP_SEED = 1729

TYPE_ORDER = ("same_file", "import_call", "two_hop", "export_import", "calls_two")
METRICS = ("all@5", "all@10", "recall", "coverage", "tokens", "selected",
           "redundancy", "ndcg@10")
BASELINES = ("porter_trunc", "entity_trunc", "graph_trunc", "rrf_trunc")
CESR = "cesr"
POOL_SYSTEMS = BASELINES + (CESR,)


# ------------------------------------------------------------------------ corpus


@dataclass
class CodeCorpus:
    """Chunk-level view of the graph: ids, file, length, symbol set, export set."""

    ids: list[str]
    position: dict[str, int]
    file_of: dict[str, str]
    dl: dict[str, int]
    symbols: dict[str, frozenset[str]]
    export_chunks: frozenset[str]
    module_chunks: dict[str, frozenset[str]]
    callers: dict[str, frozenset[str]] = field(default_factory=dict)
    defined: dict[str, frozenset[str]] = field(default_factory=dict)
    calls: dict[str, frozenset[str]] = field(default_factory=dict)


def build_corpus(graph, ix) -> CodeCorpus:
    ids = [chunk["id"] for chunk in graph.chunks]
    position = {doc: i for i, doc in enumerate(ix.doc_ids)}
    file_of = {chunk["id"]: chunk["file"] for chunk in graph.chunks}
    dl = {doc: int(ix.dl[position[doc]]) for doc in ids}

    # chunk -> symbol set, for the redundancy penalty: defined + called + mentioned
    symbols: dict[str, set[str]] = {doc: set() for doc in ids}
    for doc, names in graph.chunk_defines.items():
        symbols[doc].update(names)
    for doc, names in graph.calls.items():
        symbols[doc].update(names)
    for symbol, holders in graph.mentions.items():
        for doc in holders:
            symbols[doc].add(symbol)

    export_chunks = frozenset(
        doc for doc in ids
        if graph.chunk_defines.get(doc, set()) & graph.exports.get(file_of[doc], set())
    )
    by_module: dict[str, set[str]] = defaultdict(set)
    for doc in ids:
        for module in graph.file_imports.get(file_of[doc], ()):
            by_module[module].add(doc)

    return CodeCorpus(
        ids=ids,
        position=position,
        file_of=file_of,
        dl=dl,
        symbols={doc: frozenset(names) for doc, names in symbols.items()},
        export_chunks=export_chunks,
        module_chunks={module: frozenset(docs) for module, docs in by_module.items()},
        callers={k: frozenset(v) for k, v in graph.callers.items()},
        defined={k: frozenset(v) for k, v in graph.defined.items()},
        calls={k: frozenset(v) for k, v in graph.calls.items()},
    )


# ------------------------------------------------------------------- constraints


@dataclass
class Constraints:
    labels: list[str]
    weights: np.ndarray
    covered_by: list[frozenset[str]]
    joints: list[dict]

    def __len__(self) -> int:
        return len(self.labels)


def _callers(corpus: CodeCorpus, symbol: str) -> frozenset[str]:
    return corpus.callers.get(symbol, frozenset())


def _defines(corpus: CodeCorpus, symbol: str) -> frozenset[str]:
    return corpus.defined.get(symbol, frozenset())


def _imports(corpus: CodeCorpus, module: str) -> frozenset[str]:
    return corpus.module_chunks.get(module, frozenset())


def query_constraints(corpus: CodeCorpus, query: dict) -> Constraints:
    """Exact coverage sets per constraint, built from query metadata only.

    `qrels` is never read here. A chunk satisfies a constraint when the graph
    says so: it is in `callers[symbol]`, `defined[symbol]`, `module_chunks[module]`
    or `export_chunks`.
    """
    kind = query["type"]
    specs: list[tuple[str, frozenset[str]]] = []
    joints: list[dict] = []

    if kind == "same_file":
        callee, anchor = query["symbols"]
        specs = [
            (f"CALLS({callee})", _callers(corpus, callee)),
            (f"DEFINES_SAME_FILE({anchor})", _defines(corpus, anchor)),
        ]
        joints = [{"label": f"same_file({callee},{anchor})", "members": [0, 1],
                   "kind": "same_file"}]

    elif kind == "import_call":
        symbol, module = query["symbols"][0], query["module"]
        specs = [
            (f"CALLS({symbol})", _callers(corpus, symbol)),
            (f"DEFINES({symbol})", _defines(corpus, symbol)),
            (f"IMPORTS({module})", _imports(corpus, module)),
        ]
        joints = [{"label": f"caller_file_imports({module})", "members": [0, 2],
                   "kind": "same_file"}]

    elif kind == "two_hop":
        callee, middle = query["chain"]
        specs = [
            (f"CALLS({callee})", _callers(corpus, callee)),
            (f"CALLS({middle})", _callers(corpus, middle)),
        ]
        joints = [{
            "label": f"chain({callee}->{middle})", "members": [0, 1], "kind": "chain",
            "defined_of": _defines(corpus, middle),
            "callers_of": _callers(corpus, middle),
        }]

    elif kind == "export_import":
        symbol, module = query["symbols"][0], query["module"]
        specs = [
            ("EXPORTS(file)", corpus.export_chunks),
            (f"DEFINES({symbol})", _defines(corpus, symbol)),
            (f"IMPORTS({module})", _imports(corpus, module)),
        ]
        joints = [{"label": "export_file_identity", "members": [0, 1, 2],
                   "kind": "same_file"}]

    elif kind == "calls_two":
        first, second = query["symbols"]
        specs = [
            (f"CALLS({first})", _callers(corpus, first)),
            (f"CALLS({second})", _callers(corpus, second)),
            (f"DEFINES({first})", _defines(corpus, first)),
            (f"DEFINES({second})", _defines(corpus, second)),
        ]
        joints = [{"label": f"same_caller({first},{second})", "members": [0, 1],
                   "kind": "same_chunk"}]

    else:
        raise ValueError(f"unknown query type: {kind}")

    labels = [label for label, _ in specs]
    n = max(len(labels), 1)
    weights = np.full(len(labels), 1.0 / n, dtype=float)
    return Constraints(labels=labels, weights=weights,
                       covered_by=[chunks for _, chunks in specs], joints=joints)


def joint_satisfied(corpus: CodeCorpus, joint: dict, selected: list[set[str]],
                    extra: tuple[str, set[int]] | None = None) -> bool:
    """Is the joint relation satisfied by the selected chunks (plus `extra`)?"""
    sets: list[set[str]] = []
    extra_id, extra_cols = extra if extra is not None else (None, set())
    for column in joint["members"]:
        held = set(selected[column])
        if extra_id is not None and column in extra_cols:
            held.add(extra_id)
        if not held:
            return False
        sets.append(held)

    if joint["kind"] == "same_file":
        files = [{corpus.file_of[doc] for doc in held} for held in sets]
        return bool(set.intersection(*files))
    if joint["kind"] == "same_chunk":
        return bool(set.intersection(*sets))
    if joint["kind"] == "chain":
        middle = sets[0] & joint["defined_of"]
        upper = sets[1] & joint["callers_of"]
        return bool(middle) and bool(upper)
    raise ValueError(f"unknown joint kind: {joint['kind']}")


def answer_candidates(corpus: CodeCorpus, query: dict,
                      constraints: Constraints) -> frozenset[str]:
    """Chunks that satisfy the query's *answer* conjunction, not just its parts.

    A diagnostic, never used by the selector: when this set has more than one
    member the constraint set does not identify a unique evidence set, and no
    coverage objective can be blamed for missing the gold chunk. This is the
    CESR-code analogue of CESR-1's "constraints do not pin the evidence" finding.
    """
    kind = query["type"]
    parts = constraints.covered_by
    if kind == "same_file":
        anchors = {corpus.file_of[doc] for doc in parts[1]}
        return frozenset(doc for doc in parts[0] if corpus.file_of[doc] in anchors)
    if kind == "import_call":
        return frozenset(parts[0] & parts[2])
    if kind == "two_hop":
        return frozenset(parts[0] & _defines(corpus, query["chain"][1]))
    if kind == "export_import":
        files = {corpus.file_of[doc] for doc in parts[1] & parts[2]}
        return frozenset(doc for doc in parts[0] if corpus.file_of[doc] in files)
    if kind == "calls_two":
        return frozenset(parts[0] & parts[1])
    raise ValueError(f"unknown query type: {kind}")


# ------------------------------------------------------------------------ pool


def build_pool(ix, graph, position, query):
    """Union of the three channel top-100s, with membership provenance."""
    tokens = cg.porter(cg.code_tokens(query["text"]))
    bm25 = harness.bm25_fast(ix, tokens)
    entity = cg.entity_scores(graph, position, query["text"])
    seeds = [ix.doc_ids[i] for i in fusion.order(bm25, cg.SEED_CHUNKS)]
    graph_channel = cg.graph_scores(graph, position, seeds)

    pool: list[str] = []
    provenance: dict[str, list[str]] = {}
    for label, scores in (("porter", bm25), ("entity", entity), ("graph", graph_channel)):
        for doc in fusion.ranked_ids(ix, scores, POOL_DEPTH):
            if doc not in provenance:
                provenance[doc] = [label]
                pool.append(doc)
            else:
                provenance[doc].append(label)
    channels = {
        "porter": fusion.ranked_ids(ix, bm25, fusion.DEPTH),
        "entity": fusion.ranked_ids(ix, entity, fusion.DEPTH),
        "graph": fusion.ranked_ids(ix, graph_channel, fusion.DEPTH),
        "rrf": fusion.ranked_ids(ix, fusion.rrf(ix, [bm25, entity, graph_channel]),
                                 fusion.DEPTH),
    }
    return pool, provenance, channels


# -------------------------------------------------------------------- selector


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / (len(left) + len(right) - shared) if shared else 0.0


def greedy_select(corpus: CodeCorpus, pool: list[str], constraints: Constraints,
                  budget: int) -> tuple[list[str], list[dict], np.ndarray]:
    """Greedy constrained coverage under a token budget.

    Returns the selection in order, per-step provenance, and the covered mask.
    The loop stops on budget exhaustion, when no positive marginal score remains,
    or when nothing fits.
    """
    n, m = len(pool), len(constraints)
    weights = constraints.weights
    lengths = np.array([corpus.dl[doc] for doc in pool], dtype=float)
    cost = np.power(np.maximum(lengths + 10.0, 1.0), COST_EXPONENT)

    covers = np.zeros((n, m), dtype=bool)
    for column, chunks in enumerate(constraints.covered_by):
        for row, doc in enumerate(pool):
            covers[row, column] = doc in chunks

    covered = np.zeros(m, dtype=bool)
    selected: list[set[str]] = [set() for _ in range(m)]
    satisfied = [joint_satisfied(corpus, joint, selected) for joint in constraints.joints]
    redundancy = np.zeros(n, dtype=float)
    taken = np.zeros(n, dtype=bool)
    used, order, steps = 0, [], []

    while True:
        fits = [row for row in range(n) if not taken[row] and used + lengths[row] <= budget]
        if not fits:
            break

        best, best_score, best_bonus, best_joints = None, 0.0, 0.0, []
        for row in fits:
            covers_row = {column for column in np.flatnonzero(covers[row])}
            gain = float(weights[covers[row] & ~covered].sum())
            bonus, newly = 0.0, []
            if constraints.joints:
                for index, joint in enumerate(constraints.joints):
                    if satisfied[index]:
                        continue
                    if joint_satisfied(corpus, joint, selected, (pool[row], covers_row)):
                        bonus += PATH_BONUS
                        newly.append(constraints.joints[index]["label"])
            if gain <= 0 and not newly:
                continue
            score = gain / cost[row] - MU * redundancy[row] + bonus
            if score > best_score:
                best, best_score, best_bonus, best_joints = row, score, bonus, newly

        if best is None:
            break

        doc = pool[best]
        new_columns = covers[best] & ~covered
        covered |= covers[best]
        for column in np.flatnonzero(covers[best]):
            selected[column].add(doc)
        used += int(lengths[best])
        taken[best] = True
        order.append(doc)
        for index, joint in enumerate(constraints.joints):
            if not satisfied[index]:
                satisfied[index] = joint_satisfied(corpus, joint, selected)
        steps.append({
            "doc": doc,
            "tokens": int(lengths[best]),
            "gain": float(weights[new_columns].sum()),
            "bonus": float(best_bonus),
            "score": float(best_score),
            "redundancy": float(redundancy[best]),
            "covers": [constraints.labels[column] for column in np.flatnonzero(new_columns)],
            "joins": best_joints,
        })

        for other in range(n):
            if taken[other]:
                continue
            jaccard = _jaccard(corpus.symbols[doc], corpus.symbols[pool[other]])
            if jaccard > redundancy[other]:
                redundancy[other] = jaccard

    return order, steps, covered


def truncate_all(ranked: list[str], corpus: CodeCorpus, budgets=BUDGETS) -> dict:
    """Rank-order truncation with the repo's skip-and-continue convention."""
    chosen = {budget: [] for budget in budgets}
    used = {budget: 0 for budget in budgets}
    largest = max(budgets)
    for doc in ranked:
        length = corpus.dl[doc]
        if used[largest] + length > largest:
            continue
        for budget in budgets:
            if used[budget] + length <= budget:
                chosen[budget].append(doc)
                used[budget] += length
    return chosen


# --------------------------------------------------------------------- metrics


def coverage_of(corpus: CodeCorpus, docs: list[str], constraints: Constraints) -> float:
    picked = set(docs)
    return float(sum(
        constraints.weights[column]
        for column, chunks in enumerate(constraints.covered_by)
        if picked & chunks
    ))


def selection_metrics(corpus, docs, relevant, constraints, tokens) -> dict:
    rel = set(relevant)
    picked = set(docs)
    out = {}
    for cutoff in (5, 10):
        out[f"all@{cutoff}"] = 1.0 if rel <= set(docs[:cutoff]) else 0.0
    out["recall"] = len(rel & picked) / len(rel) if rel else 0.0
    out["coverage"] = coverage_of(corpus, docs, constraints)
    out["tokens"] = float(tokens)
    out["selected"] = float(len(docs))
    pairs = [
        _jaccard(corpus.symbols[docs[i]], corpus.symbols[docs[j]])
        for i in range(len(docs)) for j in range(i + 1, len(docs))
    ]
    out["redundancy"] = float(np.mean(pairs)) if pairs else 0.0
    dcg = sum(1 / math.log2(i + 2) for i, doc in enumerate(docs[:10]) if doc in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    out["ndcg@10"] = float(dcg / ideal) if ideal else 0.0
    return out


def summarize(rows: list[dict]) -> dict:
    out = {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
    out["queries"] = len(rows)
    return out


def slice_summary(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: summarize(values) for name, values in sorted(groups.items())}


def paired(base: list[dict], candidate: list[dict], metric: str,
           resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    left = np.array([row[metric] for row in base], dtype=float)
    right = np.array([row[metric] for row in candidate], dtype=float)
    diff = right - left
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, diff.size, size=(resamples, diff.size))
    means = diff[samples].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "delta": float(diff.mean()),
        "ci95": [float(low), float(high)],
        "wtl": [int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())],
    }


def gate(block: dict, token_ratio: dict) -> dict:
    """Pre-registered rule: beat RRF on all@10 at one budget, or tie with >=20% fewer tokens."""
    beats, matches = {}, {}
    for budget in BUDGETS:
        report = block[str(budget)][CESR]["all@10"]
        low, high = report["ci95"]
        beats[budget] = bool(report["delta"] > 0 and low > 0)
        matches[budget] = bool(low <= 0 <= high and token_ratio[budget] <= 0.8)
    beating = [budget for budget in BUDGETS if beats[budget]]
    matching = [budget for budget in BUDGETS if matches[budget]]
    return {
        "rule": (
            "CESR-code beats RRF(Porter,entity,graph) on all@10 at the same budget "
            "(delta>0, CI95 low>0) on at least one budget, OR matches all@10 (CI spans "
            "zero) with >=20% fewer tokens."
        ),
        "budgets_beating_rrf": beating,
        "budgets_matching_with_fewer_tokens": matching,
        "all_at_10_by_budget": {
            str(budget): block[str(budget)][CESR]["all@10"] for budget in BUDGETS
        },
        "token_ratio_cesr_over_rrf": {str(b): token_ratio[b] for b in BUDGETS},
        "passed": bool(beating or matching),
        "verdict": (
            "exact constraints do rescue the evidence-set objective on code"
            if (beating or matching) else
            "exact constraints do not rescue the evidence-set objective on code"
        ),
    }
