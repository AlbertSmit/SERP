"""CESR-U: CESR with universal quantification and exact witness identity.

CESR-code (lab/cesr_code.py) maximised *existence*: one `CALLS(X)` witness, one
`EXPORTS(file)` witness. Its qrels demand the universal closure -- all callers of
the middle chunk, all exports of the file -- so full coverage delivered the full
evidence set only 72.5% of the time, and 96.8% of the missing gold was another
chunk satisfying an already-covered constraint.

This module makes the constraint language able to say what the qrels mean:

    EXISTS_CALLS(X)              alternatives: callers(X)          (witness set)
    ALL_CALLERS(X)               mandatory:    callers(X)          (universal)
    EXISTS_DEFINES(X)            alternatives: defined(X)          (witness set)
    EXACT_DEFINES(X, chunk_id)   identity:     {chunk_id}          (mandatory)
    ALL_DEFINES(X)               mandatory:    defined(X)          (universal)
    ALL_EXPORTS(file)            mandatory:    export units of file
    IMPORTS(file, module)        predicate on a chunk's file, not a packable set
    SAME_FILE(a, b)              relation: file(a) == file(b)
    CALL_CHAIN(X, middle)        ALL_CALLERS(middle) + middle in callers(X) & defined(middle)
    CALLS_BOTH(X, Y)             alternatives: callers(X) & callers(Y)

Every chunk id here comes from the graph. Constraints are built from `type`,
`symbols`, `module` and `chain`; `qrels` is never read by the builder, and `qid`
is not read either (the generated qid embeds the answer chunk id, which would be
a leak). `qrels` is used afterwards, for measurement only.

The selector is not a coverage maximiser over a flat candidate list any more:

  phase 1  resolve witness constraints (alternative sets) by coverage, then by
           BM25/lexical score, then by length and id. All alternatives satisfy
           the constraint equally, so lexical score is the only discriminating
           evidence there is.
  phase 2  build the mandatory closure: every universal set, plus universal sets
           that depend on a chosen witness (SAME_FILE anchors follow the caller).
  phase 3  insert the closure first, by identity, not by rank. A mandatory chunk
           is fetched from the graph even if no channel retrieved it.
  phase 4  greedy coverage for any constraint still uncovered, with the CESR-code
           cost/redundancy score and a small lexical tie-break.
  phase 5  repair: if a universal set is still incomplete, add its members and
           report the budget violation instead of silently dropping evidence.

The qrels of the host benchmark are graph-derived, so this is still a capability
test with oracle constraints. See experiments/stages/CESR_U.md.
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
# Inherited verbatim from CESR-code so the comparison is matched. The only new
# constant is LEX_TIEBREAK, fixed before the run: it is small enough that coverage
# and cost still decide, and only separates candidates whose coverage is equal.
BUDGETS = (500, 1000, 2000)
UNBUDGETED = 1 << 30
POOL_DEPTH = 100
MU = 0.2
COST_EXPONENT = 0.35
PATH_BONUS = 0.25
LEX_TIEBREAK = 0.02
RESAMPLES = 1000
BOOTSTRAP_SEED = 1729

TYPE_ORDER = ("same_file", "import_call", "two_hop", "export_import", "calls_two")
METRICS = ("all@5", "all@10", "recall", "coverage", "tokens", "selected",
           "redundancy", "ndcg@10")
SYSTEMS = ("porter_trunc", "rrf_trunc", "cesr_code", "cesr_u")
CESR_U = "cesr_u"
CESR_CODE = "cesr"  # system id of the CESR-code baseline (cc.CESR)

# Constraint kinds.
UNIVERSAL = "universal"        # every member must be selected
UNIVERSAL_ANY = "universal_any"  # one of several alternative groups must be complete
WITNESS = "witness"            # exactly one alternative must be selected
IDENTITY = "identity"          # exactly these chunk ids must be selected


# ------------------------------------------------------------------- constraints


@dataclass(frozen=True)
class Constraint:
    """One requirement. `members` means different things per kind, by design.

    universal      members = the full required set
    universal_any   groups  = alternative required sets (one must be complete)
    witness        members = the alternatives; one must be chosen
    identity       members = the exact chunk ids the graph pins
    """

    label: str
    kind: str
    members: frozenset[str] = frozenset()
    groups: tuple[frozenset[str], ...] = ()
    weight: float = 0.0
    is_answer: bool = False

    @property
    def options(self) -> frozenset[str]:
        return self.members


@dataclass
class Plan:
    constraints: list[Constraint]
    predicates: list[dict] = field(default_factory=list)
    # universal sets that follow a witness choice:
    # {"constraint": index, "witness": index, "symbol": str}
    derived: list[dict] = field(default_factory=list)
    witness_index: int = -1
    anchors_by_caller: dict[str, frozenset[str]] = field(default_factory=dict)
    file_candidates: list[str] = field(default_factory=list)
    empty: bool = False


def exists_calls(corpus, symbol: str) -> frozenset[str]:
    return corpus.callers.get(symbol, frozenset())


def all_callers(corpus, symbol: str) -> frozenset[str]:
    return corpus.callers.get(symbol, frozenset())


def exists_defines(corpus, symbol: str) -> frozenset[str]:
    return corpus.defined.get(symbol, frozenset())


def all_defines(corpus, symbol: str) -> frozenset[str]:
    return corpus.defined.get(symbol, frozenset())


def exact_defines(corpus, symbol: str, chunk_id: str) -> frozenset[str]:
    """Identity: the named chunk, accepted only when the graph defines `symbol` there."""
    if chunk_id in corpus.defined.get(symbol, frozenset()):
        return frozenset({chunk_id})
    return frozenset()


def all_exports(corpus, graph, file: str) -> frozenset[str]:
    """Export units of a file: chunks whose name is exported there.

    Class containers are excluded. That is a declared corpus-unit convention, not
    a qrels lookup: the generator excludes them for the same reason (a 1,000-line
    class is not a retrieval unit), and so does CESR-code's export channel.
    """
    return frozenset(
        doc for symbol in graph.exports.get(file, ())
        for doc in corpus.defined.get(symbol, ())
        if corpus.file_of[doc] == file and graph.chunk_kind.get(doc) != "class"
    )


def imports(graph, file: str, module: str) -> bool:
    """IMPORTS(file, module): a file-level predicate, not a selectable chunk set."""
    return module in graph.file_imports.get(file, ())


def same_file(corpus, left: str, right: str) -> bool:
    return corpus.file_of[left] == corpus.file_of[right]


def call_chain_targets(corpus, callee: str, middle_name: str) -> frozenset[str]:
    """CALL_CHAIN(X, middle): the chunks that both call X and define `middle`."""
    return exists_calls(corpus, callee) & corpus.defined.get(middle_name, frozenset())


def calls_both(corpus, first: str, second: str) -> frozenset[str]:
    return corpus.callers.get(first, frozenset()) & corpus.callers.get(second, frozenset())


def files_defining(corpus, symbol: str) -> frozenset[str]:
    return frozenset(corpus.file_of[doc] for doc in corpus.defined.get(symbol, ()))


def _definition_requirement(label: str, chunks: frozenset[str]) -> Constraint:
    """EXACT_DEFINES when the graph pins one chunk, ALL_DEFINES otherwise.

    The distinction is semantic, not cosmetic: a singleton definition set is an
    exact identity, and the selector fetches that chunk id directly. A symbol with
    several definitions is a universal set, because any of them satisfies the
    question's "the chunk that defines X".
    """
    if len(chunks) == 1:
        return Constraint(f"EXACT_{label}", IDENTITY, frozenset(chunks))
    return Constraint(f"ALL_{label}", UNIVERSAL, frozenset(chunks))


def _weigh(constraints: list[Constraint]) -> list[Constraint]:
    """Equal weight per constraint, normalised so coverage is a fraction."""
    share = 1.0 / max(len(constraints), 1)
    return [
        Constraint(label=c.label, kind=c.kind, members=c.members, groups=c.groups,
                   weight=share, is_answer=c.is_answer)
        for c in constraints
    ]


def build_plan(corpus, graph, query: dict) -> Plan:
    """Compose the constraint set from `type`, `symbols`, `module`, `chain` only."""
    kind = query["type"]
    if kind == "same_file":
        return _same_file_plan(corpus, query)
    if kind == "import_call":
        return _import_call_plan(corpus, graph, query)
    if kind == "two_hop":
        return _two_hop_plan(corpus, query)
    if kind == "export_import":
        return _export_import_plan(corpus, graph, query)
    if kind == "calls_two":
        return _calls_two_plan(corpus, query)
    raise ValueError(f"unknown query type: {kind}")


def _same_file_plan(corpus, query: dict) -> Plan:
    """`Which chunk calls X and is defined in the same file as Y?`

    The witness is the caller. The anchor set follows the witness: every chunk
    defining Y in the caller's file. The joint is SAME_FILE(caller, anchor).
    """
    callee, anchor_symbol = query["symbols"]
    files = files_defining(corpus, anchor_symbol)
    options = frozenset(
        caller for caller in exists_calls(corpus, callee)
        if corpus.file_of[caller] in files
    )
    anchors_by_caller = {
        caller: frozenset(
            doc for doc in corpus.defined.get(anchor_symbol, ())
            if corpus.file_of[doc] == corpus.file_of[caller]
        )
        for caller in options
    }
    groups = tuple(sorted({group for group in anchors_by_caller.values() if group}))
    constraints = [
        Constraint(f"CALLS_FILE_OF({callee},{anchor_symbol})", WITNESS, options,
                   is_answer=True),
        Constraint(f"ALL_DEFINES_IN_FILE({anchor_symbol})", UNIVERSAL_ANY, frozenset(),
                   groups=groups),
    ]
    plan = Plan(constraints=_weigh(constraints),
                predicates=[{"label": f"same_file({callee},{anchor_symbol})",
                             "kind": "same_file", "witness": 0}],
                derived=[{"constraint": 1, "witness": 0,
                          "symbol": anchor_symbol}],
                witness_index=0,
                anchors_by_caller=anchors_by_caller)
    return plan


def _import_call_plan(corpus, graph, query: dict) -> Plan:
    """`Which chunk calls X in a file that imports M?`

    IMPORTS is a predicate on the caller's file. The caller is one of ~17
    alternatives; the definition is universal (all chunks defining X), which
    contains the single definition the qrels name without mirroring its tie-break.
    """
    symbol, module = query["symbols"][0], query["module"]
    importing = corpus.module_chunks.get(module, frozenset())
    options = frozenset(
        caller for caller in exists_calls(corpus, symbol) if caller in importing
    )
    constraints = [
        Constraint(f"CALLS_IMPORTING({symbol},{module})", WITNESS, options,
                   is_answer=True),
        _definition_requirement(f"DEFINES({symbol})", all_defines(corpus, symbol)),
    ]
    return Plan(constraints=_weigh(constraints),
                predicates=[{"label": f"IMPORTS(file({symbol}),{module})",
                             "kind": "imports", "witness": 0, "module": module}],
                witness_index=0)


def _two_hop_plan(corpus, query: dict) -> Plan:
    """`Which chunk calls a function that calls X?`

    The middle chunk is an identity: it calls X and defines chain[1]. Every upper
    caller of chain[1] is universal. The chain relation is
    CALL_CHAIN(X, middle) = CALLS(X) & DEFINES(middle), and each upper caller is
    in ALL_CALLERS(middle).
    """
    callee, middle_name = query["chain"]
    middle = call_chain_targets(corpus, callee, middle_name)
    constraints = [
        Constraint(f"CALL_CHAIN({callee},{middle_name})", WITNESS, middle,
                   is_answer=True),
        Constraint(f"ALL_CALLERS({middle_name})", UNIVERSAL,
                   all_callers(corpus, middle_name)),
    ]
    return Plan(constraints=_weigh(constraints),
                predicates=[{"label": f"chain({callee}->{middle_name})",
                             "kind": "chain", "witness": 0,
                             "upper": all_callers(corpus, middle_name)}],
                witness_index=0)


def _export_import_plan(corpus, graph, query: dict) -> Plan:
    """`Which exports are defined in the file that imports M and defines X?`

    The file is pinned by the graph: the unique file that imports M and defines X.
    Its export units and its chunks defining X are universal.
    """
    symbol, module = query["symbols"][0], query["module"]
    candidates = sorted(
        set(files_defining(corpus, symbol))
        & {file for file in graph.files if imports(graph, file, module)}
    )
    file = candidates[0] if candidates else None
    export_units = all_exports(corpus, graph, file) if file else frozenset()
    in_file = frozenset(
        doc for doc in corpus.defined.get(symbol, ()) if corpus.file_of[doc] == file
    ) if file else frozenset()
    constraints = [
        Constraint(f"ALL_EXPORTS({file})", UNIVERSAL, export_units),
        _definition_requirement(f"DEFINES_IN_FILE({symbol},{file})", in_file),
    ]
    plan = Plan(constraints=_weigh(constraints),
                predicates=[{"label": f"IMPORTS({file},{module})", "kind": "imports",
                             "module": module, "file": file}],
                witness_index=-1,
                file_candidates=candidates,
                empty=not candidates)
    return plan


def _calls_two_plan(corpus, query: dict) -> Plan:
    """`Which chunk calls both X and Y?`  Caller ambiguous, definitions universal."""
    first, second = query["symbols"]
    constraints = [
        Constraint(f"CALLS_BOTH({first},{second})", WITNESS,
                   calls_both(corpus, first, second), is_answer=True),
        _definition_requirement(f"DEFINES({first})", all_defines(corpus, first)),
        _definition_requirement(f"DEFINES({second})", all_defines(corpus, second)),
    ]
    return Plan(constraints=_weigh(constraints),
                predicates=[{"label": f"calls_both({first},{second})",
                             "kind": "calls_both", "witness": 0}],
                witness_index=0)


# --------------------------------------------------------------- plan diagnostics


def witness_options(plan: Plan) -> list[dict]:
    """Alternative counts of every witness constraint, from the graph alone."""
    return [
        {"label": constraint.label, "options": len(constraint.options)}
        for constraint in plan.constraints if constraint.kind in (WITNESS, IDENTITY)
    ]


def unique_answer(plan: Plan) -> bool:
    """True when every witness constraint has exactly one alternative."""
    return all(entry["options"] == 1 for entry in witness_options(plan))


def unsatisfiable(plan: Plan) -> bool:
    if plan.empty:
        return True
    return any(entry["options"] == 0 for entry in witness_options(plan))


def answer_options(plan: Plan) -> int:
    counts = [entry["options"] for entry in witness_options(plan)]
    if not counts:
        counts = [len(getattr(plan, "file_candidates", [1]))]
    return max(counts) if counts else 0


def closure(plan: Plan, witness_choice: dict[int, str] | None = None) -> frozenset[str]:
    """Everything the constraint set says must be selected, for a given witness.

    Used to check that the language can express the qrels (diagnostic only).
    """
    chosen = witness_choice or {}
    required: set[str] = set()
    for index, constraint in enumerate(plan.constraints):
        if constraint.kind in (UNIVERSAL, IDENTITY):
            required |= set(constraint.members)
        elif constraint.kind == WITNESS:
            if index in chosen:
                required.add(chosen[index])
        elif constraint.kind == UNIVERSAL_ANY:
            required |= set(_group_for_witness(plan, index, chosen))
    return frozenset(required)


def expressible(plan: Plan, qrels: set[str]) -> bool:
    """Can the constraint set contain the qrels under *some* legal witness choice?

    This separates expressiveness from disambiguation: if True, the language can
    state the answer and the only thing left is picking the right alternative. If
    False, the operator set cannot express the evidence at all. Diagnostic only;
    it is the one place where qrels is compared against the plan.
    """
    witnesses = [index for index, constraint in enumerate(plan.constraints)
                 if constraint.kind == WITNESS and constraint.options]
    if not witnesses:
        return qrels <= closure(plan)
    index = witnesses[0]
    return any(
        qrels <= closure(plan, {index: option})
        for option in plan.constraints[index].options
    )


# --------------------------------------------------------------------- retrieval


def build_context(ix, graph, position, query):
    """Same three channels as CESR-code, plus the lexical score for tie-breaks."""
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
    top = float(bm25.max()) if bm25.size and bm25.max() > 0 else 0.0
    lex = np.zeros(ix.N, dtype=float)
    if top > 0:
        lex = bm25 / top
    return pool, provenance, channels, {doc: float(lex[position[doc]]) for doc in ix.doc_ids}


# -------------------------------------------------------------------- selector


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / (len(left) + len(right) - shared) if shared else 0.0


def _witness_coverage(candidate: str, plan: Plan) -> int:
    """How many constraints a witness option belongs to. Coverage comes first."""
    return sum(
        1 for constraint in plan.constraints
        if not constraint.is_answer and candidate in constraint.members
    )


def choose_witness(plan: Plan, index: int, options: frozenset[str], corpus,
                   lex: dict[str, float], headroom: int) -> str:
    """Budget fit, then coverage, then lexical score, then length, then id.

    The headroom test comes first so a witness that would blow the residual
    budget is only chosen when no alternative fits. It can only change the choice
    among *ambiguous* alternatives, where every option satisfies the constraint
    equally; on a uniquely determined witness the option set is a singleton.
    """
    return max(
        sorted(options),
        key=lambda doc: (
            corpus.dl[doc] <= headroom,
            _witness_coverage(doc, plan),
            lex.get(doc, 0.0),
            -corpus.dl[doc],
        ),
    )


def _marginal(doc: str, constraint: Constraint, picked: set[str]) -> float:
    """Fraction of one constraint that `doc` adds, if any."""
    if constraint.kind in (UNIVERSAL, IDENTITY):
        if doc in constraint.members and doc not in picked:
            return 1.0 / max(len(constraint.members), 1)
        return 0.0
    if constraint.kind == WITNESS:
        return 1.0 if doc in constraint.options and not (picked & constraint.options) else 0.0
    if constraint.kind == UNIVERSAL_ANY:
        for group in constraint.groups:
            if doc in group and not group <= picked:
                return 1.0 / len(group)
        return 0.0
    return 0.0


def select(corpus, graph, plan: Plan, pool: list[str], lex: dict[str, float],
           budget: int = UNBUDGETED, fill: bool = True) -> dict:
    """Universal closure first, then witnesses, then optional greedy, then repair."""
    order: list[str] = []
    chosen_witness: dict[int, str] = {}

    # headroom for a witness: the budget minus universal sets that do not depend
    # on the witness. Universal_any groups are witness-dependent and excluded.
    fixed_tokens = sum(
        corpus.dl[doc]
        for constraint in plan.constraints
        if constraint.kind in (UNIVERSAL, IDENTITY)
        for doc in constraint.members
    )
    headroom = budget - fixed_tokens if budget != UNBUDGETED else UNBUDGETED

    # phase 1: resolve witnesses from alternatives (never from qrels)
    for index, constraint in enumerate(plan.constraints):
        if constraint.kind not in (WITNESS,):
            continue
        if not constraint.options:
            continue
        chosen_witness[index] = choose_witness(
            plan, index, constraint.options, corpus, lex, headroom)
    for index, constraint in enumerate(plan.constraints):
        if constraint.kind != IDENTITY:
            continue
        chosen_witness[index] = min(constraint.members)

    # phase 2: mandatory closure, including universal sets that follow a witness
    required: list[str] = []
    for index, constraint in enumerate(plan.constraints):
        if constraint.kind in (UNIVERSAL, IDENTITY):
            required.extend(sorted(constraint.members, key=lambda doc: (-lex.get(doc, 0.0), doc)))
        elif constraint.kind == UNIVERSAL_ANY:
            group = _group_for_witness(plan, index, chosen_witness)
            required.extend(sorted(group, key=lambda doc: (-lex.get(doc, 0.0), doc)))

    # phase 3: insert the closure by identity, then the witness answers
    picked: set[str] = set()
    for doc in required:
        if doc not in picked:
            picked.add(doc)
            order.append(doc)
    for index in sorted(chosen_witness):
        doc = chosen_witness[index]
        if doc not in picked:
            picked.add(doc)
            order.append(doc)

    used = sum(corpus.dl[doc] for doc in order)
    over_budget = used > budget

    # phase 4: greedy coverage for anything still uncovered (cost / redundancy / lex)
    candidates = list(dict.fromkeys(list(pool) + order))
    tokens_of = {doc: corpus.dl[doc] for doc in candidates}
    redundancy = {doc: 0.0 for doc in candidates}
    steps: list[dict] = []
    while fill and not over_budget:
        unsatisfied = [
            constraint for constraint in plan.constraints
            if _satisfaction(constraint, picked) < 1.0
        ]
        if not unsatisfied:
            break
        fits = [doc for doc in candidates
                if doc not in picked and used + tokens_of[doc] <= budget]
        best, best_score, best_gain = None, 0.0, 0.0
        for doc in fits:
            gain = sum(constraint.weight * _marginal(doc, constraint, picked)
                       for constraint in unsatisfied)
            if gain <= 0:
                continue
            score = (gain / max(tokens_of[doc] + 10, 1) ** COST_EXPONENT
                     - MU * redundancy[doc] + LEX_TIEBREAK * lex.get(doc, 0.0))
            if score > best_score:
                best, best_score, best_gain = doc, score, gain
        if best is None:
            break
        picked.add(best)
        order.append(best)
        used += tokens_of[best]
        steps.append({"doc": best, "gain": best_gain, "score": best_score,
                      "redundancy": redundancy[best]})
        for other in candidates:
            if other in picked:
                continue
            redundancy[other] = max(redundancy[other],
                                    _jaccard(corpus.symbols[best], corpus.symbols[other]))

    # phase 5: repair -- never silently drop mandatory evidence
    missing = [
        doc for index, constraint in enumerate(plan.constraints)
        if constraint.kind in (UNIVERSAL, IDENTITY)
        for doc in sorted(constraint.members - picked)
    ]
    repaired = 0
    for doc in dict.fromkeys(missing):
        picked.add(doc)
        order.append(doc)
        used += corpus.dl[doc]
        repaired += 1
    over_budget = over_budget or used > budget

    return {
        "order": order,
        "tokens": used,
        "mandatory_tokens": sum(corpus.dl[doc] for doc in required),
        "witness": {plan.constraints[index].label: doc for index, doc in chosen_witness.items()},
        "chosen_witness": chosen_witness,
        "over_budget": over_budget,
        "over_by": max(used - budget, 0) if budget != UNBUDGETED else 0,
        "repaired": repaired,
        "steps": steps,
    }


def _group_for_witness(plan: Plan, index: int, chosen: dict[int, str]) -> frozenset[str]:
    entry = next((item for item in plan.derived if item["constraint"] == index), None)
    if entry is None:
        return frozenset()
    caller = chosen.get(entry["witness"])
    return plan.anchors_by_caller.get(caller, frozenset()) if caller else frozenset()


def _satisfaction(constraint: Constraint, picked: set[str]) -> float:
    if constraint.kind in (UNIVERSAL, IDENTITY):
        if not constraint.members:
            return 1.0
        return len(constraint.members & picked) / len(constraint.members)
    if constraint.kind == WITNESS:
        return 1.0 if picked & constraint.options else 0.0
    if constraint.kind == UNIVERSAL_ANY:
        if not constraint.groups:
            return 0.0
        return max((len(group & picked) / len(group) for group in constraint.groups), default=0.0)
    return 0.0


# ---------------------------------------------------------------------- metrics


def coverage_of(plan: Plan, docs: list[str]) -> float:
    picked = set(docs)
    return float(sum(
        constraint.weight * _satisfaction(constraint, picked)
        for constraint in plan.constraints
    ))


def full_coverage(plan: Plan, docs: list[str]) -> bool:
    picked = set(docs)
    return all(_satisfaction(constraint, picked) >= 1.0 - 1e-12
               for constraint in plan.constraints)


def selection_metrics(corpus, docs, relevant, plan, tokens) -> dict:
    rel = set(relevant)
    picked = set(docs)
    out = {}
    for cutoff in (5, 10):
        out[f"all@{cutoff}"] = 1.0 if rel <= set(docs[:cutoff]) else 0.0
    out["recall"] = len(rel & picked) / len(rel) if rel else 0.0
    out["coverage"] = coverage_of(plan, docs)
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


# ------------------------------------------------------------------------- gate


def gate(block: dict, budgets=(1000, 2000)) -> dict:
    """Pre-registered rule.

    G1  CESR-U beats CESR-code on text-unique all@10 at every gated budget
        (delta > 0, CI95 low > 0).
    G2  CESR-U beats RRF on text-unique all@10 at every gated budget.
    G3  full coverage implies complete evidence at >= 0.90, measured on the
        text-unique set **overall** (the primary, stated rule). The
        constraint-determined subset is reported next to it, not substituted.
    Gated budgets are those where gold evidence fits in >= 90% of queries
    (1000 and 2000; 500 fits in 82%).
    """
    beats_code, beats_rrf = {}, {}
    for budget in budgets:
        code = block[str(budget)][CESR_U][f"vs_{CESR_CODE}"]["all@10"]
        rrf = block[str(budget)][CESR_U][f"vs_rrf_trunc"]["all@10"]
        beats_code[budget] = bool(code["delta"] > 0 and code["ci95"][0] > 0)
        beats_rrf[budget] = bool(rrf["delta"] > 0 and rrf["ci95"][0] > 0)
    completeness = block["full_coverage_completeness"]
    return {
        "rule": (
            "CESR-U beats CESR-code AND RRF on all@10 for text-unique queries at "
            "every budget where gold evidence fits in >=90% of queries "
            "(1000, 2000), and full coverage implies complete evidence at >= 0.90."
        ),
        "budgets": list(budgets),
        "beats_cesr_code": {str(b): beats_code[b] for b in budgets},
        "beats_rrf": {str(b): beats_rrf[b] for b in budgets},
        "full_coverage_completeness": completeness,
        "passed": bool(all(beats_code.values()) and all(beats_rrf.values())
                       and completeness["text_unique"] >= 0.90),
        "verdict": (
            "universal/identity constraints rescue the evidence-set objective"
            if all(beats_code.values()) and all(beats_rrf.values())
            and completeness["text_unique"] >= 0.90 else
            "universal/identity constraints do not rescue the objective on this benchmark"
        ),
    }
