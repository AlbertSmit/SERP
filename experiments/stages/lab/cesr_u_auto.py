"""CESR-U-auto: CESR-U with constraints parsed from the question text alone.

CESR-U (`CESR_U.md`) showed the objective is no longer the bottleneck: when the
constraint set is handed over, universal/identity constraints return the complete
evidence set on 161/161 constraint-determined text-unique queries. But it built
those constraints from the generator's metadata -- `type`, `symbols`, `module`,
`chain` -- which a deployed system does not have. This module deletes that input.

The only query input is `query["text"]` plus the graph. Nothing here reads `type`,
`symbols`, `module`, `chain`, `qid` or `qrels`; the generated qid embeds the answer
chunk id, so it is not touched either.

Parser v1 (frozen; no rule was fitted to the 250 queries):

  symbols   tokens in the text that exactly match a key of `graph.defined`.
            `code_graph.query_symbols` is reused for the candidate set (it is
            read-only), then filtered to exact-case token matches so that the
            "exactly match" rule is literal. Mentions are ordered by first
            occurrence in the text.
  modules   spans that look like paths or relative modules (`./x`, `../x`, `*.js`,
            `*.ts`) are resolved against the graph's import table. CESR-U's
            `IMPORTS(file, module)` predicate is literally
            `module in graph.file_imports[file]`, so a mention that occurs verbatim
            in at least one file's import list uses that literal relation. Only when
            no file contains the mention is `code_graph.resolve_module` used, via
            the set of files whose specifiers resolve to the same target. A
            relative specifier is only meaningful relative to its importer, so
            target resolution is the fallback, not the default.
  intent    fixed cue phrases: "same file" -> same_file, "calls both" -> calls_two,
            "imports"+"calls" -> import_call, "exports"+"imports"+"defines" ->
            export_import, "calls a function that calls" -> two_hop.

Two fixed parser decisions, both made before any retrieval number was computed:

1. Path spans are *excised* from the text before symbol matching. `./tokenize.js`
   contains the defined symbol `tokenize`; without this rule 49/50 `export_import`
   questions would parse two symbols from one module mention and be unsupported for
   a reason that is about tokenisation, not about the constraint.
2. A path span is one module mention. If the text yields more than one, the module
   is not pinned and the query is unsupported rather than guessed.

`two_hop` is detected and then declared **unsupported**. Its text names only the
bottom callee; the middle bridge chunk is the hidden `chain[1]` that also defines
the qrels. Parsing it from text would need a bridge extractor that does not exist
here, so the query falls back to RRF and is reported as a null slice. It is not
silently reconstructed from metadata.

An unsupported query is never guessed at. There is no oracle fallback inside
CESR-U-auto: the query is returned to the RRF baseline, unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import code_graph as cg

import cesr_u as cu

# --------------------------------------------------------------- frozen cue list
CUE_SAME_FILE = "same file"
CUE_CALLS_TWO = "calls both"
CUE_IMPORT_CALL = ("imports", "calls")
CUE_EXPORT_IMPORT = ("exports", "imports", "defines")
CUE_TWO_HOP = "calls a function that calls"

# path-like spans: relative modules first, then anything ending in a code extension
PATH_RE = re.compile(
    r"(?:\.{1,2}/)[A-Za-z0-9_./@-]+"
    r"|(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:js|ts|jsx|tsx|mjs|cjs)"
)
TOKEN_RE = re.compile(r"[A-Za-z_$][\w$]*")
SPACE_RE = re.compile(r"\s+")

PARSED = "parsed"
UNSUPPORTED = "unsupported"

INTENTS = ("same_file", "import_call", "two_hop", "export_import", "calls_two")


@dataclass
class Parse:
    """What the text alone supports, before the graph turns it into constraints."""

    intent: str
    status: str
    reason: str = ""
    plan_reason: str = ""
    symbols: list[str] = field(default_factory=list)
    module: str | None = None
    module_files: list[str] = field(default_factory=list)
    module_resolution: str = ""
    symbol_mentions: list[str] = field(default_factory=list)
    module_mentions: list[str] = field(default_factory=list)

    @property
    def parsed(self) -> bool:
        return self.status == PARSED


# ------------------------------------------------------------------- text parsing


def normalize(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def extract_modules(text: str) -> list[str]:
    """Path-like spans, in order, deduplicated."""
    return list(dict.fromkeys(match.group(0) for match in PATH_RE.finditer(text)))


def excise_modules(text: str) -> str:
    """Replace path spans with spaces so their segments cannot become symbols."""
    return PATH_RE.sub(lambda match: " " * len(match.group(0)), text)


def extract_symbols(graph, prose: str) -> list[str]:
    """Defined-symbol mentions in order of first appearance, exact-case only."""
    candidates = cg.query_symbols(graph, prose)
    words = set(TOKEN_RE.findall(prose))
    exact = {symbol for symbol in candidates if symbol in words}
    ordered: list[str] = []
    for match in TOKEN_RE.finditer(prose):
        token = match.group(0)
        if token in exact and token not in ordered:
            ordered.append(token)
    return ordered


def classify(text: str) -> tuple[str, str]:
    """Fixed cue phrases -> intent. Returns (intent, reason-if-none)."""
    lowered = text.lower()
    if CUE_TWO_HOP in lowered:
        return "two_hop", ""
    if all(cue in lowered for cue in CUE_EXPORT_IMPORT):
        return "export_import", ""
    if CUE_CALLS_TWO in lowered:
        return "calls_two", ""
    if all(cue in lowered for cue in CUE_IMPORT_CALL):
        return "import_call", ""
    if CUE_SAME_FILE in lowered:
        return "same_file", ""
    return "", "no_intent_cue"


def resolve_module_files(graph, mention: str) -> tuple[list[str], str]:
    """Files whose imports cover `mention`, literal first, target fallback."""
    literal = sorted(file for file in graph.files
                     if mention in graph.file_imports.get(file, ()))
    if literal:
        return literal, "literal"
    fileset = set(graph.files)
    targets = {
        target for file in graph.files
        for target in [cg.resolve_module(file, mention, fileset)]
        if target
    }
    if not targets:
        return [], "unresolved"
    resolved = sorted(
        file for file in graph.files
        if any(cg.resolve_module(file, module, fileset) in targets
               for module in graph.file_imports.get(file, ()))
    )
    return resolved, "resolved"


def parse_query(graph, text: str) -> Parse:
    """Intent + mentions from the question text. No constraints yet."""
    prose = excise_modules(normalize(text))
    intent, reason = classify(normalize(text))
    modules = extract_modules(normalize(text))
    symbols = extract_symbols(graph, prose)
    parse = Parse(
        intent=intent,
        status=PARSED if intent else UNSUPPORTED,
        reason=reason,
        symbols=symbols,
        module=modules[0] if len(modules) == 1 else None,
        module_mentions=modules,
        symbol_mentions=list(symbols),
    )
    if len(modules) == 1:
        parse.module_files, parse.module_resolution = resolve_module_files(graph, modules[0])
    return parse


# ------------------------------------------------------------------- plan building
#
# The operators, their weights and the selector are CESR-U's, unchanged. Only the
# source of the names differs: parsed mentions instead of query metadata.


def build_plan(graph, corpus, parse: Parse) -> tuple[cu.Plan | None, str]:
    """Constraints the text supports, or (None, reason) when it supports none."""
    if not parse.parsed:
        return None, parse.reason or "not_parsed"
    if parse.intent == "two_hop":
        return None, "bridge_name_not_in_text"
    if parse.intent == "same_file":
        return _same_file_plan(corpus, parse)
    if parse.intent == "import_call":
        return _import_call_plan(corpus, parse)
    if parse.intent == "export_import":
        return _export_import_plan(corpus, graph, parse)
    if parse.intent == "calls_two":
        return _calls_two_plan(corpus, parse)
    return None, "no_intent_cue"


def _same_file_plan(corpus, parse: Parse):
    if len(parse.symbols) != 2:
        return None, "symbol_arity"
    callee, anchor_symbol = parse.symbols
    if callee == anchor_symbol:
        return None, "symbols_identical"
    files = cu.files_defining(corpus, anchor_symbol)
    if not files:
        return None, "anchor_not_defined"
    options = frozenset(
        caller for caller in cu.exists_calls(corpus, callee)
        if corpus.file_of[caller] in files
    )
    if not options:
        return None, "no_calls_in_anchor_file"
    anchors_by_caller = {
        caller: frozenset(
            doc for doc in corpus.defined.get(anchor_symbol, ())
            if corpus.file_of[doc] == corpus.file_of[caller]
        )
        for caller in options
    }
    groups = tuple(sorted({group for group in anchors_by_caller.values() if group}))
    constraints = [
        cu.Constraint(f"CALLS_FILE_OF({callee},{anchor_symbol})", cu.WITNESS, options,
                      is_answer=True),
        cu.Constraint(f"ALL_DEFINES_IN_FILE({anchor_symbol})", cu.UNIVERSAL_ANY,
                      frozenset(), groups=groups),
    ]
    return cu.Plan(
        constraints=cu._weigh(constraints),
        predicates=[{"label": f"same_file({callee},{anchor_symbol})",
                     "kind": "same_file", "witness": 0}],
        derived=[{"constraint": 1, "witness": 0, "symbol": anchor_symbol}],
        witness_index=0,
        anchors_by_caller=anchors_by_caller,
    ), ""


def _import_call_plan(corpus, parse: Parse):
    if len(parse.symbols) != 1:
        return None, "symbol_arity"
    if len(parse.module_mentions) != 1:
        return None, "module_arity"
    if not parse.module_files:
        return None, "module_unresolved"
    symbol, module = parse.symbols[0], parse.module_mentions[0]
    importing = frozenset(
        doc for doc in corpus.ids if corpus.file_of[doc] in set(parse.module_files)
    )
    options = frozenset(
        caller for caller in cu.exists_calls(corpus, symbol) if caller in importing
    )
    if not options:
        return None, "no_importing_callers"
    constraints = [
        cu.Constraint(f"CALLS_IMPORTING({symbol},{module})", cu.WITNESS, options,
                      is_answer=True),
        cu._definition_requirement(f"DEFINES({symbol})",
                                   cu.all_defines(corpus, symbol)),
    ]
    return cu.Plan(
        constraints=cu._weigh(constraints),
        predicates=[{"label": f"IMPORTS(file({symbol}),{module})", "kind": "imports",
                     "witness": 0, "module": module}],
        witness_index=0,
    ), ""


def _export_import_plan(corpus, graph, parse: Parse):
    if len(parse.symbols) != 1:
        return None, "symbol_arity"
    if len(parse.module_mentions) != 1:
        return None, "module_arity"
    if not parse.module_files:
        return None, "module_unresolved"
    symbol, module = parse.symbols[0], parse.module_mentions[0]
    defining = {
        corpus.file_of[doc] for doc in corpus.defined.get(symbol, ())
    }
    candidates = sorted(defining & set(parse.module_files))
    if not candidates:
        return None, "no_file_imports_and_defines"
    if len(candidates) > 1:
        return None, "file_not_pinned"
    file = candidates[0]
    export_units = cu.all_exports(corpus, graph, file)
    if not export_units:
        return None, "no_export_units"
    in_file = frozenset(
        doc for doc in corpus.defined.get(symbol, ()) if corpus.file_of[doc] == file
    )
    constraints = [
        cu.Constraint(f"ALL_EXPORTS({file})", cu.UNIVERSAL, export_units),
        cu._definition_requirement(f"DEFINES_IN_FILE({symbol},{file})", in_file),
    ]
    return cu.Plan(
        constraints=cu._weigh(constraints),
        predicates=[{"label": f"IMPORTS({file},{module})", "kind": "imports",
                     "module": module, "file": file}],
        witness_index=-1,
        file_candidates=candidates,
    ), ""


def _calls_two_plan(corpus, parse: Parse):
    if len(parse.symbols) != 2:
        return None, "symbol_arity"
    first, second = parse.symbols
    if first == second:
        return None, "symbols_identical"
    options = cu.calls_both(corpus, first, second)
    if not options:
        return None, "no_common_caller"
    constraints = [
        cu.Constraint(f"CALLS_BOTH({first},{second})", cu.WITNESS, options,
                      is_answer=True),
        cu._definition_requirement(f"DEFINES({first})", cu.all_defines(corpus, first)),
        cu._definition_requirement(f"DEFINES({second})", cu.all_defines(corpus, second)),
    ]
    return cu.Plan(
        constraints=cu._weigh(constraints),
        predicates=[{"label": f"calls_both({first},{second})",
                     "kind": "calls_both", "witness": 0}],
        witness_index=0,
    ), ""


# ------------------------------------------------------------------ diagnostics


def constraint_labels(plan: cu.Plan | None) -> list[str]:
    return [constraint.label for constraint in plan.constraints] if plan else []


def coverable(constraint: cu.Constraint, pool: set[str]) -> bool:
    """Can the candidate pool fully satisfy this constraint by itself?"""
    if constraint.kind in (cu.UNIVERSAL, cu.IDENTITY):
        return bool(constraint.members) and constraint.members <= pool
    if constraint.kind == cu.WITNESS:
        return bool(constraint.options & pool)
    if constraint.kind == cu.UNIVERSAL_ANY:
        return any(group <= pool for group in constraint.groups)
    return False


def achievable_coverage(plan: cu.Plan | None, pool: set[str]) -> float:
    """Weight of the constraints the pool can satisfy completely."""
    if plan is None or not plan.constraints:
        return 0.0
    total = sum(constraint.weight for constraint in plan.constraints)
    reachable = sum(constraint.weight for constraint in plan.constraints
                    if coverable(constraint, pool))
    return float(reachable / total) if total else 0.0


def select_auto(corpus, graph, plan: cu.Plan, pool, lex, budget: int) -> dict:
    """CESR-U's closure-first selector, unchanged."""
    return cu.select(corpus, graph, plan, pool, lex, budget=budget)
