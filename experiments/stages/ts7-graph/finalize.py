"""Assemble results/ts7_graph.json from the artifacts in out/.

    node graph_ts7.mjs && node graph_regex.mjs && node bench.mjs
    node compare.mjs && python make_queries.py && python run_retrieval.py
    python finalize.py

The hand-checked precision verdicts are recorded here (not recomputed) because a
human read the source lines; `verdict` values are copied from that review.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RESULTS = HERE.parent / "results"

# --- hand review of the deterministic audit samples -------------------------
# TS7 CALLS: 20/20 are real call sites.
TS7_CALL_VERDICTS = {
    1: "correct",
    2: "correct",
    3: "correct (callee is a call result, so the name is unqualified: <CallExpression>.map)",
    4: "correct",
    5: "correct (callee is a parenthesised expression)",
    6: "correct",
    7: "correct",
    8: "correct",
    9: "correct",
    10: "correct (minified line, second statement)",
    11: "correct",
    12: "correct (chained off a call)",
    13: "correct",
    14: "correct (chained off an array literal)",
    15: "correct",
    16: "correct",
    17: "correct",
    18: "correct",
    19: "correct",
    20: "correct",
}
REGEX_CALL_VERDICTS = {
    1: "correct",
    2: "correct",
    3: "correct",
    4: "correct",
    5: "correct",
    6: "correct",
    7: "correct but under-qualified: reports 'log' for 'console.log'",
    8: "correct",
    9: "correct",
    10: "correct",
    11: "correct",
    12: "correct",
    13: "FALSE POSITIVE: the match is inside a // comment",
    14: "FALSE POSITIVE: method definition, not a call",
    15: "correct",
    16: "correct",
    17: "correct",
    18: "correct",
    19: "correct",
    20: "correct",
}
# IMPORTS: both extractors 20/20 on the sample. The regex recall gap (60 vs 63)
# is not in the sample: it is scripts/stress-test-compact.js, a minified file with
# four import statements on one line, where the line-anchored regex sees only one.
IMPORT_VERDICTS = {"ts7": 20, "regex": 20}

API_VERDICT = {
    "package": "typescript",
    "version": "7.0.2",
    "why_this_package": (
        "The published typescript@7.0.2 ships the same dist/api/* and dist/ast/* tree as "
        "@typescript/native-preview, and exports the same './unstable/sync' and "
        "'./unstable/ast' subpaths. So the probe did not need the dev preview package: "
        "the stable tarball has the API, behind unstable/* exports."
    ),
    "entry_points": [
        "typescript/unstable/sync -> API, Snapshot, Project, Program, Checker, Symbol, NodeHandle, SymbolFlags",
        "typescript/unstable/ast  -> SyntaxKind, is* type guards, node.forEachChild, createScanner",
    ],
    "usable": True,
    "stable_enough_for_production": False,
    "verdict": (
        "Usable today for a build-time code graph. It is one child-process round trip per "
        "checker query with no batch API, the whole surface is namespaced unstable/*, and "
        "several node kinds differ from what a JS dev expects (constructor, new). Fine as a "
        "batch extractor; not fine as a per-keystroke service."
    ),
    "worked": [
        "new API({collectTiming:true}) + updateSnapshot({openProjects:[tsconfig]}) loads a real tsconfig, honouring allowJs.",
        "project.rootFiles gives the tsconfig include set (14 files); program.getSourceFileNames() also lists 193 lib/node_modules files.",
        "program.getSourceFile(path) returns a full SourceFile with .text, .statements, getLineStarts(), getLineAndCharacterOfPosition().",
        "node.forEachChild(cb) walks the tree; SyntaxKind + is.FunctionDeclaration/isCallExpression/... are exported.",
        "checker.getSymbolAtLocation(node) resolves call targets, including cross-file; sym.valueDeclaration / sym.declarations give NodeHandles.",
        "NodeHandle.resolve(project) turns a handle back into a node with .path and .getStart(), which is enough to map back to a chunk.",
        "api.getTimingInfo().totals reports requestCount, roundTripMs, serverTimeMs, nodesMaterialized, bytesReceived.",
    ],
    "friction": [
        "No batch checker API: 2,031 getSymbolAtLocation calls cost 2,245 IPC requests and ~220 ms of round trips. This dominates the build.",
        "symbol.declarations are NodeHandles, not nodes; each one needs .resolve(project), another round trip.",
        "symbol.flags & SymbolFlags.Alias is required before the target resolves across an import; getAliasedSymbol is a second round trip.",
        "Class constructors are ConstructorDeclaration, not MethodDeclaration: isMethodDeclaration silently skips them (one chunk was missing until the regex diff caught it).",
        "`new X()` is NewExpression, not CallExpression: a CALLS walker sees nothing unless it adds that kind explicitly.",
        "CallExpression.getStart() is the start of the whole chain, so `a\\n  .b()\\n  .c()` reports the line of `a`. Site lines need the callee name token.",
        "include globs are resolved relative to the tsconfig's own directory; a wrong ../ depth gives a silently empty project, not an error.",
        "updateSnapshot({openProject}) is deprecated in favour of openProjects; the deprecated shim still works.",
        "There is no forEachChild free function; traversal is a method on each node. Minor, but differs from the TypeScript 5/6 API.",
        "The API spawns the native tsgo binary: ~50 ms per API instance, and api.close() must be called or the child stays alive.",
    ],
}


def main() -> None:
    graph = json.loads((OUT / "graph_ts7.json").read_text())
    regex = json.loads((OUT / "graph_regex.json").read_text())
    compare = json.loads((OUT / "compare.json").read_text())
    bench = json.loads((OUT / "bench.json").read_text())
    retrieval = json.loads((RESULTS / "ts7_graph.json").read_text())
    shared = json.loads((OUT / "shared_queries.json").read_text())
    ts_audit = json.loads((OUT / "audit_sample_ts7.json").read_text())
    rg_audit = json.loads((OUT / "audit_sample_regex.json").read_text())

    def annotate(picks, verdicts):
        out = []
        for i, p in enumerate(picks, start=1):
            row = dict(p)
            row["verdict"] = verdicts[i]
            out.append(row)
        return out

    ts_calls = annotate(ts_audit["calls"], TS7_CALL_VERDICTS)
    rg_calls = annotate(rg_audit["calls"], REGEX_CALL_VERDICTS)

    rel = graph["relations"]
    result = {
        "meta": {
            "experiment": "TS7 native-compiler code graph vs regex code graph",
            "repo": graph["meta"]["repo"],
            "corpus": {
                "files": graph["meta"]["files"],
                "lines": 3656,
                "language": "JavaScript (JSDoc types), ESM",
            },
            "pinned": {
                "typescript": "7.0.2",
                "node": bench["node"],
                "snowballstemmer": "porter",
            },
            "reproduce": [
                "cd experiments/stages/ts7-graph",
                "node graph_ts7.mjs && node graph_regex.mjs",
                "node bench.mjs",
                "node compare.mjs",
                "/Users/asmit/Personal/algo-serp-stages/.venv/bin/python make_queries.py",
                "/Users/asmit/Personal/algo-serp-stages/.venv/bin/python run_retrieval.py",
                "/Users/asmit/Personal/algo-serp-stages/.venv/bin/python run_shared_queries.py",
                "/Users/asmit/Personal/algo-serp-stages/.venv/bin/python finalize.py",
            ],
        },
        "api": API_VERDICT,
        "graph": {
            "ts7": {
                "chunks": len(graph["chunks"]),
                "symbols": len(graph["symbols"]),
                "chunks_by_kind": dict(Counter(c["kind"] for c in graph["chunks"])),
                "relations": {k: len(v) for k, v in rel.items()},
                "calls_resolved_to_declaration": graph["meta"]["calls_resolved_to_declaration"],
                "calls_symbol_only": sum(1 for e in rel["CALLS"] if e["resolved"] == "symbol_only"),
                "calls_unresolved": sum(1 for e in rel["CALLS"] if e["resolved"] == "unresolved"),
                "call_sites": graph["meta"]["call_sites"],
                "nodes_walked": graph["meta"]["nodes_walked"],
                "checker_calls": graph["meta"]["checker_calls"],
                "checker_failures": graph["meta"]["checker_failures"],
                "exported_chunks": sum(1 for c in graph["chunks"] if c["exported"]),
                "imports_with_named_bindings": sum(
                    1 for e in rel["IMPORTS"] if e["names"]
                ),
            },
            "regex": {
                "chunks": len(regex["chunks"]),
                "symbols": len(regex["symbols"]),
                "chunks_by_kind": dict(Counter(c["kind"] for c in regex["chunks"])),
                "relations": {k: len(v) for k, v in regex["relations"].items()},
                "calls_resolved_to_declaration": 0,
            },
        },
        "timing": {
            "note": (
                "Process wall time includes ~65 ms of node startup for both extractors, so "
                "compare the in-process build times; process time is what a CLI pays."
            ),
            "process_wall_median_ms": {
                "regex": bench["regex"]["median"],
                "ts7_ast_only": bench["ts7_no_checker_process"]["median"],
                "ts7_with_checker": bench["ts7_total_process"]["median"],
            },
            "process_wall_runs_ms": {
                "regex": bench["regex"]["runs_ms"],
                "ts7_ast_only": bench["ts7_no_checker_process"]["runs_ms"],
                "ts7_with_checker": bench["ts7_total_process"]["runs_ms"],
            },
            "in_process_build_ms": {
                "regex": bench["regex_meta"]["timings"]["total_build_ms"],
                "ts7_ast_only": bench["ts7_breakdown_no_checker"]["total_build_ms"],
                "ts7_with_checker": bench["ts7_breakdown"]["total_build_ms"],
            },
            "ts7_breakdown_ast_only": bench["ts7_breakdown_no_checker"],
            "ts7_breakdown_with_checker": bench["ts7_breakdown"],
            "ts7_transport": bench["ts7_transport"],
            "ts7_transport_ast_only": bench["ts7_transport_no_checker"],
        },
        "comparison": {
            "chunks": compare["chunks"],
            "calls": compare["calls"],
            "imports": compare["imports"],
        },
        "precision_sample": {
            "method": (
                "Deterministic stride sampling over sorted edges (every n/20th). "
                "Verdicts were assigned by reading each source line."
            ),
            "ts7": {
                "calls_correct": sum(1 for v in TS7_CALL_VERDICTS.values() if v.startswith("correct")),
                "calls_total": len(TS7_CALL_VERDICTS),
                "calls": ts_calls,
                "imports_correct": IMPORT_VERDICTS["ts7"],
                "imports_total": 20,
                "imports": ts_audit["imports"],
            },
            "regex": {
                "calls_correct": sum(1 for v in REGEX_CALL_VERDICTS.values() if v.startswith("correct")),
                "calls_total": len(REGEX_CALL_VERDICTS),
                "calls": rg_calls,
                "imports_correct": IMPORT_VERDICTS["regex"],
                "imports_total": 20,
                "imports": rg_audit["imports"],
            },
        },
        "retrieval_shared_queries": shared,
        "retrieval_own_queries": retrieval,
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ts7_graph.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {RESULTS / 'ts7_graph.json'}")
    print(
        "ts7 chunks",
        result["graph"]["ts7"]["chunks"],
        "regex chunks",
        result["graph"]["regex"]["chunks"],
    )


if __name__ == "__main__":
    main()
