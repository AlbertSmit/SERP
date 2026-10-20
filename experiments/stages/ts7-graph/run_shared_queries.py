"""Run the regex graph's own 250-query evidence-chain set against the TS7 graph.

The point of this script is that only the *graph* changes. Query text, qrels and
the six scoring systems are the ones in `lab/code_graph.py` and
`run_code_graph.py`; both graphs are scored by the same `score_systems`, over the
same Porter index code (`cg.build_index`).

Two things are unavoidably different and are reported rather than hidden:

1. **Chunk boundaries.** The regex graph splits a class into a header chunk plus
   method chunks and splits top-level code at blank lines. The TS7 converter below
   reproduces that shape from the AST so the corpora are the same size, but the
   chunk ids are still the extractor's own, so qrels are joined on (file, start
   line), with a fallback to the enclosing chunk.
2. **Qrels that have no TS7 counterpart** (a regex chunk that only exists because
   brace matching ended a function early) map to the enclosing AST chunk. The
   fallback count is printed.

Writes out/shared_queries.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = HERE.parent
sys.path.insert(0, str(STAGES / "lab"))
sys.path.insert(0, str(STAGES))

import code_graph as cg  # noqa: E402
import run_code_graph as runner  # noqa: E402

OUT = HERE / "out"
ID_RE = re.compile(r"^(?P<file>.+):L(?P<start>\d+)-(?P<end>\d+):(?P<name>.+)$")


# --------------------------------------------------------------- TS7 -> Graph
def ts7_graph_object(graph: dict, chunk_text: dict) -> cg.Graph:
    """Same chunk shape as the regex graph, built from AST node ranges."""
    raw = {c["id"]: c for c in graph["chunks"]}
    children: dict[str, list[dict]] = {}
    for c in graph["chunks"]:
        if c["parent"]:
            children.setdefault(c["parent"], []).append(c)

    chunks = []
    for c in graph["chunks"]:
        text = chunk_text[c["id"]]
        kind = c["kind"]
        name = c["name"]
        start, end = c["start_line"], c["end_line"]

        if kind == "method":
            name = name.rsplit(".", 1)[-1]
        elif kind == "class":
            # header and fields only, methods are their own chunks
            lines = text.split("\n")
            skip = set()
            for child in children.get(c["id"], []):
                for line in range(child["start_line"], child["end_line"] + 1):
                    skip.add(line)
            text = "\n".join(
                line for offset, line in enumerate(lines) if start + offset not in skip
            ).strip()
        elif kind == "body":
            # top-level code split at blank lines, like the regex extractor
            lines = text.split("\n")
            run_start = None
            for i, line in enumerate(lines + [""]):
                if line.strip() and run_start is None:
                    run_start = i
                elif not line.strip() and run_start is not None:
                    body = "\n".join(lines[run_start:i]).strip()
                    if len(body) > 20:
                        s = start + run_start
                        e = start + i - 1
                        chunks.append({
                            "id": f"{c['file']}:L{s}-{e}:top@L{s}",
                            "name": f"top@L{s}",
                            "kind": "block",
                            "file": c["file"],
                            "start": s,
                            "end": e,
                            "text": body,
                            "title": f"{c['file']} top@L{s}",
                        })
                    run_start = None
            continue

        chunks.append({
            "id": f"{c['file']}:L{start}-{end}:{name}",
            "name": name,
            "kind": kind,
            "file": c["file"],
            "start": start,
            "end": end,
            "text": text.strip(),
            "title": f"{c['file']} {name}",
            "parent": raw[c["parent"]]["name"].rsplit(".", 1)[-1] if c["parent"] else None,
            "exported": c["exported"],
        })

    g = cg.Graph(chunks=chunks, files=graph["files"])
    defined, chunk_defines = {}, {}
    for chunk in chunks:
        g.defined_in[chunk["id"]] = chunk["file"]
        g.chunk_kind[chunk["id"]] = chunk["kind"]
        g.contains.setdefault(chunk["file"], set()).add(chunk["id"])
        if chunk["kind"] != "block":
            defined.setdefault(chunk["name"], set()).add(chunk["id"])
            chunk_defines.setdefault(chunk["id"], set()).add(chunk["name"])
        if chunk.get("exported"):
            g.exports.setdefault(chunk["file"], set()).add(chunk["name"])

    # Exported module constants live inside a block chunk, not a declaration chunk,
    # so the AST graph would under-report EXPORTS. Register them the way the regex
    # graph does, to keep the edge counts comparable.
    EXPORTED_DECL = re.compile(r"^\s*export\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)", re.M)
    for chunk in chunks:
        for name in EXPORTED_DECL.findall(chunk["text"]):
            g.exports.setdefault(chunk["file"], set()).add(name)
            if name not in defined:
                defined[name] = {chunk["id"]}
                chunk_defines.setdefault(chunk["id"], set()).add(name)
    g.defined = defined
    g.chunk_defines = chunk_defines

    imported_names: dict[str, set[str]] = {}
    for rel in graph["files"]:
        g.file_imports.setdefault(rel, set())
        g.exports.setdefault(rel, set())
        imported_names[rel] = set()
    for edge in graph["relations"]["IMPORTS"]:
        g.file_imports.setdefault(edge["file"], set()).add(edge["module"])
        g.module_files.setdefault(edge["module"], set()).add(edge["file"])
        for name in edge["names"]:
            imported_names[edge["file"]].add(name.replace("* as ", ""))

    # CALLS: prefer the checker-resolved symbol, fall back to the syntactic name.
    # Call sites are re-homed onto the converted chunk that contains them, because
    # splitting a body run into blocks changes the chunk ids.
    by_span: dict[str, list[dict]] = {}
    for c in chunks:
        by_span.setdefault(c["file"], []).append(c)

    def containing(file, line):
        best = None
        for c in by_span.get(file, ()):
            if c["start"] <= line <= c["end"]:
                if best is None or (c["end"] - c["start"]) < (best["end"] - best["start"]):
                    best = c
        return best["id"] if best else None

    by_chunk: dict[str, set[str]] = {}
    symbols = set(defined)
    for edge in graph["relations"]["CALLS"]:
        symbol = edge.get("callee_symbol") or edge["callee"].rsplit(".", 1)[-1].replace("[]", "")
        if symbol not in symbols and symbol not in imported_names.get(edge["file"], ()):
            continue
        owner_id = containing(edge["file"], edge["line"])
        if owner_id is None or symbol in chunk_defines.get(owner_id, set()):
            continue
        by_chunk.setdefault(owner_id, set()).add(symbol)
    g.callers = {}
    for chunk_id, symbols_called in by_chunk.items():
        g.calls[chunk_id] = symbols_called
        for symbol in symbols_called:
            g.callers.setdefault(symbol, set()).add(chunk_id)
    for chunk in chunks:
        g.calls.setdefault(chunk["id"], set())

    mentions: dict[str, set[str]] = {}
    for chunk in chunks:
        tokens = set(cg.code_tokens(chunk["text"]))
        for symbol in symbols:
            if symbol.lower() in tokens:
                mentions.setdefault(symbol, set()).add(chunk["id"])
    g.mentions = mentions

    cg.resolve_imports(g)
    return g


# --------------------------------------------------------------- qrel mapping
def map_qrels(queries, target):
    """Chunk id from one graph -> the other graph's chunk id, joined on (file, start)."""
    by_start = {(c["file"], c["start"]): c["id"] for c in target.chunks}
    by_file: dict[str, list[dict]] = {}
    for c in target.chunks:
        by_file.setdefault(c["file"], []).append(c)

    def enclosing(file, line):
        best = None
        for c in by_file.get(file, ()):
            if c["start"] <= line <= c["end"]:
                if best is None or (c["end"] - c["start"]) < (best["end"] - best["start"]):
                    best = c
        return best["id"] if best else None

    mapped, exact, fallback, dropped = [], 0, 0, 0
    for query in queries:
        qrels, misses = [], 0
        for chunk_id in query["qrels"]:
            m = ID_RE.match(chunk_id)
            if not m:
                misses += 1
                continue
            file, start = m.group("file"), int(m.group("start"))
            hit = by_start.get((file, start))
            if hit:
                exact += 1
            else:
                hit = enclosing(file, start)
                if hit:
                    fallback += 1
                else:
                    misses += 1
            if hit:
                qrels.append(hit)
        qrels = list(dict.fromkeys(qrels))
        if qrels and not misses:
            mapped.append({**query, "qrels": qrels})
        else:
            dropped += 1
    return mapped, {"exact": exact, "fallback": fallback, "queries_dropped": dropped}


def paired_rows(base_rows, cand_rows, systems, metrics=("both@5", "both@10", "ndcg@10", "recall@10")):
    out = {}
    for system in systems:
        for metric in metrics:
            diffs = [b[metric] - a[metric] for a, b in zip(base_rows[system], cand_rows[system])]
            wins = sum(d > 1e-12 for d in diffs)
            losses = sum(d < -1e-12 for d in diffs)
            out.setdefault(system, {})[metric] = {
                "delta": sum(diffs) / len(diffs),
                "wtl": [wins, len(diffs) - wins - losses, losses],
            }
    return out


def main() -> None:
    ts_graph = json.loads((OUT / "graph_ts7.json").read_text())
    chunk_text = json.loads((OUT / "chunk_text.json").read_text())

    regex_graph = cg.build_graph(runner.ROOT)
    queries = cg.generate_queries(regex_graph, per_type=50, seed=cg.SEED)

    ts7 = ts7_graph_object(ts_graph, chunk_text)
    mapped, coverage = map_qrels(queries, ts7)

    # The regex graph is scored on its own query set (its qrels use its own chunk
    # ids); the TS7 graph is scored on the id-mapped copy of the same queries.
    rows, systems_summary = {}, {}
    for label, graph, qs in (("regex", regex_graph, queries), ("ts7", ts7, mapped)):
        rows[label] = runner.run(graph, qs, cg.build_index(graph))
        systems_summary[label] = runner.summarize(rows[label])

    # ---- provenance control -------------------------------------------------
    # Each query set is generated from one graph's edges. Re-generate the same five
    # query types from the TS7 graph and score both graphs on both sets: if each
    # graph wins on its own queries, the numbers measure query provenance rather
    # than parser quality.
    cross = {"meta": {
        "queries_per_source": {},
        "why": (
            "Each query set is generated from one graph's edges and then id-mapped "
            "onto both graphs, so the only asymmetry left is which graph the "
            "relations were read off."
        ),
    }, "cells": {}}
    population = {"regex": regex_graph, "ts7": ts7}
    for source, graph in population.items():
        qs = cg.generate_queries(graph, per_type=50, seed=cg.SEED)
        cross["meta"]["queries_per_source"][source] = len(qs)
        for target_label, target in population.items():
            mapped_qs, cov = map_qrels(qs, target)
            summary_t = runner.summarize(
                runner.run(target, mapped_qs, cg.build_index(target))
            )
            cross["cells"][f"queries:{source}|graph:{target_label}"] = {
                "queries": len(mapped_qs),
                "qrel_mapping": cov,
                "porter_both@10": summary_t["porter"]["both@10"],
                "porter_ndcg@10": summary_t["porter"]["ndcg@10"],
                "rrf_both@10": summary_t["rrf_porter_entity_graph"]["both@10"],
                "rrf_ndcg@10": summary_t["rrf_porter_entity_graph"]["ndcg@10"],
            }

    # ---- ablation: TS7 boundaries, regex chunk text -------------------------
    # The regex brace matcher ends ~14% of matched chunks early, so its document
    # text is truncated. This separates "different graph" from "damaged documents".
    regex_text = {(c["file"], c["start"]): c["text"] for c in regex_graph.chunks}
    swapped = 0
    for chunk in ts7.chunks:
        other = regex_text.get((chunk["file"], chunk["start"]))
        if other is not None:
            chunk["text"] = other
            swapped += 1
    rows["ts7_boundaries_regex_text"] = runner.run(ts7, mapped, cg.build_index(ts7))
    systems_summary["ts7_boundaries_regex_text"] = runner.summarize(
        rows["ts7_boundaries_regex_text"]
    )

    result = {
        "meta": {
            "kind": "same queries and qrels, same channel code, two graphs",
            "queries_regex_graph": len(queries),
            "queries_kept_for_ts7": len(mapped),
            "qrel_mapping": coverage,
            "note": (
                "qrels are graph-derived (circular by construction). The only things "
                "that change between runs are the parser behind the graph and, in the "
                "ablation, the chunk text it produced."
            ),
            "ablation": {
                "name": "ts7_boundaries_regex_text",
                "chunks_with_regex_text": swapped,
                "purpose": "isolate chunk-boundary fidelity from chunk-text truncation",
            },
        },
        "graphs": {},
        "systems": systems_summary,
        "ts7_minus_regex_same_queries": paired_rows(rows["regex"], rows["ts7"], runner.SYSTEMS),
        "ts7_minus_regex_text_ablation": paired_rows(
            rows["ts7_boundaries_regex_text"], rows["ts7"], runner.SYSTEMS
        ),
        "provenance_control": cross,
    }
    for label, graph in (("regex", regex_graph), ("ts7", ts7)):
        result["graphs"][label] = graph.stats() | {
            "chunk_kinds": dict(sorted(Counter(c["kind"] for c in graph.chunks).items()))
        }

    (OUT / "shared_queries.json").write_text(json.dumps(result, indent=1))

    print(f"queries {len(queries)} -> {len(mapped)} usable on the TS7 graph")
    print(f"qrel mapping: {coverage}  |  chunk text swapped for ablation: {swapped}")
    print(f"\n{'graph':9} {'chunks':>7} {'calls':>7} {'imports':>8} {'exports':>8} {'with_calls':>11}")
    for label, stats in result["graphs"].items():
        print(f"{label:9} {stats['chunks']:>7} {stats['call_edges']:>7} "
              f"{stats['import_edges']:>8} {stats['export_edges']:>8} "
              f"{stats['chunks_with_calls']:>11}")

    print(f"\n{'system':26} {'variant':>26} {'both@5':>8} {'both@10':>8} {'nDCG@10':>8} {'R@10':>7}")
    for system in runner.SYSTEMS:
        for label in ("regex", "ts7", "ts7_boundaries_regex_text"):
            row = systems_summary[label][system]
            print(f"{system:26} {label:>26} {row['both@5']:8.3f} {row['both@10']:8.3f} "
                  f"{row['ndcg@10']:8.3f} {row['recall@10']:7.3f}")

    print("\nprovenance control (both@10 | nDCG@10)")
    print(f"{'queries from':>14} {'graph':>7} {'n':>5} {'porter':>8} {'rrf':>8} | "
          f"{'porter':>8} {'rrf':>8}")
    for cell_key, cell in result["provenance_control"]["cells"].items():
        source, target = cell_key.split("|")
        print(f"{source.split(':')[1]:>14} {target.split(':')[1]:>7} {cell['queries']:>5} "
              f"{cell['porter_both@10']:8.3f} {cell['rrf_both@10']:8.3f} | "
              f"{cell['porter_ndcg@10']:8.3f} {cell['rrf_ndcg@10']:8.3f}")

    for title, key in (
        ("ts7 minus regex (same queries)", "ts7_minus_regex_same_queries"),
        ("ts7 minus regex-text ablation", "ts7_minus_regex_text_ablation"),
    ):
        print(f"\n{title}")
        print(f"{'system':26} {'metric':>9} {'delta':>9} {'W/T/L':>14}")
        for system, metrics in result[key].items():
            for metric, entry in metrics.items():
                w, t, l = entry["wtl"]
                print(f"{system:26} {metric:>9} {entry['delta']:+9.4f} {w:5}/{t:5}/{l:5}")


if __name__ == "__main__":
    main()
