"""Code-graph EBR: the HotpotQA entity-bridge idea moved onto exact structure.

HotpotQA entities are prose spans guessed by capitalisation. A code graph has
typed, exact relations instead: DEFINED_IN, CALLS, IMPORTS, EXPORTS, CONTAINS.
The question this probe asks is narrow: when retrieval is given exact symbol and
call/import edges, does graph expansion beat Porter BM25 on evidence-chain
queries?

Everything here is regex parsing over the repo's own JavaScript. No external
parser, no embeddings, no LLM. See experiments/stages/CODE_GRAPH.md for the
claim boundary: the qrels are generated from the same graph the retrieval
channels traverse, so this measures *capability to follow structural relations*,
not human-judged relevance.
"""
from __future__ import annotations

import copy
import math
import posixpath
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import snowballstemmer

import harness

SCAN_DIRS = ("src", "scripts", "test")
CODE_EXT = (".js",)
SEED = 1729
RRF_K = 60
SEED_CHUNKS = 10
HUB_DF_FRAC = 0.05
HOP2_WEIGHT = 0.5
IMPORT_WEIGHT = 0.5
EXPORT_WEIGHT = 0.75

PORTER = snowballstemmer.stemmer("porter")

# ---------------------------------------------------------------------------
# Tokenisation: identifiers survive intact, plus their camel/snake subwords.
# This mirrors scripts/index-codebase.js dualTokenizeCode(), in Python.
# ---------------------------------------------------------------------------
IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*|\d+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
WORD_RE = re.compile(r"[a-z0-9]+")


def split_identifier(token: str) -> list[str]:
    parts: list[str] = []
    for chunk in token.split("_"):
        parts.extend(CAMEL_RE.sub(" ", chunk).split())
    return [part.lower() for part in parts if len(part) >= 2]


def code_tokens(text: str) -> list[str]:
    """Lowercased identifiers, each also emitted as its subword parts."""
    out: list[str] = []
    for match in IDENT_RE.finditer(text):
        token = match.group(0).lower()
        out.append(token)
        if "_" in token or any(c.isupper() for c in match.group(0)):
            out.extend(split_identifier(match.group(0)))
    return out


def porter(tokens: list[str]) -> list[str]:
    return PORTER.stemWords(tokens)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
JS_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "function", "return", "else",
    "do", "try", "with", "typeof", "new", "await", "yield", "case", "throw",
    "constructor", "super", "of", "in", "delete", "void", "instanceof",
}

CLASS_DECL = re.compile(r"^(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)")
FUNCTION_DECL = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)",
)
ASSIGN_DECL = re.compile(
    r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)",
)
METHOD_DECL = re.compile(
    r"^\s{1,8}(?:async\s+)?(?:static\s+)?(?:get\s+|set\s+)?"
    r"([A-Za-z_$][\w$]*)\s*\(([^;]*)\)\s*\{",
)

IMPORT_FROM = re.compile(
    r"""^\s*import\s+(?:([\w$*{}\s,]+)\s+from\s+)?["']([^"']+)["']""",
)
EXPORT_FROM = re.compile(r"""^\s*export\s+(?:\*|\{[^}]*\})\s+from\s+["']([^"']+)["']""")
EXPORT_BLOCK = re.compile(r"^\s*export\s*\{([^}]*)\}")
EXPORT_NAMED = re.compile(r"^\s*export\s+(?:default\s+)?(?:async\s+)?"
                          r"(?:function\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)")
CALL_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*\(")


def strip_literals(line: str, in_block: bool) -> tuple[str, bool]:
    """Drop string/regex/comment content so brace counting sees only structure."""
    out: list[str] = []
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if in_block:
            if line.startswith("*/", i):
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if line.startswith("/*", i):
            in_block = True
            i += 2
            continue
        if line.startswith("//", i):
            break
        if ch in "\"'`":
            quote = ch
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), in_block


def depth_map(lines: list[str]) -> list[int]:
    """depth[i] = brace depth at the start of line i."""
    depths: list[int] = []
    depth = 0
    in_block = False
    for line in lines:
        depths.append(depth)
        clean, in_block = strip_literals(line, in_block)
        depth += clean.count("{") - clean.count("}")
    return depths


def block_end(depths: list[int], start: int) -> int:
    """Last line of the construct starting at `start`."""
    base = depths[start]
    for i in range(start + 1, len(depths)):
        if depths[i] <= base:
            return i - 1
    return len(depths) - 1


def scan_file(rel: str, source: str) -> list[dict]:
    lines = source.split("\n")
    depths = depth_map(lines)
    constructs: list[dict] = []

    i = 0
    while i < len(lines):
        if depths[i] == 0:
            clean, _ = strip_literals(lines[i], False)
            for kind, pattern in (
                ("class", CLASS_DECL), ("function", FUNCTION_DECL), ("function", ASSIGN_DECL),
            ):
                match = pattern.match(clean)
                if match:
                    end = block_end(depths, i)
                    constructs.append({"start": i, "end": end, "name": match.group(1), "kind": kind})
                    i = end + 1
                    break
            else:
                i += 1
            continue
        i += 1

    methods: list[dict] = []
    for construct in [c for c in constructs if c["kind"] == "class"]:
        for j in range(construct["start"] + 1, construct["end"] + 1):
            if depths[j] != 1:
                continue
            clean, _ = strip_literals(lines[j], False)
            match = METHOD_DECL.match(clean)
            if not match or match.group(1) in JS_KEYWORDS:
                continue
            if any(m["start"] == j for m in methods):
                continue
            end = block_end(depths, j)
            methods.append({"start": j, "end": end, "name": match.group(1), "kind": "method",
                            "parent": construct["name"]})

    return make_chunks(rel, lines, constructs, methods)


def make_chunks(rel: str, lines: list[str], constructs: list[dict], methods: list[dict]) -> list[dict]:
    method_lines: dict[int, list[dict]] = defaultdict(list)
    for method in methods:
        owner = max(
            (c for c in constructs if c["kind"] == "class" and c["start"] < method["start"] <= c["end"]),
            key=lambda c: c["start"],
        )
        method_lines[owner["start"]].append(method)

    chunks: list[dict] = []
    for construct in constructs:
        body = lines[construct["start"]:construct["end"] + 1]
        if construct["kind"] == "class":
            skip = set()
            for method in method_lines[construct["start"]]:
                skip.update(range(method["start"], method["end"] + 1))
            body = [
                line for offset, line in enumerate(body)
                if construct["start"] + offset not in skip
            ]
        name = construct["name"]
        chunks.append({
            "name": name,
            "kind": construct["kind"],
            "file": rel,
            "start": construct["start"] + 1,
            "end": construct["end"] + 1,
            "text": "\n".join(body).strip(),
        })

    for method in methods:
        chunks.append({
            "name": method["name"],
            "kind": "method",
            "parent": method["parent"],
            "file": rel,
            "start": method["start"] + 1,
            "end": method["end"] + 1,
            "text": "\n".join(lines[method["start"]:method["end"] + 1]).strip(),
        })

    covered: set[int] = set()
    for construct in constructs:
        covered.update(range(construct["start"], construct["end"] + 1))
    chunks.extend(top_level_blocks(rel, lines, covered))

    for chunk in chunks:
        chunk["id"] = f"{rel}:L{chunk['start']}-{chunk['end']}:{chunk['name']}"
        chunk["title"] = f"{rel} {chunk['name']}"
    return chunks


def top_level_blocks(rel: str, lines: list[str], covered: set[int]) -> list[dict]:
    """Top-level lines outside any function/class, split at blank lines.

    Imports and module-level constants become separate chunks, so an exported
    constant does not hide inside the import preamble.
    """
    blocks: list[dict] = []
    start = None
    for i in range(len(lines) + 1):
        free = i < len(lines) and i not in covered and bool(lines[i].strip())
        if free and start is None:
            start = i
            continue
        if not free and start is not None:
            text = "\n".join(lines[start:i]).strip()
            if len(text) > 20:
                blocks.append({
                    "name": f"top@L{start + 1}",
                    "kind": "block",
                    "file": rel,
                    "start": start + 1,
                    "end": i,
                    "text": text,
                })
            start = None
    return blocks


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
@dataclass
class Graph:
    chunks: list[dict]
    files: list[str]
    defined: dict[str, set[str]] = field(default_factory=dict)          # symbol -> chunk ids
    chunk_defines: dict[str, set[str]] = field(default_factory=dict)   # chunk id -> symbols it defines
    defined_in: dict[str, str] = field(default_factory=dict)           # chunk id -> file
    calls: dict[str, set[str]] = field(default_factory=dict)           # chunk id -> symbols
    callers: dict[str, set[str]] = field(default_factory=dict)         # symbol -> chunk ids
    file_imports: dict[str, set[str]] = field(default_factory=dict)    # file -> modules
    module_files: dict[str, set[str]] = field(default_factory=dict)    # module -> files
    module_targets: dict[str, set[str]] = field(default_factory=dict)  # file -> imported files
    module_importers: dict[str, set[str]] = field(default_factory=dict)  # file -> importing files
    exports: dict[str, set[str]] = field(default_factory=dict)         # file -> symbols
    contains: dict[str, set[str]] = field(default_factory=dict)        # file -> chunk ids
    mentions: dict[str, set[str]] = field(default_factory=dict)        # symbol -> chunk ids (textual)
    chunk_kind: dict[str, str] = field(default_factory=dict)           # chunk id -> function/class/method/block

    def stats(self) -> dict:
        call_edges = sum(len(v) for v in self.calls.values())
        import_edges = sum(len(v) for v in self.file_imports.values())
        export_edges = sum(len(v) for v in self.exports.values())
        contains_edges = sum(len(v) for v in self.contains.values())
        out_degrees = [len(v) for v in self.calls.values()]
        cross_file = sum(
            1
            for chunk_id, symbols in self.calls.items()
            for symbol in symbols
            if all(self.defined_in.get(target) != self.defined_in.get(chunk_id)
                   for target in self.defined.get(symbol, ()))
        )
        return {
            "files": len(self.files),
            "chunks": len(self.chunks),
            "symbols": len(self.defined),
            "call_edges": call_edges,
            "import_edges": import_edges,
            "import_edges_resolved": sum(len(v) for v in self.module_targets.values()),
            "export_edges": export_edges,
            "contains_edges": contains_edges,
            "defined_in_edges": len(self.defined_in),
            "cross_file_call_edges": cross_file,
            "multi_definition_symbols": sum(1 for v in self.defined.values() if len(v) > 1),
            "avg_call_out_degree": sum(out_degrees) / max(len(out_degrees), 1),
            "avg_out_degree_with_imports": (
                sum(len(self.calls.get(c, ())) + len(self.file_imports.get(self.defined_in[c], ()))
                    for c in self.defined_in)
                / max(len(self.defined_in), 1)
            ),
            "chunks_with_calls": sum(1 for v in self.calls.values() if v),
            "files_with_imports": sum(1 for v in self.file_imports.values() if v),
        }


def read_repo(root: Path) -> list[tuple[str, str]]:
    files = []
    for folder in SCAN_DIRS:
        for path in sorted((root / folder).rglob("*.js")):
            if "node_modules" in path.parts:
                continue
            files.append((path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")))
    return files


def build_graph(root: Path) -> Graph:
    files = read_repo(root)
    chunks: list[dict] = []
    for rel, source in files:
        chunks.extend(scan_file(rel, source))

    graph = Graph(chunks=chunks, files=[rel for rel, _ in files])

    defined: dict[str, set[str]] = defaultdict(set)
    chunk_defines: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        if chunk["kind"] != "block":
            defined[chunk["name"]].add(chunk["id"])
            chunk_defines[chunk["id"]].add(chunk["name"])
        graph.defined_in[chunk["id"]] = chunk["file"]
        graph.chunk_kind[chunk["id"]] = chunk["kind"]
        graph.contains.setdefault(chunk["file"], set()).add(chunk["id"])
    graph.defined = dict(defined)
    graph.chunk_defines = dict(chunk_defines)

    imported_names: dict[str, set[str]] = defaultdict(set)
    export_lines: list[tuple[str, int, str]] = []
    for rel, source in files:
        for line_no, line in enumerate(source.split("\n"), start=1):
            from_match = IMPORT_FROM.match(line)
            if from_match:
                graph.file_imports.setdefault(rel, set()).add(from_match.group(2))
                graph.module_files.setdefault(from_match.group(2), set()).add(rel)
                if from_match.group(1):
                    for name in re.findall(r"[A-Za-z_$][\w$]*", from_match.group(1)):
                        if name not in ("as",):
                            imported_names[rel].add(name)
            export_from = EXPORT_FROM.match(line)
            if export_from:
                graph.file_imports.setdefault(rel, set()).add(export_from.group(1))
                graph.module_files.setdefault(export_from.group(1), set()).add(rel)
            block = EXPORT_BLOCK.match(line)
            if block:
                for part in block.group(1).split(","):
                    name = part.split(" as ")[0].strip()
                    if name:
                        graph.exports.setdefault(rel, set()).add(name)
            named = EXPORT_NAMED.match(line)
            if named:
                graph.exports.setdefault(rel, set()).add(named.group(1))
                export_lines.append((rel, line_no, named.group(1)))
        graph.file_imports.setdefault(rel, set())
        graph.exports.setdefault(rel, set())

    resolve_export_chunks(graph, export_lines)
    symbols = set(graph.defined)

    graph.callers = defaultdict(set)
    for chunk in chunks:
        called = {
            name for name in CALL_RE.findall(chunk["text"])
            if name not in JS_KEYWORDS and (name in symbols or name in imported_names[chunk["file"]])
        }
        called.discard(chunk["name"])
        graph.calls[chunk["id"]] = called
        for name in called:
            graph.callers[name].add(chunk["id"])

    mentions: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        tokens = set(code_tokens(chunk["text"]))
        for symbol in symbols:
            if symbol.lower() in tokens:
                mentions[symbol].add(chunk["id"])
    graph.mentions = dict(mentions)

    resolve_imports(graph)
    return graph


def resolve_export_chunks(graph: Graph, export_lines: list[tuple[str, int, str]]) -> None:
    """Register `export const NAME = ...` as a symbol bound to the chunk holding it."""
    spans: dict[str, list[dict]] = defaultdict(list)
    for chunk in graph.chunks:
        spans[chunk["file"]].append(chunk)
    for rel, line_no, name in export_lines:
        if name in graph.defined:
            continue
        holders = [
            chunk for chunk in spans.get(rel, ())
            if chunk["start"] <= line_no <= chunk["end"]
        ]
        if not holders:
            continue
        holder = min(holders, key=lambda chunk: (chunk["end"] - chunk["start"], chunk["start"]))
        graph.defined[name] = {holder["id"]}
        graph.chunk_defines.setdefault(holder["id"], set()).add(name)


def resolve_imports(graph: Graph) -> None:
    """Relative module strings -> repo files, so IMPORTS is a file-to-file edge."""
    files = set(graph.files)
    graph.module_targets = defaultdict(set)
    graph.module_importers = defaultdict(set)
    for rel in graph.files:
        for module in graph.file_imports.get(rel, ()):
            target = resolve_module(rel, module, files)
            if target and target != rel:
                graph.module_targets[rel].add(target)
                graph.module_importers[target].add(rel)
    graph.module_targets = dict(graph.module_targets)
    graph.module_importers = dict(graph.module_importers)


def resolve_module(rel: str, module: str, files: set[str]) -> str | None:
    if not module.startswith("."):
        return None
    base = posixpath.dirname(rel)
    candidate = posixpath.normpath(posixpath.join(base, module))
    for path in (candidate, candidate + ".js", candidate + "/index.js"):
        if path in files:
            return path
    return None


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
def build_index(graph: Graph, stem: bool = True) -> harness.Index:
    ids = [chunk["id"] for chunk in graph.chunks]
    titles, bodies = [], []
    for chunk in graph.chunks:
        title = code_tokens(chunk["title"] + " " + chunk["kind"])
        body = code_tokens(chunk["text"])
        titles.append(porter(title) if stem else title)
        bodies.append(porter(body) if stem else body)
    return harness.Index(doc_ids=ids, title_tokens=titles, body_tokens=bodies)


def symbol_df(graph: Graph) -> dict[str, int]:
    return {symbol: len(chunks) for symbol, chunks in graph.mentions.items()}


def idf(n: int, df: int) -> float:
    return math.log(1 + (n - df + 0.5) / (df + 0.5)) if df else 0.0


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
def query_symbols(graph: Graph, raw_query: str) -> set[str]:
    """Exact defined-symbol mentions in the query. No capitalisation heuristics."""
    words = set(re.findall(r"[A-Za-z_$][\w$]*", raw_query))
    lowered = {w.lower() for w in words}
    return {symbol for symbol in graph.defined if symbol in words or symbol.lower() in lowered}


def entity_scores(graph: Graph, position: dict[str, int], raw_query: str):
    import numpy as np
    scores = np.zeros(len(position))
    df_all = symbol_df(graph)
    n = len(position)
    for symbol in sorted(query_symbols(graph, raw_query)):
        df = df_all.get(symbol, 0)
        if df < 1 or df > max(2, HUB_DF_FRAC * n):
            continue
        weight = idf(n, df)
        for chunk_id in graph.mentions.get(symbol, ()):
            scores[position[chunk_id]] += weight
    return scores


def call_neighbours(graph: Graph, chunk_id: str, df_all: dict[str, int], n: int, cap: float):
    """CALLS, both directions, anchored on the definition.

    out: chunk -> the chunks that define a symbol it calls.
    in:  chunk -> the chunks that call a symbol it defines.
    """
    out = []
    for symbol in sorted(graph.calls.get(chunk_id, ())):
        df = df_all.get(symbol, 0)
        if df < 1 or df > cap:
            continue
        weight = idf(n, df)
        for target in graph.defined.get(symbol, ()):
            if target != chunk_id:
                out.append((target, weight))
    for symbol in sorted(graph.chunk_defines.get(chunk_id, ())):
        df = df_all.get(symbol, 0)
        if df < 1 or df > cap:
            continue
        weight = idf(n, df)
        for caller in graph.callers.get(symbol, ()):
            if caller != chunk_id:
                out.append((caller, weight))
    return out


def import_neighbours(graph: Graph, chunk_id: str, n_files: int):
    file = graph.defined_in.get(chunk_id)
    if file is None:
        return []
    out = []
    for target in sorted(graph.module_targets.get(file, ())):
        importers = len(graph.module_importers.get(target, ()))
        weight = IMPORT_WEIGHT * idf(n_files, max(importers, 1))
        members = graph.contains.get(target, ())
        for other in members:
            out.append((other, weight / max(len(members), 1)))
    for importer in sorted(graph.module_importers.get(file, ())):
        importers = len(graph.module_importers.get(file, ()))
        weight = IMPORT_WEIGHT * idf(n_files, max(importers, 1))
        members = graph.contains.get(importer, ())
        for other in members:
            out.append((other, weight / max(len(members), 1)))
    return out


def export_neighbours(graph: Graph, chunk_id: str):
    """DEFINED_IN -> EXPORTS: the exported entry points of the chunk's file.

    Bounded by the export count (1-6 per file here), unlike CONTAINS, which would
    fan out to every chunk in the file. This is the path an export question asks for.
    """
    file = graph.defined_in.get(chunk_id)
    if file is None:
        return []
    out = []
    for symbol in sorted(graph.exports.get(file, ())):
        for target in graph.defined.get(symbol, ()):
            if target != chunk_id and graph.defined_in.get(target) == file:
                out.append((target, EXPORT_WEIGHT))
    return out


def graph_scores(graph: Graph, position: dict[str, int], seeds: list[str]):
    """Two-hop expansion over CALLS (both directions) and resolved IMPORTS."""
    import numpy as np
    scores = np.zeros(len(position))
    df_all = symbol_df(graph)
    n = len(position)
    n_files = len(graph.files)
    cap = max(2, HUB_DF_FRAC * n)
    frontier = [(chunk_id, 1.0 / (rank + 1)) for rank, chunk_id in enumerate(seeds)]
    for decay in (1.0, HOP2_WEIGHT):
        reached: dict[str, float] = {}
        for chunk_id, weight in frontier:
            for neighbour, edge in call_neighbours(graph, chunk_id, df_all, n, cap):
                reached[neighbour] = reached.get(neighbour, 0.0) + weight * decay * edge
            for neighbour, edge in import_neighbours(graph, chunk_id, n_files):
                if neighbour != chunk_id:
                    reached[neighbour] = reached.get(neighbour, 0.0) + weight * decay * edge
            for neighbour, edge in export_neighbours(graph, chunk_id):
                reached[neighbour] = reached.get(neighbour, 0.0) + weight * decay * edge
        for chunk_id, score in sorted(reached.items()):
            scores[position[chunk_id]] += score
        frontier = [(cid, value) for cid, value in sorted(reached.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]
        if not frontier:
            break
    return scores


# ---------------------------------------------------------------------------
# Query generation: deterministic, graph-derived qrels
# ---------------------------------------------------------------------------
def _unique_by_construction(queries: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for query in queries:
        if query["qid"] in seen:
            continue
        seen.add(query["qid"])
        out.append(query)
    return out


def same_file_queries(graph: Graph) -> list[dict]:
    """`Which chunk calls X and is defined in the same file as Y?`"""
    out = []
    for symbol, callers in sorted(graph.callers.items()):
        for caller in sorted(callers):
            file = graph.defined_in.get(caller)
            if file is None:
                continue
            out.extend(_same_file_for_caller(graph, symbol, caller, file))
    return out


def _same_file_for_caller(graph: Graph, symbol: str, caller: str, file: str) -> list[dict]:
    # the answer must be unique inside the file, otherwise the question is ambiguous
    answer_in_file = [other for other in graph.callers.get(symbol, ())
                      if graph.defined_in.get(other) == file]
    if answer_in_file != [caller]:
        return []

    caller_name = graph_chunk_name(graph, caller)
    local = sorted(
        name for name, chunks in graph.defined.items()
        if name not in (symbol, caller_name)
        and any(graph.defined_in[c] == file for c in chunks)
    )
    out = []
    for target in local:
        anchor = sorted(c for c in graph.defined[target] if graph.defined_in.get(c) == file)[0]
        if anchor == caller:
            continue
        out.append({
            "qid": f"same_file:{symbol}:{target}",
            "type": "same_file",
            "text": f"Which chunk calls {symbol} and is defined in the same file as {target}?",
            "qrels": sorted({caller, anchor}),
            "answer": caller,
            "symbols": [symbol, target],
        })
    return out


def graph_chunk_name(graph: Graph, chunk_id: str) -> str:
    for chunk in graph.chunks:
        if chunk["id"] == chunk_id:
            return chunk["name"]
    return chunk_id


def import_call_queries(graph: Graph) -> list[dict]:
    out = []
    for module, files in sorted(graph.module_files.items()):
        if not module.startswith("."):
            continue
        for file in sorted(files):
            for chunk_id in sorted(graph.contains.get(file, ())):
                for symbol in sorted(graph.calls.get(chunk_id, ())):
                    defined = sorted(graph.defined.get(symbol, ()))
                    if not defined:
                        continue
                    if any(graph.defined_in.get(c) == file for c in defined):
                        continue  # symbol is local to the importing file: no bridge to cross
                    evidence = {chunk_id, defined[0]}
                    out.append({
                        "qid": f"import_call:{module}:{symbol}:{chunk_id}",
                        "type": "import_call",
                        "text": f"Which chunk calls {symbol} in a file that imports {module}?",
                        "qrels": sorted(evidence),
                        "answer": chunk_id,
                        "symbols": [symbol],
                        "module": module,
                    })
    return out


def call_chain_queries(graph: Graph) -> list[dict]:
    """`Which chunk calls a function that calls X?` -- the bridge name is not in the query."""
    out = []
    for symbol, callers in sorted(graph.callers.items()):
        for middle in sorted(callers):
            middle_name = graph_chunk_name(graph, middle)
            if middle_name == symbol:
                continue
            up = [caller for caller in graph.callers.get(middle_name, ()) if caller != middle]
            if not up:
                continue
            evidence = {middle} | set(up)
            if len(evidence) > 4:
                continue
            out.append({
                "qid": f"two_hop:{symbol}:{middle}",
                "type": "two_hop",
                "text": f"Which chunk calls a function that calls {symbol}?",
                "qrels": sorted(evidence),
                "answer": sorted(up)[0],
                "symbols": [symbol],
                "chain": [symbol, middle_name],
            })
    return out


def export_import_queries(graph: Graph) -> list[dict]:
    """`Which exports are defined in the file that imports Z and defines W?`

    The file is pinned by (imports Z, defines W); the answers are its exported
    chunks, which do not contain Z or W and are therefore a true second hop.
    Class containers are excluded: a 1000-line class is not a retrieval unit
    comparable to a function chunk.
    """
    out = []
    for module, files in sorted(graph.module_files.items()):
        if not module.startswith("."):
            continue
        for file in sorted(files):
            export_chunks = {
                name: sorted(
                    c for c in graph.defined.get(name, set())
                    if graph.defined_in.get(c) == file and graph.chunk_kind.get(c) != "class"
                )
                for name in graph.exports.get(file, set())
            }
            export_chunks = {name: chunks for name, chunks in export_chunks.items() if chunks}
            evidence_answers = {chunk for chunks in export_chunks.values() for chunk in chunks}
            if not 1 <= len(evidence_answers) <= 3:
                continue
            for symbol in sorted(graph.defined):
                local = [c for c in graph.defined[symbol] if graph.defined_in.get(c) == file]
                if not local or symbol in export_chunks:
                    continue
                pinned = [
                    other for other in graph.files
                    if module in graph.file_imports.get(other, set())
                    and any(graph.defined_in.get(c) == other for c in graph.defined[symbol])
                ]
                if pinned != [file]:
                    continue
                evidence = set(evidence_answers) | {local[0]}
                if len(evidence) < 2:
                    continue
                out.append({
                    "qid": f"export_import:{module}:{symbol}:{file}",
                    "type": "export_import",
                    "text": (f"Which exports are defined in the file that imports {module} "
                             f"and defines {symbol}?"),
                    "qrels": sorted(evidence),
                    "answer": sorted(evidence_answers)[0],
                    "symbols": [symbol],
                    "module": module,
                })
    return out


def call_two_queries(graph: Graph) -> list[dict]:
    out = []
    for chunk_id, called in sorted(graph.calls.items()):
        symbols = sorted(s for s in called if graph.defined.get(s))
        for i, first in enumerate(symbols):
            for second in symbols[i + 1:]:
                evidence = {chunk_id}
                for symbol in (first, second):
                    defined = sorted(graph.defined[symbol])
                    if defined:
                        evidence.add(defined[0])
                if len(evidence) < 3:
                    continue
                out.append({
                    "qid": f"calls_two:{first}:{second}:{chunk_id}",
                    "type": "calls_two",
                    "text": f"Which chunk calls both {first} and {second}?",
                    "qrels": sorted(evidence),
                    "answer": chunk_id,
                    "symbols": [first, second],
                })
    return out


GENERATORS = {
    "same_file": same_file_queries,
    "import_call": import_call_queries,
    "two_hop": call_chain_queries,
    "export_import": export_import_queries,
    "calls_two": call_two_queries,
}


def generate_queries(graph: Graph, per_type: int = 50, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    queries: list[dict] = []
    for name, generator in GENERATORS.items():
        candidates = [
            query for query in _dedupe(_unique_by_construction(generator(graph)))
            if len(query["qrels"]) >= 2
        ]
        rng.shuffle(candidates)
        queries.extend(candidates[:per_type])
    return queries


def _dedupe(queries: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for query in queries:
        key = (query["type"], tuple(query["qrels"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(query)
    return out


# ---------------------------------------------------------------------------
# Graph noise: parsers are lossy, so measure how much of the advantage survives
# ---------------------------------------------------------------------------
def perturb(graph: Graph, drop: float, seed: int = SEED) -> Graph:
    """Drop a fraction of CALLS/EXPORTS/IMPORTS edges used by the retrieval channel.

    Definitions are kept: a parser rarely invents functions, but it does miss
    calls. The qrels stay clean, so this measures channel robustness, not
    benchmark corruption.
    """
    rng = random.Random(seed)
    clone = copy.copy(graph)
    clone.calls = {
        chunk_id: {symbol for symbol in sorted(symbols) if rng.random() > drop}
        for chunk_id, symbols in graph.calls.items()
    }
    callers: dict[str, set[str]] = defaultdict(set)
    for chunk_id, symbols in clone.calls.items():
        for symbol in symbols:
            callers[symbol].add(chunk_id)
    clone.callers = dict(callers)
    clone.exports = {
        file: {symbol for symbol in sorted(symbols) if rng.random() > drop}
        for file, symbols in graph.exports.items()
    }
    clone.module_targets = {
        file: {target for target in sorted(targets) if rng.random() > drop}
        for file, targets in graph.module_targets.items()
    }
    importers: dict[str, set[str]] = defaultdict(set)
    for file, targets in clone.module_targets.items():
        for target in targets:
            importers[target].add(file)
    clone.module_importers = dict(importers)
    return clone


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(ranked: list[str], qrels: list[str], k: int = 10) -> dict:
    rel = set(qrels)
    top = ranked[:k]
    out = {}
    for cutoff in (5, 10):
        hits = rel & set(ranked[:cutoff])
        out[f"both@{cutoff}"] = 1.0 if rel <= set(ranked[:cutoff]) else 0.0
        out[f"recall@{cutoff}"] = len(hits) / len(rel)
    dcg = sum(1 / math.log2(i + 2) for i, doc in enumerate(top) if doc in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    first = next((i + 1 for i, doc in enumerate(top) if doc in rel), None)
    out["ndcg@10"] = dcg / ideal if ideal else 0.0
    out["mrr@10"] = 1 / first if first else 0.0
    return out


METRICS = ("both@5", "both@10", "ndcg@10", "recall@10", "mrr@10")


def paired(base: list[dict], candidate: list[dict], metric: str, resamples: int = 1000,
           seed: int = SEED) -> dict:
    import numpy as np
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
