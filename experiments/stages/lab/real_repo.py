"""Real, PR-derived code retrieval: the non-circular test for the code-graph direction.

`code_graph.py` generates queries *from* the graph it then uses for retrieval, which
makes the graph channel correct by construction. This module removes that loop:

    query  = the title + body of a merged pull request
    qrels  = the files that pull request actually changed

Nothing in the label comes from the graph, and nothing in the retriever sees the
labels. What the graph can contribute is only what it can extract from the PR text:
symbols it defines, files it names, and relation words ("calls", "imports", "exports").

Three honesty notes that are part of the result, not caveats bolted on afterwards:

1. **PR-derived qrels are noisy.** A PR changes its test file because the test had to
   be updated, not because the test file best answers the description. Every metric
   here is therefore measured against "files the author touched", which is a broader
   and less intentional set than "files a developer would want".
2. **Temporal leakage.** The corpus and the graph are built at repo HEAD. A PR merged
   in 2023 is scored against 2026 source. `file_drift` measures this directly: the
   fraction of a file's lines that changed between the PR's base commit and HEAD.
   The 2025+ subset is reported separately and has ~20x less drift.
3. **No tuning.** Every constant below was fixed before the run. Nothing reads qrels
   except the metric functions. Two parser bugs were fixed after the first retrieval
   run (multi-line signatures were truncated; a brace-less declaration could swallow
   the next one); no constant, filter or query changed with them.

The parser is a tolerant line/brace scanner over TypeScript. `TS7_GRAPH.md` measured
this family of extractor against the TS7 AST on JavaScript and found ~11% false call
edges and 19/140 truncated chunk ranges; expect at least that much loss here. See
REAL_REPO.md for the measured declaration recall and line coverage.
"""
from __future__ import annotations

import difflib
import json
import math
import posixpath
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import snowballstemmer

import code_graph as cg
import fusion
import harness

# --------------------------------------------------------------- frozen constants

SCAN_DIRS = ("src", "tests", "examples")
CODE_EXT = (".ts", ".tsx", ".js", ".jsx")
SEED = 1729
RESAMPLES = 1000

RRF_K = 60
POOL_DEPTH = 100
SEED_CHUNKS = cg.SEED_CHUNKS
HUB_DF_FRAC = cg.HUB_DF_FRAC

# CESR-code constants, copied from cesr_code.py and *not* tuned here.
CESR_BUDGET = 2000
MU = 0.2
COST_EXPONENT = 0.35
BM25_TIE = 0.01
MAX_SYMBOL_CONSTRAINTS = 4
MAX_FILE_CONSTRAINTS = 3
SMALL_SET = 3          # a coverage set this small is treated as universal
FILE_RANK_DEPTH = 10

# PR selection, fixed before any retrieval number was produced.
MIN_MERGED = "2023-01-01"
MAX_CHANGED_FILES = 5
MIN_CHANGED_FILES = 1
MIN_QUERY_WORDS = 8
PR_LIMIT = 1000

# File basenames too generic to be a useful query signal.
FILE_STOP = {"index", "types", "type", "basic", "setup", "utils", "util", "test",
             "main", "app", "shallow", "react", "vanilla", "middleware"}

TEST_FILE_RE = re.compile(r"(^|/)(tests?|__tests__)/|\.(test|spec)\.[jt]sx?$")

PORTER = snowballstemmer.stemmer("porter")


# ------------------------------------------------------------------------ tokens


def code_tokens(text: str) -> list[str]:
    """Same dual-emit tokenisation as code_graph.py: identifier intact + subwords."""
    return cg.code_tokens(text)


def porter(tokens: list[str]) -> list[str]:
    return PORTER.stemWords(tokens)


def query_tokens(text: str) -> list[str]:
    return porter(code_tokens(text))


# ------------------------------------------------------------------ PR selection


def fetch_pr_list(root: Path, cache: Path, repo: str = "pmndrs/zustand",
                  limit: int = PR_LIMIT) -> list[dict]:
    """Merged PRs with their file lists, cached so the run is reproducible offline."""
    if cache.exists():
        return json.loads(cache.read_text())["prs"]
    out = subprocess.run(
        ["gh", "pr", "list", "--repo", repo, "--state", "merged",
         "--limit", str(limit), "--json",
         "number,title,body,mergedAt,files,author"],
        capture_output=True, text=True, check=True,
    )
    prs = json.loads(out.stdout)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "repo": repo,
        "head": git_head(root),
        "limit": limit,
        "prs": prs,
    }, indent=1))
    return prs


def git(*args: str, root: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, check=False).stdout
    except FileNotFoundError:
        return ""


def git_head(root: Path) -> str:
    return git("rev-parse", "HEAD", root=root).strip() or "unknown"


def file_drift(root: Path, path: str, merged_at: str) -> float | None:
    """1 - line similarity between the file at the PR's base commit and at HEAD.

    0.0 means the file is byte-identical to what the PR saw; 1.0 means it was
    rewritten. This is the direct measurement of temporal leakage.
    """
    commit = git("rev-list", "-1", "--before", merged_at, "HEAD", "--", path, root=root).strip()
    if not commit:
        return 1.0
    old = git("show", f"{commit}:{path}", root=root)
    try:
        new = (root / path).read_text(encoding="utf-8")
    except OSError:
        return 1.0
    if old == new:
        return 0.0
    matcher = difflib.SequenceMatcher(None, old.splitlines(), new.splitlines())
    return 1.0 - matcher.ratio()


def is_code(path: str) -> bool:
    return path.endswith(CODE_EXT)


def is_test_path(path: str) -> bool:
    return bool(TEST_FILE_RE.search(path))


DEP_BUMP_RE = re.compile(
    r"^(chore\(deps\)|build\(deps\)|chore: (update|bump)|fix\(deps\)|chore: release)", re.I)


def select_prs(raw: list[dict], root: Path, known_files: set[str],
               min_merged: str = MIN_MERGED) -> tuple[list[dict], dict]:
    """1-5 changed code files, all present at HEAD, none in docs/CI/lockfiles."""
    drops: Counter[str] = Counter()
    kept: list[dict] = []
    for pr in raw:
        files = [f["path"] for f in pr.get("files") or []]
        code = [path for path in files if is_code(path)]
        if pr["mergedAt"] < min_merged:
            drops["before_window"] += 1
            continue
        if not code:
            drops["docs_or_ci_only"] += 1
            continue
        if not MIN_CHANGED_FILES <= len(code) <= MAX_CHANGED_FILES:
            drops["file_count_out_of_range"] += 1
            continue
        if any(not path.startswith(SCAN_DIRS) for path in code):
            drops["outside_corpus_dirs"] += 1
            continue
        if (pr.get("author") or {}).get("is_bot") or "dependabot" in (
                (pr.get("author") or {}).get("login") or ""):
            drops["bot"] += 1
            continue
        if DEP_BUMP_RE.match(pr["title"]):
            drops["dependency_bump"] += 1
            continue
        if any(path not in known_files for path in code):
            drops["file_gone_at_head"] += 1
            continue
        query = clean_query_text(pr)
        if len(query.split()) < MIN_QUERY_WORDS:
            drops["query_too_short"] += 1
            continue
        kept.append({
            "number": pr["number"],
            "title": pr["title"],
            "mergedAt": pr["mergedAt"],
            "author": (pr.get("author") or {}).get("login"),
            "query": query,
            "qrels_files": sorted(code),
            "drift": _mean([file_drift(root, path, pr["mergedAt"]) for path in code]),
        })
    kept.sort(key=lambda pr: (pr["mergedAt"], pr["number"]))
    return kept, dict(drops)


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


SECTION_DROP = re.compile(
    r"^#{1,6}\s*(checklist|things to check|related issues?|screenshots?|sponsors?|"
    r"funding|additional context|before/after|preview|demo)\b", re.I)
HEADING = re.compile(r"^#{1,6}\s")
CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\]")
FENCE = re.compile(r"^\s*```")


def clean_query_text(pr: dict) -> str:
    """PR title + body, stripped of checklists, templates, fenced dumps and URLs.

    The body is where the intent lives ("fix devtools type inference for the
    initializer"), so it is kept; the template scaffolding around it is not.
    """
    body = pr.get("body") or ""
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    lines: list[str] = []
    skip_section = False
    fence_lines = 0
    for line in body.split("\n"):
        if FENCE.match(line):
            fence_lines = 0 if fence_lines else 1
            continue
        if fence_lines:
            fence_lines += 1
            if fence_lines > 8:      # a pasted stack trace or diff, not prose
                continue
        if HEADING.match(line):
            skip_section = bool(SECTION_DROP.match(line))
            if skip_section:
                continue
        if skip_section or CHECKBOX.match(line):
            continue
        lines.append(line)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", "\n".join(lines))
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]{1,40}>", " ", text)
    text = re.sub(r"[*_`#>]+", " ", text)
    words = (pr["title"] + ". " + text).split()
    return " ".join(words[:200])


# ---------------------------------------------------------------------- parsing


DECL_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("class", re.compile(
        r"^(?:export\s+)?(?:declare\s+)?(?:default\s+)?(?:abstract\s+)?class\s+"
        r"([A-Za-z_$][\w$]*)")),
    ("function", re.compile(
        r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*"
        r"([A-Za-z_$][\w$]*)")),
    ("enum", re.compile(
        r"^(?:export\s+)?(?:declare\s+)?(?:const\s+)?enum\s+([A-Za-z_$][\w$]*)")),
    ("interface", re.compile(
        r"^(?:export\s+)?(?:declare\s+)?interface\s+([A-Za-z_$][\w$]*)")),
    ("type", re.compile(
        r"^(?:export\s+)?(?:declare\s+)?type\s+([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*=")),
)
ASSIGN_TS = re.compile(
    r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+?)?=\s*"
    r"(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::\s*[^=]+?)?=>|"
    r"[A-Za-z_$][\w$]*\s*=>|<[^>]*>\s*\([^)]*\)\s*(?::\s*[^=]+?)?=>)")
METHOD_TS = re.compile(
    r"^\s+(?:(?:public|private|protected|static|readonly|abstract|override|async|"
    r"get|set)\s+)*([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*"
    r"(?::\s*[^{;]+)?\{\s*$")
PROP_ARROW_TS = re.compile(
    r"^\s+(?:(?:public|private|protected|static|readonly|abstract|override)\s+)*"
    r"([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)"
    r"\s*(?::[^=]+)?=>")
BLOCK_MAX_LINES = 40
BLOCK_MIN_CHARS = 20


def continuation_line(lines: list[str], depths: list[int], i: int,
                      limit: int = 15) -> int | None:
    """First line whose brace depth exceeds the declaration's, if the head is open.

    TypeScript declarations routinely put parameters on their own lines:
    `export function useStore<T>(\n  api: StoreApi<T>\n): T`. Brace counting alone
    ends the chunk at the signature line and dumps the body into a block chunk.
    Only an unbalanced bracket count triggers the forward scan, so a bodyless
    overload cannot swallow the next declaration.
    """
    clean, _ = cg.strip_literals(lines[i], False)
    balance = (clean.count("(") - clean.count(")")
               + clean.count("[") - clean.count("]"))
    if balance <= 0:
        return None
    base = depths[i]
    for j in range(i + 1, min(len(depths), i + limit)):
        if not lines[j].strip():
            return None      # a signature never crosses a blank line
        if depths[j] > base:
            return j
        continuation, _ = cg.strip_literals(lines[j], False)
        if _match_decl(continuation):
            return None      # the next declaration started: this one had no body
    return None


def scan_file(rel: str, source: str) -> list[dict]:
    """Chunk a TypeScript file: declarations, class methods, then 40-line blocks.

    Unknown constructs are not dropped -- they land in a block chunk -- so every
    line of the file is covered by some retrieval unit.
    """
    lines = source.split("\n")
    depths = cg.depth_map(lines)
    constructs: list[dict] = []

    i = 0
    while i < len(lines):
        if depths[i] != 0:
            i += 1
            continue
        clean, _ = cg.strip_literals(lines[i], False)
        hit = _match_decl(clean)
        if hit is None:
            i += 1
            continue
        kind, name = hit
        body = continuation_line(lines, depths, i)
        end = cg.block_end(depths, body) if body is not None else cg.block_end(depths, i)
        constructs.append({"start": i, "end": end, "name": name, "kind": kind})
        i = end + 1

    methods: list[dict] = []
    for construct in (c for c in constructs if c["kind"] == "class"):
        for j in range(construct["start"] + 1, construct["end"] + 1):
            if depths[j] != 1:
                continue
            clean, _ = cg.strip_literals(lines[j], False)
            match = METHOD_TS.match(clean) or PROP_ARROW_TS.match(clean)
            if not match or match.group(1) in cg.JS_KEYWORDS:
                continue
            body = continuation_line(lines, depths, j)
            methods.append({
                "start": j, "end": cg.block_end(depths, body if body is not None else j),
                "name": match.group(1), "kind": "method", "parent": construct["name"],
            })
    return make_chunks(rel, lines, constructs, methods)


def _match_decl(clean: str) -> tuple[str, str] | None:
    for kind, pattern in DECL_PATTERNS:
        match = pattern.match(clean)
        if match:
            return kind, match.group(1)
    match = ASSIGN_TS.match(clean)
    if match:
        return "function", match.group(1)
    return None


def make_chunks(rel: str, lines: list[str], constructs: list[dict],
                methods: list[dict]) -> list[dict]:
    by_owner: dict[int, list[dict]] = defaultdict(list)
    for method in methods:
        owner = max(
            (c for c in constructs if c["kind"] == "class"
             and c["start"] < method["start"] <= c["end"]),
            key=lambda c: c["start"],
        )
        by_owner[owner["start"]].append(method)

    chunks: list[dict] = []
    covered: set[int] = set()
    for construct in constructs:
        covered.update(range(construct["start"], construct["end"] + 1))
        if construct["kind"] == "class":
            skip = {line for method in by_owner[construct["start"]]
                    for line in range(method["start"], method["end"] + 1)}
            body = [line for offset, line in enumerate(
                lines[construct["start"]:construct["end"] + 1])
                if construct["start"] + offset not in skip]
        else:
            body = lines[construct["start"]:construct["end"] + 1]
        chunks.append(_chunk(rel, construct["name"], construct["kind"], construct["start"],
                             construct["end"], body))
    for method in methods:
        chunks.append(_chunk(rel, method["name"], "method", method["start"], method["end"],
                             lines[method["start"]:method["end"] + 1]))
    chunks.extend(_blocks(rel, lines, covered))
    for chunk in chunks:
        chunk["id"] = f"{rel}:L{chunk['start']}-{chunk['end']}:{chunk['name']}"
        chunk["title"] = f"{rel} {chunk['name']}"
    return chunks


def _chunk(rel: str, name: str, kind: str, start: int, end: int,
           body: list[str]) -> dict:
    return {"name": name, "kind": kind, "file": rel, "start": start + 1, "end": end + 1,
            "text": "\n".join(body).strip()}


def _blocks(rel: str, lines: list[str], covered: set[int]) -> list[dict]:
    """Uncovered runs, split at blank lines and capped at BLOCK_MAX_LINES."""
    out: list[dict] = []
    run: list[int] = []
    for i in range(len(lines) + 1):
        free = i < len(lines) and i not in covered and bool(lines[i].strip())
        if free and len(run) < BLOCK_MAX_LINES:
            run.append(i)
            continue
        if run:
            text = "\n".join(lines[run[0]:run[-1] + 1]).strip()
            if len(text) > BLOCK_MIN_CHARS:
                out.append(_chunk(rel, f"block@L{run[0] + 1}", "block", run[0], run[-1],
                                  lines[run[0]:run[-1] + 1]))
            run = [i] if free else []
        elif free:
            run = [i]
    return out


# -------------------------------------------------------------------- graph


IMPORT_FROM = re.compile(r"""^\s*import\s+(?:type\s+)?(?:([\w$*{}\s,]+)\s+from\s+)?"""
                         r"""["']([^"']+)["']""")
EXPORT_FROM = re.compile(r"""^\s*export\s+(?:type\s+)?(?:\*|\{[^}]*\})\s+from\s+"""
                         r"""["']([^"']+)["']""")
DYNAMIC_IMPORT = re.compile(r"""\bimport\s*\(\s*["']([^"']+)["']""")
REQUIRE = re.compile(r"""\brequire\s*\(\s*["']([^"']+)["']""")
EXPORT_BLOCK = re.compile(r"^\s*export\s*(?:type\s*)?\{([^}]*)\}")
EXPORT_NAMED = re.compile(r"^\s*export\s+(?:default\s+)?(?:async\s+)?"
                          r"(?:declare\s+)?(?:function\*?|class|const|let|var|interface|"
                          r"type|enum)\s+([A-Za-z_$][\w$]*)")
CALL_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*\(")


def read_repo(root: Path) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for folder in SCAN_DIRS:
        base = root / folder
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in CODE_EXT:
                continue
            if "node_modules" in path.parts:
                continue
            files.append((path.relative_to(root).as_posix(),
                          path.read_text(encoding="utf-8", errors="replace")))
    return files


def build_graph(root: Path) -> cg.Graph:
    """Tolerant TS graph with the same edge vocabulary as code_graph.build_graph."""
    files = read_repo(root)
    chunks: list[dict] = []
    for rel, source in files:
        chunks.extend(scan_file(rel, source))

    graph = cg.Graph(chunks=chunks, files=[rel for rel, _ in files])
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
        lines = source.split("\n")
        for line_no, line in enumerate(lines, start=1):
            for module in _imported_modules(line):
                graph.file_imports.setdefault(rel, set()).add(module)
                graph.module_files.setdefault(module, set()).add(rel)
            from_match = IMPORT_FROM.match(line)
            if from_match and from_match.group(1):
                for name in re.findall(r"[A-Za-z_$][\w$]*", from_match.group(1)):
                    if name not in ("as", "type"):
                        imported_names[rel].add(name)
            block = EXPORT_BLOCK.match(line)
            if block:
                for part in block.group(1).split(","):
                    name = part.split(" as ")[0].strip()
                    if name and name != "type":
                        graph.exports.setdefault(rel, set()).add(name)
            named = EXPORT_NAMED.match(line)
            if named:
                graph.exports.setdefault(rel, set()).add(named.group(1))
                export_lines.append((rel, line_no, named.group(1)))
        graph.file_imports.setdefault(rel, set())
        graph.exports.setdefault(rel, set())

    cg.resolve_export_chunks(graph, export_lines)
    symbols = set(graph.defined)

    graph.callers = defaultdict(set)
    for chunk in chunks:
        called = {
            name for name in CALL_RE.findall(chunk["text"])
            if name not in cg.JS_KEYWORDS
            and (name in symbols or name in imported_names[chunk["file"]])
        }
        called.discard(chunk["name"])
        graph.calls[chunk["id"]] = called
        for name in called:
            graph.callers[name].add(chunk["id"])
    graph.callers = dict(graph.callers)
    graph.calls = dict(graph.calls)

    mentions: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        tokens = set(code_tokens(chunk["text"]))
        for symbol in symbols:
            if symbol.lower() in tokens:
                mentions[symbol].add(chunk["id"])
    graph.mentions = dict(mentions)

    resolve_imports(graph)
    return graph


def _imported_modules(line: str) -> list[str]:
    modules = []
    for pattern in (IMPORT_FROM, EXPORT_FROM, DYNAMIC_IMPORT, REQUIRE):
        for match in pattern.finditer(line):
            modules.append(match.group(match.lastindex or 1))
    return modules


def resolve_imports(graph: cg.Graph) -> None:
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
    candidate = posixpath.normpath(posixpath.join(posixpath.dirname(rel), module))
    for path in (candidate, *(candidate + ext for ext in CODE_EXT),
                 *(candidate + "/index" + ext for ext in CODE_EXT)):
        if path in files:
            return path
    return None


# -------------------------------------------------------------------- corpus


@dataclass
class Corpus:
    graph: cg.Graph
    ix: harness.Index
    position: dict[str, int]
    file_of: dict[str, str]
    dl: dict[str, int]
    files: list[str]
    chunks_of: dict[str, list[str]]
    symbols: dict[str, frozenset[str]] = field(default_factory=dict)
    export_chunks: frozenset[str] = frozenset()

    def file_rank(self, chunk_ids: list[str], depth: int = FILE_RANK_DEPTH) -> list[str]:
        """File ranking by first appearance in a chunk ranking."""
        out: list[str] = []
        for chunk_id in chunk_ids:
            file = self.file_of[chunk_id]
            if file not in out:
                out.append(file)
                if len(out) >= depth:
                    break
        return out

    def aggregate_files(self, scores: np.ndarray, depth: int = FILE_RANK_DEPTH,
                        mode: str = "max") -> list[str]:
        """File ranking by aggregating chunk scores per file.

        `max` is the primary aggregation. Note that it is exactly equivalent to
        ranking chunks and taking each file's first appearance, because the first
        chunk of a file to appear is its highest scoring one; `sum` is kept as the
        alternative that rewards many weakly matching chunks (usually test files).
        """
        aggregated: dict[str, float] = {}
        for doc_id, score in zip(self.ix.doc_ids, scores):
            if score <= 0:
                continue
            file = self.file_of[doc_id]
            if mode == "sum":
                aggregated[file] = aggregated.get(file, 0.0) + float(score)
            elif score > aggregated.get(file, 0.0):
                aggregated[file] = float(score)
        return [file for file, _ in
                sorted(aggregated.items(), key=lambda kv: (-kv[1], kv[0]))[:depth]]


def build_corpus(root: Path) -> Corpus:
    graph = build_graph(root)
    ix = cg.build_index(graph)
    position = {doc: i for i, doc in enumerate(ix.doc_ids)}
    file_of = {chunk["id"]: chunk["file"] for chunk in graph.chunks}
    dl = {doc: int(ix.dl[position[doc]]) for doc in ix.doc_ids}

    symbols: dict[str, set[str]] = {doc: set() for doc in ix.doc_ids}
    for doc, names in graph.chunk_defines.items():
        symbols[doc].update(names)
    for doc, names in graph.calls.items():
        symbols[doc].update(names)
    for symbol, holders in graph.mentions.items():
        for doc in holders:
            symbols[doc].add(symbol)

    export_chunks = frozenset(
        doc for doc in ix.doc_ids
        if graph.chunk_defines.get(doc, set()) & graph.exports.get(file_of[doc], set())
    )
    return Corpus(
        graph=graph, ix=ix, position=position, file_of=file_of, dl=dl,
        files=graph.files, chunks_of={f: sorted(c) for f, c in graph.contains.items()},
        symbols={doc: frozenset(names) for doc, names in symbols.items()},
        export_chunks=export_chunks,
    )


# ------------------------------------------------------------------ constraints


@dataclass
class Constraints:
    labels: list[str]
    covered_by: list[frozenset[str]]
    mandatory: list[str]
    symbol_hits: list[str]
    file_hits: list[str]
    cues: list[str]


SYMBOL_RE = re.compile(r"[A-Za-z_$][\w$]*")
PATH_RE = re.compile(r"[A-Za-z0-9_$/.-]+\.(?:[cm]?[jt]sx?)")
CALL_CUE = re.compile(r"\b(call|calls|called|calling|invoke|invokes|invoked)\b", re.I)
IMPORT_CUE = re.compile(r"\b(import|imports|imported|importing|require|requires)\b", re.I)
EXPORT_CUE = re.compile(r"\b(export|exports|exported|exporting|public api)\b", re.I)
DEFINE_CUE = re.compile(r"\b(define|defines|defined|definition|declares?|adds?|"
                        r"introduces?|implements?)\b", re.I)


def query_symbols(corpus: Corpus, text: str) -> list[str]:
    """Identifiers in the PR text that the graph defines, rarest first.

    Exact identifier match only: no case heuristics, no substrings. Hub symbols
    (defined or mentioned in more than 5% of chunks) are dropped, as in
    code_graph.entity_scores, because they carry no information.
    """
    words = set(SYMBOL_RE.findall(text))
    lowered = {word.lower() for word in words}
    cap = max(2.0, HUB_DF_FRAC * len(corpus.ix.doc_ids))
    hits = [
        symbol for symbol in corpus.graph.defined
        if (symbol in words or symbol.lower() in lowered)
        and 1 <= len(corpus.graph.mentions.get(symbol, ())) <= cap
    ]
    return sorted(hits, key=lambda s: (len(corpus.graph.mentions.get(s, ())), s))[
        :MAX_SYMBOL_CONSTRAINTS]


def query_files(corpus: Corpus, text: str) -> list[str]:
    """Corpus files named in the PR text, by path, basename, or module stem."""
    low = text.lower()
    tokens = {token.lower() for token in SYMBOL_RE.findall(text)}
    ranked: list[tuple[int, str]] = []
    for file in corpus.files:
        base = posixpath.basename(file)
        stem = base.split(".")[0].lower()
        if file.lower() in low:
            ranked.append((3, file))
        elif len(base) > 6 and base.lower() in low:
            ranked.append((2, file))
        elif stem not in FILE_STOP and len(stem) >= 4 and stem in tokens:
            ranked.append((1, file))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    out: list[str] = []
    for _, file in ranked:
        if file not in out:
            out.append(file)
    return out[:MAX_FILE_CONSTRAINTS]


def extract_constraints(corpus: Corpus, text: str) -> Constraints:
    """Automatic constraints from the PR text alone. Never reads qrels.

    Three sources, all cheap and all graph-backed:
      * symbols the graph defines and the text names  -> DEFINES(symbol)
      * relation words next to those symbols          -> CALLS(symbol),
                                                         IMPORTS_FROM(file),
                                                         EXPORTS(file)
      * files the text names                          -> COVERS(file)
    """
    graph = corpus.graph
    symbols = query_symbols(corpus, text)
    files = query_files(corpus, text)
    cues = [name for name, pattern in (("calls", CALL_CUE), ("imports", IMPORT_CUE),
                                       ("exports", EXPORT_CUE), ("defines", DEFINE_CUE))
            if pattern.search(text)]

    labels: list[str] = []
    sets: list[frozenset[str]] = []

    def add(label: str, chunks) -> None:
        chunks = frozenset(chunks)
        if chunks and label not in labels:
            labels.append(label)
            sets.append(chunks)

    for symbol in symbols:
        add(f"DEFINES({symbol})", graph.defined.get(symbol, ()))
        if "calls" in cues:
            add(f"CALLS({symbol})", graph.callers.get(symbol, ()))
        for file in {corpus.file_of[c] for c in graph.defined.get(symbol, ())}:
            if "imports" in cues:
                importers = graph.module_importers.get(file, ())
                add(f"IMPORTS_FROM({file})",
                    [chunk for importer in importers
                     for chunk in corpus.chunks_of.get(importer, ())])
            if "exports" in cues:
                exports = graph.exports.get(file, set())
                add(f"EXPORTS({file})",
                    [chunk for chunk in corpus.chunks_of.get(file, ())
                     if graph.chunk_defines.get(chunk, set()) & exports])
    for file in files:
        add(f"COVERS({file})", corpus.chunks_of.get(file, ()))

    mandatory: list[str] = []
    for chunks in sets:
        if len(chunks) <= SMALL_SET:
            mandatory.extend(sorted(chunks))
    return Constraints(labels=labels, covered_by=sets,
                       mandatory=sorted(set(mandatory)), symbol_hits=symbols,
                       file_hits=files, cues=cues)


# --------------------------------------------------------------------- systems


def bm25_chunks(corpus: Corpus, text: str) -> np.ndarray:
    return harness.bm25_fast(corpus.ix, query_tokens(text))


def entity_chunks(corpus: Corpus, text: str) -> np.ndarray:
    return cg.entity_scores(corpus.graph, corpus.position, text)


def graph_channel(corpus: Corpus, bm25: np.ndarray) -> np.ndarray:
    seeds = [corpus.ix.doc_ids[i] for i in fusion.order(bm25, SEED_CHUNKS)]
    return cg.graph_scores(corpus.graph, corpus.position, seeds)


def candidate_pool(corpus: Corpus, channels: dict[str, np.ndarray]) -> tuple[list[str], dict[str, list[str]]]:
    pool: list[str] = []
    provenance: dict[str, list[str]] = {}
    for label, scores in channels.items():
        for doc in fusion.ranked_ids(corpus.ix, scores, POOL_DEPTH):
            if doc in provenance:
                provenance[doc].append(label)
            else:
                provenance[doc] = [label]
                pool.append(doc)
    return pool, provenance


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / (len(left) + len(right) - shared) if shared else 0.0


def cesr_select(corpus: Corpus, text: str, constraints: Constraints, pool: list[str],
                bm25: np.ndarray, budget: int = CESR_BUDGET) -> list[str]:
    """Deterministic greedy coverage with a BM25 tie-break, then BM25 backfill.

    The selection order is the ranking. Chunks that satisfy a *small* coverage set
    (`SMALL_SET`) are taken first: when a relation pins at most three chunks, the
    constraint is effectively an identity and an existential reading would silently
    drop the rest. That is the one universal quantifier this constraint language has.
    """
    if not constraints.labels:
        # No constraint to execute: fall back to lexical rank order. 40 chunks is
        # enough to name 10 distinct files, which is all the file ranking needs.
        return fusion.ranked_ids(corpus.ix, bm25, 40)

    position = corpus.position
    order_of = {doc: rank for rank, doc in enumerate(
        fusion.ranked_ids(corpus.ix, bm25, 400))}
    prior = np.zeros(len(corpus.ix.doc_ids))
    top = max(len(order_of), 1)
    for doc, rank in order_of.items():
        prior[position[doc]] = 1.0 - rank / top

    # Universal part: a coverage set of at most SMALL_SET chunks is an identity,
    # so take all of it. Lexical order decides which fits the budget first.
    mandatory = sorted(set(constraints.mandatory),
                       key=lambda doc: (order_of.get(doc, 10 ** 6), doc))
    taken: list[str] = []
    used = 0
    for doc in mandatory:
        if used + corpus.dl[doc] <= budget:
            taken.append(doc)
            used += corpus.dl[doc]
    taken_set = set(taken)
    covered = [bool(set(chunks) & taken_set) for chunks in constraints.covered_by]

    selected: list[str] = []
    while True:
        best, best_score = None, 0.0
        for doc in pool:
            if doc in taken_set or used + corpus.dl[doc] > budget:
                continue
            gain = sum(1.0 / len(constraints.labels) for column, chunks in
                       enumerate(constraints.covered_by)
                       if not covered[column] and doc in chunks)
            if gain <= 0:
                continue
            overlap = max((_jaccard(corpus.symbols[doc], corpus.symbols[other])
                           for other in taken), default=0.0)
            cost = max(corpus.dl[doc] + 10, 1) ** COST_EXPONENT
            score = gain / cost + BM25_TIE * prior[position[doc]] - MU * overlap
            if score > best_score:
                best, best_score = doc, score
        if best is None:
            break
        taken.append(best)
        taken_set.add(best)
        selected.append(best)
        used += corpus.dl[best]
        for column, chunks in enumerate(constraints.covered_by):
            if not covered[column] and best in chunks:
                covered[column] = True

    forced = len(taken) - len(selected)
    return taken[:forced] + selected


def backfill(corpus: Corpus, ordered: list[str], fallback: list[str],
             depth: int = FILE_RANK_DEPTH) -> list[str]:
    """Finish a file ranking with a lexical fallback once coverage is exhausted."""
    files = corpus.file_rank(ordered, depth=10 ** 6)
    if len(files) >= depth:
        return files[:depth]
    for file in corpus.file_rank(fallback, depth=10 ** 6):
        if file not in files:
            files.append(file)
            if len(files) >= depth:
                break
    return files[:depth]


def run_query(corpus: Corpus, text: str) -> dict:
    """All systems for one PR description. Returns chunk and file rankings."""
    bm25 = bm25_chunks(corpus, text)
    entity = entity_chunks(corpus, text)
    graph = graph_channel(corpus, bm25)
    rrf = fusion.rrf(corpus.ix, [bm25, entity, graph])
    bm25_order = fusion.ranked_ids(corpus.ix, bm25, 1000)
    rrf_order = fusion.ranked_ids(corpus.ix, rrf, 1000)

    constraints = extract_constraints(corpus, text)
    pool_bm25 = fusion.ranked_ids(corpus.ix, bm25, POOL_DEPTH)
    pool_union, provenance = candidate_pool(
        corpus, {"porter": bm25, "entity": entity, "graph": graph})

    cesr_sel = cesr_select(corpus, text, constraints, pool_bm25, bm25)
    cesr_graph_sel = cesr_select(corpus, text, constraints, pool_union, bm25)

    systems = {
        "bm25_chunk": {
            "chunks": bm25_order, "files": corpus.file_rank(bm25_order)},
        "bm25_file": {
            "chunks": [], "files": corpus.aggregate_files(bm25)},
        "bm25_file_sum": {
            "chunks": [], "files": corpus.aggregate_files(bm25, mode="sum")},
        "entity_file": {
            "chunks": [], "files": corpus.aggregate_files(entity)},
        "graph_file": {
            "chunks": [], "files": corpus.aggregate_files(graph)},
        "graph_rrf": {
            "chunks": rrf_order, "files": corpus.file_rank(rrf_order)},
        "cesr": {
            "chunks": cesr_sel, "files": backfill(corpus, cesr_sel, bm25_order)},
        "cesr_plus_graph": {
            "chunks": cesr_graph_sel,
            "files": backfill(corpus, cesr_graph_sel, rrf_order)},
    }
    return {
        "systems": systems,
        "channels": {"porter": bm25, "entity": entity, "graph": graph},
        "pool": pool_union,
        "pool_provenance": provenance,
        "constraints": constraints,
    }


SYSTEMS = ("bm25_chunk", "bm25_file", "bm25_file_sum", "entity_file", "graph_file",
           "graph_rrf", "cesr", "cesr_plus_graph")


# --------------------------------------------------------------------- metrics


def file_metrics(files: list[str], relevant: list[str], k: int = 10) -> dict:
    rel = set(relevant)
    top = files[:k]
    out = {}
    for cutoff in (1, 5, 10):
        out[f"recall@{cutoff}"] = len(rel & set(files[:cutoff])) / len(rel)
    out["all@10"] = 1.0 if rel <= set(top) else 0.0
    dcg = sum(1 / math.log2(i + 2) for i, file in enumerate(top) if file in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    out["ndcg@10"] = dcg / ideal if ideal else 0.0
    first = next((i + 1 for i, file in enumerate(top) if file in rel), None)
    out["mrr@10"] = 1 / first if first else 0.0
    return out


def chunk_recall(chunks: list[str], corpus: Corpus, relevant: list[str],
                 cutoffs=(10, 100)) -> dict:
    out = {}
    for cutoff in cutoffs:
        reach = {corpus.file_of[chunk] for chunk in chunks[:cutoff]}
        out[f"chunk_recall@{cutoff}"] = len(reach & set(relevant)) / len(relevant)
    return out


METRICS = ("recall@1", "recall@5", "recall@10", "all@10", "ndcg@10", "mrr@10")


def summarize(rows: list[dict]) -> dict:
    out = {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
    out["queries"] = len(rows)
    return out


def slice_recall(rows: list[dict], group: str) -> dict:
    """Recall@10 restricted to the qrel files in one group (test or source)."""
    subset = [row for row in rows if group in row["groups"].values()]
    if not subset:
        return {"queries": 0, "recall@10": float("nan")}
    values = []
    for row in subset:
        files = [file for file in row["relevant"] if row["groups"][file] == group]
        top = set(row["top10"])
        values.append(len([file for file in files if file in top]) / len(files))
    return {"queries": len(subset), "recall@10": float(np.mean(values))}


def paired(base: list[dict], candidate: list[dict], metric: str,
           resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    return cg.paired(base, candidate, metric, resamples=resamples, seed=seed)


def gate(blocks: dict, base: str = "bm25_file",
         challengers=("graph_rrf", "cesr", "cesr_plus_graph")) -> dict:
    """Pre-registered: the direction survives only with a CI above zero."""
    checks = {}
    for system in challengers:
        for metric in ("recall@10", "ndcg@10"):
            entry = blocks[system][metric]
            low = entry["ci95"][0]
            checks[f"{system}:{metric}"] = {
                **entry,
                "ci_above_zero": bool(entry["delta"] > 0 and low > 0),
            }
    passing = [key for key, value in checks.items() if value["ci_above_zero"]]
    return {
        "rule": (f"{' or '.join(challengers)} beats {base} on file recall@10 or "
                 "file nDCG@10 with a paired-bootstrap 95% CI lower bound above zero"),
        "checks": checks,
        "passing_checks": passing,
        "passed": bool(passing),
    }


# ------------------------------------------------------------------ PR -> query


def build_queries(prs: list[dict], corpus: Corpus) -> list[dict]:
    """Attach qrels, slices and graph reachability to each PR description."""
    queries = []
    for pr in prs:
        relevant = [file for file in pr["qrels_files"] if file in corpus.chunks_of]
        if not relevant:
            continue
        queries.append({
            "qid": str(pr["number"]),
            "number": pr["number"],
            "text": pr["query"],
            "title": pr["title"],
            "merged": pr["mergedAt"][:10],
            "drift": pr["drift"],
            "relevant": relevant,
            "groups": {file: "test" if is_test_path(file) else "source"
                       for file in relevant},
        })
    return queries


def candidate_ceiling(result: dict, relevant: list[str], corpus: Corpus) -> dict:
    pool = set(result["pool"])
    reached = {corpus.file_of[chunk] for chunk in pool}
    return {
        "files_in_pool": len(reached & set(relevant)) / len(relevant),
        "all_in_pool": 1.0 if set(relevant) <= reached else 0.0,
    }


def time_run(fn, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, (time.perf_counter() - start) * 1000
