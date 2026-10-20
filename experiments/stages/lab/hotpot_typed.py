"""Relation-typed bridge extraction for Entity-Bridge Retrieval on HotpotQA.

The untyped EBR bridge is any capitalized span in a seed paragraph. That leaks
generic noun phrases, sentence-initial words and relation-agnostic noise. This
module adds a fixed relation lexicon and a rule-based extractor so a bridge must
stand in a *known relation* that the question is also asking about.

No ML, no LLM, no embeddings: a cue-phrase lexicon, regex over the raw paragraph
text, and the untyped candidate set from `hotpot.py` (imported read-only). The
score formula, the 5% df cap, the 20 seeds and the 20-bridge limit are unchanged.

Two validity rules, applied to the untyped candidate set:

  (a) argument  - the entity is the subject or object of a detected-relation
                  instance in a seed paragraph;
  (b) link      - a detected-relation instance in the seed paragraph connects a
                  query entity to the entity.

`union` (a or b) is the pre-registered typed system. `argument` and `link` are
reported as diagnostics. If the question matches no relation at all the typed
bridge falls back to the untyped bridge, so the two systems differ only on the
`relation_detected` slice.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

import harness
import hotpot

# Same frozen constants as hotpot.py. Do not tune these on test.
TOP_SEEDS = 20
TOP_BRIDGES = 20
MAX_DF_FRAC = 0.05

# Entity span for relation arguments. Differs from hotpot.CAP_RE in one way that
# matters: "and" is not an internal connector, so "Tim Burton and Johnny Depp"
# becomes two arguments instead of one glued span.
ENT = (
    r"[A-Z][\w&'’.\-]*(?:\s+(?:(?:of|the|de|van|von|del|la|le|da|di|du)\s+)?"
    r"[A-Z][\w&'’.\-]*){0,3}"
)
ENT_RE = re.compile(ENT)

# ---------------------------------------------------------------------------
# Relation lexicon: cue phrase -> relation types it evidences.
#
# A cue can emit more than one type. "born in" is both the birth relation and a
# city filler; "located in" is both a location relation and a city filler. The
# question detector picks the type the question asks about, and only that type's
# arguments are admitted as bridges.
# ---------------------------------------------------------------------------
CUE_TYPES: dict[str, tuple[str, ...]] = {
    # people / works
    "directed by": ("directed_by",),
    "helmed by": ("directed_by",),
    "written by": ("written_by",),
    "screenplay by": ("written_by",),
    "script by": ("written_by",),
    "teleplay by": ("written_by",),
    "penned by": ("written_by",),
    "starring": ("starring",),
    "stars": ("starring",),
    "starred": ("starring",),
    "portrayed by": ("starring",),
    "played by": ("starring",),
    "featuring": ("starring",),
    "composed by": ("composed_by",),
    "music by": ("composed_by",),
    "score by": ("composed_by",),
    "songs by": ("composed_by",),
    "published by": ("publisher",),
    "released by": ("publisher",),
    # authorship (works)
    "novel by": ("author",),
    "book by": ("author",),
    "memoir by": ("author",),
    "story by": ("author",),
    "poem by": ("author",),
    "play by": ("author",),
    "series by": ("author",),
    "comic by": ("author",),
    "manga by": ("author",),
    # places
    "born in": ("born_in", "city"),
    "born at": ("born_in",),
    "born near": ("born_in",),
    "native of": ("born_in",),
    "located in": ("located_in", "city"),
    "located at": ("located_in",),
    "situated in": ("located_in", "city"),
    "based in": ("located_in", "city"),
    "headquartered in": ("located_in", "city"),
    "lies in": ("located_in",),
    "is found in": ("located_in",),
    "raised in": ("city",),
    "grew up in": ("city",),
    "died in": ("city",),
    "city in": ("country",),
    "town in": ("country",),
    "village in": ("country",),
    "hamlet in": ("country",),
    "county in": ("country",),
    "state in": ("country",),
    "province in": ("country",),
    "island in": ("country",),
    "territory in": ("country",),
    "region in": ("country",),
    "district in": ("country",),
    "capital of": ("capital_of",),
    "capital city of": ("capital_of",),
    # organisations / groups
    "member of": ("member_of",),
    "joined": ("member_of",),
    "part of": ("part_of",),
    "subsidiary of": ("part_of",),
    "division of": ("part_of",),
    "branch of": ("part_of",),
    "component of": ("part_of",),
    "founded by": ("founded_by",),
    "established by": ("founded_by",),
    "co-founded by": ("founded_by",),
    "formed by": ("founded_by",),
    "created by": ("founded_by",),
    "owned by": ("owned_by",),
    "acquired by": ("owned_by",),
    "produced by": ("produced_by",),
    # education / recognition / role
    "educated at": ("educated_at",),
    "educated in": ("educated_at",),
    "attended": ("educated_at",),
    "graduated from": ("educated_at",),
    "alumnus of": ("educated_at",),
    "won the": ("award",),
    "received the": ("award",),
    "awarded the": ("award",),
    "winner of the": ("award",),
    "worked as": ("occupation",),
    "served as": ("occupation",),
    "employed as": ("occupation",),
}

# Nationality is an adjective class, not a cue phrase. A fixed list keeps this
# rule-based; it also shows up in the report as the sparse type it is.
NATIONALITY_ADJECTIVES = (
    "American", "British", "English", "Scottish", "Welsh", "Irish", "French",
    "German", "Italian", "Spanish", "Portuguese", "Dutch", "Belgian", "Swiss",
    "Austrian", "Swedish", "Norwegian", "Danish", "Finnish", "Icelandic",
    "Polish", "Czech", "Hungarian", "Romanian", "Bulgarian", "Greek", "Turkish",
    "Russian", "Ukrainian", "Serbian", "Croatian", "Slovenian", "Slovak",
    "Estonian", "Latvian", "Lithuanian", "Albanian", "Canadian", "Australian",
    "Zealander", "Indian", "Pakistani", "Bangladeshi", "Chinese", "Japanese",
    "Korean", "Vietnamese", "Thai", "Filipino", "Indonesian", "Malaysian",
    "Singaporean", "Mexican", "Brazilian", "Argentine", "Chilean", "Peruvian",
    "Colombian", "Venezuelan", "Cuban", "Jamaican", "Nigerian", "Kenyan",
    "Ghanaian", "Ethiopian", "Egyptian", "Moroccan", "Algerian", "Tunisian",
    "Israeli", "Iranian", "Iraqi", "Syrian", "Lebanese", "Saudi", "Emirati",
    "Nepalese", "Sri Lankan", "Burmese", "Cambodian", "Mongolian", "Armenian",
    "Georgian", "Azerbaijani", "Kazakh", "Uzbek", "Congolese", "Zimbabwean",
)
NAT_RE = re.compile(
    r"(?i:\b(?:is|was|were)\s+an?\s+)(?P<n>(?:" + "|".join(NATIONALITY_ADJECTIVES) + r"))\b"
)

CUE_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(
        sorted((re.escape(cue) for cue in CUE_TYPES), key=len, reverse=True)
    ) + r")(?!\w)",
    re.IGNORECASE,
)

# The object of a relation stops at a clause boundary or at the next participial
# phrase, so "directed by Tim Burton and starring Johnny Depp" does not make
# Johnny Depp a director.
_OBJECT_BREAK_RE = re.compile(
    r"[,;:()\[\]]|\.\s|\b(?:starring|written|produced|directed|distributed|"
    r"released|based|featuring|edited|composed|published|located|situated)\b",
    re.IGNORECASE,
)
OBJECT_WINDOW = 60
SUBJECT_WINDOW = 140


def _normalize(value: str) -> str:
    return hotpot.normalize_entity(value)


def _padded(value: str) -> str:
    """Space-padded entity for token-boundary-aware containment tests."""
    return f" {value} "


def _plausible(value: str) -> bool:
    if not value or value in hotpot.STOP_ENTITIES:
        return False
    if len(value) < 2 or len(value.split()) > 5:
        return False
    return not value.isdigit()


def object_spans(text: str, start: int) -> list[str]:
    window = text[start:start + OBJECT_WINDOW]
    boundary = _OBJECT_BREAK_RE.search(window)
    if boundary:
        window = window[:boundary.start()]
    spans: list[str] = []
    for raw in ENT_RE.findall(window):
        value = _normalize(raw)
        if _plausible(value) and value not in spans:
            spans.append(value)
        if len(spans) >= 3:
            break
    return spans


def subject_before(text: str, start: int, title: str = "") -> str:
    """Best guess at the grammatical subject of the clause holding the cue.

    Passive relation phrases ("was directed by") put the topic first, so prefer a
    span that also occurs in the paragraph title; otherwise take the first
    plausible span in the clause. Topic binding is only used by the `link` rule.
    """
    window = text[max(0, start - SUBJECT_WINDOW):start]
    cut = max(window.rfind(". "), window.rfind("; "))
    if cut >= 0:
        window = window[cut + 2:]
    spans: list[str] = []
    for raw in ENT_RE.findall(window):
        value = _normalize(raw)
        if _plausible(value) and value not in spans:
            spans.append(value)
    if not spans:
        return ""
    if title:
        title_norm = _normalize(title)
        for value in spans:
            if _contains(value, title_norm):
                return value
    return spans[0]


def extract_relations(text: str, title: str = "") -> dict[str, dict]:
    """Per-paragraph relation instances: type -> {args, pairs} (normalized)."""
    found: dict[str, dict] = {}
    for match in CUE_RE.finditer(text):
        types = CUE_TYPES.get(match.group(0).lower())
        if not types:
            continue
        objects = object_spans(text, match.end())
        if not objects:
            continue
        subject = subject_before(text, match.start(), title)
        for relation in types:
            spec = found.setdefault(relation, {"args": set(), "pairs": []})
            for obj in objects:
                spec["args"].add(obj)
                if subject:
                    spec["args"].add(subject)
                    spec["pairs"].append((subject, obj))
    for match in NAT_RE.finditer(text):
        value = _normalize(match.group("n"))
        if not _plausible(value):
            continue
        spec = found.setdefault("nationality", {"args": set(), "pairs": []})
        spec["args"].add(value)
        subject = subject_before(text, match.start(), title)
        if subject:
            spec["pairs"].append((subject, value))

    return {
        relation: {
            "args": tuple(sorted(_padded(arg) for arg in spec["args"])),
            "pairs": tuple(spec["pairs"]),
        }
        for relation, spec in found.items()
    }


# ---------------------------------------------------------------------------
# Question-side relation detection.
#
# Fixed cue patterns in specificity order. The first two matching types are kept:
# a question like "who directed the film starring X" legitimately asks about both
# and its second hop may need either argument.
# ---------------------------------------------------------------------------
QUESTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("capital_of", r"\bcapital\s+(?:city\s+)?of\b"),
    ("directed_by", r"\bwho\s+directed\b|\bdirected\s+by\b|\bdirector\s+of\b|\bwhich\s+director\b|\bwhat\s+director\b"),
    ("composed_by", r"\bcomposed?\b|\bcomposer\b"),
    ("written_by", r"\bwho\s+wrote\b|\bwritten\s+by\b|\bwriter\s+of\b|\bscreenplay\b"),
    ("author", r"\bauthor\s+of\b|\bwho\s+(?:is|was)\s+the\s+author\b|\bby\s+what\s+author\b|\bauthored\b"),
    ("publisher", r"\bpublish(?:er|ed|ing)\b|\breleased\s+by\b"),
    ("educated_at", r"\beducated\b|\battend(?:ed|s)?\b|\bgraduated\b|\balumn"),
    ("founded_by", r"\bfounded\b|\bfounder\b|\bestablish(?:ed)?\b|\bformed\b"),
    ("produced_by", r"\bproduced\b|\bproducer\b"),
    ("owned_by", r"\bowned\b|\bowner\b"),
    ("member_of", r"\bmember\s+of\b|\bmembers\s+of\b"),
    ("part_of", r"\bpart\s+of\b|\bsubsidiary\b"),
    ("starring", r"\bstarring\b|\bstars?\b|\bstarred\b|\bportray(?:ed|s|ing)?\b|\bactor\s+who\b|\bactress\s+who\b"),
    ("born_in", r"\bborn\b|\bbirthplace\b"),
    ("located_in", r"\blocated\b|\bsituated\b|\bbased\s+in\b|\bheadquarter"),
    ("country", r"\b(?:which|what|same|another|a|an|the)\s+(?:\w+\s+){0,2}countr(?:y|ies)\b"),
    ("city", r"\b(?:which|what|the|a|an)\s+(?:\w+\s+){0,2}\bcity\b"),
    ("award", r"\baward(?:s)?\b|\bwho\s+won\b|\bwhich\s+award\b"),
    ("occupation", r"\boccupation\b"),
    ("nationality", r"\bnationality\b"),
)
QUESTION_RES = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in QUESTION_PATTERNS)
MAX_QUESTION_TYPES = 2
RULES = ("union", "argument", "link", "anchor")


def detect_question_relations(question: str) -> tuple[str, ...]:
    found: list[str] = []
    for name, pattern in QUESTION_RES:
        if pattern.search(question):
            found.append(name)
            if len(found) >= MAX_QUESTION_TYPES:
                break
    return tuple(found)


# ---------------------------------------------------------------------------
# Typed bridge channel.
# ---------------------------------------------------------------------------
def _contains(needle: str, haystack: str) -> bool:
    """Token-boundary containment on normalized strings, either direction."""
    if not needle or not haystack:
        return False
    a, b = _padded(needle), _padded(haystack)
    return a in b or b in a


def _any_contains(entity_padded: str, arguments) -> bool:
    for argument in arguments:
        if entity_padded in argument or argument in entity_padded:
            return True
    return False


def _linked_arguments(specs, q_entities: set[str]) -> set[str]:
    """Padded endpoints that a relation instance ties to a query entity."""
    linked: set[str] = set()
    for spec in specs:
        for subject, obj in spec["pairs"]:
            for query_entity in q_entities:
                if _contains(query_entity, subject):
                    linked.add(_padded(obj))
                elif _contains(query_entity, obj):
                    linked.add(_padded(subject))
    return linked


def typed_bridge_scores(
    data,
    qid: str,
    porter_scores,
    top_seeds: int = TOP_SEEDS,
    top_bridges: int = TOP_BRIDGES,
    max_df_frac: float = MAX_DF_FRAC,
    untyped: tuple | None = None,
    seeds: list[str] | None = None,
):
    """Bridges restricted to detected-relation arguments.

    Returns (scores_by_rule, top_by_rule, stats). Rules:

      union    - question relation, argument OR query-linked (pre-registered)
      argument - question relation, argument only
      link     - question relation, query-linked only
      anchor   - ANY relation, query-linked only (diagnostic: the bridge hop may
                 not be the relation the question asks about)

    When the question matches no relation every rule falls back to the untyped
    bridge, so systems differ only on the `relation_detected` slice.
    """
    ix = data["ix"]
    question = data["raw_queries"][qid]
    types = detect_question_relations(question)
    q_entities = hotpot.query_entities(question, data["title_index"])

    if untyped is None:
        untyped = hotpot.bridge_scores(
            data, qid, porter_scores,
            top_seeds=top_seeds, top_bridges=top_bridges, max_df_frac=max_df_frac,
        )
    untyped_scores, untyped_top = untyped
    stats = {
        "types": types,
        "untyped_bridges": len(untyped_top),
        "detected": bool(types),
    }
    if not types:
        stats.update({
            "typed_bridges": len(untyped_top), "argument_bridges": len(untyped_top),
            "link_bridges": len(untyped_top), "anchor_bridges": len(untyped_top),
            "fallback": True,
        })
        return (
            {rule: untyped_scores for rule in RULES},
            {rule: untyped_top for rule in RULES},
            stats,
        )
    stats["fallback"] = False

    if seeds is None:
        seeds = harness.rank(ix, porter_scores, top_seeds)
    relations = data["relations"]
    weights = {rule: Counter() for rule in RULES}
    for rank, doc in enumerate(seeds):
        spec_by_type = relations.get(doc)
        if not spec_by_type:
            continue
        detected_specs = [spec_by_type[t] for t in types if t in spec_by_type]
        if not detected_specs:
            continue
        arguments = set()
        for spec in detected_specs:
            arguments.update(spec["args"])
        link_arguments = _linked_arguments(detected_specs, q_entities)
        anchor_arguments = _linked_arguments(spec_by_type.values(), q_entities)

        for entity in sorted(data["paragraph_entities"].get(doc, ())):
            if entity in q_entities:
                continue
            df = data["entity_df"].get(entity, 0)
            if df < 1 or df > max_df_frac * ix.N:
                continue
            entity_padded = _padded(entity)
            argument = _any_contains(entity_padded, arguments)
            link = _any_contains(entity_padded, link_arguments)
            anchor = _any_contains(entity_padded, anchor_arguments)
            if not (argument or link or anchor):
                continue
            weight = hotpot.entity_idf(ix.N, df) / (rank + 1)
            if argument:
                weights["argument"][entity] += weight
            if link:
                weights["link"][entity] += weight
            if anchor:
                weights["anchor"][entity] += weight
            if argument or link:
                weights["union"][entity] += weight

    scores, tops = {}, {}
    for rule, counter in weights.items():
        top = counter.most_common(top_bridges)
        paragraph_scores = Counter()
        for entity, weight in top:
            for doc in data["entity_paragraphs"].get(entity, ()):
                paragraph_scores[doc] += weight
        scores[rule], tops[rule] = paragraph_scores, top
    stats.update({
        "typed_bridges": len(tops["union"]),
        "argument_bridges": len(tops["argument"]),
        "link_bridges": len(tops["link"]),
        "anchor_bridges": len(tops["anchor"]),
    })
    return scores, tops, stats


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------
CACHE_VERSION = 1


def cache_key() -> str:
    """Fingerprint the lexicon so an edited cue list cannot reuse a stale cache."""
    material = repr((
        sorted(CUE_TYPES.items()), QUESTION_PATTERNS, NATIONALITY_ADJECTIVES,
        OBJECT_WINDOW, SUBJECT_WINDOW, CACHE_VERSION,
    ))
    return hashlib.md5(material.encode()).hexdigest()


def load(path: str, cache: str | None = None):
    """hotpot.load plus raw paragraph text and the typed relation index.

    Depends on hotpot.load for the corpus, BM25 index, entity postings and qrels,
    so the typed systems are compared against exactly the untyped ones.
    """
    import pickle
    from pathlib import Path

    key = cache_key()
    if cache:
        cached = Path(cache)
        if cached.exists():
            try:
                with cached.open("rb") as handle:
                    data = pickle.load(handle)
                if data.get("cache_key") == key:
                    return data
            except Exception:
                pass

    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    texts = {}
    for row in rows:
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            if title not in texts:
                texts[title] = f"{title}. " + " ".join(sentences)

    data = hotpot.load(path)
    relations, per_type_paragraphs, per_type_instances = {}, Counter(), Counter()
    for doc_id in data["ix"].doc_ids:
        extracted = extract_relations(texts[doc_id], doc_id)
        if not extracted:
            continue
        relations[doc_id] = extracted
        for relation, spec in extracted.items():
            per_type_paragraphs[relation] += 1
            per_type_instances[relation] += len(spec["pairs"])

    data["relations"] = relations
    data["relation_stats"] = {
        "paragraphs_with_relations": len(relations),
        "per_type_paragraphs": dict(per_type_paragraphs),
        "per_type_pairs": dict(per_type_instances),
    }
    data["cache_key"] = key
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with Path(cache).open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data
