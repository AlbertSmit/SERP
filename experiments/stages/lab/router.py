"""Text-only router: multi-hop question, or single-hop lookup?

Entity-Bridge Retrieval is a multi-hop method. Its bridge channel spends the
retrieval budget on entities that the question does *not* name, which is what a
two-fact question needs and what a topical single-hop query does not. The router
therefore decides from question text alone whether EBR may run.

No training, no model, no embeddings, no corpus statistics: four fixed rules over
capitalisation, closed-class words and relation vocabulary. The rules were frozen
after looking at HotpotQA dev questions and SciFact train claims only. They were
never adjusted against an evaluation split, and never against Touche.

Rules
-----
R1 comparison   a comparison cue ("both", "same", "which of", "versus", ...)
                plus at least one detected entity.
R2 anaphora     a relative/anaphoric pronoun ("that", "who", "which", "whose",
                "where", "when") anywhere but the first token, plus an entity.
                The question refers to something it has not named yet, which is
                what a bridge question does.
R3 wh-chain     two or more distinct wh-words ("... what ... who ...").
R4 pair+relation  two or more entities plus a relation word ("born", "located",
                "director of", ...). Two named things plus a relation is the
                comparison and multi-fact shape.

Case is treated as evidence, with one correction: a headline-style query
(NFCorpus is title-cased) carries no case information, so it is lowercased before
entity detection. Otherwise every content word would look like an entity.
"""
from __future__ import annotations

import re

import hotpot

# Capitalised words that are grammar, not names. hotpot.STOP_ENTITIES covers most
# of them; these are the auxiliaries that start a yes/no question.
EXTRA_STOP = {"should", "can", "could", "would", "will", "may", "might", "must"}

COMPARISON = re.compile(r"\b(?:both|same|which of|versus|vs|compare|compares|difference between)\b")
ANAPHOR = {"that", "who", "whom", "whose", "which", "where", "when"}
WH_WORDS = {"what", "who", "whom", "whose", "which", "when", "where", "why", "how"}
RELATION = re.compile(
    r"\b(?:born|located|based|director|founded|founder|author|written|member of|part of|"
    r"capital|country|city|nationality|spouse|starring|produced by|known for|named after|"
    r"succeeded by|affiliated|published by|released by|county|state|album|film|team|"
    r"company|university)\b"
)
# Closed-class words that headline capitalisation leaves lowercase.
CLOSED_CLASS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with", "at",
    "by", "from", "as", "vs", "is", "are", "be",
}

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
CONJUNCTION = re.compile(r"\s+(?:and|or|&)\s+")


def is_headline(text: str) -> bool:
    """Title case: every open-class word is capitalised. Then case says nothing."""
    words = WORD.findall(text)
    if len(words) < 3:
        return False
    return all(word[0].isupper() for word in words if word.lower() not in CLOSED_CLASS)


def entities(text: str) -> set[str]:
    """Capitalised spans, split on conjunction so "X and Y" counts as two things."""
    if is_headline(text):
        text = text.lower()
    found = set()
    for match in hotpot.CAP_RE.finditer(text):
        for piece in CONJUNCTION.split(match.group(0)):
            value = hotpot.normalize_entity(piece)
            if len(value) < 2 or value.isdigit() or len(value.split()) > 6:
                continue
            if value in hotpot.STOP_ENTITIES or value in EXTRA_STOP:
                continue
            found.add(value)
    return found


def cues(question: str) -> dict[str, bool]:
    """Which rules fire. Exposed so a run can report *why* the router fired."""
    text = question.lower() if is_headline(question) else question
    words = WORD.findall(text.lower())
    found = entities(question)
    return {
        "comparison": bool(COMPARISON.search(text)) and len(found) >= 1,
        "anaphora": any(word in ANAPHOR for word in words[1:]) and len(found) >= 1,
        "whchain": len({word for word in words if word in WH_WORDS}) >= 2,
        "pair_relation": len(found) >= 2 and bool(RELATION.search(text)),
    }


def is_multihop(question: str) -> bool:
    return any(cues(question).values())


def route(question: str) -> str:
    return "ebr" if is_multihop(question) else "bm25"
