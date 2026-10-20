"""E2: per-query selective lexical model selector with a hard BM25 fallback.

The selector picks one of three frozen E1 scorers for a query — `bm25`, `rm3`
(the fixed RM3-style expansion of BM25) or `dph` — using features that exist at
inference time only: query statistics, corpus statistics, and diagnostics of
each candidate's own ranked list. No qrels, no qid, no dataset name.

Training is supervised on the three screening splits the project already uses
(nfcorpus/dev, scifact/train, fiqa/dev): one gradient-boosted regressor per
candidate predicts that candidate's nDCG@10 for the query. At inference the
predicted best candidate is taken only when its predicted gain over BM25 clears
a margin; otherwise BM25 runs. That fallback is the point of the design: BM25 is
the default and the selector has to earn a switch.

Every model, scorer, metric and statistic is imported from `lexical_portfolio`
(E1) and `harness`; this module adds only feature extraction and the selection
policy. Formulas and the frozen experiment definition live in
`experiments/stages/SELECTIVE_LEXICAL.md`.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

import harness
from channels import porter

# --- frozen experiment definition --------------------------------------------
CANDIDATES = ("bm25", "rm3", "dph")
MARGIN = 0.005            # predicted nDCG gain a switch must clear
MARGIN_SAFE = 0.020       # conservative variant
TOPK = 100                # depth of the list-diagnostic features
HGB_PARAMS = {
    "max_iter": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "l2_regularization": 1.0,
    "min_samples_leaf": 20,
    "random_state": 1729,
}
CLF_PARAMS = dict(HGB_PARAMS)
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 1729

# --- frozen word lists (query tokens and these lists are Porter-stemmed) -----
STOPWORDS = (
    "a about above after again against all am an and any are aren as at be because been "
    "before being below between both but by can cannot could couldn did didn do does doesn "
    "doing don down during each few for from further had hadn has hasn have haven having he "
    "her here hers herself him himself his how i if in into is isn it its itself just me more "
    "most mustn my myself no nor not now of off on once only or other ought our ours ourselves "
    "out over own same shan she should shouldn so some such than that the their theirs them "
    "themselves then there these they this those through to too under until up very was wasn we "
    "were weren what when where which while who whom why with won would wouldn you your yours "
    "yourself yourselves"
).split()

WH_WORDS = ("what", "which", "who", "whom", "whose", "when", "where", "why", "how")
BOOLEAN_WORDS = ("and", "or", "not", "without", "versus", "vs", "excluding", "except", "nor")
COMPARISON_WORDS = (
    "more", "less", "than", "greater", "fewer", "best", "better", "most", "least",
    "largest", "smallest", "highest", "lowest", "compared", "compare", "difference",
    "between", "before", "after", "versus", "vs", "increase", "decrease",
)

STOP_STEMS = frozenset(porter(list(STOPWORDS)))
WH_STEMS = frozenset(porter(list(WH_WORDS)))
BOOLEAN_STEMS = frozenset(porter(list(BOOLEAN_WORDS)))
COMPARISON_STEMS = frozenset(porter(list(COMPARISON_WORDS)))

# --- feature names ------------------------------------------------------------
QUERY_FEATURES = (
    "q_len",
    "q_unique",
    "q_stop_ratio",
    "q_oov_rate",
    "q_idf_mean",
    "q_idf_max",
    "q_idf_sum",
    "q_clarity",
    "q_wh",
    "q_boolean",
    "q_comparison",
    "q_digit_ratio",
    "q_identifier",
    "q_len_rel",
)
BM25_FEATURES = (
    "bm_top1",
    "bm_top10_mean",
    "bm_gap",
    "bm_gap_rel",
    "bm_entropy",
    "bm_pos_docs",
    "bm_pos_rate",
)
RM3_FEATURES = (
    "rm3_added_terms",
    "rm3_added_mass",
    "rm3_added_idf_mean",
    "rm3_added_idf_max",
    "rm3_orig_kept",
    "rm3_drift",
    "rm3_rank_overlap",
)
DPH_FEATURES = (
    "dph_top1",
    "dph_top10_mean",
    "dph_entropy",
    "dph_rank_overlap",
)
CORPUS_FEATURES = (
    "c_log_n",
    "c_log_avgdl",
    "c_log_vocab",
    "c_vocab_rate",
)
FEATURES = QUERY_FEATURES + BM25_FEATURES + RM3_FEATURES + DPH_FEATURES + CORPUS_FEATURES
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURES)}


def _has_digit(token: str) -> bool:
    return any(c.isdigit() for c in token)


def _is_identifier(token: str) -> bool:
    """Alphanumeric identifier such as `covid19`, `b12`, `3d` (no underscore survives tokenizing)."""
    digit = _has_digit(token)
    alpha = any(c.isalpha() for c in token)
    return digit and alpha


# --- feature blocks -----------------------------------------------------------
def query_features(ix: harness.Index, tokens: list[str]) -> dict[str, float]:
    """Query-only features. No qrels, no doc identities; IDF/clarity come from the index."""
    n = len(tokens)
    unique = set(tokens)
    total = float(ix.total_tokens) if ix.total_tokens else 1.0
    idfs = [ix.idf(t) for t in unique]
    qtf = Counter(tokens)
    clarity = 0.0
    for term, count in qtf.items():
        p_q = count / n if n else 0.0
        p_c = ix.cf.get(term, 0) / total
        if p_c > 0.0:
            clarity += p_q * math.log2(p_q / p_c)
    return {
        "q_len": float(n),
        "q_unique": float(len(unique)),
        "q_stop_ratio": (sum(1 for t in tokens if t in STOP_STEMS) / n) if n else 0.0,
        "q_oov_rate": (sum(1 for t in tokens if t not in ix.postings) / n) if n else 0.0,
        "q_idf_mean": float(np.mean(idfs)) if idfs else 0.0,
        "q_idf_max": float(np.max(idfs)) if idfs else 0.0,
        "q_idf_sum": float(np.sum(idfs)) if idfs else 0.0,
        "q_clarity": float(clarity),
        "q_wh": 1.0 if unique & WH_STEMS else 0.0,
        "q_boolean": 1.0 if unique & BOOLEAN_STEMS else 0.0,
        "q_comparison": 1.0 if unique & COMPARISON_STEMS else 0.0,
        "q_digit_ratio": (sum(1 for t in tokens if _has_digit(t)) / n) if n else 0.0,
        "q_identifier": 1.0 if any(_is_identifier(t) for t in tokens) else 0.0,
        "q_len_rel": (n / ix.avgdl) if ix.avgdl else 0.0,
    }


def ranking_stats(idx: np.ndarray, scores: np.ndarray, prefix: str, k: int = TOPK) -> dict[str, float]:
    """Top-1, top-10 mean, top1-top2 gap and normalised score entropy of one ranked list."""
    top_idx = idx[:k]
    top = np.asarray(scores[:k], dtype=np.float64)
    n = top.size
    if n == 0:
        return {f"{prefix}top1": 0.0, f"{prefix}top10_mean": 0.0, f"{prefix}gap": 0.0,
                f"{prefix}gap_rel": 0.0, f"{prefix}entropy": 0.0}
    top1 = float(top[0])
    top10 = float(top[:10].mean())
    second = float(top[1]) if n > 1 else top1
    shifted = np.exp(top - top.max())
    p = shifted / shifted.sum()
    entropy = float(-(p * np.log(p + 1e-300)).sum() / math.log(n)) if n > 1 else 0.0
    out = {
        f"{prefix}top1": top1,
        f"{prefix}top10_mean": top10,
        f"{prefix}gap": top1 - second,
        f"{prefix}gap_rel": (top1 - second) / (abs(top1) + 1e-9),
        f"{prefix}entropy": entropy,
    }
    del top_idx
    return out


def overlap(a: np.ndarray, b: np.ndarray, k: int = TOPK) -> float:
    """Fraction of the shorter list shared with the other, at depth k."""
    sa = set(int(x) for x in np.asarray(a)[:k])
    sb = set(int(x) for x in np.asarray(b)[:k])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def rm3_features(ix: harness.Index, query_tokens: list[str], weights: dict[str, float],
                 bm25_idx: np.ndarray, rm3_idx: np.ndarray) -> dict[str, float]:
    """Expansion diagnostics: how much mass the feedback model added, and how far it moved the list."""
    original = set(query_tokens)
    expanded = set(weights)
    added = expanded - original
    added_mass = float(sum(weights[t] for t in added))
    added_idfs = [ix.idf(t) for t in added]
    union = original | expanded
    return {
        "rm3_added_terms": float(len(added)),
        "rm3_added_mass": added_mass,
        "rm3_added_idf_mean": float(np.mean(added_idfs)) if added_idfs else 0.0,
        "rm3_added_idf_max": float(np.max(added_idfs)) if added_idfs else 0.0,
        "rm3_orig_kept": (len(original & expanded) / len(original)) if original else 0.0,
        "rm3_drift": (1.0 - len(original & expanded) / len(union)) if union else 0.0,
        "rm3_rank_overlap": overlap(bm25_idx, rm3_idx),
    }


def corpus_features(ix: harness.Index) -> dict[str, float]:
    """Corpus-level priors. Legitimate at inference (you know the index you search)."""
    return {
        "c_log_n": math.log1p(ix.N),
        "c_log_avgdl": math.log1p(ix.avgdl),
        "c_log_vocab": math.log1p(len(ix.df)),
        "c_vocab_rate": len(ix.df) / ix.N if ix.N else 0.0,
    }


def feature_row(*blocks: dict[str, float]) -> np.ndarray:
    merged: dict[str, float] = {}
    for block in blocks:
        merged.update(block)
    missing = [name for name in FEATURES if name not in merged]
    if missing:  # catches a feature defined in FEATURES but never computed
        raise KeyError(f"missing features: {missing}")
    row = np.asarray([merged[name] for name in FEATURES], dtype=np.float64)
    if not np.all(np.isfinite(row)):
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    return row


# --- selection policy ---------------------------------------------------------
def choose(predicted: np.ndarray, margin: float = MARGIN) -> np.ndarray:
    """Candidate index per query: BM25 unless a candidate's predicted gain clears `margin`.

    `predicted` is (n_queries, len(CANDIDATES)) predicted nDCG@10, columns in CANDIDATES order.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    best = np.argmax(predicted, axis=1)
    gain = predicted[np.arange(predicted.shape[0]), best] - predicted[:, 0]
    chosen = np.where(gain > margin, best, 0)
    return chosen.astype(np.int64)


class Selector:
    """One HistGradientBoostingRegressor per candidate, trained on per-query nDCG@10."""

    def __init__(self, margin: float = MARGIN, params: dict | None = None):
        self.margin = margin
        self.params = dict(params or HGB_PARAMS)
        self.models: dict[str, object] = {}

    def fit(self, X: np.ndarray, targets: dict[str, np.ndarray]) -> "Selector":
        from sklearn.ensemble import HistGradientBoostingRegressor
        for name in CANDIDATES:
            model = HistGradientBoostingRegressor(**self.params)
            model.fit(X, np.asarray(targets[name], dtype=np.float64))
            self.models[name] = model
        return self

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        """(n, 3) predicted nDCG@10, clipped to [0, 1]."""
        cols = [np.clip(self.models[name].predict(X), 0.0, 1.0) for name in CANDIDATES]
        return np.column_stack(cols)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return choose(self.predict_matrix(X), self.margin)


class ClassifierSelector:
    """3-class variant: predict which candidate wins, take the argmax class (no margin)."""

    def __init__(self, params: dict | None = None):
        self.params = dict(params or CLF_PARAMS)
        self.model = None

    def fit(self, X: np.ndarray, best: np.ndarray) -> "ClassifierSelector":
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(**self.params)
        self.model.fit(X, np.asarray(best, dtype=np.int64))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.int64)


def selection_diagnostics(chosen: np.ndarray, actual: dict[str, np.ndarray]) -> dict:
    """Which candidate was taken, and whether it actually beat BM25 on those queries."""
    n = int(chosen.size)
    base = np.asarray(actual["bm25"], dtype=np.float64)
    rows = {}
    for i, name in enumerate(CANDIDATES):
        mask = chosen == i
        count = int(mask.sum())
        if count == 0 or name == "bm25":
            rows[name] = {
                "count": count,
                "share": (count / n) if n else 0.0,
                "actual_win_rate": None,
                "actual_mean_delta": 0.0,
            }
            continue
        delta = np.asarray(actual[name], dtype=np.float64)[mask] - base[mask]
        rows[name] = {
            "count": count,
            "share": count / n,
            "actual_win_rate": float((delta > 1e-12).mean()),
            "actual_mean_delta": float(delta.mean()),
        }
    return rows
