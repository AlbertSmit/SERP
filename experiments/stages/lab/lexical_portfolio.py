"""E1: non-neural heterogeneous lexical portfolio over a Porter-stemmed index.

Seven classical lexical scorers with frozen constants (BM25, BM25+, BM25L, PL2,
DPH, DirichletLM, HiemstraLM), one fixed RM3-style pseudo-relevance expansion of
BM25, and fixed fusion (RRF k=60, and NAF = per-model min-max then mean).

Nothing here is tuned. The module only builds impact matrices, scores queries,
ranks them, and computes the same metrics as `harness`. Formulas and constants
are documented in `experiments/stages/LEXICAL_PORTFOLIO.md`; the docstrings below
repeat the constants so an audit can read code and report side by side.

Scoring is `s(q, d) = sum_t w_t * impact[t, d]` for every model, where the impact
matrix is built once per model over a porter-stemmed `harness.Index`. The two
language models additionally contribute a document-level term for query words
that do not occur in the document (DirichletLM's missing-term mass depends on
`dl`, HiemstraLM's does not).
"""
from __future__ import annotations

import gc
import math
import time
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

import harness
from channels import porter

# --- frozen constants (do not change without a new experiment id) -------------
K1 = 1.2
B = 0.75
DELTA = 0.5            # BM25+ and BM25L
PL2_C = 1.0
DPH_C = 1.0
MU = 1500.0            # DirichletLM
HIEM_LAMBDA = 0.7      # HiemstraLM
RM3_LAMBDA = 0.6       # weight on the original query
RM3_TERMS = 20
RM3_DOCS = 10
RRF_K = 60
DEPTH = 1000

LOG2 = math.log(2.0)
LOG2E = 1.0 / math.log(2.0)     # log2(e)
TWO_PI = 2.0 * math.pi

MODELS = ("bm25", "bm25plus", "bm25l", "pl2", "dph", "dirichletlm", "hiemstralm")
ALL_MODELS = MODELS + ("rm3",)
ABLATIONS = ("bm25_plus_pl2", "bm25_plus_dph", "bm25_plus_lm")
SYSTEMS = (
    "bm25", "bm25plus", "bm25l", "pl2", "dph", "dirichletlm", "hiemstralm", "rm3",
    "portfolio_rrf", "portfolio_naf", "portfolio_rrf_no_rm3",
    "bm25_plus_pl2", "bm25_plus_dph", "bm25_plus_lm",
)
PORTFOLIO_MEMBERS = ALL_MODELS
NO_RM3_MEMBERS = MODELS
SYSTEM_MEMBERS = {
    **{m: (m,) for m in ALL_MODELS},
    "portfolio_rrf": PORTFOLIO_MEMBERS,
    "portfolio_naf": PORTFOLIO_MEMBERS,
    "portfolio_rrf_no_rm3": NO_RM3_MEMBERS,
    "bm25_plus_pl2": ("bm25", "pl2"),
    "bm25_plus_dph": ("bm25", "dph"),
    "bm25_plus_lm": ("bm25", "dirichletlm"),
}
PRIMARY_PORTFOLIO = "portfolio_rrf"


def porter_index(ix: harness.Index) -> harness.Index:
    """One Porter-transformed index per dataset: same doc ids, stemmed title + body."""
    return harness.Index(
        doc_ids=ix.doc_ids,
        title_tokens=[porter(toks) for toks in ix.title_tokens],
        body_tokens=[porter(toks) for toks in ix.body_tokens],
    )


# --- ranking -----------------------------------------------------------------
def topk(scores: np.ndarray, k: int = DEPTH, positive_only: bool = True) -> np.ndarray:
    """Top-k doc indices ordered by (-score, doc index) like `harness.rank`.

    `positive_only` mirrors `harness.rank`'s behaviour for sparse scorers: documents
    with no matching query term (score exactly 0) never enter the list. Language
    models score every document and are ranked with `positive_only=False`.
    """
    n = scores.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n <= k:
        cand = np.arange(n, dtype=np.int64)
    else:
        cand = np.argpartition(scores, n - k)[n - k:].astype(np.int64, copy=False)
    if positive_only:
        cand = cand[scores[cand] > 0]
    if cand.size == 0:
        return cand
    if n > k and cand.size >= k:
        # include every document tied with the k-th value so the doc-id tie-break
        # matches a full sort whenever the tie set is small.
        threshold = scores[cand].min()
        tied = np.flatnonzero(scores == threshold)
        if 0 < tied.size <= 20000:
            cand = np.union1d(cand, tied)
    order = np.lexsort((cand, -scores[cand]))
    return cand[order][:k]


class Model:
    """One lexical scorer over one Porter index, as a sparse term x doc impact matrix."""

    def __init__(self, name: str, ix: harness.Index):
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}")
        self.name = name
        self.ix = ix
        self.terms = list(ix.postings)
        self.row = {t: i for i, t in enumerate(self.terms)}
        self.total = float(ix.total_tokens)
        self._dl = np.asarray(ix.dl, dtype=np.float64)
        self._avgdl = float(ix.avgdl if ix.avgdl > 0 else 1.0)
        self._norm = K1 * (1.0 - B + B * self._dl / self._avgdl)
        if name == "dirichletlm":
            self._log_dl_mu = np.log(self._dl + MU)
        if name == "hiemstralm":
            self._log_dl = np.log(np.maximum(self._dl, 1.0))
        self.M = self._build()

    def _build(self) -> csr_matrix:
        ix = self.ix
        name = self.name
        nnz = sum(len(p) for p in ix.postings.values())
        rows = np.empty(nnz, dtype=np.int32)
        cols = np.empty(nnz, dtype=np.int32)
        vals = np.empty(nnz, dtype=np.float64)
        dl, avgdl, N = self._dl, self._avgdl, ix.N
        log2 = math.log2
        pos = 0
        for i, term in enumerate(self.terms):
            postings = ix.postings[term]
            cf = ix.cf.get(term, 0)
            idf = ix.idf(term)
            lam = cf / N if N else 0.0
            p_c = cf / self.total if self.total else 0.0
            for d, tf in postings:
                doc_len = dl[d]
                if name == "bm25":
                    v = idf * tf * (K1 + 1.0) / (tf + self._norm[d])
                elif name == "bm25plus":
                    v = idf * (tf * (K1 + 1.0) / (tf + self._norm[d]) + DELTA)
                elif name == "bm25l":
                    ctd = tf / (1.0 - B + B * doc_len / avgdl) if doc_len > 0 else 0.0
                    v = idf * (ctd * (K1 + 1.0) / (ctd + K1) + DELTA)
                elif name == "pl2":
                    tfn = tf * log2(1.0 + PL2_C * avgdl / doc_len) if doc_len > 0 else 0.0
                    if tfn <= 0.0 or lam <= 0.0:
                        v = 0.0
                    else:
                        v = (1.0 / (tfn + 1.0)) * (
                            tfn * log2(tfn / lam)
                            + (lam - tfn) * LOG2E
                            + 0.5 * log2(TWO_PI * tfn)
                        )
                elif name == "dph":
                    # Terrier DPH (matching.models.DPH). The variant in the E1 brief
                    # gives a score that decreases with tf (norm 1/(tfn+1)^3); the
                    # Terrier source is authoritative. See LEXICAL_PORTFOLIO.md.
                    if tf <= 0.0 or doc_len <= 0.0 or cf <= 0:
                        v = 0.0
                    else:
                        f = tf / doc_len
                        tail = TWO_PI * tf * max(1.0 - f, 1e-12)  # tf==dl would log(0)
                        v = ((1.0 - f) ** 2 / (tf + 1.0)) * (
                            tf * log2(tf * (avgdl / doc_len) * (N / cf))
                            + 0.5 * log2(tail)
                        )
                        if not np.isfinite(v):
                            v = 0.0
                elif name == "dirichletlm":
                    # log((tf + mu p) / (mu p)): the absent-term mass is added in score()
                    if p_c <= 0.0:
                        v = 0.0
                    else:
                        v = math.log1p(tf / (MU * p_c))
                elif name == "hiemstralm":
                    # log(1 + lambda tf / (dl (1-lambda) p)) on top of the absent-term mass
                    if p_c <= 0.0 or doc_len <= 0.0:
                        v = 0.0
                    else:
                        v = math.log1p(
                            HIEM_LAMBDA * tf / (doc_len * (1.0 - HIEM_LAMBDA) * p_c)
                        )
                else:  # pragma: no cover
                    raise ValueError(name)
                rows[pos] = i
                cols[pos] = d
                vals[pos] = v
                pos += 1
        return csr_matrix((vals, (rows, cols)), shape=(len(self.terms), ix.N))

    @property
    def positive_only(self) -> bool:
        return self.name not in ("dirichletlm", "hiemstralm")

    def score(self, weights: dict[str, float]) -> np.ndarray:
        """Dense score vector for a weighted query."""
        s = np.zeros(self.ix.N, dtype=np.float64)
        idx: list[int] = []
        w: list[float] = []
        for term, weight in weights.items():
            r = self.row.get(term)
            if r is not None:
                idx.append(r)
                w.append(weight)
        if idx:
            s += self.M[idx].T @ np.asarray(w, dtype=np.float64)
        if self.name == "dirichletlm":
            const = 0.0
            mass = 0.0
            for term, weight in weights.items():
                p_c = self.ix.cf.get(term, 0) / self.total if self.total else 0.0
                if p_c <= 0.0:
                    continue            # unseen term: no evidence, Terrier ignores it
                const += weight * math.log(MU * p_c)
                mass += weight
            if mass:
                # every document pays the absent-term mass; present terms add the
                # relative boost stored in the impact matrix
                s += const - mass * self._log_dl_mu
        elif self.name == "hiemstralm":
            const = 0.0
            for term, weight in weights.items():
                p_c = self.ix.cf.get(term, 0) / self.total if self.total else 0.0
                if p_c <= 0.0:
                    continue
                const += weight * math.log((1.0 - HIEM_LAMBDA) * p_c)
            if const:
                s += const
        return s

    def free(self) -> None:
        self.M = None
        self.terms = None
        self.row = None
        gc.collect()


def ranks(scores: np.ndarray, k: int = DEPTH, positive_only: bool = True) -> np.ndarray:
    return topk(scores, k=k, positive_only=positive_only)


# --- RM3 ---------------------------------------------------------------------
def rm3_weights(ix: harness.Index, bm25: Model, query_tokens: list[str],
                exclude: int | None = None) -> dict[str, float]:
    """RM3-style relevance-model expansion of a BM25 query (not the full RM1/RM3 EM).

    p(w|d) = tf/dl over the BM25 top-10; p(q|d) = softmax of the 10 BM25 scores;
    p(w|R) = sum_d p(w|d) p(q|d), normalized; q' = 0.6 p(w|q) + 0.4 p(w|R); top 20.
    """
    qtf = Counter(query_tokens)
    scores_all = bm25.score(qtf)
    if exclude is not None:
        scores_all[exclude] = -np.inf
    top = topk(scores_all, RM3_DOCS, positive_only=True)
    if top.size == 0:
        n = sum(qtf.values()) or 1
        return {t: c / n for t, c in qtf.items()}
    scores = scores_all[top]
    shifted = np.exp(scores - scores.max())
    p_d = shifted / shifted.sum()
    agg: dict[str, float] = {}
    for d, pd in zip(top, p_d):
        dl = ix.dl[d]
        if dl <= 0:
            continue
        scale = float(pd) / dl
        for term, tf in ix.tf[d].items():
            agg[term] = agg.get(term, 0.0) + tf * scale
    mass = sum(agg.values())
    if mass <= 0.0:
        n = sum(qtf.values()) or 1
        return {t: c / n for t, c in qtf.items()}
    n = sum(qtf.values()) or 1
    combined: dict[str, float] = {}
    for term, w in agg.items():
        combined[term] = (1.0 - RM3_LAMBDA) * w / mass
    for term, c in qtf.items():
        combined[term] = combined.get(term, 0.0) + RM3_LAMBDA * c / n
    ranked = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))[:RM3_TERMS]
    return {t: w for t, w in ranked if w > 0.0}


# --- evaluation --------------------------------------------------------------
def evaluate(model: Model, queries: dict, qrels: dict, depth: int = DEPTH,
             expand=None, self_ids: dict[str, int] | None = None) -> tuple[dict, dict, float]:
    """Rank every query. Returns (lists, per_query metrics, ms/query).

    `lists[qid] = (indices int32, scores float32)` feeds fusion without keeping the
    dense matrix or the score vectors alive. `self_ids` are the BEIR ArguAna-style
    self documents (a query that is also a document); they are pushed to -inf
    before ranking, matching the frozen first_principles protocol.
    """
    ix = model.ix
    lists: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_query: dict[str, dict[str, float]] = {}
    t0 = time.perf_counter()
    for qid, q in queries.items():
        weights = expand(qid, q) if expand is not None else Counter(q)
        s = model.score(weights)
        if self_ids and qid in self_ids:
            s[self_ids[qid]] = -np.inf
        idx = ranks(s, depth, model.positive_only)
        lists[qid] = (idx.astype(np.int32, copy=False), s[idx].astype(np.float32, copy=False))
        per_query[qid] = harness.metrics([ix.doc_ids[d] for d in idx], qrels[qid])
        del s
    ms = (time.perf_counter() - t0) * 1000.0 / max(len(queries), 1)
    return lists, per_query, ms


def _fuse_query(arrays: list[tuple[np.ndarray, np.ndarray]], method: str) -> tuple[np.ndarray, np.ndarray]:
    idxs = [a[0] for a in arrays if a[0].size]
    if not idxs:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    union = np.unique(np.concatenate(idxs))
    if method == "rrf":
        concat = np.concatenate(idxs)
        ranks_cat = np.concatenate([np.arange(1, a[0].size + 1) for a in arrays if a[0].size])
        _, inv = np.unique(concat, return_inverse=True)
        fused = np.bincount(inv, weights=1.0 / (RRF_K + ranks_cat), minlength=union.size)
    elif method == "naf":
        fused = np.zeros(union.size, dtype=np.float64)
        for idx, sc in arrays:
            if idx.size == 0:
                continue
            lo = float(sc.min())
            hi = float(sc.max())
            norm = (sc - lo) / (hi - lo) if hi > lo else np.zeros(sc.shape, dtype=np.float64)
            fused[np.searchsorted(union, idx)] += norm
    else:  # pragma: no cover
        raise ValueError(method)
    order = np.lexsort((union, -fused))[:DEPTH]
    return union[order].astype(np.int32, copy=False), fused[order].astype(np.float32, copy=False)


def fuse(lists_by_model: dict[str, dict], members: tuple[str, ...], method: str,
         qids) -> dict:
    out = {}
    for qid in qids:
        out[qid] = _fuse_query([lists_by_model[m][qid] for m in members], method)
    return out


def metrics_for(ix: harness.Index, lists: dict, qrels: dict) -> dict[str, dict[str, float]]:
    return {
        qid: harness.metrics([ix.doc_ids[d] for d in idx], qrels[qid])
        for qid, (idx, _sc) in lists.items()
    }


# --- statistics --------------------------------------------------------------
def mean_metric(per_query: dict, metric: str) -> float:
    return float(np.mean([v[metric] for v in per_query.values()])) if per_query else 0.0


def paired_bootstrap(base: dict, cand: dict, metric: str = "ndcg",
                     reps: int = 1000, seed: int = 1729) -> dict:
    """Paired bootstrap over queries, 95% CI of the mean delta. Fixed seed."""
    qids = [q for q in base if q in cand]
    deltas = np.asarray([cand[q][metric] - base[q][metric] for q in qids], dtype=np.float64)
    if deltas.size == 0:
        return {"delta": 0.0, "ci95": [0.0, 0.0], "wtl": [0, 0, 0], "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, deltas.size, size=(reps, deltas.size))
    boots = np.sort(deltas[draws].mean(axis=1))
    wins = int((deltas > 1e-12).sum())
    losses = int((deltas < -1e-12).sum())
    return {
        "delta": float(deltas.mean()),
        "ci95": [float(boots[int(0.025 * reps)]), float(boots[min(int(0.975 * reps), reps - 1)])],
        "wtl": [wins, int(deltas.size - wins - losses), losses],
        "n": int(deltas.size),
    }


def gate(per_dataset: dict, primary: str, margin: float = -0.005, need_positive: int = 4) -> dict:
    """Frozen decision rule from ARGON_PLAN A4/E1.

    A portfolio passes when its macro nDCG@10 is above Porter BM25, no dataset has
    a delta below `margin` with a CI that excludes that margin, and the point
    estimate is positive on at least `need_positive` of the seven datasets.
    """
    avail = {n: d["systems"] for n, d in per_dataset.items()
             if primary in d.get("systems", {}) and "bm25" in d.get("systems", {})}
    rows = {name: avail[name][primary] for name in avail}
    positive = [n for n, r in rows.items() if r["delta_ndcg"] > 0]
    fails = [n for n, r in rows.items()
             if r["delta_ndcg"] < margin and r["ci95"][1] < margin]
    macro = float(np.mean([r["ndcg10"] for r in rows.values()]))
    macro_base = float(np.mean([avail[n]["bm25"]["ndcg10"] for n in avail]))
    checks = {
        "macro_ndcg_above_bm25": bool(macro > macro_base),
        "no_dataset_below_margin": bool(not fails),
        "positive_on_at_least_4_of_7": bool(len(positive) >= need_positive),
    }
    return {
        "primary_system": primary,
        "datasets_considered": list(rows),
        "macro_ndcg": macro,
        "macro_bm25_ndcg": macro_base,
        "macro_delta": macro - macro_base,
        "positive_datasets": positive,
        "failing_datasets": fails,
        "margin": margin,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
