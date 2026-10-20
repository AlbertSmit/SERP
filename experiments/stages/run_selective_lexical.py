#!/usr/bin/env python3
"""ARGON E2: per-query selective lexical model selector with a hard BM25 fallback.

Train on the three screening splits only (nfcorpus/dev, scifact/train, fiqa/dev):
one `HistGradientBoostingRegressor` per candidate predicts that candidate's
nDCG@10 from qrels-free features. Test once on the seven frozen BEIR test sets.
A switch happens only when the predicted gain over BM25 clears a margin; else
BM25 runs. No test query is used for training, tuning, or feature selection.

Usage:
  PYTHONHASHSEED=0 ../../.venv/bin/python -u run_selective_lexical.py
  PYTHONHASHSEED=0 ../../.venv/bin/python -u run_selective_lexical.py --datasets scifact nfcorpus
  PYTHONHASHSEED=0 ../../.venv/bin/python -u run_selective_lexical.py --limit 40   # smoke only
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lab"))

import numpy as np

import harness
import lexical_portfolio as lp
import selective_lexical as sl
from channels import porter

RESULTS = HERE / "results"
E1_RESULTS = RESULTS / "lexical_portfolio.json"
TRAIN_SETS = (("nfcorpus", "dev"), ("scifact", "train"), ("fiqa", "dev"))
TEST_SETS = (
    ("scifact", "test"),
    ("nfcorpus", "test"),
    ("fiqa", "test"),
    ("arguana", None),
    ("scidocs", None),
    ("trec-covid", None),
    ("quora", None),
)
TEST_SPLIT = dict(TEST_SETS)
RESAMPLES = 1000
CI_SEED = 1729
BOOT_SYSTEMS = ("selector", "selector_safe", "selector_clf", "always_rm3", "always_dph", "oracle")


def log(message: str) -> None:
    print(message, flush=True)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def verify_bm25(pix: harness.Index, model: lp.Model, queries: dict, n: int = 20) -> dict:
    """Reproduce E1's baseline check: our BM25 matrix must equal `harness.bm25_fast`."""
    qids = list(queries)[:n]
    worst_score, worst_top, checked = 0.0, 0, 0
    for qid in qids:
        q = queries[qid]
        fast = np.asarray(harness.bm25_fast(pix, q), dtype=np.float64)
        ours = model.score(Counter(q))
        worst_score = max(worst_score, float(np.max(np.abs(fast - ours))))
        if harness.rank(pix, fast, 100) != [pix.doc_ids[d] for d in lp.topk(ours, 100)]:
            worst_top += 1
        checked += 1
    pix.__dict__.pop("_impacts", None)
    pix.__dict__.pop("_row", None)
    gc.collect()
    ok = worst_score < 1e-8 and worst_top == 0
    log(f"    baseline check: max|score diff|={worst_score:.3e}, "
        f"top-100 mismatches={worst_top}/{checked} -> {'MATCH' if ok else 'MISMATCH'}")
    return {"queries": checked, "max_abs_score_diff": worst_score,
            "top100_mismatches": worst_top, "match": ok}


def positive_docs(model: lp.Model, weights: dict[str, float]) -> int:
    """Documents with at least one matching query term (BM25 scores are > 0 exactly there)."""
    rows = [model.row[t] for t in weights if t in model.row]
    if not rows:
        return 0
    sub = model.M[sorted(set(rows))]
    if sub.nnz == 0:
        return 0
    return int(np.count_nonzero(sub.getnnz(axis=0)))


def build_dataset(name: str, split: str | None, limit: int | None) -> dict:
    """Run the three candidates on one dataset and extract qrels-free features per query."""
    t_start = time.perf_counter()
    log(f"\n=== {name} ({split or 'default test'}) ===")
    t0 = time.perf_counter()
    raw, queries, qrels = harness.load(name, confirm_split=split)
    n_docs = int(raw.N)
    log(f"    loaded index: N={raw.N:,} docs, vocab={len(raw.df):,}, avgdl={raw.avgdl:.1f}, "
        f"{time.perf_counter() - t0:.1f}s")
    if limit:
        queries = dict(list(queries.items())[:limit])
    queries = {qid: porter(toks) for qid, toks in queries.items()}
    pix = lp.porter_index(raw)
    position = {d: i for i, d in enumerate(pix.doc_ids)}
    self_ids = {qid: position[qid] for qid in queries if qid in position}
    del raw
    gc.collect()
    log(f"    queries={len(queries):,}, qrels={sum(len(v) for v in qrels.values()):,}, "
        f"self-documents ignored={len(self_ids)}, rss={rss_gb():.2f} GB")

    per_query: dict[str, dict] = {}
    lists: dict[str, dict] = {}
    pos_docs: dict[str, int] = {}
    rm3_weights: dict[str, dict[str, float]] = {}
    timing: dict[str, float] = {}
    base_check = None

    model = lp.Model("bm25", pix)
    t0 = time.perf_counter()
    base_check = verify_bm25(pix, model, queries)
    lists["bm25"], per_query["bm25"], timing["bm25"] = lp.evaluate(
        model, queries, qrels, depth=lp.DEPTH, self_ids=self_ids
    )
    for qid, q in queries.items():
        pos_docs[qid] = positive_docs(model, Counter(q))
    log(f"    bm25        nDCG@10={lp.mean_metric(per_query['bm25'], 'ndcg'):.4f} "
        f"{timing['bm25']:.2f} ms/q ({time.perf_counter() - t0:.1f}s incl. check)")

    def expand(qid: str, q: list[str]) -> dict[str, float]:
        w = lp.rm3_weights(pix, model, q, exclude=self_ids.get(qid))
        rm3_weights[qid] = w
        return w

    lists["rm3"], per_query["rm3"], timing["rm3"] = lp.evaluate(
        model, queries, qrels, depth=lp.DEPTH, expand=expand, self_ids=self_ids
    )
    log(f"    rm3         nDCG@10={lp.mean_metric(per_query['rm3'], 'ndcg'):.4f} "
        f"{timing['rm3']:.2f} ms/q")
    model.free()
    del model
    gc.collect()

    model = lp.Model("dph", pix)
    lists["dph"], per_query["dph"], timing["dph"] = lp.evaluate(
        model, queries, qrels, depth=lp.DEPTH, self_ids=self_ids
    )
    log(f"    dph         nDCG@10={lp.mean_metric(per_query['dph'], 'ndcg'):.4f} "
        f"{timing['dph']:.2f} ms/q")
    model.free()
    del model
    gc.collect()

    t0 = time.perf_counter()
    corpus_block = sl.corpus_features(pix)
    qids = list(queries)
    X = np.empty((len(qids), len(sl.FEATURES)), dtype=np.float64)
    for i, qid in enumerate(qids):
        tokens = queries[qid]
        bm_idx, bm_scores = lists["bm25"][qid]
        rm_idx, _rm_scores = lists["rm3"][qid]
        dph_idx, dph_scores = lists["dph"][qid]
        bm_block = sl.ranking_stats(bm_idx, bm_scores, "bm_")
        bm_block["bm_pos_docs"] = float(pos_docs[qid])
        bm_block["bm_pos_rate"] = pos_docs[qid] / pix.N if pix.N else 0.0
        dph_block = sl.ranking_stats(dph_idx, dph_scores, "dph_")
        dph_block["dph_rank_overlap"] = sl.overlap(bm_idx, dph_idx)
        X[i] = sl.feature_row(
            sl.query_features(pix, tokens),
            bm_block,
            sl.rm3_features(pix, tokens, rm3_weights[qid], bm_idx, rm_idx),
            dph_block,
            corpus_block,
        )
    log(f"    features    {X.shape[0]:,} queries x {X.shape[1]} features "
        f"({time.perf_counter() - t0:.1f}s)")

    meta = {
        "split": split or "test",
        "n_docs": n_docs,
        "n_queries": len(queries),
        "n_qrels": sum(len(v) for v in qrels.values()),
        "self_documents_ignored": len(self_ids),
        "baseline_check": base_check,
        "ms_per_query": timing,
        "wall_s": time.perf_counter() - t_start,
    }
    del lists, rm3_weights, pix
    gc.collect()
    return {"qids": qids, "X": X, "per_query": per_query, "meta": meta}


def actual_matrix(per_query: dict, qids: list[str]) -> dict[str, np.ndarray]:
    return {name: np.asarray([per_query[name][q]["ndcg"] for q in qids], dtype=np.float64)
            for name in sl.CANDIDATES}


def system_per_query(per_query: dict, qids: list[str], chosen: np.ndarray) -> dict[str, dict[str, float]]:
    """Metrics of the chosen candidate's own ranked list, query by query."""
    return {qid: per_query[sl.CANDIDATES[chosen[i]]][qid] for i, qid in enumerate(qids)}


def oracle_per_query(per_query: dict, qids: list[str]) -> dict[str, dict[str, float]]:
    """Upper bound: the candidate with the best actual nDCG@10 for that query."""
    out = {}
    for qid in qids:
        best = max(sl.CANDIDATES, key=lambda n: per_query[n][qid]["ndcg"])
        out[qid] = per_query[best][qid]
    return out


def importance(selector: sl.Selector, X: np.ndarray, targets: dict[str, np.ndarray]) -> dict:
    """Permutation importance, training rows only."""
    from sklearn.inspection import permutation_importance
    out = {}
    for name in sl.CANDIDATES:
        res = permutation_importance(selector.models[name], X, targets[name],
                                     n_repeats=5, random_state=CI_SEED, scoring="r2")
        order = np.argsort(-res.importances_mean)[:12]
        out[name] = {
            "top": [{"feature": sl.FEATURES[i], "importance": float(res.importances_mean[i]),
                     "std": float(res.importances_std[i])} for i in order],
            "all": {sl.FEATURES[i]: float(res.importances_mean[i]) for i in range(len(sl.FEATURES))},
        }
    return out


def choose_for(selector, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Selection for either selector kind: the classifier takes its argmax class, no margin."""
    if isinstance(selector, sl.ClassifierSelector):
        return selector.predict(X), None
    pred = selector.predict_matrix(X)
    return sl.choose(pred, selector.margin), pred


def evaluate_selector(data: dict, selector, margin_name: str) -> dict:
    qids = data["qids"]
    actual = actual_matrix(data["per_query"], qids)
    chosen, pred = choose_for(selector, data["X"])
    per_q = system_per_query(data["per_query"], qids, chosen)
    stats = lp.paired_bootstrap(data["per_query"]["bm25"], per_q, reps=RESAMPLES, seed=CI_SEED)
    actual_stack = np.column_stack([actual[n] for n in sl.CANDIDATES])
    pred_gain = None
    if pred is not None:
        pred_gain = float(np.mean(pred[np.arange(len(qids)), chosen] - pred[:, 0]))
    return {
        "per_query": per_q,
        "metrics": {
            "ndcg10": lp.mean_metric(per_q, "ndcg"),
            "r100": lp.mean_metric(per_q, "r100"),
            "mrr": lp.mean_metric(per_q, "mrr"),
            "delta_ndcg": stats["delta"],
            "ci95": stats["ci95"],
            "wtl": stats["wtl"],
        },
        "selection": sl.selection_diagnostics(chosen, actual),
        "choice_counts": {name: int((chosen == i).sum()) for i, name in enumerate(sl.CANDIDATES)},
        "pred_gain_mean": pred_gain,
        "true_best_counts": {name: int((np.argmax(actual_stack, axis=1) == i).sum())
                             for i, name in enumerate(sl.CANDIDATES)},
        "margin": getattr(selector, "margin", None),
        "label": margin_name,
    }


def run_test_dataset(name: str, split: str | None, limit: int | None,
                     selectors: dict[str, sl.Selector]) -> dict:
    data = build_dataset(name, split, limit)
    qids = data["qids"]
    per_query = data["per_query"]
    systems: dict[str, dict] = {}

    def add(system: str, per_q: dict) -> None:
        row = {
            "ndcg10": lp.mean_metric(per_q, "ndcg"),
            "r100": lp.mean_metric(per_q, "r100"),
            "mrr": lp.mean_metric(per_q, "mrr"),
        }
        if system == "bm25":
            row.update({"delta_ndcg": 0.0, "ci95": [0.0, 0.0], "wtl": [0, len(per_q), 0]})
        else:
            stats = lp.paired_bootstrap(per_query["bm25"], per_q, reps=RESAMPLES, seed=CI_SEED)
            row.update({"delta_ndcg": stats["delta"], "ci95": stats["ci95"], "wtl": stats["wtl"]})
        row["per_query"] = {q: {k: round(v, 6) for k, v in m.items()} for q, m in per_q.items()}
        systems[system] = row

    add("bm25", per_query["bm25"])
    add("always_rm3", per_query["rm3"])
    add("always_dph", per_query["dph"])
    add("oracle", oracle_per_query(per_query, qids))

    diagnostics = {}
    for label, selector in selectors.items():
        result = evaluate_selector(data, selector, label)
        add(label, result["per_query"])
        diagnostics[label] = {
            "margin": result["margin"],
            "selection": result["selection"],
            "choice_counts": result["choice_counts"],
            "pred_gain_mean": result["pred_gain_mean"],
            "true_best_counts": result["true_best_counts"],
        }

    log(f"\n    {'system':14} {'nDCG@10':>8} {'R@100':>7} {'MRR':>7} {'dNDCG':>8} {'95% CI':>19} {'W/T/L':>13}")
    for system in ("bm25", "always_rm3", "always_dph", *selectors, "oracle"):
        row = systems[system]
        d, lo, hi = row["delta_ndcg"], row["ci95"][0], row["ci95"][1]
        w, t, l = row["wtl"]
        log(f"    {system:14} {row['ndcg10']:8.4f} {row['r100']:7.4f} {row['mrr']:7.4f} "
            f"{d:+8.4f} [{lo:+.4f},{hi:+.4f}] {w:5}/{t:5}/{l:5}")
    for label, diag in diagnostics.items():
        parts = ", ".join(f"{k}={v['count']}" for k, v in diag["selection"].items())
        log(f"    {label:14} picks: {parts}")

    out = dict(data["meta"])
    out["systems"] = systems
    out["selector"] = diagnostics
    out["max_rss_gb"] = rss_gb()
    return out


def train(limit: int | None) -> tuple[sl.Selector, sl.Selector, sl.ClassifierSelector, dict]:
    log("\n########## training set construction ##########")
    Xs, target_rows, sources = [], [], []
    for name, split in TRAIN_SETS:
        data = build_dataset(name, split, limit)
        actual = actual_matrix(data["per_query"], data["qids"])
        Xs.append(data["X"])
        target_rows.append(actual)
        sources.append({"dataset": name, "split": split, "queries": len(data["qids"])})
        del data
        gc.collect()

    X = np.vstack(Xs)
    targets = {name: np.concatenate([rows[name] for rows in target_rows]) for name in sl.CANDIDATES}
    actual_stack = np.column_stack([targets[n] for n in sl.CANDIDATES])
    true_best = np.argmax(actual_stack, axis=1)
    log(f"\n    training matrix {X.shape[0]:,} queries x {X.shape[1]} features; "
        f"true best: " + ", ".join(f"{n}={int((true_best == i).sum())}"
                                   for i, n in enumerate(sl.CANDIDATES)))

    # fixed 80/20 sanity check; the final selector is refit on all training rows
    rng = np.random.default_rng(sl.SPLIT_SEED)
    order = rng.permutation(X.shape[0])
    cut = int(round((1.0 - sl.HOLDOUT_FRACTION) * X.shape[0]))
    tr, ho = order[:cut], order[cut:]
    check = sl.Selector().fit(X[tr], {n: targets[n][tr] for n in sl.CANDIDATES})
    pred = check.predict_matrix(X[ho])
    chosen = sl.choose(pred, sl.MARGIN)
    policy = actual_stack[ho][np.arange(ho.size), chosen]
    base = actual_stack[ho][:, 0]
    rmse = {n: float(np.sqrt(np.mean((pred[:, i] - targets[n][ho]) ** 2)))
            for i, n in enumerate(sl.CANDIDATES)}
    holdout = {
        "n_train": int(tr.size), "n_holdout": int(ho.size), "seed": sl.SPLIT_SEED,
        "policy_ndcg": float(policy.mean()), "bm25_ndcg": float(base.mean()),
        "policy_delta": float((policy - base).mean()),
        "argmax_accuracy": float((np.argmax(pred, axis=1) == true_best[ho]).mean()),
        "rmse": rmse,
        "choice_counts": {n: int((chosen == i).sum()) for i, n in enumerate(sl.CANDIDATES)},
    }
    log(f"    80/20 check: policy nDCG {holdout['policy_ndcg']:.4f} vs bm25 "
        f"{holdout['bm25_ndcg']:.4f} (delta {holdout['policy_delta']:+.4f}), "
        f"argmax acc {holdout['argmax_accuracy']:.3f}, rmse {rmse}")

    full = sl.Selector(margin=sl.MARGIN).fit(X, targets)
    safe = sl.Selector(margin=sl.MARGIN_SAFE).fit(X, targets)
    clf = sl.ClassifierSelector().fit(X, true_best)
    log("    permutation importance (training rows only) ...")
    imp = importance(full, X, targets)
    for name in sl.CANDIDATES:
        top = ", ".join(f"{d['feature']}={d['importance']:+.3f}" for d in imp[name]["top"][:6])
        log(f"    {name:6} importance: {top}")

    info = {
        "sources": sources,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "features": list(sl.FEATURES),
        "candidates": list(sl.CANDIDATES),
        "hgb_params": sl.HGB_PARAMS,
        "margins": {"selector": sl.MARGIN, "selector_safe": sl.MARGIN_SAFE},
        "classifier_params": sl.CLF_PARAMS,
        "true_best_counts": {n: int((true_best == i).sum()) for i, n in enumerate(sl.CANDIDATES)},
        "target_means": {n: float(targets[n].mean()) for n in sl.CANDIDATES},
        "holdout_check": holdout,
        "permutation_importance": imp,
    }
    return full, safe, clf, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[n for n, _ in TEST_SETS])
    ap.add_argument("--limit", type=int, default=None, help="queries per dataset (smoke only)")
    ap.add_argument("--out", default=str(RESULTS / "selective_lexical.json"))
    args = ap.parse_args()

    log("E2 selective lexical selector — frozen configuration")
    log(f"python {platform.python_version()} | numpy {np.__version__} | "
        f"host {platform.machine()} | data root {harness.DATA}")
    log(f"candidates={sl.CANDIDATES} margins=({sl.MARGIN}, {sl.MARGIN_SAFE}) "
        f"hgb={sl.HGB_PARAMS}")
    log(f"train={TRAIN_SETS}")

    t_all = time.perf_counter()
    report = {
        "experiment": "selective_lexical",
        "frozen": {
            "candidates": list(sl.CANDIDATES),
            "margin": sl.MARGIN,
            "margin_safe": sl.MARGIN_SAFE,
            "topk": sl.TOPK,
            "hgb_params": sl.HGB_PARAMS,
            "holdout_fraction": sl.HOLDOUT_FRACTION,
            "split_seed": sl.SPLIT_SEED,
            "resamples": RESAMPLES,
            "ci_seed": CI_SEED,
            "train_sets": [f"{n}/{s}" for n, s in TRAIN_SETS],
            "test_sets": [f"{n}/{s or 'test'}" for n, s in TEST_SETS],
            "e1_constants": {"k1": lp.K1, "b": lp.B, "rm3_lambda": lp.RM3_LAMBDA,
                             "rm3_terms": lp.RM3_TERMS, "rm3_docs": lp.RM3_DOCS,
                             "depth": lp.DEPTH},
        },
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "machine": platform.machine(), "platform": platform.platform(),
            "data_root": str(harness.DATA),
        },
        "training": {},
        "datasets": {},
    }

    selector, safe, clf, info = train(args.limit)
    report["training"] = info
    selectors = {"selector": selector, "selector_safe": safe, "selector_clf": clf}
    Path(args.out).write_text(json.dumps(report, indent=1))

    for name in args.datasets:
        report["datasets"][name] = run_test_dataset(name, TEST_SPLIT[name], args.limit, selectors)
        Path(args.out).write_text(json.dumps(report, indent=1))

    systems = ("bm25", "always_rm3", "always_dph", "selector", "selector_safe",
               "selector_clf", "oracle")
    macro = {}
    for system in systems:
        if not all(system in d["systems"] for d in report["datasets"].values()):
            continue
        macro[system] = {
            metric: float(np.mean([report["datasets"][n]["systems"][system][metric]
                                   for n in report["datasets"]]))
            for metric in ("ndcg10", "r100", "mrr", "delta_ndcg")
        }
    report["macro"] = macro

    # E1 reference row, copied from the frozen E1 result (not recomputed)
    if E1_RESULTS.exists():
        e1 = json.loads(E1_RESULTS.read_text())
        if all(n in e1.get("datasets", {}) for n in report["datasets"]):
            report["reference_e1"] = {
                "portfolio_naf": {
                    "per_dataset_ndcg10": {n: e1["datasets"][n]["systems"]["portfolio_naf"]["ndcg10"]
                                           for n in report["datasets"]},
                    "per_dataset_r100": {n: e1["datasets"][n]["systems"]["portfolio_naf"]["r100"]
                                         for n in report["datasets"]},
                    "macro_ndcg10": float(np.mean([
                        e1["datasets"][n]["systems"]["portfolio_naf"]["ndcg10"]
                        for n in report["datasets"]])),
                    "source": "experiments/stages/results/lexical_portfolio.json",
                }
            }

    full = set(report["datasets"]) == {n for n, _ in TEST_SETS}
    if full and "selector" in macro:
        report["gate"] = lp.gate(report["datasets"], "selector")
        report["gate_safe"] = lp.gate(report["datasets"], "selector_safe")
    else:
        report["gate"] = {"skipped": "partial run: gate requires all seven test sets"}

    report["runtime_s"] = time.perf_counter() - t_all
    report["max_rss_gb"] = rss_gb()
    Path(args.out).write_text(json.dumps(report, indent=1))

    print(f"\n{'macro-average':14} {'nDCG@10':>8} {'R@100':>7} {'MRR':>7} {'dNDCG':>8}")
    for system, row in macro.items():
        print(f"{system:14} {row['ndcg10']:8.4f} {row['r100']:7.4f} {row['mrr']:7.4f} "
              f"{row['delta_ndcg']:+8.4f}")
    g = report["gate"]
    if "pass" in g:
        print(f"\ngate (selector, margin {sl.MARGIN}): {'PASS' if g['pass'] else 'FAIL'}")
        print(f"  macro nDCG {g['macro_ndcg']:.4f} vs bm25 {g['macro_bm25_ndcg']:.4f} "
              f"({g['macro_delta']:+.4f}); positive {len(g['positive_datasets'])}/7; "
              f"failing {g['failing_datasets'] or 'none'}")
        print(f"  checks: {g['checks']}")
    else:
        print(f"\ngate: {g['skipped']}")
    print(f"\nwrote {args.out} in {report['runtime_s'] / 60:.1f} min, "
          f"max rss {report['max_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()
