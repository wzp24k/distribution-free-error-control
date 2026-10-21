"""R54 -- can a stronger ranker lower the oracle?

The certificate's acceptance is capped by the oracle, and the oracle is the
largest acceptance the score can support at the budget. R52 showed the ranking is
insensitive to the model inside the paper's own family; this script asks the
question with the strongest tabular learners available on the same per-cell
marker features, on the same folds, calibration draws, controller and oracle.

Everything here is a per-cell function of the already-log1p-transformed marker
intensities, so the exchangeability the guarantee needs is untouched: no
neighbour, composition or other cross-cell feature is used, because such a
feature would make the accepted-and-wrong count non-binomial and void the bound
as stated.

LightGBM is deliberately absent: its C API raises an access violation in
``LGBM_DatasetSetField`` inside spawned worker processes on this platform
(a Windows OpenMP/DLL interaction, reproducible outside this script), so
the sweep keeps the strongest learners that fit reliably -- a widened
histogram gradient boosting, a large random forest, extra trees and
xgboost -- and takes the vote over three of them.

Output: experiments/r54_ranking.csv, experiments/r54_ranking_summary.csv
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, data as D, paths
from errctl.calibration import expected_calibration_error
from errctl.classifiers import fit_classifier
from errctl.preprocess import DomainNormaliser

OUT = os.path.join(paths.RESULTS, "r54_ranking.csv")
OUTSUM = os.path.join(paths.RESULTS, "r54_ranking_summary.csv")

N_FOLDS = 5
N_SPLITS = 10
CAL_FRAC = 0.5
MIN_CLASS_TRAIN = 3
MAX_TRAIN = 60000
DELTA = 0.10
ALPHAS = (0.05, 0.10, 0.20)
MODELS = ("hgb", "hgb_x", "rf_x", "et", "xgb", "vote")


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)), flush=True)


def make(kind, seed=0):
    if kind == "hgb":
        return fit_classifier(None, None, kind="hgb", seed=seed) or None
    raise ValueError(kind)


def fit_model(kind, X, y, seed=0):
    if kind == "hgb_x":
        clf = HistGradientBoostingClassifier(
            max_iter=1200, learning_rate=0.03, max_leaf_nodes=63,
            min_samples_leaf=10, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=30, random_state=seed)
    elif kind == "rf_x":
        clf = RandomForestClassifier(n_estimators=1200, min_samples_leaf=1,
                                     max_features="sqrt", random_state=seed, n_jobs=1)
    elif kind == "et":
        clf = ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=1,
                                   max_features="sqrt", random_state=seed, n_jobs=1)
    elif kind == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=8,
                            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                            random_state=seed, n_jobs=1, verbosity=0)
    else:
        clf = fit_classifier(X, y, kind=kind, seed=seed).estimator
        return clf
    clf.fit(X, y)
    return clf


def widened(clf, X, n_cls):
    p = np.zeros((X.shape[0], n_cls), dtype=np.float64)
    p[:, np.asarray(clf.classes_, dtype=int)] = clf.predict_proba(X)
    return p


def model_probs(job):
    ds, fold = job
    g, markers = D.gold(ds, drop_unidentified=True)
    g = g.reset_index(drop=True)
    X = g[markers].to_numpy(dtype=np.float64)
    y_all = g["class_expert"].astype(str).to_numpy()
    rng0 = np.random.default_rng(0)
    imgs = np.array(sorted(g["image"].unique()))
    rng0.shuffle(imgs)
    folds = np.array_split(imgs, N_FOLDS)
    test_imgs = set(folds[fold])
    tr = ~g["image"].isin(test_imgs).to_numpy()
    pool_idx = np.nonzero(~tr)[0]
    tr_idx = np.nonzero(tr)[0]
    if len(tr_idx) > MAX_TRAIN:
        tr_idx = np.random.default_rng(fold).choice(tr_idx, MAX_TRAIN, replace=False)
    present = pd.Series(y_all[tr_idx]).value_counts()
    classes = sorted(c for c in present.index if present[c] >= MIN_CLASS_TRAIN)
    pool_idx = np.array([i for i in pool_idx if y_all[i] in classes], dtype=int)
    tr_idx = np.array([i for i in tr_idx if y_all[i] in classes], dtype=int)
    if len(pool_idx) < 200:
        return None
    ix = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)
    ytr = np.array([ix[v] for v in y_all[tr_idx]])
    nz = DomainNormaliser("z_dataset").fit("src", X[tr_idx])
    Xtr = nz.transform("src", X[tr_idx])
    Xpool = nz.transform("src", X[pool_idx])
    ypool = np.array([ix[v] for v in y_all[pool_idx]])

    probs = {}
    for kind in ("hgb", "hgb_x", "rf_x", "et", "xgb"):
        probs[kind] = widened(fit_model(kind, Xtr, ytr, 0), Xpool, n_cls)
    probs["vote"] = (probs["hgb_x"] + probs["rf_x"] + probs["xgb"]) / 3.0
    return {"ds": ds, "fold": fold, "prod": probs, "ypool": ypool, "n_pool": len(pool_idx)}


def oracle_acceptance(s_te, e_te, alpha, min_accept=25):
    order = np.argsort(s_te, kind="mergesort")
    s, e = s_te[order], e_te[order]
    cum = np.cumsum(e)
    k = np.arange(1, s.size + 1)
    ok = (cum / k <= alpha) & (k >= min_accept)
    if not ok.any():
        return np.zeros(s_te.size, dtype=bool)
    lam = float(s[np.nonzero(ok)[0][-1]])
    return s_te <= lam


def evaluate(res):
    if res is None:
        return []
    prod, ypool, n = res["prod"], res["ypool"], res["n_pool"]
    rng = np.random.default_rng(1000 + res["fold"])
    keep = np.zeros((n, N_SPLITS), dtype=bool)
    for r in range(N_SPLITS):
        keep[rng.choice(n, int(CAL_FRAC * n), replace=False), r] = True
    rows = []
    for name, p in prod.items():
        pred = p.argmax(1)
        err = (pred != ypool).astype(float)
        s = 1.0 - p.max(1)
        ece = expected_calibration_error(p.max(1), 1.0 - err)
        for r in range(N_SPLITS):
            cal, te = keep[:, r], ~keep[:, r]
            for alpha in ALPHAS:
                ctl = C.SelectiveRiskController(alpha, DELTA).fit(s[cal], err[cal])
                acc = ctl.accept(s[te])
                k = int(acc.sum())
                risk = float(err[te][acc].mean()) if k else np.nan
                ora = oracle_acceptance(s[te], err[te], alpha)
                rows.append({"dataset": res["ds"], "fold": res["fold"], "clf": name,
                             "alpha": alpha, "rep": r, "n_pool": n, "n_te": int(te.sum()),
                             "acc": float((pred == ypool).mean()), "ece": ece,
                             "accept_rate": k / max(int(te.sum()), 1),
                             "selective_risk": risk,
                             "covered": int((not np.isfinite(risk)) or risk <= alpha),
                             "empty": int(k == 0),
                             "oracle_rate": int(ora.sum()) / max(int(te.sum()), 1)})
    return rows


def _job(j):
    try:
        return evaluate(model_probs(j))
    except Exception as exc:
        return [{"error": "%s: %r" % (j, exc)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", default="")
    args = ap.parse_args()
    datasets = [d for d in paths.MANUAL_DATASETS if not args.limit or args.limit in d]
    jobs = [(ds, f) for ds in datasets for f in range(N_FOLDS)]
    log("%d units x %d models" % (len(jobs), len(MODELS)))
    rows = []
    with mp.Pool(args.jobs) as pool:
        for i, res in enumerate(pool.imap(_job, jobs, chunksize=1), 1):
            rows.extend(res)
            log("  %d/%d" % (i, len(jobs)))
    df = pd.DataFrame(rows)
    err = df[df.get("error").notna()] if "error" in df.columns else pd.DataFrame()
    if len(err):
        log("ERRORS:\n" + err.to_string())
        df = df[df.get("error").isna()]
    if "error" in df.columns:
        df = df.drop(columns=["error"])
    df.to_csv(OUT, index=False)

    g = df.groupby(["clf", "alpha"])
    summ = g.agg(n_units=("fold", "size"), acc=("acc", "mean"), ece=("ece", "mean"),
                 accept_mean=("accept_rate", "mean"), oracle_mean=("oracle_rate", "mean"),
                 risk_nonvac=("selective_risk", "mean"), coverage=("covered", "mean"),
                 vacuous=("empty", "mean")).reset_index()
    summ["accept_pooled"] = g.apply(
        lambda d: d.accept_rate.mul(d.n_te).sum() / d.n_te.sum(), include_groups=False).values
    summ["oracle_pooled"] = g.apply(
        lambda d: d.oracle_rate.mul(d.n_te).sum() / d.n_te.sum(), include_groups=False).values
    summ["price_pp"] = (summ.oracle_mean - summ.accept_mean) * 100.0
    summ = summ.round(5)
    summ.to_csv(OUTSUM, index=False)
    log("wrote %s (%d rows) and %s" % (OUT, len(df), OUTSUM))
    log("\nacceptance (mean over units):\n"
        + summ.pivot(index="clf", columns="alpha", values="accept_mean").round(3).to_string())
    log("\noracle (mean over units):\n"
        + summ.pivot(index="clf", columns="alpha", values="oracle_mean").round(3).to_string())
    log("\naccuracy and ECE:\n"
        + summ[summ.alpha == 0.10][["clf", "acc", "ece"]].round(4).to_string(index=False))
    log("\ncoverage at alpha=0.10:\n"
        + summ[np.isclose(summ.alpha, 0.10)][["clf", "coverage", "vacuous"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
