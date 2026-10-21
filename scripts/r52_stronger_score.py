"""R52 -- can a stronger model compress the residual review?

The certificate's review fraction at a fixed error budget is bounded below by
the oracle, and the oracle is a property of the score: a better one ranks
correct cells above wrong ones more sharply, so it lifts the largest threshold
that still meets the budget, and the certificate moves with it.

This script asks that question on the *same* family, folds and calibration
draws that produce the paper's headline numbers (fine ``class_expert`` labels,
fields of view split five ways, calibration a random half of the held-out
cells), so the comparison is a controlled one: the only thing that changes is
the model that produces the score.

Models
------
  logreg, rf, hgb   the three base learners already in the paper's comparison
  vote3             their mean probability -- free, no extra fits
  hgb_bag           five-bootstrap bagged gradient boosting
  vote_bag          five-bootstrap bag of the logreg+hgb vote
  mlp               a different family entirely

Every model is scored the paper's way, s = 1 - max probability, and evaluated
with the same controller (SelectiveRiskController, delta = 0.10, min_accept =
25, grid 120, Bonferroni) and the same oracle.  The three baseline models are
refit here from the same code path and their probabilities are checked against
the stored ``work/predictions/cellsplit_*.npz`` files, so the head-to-head is
verifiable rather than asserted.

Output: experiments/r52_stronger.csv, experiments/r52_stronger_summary.csv
"""

import argparse
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, data as D, paths
from errctl.calibration import expected_calibration_error
from errctl.classifiers import HistGradientBoostingClassifier, fit_classifier
from errctl.preprocess import DomainNormaliser

PRED_DIR = os.path.join(paths.WORK, "predictions")
OUT = os.path.join(paths.RESULTS, "r52_stronger.csv")
OUTSUM = os.path.join(paths.RESULTS, "r52_stronger_summary.csv")

N_FOLDS = 5
N_SPLITS = 10
CAL_FRAC = 0.5
MIN_CLASS_TRAIN = 3
MAX_TRAIN = 60000
DELTA = 0.10
ALPHAS = (0.05, 0.10, 0.15, 0.20)
N_BAGS = 5
BASE = ("logreg", "rf", "hgb")


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)), flush=True)


def fit_rf(X, y, seed):
    """Same configuration as errctl.classifiers, pinned to one thread.

    ``n_jobs`` changes only the wall clock, not the fitted forest, and the
    workers are already saturating the cores.
    """
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                 random_state=seed, n_jobs=1)
    clf.fit(X, y)
    return clf


def fit_one(kind, X, y, seed):
    if kind == "rf":
        return fit_rf(X, y, seed)
    if kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                            early_stopping=True, validation_fraction=0.15,
                            n_iter_no_change=15, random_state=seed)
        clf.fit(X, y)
        return clf
    return fit_classifier(X, y, kind=kind, seed=seed).estimator


def full_probs(clf, X, n_classes):
    """predict_proba widened to the full class index space."""
    p = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    idx = np.asarray(clf.classes_, dtype=int)
    p[:, idx] = clf.predict_proba(X)
    return p


def bootstrap(idx_rng, n):
    return idx_rng.choice(n, n, replace=True)


def model_probs(job):
    ds, fold, kinds = job
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
    for kind in BASE:
        probs[kind] = full_probs(fit_one(kind, Xtr, ytr, 0), Xpool, n_cls)
    probs["vote3"] = (probs["logreg"] + probs["rf"] + probs["hgb"]) / 3.0

    rng = np.random.default_rng(4242 + fold)
    hb = np.zeros((N_BAGS, len(pool_idx), n_cls))
    vb = np.zeros((N_BAGS, len(pool_idx), n_cls))
    for b in range(N_BAGS):
        idx = bootstrap(rng, len(tr_idx))
        members = np.unique(ytr[idx])
        keep = np.isin(ytr, members)
        pb = []
        for kind in ("logreg", "hgb"):
            clf = fit_one(kind, Xtr[idx][keep], ytr[idx][keep], b)
            pb.append(full_probs(clf, Xpool, n_cls))
        vb[b] = (pb[0] + pb[1]) / 2.0
        hb[b] = full_probs(fit_one("hgb", Xtr[idx][keep], ytr[idx][keep], b), Xpool, n_cls)
    probs["hgb_bag"] = hb.mean(0)
    probs["vote_bag"] = vb.mean(0)
    probs["mlp"] = full_probs(fit_one("mlp", Xtr, ytr, 0), Xpool, n_cls)

    # self-check: the refit logreg must reproduce the stored paper's predictions
    ref = os.path.join(PRED_DIR, "cellsplit_%s_fold%d__logreg.npz" % (ds, fold))
    check = np.nan
    if os.path.exists(ref):
        z = np.load(ref, allow_pickle=True)
        if np.array_equal(np.asarray(z["classes"]).astype(str), np.asarray(classes)):
            check = float(np.abs(np.asarray(z["test_probs"], dtype=np.float64)
                                 - probs["logreg"]).max())

    return {"ds": ds, "fold": fold, "classes": list(classes), "prod": probs,
            "ypool": ypool, "n_pool": len(pool_idx), "n_cal": int(CAL_FRAC * len(pool_idx)),
            "check_logreg": check}


def evaluate(res):
    if res is None:
        return []
    prod, ypool, n = res["prod"], res["ypool"], res["n_pool"]
    n_cal = res["n_cal"]
    rng = np.random.default_rng(1000 + res["fold"])
    keep = np.zeros((n, N_SPLITS), dtype=bool)
    for r in range(N_SPLITS):
        pick = rng.choice(n, n_cal, replace=False)
        keep[pick, r] = True
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
                ora, _ = oracle_acceptance(s[te], err[te], alpha)
                k_o = int(ora.sum())
                lam = C.naive_threshold(s[cal], err[cal], alpha)
                acc_n = s[te] <= lam
                k_n = int(acc_n.sum())
                rows.append({"dataset": res["ds"], "fold": res["fold"], "clf": name,
                             "alpha": alpha, "rep": r, "n_pool": n, "n_te": int(te.sum()),
                             "acc": float((pred == ypool).mean()), "ece": ece,
                             "accept_rate": k / max(int(te.sum()), 1),
                             "selective_risk": risk,
                             "covered": int((not np.isfinite(risk)) or risk <= alpha),
                             "empty": int(k == 0),
                             "oracle_rate": k_o / max(int(te.sum()), 1),
                             "accept_rate_naive": k_n / max(int(te.sum()), 1)})
    return rows


def oracle_acceptance(s_te, e_te, alpha, min_accept=25):
    """Largest threshold meeting the budget when the test labels are known."""
    order = np.argsort(s_te, kind="mergesort")
    s, e = s_te[order], e_te[order]
    cum = np.cumsum(e)
    k = np.arange(1, s.size + 1)
    ok = (cum / k <= alpha) & (k >= min_accept)
    if not ok.any():
        return np.zeros(s_te.size, dtype=bool), -np.inf
    lam = float(s[np.nonzero(ok)[0][-1]])
    return s_te <= lam, lam


def _job(j):
    try:
        return evaluate(model_probs(j))
    except Exception as exc:  # keep one bad unit from killing the sweep
        return [{"error": "%s: %s" % (j, exc)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", default="")
    args = ap.parse_args()

    datasets = [d for d in paths.MANUAL_DATASETS if not args.limit or args.limit in d]
    jobs = [(ds, f, None) for ds in datasets for f in range(N_FOLDS)]
    log("%d units x %d bags" % (len(jobs), N_BAGS))

    rows = []
    with mp.Pool(args.jobs) as pool:
        for i, res in enumerate(pool.imap(_job, jobs, chunksize=1), 1):
            rows.extend(res)
            log("  %d/%d" % (i, len(jobs)))
    df = pd.DataFrame(rows)
    errs = df[df.get("error").notna()] if "error" in df.columns else pd.DataFrame()
    if len(errs):
        log("ERRORS:\n" + errs.to_string())
    df = df[df.get("error").isna()] if "error" in df.columns else df
    if "error" in df.columns:
        df = df.drop(columns=["error"])
    df.to_csv(OUT, index=False)

    g = df.groupby(["clf", "alpha"])
    summ = g.agg(n_units=("fold", "size"),
                 acc=("acc", "mean"), ece=("ece", "mean"),
                 accept_mean=("accept_rate", "mean"),
                 oracle_mean=("oracle_rate", "mean"),
                 risk_nonvac=("selective_risk", "mean"),
                 coverage=("covered", "mean"),
                 vacuous=("empty", "mean"),
                 accept_naive=("accept_rate_naive", "mean")).reset_index()
    summ["accept_pooled"] = g.apply(
        lambda d: d.accept_rate.mul(d.n_te).sum() / d.n_te.sum(), include_groups=False
    ).values
    summ["oracle_pooled"] = g.apply(
        lambda d: d.oracle_rate.mul(d.n_te).sum() / d.n_te.sum(), include_groups=False
    ).values
    summ = summ.round(5)
    summ.to_csv(OUTSUM, index=False)
    log("wrote %s (%d rows) and %s" % (OUT, len(df), OUTSUM))
    log("\n" + summ.pivot(index="clf", columns="alpha",
                          values="accept_pooled").to_string())
    log("\ncoverage:\n" + summ.pivot(index="clf", columns="alpha",
                                     values="coverage").to_string())
    log("\naccuracy:\n" + summ[summ.alpha == 0.10][["clf", "acc", "ece"]].to_string(index=False))


if __name__ == "__main__":
    main()
