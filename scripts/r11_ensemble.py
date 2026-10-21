"""R11 -- does an ensemble confidence already give the guarantee?

An ensemble is the usual answer to "the softmax is overconfident": average
several models and trust their disagreement. This script fits bagged ensembles
of the same two base learners, forms the three confidence scores an ensemble
offers -- max probability, predictive entropy, and the mutual information
between members -- and asks whether any of them controls the error rate among
accepted cells. The comparison is against the same controller used everywhere
else, so the only thing that changes is the score.

Evaluation uses the exchangeable within-FOV split, where the controller *must*
work; a score that fails there cannot be rescued by calibration.

Output: experiments/r11_ensemble.csv
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, data as D, paths
from errctl.calibration import expected_calibration_error
from errctl.classifiers import fit_classifier
from errctl.preprocess import DomainNormaliser

OUT = os.path.join(paths.RESULTS, "r11_ensemble.csv")
CACHE = os.path.join(paths.CACHE, "r08")

N_FOLDS = 5
N_SPLITS = 5
N_BAGS = 5
MAX_TRAIN = 20000
MIN_CLASS_TRAIN = 3
ALPHAS = (0.10, 0.20)


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)), flush=True)


def bag(i, n, rng):
    return rng.choice(n, n, replace=True) if i else np.arange(n)


def run_job(job):
    ds, fold, kind = job
    d = np.load(os.path.join(CACHE, "%s.npz" % ds), allow_pickle=True)
    X, y, fo = d["X"].astype(np.float64), d["y"], d["fold"]
    tr = np.nonzero(fo != fold)[0]
    pool = np.nonzero(fo == fold)[0]
    counts = pd.Series(y[tr]).value_counts()
    classes = sorted(c for c in counts.index if counts[c] >= MIN_CLASS_TRAIN)
    tr = np.array([i for i in tr if y[i] in classes])
    pool = np.array([i for i in pool if y[i] in classes])
    if len(pool) < 200 or len(tr) < 200:
        return []
    if len(tr) > MAX_TRAIN:
        tr = np.random.default_rng(fold).choice(tr, MAX_TRAIN, replace=False)
    ix = {c: i for i, c in enumerate(classes)}
    ytr = np.array([ix[v] for v in y[tr]])
    nz = DomainNormaliser("z_dataset").fit("src", X[tr])
    Xtr = nz.transform("src", X[tr])
    Xpool = nz.transform("src", X[pool])
    ypool = np.array([ix[v] for v in y[pool]])

    rng = np.random.default_rng(4242 + fold)
    probs = np.zeros((N_BAGS, len(pool), len(classes)))
    for b in range(N_BAGS):
        idx = bag(b, len(tr), rng)
        while len(np.unique(ytr[idx])) < 2:
            idx = bag(b, len(tr), rng)
        members = np.unique(ytr[idx])
        keep = np.isin(ytr, members)
        clf = fit_classifier(Xtr[idx][keep], ytr[idx][keep], kind=kind, seed=b)
        p = np.zeros((len(pool), len(classes)))
        p[:, members] = clf.predict_proba(Xpool)
        probs[b] = p
    # renormalise: members that a bootstrap left out carry zero mass
    probs = probs / np.clip(probs.sum(-1, keepdims=True), 1e-12, None)
    mean_p = probs.mean(0)
    eps = 1e-12
    ent_mean = -(mean_p * np.log(np.clip(mean_p, eps, 1))).sum(1)
    ent_each = -(probs * np.log(np.clip(probs, eps, 1))).sum(-1).mean(0)
    scores = {"ens_maxprob": 1.0 - mean_p.max(1),
              "ens_entropy": ent_mean / np.log(len(classes)),
              "ens_mi": (ent_mean - ent_each) / np.log(len(classes))}
    pred = mean_p.argmax(1)
    err = (pred != ypool).astype(float)
    ece = expected_calibration_error(mean_p.max(1), 1 - err)

    rows = []
    for rep in range(N_SPLITS):
        pick = rng.permutation(len(pool))
        cal, te = pick[:len(pool) // 2], pick[len(pool) // 2:]
        for name, s in scores.items():
            s_cal, s_te = s[cal], s[te]
            for alpha in ALPHAS:
                ctl = C.SelectiveRiskController(alpha, 0.10).fit(s_cal, err[cal])
                acc = ctl.accept(s_te)
                n = int(acc.sum())
                risk = float(err[te][acc].mean()) if n else np.nan
                lam = C.naive_threshold(s_cal, err[cal], alpha)
                acc_n = s_te <= lam
                n_n = int(acc_n.sum())
                risk_n = float(err[te][acc_n].mean()) if n_n else np.nan
                rows.append({"dataset": ds, "fold": fold, "clf": kind, "score": name,
                             "alpha": alpha, "rep": rep, "ece": ece,
                             "accept_rate": n / max(len(te), 1),
                             "selective_risk": risk,
                             "covered": int((not np.isfinite(risk)) or risk <= alpha),
                             "accept_rate_naive": n_n / max(len(te), 1),
                             "risk_naive": risk_n,
                             "covered_naive": int((not np.isfinite(risk_n)) or risk_n <= alpha)})
    return rows


def _job(j):
    return run_job(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--kinds", default="logreg,hgb")
    args = ap.parse_args()
    from errctl.classifiers import CLASSIFIERS  # noqa: F401  (import check)

    for ds in paths.MANUAL_DATASETS:
        p = os.path.join(CACHE, "%s.npz" % ds)
        if not os.path.exists(p):
            log("cache missing for", ds, "- run scripts/r08_panel.py first")
            return
    jobs = [(ds, f, k) for ds in paths.MANUAL_DATASETS
            for f in range(N_FOLDS) for k in args.kinds.split(",")]
    log("%d ensemble jobs x %d bags" % (len(jobs), N_BAGS))

    rows = []
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            for i, res in enumerate(pool.imap(_job, jobs, chunksize=1), 1):
                rows.extend(res)
                if i % 5 == 0:
                    log("  %d/%d" % (i, len(jobs)))
    else:
        for i, j in enumerate(jobs, 1):
            rows.extend(run_job(j))
            if i % 5 == 0:
                log("  %d/%d" % (i, len(jobs)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    log("wrote %s (%d rows)" % (OUT, len(df)))
    t = (df.groupby(["clf", "score", "alpha"])
           .agg(ece=("ece", "mean"),
                accept=("accept_rate", "mean"), risk=("selective_risk", "mean"),
                coverage=("covered", "mean"),
                accept_naive=("accept_rate_naive", "mean"),
                coverage_naive=("covered_naive", "mean")).round(3))
    log(t.to_string())


if __name__ == "__main__":
    main()

