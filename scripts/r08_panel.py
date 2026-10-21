"""R8 -- how much panel does the guarantee need?

The depot's panels run from 22 to 58 channels, but most published imaging
panels are far smaller. This ablation refits the annotation model on reduced
panels -- markers are dropped in order of a training-set F-score, from the full
panel down to a quarter of it -- and reports what the reduction costs in plain
accuracy, in certified acceptance rate at a fixed error budget, and in the
observed selective risk of the accepted cells.

Evaluation splits the held-out FOVs' cells into two random halves (calibration
and deployment), so the exchangeability the guarantee needs is preserved and the
only thing that changes is the panel.

Output: experiments/r08_panel.csv
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, data as D, paths
from errctl.classifiers import fit_classifier
from errctl.preprocess import DomainNormaliser

CACHE = os.path.join(paths.CACHE, "r08")
os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(paths.RESULTS, "r08_panel.csv")

FRACTIONS = (1.0, 0.75, 0.5, 0.25)
CLFS = ("logreg", "hgb")
ALPHAS = (0.10, 0.20)
N_FOLDS = 5
N_SPLITS = 5
MAX_TRAIN = 20000
MIN_CLASS_TRAIN = 3


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)), flush=True)


def prepare(ds):
    """Cache (X, y, image, fold) for one cohort so workers stay cheap."""
    path = os.path.join(CACHE, "%s.npz" % ds)
    if os.path.exists(path):
        return path
    g, markers = D.gold(ds, drop_unidentified=True)
    g = g.reset_index(drop=True)
    X = g[markers].to_numpy(dtype=np.float64)
    y = g["coarse_expert"].astype(str).to_numpy()
    imgs = g["image"].astype(str).to_numpy()
    uniq = np.array(sorted(set(imgs)))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    fold_of = {}
    for i, part in enumerate(np.array_split(uniq, N_FOLDS)):
        fold_of.update({u: i for u in part})
    fold = np.array([fold_of[u] for u in imgs])
    np.savez_compressed(path, X=X.astype(np.float32), y=y, image=imgs, fold=fold,
                        markers=np.asarray(markers))
    return path


def run_job(job):
    ds, fold, frac, kind = job
    d = np.load(os.path.join(CACHE, "%s.npz" % ds), allow_pickle=True)
    X, y, fo, markers = d["X"].astype(np.float64), d["y"], d["fold"], list(d["markers"])
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

    # rank markers on the training split only, then keep the top fraction
    F, _ = f_classif(X[tr], np.array([classes.index(v) for v in y[tr]]))
    F = np.nan_to_num(F)
    keep = np.argsort(-F)[:max(3, int(round(frac * X.shape[1])))]
    keep = np.sort(keep)
    cols = [markers[i] for i in keep]

    ix = {c: i for i, c in enumerate(classes)}
    ytr = np.array([ix[v] for v in y[tr]])
    nz = DomainNormaliser("z_dataset").fit("src", X[tr][:, keep])
    Xtr = nz.transform("src", X[tr][:, keep])
    Xpool = nz.transform("src", X[pool][:, keep])
    ypool = np.array([ix[v] for v in y[pool]])

    clf = fit_classifier(Xtr, ytr, kind=kind, seed=0)
    probs = clf.predict_proba(Xpool).astype(np.float64)
    pred = probs.argmax(1)
    acc = float((pred == ypool).mean())

    rng = np.random.default_rng(1000 + fold)
    n = len(pool)
    half = n // 2
    rows = []
    for rep in range(N_SPLITS):
        pick = rng.permutation(n)
        cal, te = pick[:half], pick[half:]
        s_cal = 1.0 - probs[cal].max(1)
        e_cal = (pred[cal] != ypool[cal]).astype(float)
        s_te = 1.0 - probs[te].max(1)
        e_te = (pred[te] != ypool[te]).astype(float)
        for alpha in ALPHAS:
            acc_m = C.SelectiveRiskController(alpha, 0.10).fit(s_cal, e_cal).accept(s_te)
            n_acc = int(acc_m.sum())
            risk = float(e_te[acc_m].mean()) if n_acc else np.nan
            rows.append({"dataset": ds, "fold": fold, "clf": kind, "frac": frac,
                         "n_markers": len(keep), "n_train": len(tr), "n_pool": n,
                         "acc": acc, "alpha": alpha, "rep": rep,
                         "accept_rate": n_acc / max(len(te), 1),
                         "selective_risk": risk,
                         "covered": int((not np.isfinite(risk)) or risk <= alpha),
                         "empty": int(n_acc == 0)})
    return rows


def _job(j):
    return run_job(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", default="")
    args = ap.parse_args()

    datasets = [d for d in paths.MANUAL_DATASETS if not args.limit or args.limit in d]
    for ds in datasets:
        log("caching", ds)
        prepare(ds)
    jobs = [(ds, f, fr, k) for ds in datasets for f in range(N_FOLDS)
            for fr in FRACTIONS for k in CLFS]
    log("%d refits" % len(jobs))

    rows = []
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            for i, res in enumerate(pool.imap(_job, jobs, chunksize=1), 1):
                rows.extend(res)
                if i % 20 == 0:
                    log("  %d/%d" % (i, len(jobs)))
    else:
        for i, j in enumerate(jobs, 1):
            rows.extend(run_job(j))
            if i % 20 == 0:
                log("  %d/%d" % (i, len(jobs)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    log("wrote %s (%d rows)" % (OUT, len(df)))
    t = (df.groupby(["clf", "frac", "alpha"])
           .agg(n_markers=("n_markers", "mean"), acc=("acc", "mean"),
                accept=("accept_rate", "mean"), risk=("selective_risk", "mean"),
                coverage=("covered", "mean")).round(3))
    log(t.to_string())


if __name__ == "__main__":
    main()


