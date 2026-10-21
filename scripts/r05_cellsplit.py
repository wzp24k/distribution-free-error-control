"""Exchangeable control experiment: calibration and test cells drawn from the
*same* set of fields of view.

The finite-sample guarantee of SelectiveRiskController needs the calibration and
deployment cells to be exchangeable.  Splitting by FOV -- which is what the
r02 units do -- deliberately breaks that, because FOVs differ in composition and
staining.  This script isolates the theory from that violation: the classifier
is still trained on other FOVs, but the calibration/test split is a random split
of the held-out cells, so the guarantee must hold with probability >= 1 - delta.

If coverage here is at the nominal level while the cross-FOV and cross-tissue
experiments are not, the gap is attributable to the loss of exchangeability
rather than to the method.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import data as D, paths
from errctl.classifiers import fit_classifier
from errctl.preprocess import DomainNormaliser

PRED_DIR = os.path.join(paths.WORK, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

N_FOLDS = 5
N_SPLITS = 10
CAL_FRAC = 0.5
MIN_CLASS_TRAIN = 3
MAX_TRAIN = 60000


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    ap.add_argument("--norm", default="z_dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    clfs = [c for c in args.clfs.split(",") if c]
    rows = []

    for ds in paths.MANUAL_DATASETS:
        g, markers = D.gold(ds, drop_unidentified=True)
        g = g.reset_index(drop=True)
        X = g[markers].to_numpy(dtype=np.float64)
        y_all = g["class_expert"].astype(str).to_numpy()
        rng0 = np.random.default_rng(args.seed)
        imgs = np.array(sorted(g["image"].unique()))
        rng0.shuffle(imgs)
        folds = np.array_split(imgs, N_FOLDS)

        for f in range(N_FOLDS):
            test_imgs = set(folds[f])
            tr = ~g["image"].isin(test_imgs).to_numpy()
            pool_idx = np.nonzero(~tr)[0]
            tr_idx = np.nonzero(tr)[0]
            if len(tr_idx) > MAX_TRAIN:
                tr_idx = np.random.default_rng(args.seed + f).choice(tr_idx, MAX_TRAIN,
                                                                     replace=False)
            present = pd.Series(y_all[tr_idx]).value_counts()
            classes = sorted(c for c in present.index if present[c] >= MIN_CLASS_TRAIN)
            pool_idx = np.array([i for i in pool_idx if y_all[i] in classes], dtype=int)
            tr_idx = np.array([i for i in tr_idx if y_all[i] in classes], dtype=int)
            if len(pool_idx) < 200:
                continue
            ix = {c: i for i, c in enumerate(classes)}
            ytr = np.array([ix[v] for v in y_all[tr_idx]])
            nz = DomainNormaliser(args.norm).fit("src", X[tr_idx])
            Xtr = nz.transform("src", X[tr_idx])
            Xpool = nz.transform("src", X[pool_idx])
            ypool = np.array([ix[v] for v in y_all[pool_idx]])

            for clf_kind in clfs:
                clf = fit_classifier(Xtr, ytr, kind=clf_kind, seed=0)
                probs = clf.predict_proba(Xpool).astype(np.float32)
                rng = np.random.default_rng(1000 + f)
                n = len(pool_idx)
                n_cal = int(n * CAL_FRAC)
                keep = np.zeros((n, N_SPLITS), dtype=bool)
                for r in range(N_SPLITS):
                    pick = rng.choice(n, n_cal, replace=False)
                    keep[pick, r] = True
                np.savez_compressed(
                    os.path.join(PRED_DIR, "cellsplit_%s_fold%d__%s.npz" % (ds, f, clf_kind)),
                    cal_probs=probs, cal_y=ypool, cal_img=g["image"].to_numpy()[pool_idx],
                    cal_rep=keep, test_rep=~keep,
                    cal_X=np.asarray(Xpool, dtype=np.float32),
                    test_probs=probs, test_y=ypool,
                    test_img=g["image"].to_numpy()[pool_idx],
                    test_cellid=g["cellID"].to_numpy()[pool_idx],
                    test_dataset=np.full(n, ds),
                    test_fold=np.full(n, f),
                    test_X=np.asarray(Xpool, dtype=np.float32),
                    classes=np.asarray(classes),
                )
                acc = float((clf.predict_index(Xpool) == ypool).mean())
                log("  cellsplit:%s:fold%d %-7s pool=%d cal=%d acc=%.3f"
                    % (ds, f, clf_kind, n, n_cal, acc))
                rows.append({"unit": "cellsplit:%s:fold%d" % (ds, f), "clf": clf_kind,
                             "pool": n, "cal": n_cal, "acc": acc})

    pd.DataFrame(rows).to_csv(os.path.join(paths.WORK, "r05_runs.csv"), index=False)
    log("wrote r05 (cellsplit) predictions")


if __name__ == "__main__":
    main()
