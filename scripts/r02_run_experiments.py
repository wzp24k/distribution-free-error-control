"""Train annotation models and store calibration/test predictions.

Design
------
Every experimental unit is a (train, calibration-pool, test) split of *cells*
built from a split of *field-of-view images*, so no image contributes cells to
more than one role. The model is trained once per unit/fold; the calibration
repeats then vary only which cells of the calibration pool are used, which is
exactly the randomness that the conformal guarantee is about.

Families
--------
* ``within:<cohort>``       train / calibrate / test inside one cohort.
* ``transfer:<src>-><tgt>`` train and calibrate on the source cohort, test on
  the target cohort, restricted to the shared (canonicalised) marker panel and
  the shared coarse label set. Features are standardised per domain, which is a
  label-free operation.

Outputs (``work/predictions/<unit>__<clf>.npz``)
-----------------------------------------------
cal_probs, cal_y, cal_img, cal_rep (n_cal x n_repeats inclusion mask),
test_probs, test_y, test_img, test_cellid, test_fold, classes,
cal_X, test_X (normalised features, needed for covariate-shift weighting).
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
from errctl.markerlogic import canonical
from errctl.preprocess import DomainNormaliser

PRED_DIR = os.path.join(paths.WORK, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

WITHIN = paths.MANUAL_DATASETS
TRANSFER_PAIRS = [
    ("GVHD_Feb1323", "CRC_Feb2623"),
    ("CRC_Feb2623", "GVHD_Feb1323"),
    ("GVHD_Feb1323", "Breast_Jun0524"),
    ("GVHD_Feb1323", "Melanoma_Oct2522"),
    ("CRC_Feb2623", "Breast_Jun0524"),
    ("CRC_Feb2623", "Melanoma_Oct2522"),
    ("Breast_Jun0524", "Melanoma_Oct2522"),
    ("Melanoma_Oct2522", "Breast_Jun0524"),
    ("Breast_Jun0524", "CRC_Feb2623"),
    ("Melanoma_Oct2522", "CRC_Feb2623"),
    ("Breast_Jun0524", "GVHD_Feb1323"),
    ("Melanoma_Oct2522", "GVHD_Feb1323"),
]

MIN_CLASS_TRAIN = 3
MIN_CLASS_TEST = 20
MAX_TRAIN = 60000
N_FOLDS = 5
N_CAL_REPEATS = 6
CAL_SUBSAMPLE = 2500


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def fold_split(images, n_folds, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.array(sorted(images))
    rng.shuffle(imgs)
    return np.array_split(imgs, n_folds)


def subsample_training(idx, y, cap, seed=0):
    if len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y[idx]):
        pool = idx[y[idx] == cls]
        take = max(3, int(round(cap * len(pool) / len(idx))))
        keep.append(rng.choice(pool, size=min(take, len(pool)), replace=False))
    return np.concatenate(keep)


def shared_panel(src_markers, tgt_markers):
    src = {canonical(m): m for m in src_markers}
    tgt = {canonical(m): m for m in tgt_markers}
    return [(k, src[k], tgt[k]) for k in sorted(set(src) & set(tgt))]


def encode(labels, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx.get(c, -1) for c in labels], dtype=int)


def run_unit(name, test_rows, test_fold, y_train_lab, y_cal_lab, y_test_lab,
             X_train, X_cal, X_test, images_cal, images_test,
             clf_kind, n_cal_repeats, out_dir, seed=0):
    keep_cls = pd.Series(y_train_lab).value_counts()
    classes = sorted(c for c in keep_cls.index if keep_cls[c] >= MIN_CLASS_TRAIN)
    tr_keep = np.isin(y_train_lab, classes)
    X_train, y_train_lab = X_train[tr_keep], np.asarray(y_train_lab)[tr_keep]
    y_train = encode(y_train_lab, classes)
    y_cal = encode(y_cal_lab, classes)
    y_test = encode(y_test_lab, classes)

    clf = fit_classifier(X_train, y_train, kind=clf_kind, seed=seed)
    if list(clf.classes_) != list(range(len(classes))):
        raise RuntimeError("class order mismatch: %r" % (clf.classes_,))
    cal_probs = clf.predict_proba(X_cal).astype(np.float32)
    test_probs = clf.predict_proba(X_test).astype(np.float32)

    rng = np.random.default_rng(12345)
    n_cal = len(X_cal)
    keep = np.zeros((n_cal, n_cal_repeats), dtype=bool)
    for r in range(n_cal_repeats):
        pick = rng.choice(n_cal, size=min(CAL_SUBSAMPLE, n_cal), replace=False)
        keep[pick, r] = True

    stub = name.replace(":", "_").replace("->", "_to_")
    np.savez_compressed(
        os.path.join(out_dir, "%s__%s.npz" % (stub, clf_kind)),
        cal_probs=cal_probs, cal_y=y_cal, cal_img=np.asarray(images_cal),
        cal_rep=keep, cal_X=np.asarray(X_cal, dtype=np.float32),
        test_probs=test_probs, test_y=y_test, test_img=np.asarray(images_test),
        test_cellid=np.asarray(test_rows["cellID"]),
        test_dataset=np.asarray(test_rows["dataset"], dtype=str),
        test_fold=np.asarray(test_fold),
        test_X=np.asarray(X_test, dtype=np.float32),
        classes=np.asarray(classes),
    )
    acc_cal = float((clf.predict_index(X_cal) == y_cal).mean())
    acc_test = float((clf.predict_index(X_test) == y_test).mean())
    log("   %-40s %-7s cal=%.3f test=%.3f  n=%d/%d/%d  K=%d  unseen_test=%d"
        % (name, clf_kind, acc_cal, acc_test, len(X_train), len(X_cal), len(X_test),
           len(classes), int((y_test < 0).sum())))
    return {"unit": name, "clf": clf_kind, "cal_acc": acc_cal,
            "test_acc": acc_test, "n_train": len(X_train), "n_cal": len(X_cal),
            "n_test": len(X_test), "n_classes": len(classes),
            "n_unseen_test": int((y_test < 0).sum())}


def prepare_cohort(ds):
    g, markers = D.gold(ds, drop_unidentified=True)
    g = g.reset_index(drop=True)
    X = g[markers].to_numpy(dtype=np.float64)
    return g, markers, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="within,transfer")
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    ap.add_argument("--only", default="")
    ap.add_argument("--norm", default="z_dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clfs = [c for c in args.clfs.split(",") if c]
    rows = []
    cached = {}

    def cohort(ds):
        if ds not in cached:
            cached[ds] = prepare_cohort(ds)
            log("loaded %s: %d cells, %d markers" % (ds, len(cached[ds][0]), len(cached[ds][1])))
        return cached[ds]

    if "within" in args.units:
        for ds in WITHIN:
            if args.only and ds not in args.only:
                continue
            g, markers, X = cohort(ds)
            y_all = g["class_expert"].astype(str).to_numpy()
            imgs = g["image"].to_numpy()
            folds = fold_split(g["image"].unique(), N_FOLDS, seed=args.seed)
            for f in range(N_FOLDS):
                test_imgs, cal_imgs = set(folds[f]), set(folds[(f + 1) % N_FOLDS])
                tr = ~g["image"].isin(test_imgs | cal_imgs).to_numpy()
                ca = g["image"].isin(cal_imgs).to_numpy()
                te = g["image"].isin(test_imgs).to_numpy()
                tr_idx = subsample_training(np.nonzero(tr)[0], y_all, MAX_TRAIN,
                                            seed=args.seed + f)
                ca_idx, te_idx = np.nonzero(ca)[0], np.nonzero(te)[0]
                if len(ca_idx) < 50 or len(te_idx) < 50:
                    continue
                nz = DomainNormaliser(args.norm).fit("src", X[tr_idx])
                Xtr = nz.transform("src", X[tr_idx])
                Xca = nz.transform("src", X[ca_idx])
                Xte = nz.transform("src", X[te_idx])
                for clf_kind in clfs:
                    rows.append(run_unit(
                        "within:%s:fold%d" % (ds, f), g.iloc[te_idx],
                        np.full(len(te_idx), f), y_all[tr_idx], y_all[ca_idx],
                        y_all[te_idx], Xtr, Xca, Xte, imgs[ca_idx], imgs[te_idx],
                        clf_kind, N_CAL_REPEATS, PRED_DIR))

    if "transfer" in args.units:
        for src, tgt in TRANSFER_PAIRS:
            if args.only and ("%s->%s" % (src, tgt)) not in args.only:
                continue
            gs, ms, Xs = cohort(src)
            gt, mt, Xt = cohort(tgt)
            panel = shared_panel(ms, mt)
            if len(panel) < 5:
                log("   skip %s->%s: only %d shared markers" % (src, tgt, len(panel)))
                continue
            src_cols = [p[1] for p in panel]
            tgt_cols = [p[2] for p in panel]
            Xs_p = gs[src_cols].to_numpy(dtype=np.float64)
            Xt_p = gt[tgt_cols].to_numpy(dtype=np.float64)

            counts_t = gt["coarse_expert"].value_counts()
            counts_s = gs["coarse_expert"].value_counts()
            shared = sorted(c for c in counts_t.index
                            if counts_t[c] >= MIN_CLASS_TEST and counts_s.get(c, 0) >= 30)
            if len(shared) < 3:
                log("   skip %s->%s: only %d shared coarse classes" % (src, tgt, len(shared)))
                continue

            sm = gs["coarse_expert"].isin(shared).to_numpy()
            tm = gt["coarse_expert"].isin(shared).to_numpy()
            gs2, Xs2 = gs[sm].reset_index(drop=True), Xs_p[sm]
            gt2, Xt2 = gt[tm].reset_index(drop=True), Xt_p[tm]
            ys = gs2["coarse_expert"].astype(str).to_numpy()
            yt = gt2["coarse_expert"].astype(str).to_numpy()

            folds = fold_split(gs2["image"].unique(), N_FOLDS, seed=args.seed + 7)
            for f in range(N_FOLDS):
                test_imgs, cal_imgs = set(folds[f]), set(folds[(f + 1) % N_FOLDS])
                tr = ~gs2["image"].isin(test_imgs | cal_imgs).to_numpy()
                ca = gs2["image"].isin(cal_imgs).to_numpy()
                tr_idx = subsample_training(np.nonzero(tr)[0], ys, MAX_TRAIN,
                                            seed=args.seed + f)
                ca_idx = np.nonzero(ca)[0]
                if len(ca_idx) < 50:
                    continue
                nz = DomainNormaliser(args.norm)
                nz.fit("src", Xs2[tr_idx])
                nz.fit("tgt", Xt2)          # label-free
                Xtr = nz.transform("src", Xs2[tr_idx])
                Xca = nz.transform("src", Xs2[ca_idx])
                Xte = nz.transform("tgt", Xt2)
                for clf_kind in clfs:
                    rows.append(run_unit(
                        "transfer:%s->%s:fold%d" % (src, tgt, f), gt2,
                        np.full(len(gt2), f), ys[tr_idx], ys[ca_idx], yt,
                        Xtr, Xca, Xte, gs2["image"].to_numpy()[ca_idx],
                        gt2["image"].to_numpy(), clf_kind, N_CAL_REPEATS, PRED_DIR))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(paths.WORK, "r02_runs.csv"), index=False)
    log("wrote r02_runs.csv (%d rows)" % len(df))


if __name__ == "__main__":
    main()
