"""R55 -- the exchangeable, cell-level split family on independent cohorts.

Why this script exists
----------------------
The paper's guarantee is verified on the *random-split* family, in which the
held-out pool of a cohort is divided at random into calibration and evaluation
halves (``scripts/r05_cellsplit.py``).  Every external test reported so far uses
the *harder* split -- image-grouped folds (``r34``, family ``ext_within``) or
cross-cohort transfer -- so the external evidence contains counterexamples only
and the positively-reproducible half of the guarantee has never been
demonstrated outside the four benchmark cohorts.  That asymmetry is the
paper's weakest point, because it leaves the failure open to the reading "the
method only ever worked where the authors chose the split".

This script closes the gap.  It applies the identical certificate, split
fraction, repeat count and metric definitions (``scripts/r03_analyse.evaluate``,
imported, not re-implemented) to the STELLAR / Hickey CODEX cell tables
(Zenodo 6863872, CC0-1.0): a different laboratory (Stanford / Nolan), a
different platform (47-plex CODEX) and three different tissues from the
benchmark.  Two families are run.

* ``ext_cellsplit`` -- train on the cohort's other images, then a cell-level
  random 50/50 split of the held-out pool.  The direct analogue of the paper's
  ``cellsplit`` family, so the guarantee must hold here.
* ``ext_targetcal`` -- train on the intestine cohort, then a cell-level random
  50/50 split *within* the tonsil cohort and within the Barrett's-oesophagus
  cohort.  This is the deployment a laboratory actually faces: a new tissue
  arrives, half of its annotated cells are spent on calibration, and the other
  half is the batch that has to be annotated.

Deviations from r05, all recorded
---------------------------------
* the STELLAR tables ship per-channel values already on an arcsinh-like scale
  (range -0.79 to 96.6, not raw counts), so ``--norm`` selects the transform.
  ``log1p_z`` is the transform the r34 external branch already uses (and hence
  the like-for-like comparison), ``z_train`` standardises on the training
  subset without the log, and ``raw`` is the identity.  The pooled result is
  reported for the default and repeated for the other two.
* the pool of a unit is capped at 20000 cells, the cap r34 uses externally, and
  the training subsample is capped at 60000, the cap r02 uses throughout.

Outputs: ``experiments/r55_stellar_<norm>.csv`` (tidy, r03 schema plus
family/source/target), ``experiments/r55_stellar_<norm>_summary.csv``,
``experiments/r55_stellar_checks.csv`` and ``work/r55_report.txt``.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from errctl import paths                      # noqa: E402
from errctl.classifiers import fit_classifier  # noqa: E402
from errctl.preprocess import DomainNormaliser  # noqa: E402
import r03_analyse as A3                      # noqa: E402

EXT_DIR = os.path.join(ROOT, "data", "external")
CACHE = os.path.join(paths.WORK, "cache")
PRED_DIR = os.path.join(paths.WORK, "predictions_r55")

N_FOLDS = 5
N_SPLITS = 10
CAL_FRAC = 0.5
MIN_CLASS_TRAIN = 3
MAX_TRAIN = 60000
POOL_CAP = 20000
IMAGE_MIN_CELLS = 60
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
SEED = 0
CLFS = ("logreg", "rf", "hgb")

# name, file, image filter, image column, label column, meta columns
COHORTS = [
    ("STELLAR_Intestine_B004", "B004_training_dryad.csv", None,
     "unique_region", "cell_type_A",
     ("x", "y", "tissue", "donor", "unique_region", "cell_type_A")),
    ("STELLAR_Tonsil", "BE_Tonsil_l3_dryad.csv", "tonsil",
     "sample_name", "cell_type", ("x", "y", "sample_name", "cell_type")),
    ("STELLAR_BarrettsEsophagus", "BE_Tonsil_l3_dryad.csv",
     "Barretts Esophagus", "sample_name", "cell_type",
     ("x", "y", "sample_name", "cell_type")),
]
COHORT_BY_NAME = {c[0]: c for c in COHORTS}
SOURCE = "STELLAR_Intestine_B004"


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


# --------------------------------------------------------------- data loading
def cohort_frame(name):
    """Per-cell marker matrix, image and expert label for one external cohort."""
    _, fn, imfilter, imgc, labc, meta = COHORT_BY_NAME[name]
    df = pd.read_csv(os.path.join(EXT_DIR, fn), index_col=0, low_memory=False)
    meta = set(meta)
    markers = [c for c in df.columns if c not in meta]
    img = df[imgc].astype(str).to_numpy()
    lab = df[labc].astype(str).to_numpy()
    V = df[markers].to_numpy(dtype=np.float64)
    ok = np.isfinite(V).all(axis=1)
    ok &= (lab != "") & (lab != "nan")
    if imfilter is not None:
        ok &= (img == imfilter)
    counts = pd.Series(img[ok]).value_counts()
    good = sorted(counts[counts >= IMAGE_MIN_CELLS].index)
    ok &= np.isin(img, np.asarray(good, dtype=object))
    idx = np.nonzero(ok)[0]
    return {"name": name, "markers": markers, "X": V[idx],
            "y": lab[idx], "img": img[idx]}


def build_cache(force=False):
    os.makedirs(CACHE, exist_ok=True)
    for name, *_ in COHORTS:
        p = os.path.join(CACHE, "r55_%s.npz" % name)
        if os.path.exists(p) and not force:
            continue
        c = cohort_frame(name)
        classes = np.asarray(sorted(set(c["y"].tolist())), dtype=object)
        ycode = np.searchsorted(classes, c["y"])
        np.savez_compressed(
            p, X=np.asarray(c["X"], dtype=np.float32), y=ycode,
            cls=classes.astype("U48"),
            img=np.asarray(c["img"], dtype="U40"),
            markers=np.asarray(c["markers"], dtype="U40"))
        log("cache %-28s %7d cells  %2d images  %2d classes  %2d markers"
            % (name, len(ycode), len(set(c["img"])), len(classes),
               len(c["markers"])))


def load_cohort(name):
    d = np.load(os.path.join(CACHE, "r55_%s.npz" % name), allow_pickle=False)
    return {"X": d["X"].astype(np.float64), "y": d["y"].astype(int),
            "cls": d["cls"].astype(str), "img": d["img"].astype(str),
            "markers": d["markers"].astype(str)}


# ------------------------------------------------------------------- features
def transform(mode, Xtr, others):
    if mode == "raw":
        return [np.asarray(X, dtype=np.float64) for X in others]
    if mode == "log1p_z":
        nz = DomainNormaliser("z_dataset").fit("s", Xtr)
        return [nz.transform("s", X) for X in others]
    if mode == "z_train":
        Xtr = np.asarray(Xtr, dtype=np.float64)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        return [(np.asarray(X, dtype=np.float64) - mu) / sd for X in others]
    raise ValueError("unknown norm %r" % mode)


# ------------------------------------------------------------------ one unit
def unit_rows(family, unit, fold, clf_kind, Xtr, ytr, Xpool, ypool, cids, imgs,
              mode, source, target, save_npz=True):
    """One (unit, classifier): fit, then every method for every repeat x alpha."""
    Xtr_t, Xpool_t = transform(mode, Xtr, [Xtr, Xpool])
    clf = fit_classifier(Xtr_t, ytr, kind=clf_kind, seed=0)
    probs = clf.predict_proba(Xpool_t).astype(np.float32)
    n = len(ypool)
    n_cal = int(n * CAL_FRAC)
    rng = np.random.default_rng(1000 + (fold if fold is not None else 0))
    keep = np.zeros((n, N_SPLITS), dtype=bool)
    for r in range(N_SPLITS):
        keep[rng.choice(n, n_cal, replace=False), r] = True
    if save_npz:
        os.makedirs(PRED_DIR, exist_ok=True)
        # ':' is the NTFS alternate-data-stream separator, so the unit tag is
        # sanitised before it is used as a file name.
        stem = unit.replace(":", "_")
        np.savez_compressed(
            os.path.join(PRED_DIR, "%s__%s.npz" % (stem, clf_kind)),
            cal_probs=probs, cal_y=ypool, cal_rep=keep, test_rep=~keep,
            cal_X=np.asarray(Xpool_t, dtype=np.float32),
            test_X=np.asarray(Xpool_t, dtype=np.float32),
            test_probs=probs, test_y=ypool, test_cellid=cids, test_img=imgs,
            test_fold=np.full(n, -1 if fold is None else fold),
            test_dataset=np.full(n, target or source),
            classes=np.asarray(sorted(set(int(v) for v in ypool))))
    acc = float((clf.predict_index(Xpool_t) == ypool).mean())
    rows = []
    for r in range(N_SPLITS):
        sel = keep[:, r]
        s_cal = 1.0 - probs[sel].max(1)
        p_cal = probs[sel].argmax(1)
        e_cal = (p_cal != ypool[sel]).astype(float)
        te = ~sel
        s_te = 1.0 - probs[te].max(1)
        p_te = probs[te].argmax(1)
        e_te = (p_te != ypool[te]).astype(float)
        for alpha in ALPHAS:
            res = dict(A3.evaluate(s_cal, e_cal, s_te, e_te, p_cal, p_te,
                                   alpha, DELTA))
            acc_mask, q, sizes = A3.conformal_set_acceptance(
                probs[sel], ypool[sel], probs[te], alpha)
            n_acc = int(acc_mask.sum())
            risk = float(e_te[acc_mask].mean()) if n_acc else np.nan
            res["conformal_set"] = {
                "method": "conformal_set", "n_accepted": n_acc,
                "accept_rate": n_acc / max(len(acc_mask), 1),
                "selective_risk": risk,
                "covered": int((not np.isfinite(risk)) or risk <= alpha),
                "empty": int(n_acc == 0),
                "n_wrong_accepted": int(e_te[acc_mask].sum()) if n_acc else 0,
                "lambda": q, "mean_set_size": float(sizes.mean())}
            cp, tp, T = A3.temp_scale(probs[sel], ypool[sel], probs[te])
            s_cal_t = 1.0 - cp.max(1)
            e_cal_t = (cp.argmax(1) != ypool[sel]).astype(float)
            s_te_t = 1.0 - tp.max(1)
            lam_t = A3.C.naive_threshold(s_cal_t, e_cal_t, alpha)
            acc_t = s_te_t <= lam_t
            n_t = int(acc_t.sum())
            res["temp_naive"] = {
                "method": "temp_naive", "n_accepted": n_t,
                "accept_rate": n_t / max(len(acc_t), 1),
                "selective_risk": float(e_te[acc_t].mean()) if n_t else np.nan,
                "covered": int(n_t == 0 or e_te[acc_t].mean() <= alpha),
                "empty": int(n_t == 0),
                "n_wrong_accepted": int(e_te[acc_t].sum()),
                "lambda": lam_t, "temperature": T}
            ctl_t = A3.C.SelectiveRiskController(alpha, DELTA).fit(s_cal_t, e_cal_t)
            acc_t2 = ctl_t.accept(s_te_t)
            n_t2 = int(acc_t2.sum())
            res["crc_temp"] = {
                "method": "crc_temp", "n_accepted": n_t2,
                "accept_rate": n_t2 / max(len(acc_t2), 1),
                "selective_risk": float(e_te[acc_t2].mean()) if n_t2 else np.nan,
                "covered": int(n_t2 == 0 or e_te[acc_t2].mean() <= alpha),
                "empty": int(n_t2 == 0),
                "n_wrong_accepted": int(e_te[acc_t2].sum()),
                "lambda": ctl_t.lambda_hat, "temperature": T}
            for name, rr in res.items():
                rows.append({
                    "family": family, "source": source, "target": target,
                    "unit": unit, "clf": clf_kind, "kind": "cellsplit",
                    "repeat": r, "alpha": alpha, "norm": mode,
                    "n_cal_pool": int(len(ypool)), "n_cal_used": int(sel.sum()),
                    "n_test_pool": int(len(ypool)), "n_test_eval": int(te.sum()),
                    "acc_clf": acc, **rr})
    log("  %-46s %-7s pool=%6d cal=%6d acc=%.3f"
        % (unit, clf_kind, n, n_cal, acc))
    return rows


# ------------------------------------------------------------------- families
def family_specs(clfs=CLFS):
    """(family, source, target, unit tag, fold) one row per unit to run."""
    specs = []
    for f in range(N_FOLDS):
        for clf in clfs:
            specs.append(("ext_cellsplit", SOURCE, None,
                          "ext_cellsplit:%s:fold%d" % (SOURCE, f), f, clf))
    for tgt in ("STELLAR_Tonsil", "STELLAR_BarrettsEsophagus"):
        for clf in clfs:
            specs.append(("ext_targetcal", SOURCE, tgt,
                          "ext_targetcal:%s" % tgt, None, clf))
    return specs


def cellsplit_unit(fold, mode):
    """Train on the cohort's other images; pool = held-out images' cells."""
    c = load_cohort(SOURCE)
    X, y, img = c["X"], c["y"], c["img"]
    imgs = np.asarray(sorted(set(img.tolist())), dtype=object)
    rng0 = np.random.default_rng(SEED)
    rng0.shuffle(imgs)
    folds = np.array_split(imgs, N_FOLDS)
    held = np.asarray(sorted(folds[fold].tolist()), dtype=object)
    tr = ~np.isin(img, held)
    tr_idx = np.nonzero(tr)[0]
    pool_idx = np.nonzero(~tr)[0]
    if len(tr_idx) > MAX_TRAIN:
        tr_idx = np.random.default_rng(SEED + fold).choice(
            tr_idx, MAX_TRAIN, replace=False)
    present = pd.Series(y[tr_idx]).value_counts()
    keep_cls = set(int(k) for k in present.index
                   if int(present[k]) >= MIN_CLASS_TRAIN)
    tr_idx = np.asarray([i for i in tr_idx if int(y[i]) in keep_cls], dtype=int)
    pool_idx = np.asarray([i for i in pool_idx if int(y[i]) in keep_cls],
                          dtype=int)
    if len(pool_idx) > POOL_CAP:
        pool_idx = np.sort(np.random.default_rng(SEED + 100 + fold).choice(
            pool_idx, POOL_CAP, replace=False))
    order = sorted(keep_cls)
    ix = {k: i for i, k in enumerate(order)}
    return (X[tr_idx], np.asarray([ix[int(v)] for v in y[tr_idx]]),
            X[pool_idx], np.asarray([ix[int(v)] for v in y[pool_idx]]),
            np.arange(1, len(pool_idx) + 1), img[pool_idx])


def targetcal_unit(target, mode):
    """Train on the intestine cohort; pool = the target cohort's own cells."""
    src, tgt = load_cohort(SOURCE), load_cohort(target)
    # The two deposited files do not carry the same channel list, so the source
    # and the target are aligned on their shared panel by name first; without
    # this the feature matrices have different widths and cannot be scored.
    sm = {m: i for i, m in enumerate(src["markers"])}
    tm = {m: i for i, m in enumerate(tgt["markers"])}
    panel = sorted(set(src["markers"]) & set(tgt["markers"]))
    sX = src["X"][:, [sm[m] for m in panel]]
    tX = tgt["X"][:, [tm[m] for m in panel]]
    sy = src["cls"][src["y"]]
    ty = tgt["cls"][tgt["y"]]
    s_idx = np.arange(len(sy))
    if len(s_idx) > MAX_TRAIN:
        s_idx = np.random.default_rng(SEED).choice(s_idx, MAX_TRAIN,
                                                   replace=False)
    present = pd.Series(sy[s_idx]).value_counts()
    shared = sorted(set(c for c in present.index
                        if int(present[c]) >= MIN_CLASS_TRAIN) & set(ty.tolist()))
    if len(shared) < 2:
        raise RuntimeError("%s shares no class with %s" % (target, SOURCE))
    sy_ok = np.isin(sy, np.asarray(shared, dtype=object))
    ty_ok = np.isin(ty, np.asarray(shared, dtype=object))
    s_idx = np.asarray([i for i in s_idx if sy_ok[i]], dtype=int)
    pool_idx = np.nonzero(ty_ok)[0]
    if len(pool_idx) > POOL_CAP:
        pool_idx = np.sort(np.random.default_rng(SEED + 7).choice(
            pool_idx, POOL_CAP, replace=False))
    ix = {c: i for i, c in enumerate(shared)}
    log("  panel %s <-> %s: %d shared channels, %d shared classes"
        % (SOURCE, target, len(panel), len(shared)))
    return (sX[s_idx], np.asarray([ix[v] for v in sy[s_idx]]),
            tX[pool_idx], np.asarray([ix[v] for v in ty[pool_idx]]),
            np.arange(1, len(pool_idx) + 1), tgt["img"][pool_idx])


def run_mode(mode, which, clfs=CLFS, save_npz=True):
    rows = []
    for fam, src, tgt, unit, fold, clf in family_specs(clfs):
        if which and fam not in which:
            continue
        if fam == "ext_cellsplit":
            Xtr, ytr, Xp, yp, cids, imgs = cellsplit_unit(fold, mode)
        else:
            Xtr, ytr, Xp, yp, cids, imgs = targetcal_unit(tgt, mode)
        rows += unit_rows(fam, unit, fold, clf, Xtr, ytr, Xp, yp, cids, imgs,
                          mode, src or "", tgt or "", save_npz=save_npz)
    return rows


# ------------------------------------------------------------------- summary
def boot_ci_unit(df, B=2000, seed=0):
    """Cluster bootstrap over units of a per-evaluation 0/1 indicator."""
    units = sorted(df["unit"].unique())
    if len(units) < 2:
        m = float(df["covered"].mean())
        return m, m, m
    rng = np.random.default_rng(seed)
    vals = {u: df.loc[df["unit"] == u, "covered"].to_numpy(float)
            for u in units}
    draws = []
    for _ in range(B):
        pick = rng.integers(0, len(units), len(units))
        cat = np.concatenate([vals[units[i]] for i in pick])
        draws.append(cat.mean())
    return (float(df["covered"].mean()), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def summarise(df, mode):
    out = []
    for (fam, alpha), g in df.groupby(["family", "alpha"], sort=True):
        c = g[g["method"] == "crc"]
        nonvac = c[~c["empty"].astype(bool)]
        cov, lo, hi = boot_ci_unit(c)
        out.append({
            "norm": mode, "family": fam, "alpha": alpha,
            "method": "crc", "n_eval": int(len(c)),
            "n_units": int(c["unit"].nunique()),
            "n_classifiers": int(c["clf"].nunique()),
            "mean_accept": float(c["accept_rate"].mean()),
            "review_frac": float(1.0 - c["accept_rate"].mean()),
            "frac_nonvacuous": float(1.0 - c["empty"].mean()),
            "mean_accept_nonvac": (float(nonvac["accept_rate"].mean())
                                   if len(nonvac) else np.nan),
            "risk_nonvac": (float(nonvac["selective_risk"].mean())
                            if len(nonvac) else np.nan),
            "coverage": cov, "coverage_lo": lo, "coverage_hi": hi,
            "coverage_nonvac": (float(nonvac["covered"].mean())
                                if len(nonvac) else np.nan),
            "n_over_budget": int((~c["covered"].astype(bool)).sum()),
            "cells_tested": int(c["n_test_eval"].sum()),
            "cells_accepted": int(c["n_accepted"].sum())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="log1p_z",
                    choices=["log1p_z", "z_train", "raw"])
    ap.add_argument("--only", default="",
                    help="comma list of families to run")
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    ap.add_argument("--tag", default="",
                    help="suffix for the output basename; defaults to --norm")
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args()
    which = [s for s in args.only.split(",") if s]
    clfs = tuple(c for c in args.clfs.split(",") if c)
    tag = args.tag or args.norm
    t0 = time.time()
    build_cache(force=args.force_cache)
    rows = run_mode(args.norm, which, clfs=clfs)
    df = pd.DataFrame(rows)
    os.makedirs(paths.RESULTS, exist_ok=True)
    df.to_csv(os.path.join(paths.RESULTS, "r55_stellar_%s.csv" % tag),
              index=False)
    summ = summarise(df, args.norm)
    summ.to_csv(os.path.join(
        paths.RESULTS, "r55_stellar_%s_summary.csv" % tag), index=False)
    print(summ.to_string(index=False))
    log("runtime %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
