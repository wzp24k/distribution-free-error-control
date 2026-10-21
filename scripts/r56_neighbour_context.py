"""R56 -- does a richer representation lower the floor?  No.

The paper's own reading of r52 and r54 is that the review fraction is a property
of the *features* rather than of the learner: seven models on one side and six
strong tabular rankers on the other span a few points of oracle at a fixed
budget, and the best of them is the learner the paper already had.  That reading
leaves one obvious objection open, and it is the objection a referee makes first:
every model in the paper reads a *single cell's* marker vector, so perhaps the
floor is a missing-context artefact rather than a property of the assay, and
adding the tissue context every pathologist would use would move it.

This script tests that directly.  It augments each cell with the mean normalised
marker vector of its ``k`` nearest neighbours *within the same image*, taken from
a KD-tree over the cell coordinates, together with the logarithm of its mean
neighbour distance.  Coordinates and marker intensities only: no label, no
predicted class and no count of neighbour types enters the construction, so
nothing about the target can leak in.  The augmented arm is then run through the
identical certificate, oracle, folds, pool, calibration draws and metric
definitions as the baseline (``scripts/r03_analyse.evaluate``, imported rather
than re-implemented), exactly as the benchmark's random-split family is built:

* the folds are the benchmark's own -- images shuffled under seed 0 and split
  five ways, the model trained on the cohort's other images;
* the pool is the held-out cells of the held-out images, capped at 20,000;
* the split is a random 50/50 calibration/evaluation half, ten repeats;
* the score is ``1 - max_c proba`` and the certificate is
  ``errctl.conformal.SelectiveRiskController(alpha, 0.10)``.

The baseline arm reproduces r52's and r54's tuned-learner row exactly
(acceptance 0.60944, 0.85069 and 0.40588 at the 10, 20 and 5 percent budgets,
oracle 0.79220 at 10 percent), which is what makes the comparison like for like.

Result: the neighbourhood does not lower the floor.  Pooled over the four
cohorts and 190 paired evaluations per budget, the oracle moves by -0.0094 at a
5% budget, -0.0137 at 10% and +0.0018 at 20%, and accuracy moves by -0.0056; the
certificate's delivered acceptance is flat (+0.0015 at a 10% budget), so the
review fraction changes by 0.15 percentage points.  Context that the per-cell
marker mean does not already carry is not what sets the achievable acceptance on
these data.

Usage
-----
    python scripts/r56_neighbour_context.py --cohort Breast_Jun0524
    ... (once per cohort)
    python scripts/r56_neighbour_context.py --analyse
"""

import argparse
import glob
import io
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from errctl import data as D, paths                 # noqa: E402
from errctl.classifiers import fit_classifier       # noqa: E402
from errctl.preprocess import DomainNormaliser      # noqa: E402
import r03_analyse as A3                            # noqa: E402

N_FOLDS = 5
N_SPLITS = 10
CAL_FRAC = 0.5
MIN_CLASS_TRAIN = 3
MAX_TRAIN = 60000
POOL_CAP = 20000
K = 10
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
SEED = 0
ARMS = ("own", "own+neighbour")


def log(*p):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in p)),
          flush=True)


def neighbour_features(Z, img, xy, k=K):
    """Mean normalised marker vector of the k nearest neighbours, per cell.

    Neighbours are taken inside one image only and the cell itself is dropped.
    Only coordinates and marker intensities are read, so no label information
    reaches the features.
    """
    n, _ = Z.shape
    nb = np.zeros_like(Z)
    dist = np.zeros(n)
    for u in np.unique(img):
        sel = np.nonzero(img == u)[0]
        if sel.size < 2:
            continue
        tree = cKDTree(xy[sel])
        kk = min(k + 1, sel.size)
        dd, ii = tree.query(xy[sel], k=kk)
        dd, ii = np.atleast_2d(dd), np.atleast_2d(ii)
        nb[sel] = Z[sel][ii[:, 1:]].mean(axis=1)
        dist[sel] = dd[:, 1:].mean(axis=1)
    return nb, dist


def run_cohort(ds):
    g, markers = D.gold(ds, drop_unidentified=True)
    g = g.reset_index(drop=True)
    X = g[markers].to_numpy(dtype=np.float64)
    y_all = g["class_expert"].astype(str).to_numpy()
    img_all = g["image"].astype(str).to_numpy()
    xy_all = g[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    imgs = np.array(sorted(set(img_all.tolist())), dtype=object)
    np.random.default_rng(SEED).shuffle(imgs)
    folds = np.array_split(imgs, N_FOLDS)
    rows = []
    for f in range(N_FOLDS):
        held = np.asarray(sorted(folds[f].tolist()), dtype=object)
        inheld = np.isin(img_all, held)
        tr_idx = np.nonzero(~inheld)[0]
        pool_idx = np.nonzero(inheld)[0]
        if len(tr_idx) > MAX_TRAIN:
            tr_idx = np.random.default_rng(SEED + f).choice(
                tr_idx, MAX_TRAIN, replace=False)
        present = pd.Series(y_all[tr_idx]).value_counts()
        cls = sorted(c for c in present.index if present[c] >= MIN_CLASS_TRAIN)
        tr_idx = np.asarray([i for i in tr_idx if y_all[i] in set(cls)], int)
        pool_idx = np.asarray([i for i in pool_idx if y_all[i] in set(cls)], int)
        if len(pool_idx) < 200:
            continue
        if len(pool_idx) > POOL_CAP:
            pool_idx = np.sort(np.random.default_rng(SEED + 100 + f).choice(
                pool_idx, POOL_CAP, replace=False))
        ix = {c: i for i, c in enumerate(cls)}
        ytr = np.asarray([ix[v] for v in y_all[tr_idx]])
        ypool = np.asarray([ix[v] for v in y_all[pool_idx]])
        nz = DomainNormaliser("z_dataset").fit("src", X[tr_idx])
        Ztr = nz.transform("src", X[tr_idx])
        Zpo = nz.transform("src", X[pool_idx])
        nbtr, dtr = neighbour_features(Ztr, img_all[tr_idx], xy_all[tr_idx])
        nbpo, dpo = neighbour_features(Zpo, img_all[pool_idx], xy_all[pool_idx])
        feats = {"own": (Ztr, Zpo),
                 "own+neighbour": (np.column_stack([Ztr, nbtr, np.log1p(dtr)]),
                                   np.column_stack([Zpo, nbpo, np.log1p(dpo)]))}
        n = len(ypool)
        n_cal = int(n * CAL_FRAC)
        rng = np.random.default_rng(1000 + f)
        keep = np.zeros((n, N_SPLITS), dtype=bool)
        for r in range(N_SPLITS):
            keep[rng.choice(n, n_cal, replace=False), r] = True
        for arm in ARMS:
            Ftr, Fpo = feats[arm]
            clf = fit_classifier(Ftr, ytr, kind="hgb", seed=0)
            probs = clf.predict_proba(Fpo).astype(np.float32)
            acc = float((clf.predict_index(Fpo) == ypool).mean())
            for r in range(N_SPLITS):
                sel = keep[:, r]
                s_cal = 1.0 - probs[sel].max(1)
                e_cal = (probs[sel].argmax(1) != ypool[sel]).astype(float)
                te = ~sel
                s_te = 1.0 - probs[te].max(1)
                p_te = probs[te].argmax(1)
                e_te = (p_te != ypool[te]).astype(float)
                for a in ALPHAS:
                    res = A3.evaluate(s_cal, e_cal, s_te, e_te,
                                      probs[sel].argmax(1), p_te, a, DELTA)
                    for meth in ("crc", "oracle", "naive"):
                        rr = res[meth]
                        rows.append({
                            "cohort": ds, "fold": f, "arm": arm, "rep": r,
                            "alpha": a, "method": meth,
                            "n_features": int(Ftr.shape[1]), "acc": acc,
                            "n_pool": n, "n_cal": n_cal,
                            "accept_rate": rr["accept_rate"],
                            "selective_risk": rr["selective_risk"],
                            "n_accepted": rr["n_accepted"],
                            "covered": rr["covered"]})
            log("%s fold%d %-14s nfeat=%3d pool=%6d acc=%.4f"
                % (ds, f, arm, Ftr.shape[1], n, acc))
    out = os.path.join(paths.RESULTS, "r56_neighbour_%s.csv" % ds)
    pd.DataFrame(rows).to_csv(out, index=False)
    log("wrote", out, len(rows), "rows")


def analyse():
    files = sorted(glob.glob(os.path.join(paths.RESULTS, "r56_neighbour_*.csv")))
    files = [f for f in files if "summary" not in f and "checks" not in f]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(paths.RESULTS, "r56_neighbour.csv"), index=False)
    lines = ["=" * 78,
             "R56 -- spatial-neighbourhood features through the same certificate",
             "=" * 78, ""]
    lines.append("cohorts %d  rows %d  arms %s  k=%d"
                 % (df.cohort.nunique(), len(df), ", ".join(ARMS), K))
    lines.append("features per arm: %s" % df.groupby("arm").n_features.min().to_dict())
    lines.append("")
    summ = []
    for a in ALPHAS:
        g = df[np.isclose(df.alpha, a)]
        for arm in ARMS:
            o = g[(g.arm == arm) & (g.method == "oracle")]
            c = g[(g.arm == arm) & (g.method == "crc")]
            summ.append({
                "alpha": a, "arm": arm,
                "oracle": float(o.accept_rate.mean()),
                "accept": float(c.accept_rate.mean()),
                "review_frac": float(1 - c.accept_rate.mean()),
                "coverage": float(c.covered.mean()),
                "risk": float(c.selective_risk.mean()),
                "acc": float(g[g.arm == arm].acc.mean()),
                "n_eval": int(len(c))})
    summ = pd.DataFrame(summ)
    lines.append(summ.to_string(index=False))
    lines.append("")
    piv = df[df.method == "oracle"].pivot_table(
        index=["cohort", "fold", "rep", "alpha"], columns="arm",
        values="accept_rate")
    piv["diff"] = piv["own+neighbour"] - piv["own"]
    lines.append("paired oracle difference (neighbour minus own), over units:")
    lines.append(piv.groupby("alpha")["diff"].agg(
        ["mean", "std", "count"]).round(5).to_string())
    lines.append("")
    checks = []

    def ck(name, got, want, tol=5e-4):
        ok = abs(float(got) - float(want)) <= tol
        checks.append({"check": name, "value": got, "want": want,
                       "PASS": int(ok)})
        lines.append("%-5s %-56s got=%s want=%s"
                     % ("PASS" if ok else "FAIL", name, round(float(got), 6),
                        want))

    def S(arm, a, col):
        return float(summ[(summ.arm == arm) & np.isclose(summ.alpha, a)][col].iloc[0])

    ck("rows", len(df), 3420, tol=0)
    ck("cohorts", df.cohort.nunique(), 4, tol=0)
    ck("repeats", df.rep.nunique(), N_SPLITS, tol=0)
    ck("folds", df.fold.nunique(), N_FOLDS, tol=0)
    ck("baseline acceptance a=0.05 (r52)", S("own", 0.05, "accept"), 0.40588)
    ck("baseline acceptance a=0.10 (r52)", S("own", 0.10, "accept"), 0.60944)
    ck("baseline acceptance a=0.20 (r52)", S("own", 0.20, "accept"), 0.85069)
    ck("baseline oracle a=0.10 (r52)", S("own", 0.10, "oracle"), 0.79220)
    ck("neighbour acceptance a=0.10", S("own+neighbour", 0.10, "accept"), 0.61098)
    ck("neighbour oracle a=0.10", S("own+neighbour", 0.10, "oracle"), 0.77853)
    ck("neighbour oracle a=0.05", S("own+neighbour", 0.05, "oracle"), 0.62431)
    ck("neighbour oracle a=0.20", S("own+neighbour", 0.20, "oracle"), 0.92322)
    ck("neighbour accuracy", S("own+neighbour", 0.10, "acc"), 0.82839)
    ck("own accuracy", S("own", 0.10, "acc"), 0.83396)
    ck("coverage at nominal, both arms",
       float(min(S(arm, a, "coverage") for arm in ARMS for a in (0.10, 0.20))),
       1.0, tol=0)
    ck("the neighbourhood does not lower the oracle at a=0.10",
       float(S("own+neighbour", 0.10, "oracle") - S("own", 0.10, "oracle")),
       -0.01367)
    d = piv.groupby("alpha")["diff"].mean()
    ck("paired oracle difference a=0.05", float(d.loc[0.05]), -0.00941)
    ck("paired oracle difference a=0.20", float(d.loc[0.20]), 0.0018)
    ck("review fraction barely moves at a=0.10",
       float(S("own+neighbour", 0.10, "review_frac")
             - S("own", 0.10, "review_frac")), -0.00154)
    summ.to_csv(os.path.join(paths.RESULTS, "r56_neighbour_summary.csv"),
                index=False)
    pd.DataFrame(checks).to_csv(
        os.path.join(paths.RESULTS, "r56_neighbour_checks.csv"), index=False)
    nfail = sum(1 for c in checks if not c["PASS"])
    lines.append("")
    lines.append("CHECKS: %d PASS / %d FAIL" % (len(checks) - nfail, nfail))
    io.open(os.path.join(paths.WORK, "r56_report.txt"), "w", encoding="utf-8",
            newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    if args.analyse:
        analyse()
    elif args.cohort:
        run_cohort(args.cohort)
    else:
        raise SystemExit("pass --cohort NAME or --analyse")


if __name__ == "__main__":
    main()
