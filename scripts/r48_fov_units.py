"""R48 -- cross-field-of-view coverage at field-of-view resolution.

The published cross-field-of-view family (the `within` units of
`scripts/r02_run_experiments.py`) holds out a *fold* of images at a time: five
folds per cohort, twenty units in all.  With twenty units the cluster bootstrap
over units cannot resolve the shortfall from the nominal level.  At a 10% budget
the point estimate is 0.817 and the interval is 0.667-0.950, so -- as the
manuscript says -- the violation is directionally clear but not established.

This script changes exactly one axis and leaves the rest alone: one image is
held out per unit, so coverage is estimated over every annotated field of view in
the four cohorts.  The model still never trains on the held-out image nor on the
images that supply its calibration cells; the calibration subsample, the six
calibration draws per unit, the certificate, the baselines and the readouts are
the published ones, imported from `scripts/r02_run_experiments.py` and
`scripts/r03_analyse.py` so that the definitions cannot drift.

Phases:
  --write    fit and store predictions under work/predictions_fov/
  --analyse  turn them into experiments/r48_metrics.csv plus the per-unit
             coverage and its bootstrap interval, written to work/r48_report.txt
"""

import argparse
import importlib.util
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from errctl import data as D, paths
from errctl import classifiers as CLS
from errctl.preprocess import DomainNormaliser

FOV_DIR = os.path.join(paths.WORK, "predictions_fov")
os.makedirs(FOV_DIR, exist_ok=True)

FAMILY = "crossfov_loo"
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
CAL_TARGET = 2500
CAL_MAX_IMAGES = 8
MIN_TRAIN_IMAGES = 3
MIN_CELLS = 50
N_BOOT = 2000


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)),
          flush=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pin_threads():
    """r02 fits the random forest with n_jobs=-1, which oversubscribes the machine
    once the driver itself fans out over units.  Pin it to one thread per worker;
    the forest is unchanged.  (fit_classifier resolves make_classifier from this
    module at call time, so patching the module attribute is enough.)"""
    orig = CLS.make_classifier

    def patched(kind, seed=0):
        clf = orig(kind, seed=seed)
        if kind == "rf":
            clf.set_params(n_jobs=1)
        return clf

    CLS.make_classifier = patched


_COHORT = {}


def cohort_frames(ds):
    if ds not in _COHORT:
        g, markers = D.gold(ds, drop_unidentified=True)
        g = g.reset_index(drop=True)
        _COHORT[ds] = (g, markers, g[markers].to_numpy(dtype=np.float64))
    return _COHORT[ds]


def calibration_images(order, i, sizes):
    """The images that supply the calibration pool for the unit whose test image
    is order[i]: the next images in sorted order, gathered until the pool holds
    CAL_TARGET cells, never more than CAL_MAX_IMAGES of them, never the test image
    itself and never so many that fewer than MIN_TRAIN_IMAGES are left to train on."""
    n = len(order)
    cap = max(1, min(CAL_MAX_IMAGES, n - 1 - MIN_TRAIN_IMAGES))
    taken, total = [], 0
    for k in range(1, n):
        if len(taken) >= cap:
            break
        taken.append(order[(i + k) % n])
        total += int(sizes[order[(i + k) % n]])
        if total >= CAL_TARGET:
            break
    return set(taken)


_SIZES = {}


def run_image(job):
    ds, image, clfs = job
    _pin_threads()
    r02 = _load("r02_run_experiments")
    g, markers, X = cohort_frames(ds)
    y_all = g["class_expert"].astype(str).to_numpy()
    imgs = g["image"].to_numpy()
    order = np.array(sorted(g["image"].unique()))
    if ds not in _SIZES:
        _SIZES[ds] = {str(k): int(v) for k, v in pd.Series(imgs).value_counts().items()}
    i = int(np.nonzero(order == image)[0][0])
    cal_imgs = calibration_images(order, i, _SIZES[ds])
    te = imgs == image
    ca = np.isin(imgs, list(cal_imgs))
    tr = ~(te | ca)
    te_idx, ca_idx, tr_idx = np.nonzero(te)[0], np.nonzero(ca)[0], np.nonzero(tr)[0]
    if len(te_idx) < MIN_CELLS or len(ca_idx) < MIN_CELLS:
        log("   skip %s:%s (test=%d cal=%d)" % (ds, image, len(te_idx), len(ca_idx)))
        return []
    tr_idx = r02.subsample_training(tr_idx, y_all, r02.MAX_TRAIN, seed=0)
    nz = DomainNormaliser("z_dataset").fit("src", X[tr_idx])
    Xtr = nz.transform("src", X[tr_idx])
    Xca = nz.transform("src", X[ca_idx])
    Xte = nz.transform("src", X[te_idx])
    rows = []
    for clf_kind in clfs:
        name = "withinFOV:%s:%s" % (ds, image)
        out = r02.run_unit(name, g.iloc[te_idx], np.zeros(len(te_idx), dtype=int),
                           y_all[tr_idx], y_all[ca_idx], y_all[te_idx],
                           Xtr, Xca, Xte, imgs[ca_idx], imgs[te_idx],
                           clf_kind, r02.N_CAL_REPEATS, FOV_DIR)
        out["dataset"] = ds
        out["image"] = image
        out["n_cal_images"] = len(cal_imgs)
        out["n_train_images"] = len(np.unique(imgs[tr_idx]))
        rows.append(out)
    return rows


def write_predictions(args):
    clfs = [c for c in args.clfs.split(",") if c]
    jobs = []
    for ds in paths.MANUAL_DATASETS:
        if args.only and ds not in args.only:
            continue
        g, markers, X = cohort_frames(ds)
        order = sorted(g["image"].unique())
        if args.limit:
            order = order[:args.limit]
        for image in order:
            jobs.append((ds, image, clfs))
    log("writing predictions for %d (cohort, image) units x %d classifiers with %d worker(s)"
        % (len(jobs), len(clfs), max(1, args.jobs)))
    rows = []
    if args.jobs > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            for i, out in enumerate(pool.imap(run_image, jobs, chunksize=1), 1):
                rows.extend(out)
                if i % 10 == 0 or i == len(jobs):
                    log("   %d/%d units done, %d rows" % (i, len(jobs), len(rows)))
    else:
        for i, job in enumerate(jobs, 1):
            rows.extend(run_image(job))
            if i % 10 == 0 or i == len(jobs):
                log("   %d/%d units done, %d rows" % (i, len(jobs), len(rows)))
    p = os.path.join(paths.WORK, "r48_runs.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    log("wrote %s (%d rows)" % (p, len(rows)))


def bootstrap_ci(values, seed=0):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(N_BOOT, values.size))
    means = values[idx].mean(axis=1)
    return (float(values.mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def summarise(m, method, alpha):
    s = m[(m["method"] == method) & np.isclose(m["alpha"], alpha)]
    if s.empty:
        return None
    per_unit = s.groupby("unit")["covered"].mean()
    joint = (s.assign(_joint=((s["covered"] > 0.5) & (s["empty"] == 0)).astype(float))
             .groupby("unit")["_joint"].mean())
    nv = s[s["empty"] == 0].groupby("unit")["covered"].mean()
    acc = s.groupby("unit")["accept_rate"].mean()
    pt, lo, hi = bootstrap_ci(per_unit.values)
    jt, jlo, jhi = bootstrap_ci(joint.values)
    ct, clo, chi = bootstrap_ci(nv.values)
    return {
        "method": method, "alpha": alpha, "n_units": int(per_unit.size),
        "coverage": pt, "cov_lo": lo, "cov_hi": hi,
        "coverage_joint": jt, "joint_lo": jlo, "joint_hi": jhi,
        "coverage_cond": ct, "cond_lo": clo, "cond_hi": chi,
        "nonvac_frac": float((s["empty"] == 0).mean()),
        "n_units_nonvacuous": int(nv.size),
        "accept": float(acc.mean()),
        "vacuous_units": float((s.groupby("unit")["empty"].mean() >= 1.0).mean()),
    }


def analyse(args):
    r03 = _load("r03_analyse")
    r03.PRED_DIR = FOV_DIR
    files = sorted(f for f in os.listdir(FOV_DIR) if f.endswith(".npz"))
    log("analysing %d stored (unit, classifier) files" % len(files))
    rows, ece = [], []
    for i, f in enumerate(files, 1):
        r, e = r03.process_file(f, list(ALPHAS), DELTA)
        rows.extend(r)
        ece.extend(e)
        if i % 50 == 0 or i == len(files):
            log("   %d/%d files" % (i, len(files)))
    m = pd.DataFrame(rows)
    m["family"] = FAMILY
    mp_path = os.path.join(paths.RESULTS, "r48_metrics.csv")
    m.to_csv(mp_path, index=False)
    pd.DataFrame(ece).to_csv(os.path.join(paths.RESULTS, "r48_ece.csv"), index=False)
    log("wrote %s (%d rows)" % (mp_path, len(m)))

    methods = ["crc", "crc_cond", "crc_mondrian", "oracle", "naive", "crc_temp",
               "conformal_set", "crc_eb", "accept_all"]
    out = [summarise(m, meth, a) for meth in methods for a in ALPHAS]
    out = pd.DataFrame([o for o in out if o is not None])
    ci_path = os.path.join(paths.RESULTS, "r48_coverage_ci.csv")
    out.to_csv(ci_path, index=False)
    log("wrote %s" % ci_path)

    per_cohort = (m[(m["method"] == "crc") & np.isclose(m["alpha"], 0.10)]
                  .groupby("unit").agg(n_test=("n_test_eval", "max"),
                                       accept=("accept_rate", "mean"),
                                       covered=("covered", "mean")))
    per_cohort["cohort"] = [u.split("_")[1] for u in per_cohort.index]

    lines = []
    def w(s=""):
        lines.append(s)
        log(s)
    w("R48 -- cross-field-of-view coverage at field-of-view resolution")
    w("=" * 72)
    w("units: %d fields of view over %d cohorts; %d stored prediction files"
      % (per_cohort.shape[0], per_cohort["cohort"].nunique(), len(files)))
    w("published design: 20 units (4 cohorts x 5 folds of images)")
    w("")
    for ds, grp in per_cohort.groupby("cohort"):
        w("  %-16s units=%3d  test cells med=%6.0f  accept=%.3f  covered=%.3f"
          % (ds, grp.shape[0], grp["n_test"].median(), grp["accept"].mean(),
             grp["covered"].mean()))
    w("")
    w("headline readouts (bootstrap over units, %d resamples)" % N_BOOT)
    w("methods: cov = P(covered) ; cov(joint) = P(covered and non-empty), the")
    w("         stronger readout ; cov(cond) = P(covered | non-empty), the")
    w("         manuscript Table-1 readout")
    hdr = "%-14s %5s %6s %6s %-16s %6s %-16s %6s" % (
        "method", "alpha", "units", "cov", "[2.5%, 97.5%]", "cov(joint)",
        "cov(cond)", "accept")
    w(hdr)
    for meth in ["crc", "crc_cond", "crc_mondrian", "oracle", "naive",
                 "crc_temp", "conformal_set"]:
        for a in ALPHAS:
            row = out[(out["method"] == meth) & np.isclose(out["alpha"], a)]
            if row.empty:
                continue
            r = row.iloc[0]
            w("%-14s %5.2f %6d %6.3f [%.3f, %.3f]  %6.3f [%.3f, %.3f]  %6.3f [%.3f, %.3f]  %6.3f"
              % (meth, a, r["n_units"], r["coverage"], r["cov_lo"], r["cov_hi"],
                 r["coverage_joint"], r["joint_lo"], r["joint_hi"],
                 r["coverage_cond"], r["cond_lo"], r["cond_hi"],
                 r["accept"]))
    w("")
    ref = os.path.join(paths.RESULTS, "r13_coverage_ci.csv")
    if os.path.exists(ref):
        d = pd.read_csv(ref)
        sel = d[(d["family"] == "within") & d["method"].isin(["crc", "oracle", "naive"])]
        w("published 5-fold numbers for the same family (experiments/r13_coverage_ci.csv)")
        for _, r in sel.iterrows():
            w("  %-8s alpha=%.2f units=%2d coverage=%.3f [%.3f, %.3f] accept=%.3f"
              % (r["method"], r["alpha"], r["n_units"], r["coverage"], r["cov_lo"],
                 r["cov_hi"], r["accept"]))
        w("")
    w("verdict")
    for a in ALPHAS:
        row = out[(out["method"] == "crc") & np.isclose(out["alpha"], a)]
        if row.empty:
            continue
        r = row.iloc[0]
        verdict = ("interval excludes" if r["cov_hi"] < 0.90 else
                   "interval includes")
        w("  alpha=%.2f  coverage=%.3f [%.3f, %.3f]  -> %s the nominal 0.90"
          % (a, r["coverage"], r["cov_lo"], r["cov_hi"], verdict))
        w("            joint %.3f [%.3f, %.3f]  cond %.3f [%.3f, %.3f]  accept %.3f"
          % (r["coverage_joint"], r["joint_lo"], r["joint_hi"],
             r["coverage_cond"], r["cond_lo"], r["cond_hi"], r["accept"]))
    p = os.path.join(paths.WORK, "r48_report.txt")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    log("wrote %s" % p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not (args.write or args.analyse):
        ap.error("pass --write and/or --analyse")
    if args.write:
        write_predictions(args)
    if args.analyse:
        analyse(args)


if __name__ == "__main__":
    main()
