"""R12 -- how does noise in the *reference* labels propagate into the guarantee?

The certified risk is risk with respect to the reference labels, so reference
noise is a first-class threat to the reading of every result in this paper.  The
marker-rule audit (R6) showed that the reference labels themselves violate the
depot's own published marker rules in a nontrivial fraction of cells: 4.0% for
GVHD expert labels, and higher for several melanoma / CRC populations.  This
script asks how much of the guarantee survives that.

Design
------
For every cohort x fold x classifier, the model is fitted on the *clean*
training folds -- training labels are never corrupted, because a noisy training
set confounds label noise with a different (and much larger) effect.  The
held-out pool supplies both the calibration split and the deployment split, and
its labels are corrupted independently per cell,

    with probability eta : y -> Uniform(other classes),   eta in {0, 1, 2, 5, 10}%

under a seeded RNG, with R independent replicates per setting.  Two regimes:

* ``cal_only`` -- only the calibration half is corrupted; the deployment truth
  stays clean.  This isolates the question "does a noisy calibration set break
  the validity of the bound, or does the controller simply become more
  conservative?" and it is the regime in which the clean risk remains the
  physically meaningful target.
* ``both`` -- calibration and deployment references are both corrupted, the
  realistic case when the pathologist's annotation is the only truth available
  everywhere.

The certified method is :class:`errctl.conformal.SelectiveRiskController`; the
uncalibrated baseline is :func:`errctl.conformal.naive_threshold`.  For every
setting we record the acceptance rate, the selective risk measured against the
CLEAN labels (physically meaningful: what a perfect oracle pathologist would
score), the selective risk measured against the NOISY labels (what the
controller actually saw), the certified threshold, and whether the clean risk
stayed inside the budget.

Coverage convention
-------------------
``covered`` follows the convention used throughout the paper: an empty accepted
set is a *vacuous* success (there is nothing that could be wrong).  That makes a
controller that abstains on everything look perfect, so the report also carries
``empty``, and the derived columns

    cov_nonvac  = P(accepted set is non-empty AND clean risk <= alpha)
    viol_nonvac = P(accepted set is non-empty AND clean risk >  alpha)

which are the honest measures of "the guarantee held on a set I actually
shipped".  Everything is measured against the CLEAN reference labels; the
noisy-label risk is reported alongside for reference only.

Output: experiments/r12_labelnoise.csv (long form, one row per
cohort x fold x clf x eta x regime x replicate x alpha x method).
"""

import os

# Keep the per-worker BLAS/OpenMP footprint small: the outer parallelism is
# process-level, and HistGradientBoosting is the only thread-hungry step.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import multiprocessing as mp
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, paths
from errctl.classifiers import fit_classifier
from errctl.preprocess import DomainNormaliser

CACHE = os.path.join(paths.CACHE, "r08")
OUT = os.path.join(paths.RESULTS, "r12_labelnoise.csv")

ETAS = (0.0, 0.01, 0.02, 0.05, 0.10)
REGIMES = ("cal_only", "both")
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
R_REPS = 10
CLFS = ("logreg", "hgb")
N_FOLDS = 5
MAX_TRAIN = 20000
MIN_CLASS_TRAIN = 3
MIN_POOL = 200
MIN_TRAIN = 200


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)),
          flush=True)


def skip(ds, fold, kind, reason):
    log("  skip %s fold%d %s: %s" % (ds, fold, kind, reason))
    return []


def stable_seed(*parts):
    """Deterministic across processes and runs (unlike the builtin hash)."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.crc32(key)) % (2 ** 31)


def corrupt_labels(y, eta, rng, n_classes):
    """Independently flip each label to a uniformly random *other* class."""
    y = np.asarray(y, dtype=np.int64)
    if eta <= 0.0 or n_classes < 2:
        return y.copy()
    out = y.copy()
    idx = np.nonzero(rng.random(y.size) < eta)[0]
    if idx.size:
        offset = 1 + np.floor(rng.random(idx.size) * (n_classes - 1)).astype(np.int64)
        offset = np.clip(offset, 1, n_classes - 1)
        out[idx] = (y[idx] + offset) % n_classes
    return out


def run_job(job):
    ds, fold, kind, n_reps = job
    path = os.path.join(CACHE, "%s.npz" % ds)
    if not os.path.exists(path):
        return skip(ds, fold, kind, "no cache file")
    d = np.load(path, allow_pickle=True)
    X = np.nan_to_num(np.asarray(d["X"], dtype=np.float64), nan=0.0,
                      posinf=0.0, neginf=0.0)
    yv = np.asarray([str(v) for v in np.asarray(d["y"])])
    fo = np.asarray(d["fold"]).astype(int)

    tr = np.nonzero(fo != fold)[0]
    pool = np.nonzero(fo == fold)[0]
    if tr.size < MIN_TRAIN or pool.size < MIN_POOL:
        return skip(ds, fold, kind, "n_train=%d n_pool=%d" % (tr.size, pool.size))

    counts = pd.Series(yv[tr]).value_counts()
    classes = sorted(c for c in counts.index if counts[c] >= MIN_CLASS_TRAIN)
    if len(classes) < 2:
        return skip(ds, fold, kind, "%d usable classes" % len(classes))
    keep = set(classes)
    tr = np.asarray([i for i in tr if yv[i] in keep], dtype=int)
    pool = np.asarray([i for i in pool if yv[i] in keep], dtype=int)
    if tr.size < MIN_TRAIN or pool.size < MIN_POOL:
        return skip(ds, fold, kind,
                    "after class filter n_train=%d n_pool=%d" % (tr.size, pool.size))

    n_cls = len(classes)
    ix = {c: i for i, c in enumerate(classes)}
    if tr.size > MAX_TRAIN:
        tr = np.sort(np.random.default_rng(1000 + int(fold)).choice(
            tr, MAX_TRAIN, replace=False))
    y_tr = np.asarray([ix[v] for v in yv[tr]], dtype=np.int64)
    y_pool = np.asarray([ix[v] for v in yv[pool]], dtype=np.int64)
    if np.unique(y_tr).size < 2 or np.unique(y_pool).size < 2:
        return skip(ds, fold, kind, "degenerate class support")

    normaliser = DomainNormaliser("z_dataset").fit("src", X[tr])
    X_tr = normaliser.transform("src", X[tr])
    X_pool = normaliser.transform("src", X[pool])

    clf = fit_classifier(X_tr, y_tr, kind=kind, seed=0)
    proba = np.asarray(clf.predict_proba(X_pool), dtype=np.float64)
    pred = proba.argmax(1)
    scores = 1.0 - proba.max(1)
    err_clean = (pred != y_pool).astype(np.float64)

    n = int(pool.size)
    half = n // 2
    pick = np.random.default_rng(2000 + int(fold)).permutation(n)
    cal_idx, te_idx = pick[:half], pick[half:]
    pool_acc = float((pred == y_pool).mean())
    base_risk = float(err_clean[te_idx].mean())

    rows = []
    for eta in ETAS:
        for regime in REGIMES:
            for rep in range(n_reps):
                # The calibration draw is deliberately shared between the two
                # regimes (the seed does not contain the regime), so that
                # ``cal_only`` and ``both`` differ in exactly one thing: whether
                # the deployment reference is corrupted as well.
                rng = np.random.default_rng(
                    stable_seed(ds, fold, kind, "%.4f" % eta, rep))
                y_cal_n = corrupt_labels(y_pool[cal_idx], eta, rng, n_cls)
                if regime == "both":
                    rng_te = np.random.default_rng(
                        stable_seed(ds, fold, kind, "%.4f" % eta, rep, "te"))
                    y_te_n = corrupt_labels(y_pool[te_idx], eta, rng_te, n_cls)
                else:
                    y_te_n = y_pool[te_idx]

                s_cal, s_te = scores[cal_idx], scores[te_idx]
                e_cal = (pred[cal_idx] != y_cal_n).astype(np.float64)
                e_te_clean = err_clean[te_idx]
                e_te_noisy = (pred[te_idx] != y_te_n).astype(np.float64)

                for alpha in ALPHAS:
                    lam_crc = C.SelectiveRiskController(alpha, DELTA).fit(
                        s_cal, e_cal).lambda_hat
                    lam_naive = C.naive_threshold(s_cal, e_cal, alpha)
                    for method, lam in (("crc", lam_crc), ("naive", lam_naive)):
                        acc_m = s_te <= lam
                        n_acc = int(acc_m.sum())
                        risk = float(e_te_clean[acc_m].mean()) if n_acc else float("nan")
                        risk_n = float(e_te_noisy[acc_m].mean()) if n_acc else float("nan")
                        rows.append({
                            "dataset": ds, "fold": int(fold), "clf": kind,
                            "eta": eta, "regime": regime, "rep": int(rep),
                            "alpha": alpha, "delta": DELTA, "method": method,
                            "n_train": int(tr.size), "n_cal": int(cal_idx.size),
                            "n_test": int(te_idx.size), "n_classes": n_cls,
                            "pool_acc": pool_acc, "base_risk_clean": base_risk,
                            "lambda_hat": float(lam),
                            "accept_rate": n_acc / float(max(te_idx.size, 1)),
                            "selective_risk": risk,
                            "selective_risk_noisy": risk_n,
                            "covered": int((not np.isfinite(risk)) or risk <= alpha),
                            "covered_noisy": int((not np.isfinite(risk_n))
                                                 or risk_n <= alpha),
                            "empty": int(n_acc == 0),
                        })
    return rows


def _job(j):
    return run_job(j)


def report(df):
    """Print the grouped tables. Coverage is always against the CLEAN labels."""
    keys = ["eta", "regime", "method", "alpha"]
    df = df.copy()
    df["lam_finite"] = df["lambda_hat"].where(np.isfinite(df["lambda_hat"]))
    df["nonvac"] = (df["empty"] == 0).astype(float)
    df["cov_nonvac"] = ((df["empty"] == 0) & (df["covered"] == 1)).astype(float)
    df["viol_nonvac"] = ((df["empty"] == 0) & (df["covered"] == 0)).astype(float)

    agg = {"accept_rate": ("accept_rate", "mean"),
           "selective_risk": ("selective_risk", "mean"),
           "risk_noisy": ("selective_risk_noisy", "mean"),
           "lambda_hat": ("lam_finite", "mean"),
           "coverage": ("covered", "mean"),
           "empty": ("empty", "mean"),
           "cov_nonvac": ("cov_nonvac", "mean"),
           "viol_nonvac": ("viol_nonvac", "mean"),
           "n": ("covered", "size")}

    log("---- summary: mean over cohorts/folds/replicates (clean-truth risk) ----")
    log("\n" + df.groupby(keys).agg(**agg).round(3).to_string())

    log("---- same, split by classifier ----")
    log("\n" + df.groupby(["clf"] + keys).agg(**agg).round(3).to_string())

    log("---- crc only, split by cohort ----")
    d2 = df[df["method"] == "crc"]
    log("\n" + d2.groupby(["dataset"] + keys).agg(**agg).round(3).to_string())

    log("---- HEADLINE (crc): does a noisy REFERENCE break the guarantee, "
        "or just shrink the accepted set? ----")
    hl = (d2.groupby(["eta", "regime", "alpha"])
            .agg(accept_rate=("accept_rate", "mean"),
                 selective_risk=("selective_risk", "mean"),
                 lambda_hat=("lam_finite", "mean"),
                 coverage=("covered", "mean"),
                 empty=("empty", "mean"),
                 cov_nonvac=("cov_nonvac", "mean"),
                 viol_nonvac=("viol_nonvac", "mean"),
                 n=("covered", "size")).round(3))
    log("\n" + hl.to_string())

    log("---- HEADLINE (naive baseline), same columns ----")
    d3 = df[df["method"] == "naive"]
    hl2 = (d3.groupby(["eta", "regime", "alpha"])
             .agg(accept_rate=("accept_rate", "mean"),
                  selective_risk=("selective_risk", "mean"),
                  coverage=("covered", "mean"),
                  empty=("empty", "mean"),
                  cov_nonvac=("cov_nonvac", "mean"),
                  viol_nonvac=("viol_nonvac", "mean"),
                  n=("covered", "size")).round(3))
    log("\n" + hl2.to_string())

    log("---- sanity: eta=0 must reproduce the exchangeable benchmark (R03/R08) ----")
    z = df[df["eta"] == 0.0]
    log("\n" + z.groupby(["clf", "method", "alpha"])
             .agg(accept_rate=("accept_rate", "mean"),
                  selective_risk=("selective_risk", "mean"),
                  coverage=("covered", "mean"),
                  pool_acc=("pool_acc", "mean")).round(3).to_string())

    log("---- cost of the corruption: acceptance lost per absolute risk budget ----")
    piv = (d2[d2["regime"] == "cal_only"]
           .pivot_table(index=["clf", "alpha"], columns="eta",
                        values="accept_rate", aggfunc="mean").round(3))
    log("\n" + piv.to_string())

    log("---- paired contrast: corrupting the DEPLOYMENT reference too "
        "(cal_only -> both, same calibration draw) ----")
    pkey = ["dataset", "fold", "clf", "eta", "rep", "alpha"]
    cols = ["accept_rate", "selective_risk", "selective_risk_noisy", "covered"]
    a = (d2[d2["regime"] == "both"].set_index(pkey)[cols]
           .add_suffix("_both"))
    b = (d2[d2["regime"] == "cal_only"].set_index(pkey)[cols]
           .add_suffix("_cal"))
    j = a.join(b, how="inner")
    j["d_accept"] = j["accept_rate_both"] - j["accept_rate_cal"]
    j["d_risk_clean"] = j["selective_risk_both"] - j["selective_risk_cal"]
    j["d_risk_noisy"] = (j["selective_risk_noisy_both"]
                         - j["selective_risk_noisy_cal"])
    ptab = (j.groupby(["clf", "eta", "alpha"])
             .agg(n=("d_accept", "size"), d_accept=("d_accept", "mean"),
                  d_risk_clean=("d_risk_clean", "mean"),
                  d_risk_noisy=("d_risk_noisy", "mean")).round(4))
    log("\n" + ptab.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--reps", type=int, default=R_REPS)
    ap.add_argument("--limit", default="")
    ap.add_argument("--analyse", default="",
                    help="skip the experiment and report on an existing CSV")
    args = ap.parse_args()

    if args.analyse:
        report(pd.read_csv(args.analyse))
        return

    datasets = [d for d in paths.MANUAL_DATASETS
                if (not args.limit or args.limit in d)
                and os.path.exists(os.path.join(CACHE, "%s.npz" % d))]
    jobs = [(ds, f, k, args.reps) for ds in datasets for f in range(N_FOLDS)
            for k in CLFS]
    log("R12 label-noise sensitivity | %d jobs | %d replicates | alphas %s"
        % (len(jobs), args.reps, list(ALPHAS)))

    rows = []
    t0 = time.time()
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            for i, res in enumerate(pool.imap(_job, jobs, chunksize=1), 1):
                rows.extend(res)
                log("  job %d/%d -> %d rows (%.0fs)"
                    % (i, len(jobs), len(res), time.time() - t0))
    else:
        for i, j in enumerate(jobs, 1):
            res = run_job(j)
            rows.extend(res)
            log("  job %d/%d -> %d rows (%.0fs)" % (i, len(jobs), len(res),
                                                    time.time() - t0))

    df = pd.DataFrame(rows)
    if df.empty:
        log("no rows produced -- nothing to write")
        return
    df["lambda_finite"] = df["lambda_hat"].where(np.isfinite(df["lambda_hat"]))
    df = df.sort_values(["dataset", "clf", "eta", "regime", "rep", "alpha",
                         "method"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    log("wrote %s (%d rows)" % (OUT, len(df)))

    report(df)
    log("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()