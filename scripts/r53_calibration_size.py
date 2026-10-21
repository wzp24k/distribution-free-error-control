"""R53 -- how much of the validity price is bought by more calibration cells?

The certificate's acceptance at a fixed budget is bounded below by the oracle and
above by the oracle minus the finite-sample correction. That correction is the
only part a laboratory can remove with more labels, and it is the part that sets
the adoption threshold: below some calibration size the correction swamps the
score and nothing is certified at all.

This script sweeps the calibration size with the deployment population held
fixed. It refits nothing -- it reads the stored per-cell probabilities of the
random-split family (`work/predictions/cellsplit_*.npz`) and re-runs the
controller with a calibration set of a chosen size drawn from the cells outside a
fixed-size test set. The oracle and the uncalibrated plug-in are recomputed on
the same test cells, so the three curves are comparable at every size.

Output: experiments/r53_calibration_size.csv,
        experiments/r53_calibration_size_summary.csv
"""

import argparse
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, paths

PRED_DIR = os.path.join(paths.WORK, "predictions")
OUT = os.path.join(paths.RESULTS, "r53_calibration_size.csv")
OUTSUM = os.path.join(paths.RESULTS, "r53_calibration_size_summary.csv")

NAME_RE = re.compile(r"(?P<unit>cellsplit_.+_fold\d+)__(?P<clf>logreg|rf|hgb)\.npz$")
ALPHAS = (0.05, 0.10, 0.20)
GRID = (25, 50, 100, 200, 400, 800, 1600, 3200, 6400)
TEST_FRAC = 0.25
DELTA = 0.10
SEED = 7000
SHORT = {"Breast_Jun0524": "Breast", "CRC_Feb2623": "CRC",
         "GVHD_Feb1323": "GVHD", "Melanoma_Oct2522": "Melanoma"}


def cohort_of(unit):
    m = re.match(r"cellsplit_(.+)_fold\d+$", unit)
    return SHORT.get(m.group(1), m.group(1))


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)), flush=True)


def oracle_acceptance(s_te, e_te, alpha, min_accept=25):
    """Largest threshold meeting the budget when the test labels are known."""
    order = np.argsort(s_te, kind="mergesort")
    s, e = s_te[order], e_te[order]
    cum = np.cumsum(e)
    k = np.arange(1, s.size + 1)
    ok = (cum / k <= alpha) & (k >= min_accept)
    if not ok.any():
        return np.zeros(s_te.size, dtype=bool)
    lam = float(s[np.nonzero(ok)[0][-1]])
    return s_te <= lam


def run_job(fname):
    m = NAME_RE.match(fname)
    if not m:
        return []
    unit, clf = m.group("unit"), m.group("clf")
    d = np.load(os.path.join(PRED_DIR, fname), allow_pickle=True)
    probs = np.asarray(d["test_probs"], dtype=np.float64)
    y = np.asarray(d["test_y"])
    n = probs.shape[0]
    err = (probs.argmax(1) != y).astype(float)
    score = 1.0 - probs.max(1)
    fold = int(re.search(r"fold(\d+)$", unit).group(1))

    rows = []
    rng = np.random.default_rng(SEED + fold)
    n_test = max(200, int(round(TEST_FRAC * n)))
    if n - n_test < 25:
        return []
    for rep in range(REPS):
        perm = rng.permutation(n)
        te, rest = perm[:n_test], perm[n_test:]
        s_te, e_te = score[te], err[te]
        for alpha in ALPHAS:
            ora = oracle_acceptance(s_te, e_te, alpha)
            k_o = int(ora.sum())
            for want in GRID:
                k = min(want, rest.size)
                if k < 25:
                    continue
                cal = rest[:k]
                ctl = C.SelectiveRiskController(alpha, DELTA).fit(score[cal], err[cal])
                acc = ctl.accept(s_te)
                n_acc = int(acc.sum())
                r = float(e_te[acc].mean()) if n_acc else np.nan
                lam = C.naive_threshold(score[cal], err[cal], alpha)
                n_n = int((s_te <= lam).sum())
                rows.append({"unit": unit, "cohort": cohort_of(unit), "clf": clf,
                             "alpha": alpha, "rep": rep,
                             "n_cal": int(k), "n_cal_wanted": int(want),
                             "capped": int(k < want),
                             "n_pool": n, "n_test": int(n_test),
                             "accept_rate": n_acc / n_test, "covered": int((not np.isfinite(r)) or r <= alpha),
                             "empty": int(n_acc == 0), "selective_risk": r,
                             "oracle_rate": k_o / n_test,
                             "accept_rate_naive": n_n / n_test})
    return rows


def _job(f):
    return run_job(f)


REPS = 10


def main():
    global REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    REPS = args.reps

    files = sorted(f for f in os.listdir(PRED_DIR) if NAME_RE.match(f))
    log("%d stored prediction files, %d reps, grid %s" % (len(files), REPS, GRID))
    rows = []
    with mp.Pool(args.jobs) as pool:
        for i, res in enumerate(pool.imap(_job, files, chunksize=1), 1):
            rows.extend(res)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    ok = df[df.capped == 0].copy()

    def summarise(frame, keys):
        g = frame.groupby(keys)
        s = g.agg(n_units=("unit", "nunique"),
                  accept_mean=("accept_rate", "mean"),
                  oracle_mean=("oracle_rate", "mean"),
                  accept_naive=("accept_rate_naive", "mean"),
                  coverage=("covered", "mean"),
                  vacuous=("empty", "mean")).reset_index()
        s["price_pp"] = (s.oracle_mean - s.accept_mean) * 100.0
        return s.round(5)

    per_cohort = summarise(ok, ["cohort", "clf", "alpha", "n_cal_wanted"])
    gmin = int(ok.groupby("unit").n_cal_wanted.max().min())
    overall = summarise(ok[ok.n_cal_wanted <= gmin], ["clf", "alpha", "n_cal_wanted"])
    overall.insert(0, "panel", "all cohorts (n_cal <= %d)" % gmin)
    capped = summarise(df[df.capped == 1], ["cohort", "clf", "alpha", "n_cal_wanted"])
    summ = pd.concat([per_cohort.assign(panel="per cohort"), overall, capped.assign(panel="capped")],
                     ignore_index=True)
    summ.to_csv(OUTSUM, index=False)
    log("wrote %s (%d rows) and %s  (balanced panel: %d units, grid <= %d)"
        % (OUT, len(df), OUTSUM, ok.groupby("unit").ngroups, gmin))

    for al in ALPHAS:
        for scope in ("Breast", "CRC", "GVHD", "Melanoma"):
            p = per_cohort[(per_cohort.cohort == scope) & np.isclose(per_cohort.alpha, al)]
            if p.empty:
                continue
            log("\n%s alpha=%.2f  acceptance | oracle | price_pp | coverage" % (scope, al))
            for r in p.itertuples():
                log("   n_cal=%-5d %-7s accept=%.3f oracle=%.3f price=%5.1f cov=%.3f"
                    % (r.n_cal_wanted, r.clf, r.accept_mean, r.oracle_mean,
                       r.price_pp, r.coverage))
    p = overall[np.isclose(overall.alpha, 0.10)]
    log("\nALL COHORTS alpha=0.10 (balanced):\n%s"
        % p.pivot(index="n_cal_wanted", columns="clf",
                  values=["accept_mean", "price_pp"]).round(3).to_string())


if __name__ == "__main__":
    main()
