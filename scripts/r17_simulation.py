"""R17 -- Monte-Carlo validation of the two new controller variants.

Companion to r09, which validates the default construction.  Two additions.

*Fixed-sequence thresholds.*  ``select="fixed_sequence"`` walks the candidate
thresholds from the safest upwards and stops at the first whose bound exceeds
the budget, instead of charging every grid point a Bonferroni share.  Coverage
here must still be at least 1 - delta; the question is whether it is and whether
the gain in acceptance is real.

*Covariate shift.*  r09 showed that the original weighted variant restores
coverage under a deliberately shifted score distribution but accepts nothing,
because an empirical-Bernstein radius on a weighted mean explodes with the
variance of the weight.  ``WeightedSelectiveRiskController`` replaces that
radius with a range-aware Hoeffding one and, when unlabelled deployment scores
are available, reads the denominator off them directly.  The settings below
include ``shift_labelshift``, where the error probability given the score itself
differs between source and deployment: there the weighting assumption is false
and coverage should fail, which is the honest boundary of the method.

Output: experiments/r17_simulation.csv and _summary.csv.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from errctl import conformal as C, paths

N_TEST = 4000
DELTA = 0.10
ALPHAS = (0.05, 0.10, 0.20)

SETTINGS = {
    "exchangeable_200":  ("uniform", "uniform", 200, 0.0, 0.0),
    "exchangeable_2000": ("uniform", "uniform", 2000, 0.0, 0.0),
    "hetero_strong":     ("uniform", "uniform", 500, 0.8, 0.0),
    "shift_mild_1000":   ("uniform", "beta2", 1000, 0.0, 0.0),
    "shift_mild_4000":   ("uniform", "beta2", 4000, 0.0, 0.0),
    "shift_strong_1000": ("uniform", "beta3", 1000, 0.0, 0.0),
    "shift_labelshift":  ("uniform", "beta2", 1000, 0.0, 0.12),
}


def draw(kind, n, rng):
    if kind == "uniform":
        return rng.random(n)
    if kind == "beta2":
        return rng.beta(2.0, 1.0, size=n)
    if kind == "beta3":
        return rng.beta(3.0, 1.0, size=n)
    raise ValueError(kind)


def true_weight(kind, s):
    if kind == "uniform":
        return np.ones_like(s)
    if kind == "beta2":
        return 2.0 * s
    if kind == "beta3":
        return 3.0 * s ** 2
    raise ValueError(kind)


def error_prob(s, u, het=0.0, offset=0.0):
    p = 0.02 + 0.40 * s + offset
    if het:
        p = p + het * (u - 0.5)
    return np.clip(p, 0.0, 0.99)


def estimated_weight(s_cal, s_te):
    from sklearn.linear_model import LogisticRegression
    x = np.concatenate([s_cal, s_te]).reshape(-1, 1)
    y = np.concatenate([np.zeros(s_cal.size), np.ones(s_te.size)]).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(x, y)
    p = np.clip(clf.predict_proba(s_cal.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1.0 - p) * (s_cal.size / float(s_te.size))
    return np.clip(w, 0.0, 20.0)


def one_rep(rep, name):
    src, tgt, n_cal, het, offset = SETTINGS[name]
    rng = np.random.default_rng(104729 * rep + 31 * len(name) + n_cal)
    s_cal = draw(src, n_cal, rng)
    s_te = draw(tgt, N_TEST, rng)
    u_cal, u_te = rng.random(n_cal), rng.random(N_TEST)
    e_cal = (rng.random(n_cal) < error_prob(s_cal, u_cal, het, 0.0)).astype(float)
    e_te = (rng.random(N_TEST) < error_prob(s_te, u_te, het, offset)).astype(float)
    shifted = tgt != src
    w_known = true_weight(tgt, s_cal) if shifted else None
    w_est = estimated_weight(s_cal, s_te) if shifted else None

    rows = []

    def emit(method, accepted, extra=None):
        accepted = np.asarray(accepted, dtype=bool)
        n_acc = int(accepted.sum())
        risk = float(e_te[accepted].mean()) if n_acc else np.nan
        row = {"setting": name, "n_cal": n_cal, "method": method, "rep": rep,
               "alpha": alpha,
               "n_accepted": n_acc, "accept_rate": n_acc / float(N_TEST),
               "selective_risk": risk,
               "covered": int((not np.isfinite(risk)) or risk <= alpha),
               "empty": int(n_acc == 0)}
        if extra:
            row.update(extra)
        rows.append(row)

    for alpha in ALPHAS:
        emit("accept_all", np.ones(N_TEST, dtype=bool))
        emit("naive", s_te <= C.naive_threshold(s_cal, e_cal, alpha))
        emit("crc", C.SelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal).accept(s_te))
        emit("crc_fs", C.SelectiveRiskController(alpha, DELTA, select="fixed_sequence")
             .fit(s_cal, e_cal).accept(s_te))
        emit("crc_cond",
             C.SelectiveRiskController(alpha, DELTA, num="cond").fit(s_cal, e_cal).accept(s_te))
        if shifted:
            emit("w_old", C.SelectiveRiskController(alpha, DELTA, weighted=True)
                 .fit(s_cal, e_cal, w_est).accept(s_te))
            c1 = C.WeightedSelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal, w_est)
            emit("w_new", c1.accept(s_te))
            c2 = C.WeightedSelectiveRiskController(alpha, DELTA)
            c2.fit(s_cal, e_cal, w_est, target_scores=s_te)
            emit("w_new_td", c2.accept(s_te))
            c3 = C.WeightedSelectiveRiskController(alpha, DELTA)
            c3.fit(s_cal, e_cal, w_known, target_scores=s_te)
            emit("w_known_td", c3.accept(s_te))
            emit("sc", C.ScoreConditionalController(alpha, DELTA)
                 .fit(s_cal, e_cal, s_te).accept(s_te))
        order = np.argsort(s_te, kind="mergesort")
        ss, ee = s_te[order], e_te[order]
        cum = np.cumsum(ee)
        kk = np.arange(1, N_TEST + 1)
        ok = (cum / kk <= alpha) & (kk >= 25)
        lam = ss[np.nonzero(ok)[0][-1]] if ok.any() else -np.inf
        emit("oracle", s_te <= lam)
    return rows


def _job(args):
    return one_rep(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default="r17_simulation.csv")
    args = ap.parse_args()

    jobs = [(r, name) for name in SETTINGS for r in range(args.reps)]
    log = lambda *p: print("[%s] %s" % (time.strftime("%H:%M:%S"),
                                       " ".join(str(x) for x in p)), flush=True)
    log("%d reps x %d settings" % (args.reps, len(SETTINGS)))
    rows = []
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            for i, res in enumerate(pool.imap(_job, jobs, chunksize=16), 1):
                rows.extend(res)
                if i % 1000 == 0:
                    log("  %d/%d" % (i, len(jobs)))
    else:
        for i, job in enumerate(jobs, 1):
            rows.extend(one_rep(*job))

    df = pd.DataFrame(rows)
    out = os.path.join(paths.RESULTS, args.out)
    df.to_csv(out, index=False)
    sub = df[df["empty"] == 0]
    summ = (df.groupby(["setting", "n_cal", "method", "alpha"])
              .agg(coverage=("covered", "mean"),
                   mean_accept=("accept_rate", "mean"),
                   nonempty=("empty", lambda x: 1.0 - float(x.mean())),
                   mean_risk=("selective_risk", "mean"),
                   reps=("covered", "size")).reset_index())
    nv = sub.groupby(["setting", "n_cal", "method", "alpha"])["covered"].mean().rename(
        "coverage_nonvac").reset_index()
    summ = summ.merge(nv, on=["setting", "n_cal", "method", "alpha"], how="left")
    summ.to_csv(out.replace(".csv", "_summary.csv"), index=False)
    log("wrote", out)
    for name in SETTINGS:
        s = summ[(summ.setting == name) & np.isclose(summ.alpha, 0.20)]
        log("\n[%s]  n_cal=%d  (coverage must be >= %.2f)" % (name, SETTINGS[name][2], 1 - DELTA))
        log(s[["method", "coverage", "coverage_nonvac", "nonempty", "mean_accept"]]
            .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
