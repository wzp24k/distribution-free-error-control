"""Monte-Carlo validation of the finite-sample guarantee.

Data-generating processes
-------------------------
* ``exchangeable``   calibration and deployment cells follow the same law. The
  score is informative, and the error probability varies with it, so the error
  rate among accepted cells is a mixture rather than a constant -- the case in
  which a conditional reading of the binomial bound stops being exact.
* ``shift``          deployment scores follow a different law (known density
  ratio, plus a variant where the ratio is estimated by a domain classifier).
  The unweighted controller has no guarantee here; the weighted one does.
* several calibration sizes, to show how the guarantee and its cost scale.

For every (setting, method, alpha) the script reports the fraction of Monte
Carlo repetitions whose observed selective risk met the budget -- the coverage
of the guarantee, which must be at least ``1 - delta`` -- together with the mean
acceptance rate and the oracle acceptance rate.

Output: experiments/r09_simulation.csv
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

ALPHAS = (0.02, 0.05, 0.10, 0.20)
DELTA = 0.10
N_TEST = 1000

# name -> (calibration law, deployment law, n_cal, error-rate heterogeneity,
#          number of score levels).  Heterogeneity is carried by a covariate the
#          score does not see: the error rate then varies *within* the accepted
#          set, which is exactly the case the conditional binomial shortcut
#          cannot represent.
SETTINGS = {
    "exchangeable_200": ("uniform", "uniform", 200, 0.0, None),
    "exchangeable_500": ("uniform", "uniform", 500, 0.0, None),
    "exchangeable_2000": ("uniform", "uniform", 2000, 0.0, None),
    "hetero_mild": ("uniform", "uniform", 500, 0.4, None),
    "hetero_strong": ("uniform", "uniform", 500, 0.8, None),
    "hetero_coarse": ("uniform", "uniform", 500, 0.8, 5),
    "shift_mild": ("uniform", "beta2", 1000, 0.0, None),
    "shift_strong": ("uniform", "beta3", 1000, 0.0, None),
}


def draw(kind, n, rng):
    if kind == "uniform":
        return rng.random(n)
    if kind == "beta2":                      # density 2 s on [0, 1]
        return rng.beta(2.0, 1.0, size=n)
    if kind == "beta3":                      # density 3 s^2 on [0, 1]
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


def error_prob(s, u=None, het=0.0):
    """Probability that the top-1 label is wrong: small scores are confident.

    ``u`` is a covariate the score never sees, so ``het > 0`` leaves the error
    rate varying among cells that share a score.
    """
    p = 0.02 + 0.40 * s
    if het and u is not None:
        p = p + het * (u - 0.5)
    return np.clip(p, 0.0, 0.99)


def estimated_weight(s_cal, s_te):
    """Density ratio p_target / p_source from a logistic domain classifier."""
    from sklearn.linear_model import LogisticRegression
    x = np.concatenate([s_cal, s_te]).reshape(-1, 1)
    y = np.concatenate([np.zeros(s_cal.size), np.ones(s_te.size)]).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(x, y)
    p = clf.predict_proba(s_cal.reshape(-1, 1))[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    w = p / (1.0 - p) * (s_cal.size / s_te.size)
    return w


def one_rep(rep, name, delta):
    src, tgt, n_cal, het, n_levels = SETTINGS[name]
    rng = np.random.default_rng(7919 * rep + 13 * len(name) + n_cal)
    s_cal = draw(src, n_cal, rng)
    s_te = draw(tgt, N_TEST, rng)
    if n_levels:
        s_cal = np.round(s_cal * (n_levels - 1)) / (n_levels - 1)
        s_te = np.round(s_te * (n_levels - 1)) / (n_levels - 1)
    u_cal = rng.random(n_cal)
    u_te = rng.random(N_TEST)
    e_cal = (rng.random(n_cal) < error_prob(s_cal, u_cal, het)).astype(float)
    e_te = (rng.random(N_TEST) < error_prob(s_te, u_te, het)).astype(float)

    w_known = true_weight(tgt, s_cal) if tgt != src else None
    w_est = estimated_weight(s_cal, s_te) if tgt != src else None

    rows = []
    for alpha in ALPHAS:
        def emit(method, accepted):
            accepted = np.asarray(accepted, dtype=bool)
            n_acc = int(accepted.sum())
            risk = float(e_te[accepted].mean()) if n_acc else np.nan
            rows.append({
                "setting": name, "n_cal": n_cal, "method": method, "alpha": alpha,
                "n_accepted": n_acc,
                "accept_rate": n_acc / float(N_TEST),
                "selective_risk": risk,
                # an empty accepted set is a vacuous success, not a failure
                "covered": int((not np.isfinite(risk)) or risk <= alpha),
            })

        emit("accept_all", np.ones(N_TEST, dtype=bool))
        emit("naive", s_te <= C.naive_threshold(s_cal, e_cal, alpha))
        emit("crc", C.SelectiveRiskController(alpha, delta).fit(s_cal, e_cal).accept(s_te))
        emit("crc_cond",
             C.SelectiveRiskController(alpha, delta, num="cond").fit(s_cal, e_cal).accept(s_te))
        emit("crc_eb",
             C.SelectiveRiskController(alpha, delta, num="eb", bound="bernstein")
             .fit(s_cal, e_cal).accept(s_te))
        emit("crc_hoeffding",
             C.SelectiveRiskController(alpha, delta, num="eb", bound="hoeffding")
             .fit(s_cal, e_cal).accept(s_te))
        if w_known is not None:
            emit("crc_weighted_known",
                 C.SelectiveRiskController(alpha, delta, weighted=True)
                 .fit(s_cal, e_cal, w_known).accept(s_te))
            emit("crc_weighted_est",
                 C.SelectiveRiskController(alpha, delta, weighted=True)
                 .fit(s_cal, e_cal, w_est).accept(s_te))
        order = np.argsort(s_te, kind="mergesort")
        s_sorted, e_sorted = s_te[order], e_te[order]
        cum = np.cumsum(e_sorted)
        k = np.arange(1, N_TEST + 1)
        ok = (cum / k <= alpha) & (k >= 25)
        lam = s_sorted[np.nonzero(ok)[0][-1]] if ok.any() else -np.inf
        emit("oracle", s_te <= lam)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="r09_simulation.csv")
    args = ap.parse_args()

    jobs = [(r, name, DELTA) for name in SETTINGS for r in range(args.reps)]
    log = lambda *p: print("[%s] %s" % (time.strftime("%H:%M:%S"),
                                       " ".join(str(x) for x in p)), flush=True)
    log("%d repetitions x %d settings" % (args.reps, len(SETTINGS)))
    rows = []
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            for i, res in enumerate(pool.imap(_one_rep_job, jobs, chunksize=8), 1):
                rows.extend(res)
                if i % 500 == 0:
                    log("  %d/%d" % (i, len(jobs)))
    else:
        for i, job in enumerate(jobs, 1):
            rows.extend(one_rep(*job))
            if i % 500 == 0:
                log("  %d/%d" % (i, len(jobs)))

    df = pd.DataFrame(rows)
    out = os.path.join(paths.RESULTS, args.out)
    df.to_csv(out, index=False)
    summary = (df.groupby(["setting", "n_cal", "method", "alpha"])
                 .agg(coverage=("covered", "mean"),
                      mean_accept=("accept_rate", "mean"),
                      mean_risk=("selective_risk", "mean"),
                      reps=("covered", "size")).reset_index())
    summary.to_csv(out.replace(".csv", "_summary.csv"), index=False)
    log("wrote", out)
    for name in SETTINGS:
        s = summary[summary.setting == name]
        log("\n[%s]  n_cal=%d  (coverage must be >= %.2f)"
            % (name, SETTINGS[name][2], 1 - DELTA))
        log(s.pivot_table(index="method", columns="alpha",
                          values="coverage").round(3).to_string())


def _one_rep_job(job):
    return one_rep(*job)


if __name__ == "__main__":
    main()


