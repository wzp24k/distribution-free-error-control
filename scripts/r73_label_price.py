"""R73 -- the label price of a bounded likelihood ratio.

This is the numerical companion to Proposition 3 in `docs/propositions.md`.
It reads only the released r19 tables, reports the effective-sample price of
the estimated transfer weights, and evaluates the exact two-point lower bound
used in the proof.  It writes `experiments/r73_label_price.csv`.

    python scripts/r73_label_price.py
"""
import csv
import io
import math
import os
import statistics

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")


def load(name):
    with io.open(os.path.join(EXP, name), "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def num(row, col):
    return float(row[col]) if row[col] not in ("", None) else float("nan")


def med(xs):
    return float(statistics.median(list(xs)))


def main():
    w = load("r19_weights.csv")
    e = load("r19_ess.csv")
    ess_frac = [num(r, "ess_pool") / num(r, "n_src_pool") for r in w]
    wp99 = [num(r, "w_p99") for r in w]
    wmax = [num(r, "w_max") for r in w]
    rows = []

    def put(block, quantity, value, detail=""):
        rows.append({"block": block, "quantity": quantity,
                     "value": value, "detail": detail})

    put("weights", "pair_folds", 60)
    put("weights", "ess_fraction_median", med(ess_frac))
    put("weights", "ess_fraction_q10", float(np.quantile(ess_frac, 0.10)))
    put("weights", "ess_fraction_q90", float(np.quantile(ess_frac, 0.90)))
    put("weights", "w_p99_median", med(wp99))
    put("weights", "w_max_median", med(wmax))
    put("weights", "w_max_max", max(wmax))
    put("weights", "implied_second_moment_median", 1.0 / med(ess_frac),
        "1/(ESS/n) under mean-one weights")

    for alpha in (0.05, 0.10, 0.20):
        d = [r for r in e if abs(num(r, "alpha") - alpha) < 5e-8]
        finite = [r for r in d if math.isfinite(num(r, "ess_star_ratio"))]
        put("certificate", "alpha_%.2f_units" % alpha, len(d))
        put("certificate", "alpha_%.2f_nonvacuous" % alpha,
            sum(int(r["nonvacuous"]) for r in d))
        put("certificate", "alpha_%.2f_ess_star_ratio_median" % alpha,
            med(num(r, "ess_star_ratio") for r in finite))
        put("certificate", "alpha_%.2f_ess_obs_median" % alpha,
            med(num(r, "ess_obs") for r in d))
        put("certificate", "alpha_%.2f_rho_star_median" % alpha,
            med(num(r, "rho_star") for r in d))

    for alpha in (0.05, 0.10, 0.20):
        t0 = alpha
        t1 = alpha * 1.5
        kl = t0 * math.log(t0 / t1) + (1 - t0) * math.log((1 - t0) / (1 - t1))
        m = math.log(1 / 0.20) / kl
        put("lower_bound", "alpha_%.2f_bernoulli_kl" % alpha, kl,
            "theta0=alpha, theta1=1.5 alpha")
        put("lower_bound", "alpha_%.2f_informative_labels" % alpha, m,
            "delta=0.10, two-point test")

    out = os.path.join(EXP, "r73_label_price.csv")
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["block", "quantity", "value", "detail"])
        wr.writeheader()
        wr.writerows(rows)

    for r in rows:
        print("%-11s %-38s %12.6f  %s" % (r["block"], r["quantity"],
                                          r["value"], r["detail"]))
    print("wrote", out)


if __name__ == "__main__":
    main()