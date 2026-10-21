"""R13 -- cluster-bootstrap confidence intervals for the headline coverage.

Coverage in r03/r04 is a mean over (unit, repeat) evaluations, and units that
share a cohort and a fold are not independent.  The interval below resamples
whole units, which is the conservative choice.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from errctl import paths

B = 2000
METHODS = ("crc", "crc_cond", "crc_eb", "crc_mondrian", "conformal_set",
           "naive", "oracle", "crc_temp", "temp_naive")


def ci(values, rng, B=B):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    idx = rng.integers(0, values.size, size=(B, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    m = pd.read_csv(os.path.join(paths.RESULTS, "r03_metrics.csv"))
    m = m.rename(columns={"kind": "family"})
    rng = np.random.default_rng(12345)
    rows = []
    for (fam, meth, alpha), g in m.groupby(["family", "method", "alpha"]):
        if meth not in METHODS:
            continue
        per_unit = g.groupby("unit")["covered"].mean()
        lo, hi = ci(per_unit.values, rng)
        acc = g.groupby("unit")["accept_rate"].mean()
        lo_a, hi_a = ci(acc.values, rng)
        rows.append({"family": fam, "method": meth, "alpha": float(alpha),
                     "n_units": int(per_unit.size),
                     "coverage": float(per_unit.mean()),
                     "cov_lo": lo, "cov_hi": hi,
                     "accept": float(acc.mean()),
                     "accept_lo": lo_a, "accept_hi": hi_a})
    out = pd.DataFrame(rows).sort_values(["family", "method", "alpha"])
    p = os.path.join(paths.RESULTS, "r13_coverage_ci.csv")
    out.to_csv(p, index=False)
    print("wrote %s (%d rows)" % (p, len(out)))
    sel = out[np.isclose(out["alpha"], 0.10) & out.method.isin(["crc", "naive", "oracle"])]
    print(sel[["family", "method", "n_units", "coverage", "cov_lo", "cov_hi",
               "accept"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()