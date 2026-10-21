"""R83 -- independent re-derivation of the IMMUcan V1 replication.

Reads the released evaluation rows, recomputes the four reported endpoints and
checks the cohort counts against the exported cell table. Exits non-zero on any
mismatch.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

EVALS = os.path.join(ROOT, "experiments", "r82_immucan_v1_evals.csv")
SUMMARY = os.path.join(ROOT, "experiments", "r82_immucan_v1_summary.csv")
CELLS = os.path.join(ROOT, "data", "external3", "immucan_v1_cells.csv")

ALPHAS = (0.05, 0.10, 0.20)
FAILS = []


def check(name, got, want, tol=1e-9):
    ok = abs(float(got) - float(want)) <= tol
    print(("OK  " if ok else "FAIL") + f" {name}: got={got} want={want}")
    if not ok:
        FAILS.append(name)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    df = pd.read_csv(EVALS)
    sm = pd.read_csv(SUMMARY)
    for alpha in ALPHAS:
        g = df[df.alpha == alpha]
        row = sm[np.isclose(sm.alpha, alpha)].iloc[0]
        k = int(g.covered.sum())
        n = len(g)
        lo, hi = wilson(k, n)
        check(f"alpha {alpha:.2f} n", n, row.n_eval, 0)
        check(f"alpha {alpha:.2f} coverage", k / n, row.coverage, 1e-12)
        check(f"alpha {alpha:.2f} coverage_lo", lo, row.coverage_lo, 1e-12)
        check(f"alpha {alpha:.2f} nonvacuous",
              float((g["empty"] == 0).mean()), row.frac_nonvacuous, 1e-12)
        check(f"alpha {alpha:.2f} acceptance",
              float(g.accept_rate.mean()), row.mean_accept, 1e-12)
        nv = g[g["empty"] == 0]
        check(f"alpha {alpha:.2f} risk",
              float(nv.selective_risk.mean()), row.risk_nonvac, 1e-12)

    primary = sm[np.isclose(sm.alpha, 0.10)].iloc[0]
    ok = (primary.coverage >= 0.90 and primary.coverage_lo >= 0.85
          and primary.frac_nonvacuous >= 0.80 and primary.mean_accept >= 0.50)
    print(f"PRIMARY {'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILS.append("primary")

    cells = pd.read_csv(CELLS, low_memory=False)
    check("samples", cells.sample_id.nunique(), 149, 0)
    check("labels", len(cells), 290226, 0)
    check("markers", cells.shape[1] - 3, 37, 0)

    print(f"FAILS={len(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
