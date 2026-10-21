"""Recompute the duodenum cross-panel summary from its released evaluations.

The script reads only `experiments/r85_duodenum_crosspanel.csv` and
`experiments/r85_duodenum_crosspanel_summary.csv`. It re-derives coverage,
acceptance, realised risk and the Wilson interval, then checks the values
quoted in the manuscript. Exit status is non-zero on any mismatch.
"""

import csv
import math
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(ROOT, "experiments", "r85_duodenum_crosspanel.csv")
SUMMARY = os.path.join(ROOT, "experiments", "r85_duodenum_crosspanel_summary.csv")


def read(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def f(row, key):
    value = row[key]
    return float(value) if value not in ("", None) else float("nan")


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return centre - half, centre + half


def close(got, want, tol, label):
    if not abs(got - want) <= tol:
        raise SystemExit("%s: got %.12g want %.12g" % (label, got, want))


def main():
    evals = read(EVALS)
    summary = read(SUMMARY)
    grouped = {}
    for row in evals:
        key = (row["arm"], round(f(row, "alpha"), 8), int(float(row["k"])))
        grouped.setdefault(key, []).append(row)

    checks = 0
    for row in summary:
        key = (row["arm"], round(f(row, "alpha"), 8), int(float(row["k"])))
        units = grouped.get(key, [])
        if not units:
            raise SystemExit("summary row without evaluations: %r" % (key,))
        n = len(units)
        covered = sum(int(float(r["covered"])) for r in units)
        nonempty = [r for r in units if int(float(r["empty"])) == 0]
        coverage = covered / n
        accept = sum(f(r, "accept_rate") for r in units) / n
        risk = (sum(f(r, "selective_risk") for r in nonempty) / len(nonempty)
                if nonempty else float("nan"))
        acc = sum(f(r, "acc") for r in units) / n
        lo, hi = wilson(covered, n)
        close(coverage, f(row, "coverage"), 1e-9, "coverage %r" % (key,))
        close(accept, f(row, "mean_accept"), 1e-9, "accept %r" % (key,))
        close(acc, f(row, "acc"), 1e-9, "accuracy %r" % (key,))
        close(lo, f(row, "coverage_lo"), 1e-9, "Wilson lower %r" % (key,))
        close(hi, f(row, "coverage_hi"), 1e-9, "Wilson upper %r" % (key,))
        if nonempty:
            close(risk, f(row, "risk_nonvac"), 1e-9, "risk %r" % (key,))
        checks += 6

    fixed = {
        ("cross_random", 0.10): (1.000000, 0.207282, 0.076755, 150),
        ("cross_random", 0.20): (1.000000, 0.628339, 0.183143, 150),
        ("cross_roi", 0.10): (0.953333, 0.168353, 0.083451, 150),
        ("srcnull", 0.10): (0.000000, 0.988373, 0.311644, 15),
    }
    lookup = {(r["arm"], round(f(r, "alpha"), 8)): r for r in summary}
    for key, (cov, accept, risk, n) in fixed.items():
        row = lookup[key]
        close(f(row, "coverage"), cov, 1e-6, "headline coverage %r" % (key,))
        close(f(row, "mean_accept"), accept, 1e-6,
              "headline acceptance %r" % (key,))
        close(f(row, "risk_nonvac"), risk, 1e-6, "headline risk %r" % (key,))
        close(float(row["n_eval"]), n, 0, "evaluation count %r" % (key,))
        checks += 4

    ladder = {
        (250, 0.000000),
        (500, 0.000000),
        (1000, 0.014646),
        (2000, 0.036070),
        (5000, 0.107033),
        (10000, 0.173129),
    }
    for k, want in sorted(ladder):
        row = next(r for r in summary
                   if r["arm"] == "crossk" and int(float(r["k"])) == k
                   and abs(f(r, "alpha") - 0.10) < 1e-9)
        close(f(row, "mean_accept"), want, 1e-6,
              "label ladder acceptance k=%d" % k)
        checks += 1

    print("r86 duodenum cross-panel check")
    print("  evaluations:", len(evals))
    print("  summary rows:", len(summary))
    print("  assertions:", checks)
    print("  primary: coverage %.3f, acceptance %.3f, risk %.3f"
          % (f(lookup[("cross_random", 0.10)], "coverage"),
             f(lookup[("cross_random", 0.10)], "mean_accept"),
             f(lookup[("cross_random", 0.10)], "risk_nonvac")))
    print("  source-only: coverage %.3f, realised risk %.3f"
          % (f(lookup[("srcnull", 0.10)], "coverage"),
             f(lookup[("srcnull", 0.10)], "risk_nonvac")))
    print("REPRODUCED: duodenum cross-panel summary is consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
