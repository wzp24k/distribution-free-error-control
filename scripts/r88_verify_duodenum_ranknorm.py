"""Recompute the rank-normalised duodenum summary from its evaluations.

Reads only `experiments/r87_duodenum_ranknorm.csv` and
`experiments/r87_duodenum_ranknorm_summary.csv`.  Re-derives coverage, the
Wilson interval, acceptance and realised risk per classifier, then checks the
values quoted in the manuscript.  Exit status is non-zero on any mismatch.
"""

import csv
import math
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(ROOT, "experiments", "r87_duodenum_ranknorm.csv")
SUMMARY = os.path.join(ROOT, "experiments",
                       "r87_duodenum_ranknorm_summary.csv")


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
        key = (row["arm"], row["clf"], round(f(row, "alpha"), 8),
               int(float(row["k"])))
        grouped.setdefault(key, []).append(row)

    checks = 0
    for row in summary:
        key = (row["arm"], row["clf"], round(f(row, "alpha"), 8),
               int(float(row["k"])))
        units = grouped.get(key, [])
        if not units:
            raise SystemExit("summary row without evaluations: %r" % (key,))
        n = len(units)
        covered = sum(int(float(r["covered"])) for r in units)
        nonempty = [r for r in units if int(float(r["empty"])) == 0]
        coverage = covered / n
        lo, hi = wilson(covered, n)
        close(coverage, f(row, "coverage"), 5e-6, "%r coverage" % (key,))
        close(lo, f(row, "coverage_lo"), 5e-6, "%r coverage_lo" % (key,))
        close(hi, f(row, "coverage_hi"), 5e-6, "%r coverage_hi" % (key,))
        close(len(nonempty) / n, f(row, "frac_nonvacuous"), 5e-6,
              "%r nonvacuous" % (key,))
        close(sum(float(r["accept_rate"]) for r in units) / n,
              f(row, "mean_accept"), 5e-6, "%r accept" % (key,))
        if nonempty:
            risk = sum(float(r["selective_risk"]) for r in nonempty) / len(nonempty)
            close(risk, f(row, "risk_nonvac"), 5e-6, "%r risk" % (key,))
            close(sum(float(r["acc"]) for r in units) / n,
                  f(row, "acc"), 5e-6, "%r acc" % (key,))
        checks += 6

    by = {}
    for row in summary:
        by[(row["arm"], row["clf"], round(f(row, "alpha"), 8))] = row

    pinned = [
        ("cross_random", "ens", 0.10, "coverage", 1.0),
        ("cross_random", "ens", 0.10, "mean_accept", 0.4952),
        ("cross_random", "ens", 0.10, "risk_nonvac", 0.0873),
        ("cross_random", "ens", 0.10, "acc", 0.7396),
        ("cross_random", "rf", 0.10, "mean_accept", 0.4793),
        ("cross_random", "logreg", 0.10, "mean_accept", 0.3393),
        ("cross_random", "hgb", 0.10, "mean_accept", 0.3281),
        ("cross_roi", "ens", 0.10, "coverage", 0.88),
        ("cross_roi", "ens", 0.10, "mean_accept", 0.4530),
        ("srcnull", "ens", 0.10, "coverage", 0.0),
        ("srcnull", "ens", 0.10, "mean_accept", 0.9978),
        ("srcnull", "ens", 0.10, "risk_nonvac", 0.2543),
        ("within_target", "ens", 0.10, "mean_accept", 0.8304),
    ]
    for arm, clf, alpha, field, want in pinned:
        row = by[(arm, clf, alpha)]
        close(f(row, field), want, 5e-4, "%s/%s/%s" % (arm, clf, field))
        checks += 1

    base = {}
    for clf in ("logreg", "rf", "hgb"):
        base[clf] = f(by[("cross_random", clf, 0.10)], "mean_accept")
    mean_accept = sum(base.values()) / 3.0
    close(mean_accept, 0.3822, 5e-4, "base-model mean acceptance")
    checks += 1

    print("r88 rank-normalised duodenum check")
    print("  evaluations: %d" % len(evals))
    print("  summary rows: %d" % len(summary))
    print("  assertions: %d" % checks)
    print("  ensemble: coverage 1.000, acceptance 0.495, risk 0.087")
    print("  base-model mean acceptance: 0.382")
    print("REPRODUCED: rank-normalised cross-panel summary is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())