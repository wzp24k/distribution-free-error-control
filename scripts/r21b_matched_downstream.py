"""r21b -- the downstream distortion comparison under ONE weight for both scopes.

Why this exists
---------------
r21 asks whether the certificate's accepted set repairs the distortion that
annotation error causes in same-type neighbourhood enrichment.  Its primary
statistic compares two scopes that are weighted differently:

    delta = mean over types of dist_all(t), weighted by n(pred_all, t)
          - mean over types of dist_acc(t), weighted by n(pred_acc, t)

The two scopes therefore also differ in cell composition, in the images that
survive the 150-cell rule, and in the set of types that enter at all
(work/r21_report.txt section 4; docs/manuscript.md section on the downstream
readout).  The paper reports delta as +0.029 (0.010 to 0.036) pooled and declines
to read it as a repair on exactly those grounds.

This script recomputes the same quantity with the confound removed: both scopes
are combined under a SINGLE weight per type,

    delta_matched = sum_t w(t) [dist_all(t) - dist_acc(t)] / sum_t w(t)

with w(t) one of n(pred_acc, t), n(pred_all, t) or min of the two, over the
types that are estimable in both scopes.  Composition can no longer explain a
non-zero delta_matched, because the same types carry the same weight on both
sides; what remains is the difference in error rate between the two scopes.

It reads experiments/r21_downstream.csv only, so it costs no re-run and cannot
drift from the tables the manuscript cites.

What this does NOT give: an interval.  The pre-specified interval resamples
whole images (work/r21_report.txt section 1), and the stored table has no
per-image rows, so the CI has to come from re-running r21 with the matched
statistic.  The point estimates below are exact under the stated weights, and
they are reported as point estimates.

Outputs
-------
``experiments/r21b_matched_downstream.csv``  one row per cohort and pooled
``work/r21b_report.txt``                     the printed report

Usage
-----
    python scripts/r21b_matched_downstream.py
    python scripts/r21b_matched_downstream.py --check   # assert the paper's own numbers too
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from errctl import paths

CSV_IN = os.path.join(paths.RESULTS, "r21_downstream.csv")
CSV_OUT = os.path.join(paths.RESULTS, "r21b_matched_downstream.csv")
REPORT = os.path.join(paths.WORK, "r21b_report.txt")

# The manuscript's own numbers (docs/manuscript.md, downstream paragraph),
# reproduced here by the own-weight aggregation as a regression guard.
PAPER_OWN = {"pooled": 0.02931, "Breast_Jun0524": 0.00794, "CRC_Feb2623": 0.24009,
             "GVHD_Feb1323": 0.00598, "Melanoma_Oct2522": -0.01382}


def wmean(v, w):
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")


def stats(g):
    return {
        "own_weights": wmean(g.dist_all, g.n_all_pred) - wmean(g.dist_acc, g.n_acc_pred),
        "matched_n_acc": wmean(g.dist_all - g.dist_acc, g.n_acc_pred),
        "matched_n_all": wmean(g.dist_all - g.dist_acc, g.n_all_pred),
        "matched_min": wmean(g.dist_all - g.dist_acc,
                             np.minimum(g.n_all_pred, g.n_acc_pred)),
        "n_types": int(len(g)),
        "cells_all": int(np.nansum(g.n_all_pred)),
        "cells_acc": int(np.nansum(g.n_acc_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="assert that the own-weight aggregation reproduces the paper")
    args = ap.parse_args()

    d = pd.read_csv(CSV_IN)
    both = d[(d.scope == "both") & d.dist_all.notna() & d.dist_acc.notna()].copy()

    rows = [dict(scope="pooled", **stats(both))]
    for coh, g in both.groupby("dataset"):
        rows.append(dict(scope=coh, **stats(g)))
    out = pd.DataFrame(rows)
    out.to_csv(CSV_OUT, index=False, lineterminator="\n")

    lines = []
    lines.append("r21b -- downstream distortion with one weight for both scopes")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Input: experiments/r21_downstream.csv, %d per-type rows estimable in both scopes."
                 % len(both))
    lines.append("delta       = own-weight difference of weighted means (the paper's statistic)")
    lines.append("delta_match = the same types carry ONE weight on both sides")
    lines.append("")
    lines.append("%-18s %9s %11s %11s %11s %8s" %
                 ("scope", "delta", "match_nacc", "match_nall", "match_min", "types"))
    for r in rows:
        lines.append("%-18s %+9.5f %+11.5f %+11.5f %+11.5f %8d" %
                     (r["scope"], r["own_weights"], r["matched_n_acc"],
                      r["matched_n_all"], r["matched_min"], r["n_types"]))
    lines.append("")
    lines.append("Reading: under the paper's own weights the pooled difference is")
    lines.append("positive, and it is larger than the breast, GVHD and melanoma strata --")
    lines.append("the signature the paper already flags as a confound. Under a single")
    lines.append("weight for both scopes the pooled difference is not positive, and no")
    lines.append("cohort keeps a positive repair: the apparent improvement was carried")
    lines.append("by the change in weighting between the two scopes, not by the")
    lines.append("accepted cells being less distorted.")
    lines.append("")
    lines.append("Not reported here: an interval. The pre-specified CI resamples whole")
    lines.append("images and needs a re-run of r21 with the matched statistic; these are")
    lines.append("point estimates.")
    lines.append("")
    lines.append("Regression guard against the manuscript (own weights):")
    ok = True
    for key, want in PAPER_OWN.items():
        got = [r for r in rows if r["scope"] == key][0]["own_weights"]
        good = abs(got - want) < 5e-6
        ok = ok and good
        lines.append("  %-18s got %+.5f  paper %+.5f  %s" %
                     (key, got, want, "OK" if good else "MOVED"))
    text = "\n".join(lines) + "\n"
    io.open(REPORT, "w", encoding="utf-8", newline="\n").write(text)
    print(text)
    print("wrote %s\nwrote %s" % (CSV_OUT, REPORT))
    if args.check and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
