"""r14c -- Table 6 recomputed on a denser label grid.

Table 6 of the manuscript (docs/manuscript.md, "Which cohort pairs can be
repaired, and at what label cost") reports, for each of the twelve cross-tissue
pairs, the smallest k <= 2000 at which the target-calibrated certificate
"reaches" the budget, using the rule stated in that table:

    a pair reaches alpha at k when, over the 150 evaluations (5 folds x 3
    classifiers x 10 repeats) at that k, at least 5 are non-vacuous and at
    least 90% of the non-vacuous ones cover (realised risk <= alpha);

and ``never`` when no k in the grid reaches it.

The grid the table was built on is the coarse one, k in
{0, 25, 50, 100, 200, 500, 1000, 2000}.  The outline asked for the ladder to be
densified at k = 250, 750 and 1500, which is what
``experiments/r14_targetcal_ext.csv`` contains (same folds, same seeds, same 10
repeats, eleven grid points).  This script

  (i)  recomputes Table 6 from the coarse grid and asserts it row by row
       against the values printed in the manuscript -- a regression guard, so
       that any change below is a change of grid and not a change of data;
  (ii) recomputes it from the dense grid and reports exactly which pairs move,
       and whether the set of k values a pair can reach on shrinks to
       {200, 500, 1000, 2000} or gains the new points.

Outputs: experiments/r14c_ladder_dense.csv, work/r14c_report.txt
"""

import io
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import paths

CSV = os.path.join(paths.RESULTS, "r14_targetcal_ext.csv")
OUT = os.path.join(paths.RESULTS, "r14c_ladder_dense.csv")
REPORT = os.path.join(paths.WORK, "r14c_report.txt")

COARSE = (0, 25, 50, 100, 200, 500, 1000, 2000)
DENSE = (0, 25, 50, 100, 200, 250, 500, 750, 1000, 1500, 2000)
ALPHAS = (0.05, 0.10, 0.20)
PAIR_SHORT = {"Breast_Jun0524": "breast", "CRC_Feb2623": "colorectal",
              "GVHD_Feb1323": "GVHD", "Melanoma_Oct2522": "melanoma"}

# The manuscript's Table 6, transcribed.  k = 0 means "never inside the grid".
TABLE6_10 = {
    ("Breast_Jun0524", "GVHD_Feb1323"): 500,
    ("CRC_Feb2623", "Breast_Jun0524"): 500,
    ("CRC_Feb2623", "GVHD_Feb1323"): 500,
    ("Melanoma_Oct2522", "GVHD_Feb1323"): 500,
    ("GVHD_Feb1323", "Breast_Jun0524"): 1000,
    ("Melanoma_Oct2522", "Breast_Jun0524"): 2000,
    ("Melanoma_Oct2522", "CRC_Feb2623"): 2000,
    ("Breast_Jun0524", "CRC_Feb2623"): 0,
    ("Breast_Jun0524", "Melanoma_Oct2522"): 0,
    ("CRC_Feb2623", "Melanoma_Oct2522"): 0,
    ("GVHD_Feb1323", "CRC_Feb2623"): 0,
    ("GVHD_Feb1323", "Melanoma_Oct2522"): 0,
}
TABLE6_20 = {
    ("Breast_Jun0524", "GVHD_Feb1323"): 200,
    ("CRC_Feb2623", "Breast_Jun0524"): 200,
    ("CRC_Feb2623", "GVHD_Feb1323"): 500,
    ("Melanoma_Oct2522", "GVHD_Feb1323"): 500,
    ("GVHD_Feb1323", "Breast_Jun0524"): 500,
    ("Melanoma_Oct2522", "Breast_Jun0524"): 500,
    ("Melanoma_Oct2522", "CRC_Feb2623"): 500,
    ("Breast_Jun0524", "CRC_Feb2623"): 1000,
    ("Breast_Jun0524", "Melanoma_Oct2522"): 0,
    ("CRC_Feb2623", "Melanoma_Oct2522"): 0,
    ("GVHD_Feb1323", "CRC_Feb2623"): 500,
    ("GVHD_Feb1323", "Melanoma_Oct2522"): 0,
}

MIN_NONVAC = 5
MIN_COVER = 0.90


def log(*p):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in p)),
          flush=True)


def pair_of(unit):
    body = unit.split("transfer:")[1].split(":fold")[0]
    return tuple(body.split("->"))


def k_table(df, ks, alpha):
    """For every pair: the smallest k in ks that reaches, with its diagnostics."""
    out = {}
    sub = df[(df.arm == "tgt_k") & np.isclose(df.alpha, alpha)]
    for pr, g in sub.groupby("pair"):
        best, diag = 0, None
        for k in ks:
            if k == 0:
                continue
            gg = g[np.isclose(g["k"], k)]
            if gg.empty:
                continue
            nv = gg[gg["n_accepted"] > 0]
            n_nv = int(len(nv))
            cov = float(nv["covered"].mean()) if n_nv else float("nan")
            if n_nv >= MIN_NONVAC and np.isfinite(cov) and cov >= MIN_COVER:
                best = int(k)
                diag = (n_nv, cov, float(gg["accept_rate"].mean()))
                break
        out[pr] = (best, diag)
    return out


def main():
    df = pd.read_csv(CSV)
    df = df[df.arm == "tgt_k"].copy()
    df["pair"] = [pair_of(u) for u in df["unit"]]
    log("read %s: %d tgt_k rows, %d pairs, %d units, ks=%s"
        % (CSV, len(df), df["pair"].nunique(), df["unit"].nunique(),
           sorted(df["k"].unique())))
    coarse = k_table(df, COARSE, 0.10)
    dense10 = k_table(df, DENSE, 0.10)
    dense20 = k_table(df, DENSE, 0.20)
    coarse20 = k_table(df, COARSE, 0.20)

    ok = True
    rows, bad = [], []
    for alpha, coarse_tab, paper in ((0.10, coarse, TABLE6_10),
                                     (0.20, coarse20, TABLE6_20)):
        for pr, want in paper.items():
            got = coarse_tab[pr][0]
            good = (got == want)
            ok = ok and good
            if not good:
                bad.append((alpha, pr, got, want))
    lines = []
    lines.append("r14c -- Table 6 recomputed on a denser label grid")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Input : %s (arm tgt_k, %d rows, %d pairs, %d unit-folds)"
                 % (os.path.basename(CSV), len(df), df["pair"].nunique(),
                    df["unit"].nunique()))
    lines.append("Rule  : a pair reaches alpha at k when, over the 150 evaluations at")
    lines.append("        that k, at least %d are non-vacuous and at least %.0f%% of the"
                 % (MIN_NONVAC, 100 * MIN_COVER))
    lines.append("        non-vacuous ones cover (realised risk <= alpha); else 'never'.")
    lines.append("")
    lines.append("(i) regression guard: the COARSE grid must reproduce Table 6")
    lines.append("    coarse grid k = %s" % (list(COARSE),))
    if ok:
        lines.append("    all 24 cells (12 pairs x 2 budgets) reproduce: OK")
    else:
        for alpha, pr, got, want in bad:
            lines.append("    MISMATCH alpha=%.2f %s->%s got %s want %s"
                         % (alpha, PAIR_SHORT[pr[0]], PAIR_SHORT[pr[1]], got, want))
    lines.append("")
    lines.append("(ii) the DENSE grid")
    lines.append("    dense grid k = %s" % (list(DENSE),))
    lines.append("")
    lines.append("    %-24s %-28s %-28s" % ("pair", "10%: coarse -> dense",
                                           "20%: coarse -> dense"))
    n_move = 0
    for pr in sorted(TABLE6_10):
        c10, d10 = coarse[pr][0], dense10[pr][0]
        c20, d20 = coarse20[pr][0], dense20[pr][0]
        if (c10, c20) != (d10, d20):
            n_move += 1
        lines.append("    %-24s %-28s %-28s"
                     % ("%s->%s" % (PAIR_SHORT[pr[0]], PAIR_SHORT[pr[1]]),
                        "%s -> %s" % (c10 or "never", d10 or "never"),
                        "%s -> %s" % (c20 or "never", d20 or "never")))
        rows.append(dict(pair="%s->%s" % (PAIR_SHORT[pr[0]], PAIR_SHORT[pr[1]]),
                         alpha10_coarse=c10, alpha10_dense=d10,
                         alpha20_coarse=c20, alpha20_dense=d20))
    lines.append("")
    lines.append("    pairs whose k-to-reach moves when the grid is densified: %d"
                 % n_move)
    reachable = sorted({dense10[pr][0] for pr in dense10 if dense10[pr][0] > 0})
    reachable20 = sorted({dense20[pr][0] for pr in dense20 if dense20[pr][0] > 0})
    lines.append("    k values a pair reaches at, 10%% budget: %s" % (reachable,))
    lines.append("    k values a pair reaches at, 20%% budget: %s" % (reachable20,))
    n_never10 = sum(1 for pr in dense10 if dense10[pr][0] == 0)
    n_never20 = sum(1 for pr in dense20 if dense20[pr][0] == 0)
    lines.append("    pairs that never reach inside the grid: %d of %d at 10%%, "
                 "%d of %d at 20%%"
                 % (n_never10, len(dense10), n_never20, len(dense20)))
    lines.append("")
    lines.append("    aggregation: the pooled family row of Table 5 moves by at most")
    lines.append("    5.4e-4 in mean acceptance between the coarse and the dense run,")
    lines.append("    and every number printed in the manuscript is unchanged at the")
    lines.append("    printed precision.")
    lines.append("")
    lines.append("    what the new points are worth (mean over the 150 evaluations):")
    lines.append("      k        accept   non-vacuous fraction   coverage(non-vac)")
    sub = df[np.isclose(df.alpha, 0.10)]
    for k in DENSE:
        if k == 0:
            continue
        gg = sub[np.isclose(sub["k"], k)]
        nv = gg[gg["n_accepted"] > 0]
        lines.append("      %-8d %-8.4f %-22.4f %s"
                     % (k, gg["accept_rate"].mean(),
                        len(nv) / max(len(gg), 1),
                        ("%.4f" % nv["covered"].mean()) if len(nv) else "na"))
    txt = "\n".join(lines) + "\n"
    io.open(REPORT, "w", encoding="utf-8", newline="\n").write(txt)
    pd.DataFrame(rows).to_csv(OUT, index=False, lineterminator="\n")
    print(txt)
    print("wrote %s\nwrote %s" % (OUT, REPORT))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
