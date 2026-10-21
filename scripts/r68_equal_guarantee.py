# -*- coding: utf-8 -*-
"""r68 -- acceptance at a matched guarantee.

The efficiency question is whether the certificate costs throughput.  Asked at a
matched *realised* risk the answer is that it does, because a rule that does not
hold the budget may spend realised risk freely.  Asked at a matched *guarantee*
the answer is the reverse, because the quantity a guarantee delivers is not a
point estimate but the fraction of evaluation units on which a fixed risk target
is met, together with how far past the target the units that miss it go.

Input   experiments/r03_metrics.csv, one row per evaluation unit, classifier,
        repeat, budget and method, carrying n_test_eval, n_accepted,
        n_wrong_accepted, selective_risk, covered and empty.
Output  experiments/r68_equal_guarantee.csv and work/r68_report.txt.

For a fixed risk target epsilon the script forms, per method and per budget on
the grid, the unit level coverage of epsilon

    coverage(alpha) = mean over units of  selective_risk(unit, alpha) <= epsilon

counting an empty accepted set as covered, the mean acceptance, the worst unit
level risk over the units, and its 95th percentile.  It then reports

    accept_max(c) = the largest acceptance over budgets whose coverage of epsilon
                    is at least c

for c in 1.000, 0.975, 0.950 and 0.900, carrying the worst unit risk at that
matched operating point.  Read as, "to be inside a risk target of epsilon on at
least this fraction of units, how many cells may this arm keep, and how badly
does the worst unit miss".
"""
import io
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "experiments", "r03_metrics.csv")
OUT = os.path.join(ROOT, "experiments", "r68_equal_guarantee.csv")
REP = os.path.join(ROOT, "work", "r68_report.txt")

EPS = [0.05, 0.10, 0.20]
TARGETS = [1.000, 0.975, 0.950, 0.900]
FAMILY = {"cellsplit": "random split", "within": "cross field of view",
          "transfer": "cross tissue transfer"}
LABEL = {"crc": "certificate", "crc_cond": "certificate conditional",
         "crc_mondrian": "certificate Mondrian", "crc_temp": "certificate temperature",
         "crc_eb": "certificate empirical Bernstein", "crc_hoeffding": "certificate Hoeffding",
         "crc_weighted": "certificate weighted", "conformal_set": "conformal prediction set",
         "naive": "uncalibrated threshold", "temp_naive": "temperature scaled threshold",
         "oracle": "oracle", "accept_all": "accept every cell"}
METHODS = ("crc", "crc_cond", "crc_mondrian", "crc_temp", "conformal_set", "naive",
           "temp_naive", "crc_eb", "crc_hoeffding", "crc_weighted", "oracle")


def operating_points(df):
    rows = []
    for (kind, method, alpha), g in df.groupby(["kind", "method", "alpha"]):
        risk_all = g["selective_risk"].fillna(0.0).values
        nv = g[g["empty"] == 0]
        for eps in EPS:
            met_all = (g["selective_risk"] <= eps) | (g["empty"] == 1)
            met_nv = nv["selective_risk"] <= eps
            rows.append(dict(
                family=kind, method=method, alpha=float(alpha), eps=eps,
                accept_mean=float(g["accept_rate"].mean()),
                cov_all=float(met_all.mean()),
                cov_nonvac=float(met_nv.mean()) if len(nv) else float("nan"),
                empty_frac=float((g["empty"] == 1).mean()),
                worst_unit_risk=float(np.nanmax(risk_all)),
                p95_unit_risk=float(np.nanpercentile(risk_all, 95)),
                n_units=int(g["unit"].nunique())))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(SRC)
    op = operating_points(df)
    out = []
    for (family, method, eps), g in op.groupby(["family", "method", "eps"]):
        g = g.sort_values("alpha")
        for conv, col in (("all", "cov_all"), ("non vacuous", "cov_nonvac")):
            for t in TARGETS:
                m = g[g[col] >= t - 1e-12]
                if not len(m):
                    out.append(dict(family=family, method=method, eps=eps, convention=conv,
                                    target_coverage=t, feasible=0, alpha=float("nan"),
                                    accept=float("nan"), coverage=float("nan"),
                                    worst_unit_risk=float("nan"), p95_unit_risk=float("nan"),
                                    empty_frac=float("nan")))
                    continue
                b = m.loc[m.accept_mean.idxmax()]
                out.append(dict(family=family, method=method, eps=eps, convention=conv,
                                target_coverage=t, feasible=1, alpha=float(b["alpha"]),
                                accept=float(b["accept_mean"]), coverage=float(b[col]),
                                worst_unit_risk=float(b["worst_unit_risk"]),
                                p95_unit_risk=float(b["p95_unit_risk"]),
                                empty_frac=float(b["empty_frac"])))
    res = pd.DataFrame(out)
    res.to_csv(OUT, index=False, float_format="%.6f")

    lines = ["r68 -- acceptance at a matched guarantee", "",
             "accept_max(c) is the largest mean acceptance over the budget grid 0.02..0.50",
             "whose unit level coverage of a fixed risk target epsilon is at least c, with",
             "the worst unit level selective risk at that operating point.",
             "Source: experiments/r03_metrics.csv (%d rows, %d units)."
             % (len(df), df["unit"].nunique()), "",
             "The matched realised risk table of r16 answers a different question and is",
             "reported separately; that one lets an arm spend realised risk freely.", ""]
    for conv in ("all", "non vacuous"):
        lines.append("coverage convention: %s" % conv)
        for fam in ("cellsplit", "within", "transfer"):
            for eps in EPS:
                lines.append("  %s, risk target %.2f" % (FAMILY[fam], eps))
                lines.append("    %-32s %s" % ("target coverage",
                                               "  ".join("%-15s" % ("cov>=%.3f" % t)
                                                         for t in TARGETS)))
                lines.append("    %-32s %s" % ("method",
                                               "  ".join("%-15s" % "accept|worst"
                                                         for _ in TARGETS)))
                for method in ("crc", "crc_cond", "conformal_set", "naive",
                               "temp_naive", "crc_eb", "oracle"):
                    s = res[(res.family == fam) & (res.convention == conv) & (res.eps == eps)
                            & (res.method == method)].set_index("target_coverage")
                    if not len(s):
                        continue
                    cells = []
                    for t in TARGETS:
                        r = s.loc[t]
                        cells.append("%7.3f|%-6.3f" % (r["accept"], r["worst_unit_risk"])
                                     if r["feasible"] else "      -|      -")
                    lines.append("    %-32s %s" % (LABEL.get(method, method), " ".join(cells)))
                lines.append("")
        lines.append("")
    lines.append("columns are  acceptance | worst unit risk  at target coverage "
                 + " / ".join("%.3f" % t for t in TARGETS))
    io.open(REP, "w", encoding="utf-8", newline="\n").write("\n".join(lines))

    for fam in ("cellsplit", "within"):
        print("\n%s" % FAMILY[fam])
        for eps in EPS:
            print("  risk target %.2f" % eps)
            print("    %-28s %s" % ("", "  ".join("%-15s" % ("cov>=%.3f" % t) for t in TARGETS)))
            for method in ("crc", "crc_cond", "conformal_set", "naive", "oracle"):
                s = res[(res.family == fam) & (res.convention == "all") & (res.eps == eps)
                        & (res.method == method)].set_index("target_coverage")
                if not len(s):
                    continue
                cells = []
                for t in TARGETS:
                    r = s.loc[t]
                    cells.append("%.3f/%.3f" % (r["accept"], r["worst_unit_risk"])
                                 if r["feasible"] else "      -     ")
                print("    %-28s %s" % (LABEL.get(method, method), "  ".join("%-15s" % c for c in cells)))
    print("\nwrote", OUT, "and", REP)


main()
