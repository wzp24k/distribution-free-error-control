"""R57 -- multiplicity and selection accounting for the three families of claims.

Three claims are stated over a family of comparisons and were reported without an
explicit family-wise correction:

  A. coverage, over the 19 exchangeable evaluation units of the benchmark and,
     separately, over the 5 units of the independent STELLAR replication;
  B. repair, over the cohort pairs of the transfer family, where the pairs called
     repairable are selected on the same data;
  C. the marker-logic violation gate, over 68 conventions crossed with the cell
     populations, where the five populations that clear the gate are selected on
     the same data.

This script adds a correction to each: a one-sided Clopper-Pearson limit at the
per-comparison level, the Bonferroni limit at the family level, and a
max-statistic cluster bootstrap that resamples the comparison unit, which is the
level at which the claims are actually made.  For C the correction is applied
twice, once across conventions (a max over conventions needs a union bound over
conventions before the max, or the reported gate is anti-conservative) and once
across the populations that are then selected on the same table.

Nothing is refitted and no model is touched: every number is read from a released
table.

Outputs: experiments/r57_multiplicity.csv,
         experiments/r57_multiplicity_summary.csv,
         work/r57_report.txt
"""

import csv
import io
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "experiments")
WORK = os.path.join(ROOT, "work")
DELTA = 0.10
B = 4000
GATE = 0.05
rng = np.random.default_rng(0)


def log(*p):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in p)), flush=True)


def cpl(x, n, d):
    """One-sided Clopper-Pearson lower limit for x successes in n at level 1-d."""
    x, n = int(x), int(n)
    if n <= 0 or x <= 0:
        return 0.0
    if x >= n:
        return float(d) ** (1.0 / n)
    return float(stats.beta.ppf(d, x, n - x + 1))


def cpu_(x, n, d):
    """One-sided Clopper-Pearson upper limit."""
    x, n = int(x), int(n)
    if n <= 0:
        return 1.0
    if x >= n:
        return 1.0
    return float(stats.beta.ppf(1.0 - d, x + 1, n - x))


rows = []


def add(**kw):
    rows.append(kw)


def coverage_family(path, fam, alpha, norm=None, fam_filter=None):
    m = pd.read_csv(path)
    m = m[(m["kind"] == "cellsplit") & (m["method"] == "crc") & np.isclose(m["alpha"], alpha)]
    if norm is not None:
        m = m[m["norm"] == norm]
    if fam_filter is not None:
        m = m[m["family"] == fam_filter]
    per = m.groupby("unit").agg(n=("covered", "size"), x=("covered", "sum"))
    per["cov"] = per.x / per.n
    mu = int(len(per))
    for u, r in per.iterrows():
        add(part="A", family=fam, unit=str(u), quantity="coverage", value=float(r["cov"]),
            n=int(r.n), x=int(r.x), m=mu, level=DELTA, bound="per_comparison",
            limit=cpl(r.x, r.n, DELTA))
        add(part="A", family=fam, unit=str(u), quantity="coverage", value=float(r["cov"]),
            n=int(r.n), x=int(r.x), m=mu, level=DELTA, bound="bonferroni",
            limit=cpl(r.x, r.n, DELTA / mu))
    covs = per["cov"].to_numpy()
    boot = np.array([covs[rng.integers(0, mu, mu)].min() for _ in range(B)])
    add(part="A", family=fam, unit="*", quantity="min_unit_coverage", value=float(covs.min()),
        n=int(per.n.sum()), x=int(per.x.sum()), m=mu, level=DELTA, bound="bootstrap_max_stat",
        limit=float(np.percentile(boot, 100.0 * DELTA)))
    add(part="A", family=fam, unit="*", quantity="pooled_coverage",
        value=float(per.x.sum() / per.n.sum()), n=int(per.n.sum()), x=int(per.x.sum()),
        m=mu, level=DELTA, bound="per_comparison", limit=cpl(per.x.sum(), per.n.sum(), DELTA))
    add(part="A", family=fam, unit="*", quantity="pooled_coverage",
        value=float(per.x.sum() / per.n.sum()), n=int(per.n.sum()), x=int(per.x.sum()),
        m=int(per.n.sum()), level=DELTA, bound="bonferroni_over_evaluations",
        limit=cpl(per.x.sum(), per.n.sum(), DELTA / max(int(per.n.sum()), 1)))
    return per


def part_b(alpha=0.10, arm="plain", k_arm="1000"):
    """Selection and multiplicity across the cohort pairs of the transfer family.

    The repair analysis asks which source-to-target pairs can be repaired with
    k target labels.  The pairs are selected on the same table the claim is made
    from, so the per-pair interval has to be widened for the family of pairs.
    The unit of the family is the pair-and-fold unit, of which there are 60 (12
    directed pairs x 5 folds), and the quantity is the realised selective risk of
    the accepted cells, for which an exact one-sided Clopper-Pearson bound is
    available from the accepted and wrong counts.
    """
    cols = ["family", "unit", "clf", "alpha", "arm", "k", "rep", "n_accepted", "n_wrong", "accept_rate"]
    m = pd.read_csv(os.path.join(RES, "r31_transfer_detail.csv"), usecols=lambda c: c in cols)
    m = m[(m.family == "transfer") & np.isclose(m.alpha, alpha)]
    m = m[(m.arm == arm) & (m.k.astype(str) == str(k_arm))]
    if len(m) == 0:
        log("part B: no rows for arm", arm, "k", k_arm)
        return
    m = m.assign(pair=m.unit.str.replace(r"_fold\d+$", "", regex=True))
    g = m.groupby("pair").agg(acc=("n_accepted", "sum"), wrong=("n_wrong", "sum"),
                              n=("n_accepted", "size"))
    mu = int(len(g))
    n_cert_pc = n_cert_bon = 0
    for q, r in g.iterrows():
        up_pc = cpu_(r.wrong, r.acc, DELTA)
        up_bon = cpu_(r.wrong, r.acc, DELTA / mu)
        if up_pc <= alpha:
            n_cert_pc += 1
        if up_bon <= alpha:
            n_cert_bon += 1
        add(part="B", family="transfer", unit=str(q), quantity="pair_risk_upper_vs_budget",
            value=float(r.wrong / max(r.acc, 1)), n=int(r.acc), x=int(r.wrong), m=mu,
            level=DELTA, budget=k_arm, bound="per_comparison", limit=up_pc)
        add(part="B", family="transfer", unit=str(q), quantity="pair_risk_upper_vs_budget",
            value=float(r.wrong / max(r.acc, 1)), n=int(r.acc), x=int(r.wrong), m=mu,
            level=DELTA, budget=k_arm, bound="bonferroni", limit=up_bon)
    log("part B: pairs=%d  certified per-comparison=%d  certified bonferroni=%d  (arm=%s k=%s)"
        % (mu, n_cert_pc, n_cert_bon, arm, k_arm))
    # the same question at the whole-family level, pooled over every accepted cell
    acc, wrong = int(g.acc.sum()), int(g.wrong.sum())
    add(part="B", family="transfer", unit="*", quantity="family_pooled_risk_upper",
        value=float(wrong / max(acc, 1)), n=acc, x=wrong, m=mu, level=DELTA, budget=k_arm,
        bound="per_comparison", limit=cpu_(wrong, acc, DELTA))
    add(part="B", family="transfer", unit="*", quantity="family_pooled_risk_upper",
        value=float(wrong / max(acc, 1)), n=acc, x=wrong, m=mu, level=DELTA, budget=k_arm,
        bound="bonferroni_over_pairs", limit=cpu_(wrong, acc, DELTA / mu))
    log("part B pooled: acc=%d wrong=%d rate=%.4f  upper(per-comp)=%.4f upper(bonf)=%.4f"
        % (acc, wrong, wrong / max(acc, 1), cpu_(wrong, acc, DELTA), cpu_(wrong, acc, DELTA / mu)))


def part_c(scope="type", source="expert", alpha_note=""):
    m = pd.read_csv(os.path.join(RES, "r18_conventions.csv"))
    m = m[(m.scope == scope)]
    if "source" in m.columns:
        m = m[m.source == source]
    m = m[m.n > 0]
    m["conv"] = (m["reading"].astype(str) + "|" + m["variant"].astype(str) + "|"
                 + m["positivity"].astype(str))
    n_conv = int(m.conv.nunique())
    key = ["dataset", "cls"]
    cells = []
    for (ds, cls), grp in m.groupby(key):
        r = grp.rate.to_numpy(dtype=float)
        j = int(np.argmin(r))
        cells.append(dict(dataset=ds, cls=cls, n_conv=int(len(grp)),
                          n=int(grp.n.iloc[j]), x=int(grp.n_violation.iloc[j]),
                          min_rate=float(r[j]), argmin_conv=grp.conv.iloc[j],
                          n_over=int(grp.over_gate.sum())))
    mp = int(len(cells))
    for c in cells:
        # union bound over conventions, then Bonferroni over populations
        lim_conv = cpl(c["x"], c["n"], DELTA / max(c["n_conv"], 1))
        lim_pop = cpl(c["x"], c["n"], DELTA / max(c["n_conv"] * mp, 1))
        add(part="C", family="marker_rule", unit="%s|%s" % (c["dataset"], c["cls"]),
            quantity="min_over_conventions_violation", value=c["min_rate"], n=c["n"], x=c["x"],
            m=mp, level=DELTA, bound="per_comparison", limit=cpl(c["x"], c["n"], DELTA))
        add(part="C", family="marker_rule", unit="%s|%s" % (c["dataset"], c["cls"]),
            quantity="min_over_conventions_violation", value=c["min_rate"], n=c["n"], x=c["x"],
            m=mp, level=DELTA, bound="simultaneous_over_conventions", limit=lim_conv)
        add(part="C", family="marker_rule", unit="%s|%s" % (c["dataset"], c["cls"]),
            quantity="min_over_conventions_violation", value=c["min_rate"], n=c["n"], x=c["x"],
            m=mp, level=DELTA, bound="bonferroni_over_conventions_and_populations", limit=lim_pop)
    return cells, n_conv, mp


def main():
    per_b = coverage_family(os.path.join(RES, "r03_metrics.csv"), "cellsplit_exchangeable", 0.10)
    log("part A benchmark: units=%d min_cov=%.4f pooled=%.4f" % (
        len(per_b), per_b["cov"].min(), per_b.x.sum() / per_b.n.sum()))
    per_x = coverage_family(os.path.join(RES, "r55_stellar.csv"), "stellar_exchangeable", 0.10,
                            norm="log1p_z", fam_filter="ext_cellsplit")
    log("part A stellar: units=%d min_cov=%.4f pooled=%.4f" % (
        len(per_x), per_x["cov"].min(), per_x.x.sum() / per_x.n.sum()))
    part_b()
    cells, n_conv, mp = part_c()
    log("part C: populations=%d conventions=%d" % (mp, n_conv))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "r57_multiplicity.csv"), index=False)
    rep = (df.groupby(["part", "family", "quantity", "bound"])
             .agg(n_units=("unit", "nunique"), value=("value", "mean"), limit=("limit", "min"))
             .round(4))
    rep.to_csv(os.path.join(RES, "r57_multiplicity_summary.csv"))
    log(rep.to_string())
    out = io.open(os.path.join(WORK, "r57_report.txt"), "w", encoding="utf-8", newline="\n")
    out.write("R57 multiplicity and selection accounting\n")
    out.write("delta = %.2f, bootstrap B = %d, gate = %.2f\n\n" % (DELTA, B, GATE))
    out.write(rep.to_string())
    out.write("\n\nPart C, populations that clear the %.0f%% gate under every convention:\n" % (100 * GATE))
    for c in sorted(cells, key=lambda c: c["min_rate"], reverse=True):
        lim_conv = cpl(c["x"], c["n"], DELTA / max(c["n_conv"], 1))
        lim_pop = cpl(c["x"], c["n"], DELTA / max(c["n_conv"] * mp, 1))
        out.write("  %-22s %-16s min=%.4f  conv-sim=%.4f  conv+pop-sim=%.4f  raw_flagged=%s  corrected_flagged=%s\n" % (
            c["dataset"], c["cls"], c["min_rate"], lim_conv, lim_pop,
            cpl(c["x"], c["n"], DELTA) > GATE, lim_pop > GATE))
    n_raw = sum(1 for c in cells if cpl(c["x"], c["n"], DELTA) > GATE)
    n_conv = sum(1 for c in cells if cpl(c["x"], c["n"], DELTA / max(c["n_conv"], 1)) > GATE)
    n_pop = sum(1 for c in cells if cpl(c["x"], c["n"], DELTA / max(c["n_conv"] * mp, 1)) > GATE)
    out.write("\npopulations clearing the gate: raw %d, over conventions %d, over conventions and populations %d, of %d\n"
              % (n_raw, n_conv, n_pop, mp))
    log("part C gates: raw=%d conv-sim=%d conv+pop-sim=%d of %d" % (n_raw, n_conv, n_pop, mp))
    out.close()
    log("wrote experiments/r57_multiplicity{,_summary}.csv and work/r57_report.txt")


if __name__ == "__main__":
    main()
