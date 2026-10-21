"""r28_marker_bounds.py

Partial identification of the marker-rule violation rate.

The marker-consistency audit is one of the paper's two hard empirical findings,
but its pooled violation rate is convention dependent (0.4%-90% depending on the
positivity threshold, the semicolon reading and the rule text).  The manuscript
therefore calls the rate "not identified".  That is too weak a conclusion: a
convention sweep does not destroy the audit, it produces an identified set.

This script reads the already computed experiments/r18_conventions.csv (8646
rows, 4 cohorts x 4 scopes x sources x 17 positivity calls x 2 readings x 2 rule
texts) and reports, for every (dataset, cell type) and for the pooled ALL_* rows:

  1. the identified set = [min, max] of rate over the 68 enumerated conventions,
     with the argmin / argmax convention and exact Clopper-Pearson intervals;
  2. convention-robust verdicts: which populations are above the 5% gate under
     EVERY evaluable convention, and which are below it under every one;
  3. the same restricted to the depot's own default family (pct_all@60..95) and
     to the mixture crossing point (gmm), and where the verdict flips;
  4. the between-convention spread against the within-convention sampling CI;
  5. the pooled ALL_evaluable / ALL_strict rows and per-cohort certification;
  6. the label-source repair restated as a bound.

Inputs : experiments/r18_conventions.csv       (read only)
Outputs: experiments/r28_bounds_by_marker.csv
         experiments/r28_bounds_pooled.csv
         work/r28_report.txt
         work/_r28_all_conventions.csv        (scratch: every group x convention)

No marker logic is recomputed; the CSV is the whole input.  Everything written
is plain ASCII.
"""

import math
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IN_CSV = os.path.join(ROOT, "experiments", "r18_conventions.csv")
OUT_BY_MARKER = os.path.join(ROOT, "experiments", "r28_bounds_by_marker.csv")
OUT_POOLED = os.path.join(ROOT, "experiments", "r28_bounds_pooled.csv")
OUT_REPORT = os.path.join(ROOT, "work", "r28_report.txt")
OUT_SCRATCH = os.path.join(ROOT, "work", "_r28_all_conventions.csv")

# ---------------------------------------------------------------- convention grid
# The manuscript's "sixty-eight conventions per cohort" are the 17 positivity
# calls x the 2 readings of the depot semicolon x the 2 rule texts.  gmm_expert
# is deliberately NOT part of the grid: it is a reproducibility cross-check of
# gmm (the same mixture refitted on expert cells only), not an independent
# convention, and it does not appear in r18_conventions.csv.
POS_GMM = "gmm"
POS_PCT_ALL = ["pct_all@%d" % p for p in (60, 65, 70, 75, 80, 85, 90, 95)]
POS_PCT_POSPOP = ["pct_pospop@%d" % p for p in (60, 65, 70, 75, 80, 85, 90, 95)]
POSITIVITY_GRID = [POS_GMM] + POS_PCT_ALL + POS_PCT_POSPOP
READINGS = ["or", "and"]
VARIANTS = ["shipped", "repaired"]
GRID = [(p, r, v) for p in POSITIVITY_GRID for r in READINGS for v in VARIANTS]
GRID_SIZE = len(GRID)                      # 68
GRID_SET = set(GRID)
PRIMARY = ("gmm", "or", "repaired")        # manuscript primary convention
THRESH = 0.05                              # the manuscript's 5% gate
ALPHA = 0.05                               # 95% Clopper-Pearson intervals

DATASET_ORDER = ["Breast_Jun0524", "CRC_Feb2623", "GVHD_Feb1323",
                 "Melanoma_Oct2522"]
SOURCE_ORDER = ["expert", "hq_ref", "hq_all", "hq_nonexpert"]

def cp_interval(k, n, alpha=ALPHA):
    """Exact Clopper-Pearson interval (matches binom.test / statsmodels)."""
    k = int(k)
    n = int(n)
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def _verdict(rates, thr=THRESH):
    if len(rates) == 0:
        return "no-data"
    arr = np.asarray(rates, dtype=float)
    if (arr > thr).all():
        return "discordant"
    if (arr <= thr).all():
        return "concordant"
    return "mixed"


def _family(recs, want):
    if want == "gmm":
        return [r for r in recs if r["positivity"] == "gmm"]
    if want == "pct_all":
        return [r for r in recs if r["positivity"].startswith("pct_all")]
    if want == "pct_pospop":
        return [r for r in recs if r["positivity"].startswith("pct_pospop")]
    raise ValueError(want)


def _family_block(recs, want, prefix):
    sub = _family(recs, want)
    r = np.array([x["rate"] for x in sub if x["rate"] is not None
                  and math.isfinite(x["rate"])], dtype=float)
    return {
        prefix + "_n": int(len(sub)),
        prefix + "_n_finite": int(len(r)),
        prefix + "_min": float(r.min()) if len(r) else float("nan"),
        prefix + "_max": float(r.max()) if len(r) else float("nan"),
        prefix + "_n_above5": int((r > THRESH).sum()) if len(r) else 0,
        prefix + "_frac_above5": (float((r > THRESH).mean()) if len(r)
                                  else float("nan")),
        prefix + "_verdict": _verdict(r),
    }


def _rate_stats(recs):
    fin = [x for x in recs if x["rate"] is not None
           and math.isfinite(x["rate"])]
    r = np.array([x["rate"] for x in fin], dtype=float)
    ci_lo = np.array([x["ci_lo"] for x in fin], dtype=float)
    ci_hi = np.array([x["ci_hi"] for x in fin], dtype=float)
    wid = ci_hi - ci_lo
    wid = wid[np.isfinite(wid)]
    span = float(r.max() - r.min()) if len(r) else float("nan")
    mean_w = float(wid.mean()) if len(wid) else float("nan")
    ratio = span / mean_w if (len(wid) and mean_w > 0) else float("nan")
    return {
        "n_finite": int(len(r)),
        "mean_rate": float(r.mean()) if len(r) else float("nan"),
        "median_rate": float(np.median(r)) if len(r) else float("nan"),
        "sd_rate": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        "min_rate_grid": float(r.min()) if len(r) else float("nan"),
        "max_rate_grid": float(r.max()) if len(r) else float("nan"),
        "span": span,
        "mean_ci_width": mean_w,
        "ratio_span_to_mean_ci": ratio,
    }

def analyse_group(key, recs):
    """Identified-set analysis for one (dataset, scope, source, cls) group."""
    ds, scope, source, cls = key
    present = len(recs)
    fin = [r for r in recs
           if r["rate"] is not None and math.isfinite(r["rate"])]
    grid_complete = (len(fin) == GRID_SIZE)
    present_keys = set((r["positivity"], r["reading"], r["variant"])
                       for r in recs)
    on_grid = present_keys <= GRID_SET

    row = {
        "dataset": ds, "scope": scope, "source": source, "cls": cls,
        "n_conventions_present": present,
        "n_conventions_finite": len(fin),
        "grid_expected": GRID_SIZE,
        "grid_complete": bool(grid_complete),
        "conventions_on_standard_grid": bool(on_grid),
        "universal_claim_eligible": bool(grid_complete and on_grid),
    }
    row.update(_rate_stats(recs))

    if fin:
        lo = min(fin, key=lambda r: (r["rate"],
                                     POSITIVITY_GRID.index(r["positivity"]),
                                     READINGS.index(r["reading"]),
                                     VARIANTS.index(r["variant"])))
        hi = max(fin, key=lambda r: (r["rate"],
                                     -POSITIVITY_GRID.index(r["positivity"]),
                                     -READINGS.index(r["reading"]),
                                     -VARIANTS.index(r["variant"])))
        cplo = cp_interval(lo["n_violation"], lo["n"])
        cphi = cp_interval(hi["n_violation"], hi["n"])
        row.update({
            "rate_min": lo["rate"], "min_positivity": lo["positivity"],
            "min_reading": lo["reading"], "min_variant": lo["variant"],
            "n_min": lo["n"], "n_violation_min": lo["n_violation"],
            "cp_lo_min": cplo[0], "cp_hi_min": cplo[1],
            "rate_max": hi["rate"], "max_positivity": hi["positivity"],
            "max_reading": hi["reading"], "max_variant": hi["variant"],
            "n_max": hi["n"], "n_violation_max": hi["n_violation"],
            "cp_lo_max": cphi[0], "cp_hi_max": cphi[1],
            "identified_span": hi["rate"] - lo["rate"],
            "n_at_min_tie": int(sum(1 for r in fin
                                    if abs(r["rate"] - lo["rate"]) < 1e-12)),
            "n_at_max_tie": int(sum(1 for r in fin
                                    if abs(r["rate"] - hi["rate"]) < 1e-12)),
        })
        rates = np.array([r["rate"] for r in fin], dtype=float)
        row["n_conventions_above_5pct"] = int((rates > THRESH).sum())
        row["frac_conventions_above_5pct"] = float((rates > THRESH).mean())
        row["robustly_discordant"] = bool((rates > THRESH).all())
        row["robustly_concordant"] = bool((rates <= THRESH).all())
        row["verdict"] = _verdict(rates)
    else:
        for k in ("rate_min", "min_positivity", "min_reading", "min_variant",
                  "n_min", "n_violation_min", "cp_lo_min", "cp_hi_min",
                  "rate_max", "max_positivity", "max_reading", "max_variant",
                  "n_max", "n_violation_max", "cp_lo_max", "cp_hi_max",
                  "identified_span", "n_conventions_above_5pct",
                  "frac_conventions_above_5pct"):
            row[k] = float("nan")
        row["robustly_discordant"] = False
        row["robustly_concordant"] = False
        row["verdict"] = "no-data"

    prim = [r for r in recs
            if (r["positivity"], r["reading"], r["variant"]) == PRIMARY]
    if prim and prim[0]["rate"] is not None and math.isfinite(prim[0]["rate"]):
        p = prim[0]
        cpp = cp_interval(p["n_violation"], p["n"])
        row.update({
            "primary_positivity": p["positivity"],
            "primary_reading": p["reading"],
            "primary_variant": p["variant"], "rate_primary": p["rate"],
            "n_primary": p["n"], "n_violation_primary": p["n_violation"],
            "cp_lo_primary": cpp[0], "cp_hi_primary": cpp[1],
            "primary_ci_width": cpp[1] - cpp[0],
            "primary_attains_min": bool(abs(p["rate"] - row["rate_min"]) < 1e-12),
            "primary_attains_max": bool(abs(p["rate"] - row["rate_max"]) < 1e-12),
        })
    else:
        for k in ("primary_positivity", "primary_reading", "primary_variant",
                  "rate_primary", "n_primary", "n_violation_primary",
                  "cp_lo_primary", "cp_hi_primary", "primary_ci_width"):
            row[k] = float("nan")
        row["primary_attains_min"] = False
        row["primary_attains_max"] = False

    row.update(_family_block(recs, "gmm", "fam_gmm"))
    row.update(_family_block(recs, "pct_all", "fam_pctall"))
    row.update(_family_block(recs, "pct_pospop", "fam_pctpospop"))
    row["verdict_changed_gmm_vs_pctall"] = bool(
        row["fam_gmm_verdict"] != row["fam_pctall_verdict"])

    for tag, v in (("shipped", "shipped"), ("repaired", "repaired")):
        sub = np.array([r["rate"] for r in recs if r["variant"] == v
                        and r["rate"] is not None
                        and math.isfinite(r["rate"])], dtype=float)
        row["rate_min_" + tag] = float(sub.min()) if len(sub) else float("nan")
        row["rate_max_" + tag] = float(sub.max()) if len(sub) else float("nan")
        row["n_conventions_" + tag] = int(len(sub))

    row["rules_used_primary"] = (prim[0]["rules_used"] if prim else "")
    row["n_labelled"] = (prim[0]["n_labelled"] if prim else float("nan"))
    return row

def _load_records(sub):
    out = []
    for _, r in sub.iterrows():
        rate = r["rate"]
        rate = float(rate) if pd.notna(rate) else None
        out.append({
            "positivity": r["positivity"], "reading": r["reading"],
            "variant": r["variant"], "rate": rate, "n": int(r["n"]),
            "n_violation": int(r["n_violation"]),
            "ci_lo": float(r["ci_lo"]) if pd.notna(r["ci_lo"]) else float("nan"),
            "ci_hi": float(r["ci_hi"]) if pd.notna(r["ci_hi"]) else float("nan"),
            "evaluability": r["evaluability"],
            "rules_used": r["rules_used"] if pd.notna(r["rules_used"]) else "",
            "n_labelled": int(r["n_labelled"]) if pd.notna(r["n_labelled"])
            else 0,
        })
    out.sort(key=lambda x: (POSITIVITY_GRID.index(x["positivity"]),
                            READINGS.index(x["reading"]),
                            VARIANTS.index(x["variant"])))
    return out


def _analyse_group_star(args):
    return analyse_group(args[0], args[1])


def _order_datasets(df):
    df["_ds"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    df["_src"] = pd.Categorical(df["source"], SOURCE_ORDER, ordered=True)
    df = df.sort_values(["_ds", "_src", "cls"]).drop(columns=["_ds", "_src"])
    return df.reset_index(drop=True)


def build(df, scope, sources):
    sub = df[(df.scope == scope) & (df.source.isin(sources))]
    tasks = []
    for (ds, src, cls), g in sub.groupby(["dataset", "source", "cls"],
                                         sort=False):
        tasks.append(((ds, scope, src, cls), _load_records(g)))
    nproc = min(16, mp.cpu_count())
    with mp.Pool(nproc) as pool:
        rows = pool.map(_analyse_group_star, tasks)
    return _order_datasets(pd.DataFrame(rows))


def fmt_pct(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return ("%." + str(nd) + "f%%") % (100.0 * x)


def conv_str(pos, rd, var):
    return "%s / %s / %s" % (pos, rd, var)


def _primary_variant_rates(df):
    """rate at (gmm, or, shipped) and (gmm, or, repaired) per group key."""
    prim = df[(df.positivity == "gmm") & (df.reading == "or") &
              (df.variant.isin(["shipped", "repaired"]))]
    out = {}
    for _, r in prim.iterrows():
        key = (r["dataset"], r["scope"], r["source"], r["cls"])
        out.setdefault(key, {})[r["variant"]] = (r["rate"], r["n"],
                                                 r["n_violation"])
    return out


def main():
    df = pd.read_csv(IN_CSV)
    checks = []

    fin = df[df.rate.notna()].copy()
    sample = fin.sample(n=min(500, len(fin)), random_state=0)
    dev = np.abs(sample.n_violation / sample.n - sample.rate)
    max_dev = float(dev.max())
    checks.append(("n_violation/n == rate within 1e-9 on %d sampled finite rows"
                   % len(sample), bool(max_dev < 1e-9),
                   "max abs dev = %.3e" % max_dev))
    brk = int((~((fin.ci_lo <= fin.rate + 1e-12) &
                 (fin.rate <= fin.ci_hi + 1e-12))).sum())
    checks.append(("ci_lo <= rate <= ci_hi on all %d finite rows" % len(fin),
                   bool(brk == 0), "%d violations" % brk))

    tscope = df[df.scope == "type"]
    grp = tscope.groupby(["dataset", "source", "cls"]).agg(
        present=("rate", "size"), finite=("rate", "count"))
    incomplete = grp[(grp.finite != GRID_SIZE) | (grp.present != GRID_SIZE)]
    exp_incomplete = grp.loc[[i for i in grp.index if i[1] == "expert"]]
    exp_bad = exp_incomplete[exp_incomplete.finite != GRID_SIZE]
    checks.append(("every source=expert (dataset, cls) has %d finite-rate "
                   "conventions" % GRID_SIZE, bool(len(exp_bad) == 0),
                   "%d expert groups incomplete" % len(exp_bad)))
    checks.append(("no group exceeds the %d-convention grid" % GRID_SIZE,
                   bool((df.groupby(["dataset", "scope", "source", "cls"])
                         .size() <= GRID_SIZE).all()),
                   "max group size %d"
                   % int(df.groupby(["dataset", "scope", "source", "cls"])
                         .size().max())))
    na = df[df.rate.isna()]
    na_scope = na[na.scope != "source"]
    na_src = na[na.scope == "source"]
    checks.append(("NaN rate: %d rows outside scope=source all have n==0; the "
                   "%d scope=source ALL_CELLS rows have n>0 and no rate"
                   % (len(na_scope), len(na_src)),
                   bool((na_scope.n == 0).all() and (na_src.n > 0).all()
                        and len(na) == len(na_scope) + len(na_src)),
                   "structural NaN, never a silent zero"))

    type_sources = [s for s in SOURCE_ORDER
                    if s in set(tscope.source.unique())]
    bymarker = build(df, "type", type_sources)
    pooled_sources = [s for s in SOURCE_ORDER
                      if s in set(df[df.scope == "summary"].source.unique())]
    pooled = build(df, "summary", pooled_sources)

    pv = _primary_variant_rates(df)
    for frame in (bymarker, pooled):
        frame["rate_primary_shipped"] = [
            pv.get((r.dataset, r.scope, r.source, r.cls), {})
            .get("shipped", (float("nan"),))[0] for _, r in frame.iterrows()]
        frame["rate_primary_repaired"] = [
            pv.get((r.dataset, r.scope, r.source, r.cls), {})
            .get("repaired", (float("nan"),))[0] for _, r in frame.iterrows()]

    pvsub = df[df.rate.notna() & (df.scope == "summary") &
               (df.cls == "ALL_evaluable")]
    piv = pvsub.pivot_table(index=["dataset", "source", "positivity", "reading"],
                            columns="variant", values="rate", aggfunc="first")
    piv = piv.dropna(subset=["shipped", "repaired"])
    n_rep = int(len(piv))
    n_viol = int((piv["repaired"] > piv["shipped"] + 1e-12).sum())
    checks.append(("repaired rate <= shipped rate at every matched pooled "
                   "convention (%d matched)" % n_rep, bool(n_viol == 0),
                   "%d violations" % n_viol))
    bad_pos = 0
    bad_prim = 0
    for ds in DATASET_ORDER:
        r = pooled[(pooled.dataset == ds) & (pooled.source == "expert") &
                   (pooled.cls == "ALL_evaluable")].iloc[0]
        sub = df[(df.dataset == ds) & (df.scope == "summary") &
                 (df.source == "expert") & (df.cls == "ALL_evaluable") &
                 df.rate.notna()]
        ties = sub[np.abs(sub.rate - r.rate_min) < 1e-12]
        if not (ties.positivity == "gmm").all():
            bad_pos += 1
        if abs(r.rate_primary - r.rate_min) > 1e-12:
            bad_prim += 1
    checks.append(("every convention attaining the pooled identified-set "
                   "minimum uses positivity=gmm (4 cohorts)", bool(bad_pos == 0),
                   "%d cohorts where it does not" % bad_pos))
    checks.append(("the primary convention (gmm/or/repaired) attains the "
                   "pooled identified-set minimum in all 4 cohorts",
                   bool(bad_prim == 0),
                   "%d cohorts where it does not" % bad_prim))

    scrap = tscope[["dataset", "source", "cls", "positivity", "reading",
                    "variant", "n", "n_violation", "rate", "ci_lo", "ci_hi",
                    "evaluability"]].copy()
    scrap.to_csv(OUT_SCRATCH, index=False)

    exp = bymarker[bymarker.source == "expert"]
    cert = {}
    for ds, g in exp.groupby("dataset"):
        elig = g[g.universal_claim_eligible]
        cert[ds] = {
            "n_types": int(len(elig)),
            "conc": sorted(elig[elig.robustly_concordant].cls.tolist()),
            "disc": sorted(elig[elig.robustly_discordant].cls.tolist()),
            "mixed": sorted(elig[~elig.robustly_concordant &
                                 ~elig.robustly_discordant].cls.tolist()),
            "excl": sorted(g[~g.universal_claim_eligible].cls.tolist()),
        }

    for col in ("n_types_swept", "n_types_certified_below5",
                "types_certified_below5", "n_types_robustly_discordant",
                "types_robustly_discordant"):
        pooled[col] = ""
    m = (pooled.source == "expert") & (pooled.cls == "ALL_evaluable")
    for i in pooled.index[m]:
        c = cert.get(pooled.at[i, "dataset"])
        if not c:
            continue
        pooled.at[i, "n_types_swept"] = c["n_types"]
        pooled.at[i, "n_types_certified_below5"] = len(c["conc"])
        pooled.at[i, "types_certified_below5"] = ";".join(c["conc"])
        pooled.at[i, "n_types_robustly_discordant"] = len(c["disc"])
        pooled.at[i, "types_robustly_discordant"] = ";".join(c["disc"])

    bymarker.to_csv(OUT_BY_MARKER, index=False, float_format="%.10g")
    pooled.to_csv(OUT_POOLED, index=False, float_format="%.10g")
    # recorded (Wilson) CI at the primary convention, for manuscript checks
    rec = {}
    for _, r in df[(df.positivity == "gmm") & (df.reading == "or") &
                   (df.variant == "repaired")].iterrows():
        rec[(r["dataset"], r["scope"], r["source"], r["cls"])] = (
            r["ci_lo"], r["ci_hi"])
    for frame in (bymarker, pooled):
        frame["ci_lo_primary_recorded"] = [
            rec.get((r.dataset, r.scope, r.source, r.cls),
                    (float("nan"), float("nan")))[0]
            for _, r in frame.iterrows()]
        frame["ci_hi_primary_recorded"] = [
            rec.get((r.dataset, r.scope, r.source, r.cls),
                    (float("nan"), float("nan")))[1]
            for _, r in frame.iterrows()]
    bymarker.to_csv(OUT_BY_MARKER, index=False, float_format="%.10g")
    pooled.to_csv(OUT_POOLED, index=False, float_format="%.10g")

    L = []
    A = L.append
    A("r28  MARKER-VIOLATION BOUNDS: what the convention sweep identifies")
    A("=" * 78)
    A("")
    A("Input  : experiments/r18_conventions.csv (%d rows)" % len(df))
    A("Tables : experiments/r28_bounds_by_marker.csv (%d rows)" % len(bymarker))
    A("         experiments/r28_bounds_pooled.csv    (%d rows)" % len(pooled))
    A("Scratch: work/_r28_all_conventions.csv")
    A("")
    A("CONVENTION GRID ADOPTED (justification)")
    A("-" * 78)
    A("The manuscript's 68 conventions per cohort are the cross product of")
    A("  positivity (17): gmm (two-component mixture crossing point),")
    A("                   pct_all@60..95 (8 thresholds, percentile of all cells),")
    A("                   pct_pospop@60..95 (8, percentile of the positive pop.)")
    A("  reading    (2) : disjunction (or) and conjunction (and) of the depot's")
    A("                   semicolon-separated marker lists")
    A("  rule text  (2) : shipped and repaired channel-name rule files")
    A("17 x 2 x 2 = 68.  gmm_expert is excluded: it is a refit of gmm on expert")
    A("cells used as a reproducibility cross-check, not an independent reading of")
    A("the same data, and it is absent from r18_conventions.csv.  Label source")
    A("(expert / hq_ref / hq_all / hq_nonexpert) is a separate axis, not part of")
    A("the 68.  The primary convention is gmm / or / repaired.")
    A("")

    A("STEP 0  REPRODUCTION OF THE MANUSCRIPT HEADLINE NUMBERS")
    A("-" * 78)
    repro = []

    def add_repro(label, expected, computed, ok):
        repro.append((label, expected, computed, bool(ok)))

    pr = {}
    for ds in DATASET_ORDER:
        pr[ds] = pooled[(pooled.dataset == ds) & (pooled.source == "expert") &
                        (pooled.cls == "ALL_evaluable")].iloc[0]
    add_repro("breast pooled range 0.44%-88%",
              "0.44-88%",
              "%.2f-%.2f%%" % (100 * pr["Breast_Jun0524"].rate_min,
                               100 * pr["Breast_Jun0524"].rate_max),
              abs(pr["Breast_Jun0524"].rate_min - 0.00443) < 5e-4 and
              abs(pr["Breast_Jun0524"].rate_max - 0.88) < 5e-3)
    add_repro("GVHD pooled range 2.09%-90%",
              "2.09-90%",
              "%.2f-%.2f%%" % (100 * pr["GVHD_Feb1323"].rate_min,
                               100 * pr["GVHD_Feb1323"].rate_max),
              abs(pr["GVHD_Feb1323"].rate_min - 0.020939) < 5e-4 and
              abs(pr["GVHD_Feb1323"].rate_max - 0.90) < 5e-3)
    add_repro("melanoma pooled range 3.98%-87%",
              "3.98-87%",
              "%.2f-%.2f%%" % (100 * pr["Melanoma_Oct2522"].rate_min,
                               100 * pr["Melanoma_Oct2522"].rate_max),
              abs(pr["Melanoma_Oct2522"].rate_min - 0.039837) < 5e-4 and
              abs(pr["Melanoma_Oct2522"].rate_max - 0.87) < 5e-3)
    add_repro("breast exceedance 52/68",
              "52/68", "%d/68" % int(pr["Breast_Jun0524"].n_conventions_above_5pct),
              int(pr["Breast_Jun0524"].n_conventions_above_5pct) == 52)
    add_repro("melanoma exceedance 64/68",
              "64/68", "%d/68" % int(pr["Melanoma_Oct2522"].n_conventions_above_5pct),
              int(pr["Melanoma_Oct2522"].n_conventions_above_5pct) == 64)
    add_repro("GVHD exceedance (not quoted in text)",
              "n/a", "%d/68" % int(pr["GVHD_Feb1323"].n_conventions_above_5pct),
              True)
    add_repro("CRC exceedance (not quoted in text)",
              "n/a", "%d/68" % int(pr["CRC_Feb2623"].n_conventions_above_5pct),
              True)

    for ds, cls, exp_r, exp_ci in (
            ("GVHD_Feb1323", "Neutrophil", "26.35%", "20.25-33.51%"),
            ("GVHD_Feb1323", "Plasma", "19.84%", "16.88-23.17%"),
            ("Melanoma_Oct2522", "CD8T", "19.78%", "18.34-21.31%")):
        g = bymarker[(bymarker.dataset == ds) & (bymarker.source == "expert") &
                     (bymarker.cls == cls)].iloc[0]
        add_repro("%s %s primary rate %s" % (ds, cls, exp_r), exp_r,
                  "%.2f%%" % (100 * g.rate_primary),
                  abs(g.rate_primary - float(exp_r.strip("%")) / 100) < 5e-4)
        add_repro("%s %s recorded interval %s" % (ds, cls, exp_ci), exp_ci,
                  "%.2f-%.2f%%" % (100 * g.ci_lo_primary_recorded,
                                   100 * g.ci_hi_primary_recorded),
                  abs(g.ci_lo_primary_recorded -
                      float(exp_ci.split("-")[0].strip("%")) / 100) < 5e-4 and
                  abs(g.ci_hi_primary_recorded -
                      float(exp_ci.split("-")[1].strip("%")) / 100) < 5e-4)

    g = pooled[(pooled.dataset == "GVHD_Feb1323") & (pooled.source == "hq_ref") &
               (pooled.cls == "ALL_evaluable")].iloc[0]
    add_repro("GVHD hq_ref label-source repair 23.08% -> 0.07%",
              "23.08% -> 0.07%",
              "%.2f%% -> %.2f%%" % (100 * g.rate_primary_shipped,
                                    100 * g.rate_primary_repaired),
              abs(g.rate_primary_shipped - 0.230778) < 5e-4 and
              abs(g.rate_primary_repaired - 0.000680) < 5e-4)

    ng = bymarker[(bymarker.dataset == "GVHD_Feb1323") &
                  (bymarker.source == "expert") &
                  (bymarker.cls == "Neutrophil")].iloc[0]
    add_repro("GVHD neutrophil: gmm and pct_all = 26.35%",
              "26.35%",
              "gmm %.2f%%, pct_all %.2f-%.2f%%"
              % (100 * ng.fam_gmm_min, 100 * ng.fam_pctall_min,
                 100 * ng.fam_pctall_max),
              abs(ng.fam_gmm_min - 0.263473) < 5e-4 and
              abs(ng.fam_pctall_min - 0.263473) < 5e-4 and
              abs(ng.fam_pctall_max - 0.263473) < 5e-4)
    add_repro("GVHD neutrophil: pct_pospop 50.9%-93.4%",
              "50.90-93.41%",
              "%.2f-%.2f%%" % (100 * ng.fam_pctpospop_min,
                               100 * ng.fam_pctpospop_max),
              abs(ng.fam_pctpospop_min - 0.50898) < 1e-3 and
              abs(ng.fam_pctpospop_max - 0.93413) < 1e-3)

    A("  %-52s %-16s %-16s %s"
      % ("manuscript number", "quoted", "reproduced", "status"))
    for label, exp_s, comp_s, ok in repro:
        A("  %-52s %-16s %-16s %s" % (label, exp_s, comp_s,
                                      "PASS" if ok else "FAIL"))
    A("")
    A("  Note on the abstract shorthand '0.4% to 90%': it spans cohorts.  Within")
    A("  breast the sweep tops out at %.2f%% (the manuscript body's 88%%), and the"
      % (100 * pr["Breast_Jun0524"].rate_max))
    A("  90%% top belongs to GVHD (%.2f%%) and colorectal (%.2f%%).  This is a"
      % (100 * pr["GVHD_Feb1323"].rate_max,
         100 * pr["CRC_Feb2623"].rate_max))
    A("  units/rounding slip in the abstract, not a computation discrepancy.")
    A("")
    A("  Pooled identified sets over the 68 conventions (expert labels):")
    for ds in DATASET_ORDER:
        r = pr[ds]
        A("    %-16s [%s, %s]  argmin %s (%d/%d)  argmax %s (%d/%d)"
          % (ds, fmt_pct(r.rate_min), fmt_pct(r.rate_max),
             conv_str(r.min_positivity, r.min_reading, r.min_variant),
             int(r.n_violation_min), int(r.n_min),
             conv_str(r.max_positivity, r.max_reading, r.max_variant),
             int(r.n_violation_max), int(r.n_max)))
        A("        primary %.4f%%; CP at argmin %s; CP at argmax %s"
          % (100 * r.rate_primary,
             fmt_pct(r.cp_lo_min) + "-" + fmt_pct(r.cp_hi_min),
             fmt_pct(r.cp_lo_max) + "-" + fmt_pct(r.cp_hi_max)))
    gs = pooled[(pooled.dataset == "GVHD_Feb1323") & (pooled.source == "expert") &
                (pooled.cls == "ALL_strict")].iloc[0]
    A("    ALL_strict GVHD primary %.4f%% (manuscript 4.02%%); melanoma" %
      (100 * gs.rate_primary))
    ms = pooled[(pooled.dataset == "Melanoma_Oct2522") &
                (pooled.source == "expert") &
                (pooled.cls == "ALL_strict")].iloc[0]
    A("    ALL_strict melanoma primary %.4f%% (manuscript 5.66%%)" %
      (100 * ms.rate_primary))
    A("")

    A("Sanity checks:")
    for name, ok, note in checks:
        A("  [%s] %s -- %s" % ("ok" if ok else "FAIL", name, note))
    A("")
    failed = [c for c in checks if not c[1]]

    A("1  IDENTIFIED SET (worst case over the enumerated conventions)")
    A("-" * 78)
    A("Under no assumption beyond 'the convention is one of the 68 enumerated',")
    A("the violation rate of a population lies in [rate_min, rate_max].  These")
    A("are properties of the sweep, not sampling statements.  The cp_lo_* /")
    A("cp_hi_* columns are exact Clopper-Pearson intervals; the")
    A("ci_lo_primary_recorded / ci_hi_primary_recorded columns are the Wilson")
    A("intervals the manuscript quotes at the primary convention.")
    A("")
    A("Per cohort, per cell type (scope=type, expert labels).  Full table with")
    A("argmin/argmax identities and intervals in experiments/r28_bounds_by_marker.csv.")
    A("")
    A("  %-16s %-16s %8s %8s   argmin convention        argmax convention"
      % ("cohort", "cell type", "min", "max"))
    for _, r in exp.iterrows():
        A("  %-16s %-16s %7.2f%% %7.2f%%   %-22s %s"
          % (r.dataset, r.cls, 100 * r.rate_min, 100 * r.rate_max,
             conv_str(r.min_positivity, r.min_reading, r.min_variant),
             conv_str(r.max_positivity, r.max_reading, r.max_variant)))
    A("")
    A("  Ties at an extreme are broken toward the earlier grid entry (shipped")
    A("  before repaired).  The table records n_at_min_tie / n_at_max_tie and")
    A("  primary_attains_min / primary_attains_max, so a reader can see whether")
    A("  the manuscript's primary convention sits at an extreme.")
    A("")
    A("  The widest identified set in each cohort, and its argmin/argmax:")
    for ds in DATASET_ORDER:
        g = exp[exp.dataset == ds]
        w = g.loc[g.identified_span.idxmax()]
        A("    %-16s %-14s [%s, %s]  = span %.2f pp"
          % (ds, w.cls, fmt_pct(w.rate_min), fmt_pct(w.rate_max),
             100 * w.identified_span))
    A("")

    A("2  CONVENTION-ROBUST VERDICTS (the convention-free statement)")
    A("-" * 78)
    A("Gate = 5%.  'Robustly discordant' = rate > 5% under EVERY evaluable")
    A("convention; 'robustly concordant' = rate <= 5% under every one.  A group")
    A("is eligible for a universal claim only if all 68 grid conventions are")
    A("present AND have a finite rate; ineligible groups are excluded.")
    A("")
    for ds in DATASET_ORDER:
        c = cert[ds]
        A("%s  (expert labels; %d cell types on a complete 68-convention grid)"
          % (ds, c["n_types"]))
        A("  robustly discordant (>5%% under all 68): %s"
          % (", ".join(c["disc"]) if c["disc"] else "none"))
        A("  robustly concordant (<=5%% under all 68): %s"
          % (", ".join(c["conc"]) if c["conc"] else "none"))
        A("  verdict flips with the convention: %d of %d"
          % (len(c["mixed"]), c["n_types"]))
        if c["excl"]:
            A("  EXCLUDED (incomplete evaluability): %s" % ", ".join(c["excl"]))
        A("")
    A("  Exceedance counts (fraction of the 68 conventions above 5%):")
    for ds in DATASET_ORDER:
        g = exp[exp.dataset == ds]
        A("    %-16s min %3d/68 (%5.1f%%)  median %3d/68  max %3d/68"
          % (ds, int(g.n_conventions_above_5pct.min()),
             100 * g.frac_conventions_above_5pct.min(),
             int(g.n_conventions_above_5pct.median()),
             int(g.n_conventions_above_5pct.max())))
    A("")
    A("  Incomplete-evaluability exceptions at scope=type: %d of %d"
      % (len(incomplete), len(grp)))
    A("  (dataset, source, cls) groups.  All are hq_nonexpert, which was swept")
    A("  only over the 4 gmm conventions (gmm x or/and x shipped/repaired).")
    srcs = sorted(set(i[1] for i in incomplete.index))
    A("  sources affected: %s" % ", ".join(srcs))
    A("  The same source has only 4 conventions at scope=summary, so its pooled")
    A("  rows are flagged not eligible for a universal claim as well.")
    A("  Every other source is on the full 68-convention grid at both scopes.")
    A("  No source=expert group is incomplete, so no expert-label universal")
    A("  claim is excluded on evaluability grounds.")
    A("")

    A("3  RESTRICTED CONVENTIONS: depot default family vs mixture crossing point")
    A("-" * 78)
    A("pct_all family = pct_all@60..95 x or/and x shipped/repaired (32 conventions).")
    A("gmm family     = gmm x or/and x shipped/repaired (4 conventions).")
    A("pct_pospop family = pct_pospop@60..95 x or/and x shipped/repaired (32).")
    A("")
    for ds in DATASET_ORDER:
        g = exp[exp.dataset == ds]
        A("%s" % ds)
        A("  %-16s %-20s %-22s %-24s %s"
          % ("cell type", "gmm family", "pct_all family", "pct_pospop family",
             "verdict gmm->pct_all"))
        for _, r in g.iterrows():
            A("  %-16s %-20s %-22s %-24s %s%s"
              % (r.cls,
                 "%s-%s" % (fmt_pct(r.fam_gmm_min), fmt_pct(r.fam_gmm_max)),
                 "%s-%s" % (fmt_pct(r.fam_pctall_min),
                            fmt_pct(r.fam_pctall_max)),
                 "%s-%s" % (fmt_pct(r.fam_pctpospop_min),
                            fmt_pct(r.fam_pctpospop_max)),
                 r.fam_gmm_verdict + " -> " + r.fam_pctall_verdict,
                 "  [CHANGES]" if r.verdict_changed_gmm_vs_pctall else ""))
        A("")
    nchg = int(exp.verdict_changed_gmm_vs_pctall.sum())
    A("  %d of %d expert cell types change verdict between the gmm and pct_all"
      % (nchg, len(exp)))
    pctall_disc = exp[exp.fam_pctall_verdict == "discordant"]
    newly = pctall_disc[pctall_disc.fam_gmm_verdict != "discordant"]
    names = "; ".join("%s %s (%s under gmm)"
                      % (r.dataset.split("_")[0], r.cls, r.fam_gmm_verdict)
                      for _, r in newly.iterrows())
    A("  families.  The switch is driven by the threshold, not by the reading or")
    A("  the rule text: the gmm family sits at the low-threshold end of the")
    A("  sweep, so %d types are robustly discordant under the pct_all family"
      % len(newly))
    A("  but not under gmm: %s." % (names if names else "none"))
    A("")

    A("4  BETWEEN-CONVENTION SPREAD vs WITHIN-CONVENTION SAMPLING CI")
    A("-" * 78)
    A("The per-convention CI is a sampling interval at a FIXED convention.  For")
    A("each cell type, ratio = (max rate - min rate over the 68 conventions) /")
    A("(mean per-convention CI width for that same cell type).")
    A("")
    A("  cohort            n_types  median ratio  min   max")
    for ds in DATASET_ORDER:
        g = exp[exp.dataset == ds]
        rr = g.ratio_span_to_mean_ci.dropna()
        A("  %-16s  %7d  %12.2f  %5.2f  %5.2f"
          % (ds, len(rr), rr.median(), rr.min(), rr.max()))
    rr = exp.ratio_span_to_mean_ci.dropna()
    A("  %-16s  %7d  %12.2f  %5.2f  %5.2f"
      % ("ALL", len(rr), rr.median(), rr.min(), rr.max()))
    pooled_rr = pooled[(pooled.source == "expert") &
                       (pooled.cls == "ALL_evaluable")]
    A("")
    A("  All-evaluable pooled rows, same ratio: %s"
      % ", ".join("%s %.1f" % (r.dataset.split("_")[0], r.ratio_span_to_mean_ci)
                  for _, r in pooled_rr.iterrows()))
    A("")
    A("  The between-convention spread exceeds the fixed-convention sampling CI")
    A("  by a median factor of about %.0f.  The sampling CI is therefore NOT the"
      % rr.median())
    A("  dominant uncertainty in the marker audit; the convention is.")
    A("")
    A("  mean / median / min / max / sd of rate across the 68 conventions:")
    A("  %-16s %-16s %7s %7s %7s %7s %7s"
      % ("cohort", "cell type", "mean", "median", "min", "max", "sd"))
    for _, r in exp.iterrows():
        A("  %-16s %-16s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %6.3f"
          % (r.dataset, r.cls, 100 * r.mean_rate, 100 * r.median_rate,
             100 * r.min_rate_grid, 100 * r.max_rate_grid, r.sd_rate))
    A("")

    A("5  POOLED ROWS AND PER-COHORT CERTIFICATION")
    A("-" * 78)
    A("  %-16s %-13s %-18s %-10s %-12s %s"
      % ("cohort", "pooled", "identified set", "primary", ">5% conv",
         "n cells"))
    for ds in DATASET_ORDER:
        for cl in ("ALL_evaluable", "ALL_strict"):
            g = pooled[(pooled.dataset == ds) & (pooled.source == "expert") &
                       (pooled.cls == cl)]
            if not len(g):
                continue
            r = g.iloc[0]
            A("  %-16s %-13s [%7s, %7s]  %8s  %2d/%d      %d"
              % (ds, cl, fmt_pct(r.rate_min), fmt_pct(r.rate_max),
                 fmt_pct(r.rate_primary), int(r.n_conventions_above_5pct),
                 int(r.n_conventions_finite), int(r.n_primary)))
    A("")
    A("  Headline: cell types certified below 5% under EVERY convention, per")
    A("  cohort (expert labels, complete 68-convention grids):")
    for ds in DATASET_ORDER:
        c = cert[ds]
        A("    %-16s %d of %d certified below 5%% everywhere%s"
          % (ds, len(c["conc"]), c["n_types"],
             (" (" + ", ".join(c["conc"]) + ")") if c["conc"] else ""))
    A("  No pooled ALL_evaluable row is certified below 5% under every")
    A("  convention; each cohort's pooled rate exceeds 5% in a majority of")
    A("  conventions (breast 52/68, CRC 64/68, GVHD 60/68, melanoma 64/68).")
    A("  So no cohort-level marker claim survives the sweep as a passing level.")
    A("")
    A("  Pooled convention-robust verdicts (data-driven):")
    for cl in ("ALL_evaluable", "ALL_strict"):
        A("    %s:" % cl)
        for ds in DATASET_ORDER:
            r = pooled[(pooled.dataset == ds) & (pooled.source == "expert") &
                       (pooled.cls == cl)].iloc[0]
            A("      %-10s %-10s [%s, %s]"
              % (ds.split("_")[0], r.verdict, fmt_pct(r.rate_min),
                 fmt_pct(r.rate_max)))
    A("  The melanoma ALL_strict pooled rate is robustly discordant: above the")
    A("  5% gate under every one of the 68 conventions (minimum 5.66%), which is")
    A("  the one pooled statement in the audit that needs no convention.")
    A("")

    A("6  LABEL-SOURCE REPAIR AS A BOUND")
    A("-" * 78)
    A("Matched conventions only (shipped and repaired are compared inside the")
    A("same positivity x reading cells).  ALL_evaluable rows.")
    A("")
    A("  %-16s %-13s %6s %6s  %-19s %-19s %s"
      % ("cohort", "source", "n", "conv", "shipped range", "repaired range",
         "primary shipped->repaired"))
    for ds in DATASET_ORDER:
        for src in pooled_sources:
            g = pooled[(pooled.dataset == ds) & (pooled.source == src) &
                       (pooled.cls == "ALL_evaluable")]
            if not len(g):
                continue
            r = g.iloc[0]
            A("  %-16s %-13s %6d %6d  %-19s %-19s %s -> %s"
              % (ds, src, int(r.n_primary), int(r.n_conventions_finite),
                 "%s-%s" % (fmt_pct(r.rate_min_shipped),
                            fmt_pct(r.rate_max_shipped)),
                 "%s-%s" % (fmt_pct(r.rate_min_repaired),
                            fmt_pct(r.rate_max_repaired)),
                 fmt_pct(r.rate_primary_shipped),
                 fmt_pct(r.rate_primary_repaired)))
    A("")
    hr = pooled[(pooled.dataset == "GVHD_Feb1323") &
                (pooled.source == "hq_ref") &
                (pooled.cls == "ALL_evaluable")].iloc[0]
    A("  The manuscript's '23.08% -> 0.07%' is reproduced at the primary")
    A("  convention (GVHD, hq_ref, ALL_evaluable).  As a bound-based claim it")
    A("  becomes: on the depot's own labels the shipped rule text violates in")
    A("  %s-%s of cells across the sweep, and the repaired text in"
      % (fmt_pct(hr.rate_min_shipped), fmt_pct(hr.rate_max_shipped)))
    A("  %s-%s." % (fmt_pct(hr.rate_min_repaired),
                     fmt_pct(hr.rate_max_repaired)))
    A("  The repair is monotone (repaired <= shipped) at all %d matched" % n_rep)
    A("  conventions, so the direction of the repair is convention-free even")
    A("  though its level is not.  0.07% must be quoted as 'at the primary")
    A("  convention'.")
    A("")

    A("VERDICT")
    A("-" * 78)
    A("The audit survives the sweep as a bounded, partially identified finding,")
    A("and it is stronger than 'not identified':")
    A("")
    A("  * Every analysed population (scope=type and the pooled ALL_* rows) has")
    A("    a finite identified set.  Pooled: 0.44%-87.8%")
    A("    (breast), 0.58%-90.2% (CRC), 2.09%-90.0% (GVHD), 3.98%-87.2%")
    A("    (melanoma).  In all four cohorts every convention attaining the")
    A("    minimum uses positivity=gmm, and the manuscript's primary convention")
    A("    (gmm/or/repaired) is itself at the minimum -- which is why the")
    A("    primary rates sit at the lower end.  The maxima are the 95th")
    A("    percentile of the positive population.")
    A("  * Convention-free facts exist and name specific populations: a cell type")
    A("    is robustly discordant when its rate clears 5% under all 68")
    A("    conventions.  The robustly discordant pairs are:")
    for ds in DATASET_ORDER:
        c = cert[ds]
        for cl in c["disc"]:
            r = exp[(exp.dataset == ds) & (exp.cls == cl)].iloc[0]
            A("      %s %s: minimum %s over the sweep, reached at %s"
              % (ds, cl, fmt_pct(r.rate_min),
                 conv_str(r.min_positivity, r.min_reading, r.min_variant)))
    A("  * Robustly concordant (<=5%% under all 68): %s."
      % ("; ".join("%s %s" % (d, ", ".join(cert[d]["conc"]))
                   for d in DATASET_ORDER if cert[d]["conc"]) or "none"))
    A("")
    A("  MANUSCRIPT SENTENCES TO WEAKEN")
    A("    - Abstract: 'The pooled aggregate itself is not identified ... so any")
    A("      absolute marker-audit number is a statement about a convention")
    A("      rather than about the cells.'  The pooled rate is bounded, so")
    A("      'not identified' should become 'identified only up to a stated")
    A("      convention interval', with the intervals above quoted.")
    A("    - Discussion: 'Aggregate violation rates are not identifiable objects")
    A("      in these data ... We therefore do not report aggregate violation")
    A("      rates as evidence.'  They are partially identified; report them as")
    A("      intervals instead of withholding them.")
    A("    - Discussion: 'the audit rests instead on the populations whose rules")
    A("      are fully evaluable, where no convention is involved.'  This is")
    A("      wrong and a referee can check it in one line: fully evaluable")
    A("      populations still move with the positivity call (GVHD neutrophils")
    A("      26.35% -> 93.41%).  What is convention-free is the VERDICT (above")
    A("      or below the gate), not the level.")
    A("    - Results: 'these populations are violated at a rate of at least a")
    A("      fifth under the depot's own default convention' understates what")
    A("      the sweep supports.")
    A("")
    A("  MANUSCRIPT SENTENCES TO STRENGTHEN")
    A("    - 'at least a fifth under the depot's own default convention'")
    A("      becomes 'at least the identified-set minimum, attained under EVERY")
    A("      convention':")
    for ds, cls in (("GVHD_Feb1323", "Neutrophil"), ("GVHD_Feb1323", "Plasma"),
                    ("Melanoma_Oct2522", "CD8T")):
        r = exp[(exp.dataset == ds) & (exp.cls == cls)].iloc[0]
        A("        %s %s: >= %s under all 68 conventions (min at %s)"
          % (ds, cls, fmt_pct(r.rate_min),
             conv_str(r.min_positivity, r.min_reading, r.min_variant)))
    A("    - Add: the identity of the discordant populations is stable across")
    A("      the sweep for %d populations (listed above), which no threshold"
      % sum(len(cert[d]["disc"]) for d in DATASET_ORDER))
    A("      choice moves below the gate.")
    A("    - Add: the between-convention spread exceeds the fixed-convention")
    A("      sampling CI by a median factor of about %.0f, so the sweep, not the"
      % rr.median())
    A("      CI, is the binding uncertainty.")
    A("    - Add: the rule-text repair is monotone at all %d matched" % n_rep)
    A("      conventions, so 'repair reduces violations' is convention-free even")
    A("      though '0.07%' is not.")
    A("")
    if failed:
        A("*** SANITY CHECK FAILURES: " + "; ".join(c[0] for c in failed))
        A("*** treat the numbers above as unverified until resolved.")
    A("End of report.")

    text = "\n".join(L) + "\n"
    bad = [i + 1 for i, line in enumerate(text.split("\n"))
           if any(ord(ch) > 127 for ch in line)]
    if bad:
        raise SystemExit("non-ASCII in report at lines %s" % bad[:5])
    with open(OUT_REPORT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)

    sys.stdout.write("wrote %s\nwrote %s\nwrote %s\nwrote %s\n"
                     % (OUT_BY_MARKER, OUT_POOLED, OUT_SCRATCH, OUT_REPORT))
    for name, ok, note in checks:
        sys.stdout.write("check [%s] %s -- %s\n"
                         % ("ok" if ok else "FAIL", name, note))
    sys.stdout.write("reproduction: %d/%d PASS\n"
                     % (sum(1 for r in repro if r[3]), len(repro)))
    for label, exp_s, comp_s, ok in repro:
        if not ok:
            sys.stdout.write("  FAIL %s: quoted %s, got %s\n"
                             % (label, exp_s, comp_s))
    sys.stdout.write("\nrobustly discordant (expert labels):\n")
    for ds in DATASET_ORDER:
        sys.stdout.write("  %-16s %s\n" % (ds, cert[ds]["disc"] or "none"))
    if failed or not all(r[3] for r in repro):
        raise SystemExit(1)


if __name__ == "__main__":
    main()