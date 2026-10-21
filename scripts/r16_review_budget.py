"""R16 -- the review budget the distribution-free guarantee actually buys.

A certificate is only worth its conservatism if the work it removes is worth
removing.  The comparison usually run -- certified acceptance against an oracle
threshold fitted on the test labels -- is not a gate: no valid rule can beat
that oracle, because the oracle is allowed to look at the answers.  The only
genuinely available alternative is to send every cell to a human.

This script therefore reports, for every family and every budget alpha:

  * the review fraction, 1 - acceptance, i.e. how much manual annotation the
    layer takes off the bench relative to reviewing every cell;
  * the realised selective risk among the retained cells, which is what makes
    the removed work safe to remove;
  * a *matched-risk* comparison.  At a target realised error rate r, how much
    can a test-peeking oracle certify, how much does the uncalibrated threshold
    certify -- and does it actually meet r? -- and how much does the certificate
    certify.  The ratio crc/oracle is the honest price of validity;
  * per-cohort review budgets and absolute annotation counts;
  * the fixed-sequence (step-down) variant of the threshold rule, which charges
    no Bonferroni share to the threshold grid, against the default, together
    with a diagnostic of the shape of the feasible set that decides whether the
    fixed-sequence shortcut can be used at all.

Outputs
    experiments/r16_review_budget.csv
    experiments/r16_matched_risk.csv
    experiments/r16_matched_risk_eval.csv
    experiments/r16_matched_risk_ratio.csv
    experiments/r16_percohort.csv
    experiments/r16_fixedseq.csv
    experiments/r16_fixedseq_delta.csv
    experiments/r16_fixedseq_geometry.csv
    work/r16_report.txt
    figures/fig7_review_budget.pdf / .png
"""

import argparse
import multiprocessing as mp
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, paths

PRED = os.path.join(paths.WORK, "predictions")
RES = paths.RESULTS
FIG = paths.FIGURES

FAMILIES = ("cellsplit", "within", "transfer")
FAM_LABEL = {"cellsplit": "random split", "within": "cross-FOV",
             "transfer": "cross-tissue"}
FAM_CLR = {"cellsplit": "#1f5fa9", "within": "#2a9d8f", "transfer": "#d1495b"}
CURVE_METHODS = ("crc", "naive", "oracle", "conformal_set", "crc_temp",
                 "temp_naive")
RISKS = (0.05, 0.10, 0.15, 0.20, 0.30)
FS_ALPHAS = (0.05, 0.10, 0.20)
# (select, min_accept, grid_size).  The default is (bonferroni, 25, 120).  The
# pairs that hold min_accept and grid_size fixed isolate what the multiplicity
# share -- not the grid resolution -- is worth.
FS_CONFIGS = (
    ("bonferroni", 25, 120),
    ("fixed_sequence", 25, 120),
    ("bonferroni", 100, 120),
    ("fixed_sequence", 100, 120),
    ("bonferroni", 25, 20),
    ("fixed_sequence", 25, 20),
)
COHORTS = ("Breast_Jun0524", "CRC_Feb2623", "GVHD_Feb1323", "Melanoma_Oct2522")
SHORT = {"Breast_Jun0524": "Breast", "CRC_Feb2623": "CRC",
         "GVHD_Feb1323": "GVHD", "Melanoma_Oct2522": "Melanoma"}
B = 2000
NAME_RE = re.compile(r"(.*)__(logreg|rf|hgb)\.npz$")


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def family_of(unit):
    if unit.startswith("cellsplit"):
        return "cellsplit"
    if unit.startswith("within"):
        return "within"
    return "transfer"


def target_of(unit):
    u = re.sub(r"_fold\d+$", "", str(unit))
    if u.startswith("transfer_"):
        return u.split("_to_")[-1]
    return re.sub(r"^(cellsplit|within)_", "", u)


def boot_ci(values, rng, B=B):
    """Percentile cluster bootstrap; the resampling unit is a run unit."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    idx = rng.integers(0, values.size, size=(B, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_r03():
    m = pd.read_csv(os.path.join(RES, "r03_metrics.csv"))
    return m.rename(columns={"kind": "family"})


# ------------------------------------------------------- A. review-budget curve
def review_budget_curve(m, rng):
    """Mean acceptance / review fraction / realised risk per family x method x alpha."""
    rows = []
    for (fam, meth, alpha), g in m.groupby(["family", "method", "alpha"]):
        if meth not in CURVE_METHODS:
            continue
        per_unit = g.groupby("unit")["accept_rate"].mean()
        lo, hi = boot_ci(per_unit.values, rng)
        nv = g[np.isfinite(g["selective_risk"])]
        pu_risk = (nv.groupby("unit")["selective_risk"].mean()
                   if len(nv) else pd.Series(dtype=float))
        rlo, rhi = boot_ci(pu_risk.values, rng)
        acc_nv = g.loc[g["empty"] == 0, "accept_rate"]
        rows.append({
            "family": fam, "method": meth, "alpha": float(alpha),
            "n_eval": int(len(g)), "n_units": int(per_unit.size),
            "mean_accept": float(per_unit.mean()),
            "accept_lo": lo, "accept_hi": hi,
            "review_frac": 1.0 - float(per_unit.mean()),
            "mean_accept_nonvac": float(acc_nv.mean()) if len(acc_nv) else np.nan,
            "frac_nonvacuous": float((g["empty"] == 0).mean()),
            "risk_nonvac": float(pu_risk.mean()) if len(pu_risk) else np.nan,
            "risk_lo": rlo, "risk_hi": rhi,
            "coverage": float(g["covered"].mean()),
            "coverage_nonvac": float(nv["covered"].mean()) if len(nv) else np.nan,
            "eval_cells_tested": float(g["n_test_eval"].sum()),
            "eval_cells_accepted": float(g["n_accepted"].sum()),
        })
    return (pd.DataFrame(rows)
            .sort_values(["family", "method", "alpha"])
            .reset_index(drop=True))


# ------------------------------------------------------- B. matched-risk comparison
def _matched_job(job):
    """At each target realised risk r, certify as much as each rule can.

    ``oracle_peek`` places the threshold so that the *test* empirical risk
    equals r -- it peeks at the answers, so it is an upper bound, not a method.
    ``naive`` places the threshold so that the *calibration* empirical risk
    equals r, which is what a practitioner does and what carries no guarantee.
    ``crc`` is told the budget r and must cover it with probability 1 - delta.
    """
    fname, risks, delta = job
    mt = NAME_RE.match(fname)
    unit, clf = mt.group(1), mt.group(2)
    fam = family_of(unit)
    d = np.load(os.path.join(PRED, fname), allow_pickle=True)
    cal_probs, cal_y, cal_rep = d["cal_probs"], d["cal_y"], d["cal_rep"]
    test_probs, test_y = d["test_probs"], d["test_y"]
    test_rep = d["test_rep"] if "test_rep" in d.files else None
    rows = []
    for r in range(cal_rep.shape[1]):
        sel = cal_rep[:, r]
        s_cal = 1.0 - cal_probs[sel].max(1)
        e_cal = (cal_probs[sel].argmax(1) != cal_y[sel]).astype(float)
        te_sel = (test_rep[:, r] if test_rep is not None
                  else np.ones(len(test_y), bool))
        n_te = int(te_sel.sum())
        s_te = 1.0 - test_probs[te_sel].max(1)
        e_te = (test_probs[te_sel].argmax(1) != test_y[te_sel]).astype(float)
        ctrls = {float(rr): C.SelectiveRiskController(rr, delta).fit(s_cal, e_cal)
                 for rr in risks}
        for rr in risks:
            lam_p = C.naive_threshold(s_te, e_te, rr)
            lam_n = C.naive_threshold(s_cal, e_cal, rr)
            ctl = ctrls[float(rr)]
            for meth, mask, lam in (("crc", ctl.accept(s_te), ctl.lambda_hat),
                                    ("oracle_peek", s_te <= lam_p, lam_p),
                                    ("naive", s_te <= lam_n, lam_n)):
                mask = np.asarray(mask, dtype=bool)
                n_acc = int(mask.sum())
                rows.append({
                    "unit": unit, "clf": clf, "family": fam, "repeat": r,
                    "risk": float(rr), "method": meth, "n_test_eval": n_te,
                    "n_accepted": n_acc,
                    "accept_rate": n_acc / max(n_te, 1),
                    "selective_risk": (float(e_te[mask].mean()) if n_acc
                                       else np.nan),
                    "empty": int(n_acc == 0),
                    "lambda": float(lam),
                })
    return rows, fname


# ------------------------------------------------------- E. threshold-rule variants
def _fs_job(job):
    fname, alphas, delta, configs = job
    mt = NAME_RE.match(fname)
    unit, clf = mt.group(1), mt.group(2)
    fam = family_of(unit)
    d = np.load(os.path.join(PRED, fname), allow_pickle=True)
    cal_probs, cal_y, cal_rep = d["cal_probs"], d["cal_y"], d["cal_rep"]
    test_probs, test_y = d["test_probs"], d["test_y"]
    test_rep = d["test_rep"] if "test_rep" in d.files else None
    rows = []
    for r in range(cal_rep.shape[1]):
        sel = cal_rep[:, r]
        s_cal = 1.0 - cal_probs[sel].max(1)
        e_cal = (cal_probs[sel].argmax(1) != cal_y[sel]).astype(float)
        te_sel = (test_rep[:, r] if test_rep is not None
                  else np.ones(len(test_y), bool))
        n_te = int(te_sel.sum())
        s_te = 1.0 - test_probs[te_sel].max(1)
        e_te = (test_probs[te_sel].argmax(1) != test_y[te_sel]).astype(float)
        for alpha in alphas:
            for select, ma, gs in configs:
                ctl = C.SelectiveRiskController(
                    alpha, delta, min_accept=ma, grid_size=gs,
                    select=select).fit(s_cal, e_cal)
                mask = ctl.accept(s_te)
                n_acc = int(mask.sum())
                risk = float(e_te[mask].mean()) if n_acc else np.nan
                ratio = getattr(ctl, "bound_ratio_", None)
                cand = getattr(ctl, "cand_", None)
                if ratio is None or ratio.size == 0:
                    grid_m, feas, flo, fhi, ibelow, iabove = 0, 0, np.nan, np.nan, 0, 0
                else:
                    grid_m = int(ratio.size)
                    ok = np.nonzero(ratio <= alpha)[0]
                    feas = int(ok.size)
                    flo = float(cand[ok[0]]) if feas else np.nan
                    fhi = float(cand[ok[-1]]) if feas else np.nan
                    ibelow = int(ok[0]) if feas else grid_m
                    iabove = int(grid_m - 1 - ok[-1]) if feas else 0
                rows.append({
                    "unit": unit, "clf": clf, "family": fam, "repeat": r,
                    "alpha": float(alpha), "select": select,
                    "min_accept": int(ma), "grid_size": int(gs),
                    "config": fs_config_label(select, ma, gs),
                    "n_test_eval": n_te, "n_accepted": n_acc,
                    "accept_rate": n_acc / max(n_te, 1),
                    "selective_risk": risk,
                    "covered": int((not np.isfinite(risk)) or risk <= alpha),
                    "empty": int(n_acc == 0),
                    "lambda": float(ctl.lambda_hat),
                    "grid_m": grid_m, "feas_count": feas,
                    "feas_lo": flo, "feas_hi": fhi,
                    "infeas_below": ibelow, "infeas_above": iabove,
                })
    return rows, fname


def fs_config_label(select, min_accept, grid_size):
    return "%s_ma%d_gs%d" % (select, int(min_accept), int(grid_size))


def run_parallel(fn, jobs, n_jobs):
    out = []
    if n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            for i, (rows, fname) in enumerate(
                    pool.imap(fn, jobs, chunksize=1), 1):
                out.extend(rows)
                if i % 25 == 0 or i == len(jobs):
                    log("    %d/%d  %s" % (i, len(jobs), fname))
    else:
        for i, job in enumerate(jobs, 1):
            rows, fname = fn(job)
            out.extend(rows)
            if i % 25 == 0 or i == len(jobs):
                log("    %d/%d  %s" % (i, len(jobs), fname))
    return pd.DataFrame(out)

# ------------------------------------------------------- matched-risk summary
def matched_summary(mr, rng):
    rows = []
    for (fam, risk, meth), g in mr.groupby(["family", "risk", "method"]):
        per_unit = g.groupby("unit")["accept_rate"].mean()
        lo, hi = boot_ci(per_unit.values, rng)
        nv = g[np.isfinite(g["selective_risk"])]
        pu = (nv.groupby("unit")["selective_risk"].mean()
              if len(nv) else pd.Series(dtype=float))
        rlo, rhi = boot_ci(pu.values, rng)
        rows.append({
            "family": fam, "risk": float(risk), "method": meth,
            "n_eval": int(len(g)), "n_units": int(per_unit.size),
            "mean_accept": float(per_unit.mean()),
            "accept_lo": lo, "accept_hi": hi,
            "review_frac": 1.0 - float(per_unit.mean()),
            "realised_risk": float(pu.mean()) if len(pu) else np.nan,
            "risk_lo": rlo, "risk_hi": rhi,
            "frac_nonvacuous": float((g["empty"] == 0).mean()),
        })
    return (pd.DataFrame(rows)
            .sort_values(["family", "method", "risk"])
            .reset_index(drop=True))


def matched_ratio(ms):
    piv = ms.pivot_table(index=["family", "risk"], columns="method",
                         values="mean_accept")
    rsk = ms.pivot_table(index=["family", "risk"], columns="method",
                         values="realised_risk")
    out = pd.DataFrame({
        "crc_accept": piv["crc"],
        "oracle_accept": piv["oracle_peek"],
        "naive_accept": piv["naive"],
        "ratio_crc_oracle": piv["crc"] / piv["oracle_peek"],
        "ratio_crc_naive": piv["crc"] / piv["naive"],
        "crc_risk": rsk["crc"],
        "oracle_risk": rsk["oracle_peek"],
        "naive_risk": rsk["naive"],
    }).reset_index()
    out["naive_risk_excess"] = out["naive_risk"] - out["risk"]
    return out


# ------------------------------------------------------- C/D. per cohort, cells
def percohort(m, alphas=(0.10, 0.20), methods=("crc", "naive", "oracle")):
    """Review fraction and absolute annotation counts per cohort, random split.

    The folds split the held-out cells by field of view, so a cohort total is
    the sum over folds of its per-fold mean: each held-out cell sits in exactly
    one fold's test pool and is counted once.
    """
    g0 = m[(m.family == "cellsplit") & m.method.isin(methods)].copy()
    g0["cohort"] = [target_of(u) for u in g0["unit"]]
    rows = []
    for (meth, alpha), g in g0.groupby(["method", "alpha"]):
        if not any(np.isclose(float(alpha), a) for a in alphas):
            continue
        for ds, gg in g.groupby("cohort"):
            pu = gg.groupby("unit").agg(accept=("accept_rate", "mean"),
                                        n_test=("n_test_eval", "mean"),
                                        n_acc=("n_accepted", "mean"),
                                        risk=("selective_risk", "mean"))
            tested = float(pu["n_test"].sum())
            accepted = float((pu["accept"] * pu["n_test"]).sum())
            fin = np.isfinite(pu["risk"]) & (pu["n_acc"] > 0)
            risk = (float((pu["risk"][fin] * pu["n_acc"][fin]).sum()
                          / pu["n_acc"][fin].sum()) if fin.any() else np.nan)
            rows.append({
                "cohort": ds, "cohort_label": SHORT.get(ds, ds),
                "alpha": float(alpha), "method": meth,
                "folds": int(pu.shape[0]),
                "mean_accept": float(pu["accept"].mean()),
                "review_frac": 1.0 - float(pu["accept"].mean()),
                "certified_risk": risk,
                "cells_tested": tested,
                "cells_accepted": accepted,
                "cells_review": tested - accepted,
                "pct_review_removed": accepted / max(tested, 1e-9),
            })
    return (pd.DataFrame(rows)
            .sort_values(["method", "alpha", "cohort"])
            .reset_index(drop=True))


# ------------------------------------------------------- fixed-sequence summary
def fs_config_label(select, min_accept, grid_size):
    return "%s_ma%d_gs%d" % (select, int(min_accept), int(grid_size))


def fs_summary(fs, rng):
    rows = []
    for (fam, alpha, cfg), g in fs.groupby(["family", "alpha", "config"]):
        per_unit = g.groupby("unit")["accept_rate"].mean()
        lo, hi = boot_ci(per_unit.values, rng)
        nv = g[np.isfinite(g["selective_risk"])]
        pu = (nv.groupby("unit")["selective_risk"].mean()
              if len(nv) else pd.Series(dtype=float))
        lam = g["lambda"].replace([-np.inf], np.nan)
        clo, chi = boot_ci(g.groupby("unit")["covered"].mean().values, rng)
        rows.append({
            "family": fam, "alpha": float(alpha), "config": cfg,
            "select": g["select"].iloc[0],
            "min_accept": int(g["min_accept"].iloc[0]),
            "grid_size": int(g["grid_size"].iloc[0]),
            "n_eval": int(len(g)), "n_units": int(per_unit.size),
            "mean_accept": float(per_unit.mean()),
            "accept_lo": lo, "accept_hi": hi,
            "coverage": float(g["covered"].mean()),
            "coverage_lo": clo, "coverage_hi": chi,
            "coverage_nonvac": float(nv["covered"].mean()) if len(nv) else np.nan,
            "realised_risk_nonvac": float(pu.mean()) if len(pu) else np.nan,
            "frac_nonvacuous": float((g["empty"] == 0).mean()),
            "median_lambda": float(lam.median()),
            "frac_infeasible": float((g["empty"] == 1).mean()),
        })
    return (pd.DataFrame(rows)
            .sort_values(["family", "alpha", "config"])
            .reset_index(drop=True))


def fs_delta(fs, rng):
    """Paired change in acceptance when the Bonferroni share is dropped.

    Each pair holds min_accept and grid_size fixed, so the only thing that
    changes is where the failure probability is spent: delta/(2m) at every one
    of the m grid points, or delta/2 once at the first failure.
    """
    piv = (fs.groupby(["family", "alpha", "min_accept", "grid_size", "unit",
                       "clf", "select"])["accept_rate"].mean().unstack("select"))
    piv["delta"] = piv["fixed_sequence"] - piv["bonferroni"]
    rows = []
    for (fam, alpha, ma, gs), g in piv.groupby(
            ["family", "alpha", "min_accept", "grid_size"]):
        lo, hi = boot_ci(g["delta"].values, rng)
        rows.append({
            "family": fam, "alpha": float(alpha),
            "min_accept": int(ma), "grid_size": int(gs),
            "n_unit_clf": int(len(g)),
            "accept_bonferroni": float(g["bonferroni"].mean()),
            "accept_fixed_sequence": float(g["fixed_sequence"].mean()),
            "delta_accept": float(g["delta"].mean()),
            "delta_lo": lo, "delta_hi": hi,
            "frac_improved": float((g["delta"] > 1e-12).mean()),
            "frac_worsened": float((g["delta"] < -1e-12).mean()),
        })
    return (pd.DataFrame(rows)
            .sort_values(["family", "alpha", "min_accept", "grid_size"])
            .reset_index(drop=True))


def fs_geometry(fs):
    """Where on the threshold grid the bounds are actually feasible.

    The Bonferroni reading spends delta/(2m) on every grid point so that any
    data-dependent pick inherits the bound; the fixed-sequence reading spends
    delta/2 once and instead certifies a running prefix of the prespecified
    order.  The two agree only if the feasible set is a prefix of that order.
    """
    g = fs[fs.config == fs_config_label("fixed_sequence", 25, 120)]
    rows = []
    for (fam, alpha), gg in g.groupby(["family", "alpha"]):
        rows.append({
            "family": fam, "alpha": float(alpha),
            "grid_points": float(gg["grid_m"].mean()),
            "feasible_points": float(gg["feas_count"].mean()),
            "feasible_from_cand": float(gg["feas_lo"].mean()),
            "feasible_to_cand": float(gg["feas_hi"].mean()),
            "infeasible_below": float(gg["infeas_below"].mean()),
            "infeasible_above": float(gg["infeas_above"].mean()),
            "frac_evals_feasible_set_is_prefix":
                float((gg["infeas_below"] == 0).mean()),
        })
    return (pd.DataFrame(rows).sort_values(["family", "alpha"])
            .reset_index(drop=True))

# ------------------------------------------------------- figure 7
def figure(curve, ms, ratio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
        "legend.frameon": False,
    })

    def panel(ax, letter, title=None):
        ax.set_title(letter, loc="left", fontweight="bold", pad=6)
        if title:
            ax.set_title(title, loc="right", fontsize=7.5, color="0.35")

    box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    alphas = sorted(curve.alpha.unique())

    ax = axes[0]
    dy = {"cellsplit": -15, "within": 10, "transfer": 10}
    for fam in FAMILIES:
        col = FAM_CLR[fam]
        s = curve[(curve.family == fam) & (curve.method == "crc")].sort_values("alpha")
        o = curve[(curve.family == fam) & (curve.method == "oracle")].sort_values("alpha")
        ax.plot(o.alpha, o.review_frac, ":", lw=1.0, color=col, alpha=0.9)
        ax.plot(s.alpha, s.review_frac, "o-", ms=3.4, lw=1.3, color=col,
                label=FAM_LABEL[fam])
        for a in (0.10, 0.20):
            row = s[np.isclose(s.alpha, a)]
            if len(row) and np.isfinite(row.risk_nonvac.iloc[0]):
                ax.annotate("%.3f" % row.risk_nonvac.iloc[0],
                            (float(row.alpha.iloc[0]),
                             float(row.review_frac.iloc[0])),
                            textcoords="offset points", xytext=(0, dy[fam]),
                            fontsize=5.4, color=col, ha="center", bbox=box)
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.text(alphas[0] * 1.05, 1.015, "review every cell", fontsize=5.6)
    ax.set_xscale("log")
    ax.set_xticks(alphas)
    ax.set_xticklabels(["%.2f" % a for a in alphas], rotation=45)
    ax.set_xlim(alphas[0] * 0.85, alphas[-1] * 1.25)
    ax.set_xlabel("error budget alpha")
    ax.set_ylabel("fraction of cells sent to review")
    ax.set_ylim(-0.06, 1.18)
    ax.legend(fontsize=6, loc="upper right",
              title="solid = certificate, dotted = oracle", title_fontsize=5.6)
    panel(ax, "a", "labels = realised risk")

    ax = axes[1]
    styles = (("crc", "-", "o", 1.5), ("oracle_peek", ":", "o", 1.1),
              ("naive", "--", "s", 1.1))
    for fam in FAMILIES:
        col = FAM_CLR[fam]
        for meth, ls, mk, lw in styles:
            s = ms[(ms.family == fam) & (ms.method == meth)].sort_values("risk")
            ax.plot(s.risk, s.mean_accept, ls, marker=mk, ms=3.0, lw=lw, color=col)
    handles = [Line2D([], [], color="0.2", ls=ls, marker=mk, ms=3.0, lw=1.2,
                      label=lab)
               for meth, ls, mk, lab in (
                   ("crc", "-", "o", "certified (ours)"),
                   ("oracle_peek", ":", "o", "oracle (peeks at test labels)"),
                   ("naive", "--", "s", "uncalibrated"))]
    ax.legend(handles=handles, fontsize=5.2, loc="center right")
    ax.text(0.397, 0.155, "acceptance / oracle at r = 0.10",
            fontsize=5.4, ha="right", va="bottom", color="0.35")
    for i, fam in enumerate(FAMILIES):
        r = ratio[ratio.family == fam].sort_values("risk")
        v = r[np.isclose(r.risk, 0.10)]
        if len(v):
            ax.text(0.397, 0.105 - 0.043 * i,
                    "%s  %.2f" % (FAM_LABEL[fam],
                                  float(v.ratio_crc_oracle.iloc[0])),
                    fontsize=5.6, ha="right", va="bottom", color=FAM_CLR[fam])
    ax.set_xlim(0.03, 0.40)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("matched realised risk r")
    ax.set_ylabel("acceptance")
    panel(ax, "b", "at the same realised risk")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "fig7_review_budget.%s" % ext),
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig7_review_budget.pdf/.png")


# ------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--limit", default="")
    ap.add_argument("--skip-threshold-arms", action="store_true")
    args = ap.parse_args()
    n_jobs = max(1, args.jobs)
    rng = np.random.default_rng(20260926)
    t0 = time.time()
    R = []

    def say(*parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        R.append(line)

    def table(df, cols=None, sort=None):
        d = df if sort is None else df.sort_values(sort)
        if cols is not None:
            d = d[cols]
        say(d.round(4).to_string(index=False))

    m = load_r03()
    alphas = sorted(float(a) for a in m.alpha.unique())
    say("R16 -- what the guarantee costs, measured against the only available")
    say("       alternative: sending every cell to a human.")
    say("")
    say("input   : experiments/r03_metrics.csv")
    say("rows    : %d" % len(m))
    say("families: %s" % ", ".join(sorted(m.family.unique())))
    say("alphas  : %s" % ", ".join("%.2f" % a for a in alphas))
    say("methods : %s" % ", ".join(sorted(m.method.unique())))
    say("delta   : %.2f  (the guarantee holds with probability >= %.2f)"
        % (args.delta, 1.0 - args.delta))
    say("")

    # ---------------------------------------------------------------- A
    curve = review_budget_curve(m, rng)
    curve.to_csv(os.path.join(RES, "r16_review_budget.csv"), index=False)
    say("=" * 78)
    say("A. review-budget curve  (experiments/r16_review_budget.csv)")
    say("   review_frac = 1 - mean certified acceptance, averaged over units;")
    say("   risk_nonvac = mean realised selective error among the retained cells;")
    say("   frac_nonvacuous = fraction of evaluations in which the layer answered")
    say("   at all (an empty answer is conservative, not wrong).")
    say("=" * 78)
    table(curve[curve.method.isin(("crc", "naive", "oracle"))],
          ["family", "method", "alpha", "n_units", "mean_accept",
           "review_frac", "risk_nonvac", "frac_nonvacuous", "coverage"],
          sort=["family", "alpha", "method"])
    say("")
    say("conformal-set and temperature-scaled arms, for reference:")
    table(curve[curve.method.isin(("conformal_set", "crc_temp", "temp_naive"))],
          ["family", "method", "alpha", "mean_accept", "review_frac",
           "risk_nonvac", "coverage"], sort=["family", "alpha", "method"])
    say("")
    say("headline -- manual annotation removed relative to reviewing every cell")
    say("(unit-averaged fraction; the pooled per-cohort cell counts are in"
        " section C/D)")
    say("%-14s %-6s %10s %12s %14s" % ("family", "alpha", "review_frac",
                                       "work removed", "certified risk"))
    for fam in FAMILIES:
        for a in (0.10, 0.20):
            s = curve[(curve.family == fam) & (curve.method == "crc")
                      & np.isclose(curve.alpha, a)]
            if not len(s):
                continue
            say("%-14s %-6.2f %10.3f %11.1f%% %14.4f"
                % (fam, a, float(s.review_frac.iloc[0]),
                   100.0 * float(s.mean_accept.iloc[0]),
                   float(s.risk_nonvac.iloc[0])))
    say("")

    # ---------------------------------------------------------------- B
    files = sorted(f for f in os.listdir(PRED) if f.endswith(".npz"))
    if args.limit:
        files = [f for f in files if args.limit in f]
    jobs = [(f, RISKS, args.delta) for f in files]
    log("B. matched-risk comparison over %d prediction files (jobs=%d)"
        % (len(jobs), n_jobs))
    mr = run_parallel(_matched_job, jobs, n_jobs)
    mr.to_csv(os.path.join(RES, "r16_matched_risk.csv"), index=False)
    ms = matched_summary(mr, rng)
    ms.to_csv(os.path.join(RES, "r16_matched_risk_eval.csv"), index=False)
    ratio = matched_ratio(ms)
    ratio.to_csv(os.path.join(RES, "r16_matched_risk_ratio.csv"), index=False)
    say("=" * 78)
    say("B. acceptance at matched realised risk")
    say("   At a target realised error rate r the oracle (peeking at test labels)")
    say("   can certify oracle_accept; the uncalibrated threshold certifies")
    say("   naive_accept while missing the budget by naive_risk_excess; the")
    say("   certificate certifies crc_accept under a <= r guarantee.  The ratio")
    say("   crc/oracle is the price of validity at equal risk -- the only")
    say("   apples-to-apples efficiency statement, since no valid rule can beat")
    say("   an oracle that reads the answers.")
    say("=" * 78)
    table(ratio, ["family", "risk", "crc_accept", "oracle_accept",
                  "naive_accept", "ratio_crc_oracle", "ratio_crc_naive",
                  "crc_risk", "oracle_risk", "naive_risk",
                  "naive_risk_excess"], sort=["family", "risk"])
    say("")
    say("per-method detail (experiments/r16_matched_risk_eval.csv):")
    table(ms, ["family", "risk", "method", "mean_accept", "accept_lo",
               "accept_hi", "realised_risk", "frac_nonvacuous"],
          sort=["family", "risk", "method"])
    say("")
    chk = curve[curve.method == "oracle"][["family", "alpha", "mean_accept"]]
    chk = chk.rename(columns={"alpha": "risk", "mean_accept": "oracle_row"})
    j = ms[ms.method == "oracle_peek"].merge(chk, on=["family", "risk"])
    if len(j):
        say("consistency check against the r03 oracle row (same selector, same")
        say("cells): max |difference| in mean acceptance = %.2e over %d cells"
            % (float(np.abs(j.mean_accept - j.oracle_row).max()), len(j)))
    say("")

    # ---------------------------------------------------------------- C/D
    pc = percohort(m)
    pc.to_csv(os.path.join(RES, "r16_percohort.csv"), index=False)
    say("=" * 78)
    say("C/D. per-cohort review budget and absolute annotation counts")
    say("     (random-split family; the folds split the held-out cells by field")
    say("     of view, so the cell totals count each held-out cell once)")
    say("=" * 78)
    table(pc, ["method", "alpha", "cohort_label", "folds", "mean_accept",
               "review_frac", "certified_risk", "cells_tested",
               "cells_accepted", "cells_review", "pct_review_removed"],
          sort=["method", "alpha", "cohort_label"])
    say("")
    say("work saved, alpha = 0.10 and 0.20, certified rule vs review-everything:")
    cr = pc[pc.method == "crc"]
    for a in (0.10, 0.20):
        s = cr[np.isclose(cr.alpha, a)]
        if not len(s):
            continue
        w = float(np.nansum(s.certified_risk * s.cells_accepted)
                  / max(s.cells_accepted.sum(), 1e-9))
        say("  alpha = %.2f: %d of %d held-out cells annotate automatically, "
            "%d still need a human (%.1f%% of the review work removed); "
            "certified error among the retained %.4f"
            % (a, int(s.cells_accepted.sum()), int(s.cells_tested.sum()),
               int(s.cells_review.sum()),
               100.0 * float(s.cells_accepted.sum() / s.cells_tested.sum()),
               w))
    orc = pc[pc.method == "oracle"]
    if len(orc):
        for a in (0.10, 0.20):
            s = orc[np.isclose(orc.alpha, a)]
            if len(s):
                say("  the test-peeking oracle, at the same nominal budget "
                    "alpha = %.2f, leaves %d of %d cells for review"
                    % (a, int(s.cells_review.sum()), int(s.cells_tested.sum())))
    say("")

    # ---------------------------------------------------------------- E
    fd = pd.DataFrame()
    fsum = pd.DataFrame()
    geom = pd.DataFrame()
    if args.skip_threshold_arms:
        log("E. threshold-rule arms skipped")
    else:
        fs_files = [f for f in files
                    if f.startswith("cellsplit") or f.startswith("within")]
        fs_jobs = [(f, FS_ALPHAS, args.delta, FS_CONFIGS) for f in fs_files]
        log("E. threshold-rule arms over %d prediction files (jobs=%d)"
            % (len(fs_jobs), n_jobs))
        fs = run_parallel(_fs_job, fs_jobs, n_jobs)
        fs.to_csv(os.path.join(RES, "r16_fixedseq.csv"), index=False)
        fsum = fs_summary(fs, rng)
        fd = fs_delta(fs, rng)
        geom = fs_geometry(fs)
        fsum.to_csv(os.path.join(RES, "r16_fixedseq_eval.csv"), index=False)
        fd.to_csv(os.path.join(RES, "r16_fixedseq_delta.csv"), index=False)
        geom.to_csv(os.path.join(RES, "r16_fixedseq_geometry.csv"), index=False)
        say("=" * 78)
        say("E. threshold-rule arms: where the failure probability is spent")
        say("   The default spends delta/(2m) at every point of an m-point grid,")
        say("   which is what makes a data-dependent pick of the largest feasible")
        say("   threshold valid.  The fixed-sequence rule spends delta/2 once, at")
        say("   the first failure in a prespecified order, and certifies the")
        say("   running prefix of that order.  The two agree only when the")
        say("   feasible set is a prefix.  min_accept is a prespecified floor on")
        say("   the accepted set; grid_size is the grid resolution.")
        say("=" * 78)
        table(fsum, ["family", "alpha", "select", "min_accept", "grid_size",
                     "n_units", "mean_accept", "accept_lo", "accept_hi",
                     "coverage", "coverage_lo", "coverage_hi",
                     "coverage_nonvac", "realised_risk_nonvac",
                     "frac_nonvacuous", "frac_infeasible"],
              sort=["family", "alpha", "min_accept", "grid_size", "select"])
        say("")
        say("paired change in acceptance, same unit x classifier x repeat, with")
        say("min_accept and grid_size held fixed (experiments/"
            "r16_fixedseq_delta.csv):")
        table(fd, ["family", "alpha", "min_accept", "grid_size",
                   "accept_bonferroni", "accept_fixed_sequence", "delta_accept",
                   "delta_lo", "delta_hi", "frac_improved", "frac_worsened"],
              sort=["family", "alpha", "min_accept", "grid_size"])
        say("")
        say("shape of the feasible set on the fixed-sequence grid (default")
        say("grid_size = 120, min_accept = 25); the bounds are infeasible at BOTH")
        say("ends -- below because the accepted count is too small to pin down")
        say("the denominator, above because the running error rate reaches the")
        say("budget -- so the feasible set is a middle interval, not a prefix:")
        table(geom, ["family", "alpha", "grid_points", "feasible_points",
                     "feasible_from_cand", "feasible_to_cand", "infeasible_below",
                     "infeasible_above", "frac_evals_feasible_set_is_prefix"],
              sort=["family", "alpha"])
        say("")
        say("reproduction check: the default cell of these arms is exactly the")
        say("controller of r03, so its per-evaluation acceptance must match the")
        say("crc rows of experiments/r03_metrics.csv:")
        base = fs[fs.config == fs_config_label("bonferroni", 25, 120)]
        key = ["unit", "clf", "repeat", "alpha"]
        cols = ["accept_rate", "covered", "selective_risk"]
        jj = base[key + cols].merge(m[m.method == "crc"][key + cols], on=key,
                                    suffixes=("_r16", "_r03"))
        if len(jj):
            say("  %d matched evaluations; max |d accept_rate| = %.2e, "
                "max |d covered| = %.2e, mean |d selective_risk| = %.2e"
                % (len(jj),
                   float(np.abs(jj.accept_rate_r16
                                - jj.accept_rate_r03).max()),
                   float(np.abs(jj.covered_r16 - jj.covered_r03).max()),
                   float(np.nanmean(np.abs(jj.selective_risk_r16
                                           - jj.selective_risk_r03)))))
        say("")
        say("net effect of the multiplicity change, averaged over families and")
        say("budgets (paired):")
        for ma, gs in sorted(set(zip(fd.min_accept, fd.grid_size))):
            s = fd[(fd.min_accept == ma) & (fd.grid_size == gs)]
            if not len(s):
                continue
            say("  min_accept=%3d grid_size=%3d : bonferroni %.4f -> "
                "fixed-sequence %.4f  (%+.4f)"
                % (ma, gs, s.accept_bonferroni.mean(),
                   s.accept_fixed_sequence.mean(), s.delta_accept.mean()))
        say("")
        say("net effect of the grid resolution, bonferroni rule only:")
        for gs in sorted(set(fsum.grid_size)):
            s = fsum[(fsum.grid_size == gs) & (fsum.select == "bonferroni")
                     & (fsum.min_accept == 25)]
            if len(s):
                say("  grid_size=%3d : mean accepted fraction %.4f, "
                    "mean coverage %.4f" % (gs, s.mean_accept.mean(),
                                            s.coverage.mean()))
        say("")

    # ---------------------------------------------------------------- figure
    figure(curve, ms, ratio)

    # ---------------------------------------------------------------- verdict
    say("=" * 78)
    say("verdict")
    say("=" * 78)
    for fam in FAMILIES:
        s = curve[(curve.family == fam) & (curve.method == "crc")]
        for a in (0.10, 0.20):
            r = s[np.isclose(s.alpha, a)]
            if not len(r):
                continue
            rr = ratio[(ratio.family == fam) & np.isclose(ratio.risk, a)]
            extra = ("" if not len(rr) else
                     " | price of validity at matched risk r = %.2f: %.2f"
                     % (a, float(rr.ratio_crc_oracle.iloc[0])))
            say("* %-11s alpha = %.2f -> %5.1f%% of cells still reviewed, "
                "realised error among the rest %.4f%s"
                % (fam, a, 100.0 * float(r.review_frac.iloc[0]),
                   float(r.risk_nonvac.iloc[0]), extra))
    if len(fsum) and len(fd):
        say("* step-down (fixed-sequence) order as implemented, min_accept = 25:")
        for fam in ("cellsplit", "within"):
            d = fd[(fd.family == fam) & (fd.min_accept == 25)
                   & (fd.grid_size == 120)]
            b = fsum[(fsum.family == fam) & (fsum.select == "bonferroni")
                     & (fsum.min_accept == 25) & (fsum.grid_size == 120)]
            s = fsum[(fsum.family == fam) & (fsum.select == "fixed_sequence")
                     & (fsum.min_accept == 25) & (fsum.grid_size == 120)]
            say("  %-11s acceptance %.3f -> %.3f (%+.3f mean over the three "
                "budgets); it certifies nothing at all at alpha <= 0.10, "
                "because the feasible set is not a prefix of the grid"
                % (fam, float(b.mean_accept.mean()),
                   float(s.mean_accept.mean()),
                   float(d.delta_accept.mean())))
        say("* step-down order with a prespecified floor of 100 accepted cells,")
        say("  which places the sequence where the bound is feasible:")
        for fam in ("cellsplit", "within"):
            for aa in (0.10, 0.20):
                d = fd[(fd.family == fam) & (fd.min_accept == 100)
                       & (fd.grid_size == 120) & np.isclose(fd.alpha, aa)]
                b = fsum[(fsum.family == fam) & (fsum.select == "bonferroni")
                         & (fsum.min_accept == 100) & (fsum.grid_size == 120)
                         & np.isclose(fsum.alpha, aa)]
                s = fsum[(fsum.family == fam)
                         & (fsum.select == "fixed_sequence")
                         & (fsum.min_accept == 100) & (fsum.grid_size == 120)
                         & np.isclose(fsum.alpha, aa)]
                if not (len(d) and len(b) and len(s)):
                    continue
                say("  %-11s alpha = %.2f: acceptance %.3f -> %.3f (%+.3f "
                    "[%+.3f, %+.3f]); coverage %.3f -> %.3f against a nominal "
                    "%.2f" % (fam, aa, float(b.mean_accept.iloc[0]),
                              float(s.mean_accept.iloc[0]),
                              float(d.delta_accept.iloc[0]),
                              float(d.delta_lo.iloc[0]),
                              float(d.delta_hi.iloc[0]),
                              float(b.coverage.iloc[0]),
                              float(s.coverage.iloc[0]), 1.0 - args.delta))
    cs20 = curve[(curve.family == "cellsplit") & (curve.method == "crc")
                 & np.isclose(curve.alpha, 0.20)]
    say("* the defensible efficiency claim is the review budget, not the oracle")
    say("  comparison.  At alpha = 0.20 on an exchangeable split the layer")
    say("  removes the annotation of %.1f%% of the cells while the realised"
        % (100.0 * float(cs20.mean_accept.iloc[0])))
    say("  error among the labels it keeps is %.4f, well under the 0.20 budget."
        % float(cs20.risk_nonvac.iloc[0]))
    say("")
    say("elapsed %.1f s" % (time.time() - t0))
    with open(os.path.join(paths.WORK, "r16_report.txt"), "w") as f:
        f.write("\n".join(R) + "\n")
    print("wrote work/r16_report.txt")


if __name__ == "__main__":
    main()
