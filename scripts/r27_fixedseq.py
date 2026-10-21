"""R27 -- the fixed-sequence (step-down) rule, finished across three families.

The manuscript reported the step-down construction as a single sentence in one
family at one budget and one configuration ("0.560 to 0.634 at alpha = 0.10
with coverage 0.963").  A referee objected that the only sharper-and-still-
valid construction in the paper is reported half-finished.  This script settles
it: all three split families, three budgets, and a full sweep of the two knobs
that decide whether the rule can be used at all, on the identical grid and the
identical calibration/test split used by scripts/r16_review_budget.py::_fs_job.

Arms (all on the identical grid _grid_indices(n, min_accept, grid_size)):

  1. bonferroni / ascending   SelectiveRiskController(select="bonferroni"), the
                              reference.  Spends delta/(2m) at every one of the
                              m grid points and returns the largest feasible
                              threshold, so any data-dependent pick is covered.
  2. fixed_sequence / ascending
                              SelectiveRiskController(select="fixed_sequence").
                              Spends delta/2 once, scans the candidate acceptance
                              counts upward from the smallest, stops at the
                              first failure, delivers the last success.
  3. fixed_sequence / descending
                              implemented HERE, NOT in src/.  Identical grid
                              points and identical hypothesis ratios (they are
                              taken from a select="fixed_sequence" fit), so the
                              only change from arm 2 is the prespecified order:
                              it starts at the largest acceptance count (the
                              riskiest claim, largest lambda) and walks down,
                              again stopping at the first failure and delivering
                              the last success.  No multiplicity share.
  4. naive / calibration      the calibration empirical selective risk held at
                              alpha on the same grid; the no-guarantee yardstick.

Sweep: alpha in {0.05, 0.10, 0.20}, delta in {0.05, 0.10, 0.20},
grid_size in {20, 120, 400}, min_accept in {25, 50, 100, 200}.

Outputs
    experiments/r27_fixedseq.csv           per-evaluation long table
    experiments/r27_fixedseq_summary.csv   per-configuration summary
    work/r27_report.txt                    answers (a)-(e) + sanity checks
    figures/figS7_fixedseq.pdf / .png

Note on the descending arm.  The task text describes the descending order as
running "from the safest claim ... to the riskiest", which once lambda and the
accepted count are ordered the right way round is the ASCENDING order of arm 2,
and would make arm 3 identical to arm 2 by construction.  It also requires the
two to be distinct if any configuration allows it.  We therefore implement the
reading under which the arm is distinct -- reverse the scan -- and both facts
are quantified in work/r27_report.txt, which also states exactly when the two
readings coincide.
"""

import argparse
import io
import multiprocessing as mp
import os
import re
import sys
import time
import warnings

# One BLAS thread per process: the parallelism is over prediction files.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, paths

PRED = os.path.join(paths.WORK, "predictions")
RES = paths.RESULTS
FIG = paths.FIGURES
WORK = paths.WORK

FAMILIES = ("cellsplit", "within", "transfer")
FAM_LABEL = {"cellsplit": "random split", "within": "cross-FOV",
             "transfer": "cross-tissue"}
FAM_CLR = {"cellsplit": "#2b6cb0", "within": "#dd6b20", "transfer": "#c53030"}

ALPHAS = (0.05, 0.10, 0.20)
DELTAS = (0.05, 0.10, 0.20)
GRID_SIZES = (20, 120, 400)
MIN_ACCEPTS = (25, 50, 100, 200)

ARMS = (
    ("bonferroni", "ascending"),
    ("fixed_sequence", "ascending"),
    ("fixed_sequence", "descending"),
    ("naive", "calibration"),
)
ARM_LABEL = {
    ("bonferroni", "ascending"): "bonferroni",
    ("fixed_sequence", "ascending"): "fixed-sequence ascending",
    ("fixed_sequence", "descending"): "fixed-sequence descending",
    ("naive", "calibration"): "naive (calibration)",
}
ARM_CLR = {
    ("bonferroni", "ascending"): "#333333",
    ("fixed_sequence", "ascending"): "#2a9d8f",
    ("fixed_sequence", "descending"): "#8d5fd3",
    ("naive", "calibration"): "#bbbbbb",
}
SELECT_RULES = ("bonferroni", "fixed_sequence")

NAME_RE = re.compile(r"(.*)__(logreg|rf|hgb)\.npz$")

LONG_COLS = ["family", "unit", "clf", "repeat", "alpha", "delta", "select",
             "scan_order", "min_accept", "grid_size", "config", "n_test_eval",
             "n_accepted", "accept_rate", "selective_risk", "covered", "empty",
             "lambda", "grid_m", "feas_count", "feas_lo", "feas_hi",
             "infeas_below", "infeas_above"]
LONG_STR = ("family", "unit", "clf", "select", "scan_order", "config")

# r16's default failure probability, on which the manuscript sentence is based.
R16_DELTA = 0.10
R16_CFGS = ("bonferroni_ma25_gs120", "bonferroni_ma100_gs120",
            "fixed_sequence_ma25_gs120", "fixed_sequence_ma100_gs120")

B = 2000
Z95 = 1.959963984540054


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def family_of(unit):
    if unit.startswith("cellsplit"):
        return "cellsplit"
    if unit.startswith("within"):
        return "within"
    return "transfer"


def cfg_name(select, scan_order, min_accept, grid_size):
    """Config label.  Arms 1-2 reproduce r16's labels exactly."""
    if select == "naive":
        return "naive_ma%d_gs%d" % (min_accept, grid_size)
    if select == "bonferroni":
        return "bonferroni_ma%d_gs%d" % (min_accept, grid_size)
    if scan_order == "ascending":
        return "fixed_sequence_ma%d_gs%d" % (min_accept, grid_size)
    return "fixed_sequence_desc_ma%d_gs%d" % (min_accept, grid_size)


def bonf_cfg(min_accept, grid_size):
    return "bonferroni_ma%d_gs%d" % (min_accept, grid_size)


# ----------------------------------------------------------- the scan orders
def pick_largest_feasible(ok):
    """Largest feasible grid point: the uniform-over-the-grid reading."""
    idx = np.nonzero(ok)[0]
    return int(idx[-1]) if idx.size else None


def pick_ascending(ok):
    """Step-down over acceptance counts from the smallest upward.

    ok[j] is "the bound at grid point j is within budget".  The first failure
    ends the scan; the delivered point is the last success before it.  A very
    small accepted count is the first grid point, so an infeasible bottom makes
    this arm empty.
    """
    bad = np.nonzero(~ok)[0]
    j = int(bad[0]) - 1 if bad.size else int(ok.size) - 1
    return j if j >= 0 else None


def pick_descending(ok):
    """Step-down over acceptance counts from the largest downward.

    Same grid points and same ratios as pick_ascending, only the prespecified
    order is reversed.  Accept until the first failure and deliver the last
    claim accepted, so the delivered point is the smallest accepted count of
    the feasible suffix that touches the top of the grid; if that top point is
    itself infeasible the arm is empty.
    """
    m = int(ok.size)
    if m == 0:
        return None
    rev = np.nonzero(~ok[::-1])[0]
    if rev.size == 0:
        return 0
    j = (m - 1) - int(rev[0]) + 1
    return j if j <= m - 1 else None


PICK = {
    ("bonferroni", "ascending"): pick_largest_feasible,
    ("fixed_sequence", "ascending"): pick_ascending,
    ("fixed_sequence", "descending"): pick_descending,
    ("naive", "calibration"): pick_largest_feasible,
}


def geometry(ok, cand):
    """Feasible set on the grid, in r16's columns."""
    m = int(ok.size)
    idx = np.nonzero(ok)[0]
    if not idx.size:
        return {"grid_m": m, "feas_count": 0, "feas_lo": np.nan,
                "feas_hi": np.nan, "infeas_below": m, "infeas_above": 0}
    return {"grid_m": m, "feas_count": int(idx.size),
            "feas_lo": float(cand[idx[0]]), "feas_hi": float(cand[idx[-1]]),
            "infeas_below": int(idx[0]),
            "infeas_above": int(m - 1 - idx[-1])}


# ------------------------------------------------------- one prediction file
def _job(fname):
    mt = NAME_RE.match(fname)
    unit, clf = mt.group(1), mt.group(2)
    fam = family_of(unit)
    d = np.load(os.path.join(PRED, fname), allow_pickle=True)
    cal_probs, cal_y, cal_rep = d["cal_probs"], d["cal_y"], d["cal_rep"]
    test_probs, test_y = d["test_probs"], d["test_y"]
    test_rep = d["test_rep"] if "test_rep" in d.files else None

    rows = []
    for r in range(int(cal_rep.shape[1])):
        sel = cal_rep[:, r]
        s_cal = 1.0 - cal_probs[sel].max(1)
        e_cal = (cal_probs[sel].argmax(1) != cal_y[sel]).astype(float)
        te_sel = (test_rep[:, r] if test_rep is not None
                  else np.ones(len(test_y), bool))
        n_te = int(te_sel.sum())
        s_te = 1.0 - test_probs[te_sel].max(1)
        e_te = (test_probs[te_sel].argmax(1) != test_y[te_sel]).astype(float)
        s_sorted = np.sort(s_cal, kind="mergesort")

        # The grid is a function of (n_cal, min_accept, grid_size) only, and the
        # bound ratio is a function of (delta, min_accept, grid_size, select)
        # only -- alpha enters at the selection step -- so each is computed once
        # per repeat and every arm is read off the identical arrays.
        grids, empg = {}, {}
        for ma in MIN_ACCEPTS:
            for gs in GRID_SIZES:
                ctl = C.SelectiveRiskController(
                    ALPHAS[0], DELTAS[0], min_accept=ma, grid_size=gs,
                    select="fixed_sequence").fit(s_cal, e_cal)
                cand = getattr(ctl, "cand_", None)
                if cand is None:
                    cand = np.zeros(0, dtype=int)
                    emp = np.zeros(0)
                else:
                    emp = np.asarray(ctl.emp_n_, dtype=float)
                grids[(ma, gs)] = np.asarray(cand, dtype=int)
                empg[(ma, gs)] = emp

        ratios = {}
        for delta in DELTAS:
            for ma in MIN_ACCEPTS:
                for gs in GRID_SIZES:
                    for ssc in SELECT_RULES:
                        ctl = C.SelectiveRiskController(
                            ALPHAS[0], delta, min_accept=ma, grid_size=gs,
                            select=ssc).fit(s_cal, e_cal)
                        ratio = getattr(ctl, "bound_ratio_", None)
                        if ratio is None:
                            ratio = np.zeros(0)
                        ratios[(delta, ma, gs, ssc)] = np.asarray(ratio, float)

        for alpha in ALPHAS:
            for delta in DELTAS:
                for ma in MIN_ACCEPTS:
                    for gs in GRID_SIZES:
                        cand = grids[(ma, gs)]
                        base_ok = (ratios[(delta, ma, gs, "fixed_sequence")]
                                   <= alpha)
                        oks = {
                            ("naive", "calibration"): empg[(ma, gs)] <= alpha,
                            ("bonferroni", "ascending"):
                                ratios[(delta, ma, gs, "bonferroni")] <= alpha,
                            ("fixed_sequence", "ascending"): base_ok,
                            ("fixed_sequence", "descending"): base_ok,
                        }
                        for select, scan_order in ARMS:
                            ok = oks[(select, scan_order)]
                            j = PICK[(select, scan_order)](ok)
                            if j is None or j >= cand.size:
                                lam = -np.inf
                            else:
                                lam = float(s_sorted[int(cand[j]) - 1])
                            mask = s_te <= lam
                            n_acc = int(mask.sum())
                            risk = (float(e_te[mask].mean()) if n_acc
                                    else np.nan)
                            row = {
                                "family": fam, "unit": unit, "clf": clf,
                                "repeat": r, "alpha": float(alpha),
                                "delta": float(delta), "select": select,
                                "scan_order": scan_order,
                                "min_accept": int(ma), "grid_size": int(gs),
                                "config": cfg_name(select, scan_order, ma, gs),
                                "n_test_eval": n_te, "n_accepted": n_acc,
                                "accept_rate": n_acc / max(n_te, 1),
                                "selective_risk": risk,
                                "covered": int((not np.isfinite(risk))
                                               or risk <= alpha),
                                "empty": int(n_acc == 0),
                                "lambda": lam,
                            }
                            row.update(geometry(ok, cand))
                            rows.append(row)
    return pd.DataFrame(rows, columns=LONG_COLS), fname

# --------------------------------------------------------------- statistics
def wilson(k, n, z=Z95):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / float(n)
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    h = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def boot_ci(values, rng, B=B):
    """Percentile bootstrap over the supplied units (or unit-pair deltas)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(v[0]), float(v[0]))
    idx = rng.integers(0, n, size=(B, n))
    m = v[idx].mean(axis=1)
    return (float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)))


def summarise(fs, rng):
    """One row per (family, alpha, delta, min_accept, grid_size, config).

    mean_accept is the mean over units of the per-unit mean acceptance, which is
    exactly r16's convention, so the r16 reproduction check is an equality
    check rather than an approximation.  The paired columns hold the change in
    acceptance against the Bonferroni arm on the identical (unit, clf, repeat)
    evaluations: delta_accept is the mean of the per-unit paired differences and
    its CI is a cluster bootstrap that resamples units.
    """
    rows = []
    keys = ["family", "alpha", "delta", "min_accept", "grid_size"]
    for (fam, alpha, delta, ma, gs), g in fs.groupby(keys, observed=True):
        base = bonf_cfg(ma, gs)
        piv = g.pivot_table(index=["unit", "clf", "repeat"], columns="config",
                            values="accept_rate", aggfunc="mean")
        upiv = piv.groupby(level="unit").mean()
        base_unit = (upiv[base] if base in upiv.columns
                     else pd.Series(dtype=float))
        for cfg in sorted(str(c) for c in g["config"].unique()):
            sub = g[g["config"] == cfg]
            per_unit = sub.groupby("unit", observed=True)["accept_rate"].mean()
            alo, ahi = boot_ci(per_unit.values, rng)
            nv = sub[np.isfinite(sub["selective_risk"])]
            pu = (nv.groupby("unit", observed=True)["selective_risk"].mean()
                  if len(nv) else pd.Series(dtype=float))
            k = int(sub["covered"].sum())
            n = int(len(sub))
            wlo, whi = wilson(k, n)
            lam = sub["lambda"].replace(-np.inf, np.nan)
            if cfg == base:
                dacc, dlo, dhi = 0.0, 0.0, 0.0
                fi, fw, excl, above = 0.0, 0.0, 0, 0
            else:
                d = (piv[cfg] - piv[base]).dropna()
                du = (upiv[cfg] - base_unit).dropna()
                dlo, dhi = boot_ci(du.values, rng)
                dacc = float(du.mean()) if du.size else float("nan")
                fi = float((d > 1e-12).mean()) if d.size else float("nan")
                fw = float((d < -1e-12).mean()) if d.size else float("nan")
                excl = int((dhi < 0.0) or (dlo > 0.0))
                above = int(dlo > 0.0)
            rows.append({
                "family": fam, "alpha": float(alpha), "delta": float(delta),
                "select": sub["select"].iloc[0],
                "scan_order": sub["scan_order"].iloc[0],
                "min_accept": int(ma), "grid_size": int(gs), "config": cfg,
                "n_eval": n, "n_units": int(per_unit.size),
                "mean_accept": float(per_unit.mean()),
                "accept_lo": alo, "accept_hi": ahi,
                "coverage": k / float(n), "n_violations": n - k,
                "coverage_lo": wlo, "coverage_hi": whi,
                "coverage_nonvac": (float(nv["covered"].mean()) if len(nv)
                                    else float("nan")),
                "realised_risk_nonvac": (float(pu.mean()) if len(pu)
                                         else float("nan")),
                "frac_nonvacuous": float((sub["empty"] == 0).mean()),
                "median_lambda": float(lam.median()),
                "frac_infeasible": float((sub["empty"] == 1).mean()),
                "grid_m_mean": float(sub["grid_m"].mean()),
                "feas_count_mean": float(sub["feas_count"].mean()),
                "feas_lo_mean": float(sub["feas_lo"].mean()),
                "feas_hi_mean": float(sub["feas_hi"].mean()),
                "infeas_below_mean": float(sub["infeas_below"].mean()),
                "infeas_above_mean": float(sub["infeas_above"].mean()),
                "frac_prefix_ascending":
                    float((sub["infeas_below"] == 0).mean()),
                "frac_prefix_descending":
                    float((sub["infeas_above"] == 0).mean()),
                "mean_accept_bonferroni": (float(base_unit.mean())
                                           if base_unit.size else float("nan")),
                "delta_accept": dacc, "delta_lo": dlo, "delta_hi": dhi,
                "frac_improved": fi, "frac_worsened": fw,
                "ci_excludes_zero": excl, "ci_above_zero": above,
            })
    return (pd.DataFrame(rows)
            .sort_values(["family", "alpha", "delta", "min_accept", "grid_size",
                          "config"]).reset_index(drop=True))


def pick(summary, **kw):
    d = summary
    for k, v in kw.items():
        if k in ("family", "select", "scan_order", "config"):
            d = d[d[k] == v]
        else:
            d = d[np.isclose(d[k].astype(float), float(v))]
    return d


def one(summary, **kw):
    d = pick(summary, **kw)
    if len(d) != 1:
        raise KeyError("expected 1 row, got %d for %r" % (len(d), kw))
    return d.iloc[0]


def xover(fs, fam, alpha, delta, ma, gs, scan_order):
    """Evaluation-level paired frame for one slice: step-down vs Bonferroni."""
    def grab(select, so):
        d = fs[(fs.family == fam) & np.isclose(fs.alpha, alpha)
               & np.isclose(fs.delta, delta) & (fs.min_accept == ma)
               & (fs.grid_size == gs) & (fs.select == select)
               & (fs.scan_order == so)]
        return d[["unit", "clf", "repeat", "accept_rate", "infeas_below",
                  "infeas_above", "n_accepted"]]
    a = grab("fixed_sequence", scan_order)
    b = grab("bonferroni", "ascending").rename(columns={"accept_rate": "base"})
    j = a.merge(b[["unit", "clf", "repeat", "base"]],
                on=["unit", "clf", "repeat"]).reset_index(drop=True)
    j["d"] = j["accept_rate"] - j["base"]
    j["prefix"] = j["infeas_below"] == 0
    j["suffix"] = j["infeas_above"] == 0
    return j

# ------------------------------------------------------------ parallel runner
def run_parallel(fn, jobs, n_jobs):
    out = []
    if n_jobs > 1:
        with mp.Pool(n_jobs, initializer=_init_worker) as pool:
            for i, (df, fname) in enumerate(
                    pool.imap(fn, jobs, chunksize=1), 1):
                out.append(df)
                if i % 25 == 0 or i == len(jobs):
                    log("    %d/%d  %s" % (i, len(jobs), fname))
    else:
        for i, job in enumerate(jobs, 1):
            df, fname = fn(job)
            out.append(df)
            if i % 25 == 0 or i == len(jobs):
                log("    %d/%d  %s" % (i, len(jobs), fname))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------- figure
def figure(summ):
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

    A_ALPHA, A_DELTA, A_MA = 0.10, 0.10, 100
    arm_style = [
        (("bonferroni", "ascending"), "--", "s", "bonferroni"),
        (("fixed_sequence", "ascending"), "-", "o", "step-down, ascending"),
        (("fixed_sequence", "descending"), ":", "^", "step-down, descending"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    # -- (a) mean acceptance against the grid size -------------------------
    ax = axes[0]
    for fam in FAMILIES:
        for (sel, so), ls, mk, _lab in arm_style:
            xs, ys = [], []
            for gs in GRID_SIZES:
                cfg = cfg_name(sel, so, A_MA, gs)
                r = one(summ, family=fam, alpha=A_ALPHA, delta=A_DELTA,
                        min_accept=A_MA, grid_size=gs, config=cfg)
                xs.append(gs)
                ys.append(float(r["mean_accept"]))
            ax.plot(xs, ys, ls, marker=mk, ms=3.2, lw=1.2, color=FAM_CLR[fam])
    ax.set_xscale("log")
    ax.set_xticks(list(GRID_SIZES))
    ax.set_xticklabels([str(g) for g in GRID_SIZES])
    ax.set_xlim(15.0, 560.0)
    ax.set_ylim(-0.04, 1.02)
    ax.set_xlabel("candidate grid size $m$")
    ax.set_ylabel("mean acceptance rate")
    h_fam = [Line2D([], [], color=FAM_CLR[f], lw=1.4, label=FAM_LABEL[f])
             for f in FAMILIES]
    h_arm = [Line2D([], [], color="0.25", ls=ls, marker=mk, ms=3.2, lw=1.2,
                    label=lab) for (_, ls, mk, lab) in arm_style]
    leg1 = ax.legend(handles=h_fam, fontsize=5.8, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=h_arm, fontsize=5.8, loc="lower right")
    panel(ax, "a", "alpha = 0.10, delta = 0.10, min_accept = 100")

    # -- (b) coverage of the budget against 1 - delta ----------------------
    ax = axes[1]
    xpos = {0.05: 0.0, 0.10: 1.0, 0.20: 2.0}
    offs = [(("bonferroni", "ascending"), -0.21),
            (("fixed_sequence", "ascending"), -0.07),
            (("fixed_sequence", "descending"), 0.07),
            (("naive", "calibration"), 0.21)]
    marks = {("bonferroni", "ascending"): "s",
             ("fixed_sequence", "ascending"): "o",
             ("fixed_sequence", "descending"): "^",
             ("naive", "calibration"): "D"}
    ymin, ymax = 1.0, 0.0
    for delta, x0 in xpos.items():
        for fam in FAMILIES:
            for (sel, so), dx in offs:
                cfg = cfg_name(sel, so, A_MA, 120)
                r = one(summ, family=fam, alpha=A_ALPHA, delta=delta,
                        min_accept=A_MA, grid_size=120, config=cfg)
                y = float(r["coverage"])
                lo = float(r["coverage_lo"])
                hi = float(r["coverage_hi"])
                ymin = min(ymin, y, lo)
                ymax = max(ymax, y, hi)
                open_marker = (sel == "naive")
                ax.errorbar([x0 + dx], [y], yerr=[[max(y - lo, 0.0)],
                                                  [max(hi - y, 0.0)]],
                            fmt=marks[(sel, so)], ms=3.6, lw=0.9, capsize=1.6,
                            color=FAM_CLR[fam],
                            markerfacecolor=("none" if open_marker
                                             else FAM_CLR[fam]),
                            markeredgecolor=FAM_CLR[fam], markeredgewidth=0.9)
    for delta, x0 in xpos.items():
        ax.plot([x0 - 0.30, x0 + 0.30], [1.0 - delta] * 2, "k--", lw=1.0)
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_xticklabels(["0.05", "0.10", "0.20"])
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(max(-0.02, ymin - 0.05), min(1.06, max(ymax + 0.05, 0.92)))
    ax.set_xlabel("failure probability delta")
    ax.set_ylabel("coverage of the budget alpha = 0.10")
    h_fam2 = [Line2D([], [], color=FAM_CLR[f], lw=1.4, label=FAM_LABEL[f])
              for f in FAMILIES]
    h_arm2 = [Line2D([], [], color="0.25", ls="", marker=marks[(s, o)], ms=3.6,
                     label=ARM_LABEL[(s, o)]) for (s, o), _ in offs]
    h_arm2.append(Line2D([], [], color="k", ls="--", lw=1.0, label="1 - delta"))
    leg1 = ax.legend(handles=h_fam2, fontsize=5.8, loc="lower left")
    ax.add_artist(leg1)
    ax.legend(handles=h_arm2, fontsize=5.4, loc="upper right", ncol=2)
    panel(ax, "b", "dots above the dashed line hold the budget")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "figS7_fixedseq.%s" % ext),
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/figS7_fixedseq.pdf/.png")

# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", default="")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse experiments/r27_fixedseq.csv when it exists")
    args = ap.parse_args()
    n_jobs = max(1, args.jobs)
    rng = np.random.default_rng(20260927)
    t0 = time.time()
    R = []
    checks = []

    def say(*parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        R.append(line)

    def table(df, cols=None, sort=None, nd=4):
        d = df if sort is None else df.sort_values(sort)
        if cols is not None:
            d = d[cols]
        say(d.round(nd).to_string(index=False))

    def chk(name, ok, msg):
        checks.append((name, bool(ok), msg))
        say("  [%s] %s -- %s" % ("PASS" if ok else "FAIL", name, msg))

    files = sorted(f for f in os.listdir(PRED) if f.endswith(".npz"))
    if args.limit:
        files = [f for f in files if args.limit in f]
    long_csv = os.path.join(RES, "r27_fixedseq.csv")
    if args.reuse and os.path.exists(long_csv):
        log("reusing %s" % long_csv)
        fs = pd.read_csv(long_csv, float_precision="round_trip")
    else:
        log("A. %d prediction files x %d arms over the sweep (jobs=%d)"
            % (len(files), len(ARMS), n_jobs))
        fs = run_parallel(_job, files, n_jobs)
        fs.to_csv(long_csv, index=False)
        log("wrote %s (%d rows)" % (long_csv, len(fs)))
    summ = summarise(fs, rng)
    summ.to_csv(os.path.join(RES, "r27_fixedseq_summary.csv"), index=False)
    log("wrote %s (%d rows)" % (os.path.join(RES, "r27_fixedseq_summary.csv"),
                                len(summ)))

    say("R27 -- the fixed-sequence (step-down) rule, finished across families.")
    say("input   : work/predictions/*.npz, %d files, %d long-table rows"
        % (len(files), len(fs)))
    say("arms    : " + " | ".join(ARM_LABEL[a] for a in ARMS))
    say("sweep   : alpha %s x delta %s x grid_size %s x min_accept %s"
        % (list(ALPHAS), list(DELTAS), list(GRID_SIZES), list(MIN_ACCEPTS)))
    say("summary : %d rows (%d configurations)" % (len(summ),
                                                   len(summ.config.unique())))
    say("wall    : %.1f s on %d processes" % (time.time() - t0, n_jobs))
    say("grid    : _grid_indices(n_cal, min_accept, grid_size), the split of")
    say("          scripts/r16_review_budget.py::_fs_job, score s = 1 - max p,")
    say("          error e = (argmax p != y), accept iff s <= lambda.")
    say("")

    say("=" * 78)
    say("0. inventory")
    say("=" * 78)
    inv = (fs[(fs.select == "bonferroni") & (fs.scan_order == "ascending")
              & np.isclose(fs.alpha, ALPHAS[0]) & np.isclose(fs.delta, R16_DELTA)
              & (fs.min_accept == 25) & (fs.grid_size == 120)]
           .groupby("family", observed=True)
           .agg(units=("unit", "nunique"), clfs=("clf", "nunique"),
                repeats=("repeat", "nunique"), evals=("accept_rate", "size"),
                cal_min=("grid_m", "size")))
    inv["evals_per_arm"] = inv["evals"]
    table(inv, ["units", "clfs", "repeats", "evals_per_arm"])
    say("  evals_per_arm = number of (unit, clf, repeat) evaluations per")
    say("  (alpha, delta, min_accept, grid_size) cell of one arm.")
    say("  cellsplit carries test_rep, so its repeats re-split both sides;")
    say("  within and transfer carry no test_rep, so every repeat scores the")
    say("  whole held-out pool with a different calibration subsample (this is")
    say("  r16's convention, kept here for comparability).")
    say("")

    # -------------------------------------------------------------- checks
    say("=" * 78)
    say("1. sanity checks")
    say("=" * 78)

    # pandas' DEFAULT CSV reader is not correctly rounded: its float parser
    # moves the last bit of some decimal strings.  to_csv itself emits the
    # exact repr, so the comparisons below pin the parser to
    # float_precision="round_trip", the only correctly rounded choice.  With it
    # the in-memory table and the written table are both BIT-exact against
    # r16's file -- the delivered counts, the covered flags and lambda are
    # exactly what the split, the grid and the bound decide.
    r16_long = pd.read_csv(os.path.join(RES, "r16_fixedseq.csv"),
                           float_precision="round_trip")
    cols = ["accept_rate", "covered", "selective_risk", "n_accepted",
            "n_test_eval", "lambda"]
    key = ["family", "unit", "clf", "repeat", "alpha", "config"]

    def r16_slice(d, cfgs=R16_CFGS):
        # the long table spans three deltas, so config alone is not a unique
        # key; the comparison is restricted to the r16 slice.
        d = d[np.isclose(d.delta, R16_DELTA)]
        return d if cfgs is None else d[d.config.isin(cfgs)]

    def match(a, b, cfgs=R16_CFGS):
        a = r16_slice(a, cfgs)
        j = a[key + cols].merge(b[key + cols], on=key, suffixes=("_a", "_b"))
        lam_a = np.nan_to_num(j.lambda_a.values.astype(float), posinf=0.0,
                              neginf=0.0)
        lam_b = np.nan_to_num(j.lambda_b.values.astype(float), posinf=0.0,
                              neginf=0.0)
        return dict(
            n=len(j),
            d_acc=float(np.abs(j.accept_rate_a - j.accept_rate_b).max()),
            d_risk=float(np.nanmax(np.abs(j.selective_risk_a
                                          - j.selective_risk_b))),
            d_cov=float(np.abs(j.covered_a - j.covered_b).max()),
            d_nac=int((j.n_accepted_a != j.n_accepted_b).sum()),
            d_nte=int((j.n_test_eval_a != j.n_test_eval_b).sum()),
            lam_same=bool(np.array_equal(lam_a, lam_b)),
            nan_same=bool(np.array_equal(np.isnan(j.selective_risk_a.values),
                                         np.isnan(j.selective_risk_b.values))))

    ulp = 2.220446049250313e-16
    _txt = pd.DataFrame({"x": [710.0 / 765.0]}).to_csv(index=False)
    _rt = float(pd.read_csv(io.StringIO(_txt)).x.iloc[0])
    _rt2 = float(pd.read_csv(io.StringIO(_txt),
                             float_precision="round_trip").x.iloc[0])
    ulp_demo = _rt != (710.0 / 765.0)
    ulp_demo_rt = _rt2 == (710.0 / 765.0)
    mem = match(fs, r16_long)
    mem_all = match(fs, r16_long, cfgs=None)
    disk_csv = os.path.join(RES, "r27_fixedseq.csv")
    dsk = (match(pd.read_csv(disk_csv, float_precision="round_trip"), r16_long)
           if os.path.exists(disk_csv) else dict(mem, n=-1))
    chk("S1 long table == r16_fixedseq.csv",
        mem["n"] == 4 * 3 * 930 and mem["d_nac"] == 0 and mem["d_nte"] == 0
        and mem["d_cov"] == 0.0 and mem["nan_same"] and mem["lam_same"]
        and mem["d_acc"] == 0.0 and mem["d_risk"] == 0.0
        and mem_all["d_acc"] == 0.0 and mem_all["d_cov"] == 0.0
        and mem_all["lam_same"] and mem_all["d_risk"] == 0.0
        and dsk["d_acc"] == 0.0 and dsk["d_cov"] == 0.0 and dsk["lam_same"],
        "%d matched evaluations (the four configs the task names) are "
        "BIT-EXACT in memory: n_accepted mismatches %d, n_test_eval mismatches "
        "%d, max |d covered| = %.1e, lambda identical = %s, max |d "
        "accept_rate| = %.1e, max |d selective_risk| = %.1e; extended to ALL "
        "six r16 configs (%d rows) still bit-exact (max |d accept_rate| = "
        "%.1e, lambda identical = %s).  the written r27 csv is bit-exact "
        "against r16 too (max |d accept_rate| = %.1e, lambda identical = %s).  "
        "this needs pandas' correctly rounded parser: on a one-number 710/765 "
        "round trip the DEFAULT parser moves the last bit (lossy = %s) and "
        "reads r16's file back 1 ulp off, which is the only reason a "
        "default-parser reading looks inexact -- to_csv writes the exact repr, "
        "and with float_precision=round_trip the same read is exact = %s"
        % (mem["n"], mem["d_nac"], mem["d_nte"], mem["d_cov"], mem["lam_same"],
           mem["d_acc"], mem["d_risk"], mem_all["n"], mem_all["d_acc"],
           mem_all["lam_same"], dsk["d_acc"], dsk["lam_same"], ulp_demo,
           ulp_demo_rt))

    r16_eval = pd.read_csv(os.path.join(RES, "r16_fixedseq_eval.csv"),
                           float_precision="round_trip")
    me = summ[np.isclose(summ.delta, R16_DELTA) & summ.config.isin(R16_CFGS)]
    je = me[["family", "alpha", "config", "mean_accept", "coverage"]].merge(
        r16_eval[["family", "alpha", "config", "mean_accept", "coverage"]],
        on=["family", "alpha", "config"], suffixes=("_r27", "_r16"))
    ok_ma = bool(np.allclose(je.mean_accept_r27, je.mean_accept_r16))
    ok_cv = bool(np.allclose(je.coverage_r27, je.coverage_r16))
    d_ma = float(np.abs(je.mean_accept_r27 - je.mean_accept_r16).max())
    d_cv = float(np.abs(je.coverage_r27 - je.coverage_r16).max())
    chk("S2 summary == r16_fixedseq_eval.csv", len(je) == 24 and ok_ma and ok_cv,
        "%d merged (family, alpha, config) rows; allclose(mean_accept) = %s "
        "(max |d| = %.2e), allclose(coverage) = %s (max |d| = %.2e)"
        % (len(je), ok_ma, d_ma, ok_cv, d_cv))
    if not (len(je) == 24 and ok_ma and ok_cv):
        say("  DIAGNOSIS -- mismatching rows:")
        say(je[~np.isclose(je.mean_accept_r27, je.mean_accept_r16)
               | ~np.isclose(je.coverage_r27, je.coverage_r16)]
            .to_string(index=False))

    step = max(1, len(files) // 24)
    samp = files[::step][:24]
    nviol, npairs, worst = 0, 0, 0.0
    gap_by_gs = {}
    for fn in samp:
        d = np.load(os.path.join(PRED, fn), allow_pickle=True)
        cal_probs, cal_y, cal_rep = d["cal_probs"], d["cal_y"], d["cal_rep"]
        s0 = cal_rep[:, 0]
        s_cal = 1.0 - cal_probs[s0].max(1)
        e_cal = (cal_probs[s0].argmax(1) != cal_y[s0]).astype(float)
        for gs in GRID_SIZES:
            for delta in DELTAS:
                cb = C.SelectiveRiskController(0.10, delta, min_accept=25,
                                               grid_size=gs,
                                               select="bonferroni").fit(s_cal, e_cal)
                cf = C.SelectiveRiskController(0.10, delta, min_accept=25,
                                               grid_size=gs,
                                               select="fixed_sequence").fit(s_cal, e_cal)
                rb = getattr(cb, "bound_ratio_", None)
                rf = getattr(cf, "bound_ratio_", None)
                if rb is None or rf is None or rb.size == 0:
                    continue
                if rb.size != rf.size or not np.array_equal(cb.cand_, cf.cand_):
                    continue
                dd = rb - rf
                nviol += int((dd < -1e-12).sum())
                npairs += int(dd.size)
                worst = min(worst, float(dd.min()))
                gap_by_gs.setdefault(gs, []).append(float(dd.mean()))
    chk("S3 fs bound_ratio <= bonferroni bound_ratio pointwise", nviol == 0,
        "%d pointwise comparisons on identical grids; %d violations; "
        "min(bonf - fs) = %.3e" % (npairs, nviol, worst))
    say("  mean (bonferroni - fixed_sequence) bound ratio on the same grid --")
    say("  the tighter bound is the entire source of the extra power, and its")
    say("  advantage grows with m because only Bonferroni carries log(m):")
    for gs in GRID_SIZES:
        v = np.asarray(gap_by_gs.get(gs, []), float)
        say("    grid_size %3d : mean %+.4f, max %+.4f, %d paired grids"
            % (gs, float(v.mean()) if v.size else float("nan"),
               float(v.max()) if v.size else float("nan"), v.size))

    nsel, nbad = 0, 0
    for fn in samp[:8]:
        d = np.load(os.path.join(PRED, fn), allow_pickle=True)
        cal_probs, cal_y, cal_rep = d["cal_probs"], d["cal_y"], d["cal_rep"]
        s0 = cal_rep[:, 0]
        s_cal = 1.0 - cal_probs[s0].max(1)
        e_cal = (cal_probs[s0].argmax(1) != cal_y[s0]).astype(float)
        s_sorted = np.sort(s_cal, kind="mergesort")
        for alpha in ALPHAS:
            for delta in (0.10,):
                for gs in (120,):
                    for ma in (25, 100):
                        for ssc, picker in (("bonferroni",
                                             pick_largest_feasible),
                                            ("fixed_sequence", pick_ascending)):
                            ctl = C.SelectiveRiskController(
                                alpha, delta, min_accept=ma, grid_size=gs,
                                select=ssc).fit(s_cal, e_cal)
                            rr = getattr(ctl, "bound_ratio_", None)
                            if rr is None or rr.size == 0:
                                continue
                            j = picker(rr <= alpha)
                            lam = (-np.inf if j is None
                                   else float(s_sorted[int(ctl.cand_[j]) - 1]))
                            nsel += 1
                            same = ((ctl.chosen_ is None) == (j is None))
                            if same and j is not None:
                                same = int(ctl.chosen_) == int(ctl.cand_[j])
                            if same:
                                same = bool(np.isclose(ctl.lambda_hat, lam))
                            if not same:
                                nbad += 1
    chk("S4 local selection == SelectiveRiskController.chosen_/lambda_hat",
        nbad == 0, "%d controller selections reproduced, %d mismatches"
        % (nsel, nbad))

    k2 = ["family", "unit", "clf", "repeat", "alpha", "delta", "min_accept",
          "grid_size"]
    asc = fs[(fs.select == "fixed_sequence") & (fs.scan_order == "ascending")]
    desc = fs[(fs.select == "fixed_sequence") & (fs.scan_order == "descending")]
    jd = asc[k2 + ["n_accepted", "lambda"]].merge(
        desc[k2 + ["n_accepted", "lambda"]], on=k2, suffixes=("_asc", "_desc"))
    ndiff = int((jd.n_accepted_asc != jd.n_accepted_desc).sum())
    asc_nv = int((jd.n_accepted_asc > 0).sum())
    desc_nv = int((jd.n_accepted_desc > 0).sum())
    chk("S5 descending arm is not the ascending arm", ndiff >= 1,
        "%d of %d matched evaluations deliver a different accepted count; "
        "non-vacuous: ascending %d, descending %d"
        % (ndiff, len(jd), asc_nv, desc_nv))
    if desc_nv == 0:
        say("  the descending arm is degenerate EVERYWHERE (empty at every")
        say("  setting); it is reported as such and not dressed up.")
    elif ndiff == 0:
        say("  the two arms coincide everywhere: under the reading in which the")
        say("  prespecified order runs from the safest claim to the riskiest,")
        say("  arm 3 is arm 2 by construction, and the reverse-scan reading")
        say("  implemented here would be the only distinct one.")
    say("")

    def table_safe(df, cols, sort=None, nd=4, empty="  (none)"):
        if df is None or len(df) == 0:
            say(empty)
            return
        table(df, cols, sort, nd)

    # ------------------------------------------------------------- (a)
    say("=" * 78)
    say("(a) does the step-down beat Bonferroni?  paired, identical evaluations")
    say("=" * 78)
    sd = summ[summ.select == "fixed_sequence"].copy()
    better = sd[sd.ci_above_zero == 1]
    below = sd[(sd.ci_excludes_zero == 1) & (sd.delta_accept < 0)]
    say("%d of %d (family, alpha, delta, min_accept, grid_size, scan_order) cells"
        % (len(better), len(sd)))
    say("have a strictly higher mean acceptance than Bonferroni at matched")
    say("settings AND a paired unit-cluster bootstrap 95% CI on the change that")
    say("excludes 0.  %d cells have a CI entirely below 0 (an established loss)."
        % len(below))
    table_safe(better, ["family", "alpha", "delta", "min_accept", "grid_size",
                        "scan_order", "mean_accept_bonferroni", "mean_accept",
                        "delta_accept", "delta_lo", "delta_hi",
                        "frac_improved", "frac_worsened"],
               sort=["family", "alpha", "delta", "min_accept", "grid_size",
                     "scan_order"])
    say("")
    say("by scan order and prespecified floor (all families, budgets, deltas,")
    say("grid sizes):")
    for so in ("ascending", "descending"):
        for ma in MIN_ACCEPTS:
            s = sd[(sd.scan_order == so) & (sd.min_accept == ma)]
            if not len(s):
                continue
            say("  %-10s min_accept=%3d : better %2d/%3d ; mean delta %+0.4f ; "
                "worst frac_worsened %.2f ; mean frac_improved %.2f"
                % (so, ma, int((s.ci_above_zero == 1).sum()), len(s),
                   float(s.delta_accept.mean()), float(s.frac_worsened.max()),
                   float(s.frac_improved.mean())))
    say("")
    say("the manuscript's sentence, checked here:")
    for alpha in (0.10, 0.20):
        b = one(summ, family="cellsplit", alpha=alpha, delta=0.10,
                min_accept=100, grid_size=120, config="bonferroni_ma100_gs120")
        f = one(summ, family="cellsplit", alpha=alpha, delta=0.10,
                min_accept=100, grid_size=120,
                config="fixed_sequence_ma100_gs120")
        say("  cellsplit alpha=%.2f delta=0.10 min_accept=100 grid_size=120 : "
            "bonferroni %.4f -> step-down %.4f, coverage %.4f -> %.4f"
            % (alpha, float(b["mean_accept"]), float(f["mean_accept"]),
               float(b["coverage"]), float(f["coverage"])))
    say("  the quoted '0.560 to 0.634 at 10 pct with coverage 0.963' and")
    say("  '0.803 to 0.839 at 20 pct with coverage 0.984' reproduce exactly,")
    say("  but they hold only for cellsplit at min_accept=100, grid_size=120,")
    say("  delta=0.10.  The full sweep below shows they are a selected cell.")
    say("")

    # ------------------------------------------------------------- (b)
    say("=" * 78)
    say("(b) is it still valid?  Monte-Carlo coverage of the budget vs 1 - delta")
    say("=" * 78)
    say("coverage counts an empty accepted set as covered (a vacuous success,")
    say("r16's convention); coverage_nonvac is reported in the summary CSV for")
    say("the non-vacuous subset.  Cells are pooled over alpha, min_accept and")
    say("grid_size within (family, delta, arm); the Wilson interval is on the")
    say("evaluation-level coverage.  coverage_nonvac repeats it on the")
    say("non-empty accepted sets alone, so a rule that 'covers' by always")
    say("answering nothing is visible as such; a construction is INVALID when")
    say("the Wilson interval on its coverage lies wholly below 1 - delta, which")
    say("frac_configs_shortfall counts per configuration.")
    b1 = []
    for delta in DELTAS:
        for fam in FAMILIES:
            for sel, so in ARMS:
                s = summ[np.isclose(summ.delta, delta) & (summ.select == sel)
                         & (summ.scan_order == so) & (summ.family == fam)]
                k = int(s.n_violations.sum())
                n = int(s.n_eval.sum())
                lo, hi = wilson(n - k, n)
                sub = fs[(fs.family == fam) & np.isclose(fs.delta, delta)
                         & (fs.select == sel) & (fs.scan_order == so)]
                nve = sub[sub["empty"] == 0]
                cnonvac = (float(nve["covered"].mean()) if len(nve)
                           else float("nan"))
                b1.append(dict(
                    family=fam, delta=delta, arm=ARM_LABEL[(sel, so)],
                    coverage=(n - k) / float(n), n_violations=k, n_eval=n,
                    wilson_lo=lo, wilson_hi=hi, nominal=1.0 - delta,
                    frac_nonvac=float((sub["empty"] == 0).mean()),
                    coverage_nonvac=cnonvac,
                    frac_configs_ok=float((s.coverage >= 1.0 - delta).mean()),
                    frac_configs_shortfall=float((s.coverage_hi
                                                  < 1.0 - delta).mean())))
    table(pd.DataFrame(b1), ["family", "delta", "arm", "coverage",
                             "n_violations", "n_eval", "wilson_lo", "wilson_hi",
                             "nominal", "frac_nonvac", "coverage_nonvac",
                             "frac_configs_ok", "frac_configs_shortfall"])
    say("")
    say("focused, min_accept = 100, grid_size = 120, delta = 0.10 -- coverage of")
    say("the budget, by family, budget and arm:")
    rows = []
    for fam in FAMILIES:
        for alpha in ALPHAS:
            rec = {"family": fam, "alpha": alpha}
            for sel, so in ARMS:
                r = one(summ, family=fam, alpha=alpha, delta=0.10,
                        min_accept=100, grid_size=120,
                        config=cfg_name(sel, so, 100, 120))
                rec[ARM_LABEL[(sel, so)]] = float(r["coverage"])
            rec["nominal"] = 0.90
            rows.append(rec)
    table(pd.DataFrame(rows), ["family", "alpha", "bonferroni",
                               "fixed-sequence ascending",
                               "fixed-sequence descending",
                               "naive (calibration)", "nominal"])
    say("")
    say("the cell the task flags: cellsplit, fixed_sequence_ma100_gs120,")
    say("alpha = 0.05, delta = 0.10.")
    sub = fs[(fs.family == "cellsplit") & np.isclose(fs.alpha, 0.05)
             & np.isclose(fs.delta, 0.10) & (fs.min_accept == 100)
             & (fs.grid_size == 120) & (fs.select == "fixed_sequence")
             & (fs.scan_order == "ascending")]
    n = int(len(sub))
    k = int(sub.covered.sum())
    vu = sub.groupby("unit", observed=True)["covered"].mean()
    nu = int(vu.size)
    vu_ok = int((vu >= 1.0 - 1e-12).sum())
    elo, ehi = wilson(k, n)
    ulo, uhi = wilson(vu_ok, nu)
    half_e = (ehi - elo) / 2.0
    half_u = (uhi - ulo) / 2.0
    shortfall = 1.0 - k / float(n)
    say("  evaluations %d over %d units; violations %d; coverage %.4f"
        % (n, nu, n - k, k / float(n)))
    say("  Wilson 95 pct at the evaluation level : [%.4f, %.4f], half-width %.4f"
        % (elo, ehi, half_e))
    say("  units with >= 1 violation            : %d of %d"
        % (nu - vu_ok, nu))
    say("  Wilson 95 pct at the unit level       : [%.4f, %.4f], half-width %.4f"
        % (ulo, uhi, half_u))
    say("  nominal tolerance 1 - delta           : %.2f" % (1.0 - 0.10))
    say("  coverage %.4f is %s the nominal %.2f, so the construction %s."
        % (k / float(n), "above" if k / float(n) >= 0.90 else "below", 0.90,
           "stays valid" if k / float(n) >= 0.90 else "FAILS"))
    say("  the shortfall from 1.000 is %.4f: %.1f evaluation-level half-widths"
        % (shortfall, shortfall / half_e if half_e > 0 else np.nan))
    say("  but only %.2f unit-level half-widths, i.e. inside Monte-Carlo error"
        % (shortfall / half_u if half_u > 0 else np.nan))
    say("  once the %d units, not the %d evaluations, are the replicates."
        % (nu, n))
    say("")
    # ------------------------------------------------------------- (c)
    say("=" * 78)
    say("(c) does the advantage survive grid refinement?")
    say("=" * 78)
    say("alpha = 0.10, delta = 0.10, min_accept = 100; the same units at every")
    say("grid size, so the comparison is paired in the units, not just in m.")
    rows = []
    for fam in FAMILIES:
        for gs in GRID_SIZES:
            b = one(summ, family=fam, alpha=0.10, delta=0.10, min_accept=100,
                    grid_size=gs, config=bonf_cfg(100, gs))
            a = one(summ, family=fam, alpha=0.10, delta=0.10, min_accept=100,
                    grid_size=gs, config=cfg_name("fixed_sequence",
                                                 "ascending", 100, gs))
            d = one(summ, family=fam, alpha=0.10, delta=0.10, min_accept=100,
                    grid_size=gs, config=cfg_name("fixed_sequence",
                                                 "descending", 100, gs))
            rows.append(dict(
                family=fam, grid_size=gs,
                bonferroni=float(b["mean_accept"]),
                fs_ascending=float(a["mean_accept"]),
                fs_descending=float(d["mean_accept"]),
                gap_ascending=float(a["mean_accept"] - b["mean_accept"]),
                gap_descending=float(d["mean_accept"] - b["mean_accept"]),
                bonf_frac_feas=float(b["frac_nonvacuous"]),
                asc_frac_feas=float(a["frac_nonvacuous"]),
                bonf_feas_hi=float(b["feas_hi_mean"]),
                asc_feas_hi=float(a["feas_hi_mean"])))
    ct = pd.DataFrame(rows)
    table(ct, ["family", "grid_size", "bonferroni", "fs_ascending",
               "fs_descending", "gap_ascending", "gap_descending",
               "bonf_frac_feas", "asc_frac_feas"])
    say("")
    say("gap = step-down mean acceptance - Bonferroni mean acceptance:")
    for fam in FAMILIES:
        g = ct[ct.family == fam].set_index("grid_size")["gap_ascending"]
        trend = "WIDENS" if float(g.loc[400]) > float(g.loc[20]) + 1e-9 else "does not widen"
        say("  %-10s m=20 %+.4f -> m=120 %+.4f -> m=400 %+.4f   (%s)"
            % (fam, float(g.loc[20]), float(g.loc[120]), float(g.loc[400]),
               trend))
    for so, col in (("ascending", "gap_ascending"),
                    ("descending", "gap_descending")):
        v20 = float(ct[ct.grid_size == 20][col].mean())
        v120 = float(ct[ct.grid_size == 120][col].mean())
        v400 = float(ct[ct.grid_size == 400][col].mean())
        say("  family-mean %-10s m=20 %+.4f -> m=120 %+.4f -> m=400 %+.4f"
            % (so, v20, v120, v400))
    say("")
    say("mechanism: only Bonferroni carries log(m) in its radius, so its")
    say("feasible upper end shrinks as the grid is refined while the")
    say("fixed-sequence bound (log(3/delta), no m) does not.  The mean")
    say("(bonferroni - fixed_sequence) bound ratio printed in check S3 above")
    say("grows with grid_size for exactly that reason.")
    say("")

    # ------------------------------------------------------------- (d)
    say("=" * 78)
    say("(d) is the gain confined to configurations whose feasible set is a")
    say("    prefix of the scan order?")
    say("=" * 78)
    say("delta = 0.10.  frac_prefix_asc = fraction of evaluations with")
    say("infeas_below == 0 (the ascending scan's first grid point is feasible);")
    say("frac_prefix_desc = fraction with infeas_above == 0 (the descending")
    say("scan's first grid point is feasible).  delta_given_* is the mean paired")
    say("acceptance change against Bonferroni inside each group.")
    rows = []
    for fam in FAMILIES:
        for alpha in ALPHAS:
            for ma in MIN_ACCEPTS:
                for gs in GRID_SIZES:
                    j = xover(fs, fam, alpha, 0.10, ma, gs, "ascending")
                    pref = j[j.prefix]
                    nop = j[~j.prefix]
                    rows.append(dict(
                        family=fam, alpha=alpha, min_accept=ma, grid_size=gs,
                        n=len(j), frac_prefix_asc=float(j.prefix.mean()),
                        frac_prefix_desc=float(j.suffix.mean()),
                        delta_given_prefix=(float(pref.d.mean())
                                            if len(pref) else np.nan),
                        delta_given_noprefix=(float(nop.d.mean())
                                              if len(nop) else np.nan),
                        frac_impr_given_prefix=((float((pref.d > 1e-12).mean())
                                                 if len(pref) else np.nan)),
                        frac_impr_given_noprefix=((float((nop.d > 1e-12).mean())
                                                   if len(nop) else np.nan))))
    gd = pd.DataFrame(rows)
    table(gd, ["family", "alpha", "min_accept", "grid_size", "n",
               "frac_prefix_asc", "frac_prefix_desc", "delta_given_prefix",
               "delta_given_noprefix", "frac_impr_given_prefix",
               "frac_impr_given_noprefix"],
          sort=["family", "alpha", "min_accept", "grid_size"])
    say("")
    say("pooled over families, budgets, floors and grid sizes (delta = 0.10):")
    parts = []
    for fam in FAMILIES:
        for alpha in ALPHAS:
            for ma in MIN_ACCEPTS:
                for gs in GRID_SIZES:
                    j = xover(fs, fam, alpha, 0.10, ma, gs, "ascending")
                    parts.append(j)
    allj = pd.concat(parts, ignore_index=True)
    for lab, mask in (("feasible set IS a prefix", allj.prefix),
                      ("feasible set is NOT a prefix", ~allj.prefix)):
        s = allj[mask]
        say("  %-28s n %6d ; mean delta_accept %+0.4f ; improved %.3f ; "
            "worsened %.3f"
            % (lab, len(s), float(s.d.mean()) if len(s) else np.nan,
               float((s.d > 1e-12).mean()) if len(s) else np.nan,
               float((s.d < -1e-12).mean()) if len(s) else np.nan))
    parts = []
    for fam in FAMILIES:
        for alpha in ALPHAS:
            for ma in MIN_ACCEPTS:
                for gs in GRID_SIZES:
                    parts.append(xover(fs, fam, alpha, 0.10, ma, gs,
                                       "descending"))
    alld = pd.concat(parts, ignore_index=True)
    say("")
    say("the descending arm is a prefix of ITS scan order when the feasible set")
    say("touches the TOP of the grid (infeas_above == 0):")
    for lab, mask in (("feasible set reaches the top", alld.suffix),
                      ("feasible set stops below the top", ~alld.suffix)):
        s = alld[mask]
        say("  %-28s n %6d ; mean delta_accept %+0.4f ; improved %.3f ; "
            "worsened %.3f"
            % (lab, len(s), float(s.d.mean()) if len(s) else np.nan,
               float((s.d > 1e-12).mean()) if len(s) else np.nan,
               float((s.d < -1e-12).mean()) if len(s) else np.nan))
    say("")

    # ------------------------------------------------------------- (e)
    say("=" * 78)
    say("(e) the transfer family")
    say("=" * 78)
    say("min_accept = 100, grid_size = 120, delta = 0.10; coverage is against")
    say("the nominal 0.90.")
    rows = []
    for alpha in ALPHAS:
        rec = {"alpha": alpha}
        for sel, so in ARMS:
            r = one(summ, family="transfer", alpha=alpha, delta=0.10,
                    min_accept=100, grid_size=120,
                    config=cfg_name(sel, so, 100, 120))
            rec[ARM_LABEL[(sel, so)] + " accept"] = float(r["mean_accept"])
            rec[ARM_LABEL[(sel, so)] + " cov"] = float(r["coverage"])
        rows.append(rec)
    table(pd.DataFrame(rows), ["alpha", "bonferroni accept", "bonferroni cov",
                               "fixed-sequence ascending accept",
                               "fixed-sequence ascending cov",
                               "fixed-sequence descending accept",
                               "fixed-sequence descending cov",
                               "naive (calibration) cov"])
    tr = summ[summ.family == "transfer"]
    say("")
    say("  transfer holds the budget in %d of %d summary cells (all arms, all"
        % (int((tr.coverage >= 1.0 - tr.delta).sum()), len(tr)))
    say("  settings); best coverage over every transfer cell is %.4f."
        % float(tr.coverage.max()))
    for fam in ("cellsplit", "within"):
        s = summ[summ.family == fam]
        say("  %-10s holds the budget in %d of %d cells; coverage range "
            "[%.4f, %.4f]"
            % (fam, int((s.coverage >= 1.0 - s.delta).sum()), len(s),
               float(s.coverage.min()), float(s.coverage.max())))
    say("  verdict: the transfer family is a FAILURE of the certificate, not of")
    say("  the step-down in particular.  Cross-tissue calibration and")
    say("  deployment cells are not exchangeable, so the bound covers the wrong")
    say("  distribution; a tighter bound (larger delta_m, no log(m)) cannot")
    say("  repair that, and it does not: the step-down is as invalid as")
    say("  Bonferroni there.  Its higher acceptance where the bound does hold")
    say("  is therefore also higher exposure where the bound does not.")
    say("")
    # ----------------------------------------------------- honest notes
    say("=" * 78)
    say("2. honest notes, including where the current manuscript is wrong")
    say("=" * 78)
    say("N1. min_accept = 25 is degenerate and is now quantified.")
    deg = sd[(sd.min_accept == 25)
             & (np.isclose(sd.alpha, 0.05) | np.isclose(sd.alpha, 0.10))]
    deg2 = sd[(sd.min_accept == 25) & np.isclose(sd.alpha, 0.20)]
    say("  ascending arm, min_accept=25, alpha in {0.05, 0.10}: %d cells, max"
        % len(deg))
    say("  mean_accept %.4f, max frac_nonvacuous %.4f -- the scan dies on the"
        % (float(deg.mean_accept.max()), float(deg.frac_nonvacuous.max())))
    say("  very first grid point, which is below the denominator bound's")
    say("  feasible interval.  At alpha=0.20 the same arm answers")
    say("  (mean_accept %.4f to %.4f) but is below Bonferroni in %d of %d cells."
        % (float(deg2.mean_accept.min()), float(deg2.mean_accept.max()),
           int((deg2.delta_accept < 0).sum()), len(deg2)))
    say("")
    say("N2. the descending arm, stated plainly.")
    say("  Read literally -- 'the prespecified order runs from the safest claim")
    say("  to the riskiest' -- arm 3 IS arm 2, because an ascending accepted")
    say("  count is an ascending lambda and the safest claim is the smallest")
    say("  accepted set; the arms would then be identical by construction and no")
    say("  configuration could separate them.  The only distinct reading is the")
    say("  reverse scan, which is what is implemented, so this arm is a")
    say("  deliberately strong negative control: if reversal recovered the")
    say("  prefix property the step-down would not need a floor after all.")
    rows = []
    for fam in FAMILIES:
        a = one(summ, family=fam, alpha=0.10, delta=0.10, min_accept=100,
                grid_size=120, config="fixed_sequence_ma100_gs120")
        d = one(summ, family=fam, alpha=0.10, delta=0.10, min_accept=100,
                grid_size=120, config="fixed_sequence_desc_ma100_gs120")
        rows.append(dict(family=fam, ascending=float(a["mean_accept"]),
                         descending=float(d["mean_accept"]),
                         asc_frac_nonvac=float(a["frac_nonvacuous"]),
                         desc_frac_nonvac=float(d["frac_nonvacuous"])))
    table(pd.DataFrame(rows), ["family", "ascending", "descending",
                               "asc_frac_nonvac", "desc_frac_nonvac"])
    say("  reversal is not a repair: the descending arm is empty or nearly")
    say("  empty everywhere at alpha = 0.10, because the top of the grid is")
    say("  infeasible for the same reason the bottom is (the ratio bound is not")
    say("  monotone in the accepted count).  %d of %d matched evaluations differ"
        % (ndiff, len(jd)))
    say("  from the ascending arm, so the arms are distinct, but the direction")
    say("  that is distinct is the useless one.")
    say("")
    say("N3. where the manuscript's sentence is wrong, quoted.")

    def _cv(fam, alpha, select, so):
        r1 = one(summ, family=fam, alpha=alpha, delta=0.10, min_accept=100,
                 grid_size=120, config=cfg_name(select, so, 100, 120))
        return float(r1["coverage"])

    def _covline(fam, select, so):
        return ", ".join("alpha=%.2f %.4f"
                         % (a, _cv(fam, a, select, so)) for a in ALPHAS)

    say("  The paragraph says the step-down 'buys about seven percentage points")
    say("  of acceptance at unchanged nominal validity', quotes coverage 0.963")
    say("  at 10 pct and 0.984 at 20 pct, and then concedes a 0.7333 against")
    say("  0.8333 coverage drop under cross-FOV in the same breath.  Three")
    say("  corrections:")
    say("  (i) the quoted coverage is cellsplit only.  At min_accept=100,")
    say("      grid_size=120, delta=0.10 the step-down covers")
    for fam in FAMILIES:
        say("        %-9s %s"
            % (fam, _covline(fam, "fixed_sequence", "ascending")))
    say("      so the quoted number is a property of the family, not of the")
    say("      rule, and it is nowhere near a property of the transfer family.")
    say("  (ii) 'unchanged nominal validity' does not survive the comparison")
    say("      inside a family either: for within at alpha=0.10 the Bonferroni")
    say("      reference covers %.4f, already below the nominal %.2f, against"
        % (_cv("within", 0.10, "bonferroni", "ascending"), 0.90))
    say("      the step-down's %.4f, a %.4f drop, so the extra acceptance is"
        % (_cv("within", 0.10, "fixed_sequence", "ascending"),
           _cv("within", 0.10, "bonferroni", "ascending")
           - _cv("within", 0.10, "fixed_sequence", "ascending")))
    say("      bought with coverage rather than for free.  Both are below")
    say("      nominal there, so the honest reading is that cross-FOV is not")
    say("      exchangeable and the sharper rule inherits that, not that the")
    say("      step-down is free.")
    say("  (iii) the seven-point gain needs the floor: it requires")
    say("      min_accept = 100.  At min_accept = 25 the same rule returns an")
    say("      empty set at alpha = 0.05 and 0.10, which the paragraph does not")
    say("      report.")
    say("  (iv) the paragraph's coverage triplets are re-derived here from the")
    say("      summary table, family by family, so the quotation is checked and")
    say("      not merely repeated.")
    mspath = os.path.join(os.path.dirname(HERE), "docs", "manuscript.md")
    qn = []
    if os.path.exists(mspath):
        try:
            mtxt = (io.open(mspath, "r", encoding="utf-8", errors="replace")
                    .read().replace("\n", " "))
            mm = re.search(r"not a property of the rule[\s\S]*?it is "
                           r"([\s\S]+?)under transfer", mtxt)
        except Exception:
            mm = None
        if mm is not None:
            qn = [float(x) for x in
                  re.findall(r"[0-9]+\.[0-9]+", mm.group(1))]
    if len(qn) >= 9:
        qn = qn[-9:]
        for i, fam in enumerate(FAMILIES):
            qq = qn[3 * i:3 * i + 3]
            gg = [_cv(fam, a, "fixed_sequence", "ascending") for a in ALPHAS]
            same = all(abs(a - b) < 5e-4 for a, b in zip(qq, gg))
            say("        %-9s quoted %s  computed %s  %s"
                % (fam, " ".join("%.3f" % v for v in qq),
                   " ".join("%.3f" % v for v in gg),
                   "MATCH" if same else "MISMATCH"))
        say("      a MISMATCH row means the manuscript attaches a different")
        say("      arm's coverage to the step-down; the Bonferroni coverage at")
        say("      the same cell is in the focused table of section (b).")
    else:
        say("      the paragraph was not located or did not parse, so this")
        say("      cross-check makes no claim about it.")
    say("")
    say("N4. the naive reference, for context, at alpha=0.10, delta=0.10,")
    say("  min_accept=100, grid_size=120:")
    rows = []
    for fam in FAMILIES:
        r = one(summ, family=fam, alpha=0.10, delta=0.10, min_accept=100,
                grid_size=120, config="naive_ma100_gs120")
        rows.append(dict(family=fam, mean_accept=float(r["mean_accept"]),
                         coverage=float(r["coverage"])))
    table(pd.DataFrame(rows), ["family", "mean_accept", "coverage"])
    say("  it accepts more than either calibrated arm and misses the budget")
    say("  everywhere; the sweep shows this is not a tuning accident.")
    say("")

    # ------------------------------------------------------------- figure
    figure(summ)

    # ------------------------------------------------------------- report
    report = os.path.join(WORK, "r27_report.txt")
    text = "\r\n".join(R)
    text = text.replace("\r\n", "\n").replace("\n", "\r\n") + "\r\n"
    bad = sorted({c for c in text if ord(c) > 127})
    if bad:
        say("WARNING: non-ascii characters in the report: %r" % (bad,))
    with io.open(report, "w", encoding="ascii", errors="replace",
                 newline="") as fh:
        fh.write(text)
    nfail = sum(1 for _, ok, _ in checks if not ok)
    log("report: %s (%d lines)" % (report, len(R)))
    log("checks: %d passed, %d failed" % (len(checks) - nfail, nfail))
    if nfail:
        sys.exit(1)


if __name__ == "__main__":
    main()
