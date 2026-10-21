# -*- coding: utf-8 -*-
"""Panel drawing code for the r66_svg_c2 SVG figure set.

Every function below is the body of one panel of one of the original figure
functions, moved out unchanged so that exactly the same drawing code produces
the per-panel SVGs and the combined figures:

    r16_review_budget.figure        -> f7a, f7b
    r22_fig8 (the writer of the
      on-disk figures/fig8_*.png)   -> f8_tables, f8_triage, f8a, f8b, f8c
    r21_downstream_repair.make_figure -> f9a, f9b
    r24_efficiency.make_figure      -> f10a, f10b
    r25_accept_deferred.plot_units  -> fs6a, fs6b
    r27_fixedseq.figure             -> fs7a, fs7b

Numbers, axis limits, ticks, legends and annotations are the ones those
functions draw today; the only edits are mechanical (the panel's own title is
not set here -- r66_svg_c2.py applies the shared house title, so every panel of
every figure carries the same title font size -- and figure-level defaults are
left to the caller).

r22_fig8.py is a script whose figure code runs at import time and would
overwrite figures/fig8_label_ladder.{pdf,png}, so it is NOT imported: its
constants and its two data-preparation steps are copied here verbatim.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _load(name):
    """Import one of the original figure modules, once, by name.

    The module is registered in sys.modules under its own name before it is
    executed: r16_review_budget draws its matched-risk jobs with
    multiprocessing, and the pickler resolves ``r16_review_budget._matched_job``
    through sys.modules -- without the registration it re-imports the file,
    finds a different function object and refuses to pickle it.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R16 = _load("r16_review_budget")
R21 = _load("r21_downstream_repair")
R24 = _load("r24_efficiency")
R25 = _load("r25_accept_deferred")
R27 = _load("r27_fixedseq")


# ===================================================================== fig 7
def f7a(ax, curve):
    """r16_review_budget.figure, left panel: review fraction against alpha."""
    alphas = sorted(curve.alpha.unique())
    dy = {"cellsplit": -15, "within": 10, "transfer": 10}
    box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
    for fam in R16.FAMILIES:
        col = R16.FAM_CLR[fam]
        s = curve[(curve.family == fam)
                  & (curve.method == "crc")].sort_values("alpha")
        o = curve[(curve.family == fam)
                  & (curve.method == "oracle")].sort_values("alpha")
        ax.plot(o.alpha, o.review_frac, ":", lw=1.0, color=col, alpha=0.9)
        ax.plot(s.alpha, s.review_frac, "o-", ms=3.4, lw=1.3, color=col,
                label=R16.FAM_LABEL[fam])
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
    ax.grid(alpha=0.25)


def f7b(ax, ms, ratio):
    """r16_review_budget.figure, right panel: acceptance at matched risk."""
    styles = (("crc", "-", "o", 1.5), ("oracle_peek", ":", "o", 1.1),
              ("naive", "--", "s", 1.1))
    for fam in R16.FAMILIES:
        col = R16.FAM_CLR[fam]
        for meth, ls, mk, lw in styles:
            s = ms[(ms.family == fam)
                   & (ms.method == meth)].sort_values("risk")
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
    for i, fam in enumerate(R16.FAMILIES):
        r = ratio[ratio.family == fam].sort_values("risk")
        v = r[np.isclose(r.risk, 0.10)]
        if len(v):
            ax.text(0.397, 0.105 - 0.043 * i,
                    "%s  %.2f" % (R16.FAM_LABEL[fam],
                                  float(v.ratio_crc_oracle.iloc[0])),
                    fontsize=5.6, ha="right", va="bottom", color=R16.FAM_CLR[fam])
    ax.set_xlim(0.03, 0.40)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("matched realised risk r")
    ax.set_ylabel("acceptance")
    ax.grid(alpha=0.25)


# ===================================================================== fig 8
# r22_fig8.py constants, copied verbatim.
F8_AL = 0.10
F8_FAMS = ["cellsplit", "within", "transfer"]
F8_LBL = {"cellsplit": "random split (exchangeable)",
          "within": "cross-field-of-view (same cohort)",
          "transfer": "cross-tissue transfer"}
F8_COL = {"cellsplit": "#2b6cb0", "within": "#dd6b20", "transfer": "#c53030"}


def _f8_grp(df, by):
    """r22_fig8.grp, verbatim."""
    g = df.groupby(by).agg(acc=("accept_rate", "mean"), nvf=("nonvac", "mean"),
                           ok=("covered_nonvac", "sum"), nv=("nonvac", "sum")
                           ).reset_index()
    g["covnv"] = g.ok / g.nv.replace(0, np.nan)
    return g


def f8_tables(u, r):
    """r22_fig8.py lines 27-45: the target / oracle / source tables.

    ``u`` is experiments/r20_units.csv sliced to min_accept 25, grid 'abs' and
    alpha 0.10 with a 'nonvac' column; ``r`` is experiments/r14_targetcal.csv
    sliced to the transfer unit with 'nonvac' and 'covered_nonvac' columns.
    """
    T = _f8_grp(u[u.arm == "tgt_k"], ["family", "gkey"]).rename(
        columns={"gkey": "k"})
    O = _f8_grp(u[u.arm == "oracle"], ["family", "gkey"]).rename(
        columns={"gkey": "k"})
    S = _f8_grp(u[u.arm == "src"], ["family", "gkey"]).rename(
        columns={"gkey": "k"})
    Tr = _f8_grp(r[r.arm == "tgt_k"], ["k"])
    Tr["family"] = "transfer"
    Or = _f8_grp(r[r.arm == "oracle"], ["k"])
    Or["family"] = "transfer"
    Sr = _f8_grp(r[r.arm == "src"], ["k"])
    Sr["family"] = "transfer"
    T = pd.concat([T[["family", "k", "acc", "nvf", "covnv"]],
                   Tr[["family", "k", "acc", "nvf", "covnv"]]], ignore_index=True)
    O = pd.concat([O[["family", "k", "acc"]], Or[["family", "k", "acc"]]],
                  ignore_index=True)
    S = pd.concat([S[["family", "k", "acc"]], Sr[["family", "k", "acc"]]],
                  ignore_index=True)
    return T, O, S


def f8_triage(M):
    """r22_fig8.py lines 76-91: the triage table plus the numbers it prints."""
    M = M[np.isclose(M.alpha, F8_AL)].copy()
    M["cens"] = M.censored == 1
    cap = 1.6 * float(M.k_max.max())
    M["y"] = np.where(M.cens, cap, M.k_reach)
    x = M.ks_scores.to_numpy(float)
    impy = np.where(M.cens.to_numpy(), M.k_max.to_numpy(float),
                    M.k_reach.to_numpy(float))
    rho, pv = stats.spearmanr(x, impy)
    z = int(((M.accept_src == 0) & M.cens).sum())
    zn = int((M.accept_src == 0).sum())
    info = dict(cap=cap, rho=float(rho), pv=float(pv), z=z, zn=zn, n=len(M),
                n_cens=int(M.cens.sum()))
    return M, info


def f8a(ax, T, O, S):
    """r22_fig8.py panel (a): acceptance vs target labels k."""
    for f in F8_FAMS:
        a = T[T.family == f].sort_values("k")
        b = O[O.family == f].sort_values("k")
        ax.plot(a.k, a.acc, "-o", ms=3.5, color=F8_COL[f], label=F8_LBL[f])
        ax.plot(b.k, b.acc, "--", lw=1.1, color=F8_COL[f], alpha=0.7)
        s0 = S[(S.family == f) & (S.k == 0)]
        if len(s0):
            ax.plot([0.55], [s0.acc.iloc[0]], marker="s", ms=6.5,
                    color=F8_COL[f], mec="k", mew=0.6)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(-0.35, 12000)
    ax.set_xticks([0, 10, 100, 1000, 8000])
    ax.set_xticklabels(["0", "10", "100", "1000", "8000"], fontsize=8)
    ax.set_xlabel("target-domain labels $k$")
    ax.set_ylabel("acceptance rate")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)


def f8b(ax, T):
    """r22_fig8.py panel (b): fraction of evaluations that answer at all."""
    for f in F8_FAMS:
        a = T[T.family == f].sort_values("k")
        ax.plot(a.k, a.nvf, "-o", ms=3.5, color=F8_COL[f])
    ax.axhline(0.05, color="k", ls=":", lw=1)
    ax.text(1400, 0.062, "usefulness floor 5%", fontsize=7)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(-0.35, 12000)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xticks([0, 10, 100, 1000, 8000])
    ax.set_xticklabels(["0", "10", "100", "1000", "8000"], fontsize=8)
    ax.set_xlabel("target-domain labels $k$")
    ax.set_ylabel("fraction of evaluations that answer at all")
    ax.grid(alpha=0.25)


def f8c(ax, M, info):
    """r22_fig8.py panel (c): label-free triage of the price."""
    for f in F8_FAMS:
        a = M[(M.family == f) & (~M.cens)]
        b = M[(M.family == f) & (M.cens)]
        ax.scatter(a.ks_scores, a.y, s=30, color=F8_COL[f], edgecolor="k",
                   linewidth=0.35, label=F8_LBL[f])
        if len(b):
            ax.scatter(b.ks_scores, b.y, s=34, color=F8_COL[f], marker="^",
                       facecolor="none", linewidth=0.9)
    ax.axhline(info["cap"], color="k", ls=":", lw=1)
    ax.text(0.015, 0.965, "never reached inside the grid",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.text(0.015, 0.04,
            "$k=0$ acceptance zero in %d of %d units;\n"
            "%d of them never reach validity"
            % (info["zn"], info["n"], info["z"]),
            transform=ax.transAxes, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(18.0, info["cap"] * 1.8)
    ax.set_yticks([25, 100, 500, 2000, info["cap"]])
    ax.set_yticklabels(["25", "100", "500", "2000", "none"], fontsize=8)
    ax.set_xlabel("label-free score shift\n"
                  "(KS distance, source calibration vs deployment)")
    ax.set_ylabel("labels $k$ needed to reach validity")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)


# ===================================================================== fig 9
def f9a(ax, csv, alphas):
    """r21_downstream_repair.make_figure, panel a: dist_acc vs dist_all."""
    pt = csv[(csv["scope"] == "both")]
    shade = {alphas[i]: (0.30 + 0.30 * i) for i in range(len(alphas))}
    for ds in R21.SHORT:
        for a in alphas:
            sel = pt[(pt["dataset"] == ds) & (np.isclose(pt["alpha"], a))]
            if sel.empty:
                continue
            ax.scatter(np.maximum(sel["dist_all"], R21.FLOOR),
                       np.maximum(sel["dist_acc"], R21.FLOOR), s=15,
                       marker=R21.MK[ds], color=R21.CLR[ds], alpha=shade[a],
                       linewidths=0)
    lim = [R21.FLOOR, 10.0]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("dist_all: |pred - expert| enrichment, all cells")
    ax.set_ylabel("dist_acc: |pred - expert| enrichment, accepted")
    h = [Line2D([], [], marker=R21.MK[ds], ls="", color=R21.CLR[ds], ms=4.5,
                label=R21.SHORT[ds]) for ds in R21.SHORT]
    h += [Line2D([], [], marker="o", ls="", color="0.35", ms=4.0,
                 alpha=shade[a], label="alpha=%.2f" % a) for a in alphas]
    ax.legend(handles=h, fontsize=6, ncol=2, loc="upper left",
              handletextpad=0.3, columnspacing=0.7)
    ax.grid(False)


def f9b(ax, res):
    """r21_downstream_repair.make_figure, panel b: the paired forest plot."""
    dss = res["dss"]
    rows = [("pooled", res["obs_pool"], res["boot"]["__pool__"]["delta"])]
    rows += [(R21.SHORT[ds], res["obs_ds"][ds], res["boot"][ds]["delta"])
             for ds in dss]
    ypos = np.arange(len(rows))[::-1]
    for y, (lab, r, bv) in zip(ypos, rows):
        dlo, dhi, _ = res["ci"](bv)
        if not np.isfinite(r["delta"]):
            ax.text(0.02, y, "not estimable (no accepted-side join)",
                    fontsize=6.5, color="0.45", va="center")
            continue
        col = "0.15" if lab == "pooled" else R21.CLR[[k for k in dss
                                                     if R21.SHORT[k] == lab][0]]
        ax.errorbar([r["delta"]], [y],
                    xerr=[[r["delta"] - dlo], [dhi - r["delta"]]],
                    fmt="o", ms=4.5, lw=1.2, capsize=2.5, color=col)
        ax.text(dhi + 0.02, y, "%d cells" % int(round(r["cells_acc"])),
                fontsize=5.8, color="0.4", va="center")
    ax.axvline(0.0, color="k", ls="--", lw=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("dist_all - dist_acc  (cell-weighted mean, accepted cells)\n"
                  "negative = accepted cells distort the spatial readout more")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(False)


# ==================================================================== fig 10
def f10a(axes, t):
    """r24_efficiency.make_figure, panel a: the three family scatter panels."""
    XMAX = {"cellsplit": 0.26, "within": 0.26, "transfer": 0.50}
    YMAX = {"cellsplit": 1.02, "within": 1.02, "transfer": 0.78}
    for i, (ax, fam) in enumerate(zip(axes, R24.FAMILIES)):
        xm, ym = XMAX[fam], YMAX[fam]
        for r in R24.R_GRID:
            if r <= xm:
                ax.axvline(r, color="0.85", lw=0.7, zorder=0)
        ax.plot([0, min(xm, ym)], [0, min(xm, ym)], color="0.5", ls=":",
                lw=0.9, zorder=1)
        for arm, mk, ls in (("oracle", "s", "--"), ("tgt_k", "o", "-"),
                            ("src", "D", "-")):
            g = t[(t.family == fam) & (t.arm == arm)]
            if not len(g):
                continue
            xs, ys = R24.curve_points([(r.realised_risk, r.accept_mean)
                                       for r in g.itertuples()])
            ax.plot(xs, ys, marker=mk, ms=3.6, lw=1.0, ls=ls,
                    color=R24.ARM_COL[arm], zorder=3)
        s = t[(t.family == fam) & (t.arm == "tgt_k")]
        su = s[~s.balanced]
        if len(su):
            ax.plot(su.realised_risk, su.accept_mean, "o", ms=3.4, mfc="white",
                    mec=R24.ARM_COL["tgt_k"], mew=0.8, ls="none", zorder=2)
        for arm in ("src", "oracle"):
            v, ok = R24.curve_at([(r.realised_risk, r.accept_mean) for r in
                                  t[(t.family == fam) & (t.arm == arm)].itertuples()],
                                 0.10)
            if ok:
                ax.plot([0.10], [v], marker="x", ms=5.5, mew=1.3,
                        color=R24.ARM_COL[arm], zorder=5)
        ax.set_xlim(0, xm)
        ax.set_ylim(0, ym)
        ax.set_xlabel("pooled realised risk")
        ax.set_title(R24.FAM_LABEL[fam], loc="right", fontsize=7.5, color="0.35")
        if fam == "cellsplit":
            ax.set_ylabel("mean acceptance")
        ax.grid(alpha=0.25)
    handles = [Line2D([], [], color=R24.ARM_COL["tgt_k"], marker="o", ms=3.6,
                      lw=1.0, label="tgt_k (target-calibrated)"),
               Line2D([], [], color=R24.ARM_COL["oracle"], marker="s", ms=3.6,
                      lw=1.0, ls="--", label="oracle (peeks at the labels)"),
               Line2D([], [], color=R24.ARM_COL["src"], marker="D", ms=3.6,
                      lw=1.0, label="src (k = 0, source-calibrated)"),
               Line2D([], [], color=R24.ARM_COL["tgt_k"], marker="o", ms=3.4,
                      mfc="white", ls="none", label="tgt_k, partial unit set"),
               Line2D([], [], color="0.5", ls=":", lw=0.9,
                      label="identity: acceptance = risk"),
               Line2D([], [], color="0.25", marker="x", ms=5.5, ls="none",
                      label="read at r = 0.10")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=6.2,
                   labelspacing=0.28, handlelength=1.7, borderpad=0.2)


def f10b(ax, t):
    """r24_efficiency.make_figure, panel b: yield per label and its marginal."""
    axr = ax.twinx()
    axr.spines["right"].set_visible(True)
    axr.grid(False)
    for fam in R24.FAMILIES:
        g = t[(t.family == fam) & (t.arm == "tgt_k")
              & np.isclose(t.alpha, 0.10)].sort_values("k")
        bal = g[g.balanced]
        unb = g[~g.balanced]
        ax.plot(bal.k, bal.yield_per_label, marker="o", ms=3.8, lw=1.4,
                color=R24.FAM_COL[fam], zorder=3)
        if len(unb):
            ax.plot(unb.k, unb.yield_per_label, "o", ms=3.2,
                    color=R24.FAM_COL[fam], mfc="white", mew=0.9, ls="none",
                    alpha=0.85, zorder=2)
        m = g.dropna(subset=["marginal_return_per100"])
        if len(m):
            axr.plot(m.k, m.marginal_return_per100, marker="^", ms=3.4, lw=1.0,
                     ls=(0, (1, 1.6)), color=R24.FAM_COL[fam], mfc="white",
                     alpha=0.75, zorder=2)
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("target labels reviewed, k")
    ax.set_ylabel("auto-annotated cells per label reviewed")
    axr.set_ylim(0, None)
    axr.set_ylabel("marginal return: extra cells per 100 labels", color="0.35")
    axr.tick_params(axis="y", colors="0.35")
    axr.axhline(0.0, color="0.85", lw=0.8)
    handles = [Line2D([], [], color=R24.FAM_COL[f], marker="o", ms=3.8, lw=1.4,
                      label="%s, balanced range" % R24.FAM_LABEL[f])
               for f in R24.FAMILIES]
    handles += [Line2D([], [], color="0.35", marker="o", ms=3.2, lw=0.0,
                       mfc="white", mew=0.9,
                       label="partial unit set (markers only)"),
                Line2D([], [], color="0.35", marker="^", ms=3.4, lw=1.0,
                       ls=(0, (1, 1.6)), mfc="white",
                       label="marginal return (right axis)")]
    ax.legend(handles=handles, loc="upper left", fontsize=6.2, ncol=2,
              labelspacing=0.3, columnspacing=1.0, handlelength=2.2,
              borderpad=0.2)
    ax.grid(alpha=0.25)


# =================================================================== fig S6
def fs6a(ax, units, st_prim):
    """r25_accept_deferred.plot_units, panel a: accepted vs deferred rates."""
    u = units[(units.scope == R25.PRIMARY_SCOPE) & units.estimable].copy()
    for fam in ("gmm", "pct_all", "pct_pospop"):
        s = u[u.family == fam]
        if not len(s):
            continue
        ax.scatter(s.rate_acc, s.rate_def, s=5, alpha=0.40, linewidths=0,
                   color=R25.FAMILY_CLR[fam], label=fam)
    hi = float(np.nanmax([u.rate_acc.max(), u.rate_def.max()])) if len(u) else 1.0
    hi = max(hi * 1.06, 1e-3)
    ax.plot([0, hi], [0, hi], color="0.35", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(-0.005 * hi, hi)
    ax.set_ylim(-0.005 * hi, hi)
    ax.set_box_aspect(1)
    ax.set_xlabel("violation rate among accepted cells")
    ax.set_ylabel("violation rate among deferred cells")
    ax.text(0.035, 0.96,
            "below the y = x line:\nthe manuscript's ordering\n"
            "%d of %d units (%.0f%%)"
            % (st_prim["n_pos"], st_prim["n"],
               100.0 * st_prim["n_pos"] / max(st_prim["n"], 1)),
            transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
            color="0.25",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
    ax.legend(loc="lower right", markerscale=2.2)
    ax.grid(alpha=0.25)


def fs6b(ax, units, st_prim):
    """r25_accept_deferred.plot_units, panel b: the per-convention ladder."""
    u = units[(units.scope == R25.PRIMARY_SCOPE) & units.estimable].copy()
    groups = list(u.groupby(["positivity", "family", "pct", "reading",
                             "ruleset"], dropna=False))
    summ = []
    for key, sub in groups:
        summ.append(dict(positivity=key[0], family=key[1], pct=key[2],
                         reading=key[3], ruleset=key[4], n=len(sub),
                         med=float(sub["diff"].median()),
                         q1=float(sub["diff"].quantile(0.25)),
                         q3=float(sub["diff"].quantile(0.75))))
    cdf = pd.DataFrame(summ).sort_values("med").reset_index(drop=True)
    rng = np.random.default_rng(0)
    for i, r in cdf.iterrows():
        sub = u[(u.positivity == r.positivity) & (u.reading == r.reading)
                & (u.ruleset == r.ruleset)]
        jit = rng.uniform(-0.20, 0.20, len(sub))
        ax.scatter(sub["diff"], i + jit, s=3.5, alpha=0.45, linewidths=0,
                   color=R25.FAMILY_CLR[r.family])
        ax.plot([r.med, r.med], [i - 0.36, i + 0.36], color="0.12", lw=1.2,
                zorder=5)
    ax.axvline(0.0, color="0.35", lw=0.8, ls="--")
    labels = [R25.short_conv(r.positivity, r.reading, r.ruleset)
              for _, r in cdf.iterrows()]
    ax.set_yticks(np.arange(len(cdf)))
    ax.set_yticklabels(labels, fontsize=4.3)
    ax.set_ylim(-0.9, len(cdf) - 0.1)
    ax.set_xlabel("violation-rate difference  (deferred - accepted)")
    ax.set_ylabel("convention, ordered by median difference")
    ax.text(0.985, 0.985,
            "sign test (accepted < deferred)\n%d / %d units, one-sided p = %s\n"
            "median difference %+.4f"
            % (st_prim["n_pos"], st_prim["n"], R25._pp(st_prim["p_greater"]),
               st_prim["median"]),
            transform=ax.transAxes, va="top", ha="right", fontsize=6,
            color="0.2",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
    ax.grid(alpha=0.25)


# =================================================================== fig S7
FS7_ALPHA, FS7_DELTA, FS7_MA = 0.10, 0.10, 100
FS7_ARM_STYLE = [
    (("bonferroni", "ascending"), "--", "s", "bonferroni"),
    (("fixed_sequence", "ascending"), "-", "o", "step-down, ascending"),
    (("fixed_sequence", "descending"), ":", "^", "step-down, descending"),
]


def fs7a(ax, summ):
    """r27_fixedseq.figure, panel a: mean acceptance against the grid size."""
    for fam in R27.FAMILIES:
        for (sel, so), ls, mk, _lab in FS7_ARM_STYLE:
            xs, ys = [], []
            for gs in R27.GRID_SIZES:
                cfg = R27.cfg_name(sel, so, FS7_MA, gs)
                r = R27.one(summ, family=fam, alpha=FS7_ALPHA, delta=FS7_DELTA,
                            min_accept=FS7_MA, grid_size=gs, config=cfg)
                xs.append(gs)
                ys.append(float(r["mean_accept"]))
            ax.plot(xs, ys, ls, marker=mk, ms=3.2, lw=1.2, color=R27.FAM_CLR[fam])
    ax.set_xscale("log")
    ax.set_xticks(list(R27.GRID_SIZES))
    ax.set_xticklabels([str(g) for g in R27.GRID_SIZES])
    ax.set_xlim(15.0, 560.0)
    ax.set_ylim(-0.04, 1.02)
    ax.set_xlabel("candidate grid size $m$")
    ax.set_ylabel("mean acceptance rate")
    h_fam = [Line2D([], [], color=R27.FAM_CLR[f], lw=1.4, label=R27.FAM_LABEL[f])
             for f in R27.FAMILIES]
    h_arm = [Line2D([], [], color="0.25", ls=ls, marker=mk, ms=3.2, lw=1.2,
                    label=lab) for (_, ls, mk, lab) in FS7_ARM_STYLE]
    leg1 = ax.legend(handles=h_fam, fontsize=5.8, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=h_arm, fontsize=5.8, loc="lower right")
    ax.grid(alpha=0.25)


def fs7b(ax, summ):
    """r27_fixedseq.figure, panel b: coverage of the budget against 1 - delta."""
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
        for fam in R27.FAMILIES:
            for (sel, so), dx in offs:
                cfg = R27.cfg_name(sel, so, FS7_MA, 120)
                r = R27.one(summ, family=fam, alpha=FS7_ALPHA, delta=delta,
                            min_accept=FS7_MA, grid_size=120, config=cfg)
                y = float(r["coverage"])
                lo = float(r["coverage_lo"])
                hi = float(r["coverage_hi"])
                ymin = min(ymin, y, lo)
                ymax = max(ymax, y, hi)
                open_marker = (sel == "naive")
                ax.errorbar([x0 + dx], [y], yerr=[[max(y - lo, 0.0)],
                                                  [max(hi - y, 0.0)]],
                            fmt=marks[(sel, so)], ms=3.6, lw=0.9, capsize=1.6,
                            color=R27.FAM_CLR[fam],
                            markerfacecolor=("none" if open_marker
                                             else R27.FAM_CLR[fam]),
                            markeredgecolor=R27.FAM_CLR[fam], markeredgewidth=0.9)
    for delta, x0 in xpos.items():
        ax.plot([x0 - 0.30, x0 + 0.30], [1.0 - delta] * 2, "k--", lw=1.0)
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_xticklabels(["0.05", "0.10", "0.20"])
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(max(-0.02, ymin - 0.05), min(1.06, max(ymax + 0.05, 0.92)))
    ax.set_xlabel("failure probability delta")
    ax.set_ylabel("coverage of the budget alpha = 0.10")
    h_fam2 = [Line2D([], [], color=R27.FAM_CLR[f], lw=1.4, label=R27.FAM_LABEL[f])
              for f in R27.FAMILIES]
    h_arm2 = [Line2D([], [], color="0.25", ls="", marker=marks[(s, o)], ms=3.6,
                     label=R27.ARM_LABEL[(s, o)]) for (s, o), _ in offs]
    h_arm2.append(Line2D([], [], color="k", ls="--", lw=1.0, label="1 - delta"))
    leg1 = ax.legend(handles=h_fam2, fontsize=5.8, loc="lower left")
    ax.add_artist(leg1)
    ax.legend(handles=h_arm2, fontsize=5.4, loc="upper right", ncol=2)
    ax.grid(alpha=0.25)
