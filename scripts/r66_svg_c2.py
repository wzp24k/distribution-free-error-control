# -*- coding: utf-8 -*-
"""r66_svg_c2 -- figures 7, 8, 9, 10, S6 and S7 as per-panel editable SVG.

For every key in KEYS this driver

  * reproduces the combined multi-panel figure (same panels, same drawing code,
    same numbers as the original figure script) and writes
    figures/svg/<key>_full.svg plus <key>_full.png (dpi 300, tight bbox,
    pad_inches 0.02), and
  * draws every panel again, on its own, inside a fresh single-panel figure
    from r66_svgstyle.new(letter, title), and writes
    figures/svg/<key>_<letter>.svg.

The drawing code lives in r66_svg_c2_panels.py and is the original panels'
code; the style (fonts, grid, spines) and the title -- one size, TITLE_SIZE,
on every panel -- come from r66_svgstyle.py, which keeps every label as real
editable text (svg.fonttype = "none").  Nothing is converted to outlines and no
combined figure is ever cropped into panels: each panel is drawn fresh.

The originals are left untouched and still write their own figures/*.png|pdf.

Producers of the six keys (grepped):

  fig7_review_budget      scripts/r16_review_budget.py::figure
  fig8_label_ladder       scripts/r22_fig8.py (it runs after r20_label_ladder.py
                          and overwrites what r20_label_ladder.py::make_figure
                          wrote; the on-disk PNG is r22's rendering -- verified
                          pixel-for-pixel against a fresh run of r22_fig8.py)
  fig9_downstream         scripts/r21_downstream_repair.py::make_figure
                          (r21b_matched_downstream.py writes no figure)
  fig10_yield             scripts/r24_efficiency.py::make_figure
  figS6_accept_deferred   scripts/r25_accept_deferred.py::plot_units
  figS7_fixedseq          scripts/r27_fixedseq.py::figure

Every number plotted is read from the same tables the original figures read
(experiments/*.csv, work/*.csv) or recomputed by the original module, and a
summary of the plotted quantities is written to work/r66_c2_numbers.json for
work/r66_c2_verify.py.

Usage
  python scripts/r66_svg_c2.py                # everything (fig9 re-runs r21)
  python scripts/r66_svg_c2.py --only fig7_review_budget
"""

import argparse
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import r66_svgstyle as svgstyle                                  # noqa: E402
import r66_svg_c2_panels as P                                    # noqa: E402

OUT = os.path.join(ROOT, "figures", "svg")
FIGDIR = os.path.join(ROOT, "figures")
EXP = os.path.join(ROOT, "experiments")
WORK = os.path.join(ROOT, "work")
NUMBERS = os.path.join(WORK, "r66_c2_numbers.json")

# panel letters in current reading order, with the short descriptors the current
# figures use (the original titles with their "(x)" / letter prefix removed;
# fig10's panel (a) has no descriptor of its own, so it is named from the
# manuscript's own caption for that panel).
DESC = {
    "fig7_review_budget": ["labels = realised risk",
                           "at the same realised risk"],
    "fig8_label_ladder": ["acceptance bought with labels\n"
                          "lines: target-calibrated;  dashed: oracle;  "
                          "squares: $k=0$",
                          "what labels actually buy\n"
                          "non-vacuous coverage is 1.000 wherever the\n"
                          "certificate answers",
                          None],                       # filled in at run time
    "fig9_downstream": ["below the line = distortion smaller when accepted",
                        "95% CI, bootstrap over images"],
    "fig10_yield": ["acceptance vs risk",
                    "certificate yield per target label, alpha = 0.10\n"
                    "solid line = balanced range (all units); open circles = "
                    "partial unit set (not comparable)"],
    "figS6_accept_deferred": ["68 conventions x 3 alpha, ALL_strict",
                              "median per convention"],
    "figS7_fixedseq": ["alpha = 0.10, delta = 0.10, min_accept = 100",
                       "dots above the dashed line hold the budget"],
}

KEYS = ["fig7_review_budget", "fig8_label_ladder", "fig9_downstream",
        "fig10_yield", "figS6_accept_deferred", "figS7_fixedseq"]

# Per-panel figure size and axes rectangle.  The house default
# (svgstyle.PANEL_W x PANEL_H, rect [0.20, 0.20, 0.76, 0.70]) is used wherever
# it fits; the sizes below follow the shape of each panel (the originals draw
# these panels at 2.4-4.3 in, and the legends and in-axes annotations need that
# room).
PANEL_SIZE = {
    ("fig7_review_budget", "a"): (3.2, 2.5, None),
    ("fig7_review_budget", "b"): (3.2, 2.5, None),
    ("fig8_label_ladder", "a"): (3.3, 2.8, None),
    ("fig8_label_ladder", "b"): (3.3, 2.8, None),
    ("fig8_label_ladder", "c"): (5.2, 3.4, None),
    # panel a carries a 46-character y label; the single-panel figure needs the
    # axes tall enough for the rotated label to stay inside the tight bbox
    # (the original combined figure is square-ish here as well).
    ("fig9_downstream", "a"): (3.0, 3.2, None),
    ("fig9_downstream", "b"): (3.4, 2.5, [0.21, 0.20, 0.75, 0.64]),
    ("fig10_yield", "a"): (9.8, 2.8, None),      # triptych, one fresh figure
    ("fig10_yield", "b"): (7.4, 2.6, [0.115, 0.20, 0.735, 0.66]),
    ("figS6_accept_deferred", "a"): (3.0, 3.0, [0.20, 0.16, 0.70, 0.70]),
    ("figS6_accept_deferred", "b"): (4.4, 5.2, [0.26, 0.06, 0.72, 0.90]),
    ("figS7_fixedseq", "a"): (3.2, 2.5, None),
    ("figS7_fixedseq", "b"): (3.2, 2.5, None),
}


def full_path(key):
    return os.path.join(OUT, key + "_full.svg")


def panel_path(key, letter):
    return os.path.join(OUT, "%s_%s.svg" % (key, letter))


def save_panel(key, letter, desc, draw):
    """One fresh single-panel figure in the house style."""
    w, h, rect = PANEL_SIZE.get((key, letter),
                                (svgstyle.PANEL_W, svgstyle.PANEL_H, None))
    fig, ax = svgstyle.new(letter, desc, w=w, h=h, rect=rect)
    draw(ax)
    svgstyle.save(fig, panel_path(key, letter))


# ------------------------------------------------------------------ fig 7
def load_fig7(jobs=1):
    m = P.R16.load_r03()
    rng = np.random.default_rng(20260926)
    curve = P.R16.review_budget_curve(m, rng)
    files = sorted(f for f in os.listdir(P.R16.PRED) if f.endswith(".npz"))
    jobs_ = [(f, P.R16.RISKS, 0.10) for f in files]
    mr = P.R16.run_parallel(P.R16._matched_job, jobs_, jobs)
    ms = P.R16.matched_summary(mr, rng)
    ratio = P.R16.matched_ratio(ms)
    return curve, ms, ratio


def build_fig7(data):
    key = "fig7_review_budget"
    curve, ms, ratio = data
    svgstyle.use()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    P.f7a(axes[0], curve)
    svgstyle.title(axes[0], "a", DESC[key][0])
    P.f7b(axes[1], ms, ratio)
    svgstyle.title(axes[1], "b", DESC[key][1])
    fig.tight_layout()
    svgstyle.save_full(fig, full_path(key))
    save_panel(key, "a", DESC[key][0], lambda ax: P.f7a(ax, curve))
    save_panel(key, "b", DESC[key][1], lambda ax: P.f7b(ax, ms, ratio))

    num = {}
    crc = curve[curve.method == "crc"]
    for fam in P.R16.FAMILIES:
        for a in (0.10, 0.20):
            s = crc[(crc.family == fam) & np.isclose(crc.alpha, a)]
            if len(s):
                num["%s_a%.2f_review_frac" % (fam, a)] = float(s.review_frac.iloc[0])
                num["%s_a%.2f_risk_nonvac" % (fam, a)] = float(s.risk_nonvac.iloc[0])
        v = ratio[(ratio.family == fam) & np.isclose(ratio.risk, 0.10)]
        if len(v):
            num["%s_ratio_crc_oracle_r010" % fam] = float(v.ratio_crc_oracle.iloc[0])
    return num


# ------------------------------------------------------------------ fig 8
def load_fig8():
    u = pd.read_csv(os.path.join(EXP, "r20_units.csv"))
    u = u[(u.min_accept == 25) & (u.grid == "abs")
          & np.isclose(u.alpha, P.F8_AL)].copy()
    u["nonvac"] = (u["empty"] == 0).astype(int)
    r = pd.read_csv(os.path.join(EXP, "r14_targetcal.csv"),
                    usecols=["unit", "alpha", "k", "arm", "accept_rate",
                             "empty", "nonvac_covered"])
    r = r[r.unit.str.startswith("transfer:")].copy()
    r = r[np.isclose(r.alpha, P.F8_AL)]
    r["nonvac"] = (r["empty"] == 0).astype(int)
    r["covered_nonvac"] = r["nonvac_covered"]
    T, O, S = P.f8_tables(u, r)
    M, info = P.f8_triage(pd.read_csv(os.path.join(WORK, "triage_reach.csv")))
    return T, O, S, M, info


def build_fig8(data):
    key = "fig8_label_ladder"
    T, O, S, M, info = data
    desc_c = ("attainability, not price, is predictable\n"
              "Spearman $\\rho$ = %+.2f (p = %.1g, n = %d) against $k$;\n"
              "open symbols: never reached (%d of %d)"
              % (info["rho"], info["pv"], info["n"], info["n_cens"],
                 info["n"]))
    DESC[key][2] = desc_c
    svgstyle.use()
    fig = plt.figure(figsize=(13.6, 4.3))
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    P.f8a(axes[0], T, O, S)
    svgstyle.title(axes[0], "a", DESC[key][0])
    P.f8b(axes[1], T)
    svgstyle.title(axes[1], "b", DESC[key][1])
    P.f8c(axes[2], M, info)
    svgstyle.title(axes[2], "c", DESC[key][2])
    fig.tight_layout()
    svgstyle.save_full(fig, full_path(key))
    save_panel(key, "a", DESC[key][0], lambda ax: P.f8a(ax, T, O, S))
    save_panel(key, "b", DESC[key][1], lambda ax: P.f8b(ax, T))
    save_panel(key, "c", DESC[key][2], lambda ax: P.f8c(ax, M, info))

    num = dict(rho=info["rho"], p=info["pv"], n=info["n"],
               n_censored=info["n_cens"], n_below_floor=info["z"],
               n_zero_accept=info["zn"], top_cap=info["cap"])
    for fam in P.F8_FAMS:
        g = T[T.family == fam]
        num["%s_k_max_shown" % fam] = int(g.k.max()) if len(g) else -1
    return num


# ------------------------------------------------------------------ fig 9
def load_fig9():
    res = P.R21.run(list(P.R21.ALPHAS), list(P.R21.CLFS), P.R21.MIN_CELLS,
                    P.R21.BOOT, P.R21.SEED)
    return res, P.R21.csv_rows(res)


def build_fig9(data):
    key = "fig9_downstream"
    res, csv = data
    svgstyle.use()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})
    P.f9a(axes[0], csv, res["alphas"])
    svgstyle.title(axes[0], "a", DESC[key][0])
    P.f9b(axes[1], res)
    svgstyle.title(axes[1], "b", DESC[key][1])
    fig.tight_layout()
    svgstyle.save_full(fig, full_path(key))
    save_panel(key, "a", DESC[key][0],
               lambda ax: P.f9a(ax, csv, res["alphas"]))
    save_panel(key, "b", DESC[key][1], lambda ax: P.f9b(ax, res))

    lo, hi, nb = res["ci"](res["boot"]["__pool__"]["delta"])
    num = dict(pooled_delta=float(res["obs_pool"]["delta"]),
               pooled_ci_lo=float(lo), pooled_ci_hi=float(hi), boot_n=nb)
    for ds in res["dss"]:
        dlo, dhi, _ = res["ci"](res["boot"][ds]["delta"])
        num["%s_delta" % ds] = float(res["obs_ds"][ds]["delta"])
        num["%s_ci_lo" % ds] = float(dlo)
        num["%s_ci_hi" % ds] = float(dhi)
        num["%s_cells_acc" % ds] = int(round(res["obs_ds"][ds]["cells_acc"]))
    num["pooled_cells_acc"] = int(round(res["obs_pool"]["cells_acc"]))
    return num


# ----------------------------------------------------------------- fig 10
def load_fig10():
    return P.R24.add_marginal(P.R24.arm_table(P.R24.load_r20()))


def build_fig10(data):
    key = "fig10_yield"
    t = data
    svgstyle.use()
    fig = plt.figure(figsize=(9.8, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.70,
                          wspace=0.34)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axb = fig.add_subplot(gs[1, :])
    P.f10a(axes, t)
    svgstyle.title(axes[0], "a", DESC[key][0])
    P.f10b(axb, t)
    svgstyle.title(axb, "b", DESC[key][1])
    svgstyle.save_full(fig, full_path(key))

    # panel a is a triptych: one fresh single-panel figure with three axes
    fig, ax0 = svgstyle.new("a", DESC[key][0], w=9.8, h=2.8)
    ax0.set_position([0.05, 0.20, 0.245, 0.66])
    axes = [ax0, fig.add_axes([0.385, 0.20, 0.245, 0.66]),
            fig.add_axes([0.72, 0.20, 0.245, 0.66])]
    P.f10a(axes, t)
    for ax in axes:
        ax.grid(alpha=0.25)
    svgstyle.save(fig, panel_path(key, "a"))
    save_panel(key, "b", DESC[key][1], lambda ax: P.f10b(ax, t))

    num = {}
    for fam in P.R24.FAMILIES:
        g = t[(t.family == fam) & (t.arm == "tgt_k")
              & np.isclose(t.alpha, 0.10)].sort_values("k")
        b = g[g.balanced]
        num["%s_k" % fam] = [int(v) for v in b.k]
        num["%s_yield_per_label" % fam] = [float(v) for v in b.yield_per_label]
        num["%s_marginal_per100" % fam] = [
            None if not np.isfinite(v) else float(v)
            for v in b.marginal_return_per100]
        for arm in ("src", "oracle"):
            v, ok = P.R24.curve_at(
                [(r.realised_risk, r.accept_mean)
                 for r in t[(t.family == fam) & (t.arm == arm)].itertuples()],
                0.10)
            num["%s_%s_accept_at_r010" % (fam, arm)] = float(v) if ok else None
    return num


# ----------------------------------------------------------------- fig S6
def load_figs6():
    units = pd.read_csv(os.path.join(WORK, "_r25_units.csv"))
    prim = units[(units.scope == P.R25.PRIMARY_SCOPE) & units.estimable]
    return units, P.R25.sign_test(prim["diff"])


def build_figs6(data):
    key = "figS6_accept_deferred"
    units, st = data
    svgstyle.use()
    fig = plt.figure(figsize=(10.0, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    P.fs6a(ax1, units, st)
    svgstyle.title(ax1, "a", DESC[key][0])
    P.fs6b(ax2, units, st)
    svgstyle.title(ax2, "b", DESC[key][1])
    svgstyle.save_full(fig, full_path(key))
    save_panel(key, "a", DESC[key][0], lambda ax: P.fs6a(ax, units, st))
    save_panel(key, "b", DESC[key][1], lambda ax: P.fs6b(ax, units, st))

    u = units[(units.scope == P.R25.PRIMARY_SCOPE) & units.estimable]
    n_conv = int(len(list(u.groupby(["positivity", "family", "pct", "reading",
                                     "ruleset"], dropna=False))))
    return dict(n_pos=st["n_pos"], n_neg=st["n_neg"], n_tie=st["n_tie"],
                n=st["n"], median=st["median"],
                p_greater=st["p_greater"], n_conventions=n_conv,
                n_estimable_units=int(len(u)))


# ----------------------------------------------------------------- fig S7
def load_figs7():
    return pd.read_csv(os.path.join(EXP, "r27_fixedseq_summary.csv"))


def build_figs7(data):
    key = "figS7_fixedseq"
    summ = data
    svgstyle.use()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    P.fs7a(axes[0], summ)
    svgstyle.title(axes[0], "a", DESC[key][0])
    P.fs7b(axes[1], summ)
    svgstyle.title(axes[1], "b", DESC[key][1])
    fig.tight_layout()
    svgstyle.save_full(fig, full_path(key))
    save_panel(key, "a", DESC[key][0], lambda ax: P.fs7a(ax, summ))
    save_panel(key, "b", DESC[key][1], lambda ax: P.fs7b(ax, summ))

    num = {}
    cfgs = ["bonferroni_ma100_gs120", "fixed_sequence_ma100_gs120",
            "fixed_sequence_desc_ma100_gs120"]
    for fam in P.R27.FAMILIES:
        for cfg in cfgs:
            for gs in P.R27.GRID_SIZES:
                r = P.R27.one(summ, family=fam, alpha=0.10, delta=0.10,
                              min_accept=100, grid_size=gs,
                              config=cfg.replace("gs120", "gs%d" % gs))
                num["%s_%s_gs%d_mean_accept" % (fam, cfg, gs)] = float(
                    r["mean_accept"])
        for d in (0.05, 0.10, 0.20):
            r = P.R27.one(summ, family=fam, alpha=0.10, delta=d, min_accept=100,
                          grid_size=120,
                          config="fixed_sequence_ma100_gs120")
            num["%s_fs_gs120_d%.2f_coverage" % (fam, d)] = float(r["coverage"])
            num["%s_fs_gs120_d%.2f_coverage_lo" % (fam, d)] = float(
                r["coverage_lo"])
            num["%s_fs_gs120_d%.2f_coverage_hi" % (fam, d)] = float(
                r["coverage_hi"])
    return num


LOADERS = {"fig7_review_budget": load_fig7, "fig8_label_ladder": load_fig8,
           "fig9_downstream": load_fig9, "fig10_yield": load_fig10,
           "figS6_accept_deferred": load_figs6, "figS7_fixedseq": load_figs7}
BUILDERS = {"fig7_review_budget": build_fig7, "fig8_label_ladder": build_fig8,
            "fig9_downstream": build_fig9, "fig10_yield": build_fig10,
            "figS6_accept_deferred": build_figs6, "figS7_fixedseq": build_figs7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="build one key only (substring match)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="workers for the r16 matched-risk jobs")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    numbers = {}
    for key in KEYS:
        if args.only and args.only not in key:
            continue
        t0 = time.time()
        print("[%s] %s: loading" % (time.strftime("%H:%M:%S"), key), flush=True)
        data = LOADERS[key](args.jobs) if key == "fig7_review_budget" \
            else LOADERS[key]()
        numbers[key] = BUILDERS[key](data)
        print("[%s] %s: done in %.1fs" % (time.strftime("%H:%M:%S"), key,
                                          time.time() - t0), flush=True)
    old = {}
    if os.path.exists(NUMBERS):
        try:
            old = json.load(io.open(NUMBERS, encoding="utf-8"))
        except ValueError:
            old = {}
    # Record the exact house title of every panel that was drawn in this run,
    # so work/r66_c2_verify.py can assert that each panel SVG carries that
    # title at the one house size (svgstyle.TITLE_SIZE).  Entries for keys not
    # built in this run (--only) are kept from the previous file.
    meta = dict(old.get("_meta", {}))
    for key in KEYS:
        if args.only and args.only not in key:
            continue
        panels = []
        for i, desc in enumerate(DESC[key]):
            letter = "abcdefgh"[i]
            size = PANEL_SIZE.get((key, letter),
                                  (svgstyle.PANEL_W, svgstyle.PANEL_H, None))
            panels.append({"letter": letter,
                           "title": ("%s  %s" % (letter, desc)) if desc
                                    else letter,
                           "w": size[0], "h": size[1]})
        meta[key] = panels
    meta["_title_size"] = svgstyle.TITLE_SIZE
    old["_meta"] = meta
    old.update(numbers)
    io.open(NUMBERS, "w", encoding="utf-8", newline="\n").write(
        json.dumps(old, indent=1, sort_keys=True) + "\n")
    print("wrote %s and figures/svg/" % NUMBERS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
