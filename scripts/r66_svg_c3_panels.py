# -*- coding: utf-8 -*-
"""r66_svg_c3_panels -- the drawing code of every regenerated panel.

Each ``sNN<letter>(ax, ctx)`` draws one panel of one figure into the axes it is
handed, with the same drawing calls and the same numbers as the figure function
it was factored out of:

    figS3   scripts/r15_marker_sensitivity.py   figure()
    figS4   scripts/r18_marker_sources.py       figure()
    figS5   scripts/r19_weight_shift_diag.py    make_figure()
    figS8   scripts/r35_figures.py              figS8()
    figS9   scripts/r35_figures.py              figS9()
    figS10  scripts/r35_figures.py              figS10()

The matching ``ctx_<key>()`` builder reproduces the data those figure functions
were handed, loaded from the CSVs the original scripts write, so the original
scripts stay untouched and runnable.

The panel title is deliberately *not* drawn here: scripts/r66_svg_c3.py applies
the shared house style (scripts/r66_svgstyle.py) after calling the panel, so
every panel of every figure carries its title at one font size.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as ss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedFormatter, FixedLocator, NullFormatter,
                               NullLocator)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "experiments")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from errctl import paths

import r35_figures as R35

# r35's own data helpers and constants, reused verbatim so the numbers the
# panels draw are the ones the original figure scripts computed.
load = R35.load
one = R35.one
near = R35.near
rec = R35.rec


def _csv(name):
    return pd.read_csv(os.path.join(EXP, name))


# ============================================================ figS3 (r15)
S3_ORDER = list(paths.MANUAL_DATASETS)
S3_SHORT = {"Breast_Jun0524": "Breast", "CRC_Feb2623": "CRC",
            "GVHD_Feb1323": "GVHD", "Melanoma_Oct2522": "Melanoma"}
S3_COL = {"Breast_Jun0524": "#1f5fa9", "CRC_Feb2623": "#2a9d8f",
          "GVHD_Feb1323": "#d1495b", "Melanoma_Oct2522": "#e6a817"}
S3_FLOOR = 2.5e-4


def ctx_s3():
    return {"types": _csv("r15_types.csv"), "thr": _csv("r15_threshold.csv"),
            "ORDER": S3_ORDER, "SHORT": S3_SHORT, "COL": S3_COL,
            "FLOOR": S3_FLOOR,
            "titles": {"a": "class detail, bar = cohort pool",
                       "b": "threshold sensitivity, dash-dot = mixture"}}


def s3a(ax, ctx):
    types = ctx["types"]
    ORDER, SHORT, COL, FLOOR = (ctx["ORDER"], ctx["SHORT"], ctx["COL"],
                                ctx["FLOOR"])
    sel = types[(types.scope == "type") & (types.variant == "repaired")
                & (types.reading == "or") & (types.source == "expert")
                & (types.n >= 50)]
    x, ticks, ticklab = 0, [], []
    for ds in ORDER:
        sub = sel[sel.dataset == ds].sort_values("violation_rate", ascending=False)
        if sub.empty:
            continue
        start = x
        for _, r in sub.iterrows():
            y = max(float(r.violation_rate), FLOOR)
            if r.n_violation > 0 and np.isfinite(r.ci_lo):
                lo = max(float(r.ci_lo), FLOOR)
                ax.errorbar([x], [y], yerr=[[y - lo], [max(float(r.ci_hi), FLOOR) - y]],
                            fmt="o", ms=2.4, lw=0.8, capsize=1.1, color=COL[ds],
                            zorder=3)
            else:
                ax.plot([x], [FLOOR], marker="o", ms=2.4, mfc="white", mec=COL[ds],
                        mew=0.7, ls="none", zorder=3)
            ticks.append(x)
            ticklab.append(r.cls)
            x += 1
        p = types[(types.scope == "summary") & (types.cls == "ALL_evaluable")
                  & (types.dataset == ds) & (types.variant == "repaired")
                  & (types.reading == "or") & (types.source == "expert")]
        if not p.empty:
            ax.plot([start - 0.4, x - 0.6], [float(p.violation_rate.iloc[0])] * 2,
                    "-", color=COL[ds], lw=1.5, alpha=0.85, zorder=2)
        ax.text((start + x - 1) / 2.0, 1.15, SHORT[ds], ha="center", fontsize=6.2,
                color=COL[ds])
        x += 2
    ax.axhline(0.05, color="k", ls=":", lw=1.0)
    ax.text(x - 0.8, 0.056, "5%", fontsize=5.8, ha="right", color="0.25")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.6, 1.5)
    ax.set_xlim(-1.2, x - 0.2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklab, rotation=90, fontsize=4.4)
    ax.set_ylabel("violation rate of the expert label")


def s3b(ax, ctx):
    thr = ctx["thr"]
    ORDER, SHORT, COL = ctx["ORDER"], ctx["SHORT"], ctx["COL"]
    t = thr[(thr.variant.isin(["pct_all", "pct_pospop"]))
            & (thr.ruleset == "repaired") & (thr.reading == "or")
            & (thr.summary == "ALL_evaluable")]
    for ds in ORDER:
        s = t[(t.dataset == ds) & (t.variant == "pct_all")].sort_values("positivity_pct")
        if not s.empty:
            ax.plot(s.positivity_pct, s.violation_rate, "o-", ms=2.5, lw=1.1,
                    color=COL[ds], label=SHORT[ds])
        s2 = t[(t.dataset == ds) & (t.variant == "pct_pospop")].sort_values("positivity_pct")
        if not s2.empty:
            ax.plot(s2.positivity_pct, s2.violation_rate, "s--", ms=2.1, lw=0.8,
                    color=COL[ds], alpha=0.55)
        g = thr[(thr.dataset == ds) & (thr.variant == "gmm")
                & (thr.ruleset == "repaired") & (thr.reading == "or")
                & (thr.summary == "ALL_evaluable")]
        if not g.empty:
            ax.axhline(float(g.violation_rate.iloc[0]), color=COL[ds], lw=0.7,
                       ls="-.", alpha=0.7)
    ax.axhline(0.05, color="k", ls=":", lw=1.0)
    ax.set_xlabel("positivity percentile")
    ax.set_ylabel("violation rate, all evaluable classes")
    ax.set_ylim(0, None)
    ax.legend(ncol=2, fontsize=5.8, title="solid: all cells; dashed: mixture-positive",
              title_fontsize=5.6)


# ============================================================ figS4 (r18)
S4_PRIMARY_POSITIVITY = "gmm"
S4_MIN_N = 50
S4_GATE = 0.05
S4_SHORT = {"Breast_Jun0524": "Breast", "CRC_Feb2623": "CRC",
            "GVHD_Feb1323": "GVHD", "Melanoma_Oct2522": "Melanoma"}
S4_COLORS = {"expert": "#1f4e79", "hq_ref": "#2e9e8f", "hq_all": "#d1743a",
             "hq_nonexpert": "#9b8bc4"}


def ctx_s4():
    types = _csv("r18_hql.csv")
    conv = _csv("r18_conventions.csv")
    agree = _csv("r18_agreement.csv").to_dict("records")
    min_n = S4_MIN_N
    prim = types[types.positivity == S4_PRIMARY_POSITIVITY]
    cohorts = sorted(prim.dataset.unique())
    srcs = ["expert", "hq_ref", "hq_all", "hq_nonexpert"]
    srcs = [q for q in srcs if q in set(prim.source)]
    ag = pd.DataFrame(agree)
    kappa = ", ".join("%s %.2f" % (S4_SHORT.get(r["dataset"], ""),
                                   r["coarse_kappa"])
                      for _, r in ag.iterrows())
    return {"types": types, "conv": conv, "agree": agree, "min_n": min_n,
            "prim": prim, "cohorts": cohorts, "srcs": srcs, "ag": ag,
            "short": S4_SHORT, "colors": S4_COLORS, "GATE": S4_GATE,
            "titles": {
                "a": "cell-pooled rate, by label source\n"
                     "bars: ';' as OR   circles: ';' as AND",
                "b": "populations with n>=%d (OR, repaired)" % min_n,
                "c": "convention sensitivity (positivity x reading x rules)",
                "d": "agreement of the depot's two label sources\n"
                     "(same cells, coarse kappa %s)" % kappa}}


def s4a(ax, ctx):
    cohorts, srcs, colors = ctx["cohorts"], ctx["srcs"], ctx["colors"]
    prim, GATE, short = ctx["prim"], ctx["GATE"], ctx["short"]

    def summ(ds, source, reading, variant):
        s = prim[(prim.dataset == ds) & (prim.source == source)
                 & (prim.scope == "summary") & (prim.cls == "ALL_evaluable")
                 & (prim.reading == reading) & (prim.variant == variant)]
        return np.nan if s.empty else float(s["rate"].iloc[0])

    xs = np.arange(len(cohorts))
    w = 0.8 / max(1, len(srcs))
    for k, source in enumerate(srcs):
        vals = [max(summ(ds, source, "or", "repaired"), 1e-6) for ds in cohorts]
        ax.bar(xs + k * w - 0.4 + w / 2, vals, width=w * 0.92,
               color=colors.get(source, "gray"), alpha=0.55, label=source)
        for i, ds in enumerate(cohorts):
            v = summ(ds, source, "and", "repaired")
            if np.isfinite(v):
                ax.plot([xs[i] + k * w - 0.4 + w / 2], [max(v, 1e-6)],
                        marker="o", ms=3.2, mfc="none",
                        mec=colors.get(source, "gray"), mew=0.9)
    ax.axhline(GATE, color="crimson", lw=1.0, ls="--")
    ax.text(len(cohorts) - 0.55, GATE * 1.35, "5% gate", color="crimson",
            fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([short.get(c, c) for c in cohorts])
    ax.set_ylabel("pooled violation rate (log)")
    ax.legend(loc="upper left", frameon=False)


def s4b(ax, ctx):
    cohorts, srcs, colors = ctx["cohorts"], ctx["srcs"], ctx["colors"]
    prim, min_n, GATE, short = (ctx["prim"], ctx["min_n"], ctx["GATE"],
                                ctx["short"])
    pos = 0.0
    ticks, labels = [], []
    for ds in cohorts:
        for source in srcs:
            sel = prim[(prim.dataset == ds) & (prim.source == source)
                       & (prim.scope == "type") & (prim.reading == "or")
                       & (prim.variant == "repaired") & (prim.n >= min_n)
                       & (prim.evaluability != "none")]
            if sel.empty:
                continue
            r = sel["rate"].to_numpy(dtype=float)
            ax.scatter(np.full(r.size, pos) + np.linspace(-0.16, 0.16, r.size),
                       np.maximum(r, 3e-4), s=11, alpha=0.8,
                       color=colors.get(source, "gray"), edgecolors="none",
                       zorder=3)
            top = sel.loc[sel["rate"].idxmax()]
            ax.annotate("%s %.0f%%" % (top["cls"], 100 * top["rate"]),
                        (pos, max(top["rate"], 3e-4)), fontsize=5.2,
                        xytext=(2, 3), textcoords="offset points")
            ticks.append(pos)
            labels.append("%s\n%s" % (short.get(ds, ds), source))
            pos += 1.0
    ax.axhline(GATE, color="crimson", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=5.0, rotation=90)
    ax.set_ylabel("per-population violation rate (log)")


def s4c(ax, ctx):
    cohorts, srcs, colors = ctx["cohorts"], ctx["srcs"], ctx["colors"]
    conv, GATE, short = ctx["conv"], ctx["GATE"], ctx["short"]
    pos = 0.0
    ticks, labels = [], []
    for ds in cohorts:
        for source in srcs:
            sel = conv[(conv.dataset == ds) & (conv.source == source)
                       & (conv.scope == "summary")
                       & (conv.cls == "ALL_evaluable")]
            if sel.empty:
                continue
            r = np.maximum(sel["rate"].to_numpy(dtype=float), 3e-4)
            ax.scatter(np.full(r.size, pos) + np.linspace(-0.18, 0.18, r.size),
                       r, s=7, alpha=0.5, color=colors.get(source, "gray"),
                       edgecolors="none")
            ax.plot([pos - 0.3, pos + 0.3],
                    [max(np.nanmedian(r), 3e-4)] * 2, color="black", lw=1.1)
            ticks.append(pos)
            labels.append("%s\n%s" % (short.get(ds, ds), source))
            pos += 1.0
    ax.axhline(GATE, color="crimson", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=5.0, rotation=90)
    ax.set_ylabel("pooled rate over all conventions (log)")


def s4d(ax, ctx):
    ag, short = ctx["ag"], ctx["short"]
    if not ag.empty:
        xs = np.arange(ag.shape[0])
        ax.bar(xs - 0.18, ag["coarse_agreement"], width=0.34, color="#2e9e8f",
               alpha=0.85, label="coarse classes")
        ax.bar(xs + 0.18, ag["native_agreement"], width=0.34, color="#d1743a",
               alpha=0.85, label="native classes")
        for i, r in ag.reset_index(drop=True).iterrows():
            ax.annotate("%.2f" % r["coarse_agreement"],
                        (xs[i] - 0.18, r["coarse_agreement"]), fontsize=5.6,
                        ha="center", va="bottom")
            ax.annotate("%.2f" % r["native_agreement"],
                        (xs[i] + 0.18, r["native_agreement"]), fontsize=5.6,
                        ha="center", va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels([short.get(d, d) for d in ag["dataset"]])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("fraction of shared cells")
        ax.legend(loc="lower right", frameon=False)


# ============================================================ figS5 (r19)
def ctx_s5():
    w = _csv("r19_weights.csv")
    e = _csv("r19_ess.csv")
    s = _csv("r19_sc.csv")
    cfg = {"alphas": [0.05, 0.10, 0.20], "clfs": ["logreg", "rf", "hgb"]}
    return {"w": w, "e": e, "s": s, "cfg": cfg,
            "titles": {"a": "the weights concentrate, and are not estimable",
                       "b": "a few cells carry the extrapolation",
                       "c": "every unit sits above the diagonal",
                       "d": "the violation is invisible without labels"}}


def s5a(ax, ctx):
    w = ctx["w"]
    ax.scatter(w.auc_domain, w.ess_pool_ratio, s=26, alpha=0.8,
               label="random cell split")
    ax.scatter(w.auc_domain_blocked, w.ess_pool_ratio, s=26, alpha=0.8,
               marker="^", label="whole images held out")
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("domain-classifier AUC")
    ax.set_ylabel("ESS / n   (source calibration pool)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")


def s5b(ax, ctx):
    w = ctx["w"]
    med = w.groupby("pair").agg(ess=("ess_pool_ratio", "median"),
                                m5=("mass_top5pct", "median"),
                                wmax=("w_max", "median")).sort_values("ess")
    yy = np.arange(len(med))
    ax.barh(yy, med.m5, color="#b2182b", alpha=0.85)
    ax.set_yticks(yy)
    ax.set_yticklabels(med.index, fontsize=6)
    ax.set_xlabel("fraction of weight mass held by the top 5% of cells")
    ax.set_xlim(0, 1.12)
    for i, (m5, wm) in enumerate(zip(med.m5, med.wmax)):
        ax.text(m5 + 0.01, i, "max w=%.0f" % wm, va="center", fontsize=5.4)


def s5c(ax, ctx):
    e, cfg = ctx["e"], ctx["cfg"]
    sub = e[e.alpha.isin(list(cfg["alphas"]))]
    for al in cfg["alphas"]:
        g = sub[np.isclose(sub.alpha, al)]
        ax.scatter(g.ess_obs, g.ess_star, s=16, alpha=0.75,
                   label="alpha=%.2f" % al)
    lo = float(np.nanmin(sub.ess_obs)) * 0.7
    hi = float(np.nanmax(np.concatenate(
        [sub.ess_obs, sub.ess_star[np.isfinite(sub.ess_star)]]))) * 1.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="ESS* = ESS")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ESS observed (source pool)")
    ax.set_ylabel("ESS* needed for rho* <= alpha")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    nun = int((e.ess_star_kind == "no_weighting").sum())
    ax.text(0.03, 0.06, "%d/%d units need ESS* = n:\nthe weights must vanish"
            % (nun, len(e)), transform=ax.transAxes, fontsize=7.5, va="bottom")


def s5d(ax, ctx):
    s, cfg = ctx["s"], ctx["cfg"]
    for al in cfg["alphas"]:
        g = s[np.isclose(s.alpha, al)]
        ax.scatter(g.tv_score, g.tv_gap, s=14, alpha=0.7,
                   label="alpha=%.2f" % al)
    g = s[(np.isfinite(s.tv_score)) & (np.isfinite(s.tv_gap))]
    if len(g) > 5:
        rho_s, _ = ss.spearmanr(g.tv_score, g.tv_gap)
        ax.text(0.03, 0.94, "Spearman rho = %.2f" % rho_s,
                transform=ax.transAxes, fontsize=8, va="top")
    ax.set_xlabel("score-marginal TV  (visible WITHOUT target labels)")
    ax.set_ylabel("error-curve TV  (needs target labels)")
    ax.legend(fontsize=6.5, frameon=False)


# ============================================================ figS8 (r35)
def ctx_s8():
    w = load("r31_within_summary.csv")
    pair = load("r31_paired.csv")
    lad = load("r31_ladder_summary.csv")

    arms = ["plain", "sc3", "sc10", "mond_class"]
    alphas = [0.05, 0.10]
    aclr = {0.05: "#2b6cb0", 0.10: "#c53030"}
    expect = {
        ("plain", 0.05): (0.8333333333333334, 0.5027777777777778, 0.6694444444444445),
        ("plain", 0.10): (0.8166666666666667, 0.5944444444444444, 0.7777777777777778),
        ("sc3", 0.05): (0.9500000000000000, 0.5444444444444444, 0.5944444444444444),
        ("sc3", 0.10): (0.9166666666666666, 0.6250000000000000, 0.7083333333333333),
        ("sc10", 0.05): (0.9750000000000000, 0.3222222222222222, 0.3472222222222222),
        ("sc10", 0.10): (0.9638888888888889, 0.6416666666666667, 0.6777777777777778),
        ("mond_class", 0.05): (0.9500000000000000, 0.4972222222222222, 0.5472222222222223),
        ("mond_class", 0.10): (0.9444444444444444, 0.7083333333333334, 0.7638888888888888),
    }
    got = {}
    for arm in arms:
        for al in alphas:
            r = one(w[(w.arm == arm) & np.isclose(w.alpha, al)],
                    "r31_within_summary arm=%s alpha=%.2f" % (arm, al))
            ca, cn, nf = (float(r["coverage_all"]), float(r["coverage_nonvac"]),
                          float(r["nonvac_frac"]))
            cc = float(r["coverage_nonvac_cond"])
            ea, en, ef = expect[(arm, al)]
            near(ca, ea, what="within coverage_all %s a=%.2f" % (arm, al))
            near(cn, en, what="within coverage_nonvac %s a=%.2f" % (arm, al))
            near(nf, ef, what="within nonvac_frac %s a=%.2f" % (arm, al))
            if not cn - 1e-9 <= cc <= ca + 1e-9:
                raise AssertionError(
                    "within coverage_nonvac_cond %s a=%.2f = %.6f is outside "
                    "[nonvac %.6f, coverage_all %.6f]" % (arm, al, cc, cn, ca))
            if not 0.0 <= cc <= 1.0:
                raise AssertionError("conditional readout outside [0, 1]")
            got[(arm, al)] = (ca, cn, nf, cc)
    r = one(w[(w.arm == "plain") & np.isclose(w.alpha, 0.10)], "plain a=0.10")
    near(r["coverage_nonvac_cond"], 0.7642857142857142,
         what="within plain a=0.10 coverage_nonvac_cond (manuscript 0.7643)")
    worst = max(v[1] for v in got.values())
    if not worst < R35.NOMINAL:
        raise AssertionError("within non-vacuous coverage reaches %.4f >= %.2f"
                             % (worst, R35.NOMINAL))

    paf = [("sc3", 0.030555555555555558, -0.05185185185185185, 0.1287037037037037),
           ("sc10", -0.04722222222222223, -0.1472222222222222, 0.06666666666666668),
           ("mond_class", 0.044444444444444446, -0.075, 0.1935185185185185)]
    pgot = []
    for arm, md, lo, hi in paf:
        r = one(pair[(pair.arm == arm) & (pair.ref == "plain")
                     & (pair.value == "covered_nonvac")],
                "r31_paired arm=%s ref=plain covered_nonvac" % arm)
        near(r["mean_diff"], md, what="paired mean_diff %s" % arm)
        near(r["ci_lo"], lo, what="paired ci_lo %s" % arm)
        near(r["ci_hi"], hi, what="paired ci_hi %s" % arm)
        if int(r["excludes_zero"]) != 0:
            raise AssertionError("paired %s unexpectedly excludes zero" % arm)
        if not (float(r["ci_lo"]) < 0.0 < float(r["ci_hi"])):
            raise AssertionError("paired %s interval does not cross zero" % arm)
        pgot.append((arm, md, lo, hi, int(r["n_pairs"]), int(r["n_units"])))

    tl = lad[lad.family == "transfer"].copy()
    if len(tl) == 0:
        raise AssertionError("r31_ladder_summary.csv has no transfer rows")
    ladder, acc_at = {}, {}
    for arm in sorted(tl.arm.unique()):
        for al in sorted(tl.alpha.unique()):
            r = one(tl[(tl.arm == arm) & np.isclose(tl.alpha, al)],
                    "r31_ladder transfer %s a=%.2f" % (arm, al))
            ladder[(arm, float(al))] = (int(r["k_first_nominal_all"]),
                                        int(r["k_first_nominal_useful"]))
            acc_at[(arm, float(al))] = (float(r["accept_at_all_k"])
                                        if np.isfinite(r["accept_at_all_k"])
                                        else float("nan"))
    ladder_expect = {
        ("mond_class", 0.05): 1000, ("mond_class", 0.10): 500,
        ("mond_class", 0.20): 200, ("prop_mond", 0.05): 1000,
        ("prop_mond", 0.10): 500, ("prop_mond", 0.20): 200,
        ("twostage_nw", 0.05): 1000, ("twostage_nw", 0.10): 500,
        ("twostage_nw", 0.20): 200, ("plain", 0.05): 1000,
        ("plain", 0.10): 500, ("plain", 0.20): 100,
    }
    for key, exp in ladder_expect.items():
        if ladder[key][0] != exp:
            raise AssertionError("ladder %s = %d, expected %d"
                                 % (str(key), ladder[key][0], exp))
    for arm in sorted(tl.arm.unique()):
        for al in sorted(tl.alpha.unique()):
            if ladder[(arm, float(al))][1] != -1:
                raise AssertionError(
                    "ladder useful readout for %s a=%.2f is not -1" % (arm, al))
    for arm in ("src", "twostage_iw"):
        for al in (0.05, 0.10, 0.20):
            if ladder[(arm, al)][0] != -1:
                raise AssertionError("ladder %s a=%.2f should never reach"
                                     % (arm, al))

    max_acc = max(acc_at[(a, al)] for a in sorted(set(x[0] for x in ladder))
                  if a != "oracle" for al in (0.05, 0.10, 0.20)
                  if np.isfinite(acc_at[(a, al)]))
    bound_val = np.ceil(round(max_acc, 9) * 1e4 - 1e-9) / 1e4
    bound_txt = "%.4f" % bound_val
    if bound_val < max_acc - 1e-12:
        raise AssertionError(
            "printed acceptance bound %s is below the computed maximum %.9f"
            % (bound_txt, max_acc))
    if bound_val - max_acc >= 1e-4:
        raise AssertionError(
            "printed acceptance bound %s is looser than the 4-dp ceiling of "
            "the computed maximum %.9f" % (bound_txt, max_acc))
    oracle_acc = max(acc_at[("oracle", al)] for al in (0.05, 0.10, 0.20)
                     if np.isfinite(acc_at[("oracle", al)]))
    if not oracle_acc > max_acc:
        raise AssertionError("the oracle reference should exceed the repair bound")
    rec("figS8 c", "max accept_at_all_k over repair arms", max_acc,
        "experiments/r31_ladder_summary.csv",
        "family==transfer & arm!=oracle & k_first_nominal_all>0", "accept_at_all_k")
    rec("figS8 c", "oracle accept_at_all_k (max)", oracle_acc,
        "experiments/r31_ladder_summary.csv",
        "family==transfer & arm==oracle & k_first_nominal_all>0", "accept_at_all_k")

    k500 = ladder[("plain", 0.10)][0]
    k200 = ladder[("mond_class", 0.20)][0]
    if k500 != 500 or k200 != 200:
        raise AssertionError("ladder grid changed: 500 -> %d, 200 -> %d"
                             % (k500, k200))

    arms_all = ["iw_sel", "mond_class", "oracle", "plain", "pool_iw",
                "prop_mond", "src", "twostage_iw", "twostage_nw", "unc_iw",
                "w_src"]
    hl = {"mond_class": "#1f5fa9", "prop_mond": "#2a9d8f",
          "twostage_nw": "#dd6b20", "plain": R35.CLR["grey"]}
    offs = dict(zip(arms_all, np.linspace(-0.16, 0.16, len(arms_all))))
    return {"w": w, "pair": pair, "lad": lad, "arms": arms, "alphas": alphas,
            "aclr": aclr, "got": got, "pgot": pgot, "ladder": ladder,
            "acc_at": acc_at, "max_acc": max_acc, "oracle_acc": oracle_acc,
            "bound_val": bound_val, "bound_txt": bound_txt, "k500": k500,
            "k200": k200, "arms_all": arms_all, "hl": hl, "offs": offs,
            "never": 2500.0,
            "titles": {"a": "within (cross-FOV), delta = 0.10",
                       "b": "paired vs plain, 20 units",
                       "c": "transfer ladder, 60 unit-folds"}}


def s8a(ax, ctx):
    arms, alphas, got, aclr = (ctx["arms"], ctx["alphas"], ctx["got"],
                               ctx["aclr"])
    x = 0.0
    ticks, lab = [], []
    for arm in arms:
        for al in alphas:
            ca, cn, nf, cc = got[(arm, al)]
            xx = x + (-0.17 if al == 0.05 else 0.17)
            ax.plot([xx, xx, xx], [cn, cc, ca], "-", color="0.75", lw=0.7,
                    zorder=2)
            ax.plot([xx], [ca], "o", ms=4.0, color=aclr[al], zorder=3)
            ax.plot([xx], [cc], "^", ms=4.2, mfc="white", mec=aclr[al],
                    mew=0.9, zorder=3)
            ax.plot([xx], [cn], "s", ms=4.0, mfc="white", mec=aclr[al],
                    mew=0.9, zorder=3)
            rec("figS8 a", "coverage_all    arm=%-10s alpha=%.2f" % (arm, al), ca,
                "experiments/r31_within_summary.csv",
                "arm==%s & alpha==%.2f" % (arm, al), "coverage_all")
            rec("figS8 a", "coverage_nonvac arm=%-10s alpha=%.2f" % (arm, al), cn,
                "experiments/r31_within_summary.csv",
                "arm==%s & alpha==%.2f" % (arm, al), "coverage_nonvac")
            rec("figS8 a", "nonvac_frac     arm=%-10s alpha=%.2f" % (arm, al), nf,
                "experiments/r31_within_summary.csv",
                "arm==%s & alpha==%.2f" % (arm, al),
                "nonvac_frac (annotation only)")
            rec("figS8 a", "nonvac_cond     arm=%-10s alpha=%.2f" % (arm, al), cc,
                "experiments/r31_within_summary.csv",
                "arm==%s & alpha==%.2f" % (arm, al),
                "coverage_nonvac_cond (P(covered | accepted set non-empty))")
        ticks.append(x)
        lab.append(arm)
        x += 1.0
    ax.axhline(R35.NOMINAL, color="k", ls="--", lw=1.0, zorder=1)
    ax.text(0.02, R35.NOMINAL + 0.020, "nominal %.2f" % R35.NOMINAL,
            fontsize=4.4, ha="left", va="bottom", color="0.25",
            transform=ax.get_yaxis_transform())
    ax.set_xticks(ticks)
    ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.set_ylim(0.26, 1.05)
    ax.set_ylabel("coverage of the error budget")
    ax.set_xlabel("repair arm")
    h_a = [Line2D([], [], color=aclr[a], marker="o", ls="none", ms=4.0,
                  label="alpha = %.2f" % a) for a in alphas]
    h_a.append(Line2D([], [], color="0.35", marker="o", ls="none", ms=4.0,
                      label="vacuous-inclusive"))
    h_a.append(Line2D([], [], color="0.35", marker="s", ls="none", ms=4.0,
                      mfc="white", label="non-vacuous"))
    h_a.append(Line2D([], [], color="0.35", marker="^", ls="none", ms=4.2,
                      mfc="white", label="P(cov | non-empty)"))
    ax.legend(handles=h_a, fontsize=4.6, loc="upper center", ncol=3,
              handletextpad=0.4, columnspacing=0.8,
              bbox_to_anchor=(0.5, -0.22), frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.92)
    ax.text(0.015, 0.185, "non-vacuous coverage\nstays below nominal\nfor every arm",
            transform=ax.transAxes, fontsize=4.6, color="0.3", va="center",
            ha="left", bbox=R35.BOX, zorder=6)


def s8b(ax, ctx):
    pgot = ctx["pgot"]
    for i, (arm, md, lo, hi, npr, nun) in enumerate(pgot):
        ax.errorbar([md], [i], xerr=[[max(md - lo, 0.0)], [max(hi - md, 0.0)]],
                    fmt="o", ms=4.0, lw=1.0, capsize=2.0,
                    color=R35.CLR[arm] if arm in R35.CLR else R35.CLR["ink"])
        ax.text(0.99, i + 0.20, "%+.3f\n[%+.3f, %+.3f]" % (md, lo, hi),
                fontsize=4.4, color="0.25", ha="right", va="center",
                transform=ax.get_yaxis_transform(), bbox=R35.BOX, zorder=6)
        rec("figS8 b", "paired covered_nonvac %s - plain" % arm, md,
            "experiments/r31_paired.csv",
            "arm==%s & ref==plain & value==covered_nonvac" % arm, "mean_diff")
        rec("figS8 b", "paired covered_nonvac %s - plain ci_lo" % arm, lo,
            "experiments/r31_paired.csv",
            "arm==%s & ref==plain & value==covered_nonvac" % arm, "ci_lo")
        rec("figS8 b", "paired covered_nonvac %s - plain ci_hi" % arm, hi,
            "experiments/r31_paired.csv",
            "arm==%s & ref==plain & value==covered_nonvac" % arm, "ci_hi")
        rec("figS8 b", "paired covered_nonvac %s - plain excludes_zero" % arm,
            0.0, "experiments/r31_paired.csv",
            "arm==%s & ref==plain & value==covered_nonvac" % arm,
            "excludes_zero (asserted == 0)")
    ax.axvline(0.0, color="k", ls="--", lw=1.0)
    ax.set_yticks(range(len(pgot)))
    ax.set_yticklabels([p[0] for p in pgot], fontsize=6.5)
    ax.set_ylim(-0.45, len(pgot) - 0.30)
    ax.set_xlim(-0.22, 0.24)
    ax.set_ylabel("arm")
    ax.set_xlabel("non-vacuous coverage minus plain\n"
                  "(95%% unit bootstrap, %d pairs)" % pgot[0][4])
    ax.text(0.985, 0.02, "every interval\ncrosses zero", transform=ax.transAxes,
            fontsize=4.9, color="0.3", ha="right", va="bottom", bbox=R35.BOX,
            zorder=6)


def s8c(ax, ctx):
    ladder, arms_all, hl = ctx["ladder"], ctx["arms_all"], ctx["hl"]
    offs, never = ctx["offs"], ctx["never"]
    max_acc, oracle_acc = ctx["max_acc"], ctx["oracle_acc"]
    bound_txt = ctx["bound_txt"]
    for arm in arms_all:
        xs, ys, raw = [], [], []
        for al in (0.05, 0.10, 0.20):
            k = ladder[(arm, al)][0]
            xs.append(al * (1.0 + offs[arm]))
            ys.append(never if k <= 0 else float(k))
            raw.append(k)
        if arm in hl:
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], "-",
                    marker="o", ms=3.4, lw=1.1, color=hl[arm], zorder=4,
                    label=arm)
        else:
            ax.plot(xs, ys, ls="none", marker="o", ms=2.6, mfc="none",
                    mec=R35.CLR["light"], mew=0.8, zorder=3)
        for al, k in zip((0.05, 0.10, 0.20), raw):
            rec("figS8 c", "k_first_nominal_all %-12s alpha=%.2f" % (arm, al),
                float(k), "experiments/r31_ladder_summary.csv",
                "family==transfer & arm==%s & alpha==%.2f" % (arm, al),
                "k_first_nominal_all (-1 never / 0 vacuous at k=0 -> top band)")
            rec("figS8 c", "k_first_nominal_useful %-6s alpha=%.2f" % (arm, al),
                float(ladder[(arm, al)][1]),
                "experiments/r31_ladder_summary.csv",
                "family==transfer & arm==%s & alpha==%.2f" % (arm, al),
                "k_first_nominal_useful")
    ax.axhline(2500.0, color="0.72", ls=":", lw=0.8, zorder=1)
    ax.text(0.040, 32.0, "no valid repair:\nnever, or only at\nk = 0 (vacuous)",
            fontsize=4.3, color="0.35", va="bottom", ha="left", bbox=R35.BOX,
            zorder=6)
    ax.annotate("%d at\nalpha = 0.10" % ctx["k500"], xy=(0.10, float(ctx["k500"])),
                xytext=(0.0435, 120.0), fontsize=4.7, color="0.25",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.5"))
    ax.annotate("%d only at\nalpha = 0.20" % ctx["k200"],
                xy=(0.205, float(ctx["k200"])),
                xytext=(0.12, 1100.0), fontsize=4.7, color="0.25",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.5"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.10, 0.20]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.05", "0.10", "0.20"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.get_offset_text().set_visible(False)
    ax.set_xlim(0.038, 0.27)
    ax.yaxis.set_major_locator(FixedLocator([25.0, 50.0, 100.0, 200.0, 500.0,
                                             1000.0, 2500.0]))
    ax.yaxis.set_major_formatter(FixedFormatter(["25", "50", "100", "200",
                                                 "500", "1000", "never"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.get_offset_text().set_visible(False)
    ax.set_ylim(12.0, 8000.0)
    ax.set_xlabel("failure budget alpha")
    ax.set_ylabel("target labels $k$ for a valid repair")
    h_c = [Line2D([], [], color=hl[a], marker="o", ls="-", ms=3.4, lw=1.1)
           for a in ("mond_class", "prop_mond", "twostage_nw", "plain")]
    h_c.append(Line2D([], [], color=R35.CLR["light"], marker="o", ls="none",
                      ms=2.6, mfc="none"))

    def acc_txt(a):
        return "/".join("n/a" if not np.isfinite(ctx["acc_at"][(a, al)])
                        else "%.3f" % ctx["acc_at"][(a, al)]
                        for al in (0.05, 0.10, 0.20))

    lab_c = ["%s %s" % (a, acc_txt(a))
             for a in ("mond_class", "prop_mond", "twostage_nw", "plain")]
    lab_c.append("other arms")
    ax.legend(h_c, lab_c, fontsize=4.3, loc="upper right", ncol=2,
              handletextpad=0.35, columnspacing=0.7, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.92)
    bound_note = ax.text(0.5, 0.012, "acceptance at that rung <= %s for the "
                         "repair arms\n(oracle reference %.4f at alpha = 0.20); "
                         "useful readout is -1"
                         % (bound_txt, oracle_acc),
                         transform=ax.transAxes, fontsize=4.2, color="0.3",
                         va="bottom", ha="center", bbox=R35.BOX, zorder=6)
    printed_bound = float(bound_note.get_text().split("<=")[1].strip()
                          .split()[0])
    if printed_bound < max_acc - 1e-12:
        raise AssertionError(
            "the printed acceptance bound %.4f lies below the true maximum "
            "%.9f" % (printed_bound, max_acc))


# ============================================================ figS9 (r35)
def ctx_s9():
    s = load("r32_sequential_summary.csv")
    sim = load("r32_sim.csv")

    fam = "cellsplit"
    arms = {"seq_eproc_bonf": "#1f5fa9", "cert_fixed_full": "#d1495b",
            "cert_fixed_union": "#2a9d8f"}
    mk = {"seq_eproc_bonf": "o", "cert_fixed_full": "s", "cert_fixed_union": "^"}
    show = [0.10, 0.20]
    expect = {
        ("seq_eproc_bonf", 0.10): (0.674395, 0.744707, 0.648188, 0.835717,
                                   0.911369, 0.837512, 0.969354),
        ("seq_eproc_bonf", 0.20): (0.389756, 0.420817, 0.290156, 0.580920,
                                   0.517411, 0.382712, 0.668562),
        ("cert_fixed_full", 0.10): (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ("cert_fixed_full", 0.20): (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ("cert_fixed_union", 0.10): (0.582184, 0.663074, 0.550455, 0.777730,
                                     0.864522, 0.767135, 0.944267),
        ("cert_fixed_union", 0.20): (0.350287, 0.383187, 0.252875, 0.531912,
                                     0.471199, 0.328472, 0.622048),
    }
    got = {}
    for arm in arms:
        for al in show:
            r = one(s[(s.family == fam) & (s.arm == arm)
                      & np.isclose(s.delta, 0.10) & np.isclose(s.alpha, al)],
                    "r32_sequential_summary %s a=%.2f d=0.10" % (arm, al))
            vals = (float(r["mean_t25_frac"]), float(r["mean_t50_frac"]),
                    float(r["mean_t50_frac_lo"]), float(r["mean_t50_frac_hi"]),
                    float(r["mean_t75_frac"]), float(r["mean_t75_frac_lo"]),
                    float(r["mean_t75_frac_hi"]))
            for j, tag in enumerate(("t25", "t50", "t50_lo", "t50_hi",
                                     "t75", "t75_lo", "t75_hi")):
                near(vals[j], expect[(arm, al)][j],
                     what="r32 %s a=%.2f %s" % (arm, al, tag))
            got[(arm, al)] = vals
    for al in show:
        e = got[("seq_eproc_bonf", al)][4]
        f = got[("cert_fixed_full", al)][4]
        u = got[("cert_fixed_union", al)][4]
        if not e < f:
            raise AssertionError("e-process does not beat cert_fixed_full at "
                                 "alpha=%.2f (%.4f vs %.4f)" % (al, e, f))
        if not e > u:
            raise AssertionError("e-process does not lose to cert_fixed_union "
                                 "at alpha=%.2f (%.4f vs %.4f)" % (al, e, u))

    s3 = {}
    for al in (0.05, 0.10, 0.20):
        r = one(s[(s.family == fam) & (s.arm == "seq_eproc_bonf")
                  & np.isclose(s.delta, 0.10) & np.isclose(s.alpha, al)],
                "r32 eproc s3 a=%.2f" % al)
        s3[al] = (float(r["s3_t75_frac"]), float(r["s3_t75_frac_lo"]),
                  float(r["s3_t75_frac_hi"]))
    for al, exp in ((0.05, (0.004401, 0.0, 0.013189)),
                    (0.10, (0.046847, 0.022885, 0.071496)),
                    (0.20, (0.046212, 0.033046, 0.059218))):
        for j, tag in enumerate(("mean", "lo", "hi")):
            near(s3[al][j], exp[j], what="r32 s3_t75_frac a=%.2f %s" % (al, tag))
        if not (s3[al][1] >= 0.0 and s3[al][0] > 0.0):
            raise AssertionError("s3_t75_frac at alpha=%.2f is not positive" % al)

    nominal = 0.05
    sm = sim[np.isclose(sim.nominal, nominal)]
    if len(sm) == 0:
        raise AssertionError("r32_sim.csv has no nominal == 0.05 rows")
    if "naive_peek_tight" in set(sim.bound):
        raise AssertionError("r32_sim.csv unexpectedly names naive_peek_tight")
    bounds = {}
    for b in ("seq_eproc_anytime", "seq_cs_hoeffding", "control_peek_hoeffding",
              "control_peek_cp"):
        cur = []
        for mu in (0.05, 0.10, 0.20):
            r = one(sm[(sm.bound == b) & np.isclose(sm.mu, mu)],
                    "r32_sim %s mu=%.2f" % (b, mu))
            cur.append((float(r["emp"]), float(r["mc_se"]), int(r["crossings"])))
        bounds[b] = cur
    near(bounds["seq_eproc_anytime"][0][0], 0.02175, what="sim eproc mu=0.05")
    near(bounds["seq_eproc_anytime"][1][0], 0.03275, what="sim eproc mu=0.10")
    near(bounds["seq_eproc_anytime"][2][0], 0.02975, what="sim eproc mu=0.20")
    near(bounds["control_peek_cp"][0][0], 0.22150, what="sim peek mu=0.05")
    near(bounds["control_peek_cp"][1][0], 0.25725, what="sim peek mu=0.10")
    near(bounds["control_peek_cp"][2][0], 0.27275, what="sim peek mu=0.20")
    for i in range(3):
        if not bounds["seq_eproc_anytime"][i][0] <= nominal:
            raise AssertionError("the e-process exceeds nominal in the sim")
        if bounds["seq_cs_hoeffding"][i][0] != 0.0:
            raise AssertionError("the Hoeffding sequence crosses in the sim")
        if not bounds["control_peek_cp"][i][0] > nominal:
            raise AssertionError("the peeking control does not break the level")
    ratio_max = bounds["control_peek_cp"][2][0] / nominal
    near(ratio_max, 5.455, tol=5e-3, what="sim peeking control break factor")

    return {"s": s, "sim": sim, "arms": arms, "mk": mk, "show": show,
            "got": got, "s3": s3, "bounds": bounds, "nominal": nominal,
            "ratio_max": ratio_max,
            "titles": {"a": "cellsplit, delta = 0.10", "b": "paired vs union",
                       "c": "level, delta = 0.05"}}


def s9a(ax, ctx):
    arms, mk, show, got = ctx["arms"], ctx["mk"], ctx["show"], ctx["got"]
    for arm in ("seq_eproc_bonf", "cert_fixed_full", "cert_fixed_union"):
        for al in show:
            t25, t50, lo50, hi50, t75, lo75, hi75 = got[(arm, al)]
            ls = "-" if al == 0.10 else "--"
            ax.fill_between([0.50, 0.75], [lo50, lo75], [hi50, hi75],
                            color=arms[arm],
                            alpha=(0.12 if al == 0.10 else 0.07), lw=0, zorder=1)
            ax.plot([0.25], [t25], marker=mk[arm], ms=3.6, ls="none",
                    color=arms[arm], zorder=3)
            ax.errorbar([0.50, 0.75], [t50, t75],
                        yerr=[[max(t50 - lo50, 0.0), max(t75 - lo75, 0.0)],
                              [max(hi50 - t50, 0.0), max(hi75 - t75, 0.0)]],
                        fmt=mk[arm], ms=3.6, lw=1.1, capsize=1.6, color=arms[arm],
                        ls=ls, zorder=3)
            for tag, val in (("t25", t25), ("t50", t50), ("t75", t75)):
                rec("figS9 a", "%s alpha=%.2f %s pool fraction" % (arm, al, tag),
                    val, "experiments/r32_sequential_summary.csv",
                    "family==cellsplit & arm==%s & alpha==%.2f & delta==0.10"
                    % (arm, al), "mean_%s_frac" % tag)
            for tag, val in (("t50_lo", lo50), ("t50_hi", hi50),
                             ("t75_lo", lo75), ("t75_hi", hi75)):
                rec("figS9 a", "%s alpha=%.2f %s" % (arm, al, tag), val,
                    "experiments/r32_sequential_summary.csv",
                    "family==cellsplit & arm==%s & alpha==%.2f & delta==0.10"
                    % (arm, al), "mean_%s_frac" % tag)
    ax.axvline(0.75, color="0.55", ls=":", lw=0.9, zorder=0)
    ax.text(0.700, 1.052, "t75", fontsize=5.0, color="0.35", ha="right",
            va="top")
    ax.text(0.775, got[("seq_eproc_bonf", 0.10)][4],
            "%.3f" % got[("seq_eproc_bonf", 0.10)][4], fontsize=5.0,
            color=arms["seq_eproc_bonf"], ha="left", va="center")
    ax.text(0.775, got[("seq_eproc_bonf", 0.20)][4],
            "%.3f" % got[("seq_eproc_bonf", 0.20)][4], fontsize=5.0,
            color=arms["seq_eproc_bonf"], ha="left", va="center")
    ax.set_xticks([0.25, 0.50, 0.75])
    ax.set_xticklabels(["0.25", "0.50", "0.75"])
    ax.set_xlim(0.15, 0.90)
    ax.set_ylim(0.235, 1.06)
    ax.set_xlabel("acceptance target $t$ reached\n"
                  "(t25 has no interval column)")
    ax.set_ylabel("mean fraction of the pool needed to reach $t$\n"
                  "over all runs; never-reached counts as 1.0")
    disp = {"seq_eproc_bonf": "seq_eproc_bonf",
            "cert_fixed_full": "cert_fixed_full (whole pool = 1.0000)",
            "cert_fixed_union": "cert_fixed_union"}
    h_ls = [Line2D([], [], color=arms[a], marker=mk[a], ls="none", ms=3.6,
                   label=disp[a]) for a in ("seq_eproc_bonf", "cert_fixed_full",
                                            "cert_fixed_union")]
    h_ls.append(Line2D([], [], color="0.25", ls="-", lw=1.1,
                       label="alpha = 0.10"))
    h_ls.append(Line2D([], [], color="0.25", ls="--", lw=1.1,
                       label="alpha = 0.20"))
    h_ls.append(Line2D([], [], color="0.6", ls="-", lw=2.4, alpha=0.4,
                       label="95% unit bootstrap"))
    ax.legend(handles=h_ls, fontsize=4.2, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.26), handletextpad=0.35,
              columnspacing=0.7, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.92)


def s9b(ax, ctx):
    s3, arms = ctx["s3"], ctx["arms"]
    xs = [0.05, 0.10, 0.20]
    ys = [s3[a][0] for a in xs]
    lo = [max(s3[a][0] - s3[a][1], 0.0) for a in xs]
    hi = [max(s3[a][2] - s3[a][0], 0.0) for a in xs]
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=4.2, lw=1.1, capsize=2.0,
                color=arms["seq_eproc_bonf"])
    ax.axhline(0.0, color="k", ls="--", lw=1.0)
    for a in xs:
        ax.text(a, s3[a][2] + 0.006, "%+.4f" % s3[a][0], fontsize=4.6,
                ha="center", va="bottom", color="0.25")
        rec("figS9 b", "s3_t75_frac eproc - union alpha=%.2f" % a, s3[a][0],
            "experiments/r32_sequential_summary.csv",
            "family==cellsplit & arm==seq_eproc_bonf & alpha==%.2f & delta==0.10"
            % a, "s3_t75_frac")
        rec("figS9 b", "s3_t75_frac alpha=%.2f ci_lo" % a, s3[a][1],
            "experiments/r32_sequential_summary.csv",
            "family==cellsplit & arm==seq_eproc_bonf & alpha==%.2f & delta==0.10"
            % a, "s3_t75_frac_lo")
        rec("figS9 b", "s3_t75_frac alpha=%.2f ci_hi" % a, s3[a][2],
            "experiments/r32_sequential_summary.csv",
            "family==cellsplit & arm==seq_eproc_bonf & alpha==%.2f & delta==0.10"
            % a, "s3_t75_frac_hi")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0.05", "0.10", "0.20"])
    ax.set_xlim(0.015, 0.245)
    ax.set_ylim(-0.014, 0.100)
    ax.set_xlabel("failure budget alpha")
    ax.set_ylabel("paired t75 pool fraction,\ne-process minus union")
    ax.text(0.02, 0.98, "positive = the e-process\nneeds MORE labels",
            fontsize=4.9, color="0.3", va="top", ha="left",
            transform=ax.transAxes, bbox=R35.BOX)


def s9c(ax, ctx):
    bounds, nominal, ratio_max = ctx["bounds"], ctx["nominal"], ctx["ratio_max"]
    order = ["seq_eproc_anytime", "seq_cs_hoeffding", "control_peek_hoeffding",
             "control_peek_cp"]
    bclr = {"seq_eproc_anytime": "#1f5fa9", "seq_cs_hoeffding": "#2a9d8f",
            "control_peek_hoeffding": R35.CLR["light"], "control_peek_cp": "#d1495b"}
    blab = {"seq_eproc_anytime": "e-process (anytime valid)",
            "seq_cs_hoeffding": "Hoeffding sequence (valid)",
            "control_peek_hoeffding": "Hoeffding fixed-sample, no correction",
            "control_peek_cp": "CP fixed-sample at every look (naive peek)"}
    mus = [0.05, 0.10, 0.20]
    base = np.arange(3, dtype=float)
    for j, b in enumerate(order):
        xx = base + (j - 1.5) * 0.13
        yy = [bounds[b][i][0] for i in range(3)]
        ee = [bounds[b][i][1] for i in range(3)]
        ax.errorbar(xx, yy, yerr=ee, fmt="o", ms=3.4, lw=0.9, capsize=1.4,
                    color=bclr[b], label=blab[b])
        for i in range(3):
            rec("figS9 c", "%s mu=%.2f empirical crossing rate" % (b, mus[i]),
                yy[i], "experiments/r32_sim.csv",
                "bound==%s & mu==%.2f & nominal==0.05" % (b, mus[i]), "emp")
            rec("figS9 c", "%s mu=%.2f mc_se" % (b, mus[i]), ee[i],
                "experiments/r32_sim.csv",
                "bound==%s & mu==%.2f & nominal==0.05" % (b, mus[i]), "mc_se")
    ax.axhline(nominal, color="k", ls="--", lw=1.0)
    ax.text(-0.40, nominal + 0.008, "nominal %.2f" % nominal, fontsize=5.4,
            color="0.25")
    ax.set_xticks(base)
    ax.set_xticklabels(["0.05", "0.10", "0.20"])
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-0.014, 0.345)
    ax.set_xlabel("drift $\\mu$ of the stream")
    ax.set_ylabel("empirical crossing rate, 4000 streams")
    ax.legend(fontsize=4.4, loc="upper left", ncol=2, handletextpad=0.35,
              columnspacing=0.7)
    ax.text(0.98, 0.40, "%.2f / %.2f = %.1fx\nat mu = 0.20"
            % (bounds["control_peek_cp"][2][0], nominal, ratio_max),
            transform=ax.transAxes, fontsize=5.0, ha="right", color="0.25",
            bbox=R35.BOX)


# ============================================================ figS10 (r35)
def ctx_s10():
    cs = load("r34_phase2_certificate_summary.csv")
    src = load("r34_phase2_by_source.csv")
    rat = load("r34_phase2_ratio.csv")
    met = load("r34_phase2_metrics.csv")
    pnl = load("r34_phase2_transfer_panels.csv")
    r03 = load("r03_summary.csv")
    r26 = load("r26_plugin_baseline_summary.csv")
    r72 = _csv("r72_panel_repair.csv")

    cert = {}
    for family, acc, risk, cov in (
            ("ext_within", 0.7264066000477634, 0.07784141183551987,
             0.6555555555555556),
            ("ext_transfer", 0.8394325484270428, 0.48929001571630854,
             0.03333333333333333)):
        r = one(cs[(cs.family == family) & np.isclose(cs.alpha, R35.ALPHA_MAIN)],
                "r34_phase2_certificate_summary %s a=0.10" % family)
        near(r["mean_accept"], acc, what="%s mean_accept" % family)
        near(r["risk_nonvac"], risk, what="%s risk_nonvac" % family)
        near(r["coverage"], cov, what="%s coverage" % family)
        if int(r["frac_nonvacuous"]) != 1:
            raise AssertionError("%s is not fully non-vacuous" % family)
        cert[family] = (float(r["mean_accept"]), float(r["risk_nonvac"]),
                        float(r["coverage"]))
    if abs(cert["ext_within"][2] - 0.6556) > 5e-4:
        raise AssertionError("ext_within coverage != 0.6556")
    bench = {}
    for kind, want in (("cellsplit", 1.0000000000000000),
                       ("within", 0.8166666666666667),
                       ("transfer", 0.3972222222222222)):
        r = one(r03[(r03.kind == kind) & (r03.method == "crc")
                    & np.isclose(r03.alpha, R35.ALPHA_MAIN)],
                "r03_summary %s crc a=0.10" % kind)
        near(r["coverage"], want, what="benchmark %s coverage" % kind)
        bench[kind] = float(r["coverage"])
    for family in ("within", "transfer"):
        r = one(r26[(r26.family == family) & (r26.arm == "certificate")
                    & np.isclose(r26.alpha, R35.ALPHA_MAIN)
                    & np.isclose(r26.delta, R35.DELTA)],
                "r26 %s certificate a=0.10 d=0.10" % family)
        near(r["coverage"], bench[family], what="r26 cross-check %s" % family)

    bys, famof = {}, {"STELLAR_Intestine_B004": "ext_within",
                      "STELLAR_BarrettsEsophagus": "ext_transfer",
                      "STELLAR_Tonsil": "ext_transfer"}
    for target, family in famof.items():
        r = one(src[(src.family == family) & (src.target == target)
                    & np.isclose(src.alpha, R35.ALPHA_MAIN)],
                "r34_phase2_by_source %s -> %s a=0.10" % (family, target))
        bys[target] = (float(r["coverage"]), float(r["mean_accept"]),
                       float(r["mean_risk"]), int(r["n_eval"]))
    near(bys["STELLAR_BarrettsEsophagus"][0], 0.06666666666666667,
         what="Barrett's coverage a=0.10")
    if bys["STELLAR_Tonsil"][0] != 0.0:
        raise AssertionError("tonsil coverage is not exactly 0.0000")
    panels = {}
    for _, r in pnl.iterrows():
        panels[r["unit_tag"].split("->")[-1]] = (int(r["n_panel"]),
                                                 int(r["n_shared_classes"]),
                                                 int(r["n_target_cells"]))
    if len(pnl) != 2:
        raise AssertionError("r34_phase2_transfer_panels.csv should have 2 rows")
    for tgt, exp in (("STELLAR_BarrettsEsophagus", (26, 5, 9869)),
                     ("STELLAR_Tonsil", (26, 5, 5346))):
        if panels[tgt] != exp:
            raise AssertionError("panel for %s is %s, expected %s"
                                 % (tgt, str(panels[tgt]), str(exp)))

    key = ["unit", "clf", "repeat", "alpha", "family"]
    cr = met[(met.method == "crc") & np.isclose(met.alpha, R35.ALPHA_MAIN)]
    orc = met[(met.method == "oracle") & np.isclose(met.alpha, R35.ALPHA_MAIN)]
    j = cr.merge(orc[key + ["accept_rate"]].rename(
        columns={"accept_rate": "acc_or"}), on=key, how="inner",
        validate="one_to_one")
    j["ratio"] = j["accept_rate"] / j["acc_or"].where(j["acc_or"] > 0)
    j["over"] = (j["covered"] == 0)
    j["flagged"] = j["ratio"] > 1.0
    te = j[j.family == "ext_transfer"]
    if len(te) != 180:
        raise AssertionError("ext_transfer evaluation count is %d" % len(te))
    counts = (int(te.over.sum()), int(te.flagged.sum()),
              int((te.over & ~te.flagged).sum()),
              int((~te.over & te.flagged).sum()),
              int((te.acc_or == 0).sum()))
    r = one(rat[(rat.level == "eval") & (rat.family == "ext_transfer")
                & np.isclose(rat.alpha, R35.ALPHA_MAIN)],
            "r34_phase2_ratio eval ext_transfer a=0.10")
    stored = (int(r["n_over_budget"]), int(r["n_flagged"]),
              int(r["over_not_flagged"]), int(r["flagged_not_over"]),
              int(r["n_ratio_undefined"]))
    if counts != stored:
        raise AssertionError("ratio recount %s != stored %s"
                             % (str(counts), str(stored)))
    near(r["accept_ratio"], 2.3407583793528226, what="ext_transfer ratio a=0.10")
    near(r["mean_risk_crc"], 0.48929001571630837, what="ratio mean risk")
    near(r["coverage_crc"], 0.03333333333333333, what="ratio coverage")
    if counts[2] != 60:
        raise AssertionError("missed over-budget evaluations is %d, not 60"
                             % counts[2])
    if counts[4] != 60:
        raise AssertionError("undefined ratios is %d, not 60" % counts[4])
    if counts[3] != 0:
        raise AssertionError("the diagnostic raises a false alarm")

    return {"cs": cs, "src": src, "rat": rat, "met": met, "pnl": pnl,
            "r03": r03, "r26": r26, "r72": r72, "cert": cert,
            "bench": bench, "bys": bys, "famof": famof, "panels": panels,
            "counts": counts, "te": te,
            "titles": {"a": "guarantee before repair", "b": "coverage by deployment source",
                       "c": "transfer, alpha = 0.10",
                       "d": "net review removed, frozen vs refitted"}}


def s10a(ax, ctx):
    cert, bench = ctx["cert"], ctx["bench"]
    groups = [("acceptance", "mean_accept", None, None),
              ("realised\nrisk", "risk_nonvac", R35.ALPHA_MAIN,
               "risk 0.10"),
              ("coverage", "coverage", R35.NOMINAL,
               "90% coverage reference")]
    cfamily = {"ext_within": "#1f5fa9", "ext_transfer": "#d1495b"}
    for gi, (glab, col, nomv, nomlab) in enumerate(groups):
        for fi, family in enumerate(("ext_within", "ext_transfer")):
            xx = gi + (-0.19 if fi == 0 else 0.19)
            val = {"mean_accept": cert[family][0], "risk_nonvac": cert[family][1],
                   "coverage": cert[family][2]}[col]
            ax.bar([xx], [val], width=0.34, color=cfamily[family],
                   alpha=0.95 if fi == 0 else 0.55,
                   edgecolor=cfamily[family], linewidth=0.6, zorder=2)
            lift = 0.055 if col == "risk_nonvac" else 0.028
            ax.text(xx, min(val + lift, 1.08),
                    "%.3f" % val, fontsize=5.0, ha="center", color="0.2",
                    zorder=4)
            rec("figS10 a", "%s %s alpha=0.10" % (family, col), val,
                "experiments/r34_phase2_certificate_summary.csv",
                "family==%s & alpha==0.10" % family, col)
        if nomv is not None:
            ax.plot([gi - 0.30, gi + 0.30], [nomv, nomv], ls="--", lw=1.0,
                    color="k", zorder=3, solid_capstyle="butt")
            ax.text(gi + 0.42, nomv + 0.014, nomlab.replace("risk", "nominal"),
                    fontsize=4.2, ha="left", va="bottom", color="0.25")
    for kind, lab0, m in (("cellsplit", "random", "D"),
                          ("within", "cross-FOV", "s"),
                          ("transfer", "transfer", "^")):
        lab = "%s\n%.3f" % (lab0, bench[kind])
        ax.plot([3.24], [bench[kind]], marker=m, ms=4.0, mfc="white",
                mec=R35.CLR["ink"], mew=0.9, ls="none", zorder=5)
        ax.text(3.10, bench[kind], lab, fontsize=4.6, va="center", ha="right",
                color="0.25")
        rec("figS10 a", "benchmark %s coverage alpha=0.10" % kind, bench[kind],
            "experiments/r03_summary.csv",
            "kind==%s & method==crc & alpha==0.10" % kind, "coverage")
    ax.text(3.10, 1.185, "depot benchmark", fontsize=4.6, va="bottom", ha="right",
            color="0.25")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=5.4)
    ax.set_xlim(-0.55, 3.45)
    ax.set_ylim(0.0, 1.26)
    ax.set_ylabel("value")
    ax.set_xlabel("readout at 10% error budget (delta = 0.10)")
    h_a = [Line2D([], [], color=cfamily[f], marker="s", ls="none", ms=4.0,
                  alpha=0.95 if f == "ext_within" else 0.55,
                  label="cross-FOV (intestine)" if f == "ext_within"
                  else "transfer (tonsil/Barrett)")
           for f in ("ext_within", "ext_transfer")]
    ax.legend(handles=h_a, fontsize=4.4, loc="upper left", ncol=1,
              handletextpad=0.35, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.92)


def s10b(ax, ctx):
    bys, famof, panels = ctx["bys"], ctx["famof"], ctx["panels"]
    order = ["STELLAR_Intestine_B004", "STELLAR_BarrettsEsophagus",
             "STELLAR_Tonsil"]
    labs = ["intestine\n(cross-FOV)", "Barrett's\noesophagus", "tonsil"]
    sclr = ["#1f5fa9", "#e6a817", "#d1495b"]
    for i, (tgt, lab, cl) in enumerate(zip(order, labs, sclr)):
        cov, acc, risk, nev = bys[tgt]
        ax.bar([i], [cov], width=0.56, color=cl, alpha=0.9, edgecolor=cl,
               linewidth=0.6, zorder=2)
        ax.text(i, max(cov, 0.0) + 0.055, "%.4f" % cov, fontsize=5.6,
                ha="center", color="0.2", zorder=4)
        if cov == 0.0:
            ax.plot([i - 0.28, i + 0.28], [0.0, 0.0], color=cl, lw=1.4,
                    zorder=4)
        ax.text(i, -0.055, "accept %.4f\nrisk %.4f" % (acc, risk), fontsize=4.3,
                ha="center", va="top", color="0.3")
        f = famof[tgt]
        for col, val in (("coverage", cov), ("mean_accept", acc),
                         ("mean_risk", risk)):
            rec("figS10 b", "%s %s alpha=0.10" % (col, tgt), val,
                "experiments/r34_phase2_by_source.csv",
                "family==%s & target==%s & alpha=0.10" % (f, tgt), col)
    for tgt, i in (("STELLAR_BarrettsEsophagus", 1), ("STELLAR_Tonsil", 2)):
        npn, nsh, ntg = panels[tgt]
        ax.text(i, 1.34, "%d shared markers\n%d shared classes\n%d cells"
                % (npn, nsh, ntg), fontsize=4.3, ha="center", va="top",
                color="0.3")
        rec("figS10 b", "transfer panel %s target cells" % tgt, float(ntg),
            "experiments/r34_phase2_transfer_panels.csv",
            "unit_tag endswith %s" % tgt,
            "n_panel=%d n_shared_classes=%d n_target_cells=%d" % (npn, nsh, ntg))
    ax.axhline(R35.NOMINAL, color="k", ls="--", lw=1.0, zorder=3)
    ax.text(2.45, R35.NOMINAL + 0.03, "nominal %.2f" % R35.NOMINAL,
            fontsize=5.0, ha="right", va="bottom", color="0.2")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=5.2)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.20, 1.44)
    ax.set_ylabel("coverage of the error budget")
    ax.set_xlabel("deployment source, alpha = 0.10, 90\nevaluations each")


def s10c(ax, ctx):
    te, counts = ctx["te"], ctx["counts"]
    undef_y = 0.55
    ax.axhspan(0.36, 0.68, color="0.93", zorder=0)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, zorder=1)
    ax.axvline(R35.ALPHA_MAIN, color="0.4", ls=":", lw=1.0, zorder=1)
    ov, win = te[te.over], te[~te.over]
    unc, far = ov[ov.acc_or == 0], ov[ov.acc_or > 0]
    ax.plot(win.selective_risk, win.ratio, ls="none", marker="o", ms=3.2,
            mfc="white", mec=R35.CLR["within"], mew=0.8, zorder=3)
    ax.plot(far.selective_risk, far.ratio, ls="none", marker="o", ms=3.2,
            color=R35.CLR["over"], zorder=3)
    ax.plot(unc.selective_risk, np.full(len(unc), undef_y), ls="none",
            marker="x", ms=3.4, color=R35.CLR["over"], zorder=3)
    for _, rr in far.iterrows():
        rec("figS10 c", "eval ratio over budget (defined)", rr["ratio"],
            "experiments/r34_phase2_metrics.csv (crc joined to oracle)",
            "family==ext_transfer & alpha==0.10 & covered==0 & oracle>0",
            "accept_rate_crc / accept_rate_oracle")
        rec("figS10 c", "eval realised risk over budget", rr["selective_risk"],
            "experiments/r34_phase2_metrics.csv",
            "family==ext_transfer & alpha==0.10 & covered==0 (method==crc)",
            "selective_risk")
    for _, rr in win.iterrows():
        rec("figS10 c", "eval ratio within budget", rr["ratio"],
            "experiments/r34_phase2_metrics.csv (crc joined to oracle)",
            "family==ext_transfer & alpha==0.10 & covered==1",
            "accept_rate_crc / accept_rate_oracle")
    for _, rr in unc.iterrows():
        rec("figS10 c", "eval risk, over budget, ratio undefined",
            rr["selective_risk"], "experiments/r34_phase2_metrics.csv",
            "family==ext_transfer & alpha==0.10 & covered==0 & oracle==0",
            "selective_risk")
    rec("figS10 c", "over-budget evaluations", float(len(ov)),
        "experiments/r34_phase2_metrics.csv",
        "family==ext_transfer & alpha==0.10 & covered==0", "count")
    rec("figS10 c", "over-budget and flagged", float(len(far)),
        "experiments/r34_phase2_metrics.csv",
        "family==ext_transfer & alpha==0.10 & covered==0 & ratio>1", "count")
    rec("figS10 c", "over-budget and missed", float(len(unc)),
        "experiments/r34_phase2_metrics.csv",
        "family==ext_transfer & alpha==0.10 & covered==0 & oracle==0", "count")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([undef_y, 1.0, 3.0, 10.0, 30.0,
                                             100.0]))
    ax.yaxis.set_major_formatter(FixedFormatter(["undefined", "1", "3", "10",
                                                 "30", "100"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.get_offset_text().set_visible(False)
    ax.set_ylim(0.34, 140.0)
    ax.set_xlim(0.0, 1.02)
    ax.xaxis.set_major_locator(FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.0", "0.2", "0.4", "0.6",
                                                 "0.8", "1.0"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("realised risk among\naccepted cells\n"
                  "(budget alpha = 0.10)")
    ax.set_ylabel("acceptance ratio, certificate / oracle")
    h_c = [Line2D([], [], color=R35.CLR["over"], marker="o", ls="none", ms=3.2,
                  label="over budget"),
           Line2D([], [], color=R35.CLR["over"], marker="x", ls="none", ms=3.4,
                  label="ratio undefined"),
           Line2D([], [], color=R35.CLR["within"], marker="o", ls="none", ms=3.2,
                  mfc="white", label="within budget"),
           Line2D([], [], color="k", ls="--", lw=1.0, label="ratio = 1")]
    ax.legend(handles=h_c, fontsize=4.4, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.42), handletextpad=0.35,
              columnspacing=0.7)
    ax.text(0.13, 0.985, "%d of %d over-budget\ntransfer evaluations are\n"
            "missed; every one has\nan undefined ratio\n(oracle accepts no cell)"
            % (counts[2], counts[0]),
            transform=ax.transAxes, fontsize=4.5, color="0.25", ha="left",
            va="top", bbox=R35.BOX, zorder=6)


def s10d(ax, ctx):
    """The repair, priced: net review removed against the labels it costs.

    The four deployments are the two same-tissue panel changes on the MIBI-TOF
    microarray (colon and breast cores), the same array's tonsil cores against
    the STELLAR tonsil cohort, which is CODEX rather than MIBI-TOF, and the
    depot colorectal to GVHD change, which moves tissue as well as panel.  For
    each, the labels are drawn from the deployment cohort and either all go to
    the calibrator of a frozen model (dashed) or half refit the model and half
    calibrate it (solid).  Net is accepted minus labels read over the batch, so
    a rung below the zero line costs more labels than it saves."""
    d = ctx["r72"]
    AL = R35.ALPHA_MAIN
    ks = [250, 500, 1000, 2000, 4000]
    pos = {k: i for i, k in enumerate(ks)}
    groups = [
        ("B_same_tissue_panel", None, "colon, 13 shared", "#1f5fa9"),
        ("C_same_tissue_panel_breast", None, "breast, 8 shared", "#d1495b"),
        ("D_cross_platform_tonsil", None, "tonsil, 11 shared", "#e6a817"),
        ("A_cross_tissue_panel", "CRC_Feb2623->GVHD_Feb1323",
         "colorectal to GVHD, 27 shared", "#2a9d8f"),
    ]
    ax.axhspan(-0.95, 0.0, color="0.94", zorder=0)
    ax.axhline(0.0, color="0.35", lw=0.9, zorder=1)
    handles = []
    for part, pair, lab, col in groups:
        s = d[d.part == part]
        if pair is not None:
            s = s[s.pair == pair]
        s = s[np.isclose(s.alpha, AL)]
        for arm_pat, ls, al, mfc, want, tag in (
                (r"tgt_k\d+", (0, (3.4, 2.0)), 0.80, "white", False, "frozen"),
                (r"adapt_k\d+", "-", 1.0, col, True, lab)):
            rows = s[s.arm.str.fullmatch(arm_pat)]
            xs, ys = [], []
            for k in ks:
                g = rows[rows.k == k]
                if not len(g):
                    continue
                v = float((((g.n_accepted - g.n_labels) / g.n_pool).mean()))
                xs.append(pos[k])
                ys.append(v)
                if want:
                    rec("figS10 d", "%s net removed, k=%d, alpha=0.10" % (part, k),
                        v, "experiments/r72_panel_repair.csv",
                        "part==%s & arm==%s & alpha==0.10" % (part, "adapt_k%d" % k),
                        "(n_accepted - n_labels) / n_pool")
            if not xs:
                continue
            ax.plot(xs, ys, ls=ls, lw=1.15, color=col, alpha=al, marker="o",
                    ms=3.0, mfc=mfc, mec=col, mew=0.9, zorder=3)
            if want:
                handles.append(Line2D([], [], color=col, ls="-", lw=1.15,
                                      marker="o", ms=3.0, mfc=col, mec=col,
                                      label=lab))
    handles.append(Line2D([], [], color=R35.CLR["ink"], ls=(0, (3.4, 2.0)),
                          lw=1.15, marker="o", ms=3.0, mfc="white",
                          mec=R35.CLR["ink"], label="frozen model"))
    ax.text(0.03, 0.055, "below zero: the labels cost\nmore than they save",
            transform=ax.transAxes, fontsize=4.3, color="0.35", ha="left",
            va="bottom")
    bc = d[(d.part == "C_same_tissue_panel_breast")
           & np.isclose(d.alpha, AL) & (d.arm == "adapt_k1000")]
    if len(bc):
        v = float((((bc.n_accepted - bc.n_labels) / bc.n_pool).mean()))
        ax.text(pos[1000] + 0.12, v + 0.05, "%.2f" % v, fontsize=4.8,
                color="#d1495b", ha="left", va="bottom")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks], fontsize=5.2)
    ax.set_xlim(-0.45, len(ks) - 0.55)
    ax.set_ylim(-0.95, 0.86)
    ax.set_xlabel("target labels drawn from the deployment cohort\n"
                  "(budget alpha = 0.10)")
    ax.set_ylabel("net share of the batch removed")
    ax.legend(handles=handles, fontsize=4.3, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.34), handletextpad=0.35,
              columnspacing=0.7)


# ------------------------------------------------------------- full layouts
def layout_s3(fig):
    axes = fig.subplots(1, 2)
    return [axes[0], axes[1]]


def layout_s4(fig):
    axes = fig.subplots(2, 2)
    return [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]


def layout_s5(fig):
    axes = fig.subplots(2, 2)
    return [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]


def layout_s8(fig):
    gs = R35.grid(fig, [1.18, 0.98, 1.32])
    return [fig.add_subplot(gs[0, i]) for i in range(3)]


def layout_s9(fig):
    gs = R35.grid(fig, [1.20, 0.90, 1.16], wspace=0.54, bottom=0.245)
    return [fig.add_subplot(gs[0, i]) for i in range(3)]


def layout_s10(fig):
    gs = fig.add_gridspec(1, 4, width_ratios=[1.30, 1.02, 1.06, 1.30],
                          wspace=0.46, left=0.060, right=0.988,
                          bottom=0.275, top=0.865)
    return [fig.add_subplot(gs[0, i]) for i in range(4)]


# ------------------------------------------------------------- figure specs
FIGS = {
    "figS3_marker_sensitivity": {
        "ctx": ctx_s3, "size": (7.4, 3.3), "layout": layout_s3, "tight": True,
        "panels": [("a", s3a), ("b", s3b)]},
    "figS4_marker_sources": {
        "ctx": ctx_s4, "size": (7.2, 5.4), "layout": layout_s4, "tight": True,
        "panels": [("a", s4a), ("b", s4b), ("c", s4c), ("d", s4d)]},
    "figS5_weight_shift": {
        "ctx": ctx_s5, "size": (11.0, 8.6), "layout": layout_s5, "tight": True,
        "panels": [("a", s5a), ("b", s5b), ("c", s5c), ("d", s5d)]},
    "figS8_repair": {
        "ctx": ctx_s8, "size": (7.4, 3.0), "layout": layout_s8, "tight": False,
        "panels": [("a", s8a), ("b", s8b), ("c", s8c)]},
    "figS9_sequential": {
        "ctx": ctx_s9, "size": (7.4, 3.0), "layout": layout_s9, "tight": False,
        "panels": [("a", s9a), ("b", s9b), ("c", s9c)]},
    "figS10_external": {
        "ctx": ctx_s10, "size": (9.6, 3.0), "layout": layout_s10, "tight": False,
        "panels": [("a", s10a), ("b", s10b), ("c", s10c), ("d", s10d)]},
}

KEYS = ["figS3_marker_sensitivity", "figS4_marker_sources",
        "figS5_weight_shift", "figS8_repair", "figS9_sequential",
        "figS10_external"]
