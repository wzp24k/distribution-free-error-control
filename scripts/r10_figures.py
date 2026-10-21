"""R10 -- manuscript figures.

Fig 1  benchmark composition and the data audit
Fig 2  calibration collapses under transfer
Fig 3  what the controller does (real example + schematic)
Fig 4  does the guarantee hold (benchmark families, simulation, calibration size)
Fig 5  what the guarantee buys (risk-coverage, review budget)
Fig 6  marker consistency and its downstream effect

Every figure is written to figures/ as both PDF (vector, for submission) and
PNG (for reading).  Panels are labelled a, b, c, ... in the order they are added.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, data as D, paths
from errctl.labels import COARSE_MAP
from errctl.markerlogic import canonical
from errctl.calibration import reliability_curve

FIG = paths.FIGURES
RES = paths.RESULTS
PRED = os.path.join(paths.WORK, "predictions")
COHORTS = paths.MANUAL_DATASETS
SHORT = {"Breast_Jun0524": "Breast", "CRC_Feb2623": "CRC",
         "GVHD_Feb1323": "GVHD", "Melanoma_Oct2522": "Melanoma"}
CLR = {"crc": "#1f5fa9", "naive": "#d1495b", "oracle": "#8d99ae",
       "conformal_set": "#e6a817", "crc_mondrian": "#2a9d8f",
       "crc_cond": "#6ab187", "crc_eb": "#b0b0b0", "accept_all": "#444444",
       "crc_weighted": "#7b4fa0", "expert": "#1f5fa9", "depot_HQ": "#d1495b"}

_SHORT2FULL = {v: k for k, v in SHORT.items()}

SETCLR = {"exchangeable_500": "#1f5fa9", "hetero_strong": "#e6a817",
          "shift_mild": "#2a9d8f", "shift_strong": "#d1495b"}


def _row_label(fig, ax, text, dy=0.055):
    """Descriptor above a row of axes, clear of the panel letters."""
    bb = ax.get_position()
    fig.text(bb.x0, bb.y1 + dy, text, fontsize=8, color="0.35", ha="left")


def _unit_target(unit):
    """Run unit -> target cohort.

    transfer_Breast_Jun0524_to_GVHD_Feb1323_fold2 -> GVHD_Feb1323
    cellsplit_Breast_Jun0524_fold0                 -> Breast_Jun0524
    """
    u = re.sub(r"_fold\d+$", "", str(unit))
    if u.startswith("transfer_"):
        return u.split("_to_")[-1]
    return re.sub(r"^(cellsplit|within)_", "", u)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "font.family": "DejaVu Sans", "legend.frameon": False,
})


def panel(ax, letter, title=None):
    ax.set_title(letter.upper(), loc="left", fontweight="bold", pad=6)
    if title:
        ax.set_title(title, loc="right", fontsize=7.5, color="0.35")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/%s.pdf/.png" % name)


def load_master(ds):
    return pd.read_parquet(os.path.join(paths.DATASETS, "master_%s.parquet" % ds))


def scores_of(probs, kind="maxprob"):
    probs = np.asarray(probs, dtype=np.float64)
    if kind == "maxprob":
        return 1.0 - probs.max(1)
    p = np.clip(probs, 1e-12, 1.0)
    if kind == "entropy":
        return -(p * np.log(p)).sum(1) / np.log(p.shape[1])
    part = np.sort(p, axis=1)
    return -(part[:, -1] - part[:, -2])


# --------------------------------------------------------------------- Fig 1
def load_masters():
    """The per-cohort master tables, loaded once per figure."""
    return {ds: load_master(ds) for ds in COHORTS}


def p1a(ax, masters, letter="a", title="benchmark"):
    """Panel a -- expert-labelled cells per cohort, by coarse class."""
    tab = []
    for ds in COHORTS:
        g = masters[ds]
        g = g[g["is_expert"] & (g["coarse_expert"] != "Unidentified")]
        tab.append(g["coarse_expert"].value_counts().rename(SHORT[ds]))
    comp = pd.concat(tab, axis=1).fillna(0).astype(int)
    top = comp.sum(1).sort_values(ascending=False).index[:8]
    comp = pd.concat([comp.loc[top], comp.drop(top).sum().rename("other").to_frame().T])
    bottom = np.zeros(comp.shape[1])
    cmap = plt.get_cmap("tab20")
    for i, cls in enumerate(comp.index):
        ax.bar(comp.columns, comp.loc[cls], bottom=bottom, label=cls,
               color=cmap(i % 20), width=0.62)
        bottom += comp.loc[cls].values
    for i, c in enumerate(comp.columns):
        del i, c
    ax.set_xticks(range(comp.shape[1]))
    ax.set_xticklabels(["%s\n%s" % (c, format(int(bottom[i]), ","))
                        for i, c in enumerate(comp.columns)], fontsize=6.5)
    ax.set_ylim(0, float(bottom.max()) * 1.34)
    ax.set_ylabel("usable expert-labelled cells")
    ax.legend(ncol=3, fontsize=5.2, loc="upper center", columnspacing=0.8,
              handlelength=1.1, handletextpad=0.35)
    ax.grid(False)
    panel(ax, letter, title)


def p1b(ax, masters, letter="b", title="panel overlap"):
    """Panel b -- Jaccard overlap of the marker panels of the four cohorts."""
    panels = {}
    for ds in COHORTS:
        cols = [c for c in masters[ds].columns if c in D.marker_table(
            ds, kind="HighQualityLabels" if ds in paths.HQ_DATASETS else "ManualLabels")[0]]
        panels[ds] = {canonical(c) for c in cols}
    J = np.zeros((4, 4))
    for i, a in enumerate(COHORTS):
        for j, b in enumerate(COHORTS):
            inter = len(panels[a] & panels[b])
            union = len(panels[a] | panels[b])
            J[i, j] = inter / union if union else np.nan
    # A deliberately light scale keeps black editable labels legible even in
    # the diagonal cells, where Jaccard is 1.00.
    from matplotlib.colors import LinearSegmentedColormap
    light_blues = LinearSegmentedColormap.from_list(
        "panel_overlap", ["#ffffff", "#e6f0fa", "#9ecae1"])
    im = ax.imshow(J, vmin=0, vmax=1.0, cmap=light_blues)
    ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[d] for d in COHORTS], rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels([SHORT[d] for d in COHORTS])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, "%.2f" % J[i, j], ha="center", va="center",
                    fontsize=6.5, color="black")
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label="Jaccard of marker panels")
    panel(ax, letter, title)


def p1c(ax, masters, letter="c", title="the shipped HQ labels"):
    """Panel c -- expert against the shipped semi-supervised HQ labels."""
    labels, vals = [], []
    for ds in paths.HQ_DATASETS:
        g = masters[ds]
        g = g[g["is_expert"] & g["class_hq"].notna()]
        ok = (g["coarse_hq"] == g["coarse_expert"]).mean()
        labels.append(SHORT[ds]); vals.append(ok)
    ax.bar(labels, [1 - v for v in vals], color=CLR["depot_HQ"], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, 1 - v + 0.012, "disagreement\n%.0f%%" % (100 * (1 - v)), ha="center", fontsize=6.5)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("expert vs HQ\ndisagreement", fontsize=7)
    ax.grid(False)
    panel(ax, letter, title)


def p1d(ax, letter="d", title="delivery audit (all FOVs)"):
    """Panel d -- are the labelled cellIDs present in the shipped masks?"""
    rows = []
    for ds in COHORTS:
        lab = D.load_manual_labels(ds)
        base = paths.manual_dir(ds)
        hit = tot = 0
        # All labelled fields of view, not a sample: an earlier six-FOV sample
        # gave 6.7% for GVHD against 17.5% over every field, and the discrepancy
        # was pure sampling noise.
        for img in sorted(lab["image"].unique()):
            d = os.path.join(base, "Images", img)
            if not os.path.isdir(d):
                continue
            f = next((x for x in os.listdir(d) if "segmentation" in x.lower()), None)
            if f is None:
                continue
            ids = set(np.unique(tifffile.imread(os.path.join(d, f))).tolist())
            want = set(lab.loc[lab["image"] == img, "cellID"].astype(int))
            hit += len(want & ids); tot += len(want)
        rows.append((SHORT[ds], hit / tot if tot else np.nan, tot))
    df = pd.DataFrame(rows, columns=["cohort", "rate", "n"])
    ax.bar(df["cohort"], df["rate"], color=[CLR["expert"] if r > 0.5 else CLR["naive"]
                                            for r in df["rate"]], width=0.5)
    for i, r in enumerate(df["rate"]):
        lab = "100%" if r >= 0.995 else "%.1f%%" % (100 * r)
        ax.text(i, r + 0.02, lab, ha="center", fontsize=6.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("labelled cellIDs found in the shipped mask")
    ax.grid(False)
    panel(ax, letter, title)


def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    masters = load_masters()
    p1a(axes[0, 0], masters)
    p1b(axes[0, 1], masters)
    p1c(axes[1, 0], masters)
    p1d(axes[1, 1])
    fig.tight_layout()
    save(fig, "fig1_benchmark")


# --------------------------------------------------------------------- Fig 2
def load_ece():
    """ECE / accuracy table, keyed by split family and domain."""
    ece = pd.read_csv(os.path.join(RES, "r03_ece.csv"))
    ece = ece.rename(columns={"kind": "family"})
    ece["cohort"] = ece["unit"].map(_unit_target)
    return ece


def _reliability_panel(ax, ds, letter, title, first=False, legend=False):
    """Reliability diagram of one transfer model, in both domains."""
    # Sources disagree on the class inventory, so their probability matrices
    # cannot be pooled.  Confidence and correctness are scalars and pool
    # cleanly across sources, which is all a reliability curve needs.
    cs, cw, ts, tw = [], [], [], []
    for f in sorted(os.listdir(PRED)):
        if not (f.startswith("transfer_") and f.endswith("__hgb.npz") and ("to_%s_" % ds) in f):
            continue
        d = np.load(os.path.join(PRED, f), allow_pickle=True)
        for P, Y, S, W in ((d["cal_probs"], d["cal_y"], cs, cw),
                           (d["test_probs"], d["test_y"], ts, tw)):
            ok = np.asarray(Y) >= 0
            S.append(np.asarray(P[ok].max(1), dtype=np.float64))
            W.append((P[ok].argmax(1) == np.asarray(Y)[ok]).astype(float))
    if not cs:
        return
    for S, W, c, lab in ((cs, cw, CLR["expert"], "training domain"),
                         (ts, tw, CLR["naive"], "deployment domain")):
        xs, ysb, _ = reliability_curve(np.concatenate(S), np.concatenate(W), n_bins=10)
        ax.plot(xs, ysb, "o-", ms=2.5, lw=1.0, color=c, label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # The cohort name is used as the descriptor so that it cannot collide
    # with the panel letter inside these very narrow axes.
    panel(ax, letter, title)
    if first:
        ax.set_ylabel("accuracy")
    ax.set_xlabel("confidence")
    if legend:
        ax.legend(loc="lower right", fontsize=6)


def p2a(ax, letter="a", title="Breast"):
    _reliability_panel(ax, COHORTS[0], letter, title, first=True, legend=True)


def p2b(ax, letter="b", title="CRC"):
    _reliability_panel(ax, COHORTS[1], letter, title)


def p2c(ax, letter="c", title="GVHD"):
    _reliability_panel(ax, COHORTS[2], letter, title)


def p2d(ax, letter="d", title="Melanoma"):
    _reliability_panel(ax, COHORTS[3], letter, title)


def _transfer_bars(ax, ece, col, ylab, letter, title, legend=False):
    """One calibration measure under transfer: both domains, side by side."""
    t = (ece[ece.family == "transfer"]
         .pivot_table(index="cohort", columns="domain", values=col))
    t = t.reindex([ds for ds in COHORTS if ds in t.index])
    x = np.arange(len(t))
    ax.bar(x - 0.19, t["source_cal"], width=0.36, color=CLR["expert"], label="training")
    ax.bar(x + 0.19, t["target"], width=0.36, color=CLR["naive"], label="deployment")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[d] for d in t.index], rotation=30, ha="right",
                       fontsize=5.8)
    ax.set_ylabel(ylab)
    ax.grid(False)
    if legend:
        ax.legend(fontsize=6)
    panel(ax, letter, title)


def p2e(ax, ece, letter="e", title="under transfer"):
    _transfer_bars(ax, ece, "ece", "ECE", letter, title, legend=True)


def p2f(ax, ece, letter="f", title="under transfer"):
    _transfer_bars(ax, ece, "acc", "accuracy", letter, title)


def p2g(ax, ece, letter="g", title="by split family"):
    """ECE by split family: only cross-tissue transfer degrades."""
    t = (ece.pivot_table(index="family", columns="domain", values="ece"))
    t = t.reindex(["cellsplit", "within", "transfer"])
    x = np.arange(len(t))
    ax.bar(x - 0.19, t["source_cal"], width=0.36, color=CLR["expert"])
    ax.bar(x + 0.19, t["target"], width=0.36, color=CLR["naive"])
    ax.set_xticks(x)
    ax.set_xticklabels(["random\nsplit", "cross-\nFOV", "cross-\ntissue"], fontsize=6.2)
    ax.set_ylabel("ECE")
    ax.grid(False)
    panel(ax, letter, title)


def p2h(ax, ece, letter="h", title="relative ECE"):
    """Deployment ECE as a multiple of the training-domain ECE."""
    t = ece.pivot_table(index="family", columns="domain", values="ece")
    t = t.reindex(["cellsplit", "within", "transfer"])
    ratio = (t["target"] / t["source_cal"]).to_numpy()
    x = np.arange(len(t))
    colors = [CLR["naive"] if r > 1.5 else CLR["expert"] for r in ratio]
    ax.bar(x, ratio, width=0.55, color=colors)
    ax.axhline(1.0, color="0.35", ls="--", lw=0.8)
    for xi, r in zip(x, ratio):
        ax.text(xi, r + 0.08, "%.2fx" % r, ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["random\nsplit", "cross-\nFOV", "cross-\ntissue"],
                       fontsize=6.2)
    ax.set_ylabel("ECE ratio")
    ax.set_ylim(0, max(2.8, float(ratio.max()) * 1.15))
    ax.grid(False)
    panel(ax, letter, title)


def fig2():
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.6),
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    ece = load_ece()
    p2a(axes[0, 0], "a"); p2b(axes[0, 1], "b")
    p2c(axes[0, 2], "c"); p2d(axes[0, 3], "d")
    p2e(axes[1, 0], ece, "e"); p2f(axes[1, 1], ece, "f")
    p2g(axes[1, 2], ece, "g"); p2h(axes[1, 3], ece, "h")
    # tight_layout cannot reserve the gap this two-row layout needs, so the
    # margins are set explicitly and the top row is labelled by hand.
    fig.subplots_adjust(left=0.065, right=0.99, top=0.83, bottom=0.10,
                        hspace=0.85, wspace=0.62)
    fig.text(0.065, 0.90, "reliability under cross-tissue transfer",
             fontsize=8, color="0.35", ha="left")
    save(fig, "fig2_calibration")


# --------------------------------------------------------------------- Fig 3
def load_fig3_ctx():
    """One within-cohort run plus its fitted controller, for all Fig 3 panels."""
    f = "within_GVHD_Feb1323_fold0__hgb.npz"
    d = np.load(os.path.join(PRED, f), allow_pickle=True)
    cal_probs, cal_y = d["cal_probs"], d["cal_y"]
    test_probs, test_y = d["test_probs"], d["test_y"]
    rep = np.nonzero(d["cal_rep"][:, 0])[0]
    s_cal = scores_of(cal_probs[rep]); e_cal = (cal_probs[rep].argmax(1) != cal_y[rep]).astype(float)
    s_te = scores_of(test_probs); e_te = (test_probs.argmax(1) != test_y).astype(float)
    alpha = 0.10
    ctl = C.SelectiveRiskController(alpha, 0.10).fit(s_cal, e_cal)
    return {"s_cal": s_cal, "e_cal": e_cal, "s_te": s_te, "e_te": e_te,
            "alpha": alpha, "ctl": ctl}


def p3a(ax, ctx, letter="a", title="choosing the threshold"):
    """Panel a -- where the calibration scores sit, and the chosen threshold."""
    s_cal, e_cal, ctl = ctx["s_cal"], ctx["e_cal"], ctx["ctl"]
    ax.hist(s_cal, bins=40, color="0.75", label="calibration cells")
    ax.hist(s_cal[e_cal > 0], bins=40, color=CLR["naive"], label="of which wrong")
    ax.axvline(ctl.lambda_hat, color=CLR["crc"], lw=1.4)
    ax.annotate("accept\nbelow", xy=(ctl.lambda_hat, ax.get_ylim()[1] * 0.85),
                xytext=(ctl.lambda_hat * 0.35, ax.get_ylim()[1] * 0.95), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate("review", xy=(ctl.lambda_hat, ax.get_ylim()[1] * 0.6),
                xytext=(min(1.0, ctl.lambda_hat * 1.25), ax.get_ylim()[1] * 0.8), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("1 - max probability (score)")
    ax.set_ylabel("cells")
    ax.legend(fontsize=6, loc="center right")
    ax.grid(False)
    panel(ax, letter, title)


def p3b(ax, ctx, letter="b", title="the bound is the constraint"):
    """Panel b -- empirical risk against the certified upper bound."""
    s_cal, e_cal = ctx["s_cal"], ctx["e_cal"]
    ctl, alpha = ctx["ctl"], ctx["alpha"]
    order = np.argsort(s_cal, kind="mergesort")
    s_sorted, e_sorted = s_cal[order], e_cal[order]
    k = np.arange(1, s_sorted.size + 1)
    risk = np.cumsum(e_sorted) / k
    ax.plot(s_sorted, risk, color=CLR["naive"], lw=1.2, label="calibration risk")
    lam_grid = s_sorted[ctl.cand_ - 1]
    ax.plot(lam_grid, ctl.upper_n_ / np.maximum(ctl.lower_d_, 1e-12), color=CLR["crc"],
            lw=1.2, label="certified upper bound")
    feasible = ctl.bound_ratio_ <= alpha
    if feasible.any():
        ax.fill_between(lam_grid, 0, min(1.0, 4 * alpha), where=feasible,
                        color=CLR["crc"], alpha=0.08, lw=0)
    ax.axhline(alpha, color="k", ls="--", lw=0.8)
    ax.axvline(ctl.lambda_hat, color=CLR["crc"], ls=":", lw=1.0)
    ax.set_xlabel("threshold on the score")
    ax.set_ylabel("error rate among accepted")
    ax.set_ylim(0, min(1.0, 4 * alpha))
    ax.legend(fontsize=6)
    panel(ax, letter, title)


def p3c(ax, letter="c", title="the rule"):
    """Panel c -- schematic of the accept / abstain split."""
    ax.axis("off")
    ax.add_patch(Rectangle((0.01, 0.62), 0.45, 0.26, fc="#dce9f7", ec="0.4", lw=0.6))
    ax.text(0.235, 0.75, "annotated\nset", ha="center", va="center", fontsize=6.4)
    ax.add_patch(Rectangle((0.54, 0.62), 0.45, 0.26, fc="#f6d6da", ec="0.4", lw=0.6))
    ax.text(0.765, 0.75, "review\nqueue", ha="center", va="center", fontsize=6.4)
    ax.add_patch(Rectangle((0.02, 0.18), 0.96, 0.24, fc="0.94", ec="0.4", lw=0.6))
    ax.text(0.50, 0.30, "score\ns(x) = 1 - p_max", ha="center", va="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((0.5, 0.62), (0.5, 0.44), arrowstyle="-|>", lw=0.9,
                                 color="0.3", mutation_scale=8))
    ax.add_patch(FancyArrowPatch((0.23, 0.44), (0.23, 0.62), arrowstyle="-|>", lw=0.9,
                                 color=CLR["crc"], mutation_scale=8))
    ax.add_patch(FancyArrowPatch((0.77, 0.44), (0.77, 0.62), arrowstyle="-|>", lw=0.9,
                                 color=CLR["naive"], mutation_scale=8))
    ax.text(0.5, 0.04, "lambda_hat = max { lambda :\n    N+(lambda) <= alpha . D-(lambda) }",
            ha="center", va="bottom", fontsize=6.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel(ax, letter, title)


def fig3():
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.0], wspace=0.32)
    ctx = load_fig3_ctx()
    p3a(fig.add_subplot(gs[0, 0]), ctx)
    p3b(fig.add_subplot(gs[0, 1]), ctx)
    p3c(fig.add_subplot(gs[0, 2]))
    save(fig, "fig3_method")


# --------------------------------------------------------------------- Fig 4
def load_metrics():
    """Per-run coverage / acceptance metrics, keyed by split family."""
    return pd.read_csv(os.path.join(RES, "r03_metrics.csv")).rename(columns={"kind": "family"})


def p4a(ax, M, letter="a", title="guarantee coverage"):
    """Panel a -- coverage at a 10% budget, by family and method."""
    methods = ["crc", "crc_cond", "conformal_set", "crc_mondrian", "naive"]
    mlabel = {"crc": "certified", "crc_cond": "conditional",
              "conformal_set": "conformal set", "crc_mondrian": "Mondrian",
              "naive": "uncalibrated"}
    fams = ["cellsplit", "within", "transfer"]
    sub = M[np.isclose(M["alpha"], 0.10)]
    w = 0.15
    for i, m in enumerate(methods):
        vals = [sub[(sub.family == f) & (sub.method == m)]["covered"].mean() for f in fams]
        ax.bar(np.arange(3) + (i - 2) * w, vals, width=w,
               label=mlabel[m], color=CLR.get(m, "0.5"))
    ax.axhline(0.90, color="k", ls="--", lw=0.9)
    ax.text(1.98, 0.925, "nominal 0.90", fontsize=6.2, ha="center")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["random split\n(exchangeable)", "cross-FOV", "cross-tissue"])
    ax.set_ylabel("coverage at alpha = 0.10")
    ax.set_ylim(0, 1.42)
    ax.legend(ncol=3, fontsize=5.4, loc="upper center", columnspacing=0.8,
              handlelength=1.1, handletextpad=0.35)
    ax.grid(False)
    panel(ax, letter, title)


def p4b(ax, M, letter="b", title="coverage vs budget"):
    """Panel b -- coverage against the budget, controller and baseline."""
    fams = ["cellsplit", "within", "transfer"]
    flabel = {"cellsplit": "random split", "within": "cross-FOV",
              "transfer": "cross-tissue"}
    for f, c in zip(fams, ("#1f5fa9", "#e6a817", "#d1495b")):
        s = M[(M.family == f) & (M.method == "crc")]
        t = s.groupby("alpha")["covered"].mean()
        ax.plot(t.index, t.values, "o-", ms=3, lw=1.1, color=c,
                label="certified, %s" % flabel[f])
        s2 = M[(M.family == f) & (M.method == "naive")]
        t2 = s2.groupby("alpha")["covered"].mean()
        ax.plot(t2.index, t2.values, "s--", ms=2.5, lw=0.9, color=c, alpha=0.5,
                label="uncalibrated, %s" % flabel[f])
    ax.plot([0.02, 0.5], [0.9, 0.9], "k--", lw=0.9)
    ax.set_xlabel("budget alpha"); ax.set_ylabel("coverage")
    ax.set_ylim(-0.30, 1.06)
    ax.legend(fontsize=5.4, ncol=3, loc="lower center", frameon=False,
              handlelength=1.6, columnspacing=1.0)
    panel(ax, letter, title)


def p4c(ax, letter="c", title="valid but vacuous?"):
    """Panel c -- Monte-Carlo coverage against acceptance under shift."""
    p = os.path.join(RES, "r09_simulation_summary.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        # Coverage on its own is not enough to judge a controller: under shift
        # the guarantee can be held by accepting almost nothing, which is valid
        # but useless.  A certified method is only interesting in the upper
        # right of this plane, so coverage and acceptance are shown jointly.
        s = s[np.isclose(s["alpha"], 0.20)]
        sets = ["exchangeable_500", "hetero_strong", "shift_mild", "shift_strong"]
        methods = ["crc", "crc_cond", "crc_weighted_known", "naive"]
        mk = {"crc": "o", "crc_cond": "^", "crc_weighted_known": "s", "naive": "X"}
        mlab = {"crc": "certified", "crc_cond": "conditional",
                "crc_weighted_known": "weighted", "naive": "uncalibrated"}
        for st in sets:
            for m in methods:
                r = s[(s.setting == st) & (s.method == m)]
                if not len(r):
                    continue
                # clamp at the left edge so that a zero-acceptance run is still
                # visible rather than drawn at an unplottable log coordinate
                a = max(float(r["mean_accept"].iloc[0]), 3e-3)
                ax.scatter(a, float(r["coverage"].iloc[0]), marker=mk[m], s=24,
                           color=SETCLR[st], edgecolor="0.2", linewidth=0.4, zorder=3)
        ax.axhline(0.90, color="k", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlim(1.8e-3, 1.6)
        ax.set_ylim(0, 1.14)
        ax.set_xlabel("mean acceptance (log scale)")
        ax.set_ylabel("coverage, alpha = 0.20")
        h = [Line2D([], [], marker=mk[m], ls="", color="0.35", ms=4.5,
                    label=mlab[m])
             for m in methods]
        ax.legend(handles=h, fontsize=5.4, loc="lower right")
    panel(ax, letter, title)


def p4d(ax, letter="d", title="cost of few labels"):
    """Panel d -- what a small calibration set costs in acceptance."""
    p = os.path.join(RES, "r07_ablations.csv")
    if os.path.exists(p):
        b = pd.read_csv(p)
        s = b[(b.ablation == "n_cal") & (b.family == "cellsplit") & np.isclose(b["alpha"], 0.10)]
        t = s.groupby("value").agg(accept=("accept_rate", "mean"), cov=("covered", "mean"))
        t.index = t.index.astype(float)
        t = t.sort_index()
        ax.plot(t.index, t["accept"], "o-", ms=3, color=CLR["crc"], label="certified acceptance")
        ax.plot(t.index, t["cov"], "s--", ms=2.5, color="0.5", label="coverage")
        ax.axhline(0.90, color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("calibration cells")
        ax.set_ylabel("at alpha = 0.10")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)
    panel(ax, letter, title)


def fig4():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    M = load_metrics()
    p4a(axes[0, 0], M)
    p4b(axes[0, 1], M)
    p4c(axes[1, 0])
    p4d(axes[1, 1])
    fig.tight_layout()
    save(fig, "fig4_coverage")


# --------------------------------------------------------------------- Fig 5
def load_efficiency():
    """Acceptance / oracle-ratio table behind Fig 5."""
    return pd.read_csv(os.path.join(RES, "r04_t2_efficiency.csv"))


def p5a(ax, letter="a", title="risk-coverage (hgb)"):
    """Panel a -- selective risk against coverage, random split."""
    for ds in COHORTS:
        f = sorted(x for x in os.listdir(PRED)
                   if x.startswith("cellsplit_%s" % ds) and x.endswith("__hgb.npz"))
        if not f:
            continue
        d = np.load(os.path.join(PRED, f[0]), allow_pickle=True)
        probs, y = d["test_probs"], d["test_y"]
        s = scores_of(probs); e = (probs.argmax(1) != y).astype(float)
        cov, risk = C.risk_coverage_curve(s, e, n_points=120)
        ax.plot(cov, risk, lw=1.1, label=SHORT[ds])
    ax.axhline(0.10, color="k", ls="--", lw=0.8)
    ax.set_xlabel("coverage"); ax.set_ylabel("selective risk")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.5)
    ax.legend(fontsize=6)
    panel(ax, letter, title)


def p5b(ax, t2, letter="b", title="acceptance vs oracle"):
    """Panel b -- acceptance of each rule, by split family."""
    s = t2[t2.cohort == "ALL"]
    fams = ["cellsplit", "within", "transfer"]
    s = s.set_index("family").reindex(fams)
    w = 0.2
    for i, (col, lab) in enumerate((("accept_crc", "certified (ours)"),
                                    ("accept_oracle", "oracle on test"),
                                    ("accept_naive", "uncalibrated"),
                                    ("accept_conformal", "conformal set"))):
        ax.bar(np.arange(3) + (i - 1.5) * w, s[col], width=w, label=lab,
               color=[CLR["crc"], CLR["oracle"], CLR["naive"], CLR["conformal_set"]][i])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["random\nsplit", "cross-FOV", "cross-\ntissue"])
    ax.set_ylabel("acceptance at alpha = 0.10")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6)
    ax.grid(False)
    panel(ax, letter, title)


def p5c(ax, t2, letter="c", title="the tell-tale ratio"):
    """Panel c -- certified acceptance relative to the oracle."""
    fams = ["cellsplit", "within", "transfer"]
    s = t2[t2.cohort != "ALL"].copy()
    s["target"] = (s["cohort"].str.replace(r"^(cellsplit|within|transfer)_", "", regex=True)
                   .map(lambda c: _SHORT2FULL.get(c, c)))
    lab = {"cellsplit": "random split", "within": "cross-FOV", "transfer": "cross-tissue"}
    x = np.arange(len(COHORTS))
    for i, fam in enumerate(fams):
        v = [s[(s.family == fam) & (s.target == ds)]["frac_of_oracle"].values for ds in COHORTS]
        v = np.array([float(a[0]) if len(a) else np.nan for a in v])
        ax.bar(x + (i - 1) * 0.26, np.clip(v, 0.03, None) - 0.03, bottom=0.03, width=0.26,
               label=lab[fam], color=[CLR["crc"], CLR["crc_mondrian"], CLR["naive"]][i])
        for xi, vi in zip(x + (i - 1) * 0.26, v):
            if np.isfinite(vi) and vi > 1.0:
                ax.text(xi, vi * 1.15, "%.1f" % vi, ha="center", fontsize=5.4, color="0.3")
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 60)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in COHORTS], rotation=20)
    ax.set_ylabel("certified acceptance / oracle")
    ax.grid(False)
    ax.legend(fontsize=6)
    panel(ax, letter, title)


def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    t2 = load_efficiency()
    p5a(axes[0])
    p5b(axes[1], t2)
    p5c(axes[2], t2)
    fig.tight_layout()
    save(fig, "fig5_efficiency")


# --------------------------------------------------------------------- Fig 6
def load_markers():
    """The three marker-consistency tables used by Fig 6."""
    a = pd.read_csv(os.path.join(RES, "r06_marker_audit.csv"))
    a = a[a.reading == "or"]
    s = a[a.coarse_class == "ALL_complete"]
    sl = pd.read_csv(os.path.join(RES, "r06_selective.csv"))
    sp = pd.read_csv(os.path.join(RES, "r06_spatial.csv"))
    return {"audit": s, "selective": sl, "spatial": sp}


def p6a(ax, s, letter="a", title="who contradicts the rules"):
    """Panel a -- marker-rule violation rate by cohort and label source."""
    srcs = ["expert", "depot_HQ", "pred_logreg", "pred_rf", "pred_hgb"]
    w = 0.16
    for i, src in enumerate(srcs):
        v = []
        for ds in COHORTS:
            r = s[(s.dataset == ds) & (s.source == src)]
            v.append(float(r["violation_rate"].iloc[0]) if len(r) else np.nan)
        ax.bar(np.arange(4) + (i - 2) * w, v, width=w, label=src.replace("pred_", ""),
               color=[CLR["expert"], CLR["depot_HQ"], "#8ecae6", "#219ebc", "#023047"][i])
    ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[d] for d in COHORTS], rotation=20)
    ax.set_ylabel("marker-rule violation rate")
    ax.set_ylim(0, 0.135)
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(False)
    panel(ax, letter, title)


def p6b(ax, sl, letter="b", title="keeps plausible cells"):
    """Panel b -- violations among accepted against deferred cells."""
    sl = sl[np.isclose(sl["alpha"], 0.10)]
    t = sl.pivot_table(index=["dataset", "clf"], columns="group", values="violation_rate")
    ds_order = [d for d in COHORTS]
    for i, ds in enumerate(ds_order):
        r = t.loc[ds] if ds in t.index.get_level_values(0) else None
        if r is None:
            continue
        ax.scatter(r["accepted"].values, r["deferred"].values, s=18, label=SHORT[ds])
    lim = max(0.001, float(np.nanmax(t.values)) * 1.15)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("violations among accepted")
    ax.set_ylabel("violations among deferred")
    ax.legend(fontsize=6)
    ax.grid(False)
    panel(ax, letter, title)


def p6c(ax, sp, letter="c", title="downstream spatial statistic"):
    """Panel c -- neighbourhood enrichment, predicted against expert labels."""
    for ds, mk in zip(COHORTS, "os^D"):
        e = sp[(sp.dataset == ds) & (sp.source == "expert")]
        p = sp[(sp.dataset == ds) & (sp.source == "pred_logreg")]
        if e.empty or p.empty:
            continue
        j = e.merge(p, on="type", suffixes=("_e", "_p"))
        ax.scatter(j["enrichment_e"], j["enrichment_p"], s=16, marker=mk, label=SHORT[ds])
    lim = [0.2, 6]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("enrichment, expert labels")
    ax.set_ylabel("enrichment, predicted labels")
    ax.legend(fontsize=6)
    ax.grid(False)
    panel(ax, letter, title)


def fig6():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    mk = load_markers()
    p6a(axes[0], mk["audit"])
    p6b(axes[1], mk["selective"])
    p6c(axes[2], mk["spatial"])
    fig.tight_layout()
    save(fig, "fig6_markers")


# ------------------------------------------------------- Extended Data Fig 1
def load_labelnoise():
    """Reference-label noise sweep at delta = 0.10, calibration-only regime."""
    p = os.path.join(RES, "r12_labelnoise.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[np.isclose(d["delta"], 0.10) & (d["regime"] == "cal_only")]
    return d


def _noise_axes(d):
    etas = sorted(d["eta"].unique())
    alphas = sorted(d["alpha"].unique())
    acol = {a: c for a, c in zip(alphas, ("#1f5fa9", "#e6a817", "#d1495b"))}
    return etas, alphas, acol


def pS1a(ax, d, letter="a", title="throughput cost"):
    """Panel a -- acceptance against reference-label noise, three budgets."""
    etas, alphas, acol = _noise_axes(d)
    for a in alphas:
        s = d[(d.method == "crc") & np.isclose(d["alpha"], a)]
        t = s.groupby("eta")["accept_rate"].mean().reindex(etas)
        ax.plot(t.index, t.values, "o-", ms=3, lw=1.1, color=acol[a], label="crc, alpha=%.2f" % a)
    for a in alphas:
        s = d[(d.method == "naive") & np.isclose(d["alpha"], a)]
        t = s.groupby("eta")["accept_rate"].mean().reindex(etas)
        ax.plot(t.index, t.values, "s--", ms=2.5, lw=0.9, color=acol[a], alpha=0.45)
    ax.set_xlabel("reference-label noise rate (eta)")
    ax.set_ylabel("acceptance")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=5.4, title="solid = certified", title_fontsize=5.4)
    panel(ax, letter, title)


def pS1b(ax, d, letter="b", title="validity survives"):
    """Panel b -- coverage is unchanged by reference-label noise."""
    etas = sorted(d["eta"].unique())
    for m, ls in (("crc", "o-"), ("naive", "s--")):
        s = d[(d.method == m) & np.isclose(d["alpha"], 0.10)]
        t = s.groupby("eta")["covered"].mean().reindex(etas)
        ax.plot(t.index, t.values, ls, ms=3, lw=1.1,
                color=CLR["crc"] if m == "crc" else CLR["naive"], label=m)
    ax.axhline(0.90, color="k", ls="--", lw=0.9)
    ax.set_xlabel("reference-label noise rate (eta)")
    ax.set_ylabel("coverage, alpha = 0.10")
    ax.set_ylim(0, 1.06)
    ax.legend(fontsize=6)
    panel(ax, letter, title)


def pS1c(ax, d, letter="c", title="when it stops answering"):
    """Panel c -- probability of a non-vacuous answer as noise grows."""
    etas, alphas, acol = _noise_axes(d)
    for a in alphas:
        s = d[(d.method == "crc") & np.isclose(d["alpha"], a)]
        t = 1.0 - s.groupby("eta")["empty"].mean().reindex(etas)
        ax.plot(t.index, t.values, "o-", ms=3, lw=1.1, color=acol[a],
                label="alpha=%.2f" % a)
    ax.set_xlabel("reference-label noise rate (eta)")
    ax.set_ylabel("P(non-vacuous answer)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5.6)
    panel(ax, letter, title)


def figS1():
    """Reference-label noise: the certificate survives, the throughput does not."""
    d = load_labelnoise()
    if d is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    pS1a(axes[0], d)
    pS1b(axes[1], d)
    pS1c(axes[2], d)
    fig.tight_layout()
    save(fig, "figS1_labelnoise")


def main():
    os.makedirs(FIG, exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    figS1()


if __name__ == "__main__":
    main()
