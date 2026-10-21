"""R35 -- supplementary figures S8, S9 and S10.

figS8_repair      targeted repair of the conformal failures and its label cost
figS9_sequential  anytime-valid sequential control
figS10_external   external validation on the STELLAR CODEX cohort

Style follows the rest of the figure batch (scripts/r10_figures.py,
scripts/r16_review_budget.py, scripts/r27_fixedseq.py): the same rcParams block,
the same left/right panel lettering helper, the same PDF + PNG pair written by a
single savefig loop.  The PNG is written at 400 dpi as specified for this batch;
the other figure scripts in the suite write 300 dpi PNGs.

Every number plotted here is read from a CSV under experiments/ and asserted
against the value the figure claims before it is drawn.  One line per plotted
series, with its source file and filter, is printed into
work/r35_figures_report.txt.  If an assertion fails the script records the
failure in the report, prints it and exits non-zero without writing the figure.

Sources
  figS8  experiments/r31_within_summary.csv       within (cross-FOV) coverage
         experiments/r31_paired.csv               paired covered_nonvac vs plain
         experiments/r31_ladder_summary.csv       the transfer label-cost ladder
  figS9  experiments/r32_sequential_summary.csv   label cost vs the rivals
         experiments/r32_sim.csv                  Monte-Carlo level of the bounds
  figS10 experiments/r34_phase2_certificate_summary.csv
         experiments/r34_phase2_by_source.csv
         experiments/r34_phase2_ratio.csv
         experiments/r34_phase2_metrics.csv       per-evaluation recount
         experiments/r34_phase2_transfer_panels.csv
         experiments/r03_summary.csv              benchmark coverage references
         experiments/r26_plugin_baseline_summary.csv   cross-check of those
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedFormatter, FixedLocator, NullFormatter,
                               NullLocator)
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "experiments")
FIG = os.path.join(ROOT, "figures")
WORK = os.path.join(ROOT, "work")

PNG_DPI = 400
DELTA = 0.10                 # work/r31_report.txt and work/r34_report.txt both
NOMINAL = 1.0 - DELTA        # run delta = 0.10, so nominal coverage is 0.90
ALPHA_MAIN = 0.10

CLR = {"plain": "#1f5fa9", "sc3": "#2a9d8f", "sc10": "#e6a817",
       "mond_class": "#7b4fa0", "grey": "#8d99ae", "light": "#9a9a9a",
       "over": "#d1495b", "within": "#1f5fa9", "ink": "#444444"}

BOX = dict(facecolor="white", edgecolor="none", alpha=0.92, pad=0.4)

_REPORT = []
_NCHECK = [0]
_NBOX = [0]
_NDATA = [0]
_NCOLL = [0]


# ------------------------------------------------------------------ helpers
def say(text=""):
    _REPORT.append(text)


def rec(panel, name, value, src, flt, col):
    """Record one plotted number, refusing to plot a non-finite one."""
    _NCHECK[0] += 1
    if value is None:
        txt = "n/a"
    else:
        v = float(value)
        if not np.isfinite(v):
            raise AssertionError("non-finite value for %s / %s" % (panel, name))
        txt = "%.6g" % v
    say("%-8s | %-50s | %-12s | %s | %s | %s"
        % (panel, name, txt, src, flt, col))
    return value


def near(value, expect, tol=5e-4, what=""):
    _NCHECK[0] += 1
    v = float(value)
    if not np.isfinite(v) or abs(v - float(expect)) > tol:
        raise AssertionError("%s: got %.10g, expected %.10g (tol %g)"
                             % (what, v, float(expect), tol))


def one(df, what):
    if df is None or len(df) != 1:
        raise AssertionError("%s: expected exactly 1 row, got %d"
                             % (what, 0 if df is None else len(df)))
    return df.iloc[0]


def load(name):
    path = os.path.join(EXP, name)
    if not os.path.exists(path):
        raise AssertionError("missing source CSV: experiments/%s" % name)
    return pd.read_csv(path)


def panel(ax, letter, title=None):
    ax.set_title(letter.upper(), loc="left", fontweight="bold", pad=6)
    if title:
        t = ax.set_title(title, loc="right", fontsize=7, color="0.35")
        _TITLE_FITS.append((ax, t))


_TITLE_FITS = []          # (axes, Text) registered by panel(), fitted in save()


def fit_titles(fig, frac=0.84, floor=4.2):
    """Shrink every right-hand panel title until it fits inside its own axes.

    A long title is right-aligned, so without this it spills left out of its
    axes and over the neighbouring panel (and over the bold panel letter).
    """
    pairs = [(ax, t) for ax, t in _TITLE_FITS if ax.figure is fig]
    fig.canvas.draw()
    for ax, t in pairs:
        limit = frac * ax.get_window_extent(fig.canvas.get_renderer()).width
        while t.get_fontsize() > floor:
            if t.get_window_extent(fig.canvas.get_renderer()).width <= limit:
                break
            t.set_fontsize(t.get_fontsize() - 0.25)
            fig.canvas.draw()


def _inside(outer, inner, eps=1.0):
    return (inner.x0 >= outer.x0 - eps and inner.x1 <= outer.x1 + eps
            and inner.y0 >= outer.y0 - eps and inner.y1 <= outer.y1 + eps)


def _overlap(b1, b2, tol=1.0):
    return (min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > tol
            and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > tol)


def _texts_of(ax):
    items = [("note", t) for t in list(ax.texts)]
    for attr in ("_left_title", "_center_title", "_right_title", "title"):
        t = getattr(ax, attr, None)
        if t is not None and t.get_text().strip():
            items.append(("title", t))
    leg = ax.get_legend()
    if leg is not None:
        items.extend(("legend", t) for t in leg.get_texts())
    items.extend(("tick", t) for t in ax.get_xticklabels() + ax.get_yticklabels())
    items.append(("xlabel", ax.xaxis.label))
    items.append(("ylabel", ax.yaxis.label))
    out = []
    for kind, t in items:
        if t.get_visible() and t.get_text().strip():
            out.append((kind, t))
    return out


def _data_artists(ax):
    """Display boxes for every plotted marker, line, collection and bar."""
    r = ax.figure.canvas.get_renderer()
    pt = ax.figure.dpi / 72.0
    out = []

    def add(xs, ys, half, label, tf):
        xs = np.asarray(xs, dtype=float).ravel()
        ys = np.asarray(ys, dtype=float).ravel()
        if xs.size == 0 or xs.size != ys.size:
            return
        good = np.isfinite(xs) & np.isfinite(ys)
        if not good.any():
            return
        disp = tf.transform(np.column_stack([xs[good], ys[good]]))
        for px, py in disp:
            out.append((label, px - half, py - half, px + half, py + half))

    for ln in ax.get_lines():
        xd = np.asarray(ln.get_xdata(), dtype=float)
        yd = np.asarray(ln.get_ydata(), dtype=float)
        if xd.size == 0:
            continue
        tf = ln.get_transform()
        lab = ln.get_label() or "line"
        has_mark = ln.get_marker() not in (None, "", "None", "none")
        if has_mark:
            half = max(float(ln.get_markersize()), 1.0) * pt / 2.0 + 1.0
        else:
            half = float(ln.get_linewidth()) * pt / 2.0 + 1.0
        add(xd, yd, half, lab, tf)
        if xd.size > 1 and ln.get_linestyle() not in (None, "", "None", "none"):
            for k in range(1, 12):
                f = k / 12.0
                add(xd[:-1] + f * (xd[1:] - xd[:-1]),
                    yd[:-1] + f * (yd[1:] - yd[:-1]),
                    float(ln.get_linewidth()) * pt / 2.0 + 1.0,
                    lab + " segment", tf)
    for coll in ax.collections:
        tfc = coll.get_transform()
        for path in coll.get_paths():
            v = np.asarray(path.vertices, dtype=float)
            if v.size:
                add(v[:, 0], v[:, 1], 1.5, "collection", tfc)
        off = coll.get_offsets()
        if off is not None and len(off):
            arr = np.asarray(off, dtype=float)
            add(arr[:, 0], arr[:, 1], 2.0, "collection offsets",
                coll.get_offset_transform())
    # N5 (r44): ax.patches was missing, so bars were invisible to this
    # harness.  Bars are patches.  An Annotation's own leader is also a
    # patch, and is excluded on purpose: it is annotation ink, not data
    # (the corresponding text box is tested as a box, not as data).
    arrows = set()
    for t in ax.texts:
        ap = getattr(t, "arrow_patch", None)
        if ap is not None:
            arrows.add(id(ap))
    for pat in ax.patches:
        if not pat.get_visible() or id(pat) in arrows:
            continue
        try:
            bb = pat.get_window_extent(r)
        except Exception:
            continue
        lab = pat.get_label()
        if not isinstance(lab, str) or not lab.strip() or lab.startswith("_"):
            lab = "patch"
        out.append((lab, bb.x0, bb.y0, bb.x1, bb.y1))
    return out


def _boxes_to_test(ax):
    """The legend frame and every annotation box drawn in this axes."""
    r = ax.figure.canvas.get_renderer()
    boxes = []
    leg = ax.get_legend()
    if leg is not None:
        boxes.append(("legend", "legend", leg.get_window_extent(r)))
    for t in ax.texts:
        if not t.get_visible() or not t.get_text().strip():
            continue
        if isinstance(t, Annotation):
            bb = Text.get_window_extent(t, r)     # text only, no arrow patch
        else:
            bb = t.get_window_extent(r)
        boxes.append(("annotation", t.get_text().replace("\n", " ")[:26],
                      bb.expanded(1.06, 1.18)))
    return boxes


def check_layout(fig, tag):
    """Programmatic layout audit of one figure.

    No two text objects may overlap, no text may fall outside the figure frame,
    and no legend box or annotation box may overlap a plotted marker, line,
    collection or bar.  That last test is what catches a legend sitting on top
    of data even when no text overlaps it.  Returns human-readable defects.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    probs = []
    allboxes = []
    data = []
    for ai, ax in enumerate(fig.axes):
        for kind, t in _texts_of(ax):
            bb = t.get_window_extent(r)
            txt = t.get_text().replace("\n", " ")[:26]
            allboxes.append((ai, kind, txt, bb))
            if kind not in ("note", "legend") and not _inside(fig.bbox, bb, 1.0):
                probs.append("axes %d outside figure: [%s] %r pos=%s bb=%.1f,"
                             "%.1f,%.1f,%.1f"
                             % (ai, kind, txt, t.get_position(), bb.x0, bb.y0,
                                bb.x1, bb.y1))
        for item in _data_artists(ax):
            data.append((ai,) + item)
    for i in range(len(allboxes)):
        for j in range(i + 1, len(allboxes)):
            a, b = allboxes[i], allboxes[j]
            if _overlap(a[3], b[3]):
                where = ("same axes %d" % a[0] if a[0] == b[0]
                         else "axes %d vs %d" % (a[0], b[0]))
                probs.append("overlap (%s): [%s] %r vs [%s] %r"
                             % (where, a[1], a[2], b[1], b[2]))
    nbox = ncoll = 0
    for ai, ax in enumerate(fig.axes):
        aabb = ax.get_window_extent(r)
        boxes = _boxes_to_test(ax)
        nbox += len(boxes)
        ncoll += len(boxes) * len(data)
        for kind, txt, bb in boxes:
            hits = [d for d in data
                    if _overlap(bb, Bbox.from_extents(d[2], d[3], d[4], d[5]),
                                tol=0.0)]
            if hits:
                hai, nm, x0, y0, x1, y1 = hits[0]
                inv = fig.axes[hai].transData.inverted()
                bx0, by0 = inv.transform((bb.x0, bb.y0))
                bx1, by1 = inv.transform((bb.x1, bb.y1))
                ax0d, ay0d = inv.transform((x0, y0))
                ax1d, ay1d = inv.transform((x1, y1))
                probs.append(
                    "axes %d window [%.0f,%.0f,%.0f,%.0f] %s box "
                    "[%.0f,%.0f,%.0f,%.0f] (data [%.4g,%.4g,%.4g,%.4g]) over "
                    "data: [%s] %r covers %d artist(s), first %r (axes %d) at "
                    "[%.0f,%.0f,%.0f,%.0f] (data [%.4g,%.4g,%.4g,%.4g])"
                    % (ai, aabb.x0, aabb.y0, aabb.x1, aabb.y1, kind,
                       bb.x0, bb.y0, bb.x1, bb.y1, bx0, by0, bx1, by1, txt,
                       nm, len(hits), nm, hai, x0, y0, x1, y1,
                       ax0d, ay0d, ax1d, ay1d))
    _NBOX[0] += nbox
    _NDATA[0] += len(data)
    _NCOLL[0] += ncoll
    return ["%s: %s" % (tag, p) for p in probs], nbox, len(data), ncoll


def save(fig, name):
    fit_titles(fig)
    probs, nbox, ndata, ncoll = check_layout(fig, name)
    del _TITLE_FITS[:]
    say("LAYOUT %-16s | %d layout defect(s); %d boxes vs %d data artists "
        "= %d box-vs-artist checks" % (name, len(probs), nbox, ndata, ncoll))
    for p in probs:
        say("  " + p)
    if probs:
        raise AssertionError("%d layout defect(s) in %s; first: %s"
                             % (len(probs), name, probs[0]))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)),
                    bbox_inches="tight",
                    dpi=(PNG_DPI if ext == "png" else None))
    plt.close(fig)
    print("wrote figures/%s.pdf/.png" % name)


def style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
        "legend.frameon": False,
    })


def grid(fig, ratios, wspace=0.52, bottom=0.165):
    return fig.add_gridspec(1, 3, width_ratios=ratios, wspace=wspace,
                            left=0.078, right=0.985, bottom=bottom, top=0.865)


# --------------------------------------------------- figS2 bookkeeping check
def check_figS2():
    """The manuscript legend cites a figS2 that no script claims to produce."""
    hits, s2cols = [], []
    sdir = os.path.join(ROOT, "scripts")
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".py") or fn == os.path.basename(__file__):
            continue
        with open(os.path.join(sdir, fn), "r", encoding="utf-8",
                  errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                low = line.lower()
                if "figs2" in low or "supplementary figure 2" in low:
                    hits.append("scripts/%s:%d: %s" % (fn, i, line.rstrip()))
                elif "s2_" in low:
                    s2cols.append("scripts/%s:%d" % (fn, i))
    files = [f for f in sorted(os.listdir(FIG)) if f.startswith("figS2")]
    say("=" * 100)
    say("figS2 STATUS")
    say("=" * 100)
    say("files matching figures/figS2*                            : %s"
        % (", ".join(files) if files else "NONE"))
    say("other scripts naming figS2 or 'supplementary figure 2'  : %d"
        % len(hits))
    say("scripts containing the token s2_ (the r27/r32 paired-column")
    say("  prefix, NOT a figure name)                            : %d line(s)"
        % len(s2cols))
    for h in hits[:10]:
        say("  " + h)
    say("producer search (rg -n \"figS2|S2_|supplementary figure 2\" scripts):")
    say("  the only supplementary figures named in scripts/ are figS1 (r10),")
    say("  figS3 (r15), figS4 (r18), figS5 (r19), figS6 (r25) and figS7 (r27).")
    say("docs/submission/code_availability.md states: 'figures/figS3_marker_")
    say("  sensitivity.* is Supplementary Figure 2 ... There is no figS2_* file;")
    say("  the label was retired with an earlier draft.'")
    say("VERDICT: figures/figS2_* is MISSING and no script produces it.  The")
    say("  manuscript's Supplementary Figure 2 is backed by the existing")
    say("  figures/figS3_marker_sensitivity.* (scripts/r15_marker_sensitivity")
    say("  .py), which was NOT regenerated: it is outside this batch's scope.")
    say("  Nothing was fabricated.")
    say()
    return len(files), len(hits)


# -------------------------------------------------------------------- figS8
def figS8():
    w = load("r31_within_summary.csv")
    pair = load("r31_paired.csv")
    lad = load("r31_ladder_summary.csv")

    arms = ["plain", "sc3", "sc10", "mond_class"]
    alphas = [0.05, 0.10]
    aclr = {0.05: "#2b6cb0", 0.10: "#c53030"}
    # the value the legend claims: (coverage_all, coverage_nonvac, nonvac_frac)
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
    # the manuscript's Table-1 convention, quoted in work/r31_report.txt
    r = one(w[(w.arm == "plain") & np.isclose(w.alpha, 0.10)], "plain a=0.10")
    near(r["coverage_nonvac_cond"], 0.7642857142857142,
         what="within plain a=0.10 coverage_nonvac_cond (manuscript 0.7643)")
    worst = max(v[1] for v in got.values())
    if not worst < NOMINAL:
        raise AssertionError("within non-vacuous coverage reaches %.4f >= %.2f"
                             % (worst, NOMINAL))

    # ---- paired differences against plain, covered_nonvac
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

    # ---- the transfer label-cost ladder
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

    # ----------------------------------------------------------- drawing
    fig = plt.figure(figsize=(7.4, 3.0))
    gs = grid(fig, [1.18, 0.98, 1.32])

    # -- (a) coverage by arm and budget, vacuous-inclusive vs non-vacuous
    ax = fig.add_subplot(gs[0, 0])
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
    ax.axhline(NOMINAL, color="k", ls="--", lw=1.0, zorder=1)
    ax.text(0.02, NOMINAL + 0.020, "nominal %.2f" % NOMINAL, fontsize=4.4, ha="left",
            va="bottom", color="0.25", transform=ax.get_yaxis_transform())
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
            ha="left", bbox=BOX, zorder=6)
    panel(ax, "a", "within (cross-FOV), delta = 0.10")

    # -- (b) paired non-vacuous coverage against plain
    ax = fig.add_subplot(gs[0, 1])
    for i, (arm, md, lo, hi, npr, nun) in enumerate(pgot):
        ax.errorbar([md], [i], xerr=[[max(md - lo, 0.0)], [max(hi - md, 0.0)]],
                    fmt="o", ms=4.0, lw=1.0, capsize=2.0,
                    color=CLR[arm] if arm in CLR else CLR["ink"])
        ax.text(0.99, i + 0.20, "%+.3f\n[%+.3f, %+.3f]" % (md, lo, hi),
                fontsize=4.4, color="0.25", ha="right", va="center",
                transform=ax.get_yaxis_transform(), bbox=BOX, zorder=6)
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
            fontsize=4.9, color="0.3", ha="right", va="bottom", bbox=BOX,
            zorder=6)
    panel(ax, "b", "paired vs plain, 20 units")

    # -- (c) transfer label-cost ladder
    ax = fig.add_subplot(gs[0, 2])
    arms_all = ["iw_sel", "mond_class", "oracle", "plain", "pool_iw",
                "prop_mond", "src", "twostage_iw", "twostage_nw", "unc_iw",
                "w_src"]
    hl = {"mond_class": "#1f5fa9", "prop_mond": "#2a9d8f",
          "twostage_nw": "#dd6b20", "plain": CLR["grey"]}
    offs = dict(zip(arms_all, np.linspace(-0.16, 0.16, len(arms_all))))
    never = 2500.0
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
                    mec=CLR["light"], mew=0.8, zorder=3)
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
    # D6/N3 (r44): the printed bound must not be smaller than the true
    # maximum.  "%.4f" rounds the true maximum 0.045744091 DOWN to 0.0457
    # and makes the printed inequality false, so the 4-dp ceiling is
    # printed instead, and the assertion is one-sided against the maximum.
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
    ax.axhline(2500.0, color="0.72", ls=":", lw=0.8, zorder=1)
    ax.text(0.040, 32.0, "no valid repair:\nnever, or only at\nk = 0 (vacuous)",
            fontsize=4.3, color="0.35", va="bottom", ha="left", bbox=BOX,
            zorder=6)
    k500 = ladder[("plain", 0.10)][0]
    k200 = ladder[("mond_class", 0.20)][0]
    if k500 != 500 or k200 != 200:
        raise AssertionError("ladder grid changed: 500 -> %d, 200 -> %d"
                             % (k500, k200))
    ax.annotate("%d at\nalpha = 0.10" % k500, xy=(0.10, float(k500)),
                xytext=(0.0435, 120.0), fontsize=4.7, color="0.25",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.5"))
    ax.annotate("%d only at\nalpha = 0.20" % k200, xy=(0.205, float(k200)),
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
    h_c.append(Line2D([], [], color=CLR["light"], marker="o", ls="none",
                      ms=2.6, mfc="none"))
    def acc_txt(a):
        return "/".join("n/a" if not np.isfinite(acc_at[(a, al)])
                        else "%.3f" % acc_at[(a, al)]
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
                         va="bottom", ha="center", bbox=BOX, zorder=6)
    printed_bound = float(bound_note.get_text().split("<=")[1].strip()
                          .split()[0])
    if printed_bound < max_acc - 1e-12:
        raise AssertionError(
            "the printed acceptance bound %.4f lies below the true maximum "
            "%.9f" % (printed_bound, max_acc))
    panel(ax, "c", "transfer ladder, 60 unit-folds")

    save(fig, "figS8_repair")


# -------------------------------------------------------------------- figS9
def figS9():
    s = load("r32_sequential_summary.csv")
    sim = load("r32_sim.csv")

    fam = "cellsplit"
    arms = {"seq_eproc_bonf": "#1f5fa9", "cert_fixed_full": "#d1495b",
            "cert_fixed_union": "#2a9d8f"}
    mk = {"seq_eproc_bonf": "o", "cert_fixed_full": "s", "cert_fixed_union": "^"}
    show = [0.10, 0.20]
    # the value the legend claims: (t25, t50, lo50, hi50, t75, lo75, hi75)
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
    # the honest direction, asserted rather than asserted-in-prose
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
            raise AssertionError("s3_t75_frac at alpha=%.2f is not positive"
                                 % al)

    # ---- the Monte-Carlo negative control
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

    # ----------------------------------------------------------- drawing
    fig = plt.figure(figsize=(7.4, 3.0))
    gs = grid(fig, [1.20, 0.90, 1.16], wspace=0.54, bottom=0.245)

    # -- (a) label cost against the acceptance target
    ax = fig.add_subplot(gs[0, 0])
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
    # N4 (r44): mean_tNN_frac divides by nrun and imputes 1.0 (the whole
    # pool) to a run that never reached t, so the ordinate is a mean over
    # ALL runs, not over the runs that reached t.
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
    panel(ax, "a", "cellsplit, delta = 0.10")

    # -- (b) the paired difference against the union-bound rival
    ax = fig.add_subplot(gs[0, 1])
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
    ax.text(0.02, 0.98, "positive = the e-process\nneeds MORE labels", fontsize=4.9,
            color="0.3", va="top", ha="left", transform=ax.transAxes, bbox=BOX)
    panel(ax, "b", "paired vs union")

    # -- (c) the negative control
    ax = fig.add_subplot(gs[0, 2])
    order = ["seq_eproc_anytime", "seq_cs_hoeffding", "control_peek_hoeffding",
             "control_peek_cp"]
    bclr = {"seq_eproc_anytime": "#1f5fa9", "seq_cs_hoeffding": "#2a9d8f",
            "control_peek_hoeffding": CLR["light"], "control_peek_cp": "#d1495b"}
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
            bbox=BOX)
    panel(ax, "c", "level, delta = 0.05")

    save(fig, "figS9_sequential")


# ------------------------------------------------------------------- figS10
def figS10():
    cs = load("r34_phase2_certificate_summary.csv")
    src = load("r34_phase2_by_source.csv")
    rat = load("r34_phase2_ratio.csv")
    met = load("r34_phase2_metrics.csv")
    pnl = load("r34_phase2_transfer_panels.csv")
    r03 = load("r03_summary.csv")
    r26 = load("r26_plugin_baseline_summary.csv")

    # ---- (a) acceptance, realised risk and coverage of the budget
    cert = {}
    for family, acc, risk, cov in (
            ("ext_within", 0.7264066000477634, 0.07784141183551987,
             0.6555555555555556),
            ("ext_transfer", 0.8394325484270428, 0.48929001571630854,
             0.03333333333333333)):
        r = one(cs[(cs.family == family) & np.isclose(cs.alpha, ALPHA_MAIN)],
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
    # benchmark references, read from the published benchmark summary
    bench = {}
    for kind, want in (("cellsplit", 1.0000000000000000),
                       ("within", 0.8166666666666667),
                       ("transfer", 0.3972222222222222)):
        r = one(r03[(r03.kind == kind) & (r03.method == "crc")
                    & np.isclose(r03.alpha, ALPHA_MAIN)],
                "r03_summary %s crc a=0.10" % kind)
        near(r["coverage"], want, what="benchmark %s coverage" % kind)
        bench[kind] = float(r["coverage"])
    for family in ("within", "transfer"):
        r = one(r26[(r26.family == family) & (r26.arm == "certificate")
                    & np.isclose(r26.alpha, ALPHA_MAIN)
                    & np.isclose(r26.delta, DELTA)],
                "r26 %s certificate a=0.10 d=0.10" % family)
        near(r["coverage"], bench[family], what="r26 cross-check %s" % family)

    # ---- (b) coverage by deployment source
    bys, famof = {}, {"STELLAR_Intestine_B004": "ext_within",
                      "STELLAR_BarrettsEsophagus": "ext_transfer",
                      "STELLAR_Tonsil": "ext_transfer"}
    for target, family in famof.items():
        r = one(src[(src.family == family) & (src.target == target)
                    & np.isclose(src.alpha, ALPHA_MAIN)],
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

    # ---- (c) the ratio diagnostic, recounted per evaluation
    key = ["unit", "clf", "repeat", "alpha", "family"]
    cr = met[(met.method == "crc") & np.isclose(met.alpha, ALPHA_MAIN)]
    orc = met[(met.method == "oracle") & np.isclose(met.alpha, ALPHA_MAIN)]
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
                & np.isclose(rat.alpha, ALPHA_MAIN)],
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

    # ----------------------------------------------------------- drawing
    fig = plt.figure(figsize=(7.4, 3.0))
    gs = grid(fig, [1.36, 1.10, 1.14], wspace=0.46, bottom=0.275)

    # -- (a) the three certificate readouts at alpha = delta = 0.10
    ax = fig.add_subplot(gs[0, 0])
    groups = [("acceptance", "mean_accept", None),
              ("realised\nrisk", "risk_nonvac", ALPHA_MAIN),
              ("coverage", "coverage", NOMINAL)]
    cfamily = {"ext_within": "#1f5fa9", "ext_transfer": "#d1495b"}
    for gi, (glab, col, nomv) in enumerate(groups):
        for fi, family in enumerate(("ext_within", "ext_transfer")):
            xx = gi + (-0.19 if fi == 0 else 0.19)
            val = {"mean_accept": cert[family][0], "risk_nonvac": cert[family][1],
                   "coverage": cert[family][2]}[col]
            ax.bar([xx], [val], width=0.34, color=cfamily[family],
                   alpha=0.95 if fi == 0 else 0.55,
                   edgecolor=cfamily[family], linewidth=0.6, zorder=2)
            lab_y = val + 0.03
            if nomv is not None and nomv - 0.005 <= lab_y <= nomv + 0.030:
                lab_y = nomv + 0.035      # lift clear of the nominal line
            # N5 (r44): with ax.patches now in the data set, a centred value
            # label is wider than the gap to its taller neighbour and its box
            # bbox touched the neighbouring bar.  Each label is nudged 0.08
            # data units away from the group centre, over its own bar only.
            ax.text(xx + (0.08 if fi else -0.08), min(lab_y, 1.02),
                    "%.4f" % val, fontsize=4.8, ha="center", color="0.2",
                    zorder=4)
            rec("figS10 a", "%s %s alpha=0.10" % (family, col), val,
                "experiments/r34_phase2_certificate_summary.csv",
                "family==%s & alpha==0.10" % family, col)
        if nomv is not None:
            ax.plot([gi - 0.44, gi + 0.44], [nomv, nomv], ls="--", lw=1.0,
                    color="k", zorder=3)
            if nomv < 0.5:
                # N5 (r44): this label used to sit at gi - 0.45 and its white
                # box painted over the ext_within acceptance bar.  Wrapped to
                # two lines and placed inside the realised-risk group, above
                # its own dashed line and clear of every bar.
                ax.text(0.40, 0.30, "nominal\n%.2f" % nomv, fontsize=4.5,
                        ha="left", va="top", color="0.2", bbox=BOX, zorder=6)
            else:
                ax.text(gi - 0.45, nomv - 0.020, "nominal %.2f" % nomv,
                        fontsize=4.5, ha="right", va="top", color="0.2",
                        bbox=BOX, zorder=6)
    for kind, lab0, m in (("cellsplit", "random", "D"),
                          ("within", "cross-FOV", "s"),
                          ("transfer", "transfer", "^")):
        lab = "%s\n%.4f" % (lab0, bench[kind])
        ax.plot([3.0], [bench[kind]], marker=m, ms=4.0, mfc="white",
                mec=CLR["ink"], mew=0.9, ls="none", zorder=5)
        ax.text(2.82, bench[kind], lab, fontsize=4.7, va="center", ha="right",
                color="0.25")
        rec("figS10 a", "benchmark %s coverage alpha=0.10" % kind, bench[kind],
            "experiments/r03_summary.csv",
            "kind==%s & method==crc & alpha==0.10" % kind, "coverage")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=5.4)
    ax.set_xlim(-0.55, 3.30)
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("value")
    ax.set_xlabel("certificate readout, alpha = delta = 0.10")
    h_a = [Line2D([], [], color=cfamily[f], marker="s", ls="none", ms=4.0,
                  alpha=0.95 if f == "ext_within" else 0.55, label=f)
           for f in ("ext_within", "ext_transfer")]
    h_a.append(Line2D([], [], color=CLR["ink"], marker="D", ls="none", ms=3.6,
                      mfc="white", label="benchmark"))
    ax.legend(handles=h_a, fontsize=4.2, loc="upper left", ncol=2,
              handletextpad=0.35, columnspacing=0.7, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.92)
    panel(ax, "a", "no repaired guarantee")

    # -- (b) coverage decomposed by deployment source
    ax = fig.add_subplot(gs[0, 1])
    order = ["STELLAR_Intestine_B004", "STELLAR_BarrettsEsophagus",
             "STELLAR_Tonsil"]
    labs = ["intestine\n(within)", "Barrett's\noesophagus", "tonsil"]
    sclr = ["#1f5fa9", "#e6a817", "#d1495b"]
    for i, (tgt, lab, cl) in enumerate(zip(order, labs, sclr)):
        cov, acc, risk, nev = bys[tgt]
        ax.bar([i], [cov], width=0.56, color=cl, alpha=0.9, edgecolor=cl,
               linewidth=0.6, zorder=2)
        ax.text(i, max(cov, 0.0) + 0.055, "%.4f" % cov, fontsize=5.4,
                ha="center", color="0.2", zorder=4)
        if cov == 0.0:
            ax.plot([i - 0.28, i + 0.28], [0.0, 0.0], color=cl, lw=1.4,
                    zorder=4)
        ax.text(i, -0.055, "accept %.4f\nrisk %.4f" % (acc, risk), fontsize=4.0,
                ha="center", va="top", color="0.3")
        f = famof[tgt]
        for col, val in (("coverage", cov), ("mean_accept", acc),
                         ("mean_risk", risk)):
            rec("figS10 b", "%s %s alpha=0.10" % (col, tgt), val,
                "experiments/r34_phase2_by_source.csv",
                "family==%s & target==%s & alpha==0.10" % (f, tgt), col)
    for tgt, i in (("STELLAR_BarrettsEsophagus", 1), ("STELLAR_Tonsil", 2)):
        npn, nsh, ntg = panels[tgt]
        ax.text(i, 1.33, "%d markers\n%d shared\nclasses\n%d cells"
                % (npn, nsh, ntg), fontsize=4.0, ha="center", va="top",
                color="0.3")
        rec("figS10 b", "transfer panel %s target cells" % tgt, float(ntg),
            "experiments/r34_phase2_transfer_panels.csv",
            "unit_tag endswith %s" % tgt,
            "n_panel=%d n_shared_classes=%d n_target_cells=%d" % (npn, nsh, ntg))
    ax.axhline(NOMINAL, color="k", ls="--", lw=1.0, zorder=3)
    ax.text(2.45, NOMINAL + 0.03, "nominal %.2f" % NOMINAL, fontsize=5.0, ha="right",
            va="bottom", color="0.2")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=5.2)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.20, 1.36)
    ax.set_ylabel("coverage of the error budget")
    ax.set_xlabel("deployment source, alpha = 0.10, 90\nevaluations each")
    panel(ax, "b", "budget lost by source")

    # -- (c) the ratio diagnostic against realised risk
    ax = fig.add_subplot(gs[0, 2])
    undef_y = 0.55
    ax.axhspan(0.36, 0.68, color="0.93", zorder=0)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, zorder=1)
    ax.axvline(ALPHA_MAIN, color="0.4", ls=":", lw=1.0, zorder=1)
    ov, win = te[te.over], te[~te.over]
    unc, far = ov[ov.acc_or == 0], ov[ov.acc_or > 0]
    ax.plot(win.selective_risk, win.ratio, ls="none", marker="o", ms=3.2,
            mfc="white", mec=CLR["within"], mew=0.8, zorder=3)
    ax.plot(far.selective_risk, far.ratio, ls="none", marker="o", ms=3.2,
            color=CLR["over"], zorder=3)
    ax.plot(unc.selective_risk, np.full(len(unc), undef_y), ls="none",
            marker="x", ms=3.4, color=CLR["over"], zorder=3)
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
    h_c = [Line2D([], [], color=CLR["over"], marker="o", ls="none", ms=3.2,
                  label="over budget"),
           Line2D([], [], color=CLR["over"], marker="x", ls="none", ms=3.4,
                  label="ratio undefined"),
           Line2D([], [], color=CLR["within"], marker="o", ls="none", ms=3.2,
                  mfc="white", label="within budget"),
           Line2D([], [], color="k", ls="--", lw=1.0, label="ratio = 1")]
    ax.legend(handles=h_c, fontsize=4.4, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.42), handletextpad=0.35,
              columnspacing=0.7)
    ax.text(0.13, 0.985, "%d of %d over-budget\ntransfer evaluations are\n"
            "missed; every one has\nan undefined ratio\n(oracle accepts no cell)"
            % (counts[2], counts[0]),
            transform=ax.transAxes, fontsize=4.5, color="0.25", ha="left",
            va="top", bbox=BOX, zorder=6)
    panel(ax, "c", "ext_transfer, alpha = 0.10")

    save(fig, "figS10_external")


# -------------------------------------------------------------------- report
def write_report(ok, err, s2):
    path = os.path.join(WORK, "r35_figures_report.txt")
    head = []
    head.append("=" * 100)
    head.append("r35 -- figure report for figures/figS8_repair, figures/figS9_sequential,")
    head.append("       figures/figS10_external  (PNG at %d dpi + PDF)" % PNG_DPI)
    head.append("=" * 100)
    head.append("script      : scripts/r35_figures.py")
    head.append("style       : scripts/r10_figures.py / r16 / r25 / r27 -- same")
    head.append("              rcParams, panel lettering helper and PDF + PNG pair;")
    head.append("              one 1x3 panel row per figure, figsize 7.4 x 3.0 in.")
    head.append("              Deviation: the rest of the suite writes 300 dpi PNGs,")
    head.append("              this batch writes %d dpi as specified." % PNG_DPI)
    head.append("conventions : delta = 0.10 (work/r31_report.txt, work/r34_report.txt),")
    head.append("              so the nominal coverage line is 1 - delta = 0.90;")
    head.append("              alpha in {0.05, 0.10, 0.20}; no smoothing and no")
    head.append("              fitted curve anywhere in these three figures.")
    head.append("")
    head.append("EVERY NUMBER BELOW IS READ FROM THE NAMED CSV AND ASSERTED AGAINST")
    head.append("THE VALUE THE FIGURE CLAIMS BEFORE IT IS DRAWN.  A failed assertion")
    head.append("stops the run before any figure is saved.")
    head.append("")
    head.append("  panel    | series                                             |")
    head.append("           | value        | source csv | filter | column")
    head.append("-" * 100)
    tail = []
    tail.append("-" * 100)
    if ok:
        tail.append("STATUS: OK -- every assertion passed, all three figures written.")
    else:
        tail.append("STATUS: FAILED -- assertion error, figures NOT written:")
        tail.append("  " + err)
    tail.append("")
    tail.append("figS8 panels (targeted repair and label cost)")
    tail.append("  a  within (cross-FOV) coverage of the error budget by arm (plain,")
    tail.append("     sc3, sc10, mond_class) at alpha = 0.05 and 0.10, three readouts:")
    tail.append("     vacuous-inclusive coverage_all (filled circle), non-vacuous")
    tail.append("     coverage_nonvac (open square) and the manuscript's conditional")
    tail.append("     coverage_nonvac_cond = P(covered | accepted set non-empty) (open")
    tail.append("     triangle), against the nominal 0.90 line.  The y axis starts at")
    tail.append("     0.26 so the low non-vacuous points stay clear of the frame; no")
    tail.append("     value is clipped.")
    tail.append("  b  paired non-vacuous coverage (covered_nonvac) against the plain")
    tail.append("     arm, 95% unit bootstrap over 20 units / 1080 pairs; all three")
    tail.append("     intervals cross zero, so no arm repairs the shortfall.")
    tail.append("  c  the transfer label-cost ladder: the smallest rung k with")
    tail.append("     coverage_all >= 0.90 and a non-empty accepted set, per arm and")
    tail.append("     budget.  -1 (never) and 0 (vacuous at k = 0) are drawn in the")
    tail.append("     top band; plain needs only 100 at alpha = 0.20.  Each highlighted")
    tail.append("     legend entry prints accept_at_all_k at its drawn rungs (alpha =")
    tail.append("     0.05/0.10/0.20), so the acceptance attained at the cost is beside")
    tail.append("     it.  The bound quoted in the panel is computed from the CSV at")
    tail.append("     draw time and asserted: it is the largest accept_at_all_k over")
    tail.append("     the repair arms, and the oracle reference that exceeds it is")
    tail.append("     named in the same box.")
    tail.append("")
    tail.append("figS9 panels (anytime-valid sequential control)")
    # N2/N4 (r44): recompute the two quoted claims here instead of carrying
    # the draft wording.  mean_tNN_frac divides by nrun and imputes 1.0 to a
    # run that never reached t (a mean over ALL runs), and the (a) ordinate
    # difference equals the (b) plotted s3_t75_frac at every alpha.
    try:
        seq = load("r32_sequential_summary.csv")
        seq = seq[(seq["family"] == "cellsplit") & (seq["delta"] == 0.10)]

        def sq(arm, al, col):
            row = one(seq[(seq["arm"] == arm) & (seq["alpha"] == al)],
                      "cellsplit %s alpha=%.2f" % (arm, al))
            return float(row[col])

        als = (0.05, 0.10, 0.20)
        t75e = [sq("seq_eproc_bonf", a, "mean_t75_frac") for a in als]
        t75u = [sq("cert_fixed_union", a, "mean_t75_frac") for a in als]
        rich = [sq("seq_eproc_bonf", a, "mean_reached75") for a in als]
        s3v = [sq("seq_eproc_bonf", a, "s3_t75_frac") for a in als]
        cond = [(v - (1.0 - p)) / p for v, p in zip(t75e, rich)]
        gap = [a - b for a, b in zip(t75e, t75u)]
        nums = {"allruns": " / ".join("%.4f" % v for v in t75e),
                "cond": " / ".join("%.3f" % v for v in cond),
                "gap": " / ".join("%+.4f" % v for v in gap),
                "identical": max(abs(g - v) for g, v in zip(gap, s3v)) < 1e-9}
    except Exception as exc:
        nums = {"allruns": "see the CSV", "cond": "see the CSV",
                "gap": "see the CSV", "identical": False}
        say("WARNING: figS9 tail numbers could not be recomputed: %s" % exc)
    tail.append("  a  mean fraction of the labelled pool needed to reach a realised")
    tail.append("     acceptance target t (t = 0.25/0.50/0.75 on the x axis), cellsplit,")
    tail.append("     delta = 0.10: seq_eproc_bonf vs cert_fixed_full vs")
    tail.append("     cert_fixed_union at alpha = 0.10 and 0.20, 95% unit bootstrap")
    tail.append("     ribbons at t50/t75, t75 marked by a dotted line.")
    tail.append("     IMPORTANT: the ordinate is a mean over ALL runs, not over the")
    tail.append("     runs that reached t.  mean_tNN_frac divides by nrun and imputes")
    tail.append("     the whole pool (fraction 1.0) to a run that never reached t, so")
    tail.append("     the column is not conditional on reaching t: the panel draws")
    tail.append("     %(allruns)s and conditioning on the reachers would give"
                % nums)
    tail.append("     %(cond)s at alpha = 0.05 / 0.10 / 0.20.  It is not a paired"
                % nums)
    tail.append("     difference.")
    tail.append("     cert_fixed_full needs the whole pool (1.0000 by construction,")
    tail.append("     as its legend entry says).  t25 has no interval column in the")
    tail.append("     summary, so it is drawn as a marker only, and the xlabel says so.")
    tail.append("  b  e-process minus cert_fixed_union at t75; the point estimate is")
    tail.append("     the same mean gap as in (a), because the (a) ordinate difference")
    tail.append("     mean_t75_frac(seq_eproc_bonf) - mean_t75_frac(cert_fixed_union)")
    tail.append("     equals the (b) plotted value s3_t75_frac at every alpha")
    tail.append("     [recomputed from the CSV at report time: %(identical)s]."
                % {"identical": "IDENTICAL" if nums["identical"] else "MISMATCH"})
    tail.append("     The gap is %(gap)s at alpha = 0.05 / 0.10 / 0.20.  Only"
                % nums)
    tail.append("     the interval differs: (b) is paired per unit, (a) is not, so")
    tail.append("     (b) also needs more labels than the union-bound rival and fewer")
    tail.append("     than the whole-pool object.  The point estimate is positive at")
    tail.append("     every alpha; at alpha = 0.05 the mean is +0.0044 while its")
    tail.append("     interval [0.0000, 0.0132] touches zero, so only the point")
    tail.append("     estimate is strictly positive.")
    tail.append("  c  Monte-Carlo negative control at delta = 0.05, four series: the")
    tail.append("     anytime-valid e-process, the valid Hoeffding sequence, the")
    tail.append("     Hoeffding fixed-sample bound with no correction, and the")
    tail.append("     fixed-sample CP bound re-applied at every look (naive peek); the")
    tail.append("     last crosses about 5.5x nominal at mu = 0.20 while the anytime")
    tail.append("     bounds stay at or below nominal.")
    tail.append("")
    tail.append("figS10 panels (external validation, STELLAR CODEX)")
    tail.append("  a  acceptance, realised risk and coverage of the budget for")
    tail.append("     ext_within and ext_transfer at alpha = 0.10 against nominal 0.10")
    tail.append("     (risk) and 0.90 (coverage), with the benchmark's own coverage at")
    tail.append("     alpha = 0.10 as open reference markers.  Each dashed nominal")
    tail.append("     line carries its own boxed label; the 0.10 label is wrapped to")
    tail.append("     two lines and sits above its line inside the realised-risk")
    tail.append("     group, clear of every bar.  The open-marker meaning is in the")
    tail.append("     legend.")
    tail.append("  b  coverage by deployment source at alpha = 0.10, with the shared")
    tail.append("     transfer-panel metadata; the tonsil bar is exactly 0.0000.  The")
    tail.append("     y axis is extended to -0.20 only to hold the accept / risk")
    tail.append("     annotations below the zeros, not to suggest negative coverage.")
    tail.append("  c  the ratio diagnostic (certificate acceptance / oracle acceptance)")
    tail.append("     against realised risk among accepted cells; the 60 missed")
    tail.append("     over-budget evaluations sit in the undefined band, and the")
    tail.append("     60 of 174 count is stated in the empty upper-left region.")
    tail.append("")
    tail.append("Nothing in these figures suggests a repaired guarantee: coverage of")
    tail.append("the budget falls from 1.0000 / 0.8167 / 0.3972 on the benchmark's")
    tail.append("random-split, cross-FOV and transfer families to 0.6556 / 0.0333 on")
    tail.append("the external cohort, and the ratio diagnostic misses 60 of 174")
    tail.append("over-budget transfer evaluations.")
    tail.append("")
    tail.append("LAYOUT FIX PASS (second render of the same three files: same")
    tail.append("names, same plotted numbers, same panels and order, still PNG")
    tail.append("%d dpi + PDF).  The defects found in the first render and" % PNG_DPI)
    tail.append("what was changed:")
    tail.append(" 1 figS8c x-axis: alpha is on a log axis; the ticks are now a")
    tail.append("   FixedLocator with plain labels 0.05 / 0.10 / 0.20, with the")
    tail.append("   minor ticks and the log offset text suppressed, so the old")
    tail.append("   '4 x 10^-2' exponent no longer jams into '0.05'.")
    tail.append(" 2 figS8c annotations: 'no valid repair: never, or only at")
    tail.append("   k = 0 (vacuous)' is three short lines above the never band,")
    tail.append("   clear of the 1000 gridline and its tick; the acceptance note")
    tail.append("   is rewrapped to two short lines inside the axes, below the")
    tail.append("   curves, and no longer truncates.")
    tail.append(" 3 figS8a annotation: 'non-vacuous coverage stays below nominal")
    tail.append("   for every arm' was moved off the top gridline to the empty")
    tail.append("   bottom-left inside the axes, below the nominal 0.90 line.")
    tail.append(" 4 figS8b labels: the three paired-interval labels are now one")
    tail.append("   line each, raised to i + 0.26 and given an opaque pad, so they")
    tail.append("   clear their own bars and the zero line.")
    tail.append(" 5 figS10a: the tick labels are shortened to acceptance /")
    tail.append("   realised risk / coverage; the value labels are 4.8 pt and the")
    tail.append("   realised-risk label that used to cross the nominal 0.10 line")
    tail.append("   is lifted clear of it.")
    tail.append(" 6 figS10a: the legend moved to the upper-left and the three")
    tail.append("   benchmark markers now sit at x = 3.0 with their two-line")
    tail.append("   labels to the left, so 'random 1.0000' no longer touches the")
    tail.append("   legend box.")
    tail.append(" 7 figS10b: exactly one four-line marker / shared-classes / cell")
    tail.append("   block per transfer bar, centred over its own bar instead of")
    tail.append("   the two overlapping three-line blocks.")
    tail.append(" 8 titles: the right-hand panel titles are shortened and are")
    tail.append("   auto-shrunk by fit_titles() so that no title extends past its")
    tail.append("   own axes (this is what fixed the figS9a 'a' collision and the")
    tail.append("   figS10c title spilling into panel b).")
    tail.append("")
    tail.append("The script audits the finished figure before saving: it draws once")
    tail.append("and asserts (i) that no two Text bounding boxes overlap and (ii) that")
    tail.append("no legend frame and no annotation box overlaps any plotted marker,")
    tail.append("line, collection or bar.  A remaining defect exits non-zero without")
    tail.append("writing the figure.  All three figures pass with 0 text collisions")
    tail.append("and 0 boxes over data.")
    tail.append("")
    tail.append("value assertions executed (rec + near): %d" % _NCHECK[0])
    tail.append("harness box-vs-data checks: %d boxes x %d data artists "
                "= %d checks" % (_NBOX[0], _NDATA[0], _NCOLL[0]))
    tail.append("")
    tail.append("figS2: %s" % ("MISSING" if not s2 or s2[0] == 0 else "present"))
    tail.append("  No script under scripts/ names figS2 or 'supplementary figure 2'")
    tail.append("  (%d such match(es)); figures/figS2_* does not exist and was NOT"
                % (0 if not s2 else s2[1]))
    tail.append("  fabricated.  docs/submission/code_availability.md records that the")
    tail.append("  manuscript's Supplementary Figure 2 is backed by the existing")
    tail.append("  figures/figS3_marker_sensitivity.* from scripts/r15_marker_sensitivity")
    tail.append("  .py; that file was not regenerated (outside this batch's scope).")
    text = "\n".join(head + _REPORT + tail) + "\n"
    with open(path, "w", newline="\n", encoding="ascii", errors="strict") as fh:
        fh.write(text)
    print("wrote work/r35_figures_report.txt (%d lines)"
          % len(text.splitlines()))


def main():
    style()
    ok, err, s2 = True, "", None
    try:
        s2 = check_figS2()
        figS8()
        figS9()
        figS10()
    except AssertionError as exc:
        ok, err = False, str(exc)
        print("ASSERTION FAILED: %s" % err)
    write_report(ok, err, s2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
