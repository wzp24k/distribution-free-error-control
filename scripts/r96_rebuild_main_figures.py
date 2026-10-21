# -*- coding: utf-8 -*-
"""Rebuild the seven main figures from their result tables.

The script writes one complete, assembled figure per manuscript display item:

    figures/Figure 1.svg/.png   benchmark composition and audit
    figures/Figure 2.svg/.png   calibration under transfer
    figures/Figure 3.svg/.png   coverage and validity
    figures/Figure 4.svg/.png   acceptance and the exchangeability diagnostic
    figures/Figure 5.svg/.png   the certificate as a review budget
    figures/Figure 6.svg/.png   external validation
    figures/Figure 7.svg/.png   deployment decision procedure

No panel is produced by cropping a combined image.  Every text object is set to
black before rendering, and ``svg.fonttype`` is ``none`` so all labels remain
editable SVG text.
"""

import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "experiments")
os.makedirs(FIG, exist_ok=True)


def house_style():
    """Shared typography; text colours are forced to black again at save time."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 6.2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def blacken(fig):
    """Force every rendered text object, including ticks and legends, to black."""
    for text in fig.findobj(Text):
        text.set_color("black")


def force_svg_text_black(path):
    """Belt-and-braces pass on the SVG text elements (keeps text editable)."""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(path)
    for elem in tree.iter():
        if elem.tag.rsplit("}", 1)[-1] not in ("text", "tspan"):
            continue
        style = elem.get("style", "")
        if "fill:" in style:
            style = re.sub(r"fill\s*:\s*[^;]+", "fill: #000000", style)
        else:
            style = (style + ";" if style else "") + "fill: #000000"
        elem.set("style", style)
        elem.set("fill", "#000000")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def save_black(fig, number):
    """Save one complete figure as editable SVG and 300 dpi PNG."""
    blacken(fig)
    base = os.path.join(FIG, "Figure %d" % number)
    fig.savefig(base + ".svg", format="svg", bbox_inches="tight",
                pad_inches=0.03)
    fig.savefig(base + ".png", format="png", dpi=300, bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)
    force_svg_text_black(base + ".svg")
    print("wrote %s.svg/.png" % base, flush=True)


def audit_svg(number):
    path = os.path.join(FIG, "Figure %d.svg" % number)
    tree = ET.parse(path)
    texts = [e for e in tree.iter() if e.tag.rsplit("}", 1)[-1] == "text"]
    nonblack = []
    for e in texts:
        style = e.get("style", "")
        fill = re.search(r"fill\s*:\s*([^;]+)", style)
        value = fill.group(1).strip().lower() if fill else "black"
        if value not in ("#000000", "#000", "black", "none"):
            nonblack.append(value)
    return {"text_elements": len(texts), "nonblack_text_fills": nonblack}


def draw_fig1_to_4():
    import r10_figures as r10

    targets = {
        "fig1_benchmark": 1,
        "fig2_calibration": 2,
        "fig4_coverage": 3,
        "fig5_efficiency": 4,
    }
    original_save = r10.save

    def save_r10(fig, name):
        save_black(fig, targets[name])

    house_style()
    r10.save = save_r10
    r10.fig1()
    r10.fig2()
    r10.fig4()
    r10.fig5()
    r10.save = original_save


def draw_fig5():
    import r66_svgstyle as sg
    import r66_svg_c2 as C2
    import r66_svg_c2_panels as P

    house_style()
    jobs = max(1, min(8, os.cpu_count() or 4))
    curve, matched, ratio = C2.load_fig7(jobs=jobs)
    metrics = pd.read_csv(os.path.join(RES, "r03_metrics.csv"))
    pooled = (metrics[metrics.method.isin(["crc", "oracle"])]
              .groupby(["kind", "method", "alpha"])
              .agg(n_accepted=("n_accepted", "sum"),
                   n_wrong=("n_wrong_accepted", "sum"),
                   n_eval=("n_test_eval", "sum")))
    pooled["review"] = 1.0 - pooled["n_accepted"] / pooled["n_eval"]
    pooled["risk"] = pooled["n_wrong"] / pooled["n_accepted"].clip(lower=1)

    def review_panel(ax):
        colors = {"cellsplit": "#1f5fa9", "within": "#2a9d8f",
                  "transfer": "#d1495b"}
        labels = {"cellsplit": "random split", "within": "cross-FOV",
                  "transfer": "cross-tissue"}
        offsets = {"cellsplit": -15, "within": 10, "transfer": 10}
        box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
        for fam in ("cellsplit", "within", "transfer"):
            color = colors[fam]
            oracle = pooled.loc[(fam, "oracle")]
            cert = pooled.loc[(fam, "crc")]
            ax.plot(oracle.index, oracle["review"], ":", lw=1.0, color=color)
            ax.plot(cert.index, cert["review"], "o-", ms=3.4, lw=1.3,
                    color=color, label=labels[fam])
            for alpha in (0.10, 0.20):
                row = cert.loc[alpha]
                ax.annotate("%.3f" % row["risk"],
                            (alpha, row["review"]), textcoords="offset points",
                            xytext=(0, offsets[fam]), fontsize=5.4,
                            color=color, ha="center", bbox=box)
        ax.axhline(1.0, color="black", ls="--", lw=0.9)
        ax.text(0.021, 1.015, "review every cell", fontsize=5.6)
        ax.set_xscale("log")
        ax.set_xticks(sorted(pooled.loc["cellsplit", "crc"].index))
        ax.set_xticklabels(["%.2f" % a
                            for a in sorted(pooled.loc["cellsplit", "crc"].index)],
                           rotation=45)
        alphas = sorted(pooled.loc["cellsplit", "crc"].index)
        ax.set_xlim(alphas[0] * 0.85, alphas[-1] * 1.25)
        ax.set_xlabel("error budget alpha")
        ax.set_ylabel("fraction of cells sent to review")
        ax.set_ylim(-0.06, 1.18)
        ax.legend(fontsize=6, loc="upper right",
                  title="solid = certificate, dotted = oracle",
                  title_fontsize=5.6)

    sg.use()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    review_panel(axes[0])
    sg.title(axes[0], "a", "labels = realised risk")
    P.f7b(axes[1], matched, ratio)
    sg.title(axes[1], "b", "at the same realised risk")
    fig.tight_layout()
    save_black(fig, 5)
    review = {
        "%s_a%.2f" % (fam, alpha):
            float(pooled.loc[(fam, "crc", alpha), "review"])
        for fam in ("cellsplit", "within", "transfer")
        for alpha in (0.10, 0.20)
    }
    risk = {
        "%s_a%.2f" % (fam, alpha):
            float(pooled.loc[(fam, "crc", alpha), "risk"])
        for fam in ("cellsplit", "within", "transfer")
        for alpha in (0.10, 0.20)
    }
    return {
        "review_frac": review,
        "risk": risk,
        "ratio_at_r010": {
            fam: float(ratio[(ratio.family == fam)
                             & np.isclose(ratio.risk, 0.10)]
                       .ratio_crc_oracle.iloc[0])
            for fam in P.R16.FAMILIES
        },
    }


def draw_fig6():
    import r66_svgstyle as sg
    import r66_svg_c3_panels as P

    house_style()
    ctx = P.ctx_s10()
    sg.use()
    fig = plt.figure(figsize=(8.0, 6.3))
    gs = fig.add_gridspec(
        2, 2,
        left=0.075, right=0.985, bottom=0.145, top=0.94,
        wspace=0.37, hspace=0.62,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for (letter, fn), ax in zip(
            [("a", P.s10a), ("b", P.s10b),
             ("c", P.s10c), ("d", P.s10d)], axes):
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
        fn(ax, ctx)
        sg.title(ax, letter, ctx["titles"].get(letter))
    save_black(fig, 6)
    return {
        "certificate": {k: list(v) for k, v in ctx["cert"].items()},
        "by_source": {k: list(v) for k, v in ctx["bys"].items()},
        "ratio_counts": list(ctx["counts"]),
    }


def draw_fig7():
    import r66_svgstyle as sg
    import r90_figure7_workflow as R

    house_style()
    e2e = R.load_end_to_end()
    pilot = R.load_pilot()
    sg.use()
    plt.rcParams.update({
        "font.size": 7.0, "axes.titlesize": 9.0, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
    })
    fig = plt.figure(figsize=(10.8, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.55],
                          width_ratios=[1.0, 1.0],
                          left=0.055, right=0.985, top=0.94, bottom=0.15,
                          hspace=0.58, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    R.draw_workflow(ax_a)
    R.draw_end_to_end(ax_b, e2e)
    R.draw_pilot(ax_c, pilot)
    save_black(fig, 7)
    return {
        "hours_saved": {str(a): float(v["saved_h"]) for a, v in e2e.items()},
        "full_manual_hours": float(e2e[0.05]["full_h"]),
        "acceptance": {str(a): float(v["accept"]) for a, v in e2e.items()},
        "pilot": pilot,
    }


def data_audit():
    """Recompute the key quantities displayed in Figures 1, 2 and 4."""
    out = {}
    ece = pd.read_csv(os.path.join(RES, "r03_ece.csv"))
    fam = ece.groupby(["kind", "domain"])["ece"].mean().unstack()
    fam["ratio"] = fam["target"] / fam["source_cal"]
    out["fig2_ece"] = {
        k: {
            "training": float(row["source_cal"]),
            "deployment": float(row["target"]),
            "ratio": float(row["ratio"]),
        }
        for k, row in fam.iterrows()
    }
    weighted = {
        dom: float(np.average(g["ece"], weights=g["n"]))
        for dom, g in ece[ece.kind == "transfer"].groupby("domain")
    }
    out["fig2_transfer_cell_weighted"] = {
        "training": weighted["source_cal"],
        "deployment": weighted["target"],
        "ratio": weighted["target"] / weighted["source_cal"],
    }

    delivery = pd.read_csv(os.path.join(RES, "r44_gvhd_delivery.csv"))
    d = delivery.groupby("cohort")[["n_labelled", "n_matched"]].sum()
    out["fig1_delivery_fraction"] = {
        k: float(r.n_matched / r.n_labelled) for k, r in d.iterrows()
    }

    t2 = pd.read_csv(os.path.join(RES, "r04_t2_efficiency.csv"))
    allrows = t2[t2.cohort == "ALL"].set_index("family")
    out["fig4_acceptance_all"] = {
        fam: {
            "certificate": float(allrows.loc[fam, "accept_crc"]),
            "oracle": float(allrows.loc[fam, "accept_oracle"]),
            "naive": float(allrows.loc[fam, "accept_naive"]),
            "ratio": float(allrows.loc[fam, "frac_of_oracle"]),
        }
        for fam in ("cellsplit", "within", "transfer")
    }
    return out


def main():
    house_style()
    draw_fig1_to_4()
    audit = {"fig5": draw_fig5(), "fig6": draw_fig6(), "fig7": draw_fig7()}
    audit.update(data_audit())
    audit["svg"] = {str(n): audit_svg(n) for n in range(1, 8)}

    jpath = os.path.join(ROOT, "work", "r96_figure_audit.json")
    with io.open(jpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)
    lines = ["R96 main-figure rebuild and data audit", ""]
    for n in range(1, 8):
        p = os.path.join(FIG, "Figure %d.svg" % n)
        a = audit["svg"][str(n)]
        lines.append("Figure %d: %d text elements, non-black=%d"
                     % (n, a["text_elements"], len(a["nonblack_text_fills"])))
    lines += [
        "",
        "ECE by split family (mean over units):",
        json.dumps(audit["fig2_ece"], indent=2),
        "",
        "Cell-weighted transfer ECE:",
        json.dumps(audit["fig2_transfer_cell_weighted"], indent=2),
        "",
        "Figure 5 review fractions:",
        json.dumps(audit["fig5"]["review_frac"], indent=2),
        "",
        "Figure 5 ratio at r=0.10:",
        json.dumps(audit["fig5"]["ratio_at_r010"], indent=2),
        "",
        "Figure 7 hours saved:",
        json.dumps(audit["fig7"]["hours_saved"], indent=2),
    ]
    with io.open(os.path.join(ROOT, "work", "r96_figure_audit.txt"),
                 "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
