# -*- coding: utf-8 -*-
"""Figure 7 -- from a certificate to a deployment decision.

The panel is built from the released result tables, not from a mock-up:

A  the five operations a laboratory performs, with the decision points that
   determine whether a repair is attempted;
B  the end-to-end liver cohort run, showing net expert hours saved after the
   calibration labels are paid for;
C  the leave-one-fold-out pilot gate, showing how many units are admitted for
   repair in the same-tissue and cross-platform arms.

Outputs are the assembled editable SVG and PNG consumed by the Word build,
plus one standalone editable SVG per panel.  All text stays as text because
``svg.svg.fonttype`` is ``none`` in the shared house style.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import r66_svgstyle as sg

EXP = os.path.join(ROOT, "experiments")
OUT = os.path.join(ROOT, "figures", "svg")
SEC_PER_CELL = 20.0

E2E = os.path.join(EXP, "r70_e2e_demo.csv")
PILOT = os.path.join(EXP, "r77_pilot_decision.csv")


def title(ax, letter, text):
    sg.title(ax, letter, text)
    ax.title.set_fontsize(9.0)


def _need(path):
    if not os.path.exists(path):
        raise SystemExit("missing source table: %s" % path)
    return pd.read_csv(path)


def load_end_to_end():
    d = _need(E2E)
    d = d[(d["arm"] == "layer") & (d["k"] == 500)].copy()
    out = {}
    for alpha in (0.05, 0.20):
        r = d[np.isclose(d["alpha"], alpha)]
        if len(r) != 4:
            raise AssertionError("expected four liver deployment units at alpha %.2f, got %d"
                                 % (alpha, len(r)))
        accept = float(r["n_accepted"].sum()) / float(r["n_eval"].sum())
        risk = float((r["realised_risk"].fillna(0) * r["n_accepted"]).sum()
                     / max(r["n_accepted"].sum(), 1))
        full_h = float(r["n_eval"].sum()
                       + r["n_cal"].clip(lower=0).sum()) * SEC_PER_CELL / 3600.0
        saved_h = float(r["n_accepted"].sum()) * SEC_PER_CELL / 3600.0
        review = int((r["n_eval"] - r["n_accepted"]).sum())
        out[alpha] = dict(accept=accept, risk=risk, full_h=full_h,
                          saved_h=saved_h, review=review,
                          n=int(r["n_eval"].sum()))
    # values quoted in the Results text and in work/r70_report.txt
    if abs(out[0.05]["saved_h"] - 145.1) > 0.2:
        raise AssertionError("5%% net hours changed: %.2f" % out[0.05]["saved_h"])
    if abs(out[0.20]["saved_h"] - 158.8) > 0.2:
        raise AssertionError("20%% net hours changed: %.2f" % out[0.20]["saved_h"])
    if abs(out[0.05]["full_h"] - 170.3) > 0.2:
        raise AssertionError("full manual hours changed: %.2f"
                             % out[0.05]["full_h"])
    if abs(out[0.20]["accept"] - 0.998) > 0.002:
        raise AssertionError("20%% acceptance changed: %.4f" % out[0.20]["accept"])
    return out


def load_pilot():
    d = _need(PILOT)
    r = d[(d["policy"] == "lofo")
          & (d["part"].isin(["B_same_tissue_panel", "E_cross_platform_colon"]))]
    if len(r) != 2:
        raise AssertionError("expected two LOFO pilot arms, got %d" % len(r))
    out = {}
    for _, x in r.iterrows():
        key = "same" if x["part"].startswith("B_") else "cross"
        out[key] = dict(n=int(x["n_units"]), go=int(x["go"]),
                        no=int(x["n_units"] - x["go"]),
                        delta=float(x["delta_mean"]))
    if out["same"]["go"] != 43 or out["cross"]["go"] != 2:
        raise AssertionError("pilot decision counts changed: %s" % out)
    return out


def draw_workflow(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 21)
    ax.axis("off")
    title(ax, "A", "Decision procedure")

    steps = [
        ("1", "Calibration pilot",
         "250 to 2,000 labels for a pilot;\nup to 10,000 for a panel change"),
        ("2", "Score the batch",
         "any classifier,\ns = 1 - max p"),
        ("3", "Certify threshold",
         "selective error <= alpha\nat confidence 1 - delta"),
        ("4", "Commit or review",
         "keep certified labels;\nrank the rest"),
        ("5", "Audit and price",
         "ratio > 1 -> repair;\nlabels versus batch"),
    ]
    x0, y0, w, h = 1.0, 4.0, 16.6, 12.6
    gap = 3.6
    for i, (num, head, body) in enumerate(steps):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch((x, y0), w, h,
                             boxstyle="round,pad=0.25,rounding_size=0.6",
                             linewidth=0.7, edgecolor="#5f6b7a",
                             facecolor="#f7f8fa", zorder=1)
        ax.add_patch(box)
        ax.text(x + 1.0, y0 + h - 2.4, num, ha="left", va="center",
                fontsize=8.4, fontweight="bold", color=sg.C["blue"], zorder=2)
        ax.text(x + 3.0, y0 + h - 2.4, head, ha="left", va="center",
                fontsize=8.0, fontweight="bold", color="#1b1b1b", zorder=2)
        ax.text(x + 1.0, y0 + 4.5, body, ha="left", va="center",
                fontsize=7.4, color="#333333", linespacing=1.35, zorder=2)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.25, y0 + h / 2.0),
                (x + w + gap - 0.25, y0 + h / 2.0),
                arrowstyle="-|>", mutation_scale=6.5,
                linewidth=0.8, color="#5f6b7a", zorder=2))
    ax.text(50, 1.2,
            "The guarantee holds when calibration represents deployment. "
            "The audit detects when it does not.",
            ha="center", va="center", fontsize=7.1, color="#555555")


def draw_end_to_end(ax, e2e):
    title(ax, "B", "One batch, end to end")
    budgets = ["5%", "20%"]
    vals = [e2e[0.05]["saved_h"], e2e[0.20]["saved_h"]]
    full = e2e[0.05]["full_h"]
    y = np.arange(len(budgets))[::-1]
    ax.barh(y, vals, height=0.48, color=[sg.C["blue"], sg.C["blue"]],
            edgecolor=sg.C["blue"], linewidth=0.5, zorder=3)
    ax.axvline(full, color="#333333", linestyle=(0, (4, 3)),
               linewidth=0.8, zorder=2)
    for yi, v, alpha in zip(y, vals, (0.05, 0.20)):
        ax.text(v + 1.5, yi, "%.1f h" % v, va="center", ha="left",
                fontsize=7.4, fontweight="bold", color="#1b1b1b")
        ax.text(0.8, yi - 0.31,
                "accept %.3f   selective error %.3f   review %s cells"
                % (e2e[alpha]["accept"], e2e[alpha]["risk"],
                   format(e2e[alpha]["review"], ",")),
                va="center", ha="left", fontsize=6.3, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels(budgets)
    ax.set_xlim(0, 190)
    ax.set_xlabel("net expert hours saved after calibration")
    ax.text(full - 2.0, 1.50, "full manual\npass %.1f h" % full,
            ha="right", va="top", fontsize=6.3, color="#333333")
    ax.set_ylim(-0.70, 1.55)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_pilot(ax, pilot):
    title(ax, "C", "250-label pilot gate")
    labels = ["Same tissue", "Cross-platform"]
    go = np.array([pilot["same"]["go"], pilot["cross"]["go"]], dtype=float)
    no = np.array([pilot["same"]["no"], pilot["cross"]["no"]], dtype=float)
    total = go + no
    y = np.arange(2)[::-1]
    ax.barh(y, go / total, color=sg.C["teal"], height=0.44,
            edgecolor=sg.C["teal"], linewidth=0.5, zorder=3)
    ax.barh(y, no / total, left=go / total, color="#d9dde3", height=0.44,
            edgecolor="#d9dde3", linewidth=0.5, zorder=3)
    for yi, g, n, t in zip(y, go, no, total):
        if g / t > 0.15:
            ax.text(g / t / 2.0, yi, "%d" % int(g), ha="center", va="center",
                    fontsize=7.4, color="white", fontweight="bold")
        ax.text(1.015, yi, "%d/%d admitted" % (int(g), int(t)), va="center",
                ha="left", fontsize=6.3, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("share of units admitted for repair")
    ax.set_ylim(-0.55, 1.55)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, -0.43,
            "leave-one-fold-out gate; counts are units",
            ha="left", va="center", fontsize=6.4, color="#555555")


def save_panel(fig, name):
    sg.save_full(fig, os.path.join(OUT, name + ".svg"))


def main():
    e2e = load_end_to_end()
    pilot = load_pilot()
    os.makedirs(OUT, exist_ok=True)

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
    draw_workflow(ax_a)
    draw_end_to_end(ax_b, e2e)
    draw_pilot(ax_c, pilot)
    save_panel(fig, "fig11_decision_workflow_full")
    print("wrote figures/svg/fig11_decision_workflow_full.svg/.png")

    # standalone editable panels
    fig, ax = plt.subplots(figsize=(10.4, 2.55))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.08)
    draw_workflow(ax)
    sg.save(fig, os.path.join(OUT, "fig11_decision_workflow_a.svg"))

    fig, ax = plt.subplots(figsize=(5.1, 3.25))
    fig.subplots_adjust(left=0.24, right=0.96, top=0.84, bottom=0.18)
    draw_end_to_end(ax, e2e)
    sg.save(fig, os.path.join(OUT, "fig11_decision_workflow_b.svg"))

    fig, ax = plt.subplots(figsize=(5.1, 3.25))
    fig.subplots_adjust(left=0.24, right=0.84, top=0.84, bottom=0.18)
    draw_pilot(ax, pilot)
    sg.save(fig, os.path.join(OUT, "fig11_decision_workflow_c.svg"))

    print("wrote three standalone editable panel SVGs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
