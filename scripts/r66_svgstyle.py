
# -*- coding: utf-8 -*-
"""Shared style for the regenerated per-panel editable SVG figures.

Every panel SVG in figures/svg/ is produced through this module so that the
whole figure set shares one visual language and one title font size.  Text is
kept as text (svg.fonttype = none) so every label stays editable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"blue": "#1f5fa9", "red": "#d1495b", "grey": "#8d99ae",
     "gold": "#e6a817", "teal": "#2a9d8f", "green": "#6ab187",
     "purple": "#7b4fa0", "black": "#222222", "light": "#b0b0b0"}
CYCLE = [C["blue"], C["red"], C["teal"], C["gold"], C["purple"],
         C["green"], C["grey"], C["black"]]

TITLE_SIZE = 8.0
LABEL_SIZE = 7.0
TICK_SIZE = 6.0
LEGEND_SIZE = 5.8
ANNOT_SIZE = 6.0
LINE_W = 1.1
MARKER_SIZE = 2.6
PANEL_W = 2.45
PANEL_H = 2.05


def use():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "0.25",
        "axes.labelcolor": "0.15",
        "text.color": "0.15",
        "xtick.color": "0.25",
        "ytick.color": "0.25",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "grid.color": "0.85",
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def title(ax, letter, text=None):
    """The one house title for a panel: same size on every panel."""
    letter = letter.upper()
    lab = letter if text is None else "%s  %s" % (letter, text)
    ax.set_title(lab, fontsize=TITLE_SIZE, loc="left", pad=3.0)


def new(letter=None, text=None, w=PANEL_W, h=PANEL_H, rect=None):
    """A fresh single-panel figure in the house style."""
    use()
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes(rect or [0.20, 0.20, 0.76, 0.70])
    if letter is not None:
        title(ax, letter, text)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, path_svg):
    fig.savefig(path_svg, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_full(fig, path_svg):
    """Combined figure: editable SVG plus a PNG for the Word embedding."""
    fig.savefig(path_svg, format="svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path_svg[:-4] + ".png", format="png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
