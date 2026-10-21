# -*- coding: utf-8 -*-
"""Regenerate Figure 6 as a 2 x 2 editable-text SVG/PNG.

Panels A-D reuse the released drawing functions and data checks from
``r66_svg_c3_panels``. Only the multi-panel layout changes, from one row of
four panels to two rows of two panels.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import r66_svgstyle as sg
import r66_svg_c3_panels as panels

OUT = os.path.join(ROOT, "figures", "svg")
STEM = "figS10_external"


def draw_full(ctx):
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
            [("a", panels.s10a), ("b", panels.s10b),
             ("c", panels.s10c), ("d", panels.s10d)], axes):
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
        fn(ax, ctx)
        sg.title(ax, letter, ctx["titles"].get(letter))
    sg.save_full(fig, os.path.join(OUT, STEM + "_full.svg"))
    print("wrote figures/svg/%s_full.svg/.png" % STEM)


def draw_panels(ctx):
    for letter, fn in [("a", panels.s10a), ("b", panels.s10b),
                       ("c", panels.s10c), ("d", panels.s10d)]:
        fig, ax = sg.new(letter, ctx["titles"].get(letter))
        fn(ax, ctx)
        sg.save(fig, os.path.join(OUT, "%s_%s.svg" % (STEM, letter)))
    print("wrote four standalone editable panel SVGs")


def main():
    ctx = panels.ctx_s10()
    os.makedirs(OUT, exist_ok=True)
    draw_full(ctx)
    draw_panels(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
