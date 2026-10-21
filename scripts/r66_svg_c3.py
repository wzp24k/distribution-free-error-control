# -*- coding: utf-8 -*-
"""r66_svg_c3 -- regenerate six supplementary figures as editable-text SVG.

For each figure this writes, into figures/svg/:

  <key>_full.svg        the combined multi-panel figure, one editable SVG
  <key>_full.png        the same at 300 dpi for the Word embedding
  <key>_<letter>.svg    every panel on its own, drawn fresh by its own function

Every panel is drawn by the matching function in scripts/r66_svg_c3_panels.py,
which carries the drawing code of the original producer (r15 / r18 / r19 /
r35).  Titles come from the shared house style scripts/r66_svgstyle.py, so the
title font size is identical on every panel of every figure, and
svg.fonttype = "none" keeps every label as real, editable <text>.

The combined figures reproduce the original panel set, layout and numbers; only
the type styling is the shared house style.  The original scripts are untouched
and still write their own figures/*.png and figures/*.pdf.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt

import r66_svgstyle as svgstyle
import r66_svg_c3_panels as P

SVG_DIR = os.path.join(ROOT, "figures", "svg")


def _prep(ax):
    """The axes dressing svgstyle.new() applies, reused for full figures."""
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


def run(key):
    spec = P.FIGS[key]
    ctx = spec["ctx"]()
    os.makedirs(SVG_DIR, exist_ok=True)

    svgstyle.use()
    fig = plt.figure(figsize=spec["size"])
    axes = spec["layout"](fig)
    for (letter, fn), ax in zip(spec["panels"], axes):
        _prep(ax)
        fn(ax, ctx)
        svgstyle.title(ax, letter, ctx["titles"].get(letter))
    if spec["tight"]:
        fig.tight_layout()
    svgstyle.save_full(fig, os.path.join(SVG_DIR, key + "_full.svg"))
    print("wrote figures/svg/%s_full.svg/.png" % key)

    for letter, fn in spec["panels"]:
        fig2, ax2 = svgstyle.new(letter, ctx["titles"].get(letter))
        fn(ax2, ctx)
        svgstyle.save(fig2, os.path.join(SVG_DIR, "%s_%s.svg" % (key, letter)))
    print("wrote %d panel SVG(s) for %s" % (len(spec["panels"]), key))


def main():
    for key in P.KEYS:
        run(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
