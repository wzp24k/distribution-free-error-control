# -*- coding: utf-8 -*-
"""R66-C1 -- the r10 manuscript figures, redrawn as per-panel editable SVG.

For every figure in ``scripts/r10_figures.py`` (fig1_benchmark, fig2_calibration,
fig3_method, fig4_coverage, fig5_efficiency, fig6_markers, figS1_labelnoise) this
driver writes two kinds of file into ``figures/svg/``:

* ``<key>_full.svg`` and ``<key>_full.png`` -- the assembled multi-panel figure,
  reproduced through the very same panel functions that r10_figures.py uses, so
  the panels, their layout and every number in them are the ones the manuscript
  already shows (the PNG behind ``figures/<key>.png`` is left untouched).
* ``<key>_<letter>.svg`` -- one fresh single-panel figure per panel, drawn by the
  panel's own function from ``scripts/r10_figures.py`` on an axes created by
  ``scripts/r66_svgstyle.py``.  ``svg.fonttype`` is "none", so every tick, label
  and title stays a real ``<text>`` element: nothing is converted to outlines.

Panels are lettered a, b, c, ... in reading order (left to right, top to
bottom).  The title of a panel is the same short descriptor the assembled
figure already carries, at the one house title size (svgstyle.TITLE_SIZE).

The assembled figure keeps r10's own style so that it reproduces today's
rendering; only the panel files are drawn in the house style.
"""

import functools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
import matplotlib.pyplot as plt

import r10_figures as r10
import r66_svgstyle as sg

OUT = os.path.join(ROOT, "figures", "svg")

# r10_figures applies its own rcParams at import; keep a copy so the assembled
# figures can be drawn in exactly that style after svgstyle has been used.
R10_RC = dict(matplotlib.rcParams)
_R10_PANEL = r10.panel


def _load_noise():
    d = r10.load_labelnoise()
    return None if d is None else {"d": d}


# --------------------------------------------------------------------- table
# (key, builder, loader, [(letter, title, panel function, data key), ...])
FIGURES = [
    ("fig1_benchmark", r10.fig1,
     lambda: {"masters": r10.load_masters()},
     [("a", "benchmark", r10.p1a, "masters"),
      ("b", "panel overlap", r10.p1b, "masters"),
      ("c", "the shipped HQ labels", r10.p1c, "masters"),
      ("d", "delivery audit (all FOVs)", r10.p1d, None)]),
    ("fig2_calibration", r10.fig2,
     lambda: {"ece": r10.load_ece()},
     [("a", "Breast", r10.p2a, None),
      ("b", "CRC", r10.p2b, None),
      ("c", "GVHD", r10.p2c, None),
      ("d", "Melanoma", r10.p2d, None),
      ("e", "under transfer", r10.p2e, "ece"),
      ("f", "under transfer", r10.p2f, "ece"),
      ("g", "by split family", r10.p2g, "ece"),
      ("h", "relative ECE", r10.p2h, "ece")]),
    ("fig3_method", r10.fig3,
     lambda: {"ctx": r10.load_fig3_ctx()},
     [("a", "choosing the threshold", r10.p3a, "ctx"),
      ("b", "the bound is the constraint", r10.p3b, "ctx"),
      ("c", "the rule", r10.p3c, None)]),
    ("fig4_coverage", r10.fig4,
     lambda: {"M": r10.load_metrics()},
     [("a", "guarantee coverage", r10.p4a, "M"),
      ("b", "coverage vs budget", r10.p4b, "M"),
      ("c", "valid but vacuous?", r10.p4c, None),
      ("d", "cost of few labels", r10.p4d, None)]),
    ("fig5_efficiency", r10.fig5,
     lambda: {"t2": r10.load_efficiency()},
     [("a", "risk-coverage (hgb)", r10.p5a, None),
      ("b", "acceptance vs oracle", r10.p5b, "t2"),
      ("c", "the tell-tale ratio", r10.p5c, "t2")]),
    ("fig6_markers", r10.fig6,
     lambda: r10.load_markers(),
     [("a", "who contradicts the rules", r10.p6a, "audit"),
      ("b", "keeps plausible cells", r10.p6b, "selective"),
      ("c", "downstream spatial statistic", r10.p6c, "spatial")]),
    ("figS1_labelnoise", r10.figS1,
     _load_noise,
     [("a", "throughput cost", r10.pS1a, "d"),
      ("b", "validity survives", r10.pS1b, "d"),
      ("c", "when it stops answering", r10.pS1c, "d")]),
]


# ------------------------------------------------------------------ drawing
def _save_full(fig, name):
    """r10's ``save`` hook: the assembled figure, as editable SVG plus PNG."""
    sg.save_full(fig, os.path.join(OUT, "%s_full.svg" % name))


def render_combined(build):
    """Redraw one assembled figure in r10's style and save it under OUT."""
    matplotlib.rcParams.update(R10_RC)
    matplotlib.rcParams["svg.fonttype"] = "none"
    r10.panel = _R10_PANEL
    r10.save = _save_full
    build()


def render_panel(key, letter, title, func, data, argkey):
    """Draw one panel on its own fresh house-style figure and save the SVG."""
    fig, ax = sg.new(letter, title)
    r10.panel = sg.title          # one title size on every panel
    if argkey is None:
        func(ax, letter=letter, title=title)
    else:
        func(ax, data[argkey], letter=letter, title=title)
    sg.save(fig, os.path.join(OUT, "%s_%s.svg" % (key, letter)))


def main():
    os.makedirs(OUT, exist_ok=True)
    # the master tables are read once and shared by the assembled figure and
    # by every panel that needs them
    r10.load_master = functools.lru_cache(maxsize=None)(r10.load_master)
    for key, build, load, panels in FIGURES:
        render_combined(build)
        data = load()
        if data is None:
            print("skipped %s: its result table is missing" % key)
            continue
        for letter, title, func, argkey in panels:
            render_panel(key, letter, title, func, data, argkey)
        print("wrote figures/svg/%s_full.svg+png and %d panel SVGs"
              % (key, len(panels)))
    plt.close("all")


if __name__ == "__main__":
    main()
