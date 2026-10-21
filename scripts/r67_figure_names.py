# -*- coding: utf-8 -*-
"""r67 -- name the editable SVG figures after their position in the manuscript.

The Nature Communications build numbers the main display items 1..6 and the
supplementary ones separately, so this driver writes

    figures/Figure <N>.svg                  the assembled main figure, all panels
    figures/Figure <N><L>.svg               one standalone panel, L in A, B, C, ...
    figures/Supplementary Figure <N>.svg    the assembled supplementary figure
    figures/Supplementary Figure <N><L>.svg one standalone supplementary panel

from the per panel SVGs produced by the r66 drivers.  The panels are the same
files, copied under the name the manuscript uses, so the folder is readable on
its own and its numbering agrees with the legends in
``1015manuscript_NC.docx`` and ``1015supplementary_NC.docx``.  Nothing is
redrawn and no text is outlined.
"""
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SVG = os.path.join(ROOT, "figures", "svg")
FIG = os.path.join(ROOT, "figures")
ASSETS = os.path.join(ROOT, "docs", "figure_assets.json")

# manuscript figure number -> asset key in docs/figure_assets.json, in the order
# the two documents insert them (r65_nc_docx.py NC_ASSETS, supplementary_nc.md)
NC_MAIN = [(1, "1"), (2, "2"), (3, "3"), (4, "5"), (5, "10"), (6, "17")]
NC_SUPP = [(1, "4"), (2, "6"), (3, "7"), (4, "8"), (5, "9"), (6, "11"),
           (7, "12"), (8, "13"), (9, "14"), (10, "15"), (11, "16")]

assets = json.loads(io.open(ASSETS, encoding="utf-8").read())
have = set(os.listdir(SVG))

PLAN = [("Figure %d" % n, key) for n, key in NC_MAIN] + \
       [("Supplementary Figure %d" % n, key) for n, key in NC_SUPP]

made, rows, missing = 0, [], []
for name, key in PLAN:
    stem = assets[key]["stem"]
    full = "%s_full.svg" % stem
    if full in have:
        with io.open(os.path.join(SVG, full), encoding="utf-8") as fh:
            body = fh.read()
        io.open(os.path.join(FIG, "%s.svg" % name), "w", encoding="utf-8",
                newline="").write(body)
        made += 1
    else:
        missing.append(full)
    letters = sorted(m.group(1) for f in have
                     for m in [re.fullmatch(re.escape(stem) + r"_([a-z])\.svg", f)]
                     if m)
    for L in letters:
        with io.open(os.path.join(SVG, "%s_%s.svg" % (stem, L)),
                     encoding="utf-8") as fh:
            body = fh.read()
        io.open(os.path.join(FIG, "%s%s.svg" % (name, L.upper())), "w",
                encoding="utf-8", newline="").write(body)
        made += 1
    rows.append((name, stem, len(letters), "".join(x.upper() for x in letters),
                 os.path.exists(os.path.join(SVG, "%s_full.png" % stem))))

# drop the files the old numbering had put there
keep = set("%s.svg" % n for n, _ in PLAN)
keep |= set("%s%s.svg" % (n, L) for n, _ in PLAN for L in "ABCDEFGH")
removed = []
for f in sorted(os.listdir(FIG)):
    if re.fullmatch(r"(Supplementary )?Figure \d+[A-Z]?\.svg", f) and f not in keep:
        os.remove(os.path.join(FIG, f))
        removed.append(f)

print("wrote %d files into figures/" % made)
print("%-28s %-28s %5s %-10s %s" % ("manuscript name", "stem", "panel", "letters", "png"))
for name, stem, np_, letters, png in rows:
    print("%-28s %-28s %5d %-10s %s" % (name, stem, np_, letters, png))
if removed:
    print("removed %d stale files: %s" % (len(removed), ", ".join(removed[:6])
                                          + (" ..." if len(removed) > 6 else "")))
if missing:
    print("MISSING assembled svg:", missing)
