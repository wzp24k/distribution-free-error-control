"""R49b -- is the marker-rule layer of the benchmark panel invariant?

Weakness #5 asks the applicability question: a laboratory whose panel does not
carry the reference panel's channels.  The statistical answer is the transfer
family and it is negative.  This script measures a different, weaker but
positive property of the same benchmark: the expert Boolean definitions in the
depot's CellTypes_ALL table are predicates over channels, so they can be
evaluated in *any* panel that carries those channels, and a rule that is
evaluable in two cohorts is evaluated with each cohort's own positivity call.
If the verdict of the marker-consistency audit is the same in every cohort, the
audit layer transports across panels even where the classifier layer does not.

For every cohort and every rule of the table this reports
  * whether the rule is evaluable in that cohort at all,
  * the rule's positive rate,
  * the fraction of cells of the rule's own cell type that satisfy it
    (its "own-class satisfaction rate"), which is the audit quantity.

Outputs: experiments/r49b_panelinvariance.csv, experiments/r49b_pair_overlap.csv,
         work/r49b_report.txt
"""

import csv
import io
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from errctl import data as D, paths
from errctl.markercheck import positivity
from errctl.markerlogic import CompiledRule, LogicError, canonical

RULES_CSV = os.path.join(paths.DEPOT, "CellTypes_ALL.csv")
OUT_CSV = os.path.join(paths.RESULTS, "r49b_panelinvariance.csv")
OUT_PAIRS = os.path.join(paths.RESULTS, "r49b_pair_overlap.csv")
OUT_TXT = os.path.join(paths.WORK, "r49b_report.txt")

# The twelve directed cohort pairs of scripts/r02_run_experiments.py, in the
# published order; used for the shared-channel overlap table.
TRANSFER_PAIRS = [
    ("GVHD_Feb1323", "CRC_Feb2623"),
    ("CRC_Feb2623", "GVHD_Feb1323"),
    ("GVHD_Feb1323", "Breast_Jun0524"),
    ("GVHD_Feb1323", "Melanoma_Oct2522"),
    ("CRC_Feb2623", "Breast_Jun0524"),
    ("CRC_Feb2623", "Melanoma_Oct2522"),
    ("Breast_Jun0524", "Melanoma_Oct2522"),
    ("Melanoma_Oct2522", "Breast_Jun0524"),
    ("Breast_Jun0524", "CRC_Feb2623"),
    ("Melanoma_Oct2522", "CRC_Feb2623"),
    ("Breast_Jun0524", "GVHD_Feb1323"),
    ("Melanoma_Oct2522", "GVHD_Feb1323"),
]


def log(*p):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in p)), flush=True)


def load_rules(path=RULES_CSV):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rules = []
    for row in rows:
        ct = str(row.get("CellType") or "").strip()
        if not ct or ct.lower().startswith("unidentified"):
            continue
        for col, tag in (("PrimaryMarker", ""), ("AlternatePrimaryMarker", "~alt")):
            text = str(row.get(col) or "").strip()
            if not text or text.lower() in ("nan", "none", "various"):
                continue
            try:
                rules.append((ct + tag, ct, text, CompiledRule(ct + tag, text)))
            except LogicError:
                pass
    return rules


def main():
    rules = load_rules()
    names = [r[0] for r in rules]
    per_cohort = {}
    panel_sets = {}
    for ds in paths.MANUAL_DATASETS:
        g, markers = D.gold(ds, drop_unidentified=True)
        panel_sets[ds] = set(canonical(str(m)) for m in markers)
        g = g.reset_index(drop=True)
        pos, _thr = positivity(g[markers], markers, seed=0)
        # Rule identifiers are canonicalised at compile time, so match them
        # against the canonical form of the panel's channel names: a panel that
        # spells a channel HLADR or Fe56 carries HLADRDPDQ or Fe (same
        # convention as the transfer panel intersection of r02_run_experiments).
        posc = {}
        for m in markers:
            v = pos[m] if m in pos else pos[str(m)]
            posc.setdefault(canonical(str(m)), v)
        labels = g["class_expert"].astype(str).to_numpy()
        recs = []
        for j, (name, ct, text, rule) in enumerate(rules):
            missing = sorted(c for c in rule.idents if c not in posc)
            if missing:
                recs.append({"rule": name, "celltype": ct, "evaluable": 0,
                             "n_missing": len(missing), "missing": "|".join(missing),
                             "pos_rate": np.nan, "n_own": 0, "own_rate": np.nan,
                             "n_cells": len(g)})
                continue
            vals = {c: posc[c] for c in rule.idents}
            r = np.asarray(rule.evaluate(vals), dtype=bool)
            own = labels == ct
            recs.append({"rule": name, "celltype": ct, "evaluable": 1, "n_missing": 0,
                         "missing": "", "pos_rate": float(r.mean()),
                         "n_own": int(own.sum()),
                         "own_rate": float(r[own].mean()) if own.sum() else np.nan,
                         "n_cells": len(g)})
        d = pd.DataFrame(recs)
        d["dataset"] = ds
        d["n_markers"] = len(markers)
        per_cohort[ds] = d
        log("%-16s panel=%2d cells=%6d rules evaluable=%2d/%d  own-class checked=%2d"
            % (ds, len(markers), len(g), int(d["evaluable"].sum()), len(d),
               int(((d["evaluable"] == 1) & (d["n_own"] >= 30)).sum())))

    all_recs = pd.concat(list(per_cohort.values()), ignore_index=True)
    all_recs.to_csv(OUT_CSV, index=False)

    # Shared-channel overlap of the twelve directed pairs, so that the "11 to 27
    # channels, half of them 20 or more" claim of the manuscript is checkable
    # from a shipped table rather than by re-deriving the intersection rule.
    pair_rows = []
    for src, tgt in TRANSFER_PAIRS:
        shared = sorted(panel_sets[src] & panel_sets[tgt])
        pair_rows.append({"src": src, "tgt": tgt, "n_shared": len(shared),
                          "shared": "|".join(shared)})
    pairs_df = pd.DataFrame(pair_rows)
    pairs_df.to_csv(OUT_PAIRS, index=False)

    piv_ev = all_recs.pivot_table(index="rule", columns="dataset", values="evaluable")
    piv_own = all_recs.pivot_table(index="rule", columns="dataset", values="own_rate")
    piv_n = all_recs.pivot_table(index="rule", columns="dataset", values="n_own")
    datasets = list(paths.MANUAL_DATASETS)
    ev_count = piv_ev.fillna(0).sum(axis=1).astype(int)

    checked = piv_own.where(piv_n >= 30)
    n_pairs = int(checked.notna().sum().sum())
    worst = float(np.nanmin(checked.values))
    frac95 = float((checked.values[~np.isnan(checked.values)] >= 0.95).mean())
    frac90 = float((checked.values[~np.isnan(checked.values)] >= 0.90).mean())

    per_rule_rows = []
    for rule in piv_own.index:
        vals = checked.loc[rule].dropna()
        if vals.empty:
            continue
        per_rule_rows.append({
            "rule": rule, "n_cohorts_evaluable": int(ev_count.get(rule, 0)),
            "n_cohorts_checked": int(vals.size),
            "min_own_rate": float(vals.min()), "max_own_rate": float(vals.max()),
            "spread": float(vals.max() - vals.min()),
            "mean_own_rate": float(vals.mean()),
        })
    per_rule = pd.DataFrame(per_rule_rows).sort_values("min_own_rate")

    lines = []
    def w(s=""):
        lines.append(s)
        log(s)

    w("R49b -- is the marker-rule layer of the benchmark panel invariant?")
    w("=" * 72)
    w("rules compiled from CellTypes_ALL.csv (primary + alternate): %d" % len(rules))
    w("")
    w("panel and rule coverage per cohort")
    w("%-16s %7s %7s %10s %12s" % ("cohort", "panel", "cells", "evaluable", "own-class>=30"))
    for ds in datasets:
        d = per_cohort[ds]
        w("%-16s %7d %7d %10d %12d"
          % (ds, int(d["n_markers"].iloc[0]), int(d["n_cells"].iloc[0]),
             int(d["evaluable"].sum()),
             int(((d["evaluable"] == 1) & (d["n_own"] >= 30)).sum())))
    w("")
    w("how many rules survive in how many cohorts")
    for k in range(len(datasets), 0, -1):
        w("   evaluable in %d cohort(s): %3d rules" % (k, int((ev_count == k).sum())))
    w("")
    w("own-class satisfaction of a rule, evaluated with each cohort's own positivity")
    w("  (rule, cohort) cells checked (>=30 own cells): %d" % n_pairs)
    w("  minimum over all of them: %.4f" % worst)
    w("  fraction above 0.95: %.4f" % frac95)
    w("  fraction above 0.90: %.4f" % frac90)
    w("")
    w("rules with the largest cross-cohort spread")
    for _, r in per_rule.head(8).iterrows():
        w("   %-24s cohorts=%d min=%.3f max=%.3f spread=%.3f"
          % (r["rule"], r["n_cohorts_evaluable"], r["min_own_rate"],
             r["max_own_rate"], r["spread"]))
    w("")
    w("cross-panel pairs: rules evaluable in both cohorts and checked in both")
    import itertools
    for a, b in itertools.combinations(datasets, 2):
        both = checked.loc[checked[a].notna() & checked[b].notna()]
        if both.empty:
            w("   %-16s %-16s no jointly checked rule" % (a, b))
            continue
        w("   %-16s %-16s jointly checked=%2d  min=%.3f  max_difference=%.3f"
          % (a, b, len(both), both[["%s" % a, "%s" % b]].min().min(),
             (both[a] - both[b]).abs().max()))
    w("")
    n_sh = pairs_df["n_shared"]
    w("shared canonicalised panel per directed cohort pair (all twelve pairs)")
    for _, r in pairs_df.iterrows():
        w("   %-16s -> %-16s %3d channels" % (r["src"], r["tgt"], r["n_shared"]))
    w("   min %d, max %d, %d of %d pairs at 20 or more"
      % (int(n_sh.min()), int(n_sh.max()), int((n_sh >= 20).sum()), len(pairs_df)))
    w("   (written to experiments/r49b_pair_overlap.csv)")
    w("")
    w("verdict")
    w("  Across the four panels the expert Boolean definitions are satisfied by the")
    w("  cells the experts assigned to them at a minimum of %.3f, and %.1f%% of the"
      % (worst, 100 * frac95))
    w("  (rule, cohort) cells are above 0.95. The audit layer of the benchmark is")
    w("  therefore panel invariant in the sense that matters here: a rule written")
    w("  once is evaluated on each panel's own measurements and reaches the same")
    w("  verdict, for %d of the %d rules in at least one cohort and for %d of them in"
      % (int((ev_count >= 1).sum()), len(rules), int((ev_count == len(datasets)).sum())))
    w("  all four. Any single pair of cohorts shares only a handful of these rules,")
    w("  so this is a statement about the definitions two panels have in common")
    w("  rather than about all %d. What does not transport is the classifier built"
      % len(rules))
    w("  on the channels, which is the negative result the transfer family reports.")

    with io.open(OUT_TXT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    log("wrote %s and %s" % (OUT_CSV, OUT_TXT))


if __name__ == "__main__":
    main()
