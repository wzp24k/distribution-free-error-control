"""R6 -- is an annotation biologically consistent with the marker logic?

Accuracy against expert labels cannot audit the labels themselves: if the
ground truth is wrong, a classifier that reproduces it scores perfectly. The
depot ships an independent source of truth -- each population is defined there by
a Boolean rule over the marker channels (CD4&CD3, (CD56|CD57)&CD45, ...). This
script turns those rules into a per-cell consistency check and asks four
questions:

1. How often does an *expert* label violate the rule of the population it names
   -- the irreducible inconsistency of the ground truth?
2. How often do the depot's own semi-supervised HighQuality labels violate it?
3. How often do the classifiers' predictions violate it, and how does the
   violation rate split between cells the classifier gets right and wrong?
4. Do the cells the risk controller accepts look more consistent than the ones
   it defers -- i.e. does error control also buy biological plausibility?

A fifth section reports what the same errors do to a downstream spatial
statistic (same-type neighbourhood enrichment).

Outputs: experiments/r06_marker_audit.csv, r06_selective.csv, r06_violations.csv,
r06_spatial.csv, and a readable work/r06_report.txt.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import data as D, markercheck as MC, paths
from errctl.labels import COARSE_MAP
from errctl.markerlogic import canonical, compile_rules

# channels that are not proteins and must never drive a positivity call
NON_MARKERS = {"segmentation_borders", "DRAQ5", "DNA", "Morphology", "nearCD31"}

PRED_DIR = os.path.join(paths.WORK, "predictions")
OUT = paths.RESULTS


def log(msg=""):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(paths.WORK, "r06_report.txt"), "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def canonical_frame(g, markers):
    """Marker columns renamed to canonical channel names, duplicates dropped."""
    taken, cols = set(), []
    for m in markers:
        c = canonical(m)
        if c in taken or c in NON_MARKERS or m in NON_MARKERS:
            continue
        taken.add(c)
        cols.append((c, m))
    ren = {m: c for c, m in cols}
    frame = g[list(ren)].rename(columns=ren)
    return frame, [c for c, _ in cols]


def build_groups(ds, semicolon="or"):
    """{coarse class: [rule, ...]} with the OR semantics of the coarse map.

    The depot writes some definitions with ";" ("CD3;CD4", "CD68;CD163;CD11b")
    and nowhere states whether that separator means AND or OR -- in
    ``Channels.csv`` the same character separates alternative annotations. Both
    readings are therefore reported.
    """
    table = D.celltype_table(ds).copy()
    op = "|" if semicolon == "or" else "&"
    table["PrimaryMarker"] = (table["PrimaryMarker"].fillna("").astype(str)
                              .str.replace(";", op, regex=False))
    rules, failed = compile_rules(table)
    groups = {}
    for ct, rule in rules.items():
        coarse = COARSE_MAP.get(ct)
        if not coarse or coarse == "Unidentified":
            continue
        groups.setdefault(coarse, []).append(rule)
    return groups, failed


def complete_classes(groups, pos):
    """Classes whose *every* published rule is evaluable on this panel.

    A class whose group lost a rule to a missing channel -- ``Morphology`` or a
    misspelt channel, say -- can only be checked partially, and a violation then
    says more about the missing channel than about the annotation.
    """
    known = set(pos)
    return {c for c, rs in groups.items() if rs and all(r.idents <= known for r in rs)}


def evaluate_source(labels, groups, pos, restrict=None):
    """(evaluable, violated) masks for cells carrying the given coarse labels."""
    labels = np.asarray(labels, dtype=object)
    n = labels.size
    evaluable = np.zeros(n, dtype=bool)
    violated = np.zeros(n, dtype=bool)
    for coarse, rules in groups.items():
        if restrict is not None and coarse not in restrict:
            continue
        m = labels == coarse
        if not m.any():
            continue
        usable = [r for r in rules if r.idents <= set(pos)]
        if not usable:
            continue
        ok = np.zeros(int(m.sum()), dtype=bool)
        for r in usable:
            values = {i: pos[i][m] for i in r.idents}
            ok |= np.asarray(r.evaluate(values), dtype=bool)
        evaluable[m] = True
        violated[m] = ~ok
    return evaluable, violated


def load_predictions(ds, clf):
    """(image, cellID) -> (predicted coarse label, confidence) from within units."""
    out = {}
    for f in range(5):
        p = os.path.join(PRED_DIR, "within_%s_fold%d__%s.npz" % (ds, f, clf))
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        classes = np.asarray(d["classes"], dtype=object)
        probs = np.asarray(d["test_probs"], dtype=np.float64)
        pred = classes[probs.argmax(1)]
        conf = probs.max(1)
        for img, cid, lab, cf in zip(d["test_img"], d["test_cellid"], pred, conf):
            out[(str(img), int(cid))] = (str(lab), float(cf))
    return out


def rate_table(ds, source, labels, groups, pos, complete=None):
    if complete is None:
        complete = complete_classes(groups, pos)
    labels = np.asarray(labels, dtype=object)
    evaluable, violated = evaluate_source(labels, groups, pos)
    rows = []
    for coarse in sorted(set(labels[evaluable])):
        m = evaluable & (labels == coarse)
        rules = groups[coarse]
        used = [r for r in rules if r.idents <= set(pos)]
        rows.append({"dataset": ds, "source": source, "coarse_class": coarse,
                     "n": int(m.sum()), "n_violation": int(violated[m].sum()),
                     "violation_rate": float(violated[m].mean()),
                     "n_rules_used": len(used), "n_rules_total": len(rules),
                     "rule_complete": int(coarse in complete)})
    in_complete = np.isin(labels, list(complete))
    for tag, mask in (("ALL_complete", evaluable & in_complete),
                      ("ALL_evaluable", evaluable)):
        rows.append({"dataset": ds, "source": source, "coarse_class": tag,
                     "n": int(mask.sum()), "n_violation": int(violated[mask].sum()),
                     "violation_rate": float(violated[mask].mean()) if mask.any() else np.nan,
                     "n_rules_used": np.nan, "n_rules_total": np.nan,
                     "rule_complete": int(tag == "ALL_complete")})
    return pd.DataFrame(rows), evaluable, violated


def spatial_by_image(frame, labels, radius, min_cells=150):
    """Mean same-type neighbourhood enrichment over images, weighted by cells."""
    acc = {}
    tmp = frame[["image", "centroid_x", "centroid_y"]].copy()
    tmp["_lab"] = np.asarray(labels, dtype=object)
    for img, sub in tmp.groupby("image"):
        if len(sub) < min_cells or sub["centroid_x"].isna().any():
            continue
        coords = sub[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
        tab = MC.neighbourhood_enrichment(coords, sub["_lab"].to_numpy(), sorted(set(sub["_lab"])), radius)
        for _, r in tab.iterrows():
            a = acc.setdefault(r["type"], {"n": 0, "w": 0.0, "w2": 0.0})
            a["n"] += int(r["n"])
            a["w"] += float(r["enrichment"]) * int(r["n"])
            a["w2"] += int(r["n"])
    rows = [{"type": t, "n": v["n"], "enrichment": v["w"] / v["w2"]}
            for t, v in acc.items() if v["w2"] > 0]
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def median_nn_radius(frame, cap=4000, seed=0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    ds = []
    for img, sub in frame.groupby("image"):
        if len(sub) < 50 or sub["centroid_x"].isna().any():
            continue
        xy = sub[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
        if xy.shape[0] > cap:
            xy = xy[rng.choice(xy.shape[0], cap, replace=False)]
        tree = cKDTree(xy)
        d, _ = tree.query(xy, k=2)
        ds.append(np.median(d[:, 1]))
    return 2.5 * float(np.median(ds)) if ds else 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    clfs = [c for c in args.clfs.split(",") if c]
    open(os.path.join(paths.WORK, "r06_report.txt"), "w").close()

    rng = np.random.default_rng(0)
    audit, selective, percell, spatial = [], [], [], []

    for ds in paths.MANUAL_DATASETS:
        log("=" * 78)
        log("%s" % ds)
        log("=" * 78)
        g, markers = D.gold(ds, drop_unidentified=True)
        g = g.reset_index(drop=True)
        frame, chans = canonical_frame(g, markers)
        pos, thr = MC.positivity(frame, chans)
        groups, failed = build_groups(ds, "or")
        groups_sens, _ = build_groups(ds, "and")
        missing = sorted({i for rs in groups.values() for r in rs for i in r.idents
                          if i not in pos})
        log("  markers=%d  channels with a positivity call=%d  rules compiled=%d"
            % (len(markers), sum(1 for c in chans if np.isfinite(thr.get(c, np.inf))),
               sum(len(v) for v in groups.values())))
        log("  coarse classes with an evaluable rule: %s"
            % ", ".join(sorted(c for c, rs in groups.items()
                               if any(r.idents <= set(pos) for r in rs))))
        if missing:
            log("  rule channels absent from this panel (rule dropped): %s"
                % ", ".join(missing))

        # ---- 1/2. expert and depot labels
        complete = complete_classes(groups, pos)
        log("  classes with a fully evaluable rule (';' read as OR): %s"
            % ", ".join(sorted(complete)))
        for reading, gr in (("or", groups), ("and", groups_sens)):
            comp = complete_classes(gr, pos)
            t, ev, _ = rate_table(ds, "expert", g["coarse_expert"].to_numpy(), gr, pos, comp)
            t["reading"] = reading
            audit.append(t)
            if "coarse_hq" in g.columns and g["coarse_hq"].notna().any():
                t, _, _ = rate_table(ds, "depot_HQ", g["coarse_hq"].to_numpy(), gr, pos, comp)
                t["reading"] = reading
                audit.append(t)

        labels_pred = np.full(len(g), None, dtype=object)
        conf = np.full(len(g), np.nan)
        for clf in clfs:
            pred = load_predictions(ds, clf)
            lab = np.array([pred.get((str(i), int(c)), (None, np.nan))[0]
                            for i, c in zip(g["image"], g["cellID"])], dtype=object)
            cf = np.array([pred.get((str(i), int(c)), (None, np.nan))[1]
                           for i, c in zip(g["image"], g["cellID"])], dtype=np.float64)
            ok = np.array([v is not None for v in lab])
            log("  %-7s predictions matched to %d/%d expert cells" % (clf, ok.sum(), len(g)))
            for reading, gr in (("or", groups), ("and", groups_sens)):
                t, _, _ = rate_table(ds, "pred_%s" % clf, lab[ok], gr, pos,
                                     complete_classes(gr, pos))
                t["reading"] = reading
                audit.append(t)
            if clf == clfs[0]:
                labels_pred, conf = lab, cf

        # ---- 3. does a violation come with a label error?
        for clf in clfs:
            pred = load_predictions(ds, clf)
            lab = np.array([pred.get((str(i), int(c)), (None, np.nan))[0]
                            for i, c in zip(g["image"], g["cellID"])], dtype=object)
            ok = np.array([v is not None for v in lab])
            ev, vi = evaluate_source(lab[ok], groups, pos)
            right = lab[ok] == g["coarse_expert"].to_numpy()[ok]
            for group, m in (("prediction_correct", right), ("prediction_wrong", ~right)):
                mm = ev & m
                if mm.sum() == 0:
                    continue
                percell.append({"dataset": ds, "clf": clf, "group": group,
                                "n": int(mm.sum()),
                                "violation_rate": float(vi[mm].mean())})

        # ---- 4. selective: accepted vs deferred, using the stored thresholds
        met = pd.read_csv(os.path.join(OUT, "r03_metrics.csv"))
        for clf in clfs:
            pred = load_predictions(ds, clf)
            lab = np.array([pred.get((str(i), int(c)), (None, np.nan))[0]
                            for i, c in zip(g["image"], g["cellID"])], dtype=object)
            cf = np.array([pred.get((str(i), int(c)), (None, np.nan))[1]
                           for i, c in zip(g["image"], g["cellID"])], dtype=np.float64)
            # evaluated over the whole frame: cells without a prediction simply
            # never match a coarse class and drop out of the evaluable mask
            ev, vi = evaluate_source(lab, groups, pos, restrict=complete)
            for fold in range(5):
                unit = "within_%s_fold%d" % (ds, fold)
                sub = met[(met.unit == unit) & (met.clf == clf) & (met.method == "crc")]
                if sub.empty:
                    continue
                files = os.path.join(PRED_DIR, "within_%s_fold%d__%s.npz" % (ds, fold, clf))
                d = np.load(files, allow_pickle=True)
                classes = np.asarray(d["classes"], dtype=object)
                probs = np.asarray(d["test_probs"], dtype=np.float64)
                score = 1.0 - probs.max(1)
                key = {}
                for j, (img, cid) in enumerate(zip(d["test_img"], d["test_cellid"])):
                    key[(str(img), int(cid))] = j
                idx = np.array([key.get((str(i), int(c)), -1)
                                for i, c in zip(g["image"], g["cellID"])])
                here = idx >= 0
                for alpha in alphas:
                    s_a = sub[np.isclose(sub.alpha, alpha)]
                    for rep, grp in s_a.groupby("repeat"):
                        lam = float(grp["lambda"].iloc[0])
                        if not np.isfinite(lam):
                            continue
                        acc = np.zeros(len(g), dtype=bool)
                        acc[here] = score[idx[here]] <= lam
                        ev_f, vi_f = ev[here], vi[here]
                        for name, m in (("accepted", acc[here]), ("deferred", ~acc[here])):
                            mm = m & ev_f
                            if mm.sum() < 20:
                                continue
                            selective.append({"dataset": ds, "clf": clf, "alpha": alpha,
                                              "repeat": int(rep), "group": name,
                                              "n": int(mm.sum()),
                                              "violation_rate": float(vi_f[mm].mean())})

        # ---- 5. downstream spatial statistic
        radius = median_nn_radius(g)
        exp_sp = spatial_by_image(g, g["coarse_expert"].to_numpy(), radius)
        exp_sp["dataset"], exp_sp["source"], exp_sp["radius"] = ds, "expert", radius
        spatial.append(exp_sp)
        ok = np.array([v is not None for v in labels_pred])
        if ok.sum() > 0:
            pr = spatial_by_image(g[ok].reset_index(drop=True), np.asarray(labels_pred)[ok], radius)
            pr["dataset"], pr["source"], pr["radius"] = ds, "pred_%s" % clfs[0], radius
            spatial.append(pr)
            j = exp_sp.merge(pr, on="type", how="inner", suffixes=("_expert", "_pred"))
            if len(j):
                log("  neighbourhood enrichment changes most for: %s"
                    % ", ".join((j.assign(d=(j.enrichment_pred - j.enrichment_expert)
                                            .abs() / j.enrichment_expert)
                                  .sort_values("d", ascending=False)["type"].head(4))))
        log("  median-nn radius %.1f px" % radius)

    for name, df in (("r06_marker_audit.csv", pd.concat(audit, ignore_index=True)),
                     ("r06_selective.csv", pd.DataFrame(selective)),
                     ("r06_violations.csv", pd.DataFrame(percell)),
                     ("r06_spatial.csv", pd.concat(spatial, ignore_index=True))):
        df.to_csv(os.path.join(OUT, name), index=False)
        log("wrote %s (%d rows)" % (name, len(df)))

    a = pd.concat(audit, ignore_index=True)
    log("")
    for reading in ("or", "and"):
        log("")
        log("SUMMARY [reading ';' as %s] -- violation rate by label source, classes" % reading.upper())
        log("           whose published rules are fully evaluable on this panel")
        log(a[(a.coarse_class == "ALL_complete") & (a.reading == reading)]
            .pivot_table(index="dataset", columns="source", values="violation_rate")
            .round(3).to_string())
    log("")
    log("Audit coverage: expert cells whose label has a fully evaluable rule")
    log(a[(a.source == "expert") & (a.coarse_class.isin(["ALL_complete", "ALL_evaluable"]))]
        .pivot_table(index="dataset", columns="coarse_class", values="n").to_string())
    if selective:
        sl = pd.DataFrame(selective)
        log("")
        log("SUMMARY -- violation rate of accepted vs deferred cells (crc, pooled)")
        log(sl.pivot_table(index=["dataset", "clf"], columns="group",
                           values="violation_rate").round(3).to_string())


if __name__ == "__main__":
    main()





