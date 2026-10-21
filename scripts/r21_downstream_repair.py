"""R21 -- does the certificate's accepted set repair the DOWNSTREAM distortion
that annotation error causes in a spatial statistic?

Motivation.  R06 showed that the same-type neighbourhood enrichment computed
from predicted labels differs from the one computed from expert labels, and
that the risk controller's accepted cells look more consistent with the
depot's published marker rules than the deferred ones.  The obvious follow-up
question -- and this script -- is whether the accepted set also repairs the
*downstream* spatial statistic.

Pre-specified design (fixed before any result was looked at)
-----------------------------------------------------------
For each cohort D, classifier C and error budget alpha in {0.05, 0.10, 0.20},
on the ``within`` (cross-field-of-view) split, over the pooled replicates r15
pools over:

  S      = cells matched to a cached prediction AND whose expert coarse label
           has a fully evaluable marker rule under the repaired reading, i.e.
           r15's ``ALL_strict`` evaluability mask ev_s.
  S_acc  = the accepted subset of S, accepted exactly as r15 does it: score
           s = 1 - max(class prob) from work/predictions/within_*_fold*__*.npz,
           accepted when s <= lambda for the cell's own fold, lambda read from
           experiments/r03_metrics.csv (method == "crc", one row per
           unit/repeat/alpha/clf).

  Four labelings, each on its own restricted cell set, all through the SAME
  spatial routine (r06's ``spatial_by_image``, reproduced verbatim below):
      expert_all : expert coarse_expert labels on S
      pred_all   : predicted labels               on S
      expert_acc : expert coarse_expert labels on S_acc
      pred_acc   : predicted labels               on S_acc

  For every cell type t present in BOTH the all-join and the acc-join:
      dist_all(t) = |enrichment(pred_all, t) - enrichment(expert_all, t)|
      dist_acc(t) = |enrichment(pred_acc, t) - enrichment(expert_acc, t)|
  plus the signed differences and the prediction-side n's.

  PRIMARY STATISTIC (pre-specified):
      delta = [ sum_t w_t dist_all(t) / sum_t w_t ]
            - [ sum_t u_t dist_acc(t) / sum_t u_t ]
  over the types t in both joins, with w_t = n(pred_all, t) and
  u_t = n(pred_acc, t) (the prediction-side cell counts).  Reported per cohort
  and pooled over cohorts; pooling is a single cell-weighted mean over all
  (cohort, classifier, alpha, type) contributions.

  95% CI by resampling WHOLE IMAGES with replacement (2000 replicates; cells
  inside one image are strongly spatially correlated, so the image is the
  honest resampling unit).  Resampling is done separately inside each cohort,
  so clusters (images) are nested in cohorts.

  Secondary paired statistic: median over the same types of
  (dist_all(t) - dist_acc(t)), same bootstrap.

  Verdict rule (pre-specified): the pooled CI decides.
      CI entirely above 0 -> "distortion shrinks among accepted cells"
      CI entirely below 0 -> "distortion worsens among accepted cells"
      CI contains 0       -> "no detectable change"
      statistic undefined -> "not estimable (state why)"

Documented departures from the letter of the recipe (all forced or explicit)
---------------------------------------------------------------------------
1. Replicates.  lambda is read at repeat 0 (r15's first replicate).  For
   Breast, GVHD and CRC the whole lambda vector is identical across the six
   repeats, so repeat 0 *is* r15's pooled answer there; only Melanoma has
   replicate-varying lambda, and r15's pooling over its distinct configurations
   cannot define one cell set S_acc.  A per-repeat robustness table for those
   strata is reported.
2. Per-image tables instead of per-image aggregation.  ``spatial_by_image``
   cell-weight-averages per-image enrichment rows; we materialise those rows
   and apply r06's two image rules (drop an image with fewer than ``min_cells``
   restricted cells, drop an image with any missing centroid) at aggregation
   time.  The weighted mean is algebraically identical to r06's, the bootstrap
   can reuse the rows, and min_cells can be varied without recomputation.
   Primary min_cells is r06's 150; a 50 sensitivity is reported.
3. NaN enrichment.  r06 propagates a NaN ``enrichment`` (possible when a type's
   expected neighbour count is 0) into the whole type aggregate.  We instead
   drop the offending (image, type) row and report how many were dropped.
   The count is printed in the log and in the report.
4. No colour, statistic variant or cohort subset is chosen on the basis of
   results.

Outputs: experiments/r21_downstream.csv, work/r21_report.txt,
figures/fig9_downstream.pdf/.png.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import data as D, markercheck as MC, paths
from errctl.labels import COARSE_MAP
from errctl.markerlogic import canonical, compile_rules

RESULTS = paths.RESULTS
FIGURES = paths.FIGURES
WORK = paths.WORK
PRED_DIR = os.path.join(WORK, "predictions")
REPORT_PATH = os.path.join(WORK, "r21_report.txt")
CSV_PATH = os.path.join(RESULTS, "r21_downstream.csv")

# ---- every number that was fixed before the first result was inspected
SEED = 20260926          # image-bootstrap seed (printed in the report)
BOOT = 2000              # bootstrap replicates
ALPHAS = (0.05, 0.10, 0.20)
CLFS = ("logreg", "rf", "hgb")
REPEAT = 0               # metrics-table replicate used for lambda
MIN_CELLS = 150          # r06 spatial_by_image default
MIN_CELLS_SENS = 50      # sensitivity only, never the headline
RADIUS_SEED = 0          # r06 median_nn_radius default
RADIUS_CAP = 4000        # r06 median_nn_radius default
POSITIVITY_SEED = 0      # MC.positivity default

SHORT = {"Breast_Jun0524": "Breast", "CRC_Feb2623": "CRC",
         "GVHD_Feb1323": "GVHD", "Melanoma_Oct2522": "Melanoma"}
CLR = {"Breast_Jun0524": "#1f5fa9", "CRC_Feb2623": "#d1495b",
       "GVHD_Feb1323": "#2a9d8f", "Melanoma_Oct2522": "#e6a817"}
MK = {"Breast_Jun0524": "o", "CRC_Feb2623": "s",
      "GVHD_Feb1323": "^", "Melanoma_Oct2522": "D"}
FLOOR = 1e-3             # figure only: log axes clip dist below this

# r06's non-marker blacklist and r15's repair aliases, copied verbatim
NON_MARKERS = {"segmentation_borders", "DRAQ5", "DNA", "Morphology", "nearCD31"}
ALIASES = {"NAKATPASE_hi": "NAKATPASE", "IgaA": "IgA"}


def log(msg=""):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# --------------------------------------------------------------------- rule layer
def canonical_frame(g, markers):
    """Marker columns renamed to canonical channel names, duplicates dropped.

    Copied from r06_marker_audit.py (identical to r15's canonical_frame)."""
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


def rule_table(ds, reading="or", repair=True):
    """Compile the depot's cell-type table into evaluable predicates.
    Copied from r15_marker_sensitivity.py."""
    t = D.celltype_table(ds).copy()
    op = "|" if reading == "or" else "&"
    txt = t["PrimaryMarker"].fillna("").astype(str).str.replace(";", op, regex=False)
    if repair:
        for a, b in ALIASES.items():
            txt = txt.str.replace(r"\b%s\b" % re.escape(a), b, regex=True)
    t["PrimaryMarker"] = txt
    rules, failed = compile_rules(t)
    return rules, failed


def group_rules(rules):
    """(coarse class -> [rule, ...]) using COARSE_MAP.  Copied from r15."""
    groups = {}
    for ct, rule in rules.items():
        coarse = COARSE_MAP.get(ct)
        if not coarse or coarse == "Unidentified":
            continue
        groups.setdefault(coarse, []).append(rule)
    return groups


def evaluate(labels, groups, pos, restrict=None):
    """(evaluable, violated) masks; a class needs at least one usable rule.
    Copied from r15_marker_sensitivity.py."""
    labels = np.asarray(labels, dtype=object)
    ev = np.zeros(labels.size, dtype=bool)
    vi = np.zeros(labels.size, dtype=bool)
    used, missing = {}, {}
    for coarse, rules in groups.items():
        if restrict is not None and coarse not in restrict:
            continue
        missing[coarse] = sorted({i for r in rules for i in r.idents if i not in pos})
        usable = [r for r in rules if r.idents <= set(pos)]
        used[coarse] = [r.text for r in usable]
        m = labels == coarse
        if not m.any() or not usable:
            continue
        ok = np.zeros(int(m.sum()), dtype=bool)
        for r in usable:
            vals = {i: pos[i][m] for i in r.idents}
            ok |= np.asarray(r.evaluate(vals), dtype=bool)
        ev[m] = True
        vi[m] = ~ok
    return ev, vi, used, missing


def evaluability(missing, used):
    """Copied from r15_marker_sensitivity.py."""
    out = {}
    for c in used:
        if used[c] and not missing.get(c):
            out[c] = "complete"
        elif used[c]:
            out[c] = "partial"
        else:
            out[c] = "none"
    return out


# ------------------------------------------------------------------ spatial layer
def median_nn_radius(frame, cap=RADIUS_CAP, seed=RADIUS_SEED):
    """One radius per cohort, from the median nearest-neighbour distance.

    Copied verbatim from r06_marker_audit.py (same cap, same seed, same 2.5x
    convention), so this is the radius r06 used."""
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


def spatial_by_image(frame, labels, radius, min_cells=MIN_CELLS):
    """r06_marker_audit.spatial_by_image, verbatim.

    Kept because it is the routine the analysis claims to use.  The analysis
    itself calls ``per_image_enrichment`` below, which returns the same
    per-image rows that this function cell-weight-averages, so the two give
    bit-identical aggregates for any min_cells (checked in the log)."""
    acc = {}
    tmp = frame[["image", "centroid_x", "centroid_y"]].copy()
    tmp["_lab"] = np.asarray(labels, dtype=object)
    for img, sub in tmp.groupby("image"):
        if len(sub) < min_cells or sub["centroid_x"].isna().any():
            continue
        coords = sub[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
        tab = MC.neighbourhood_enrichment(coords, sub["_lab"].to_numpy(),
                                          sorted(set(sub["_lab"])), radius)
        for _, r in tab.iterrows():
            a = acc.setdefault(r["type"], {"n": 0, "w": 0.0, "w2": 0.0})
            a["n"] += int(r["n"])
            a["w"] += float(r["enrichment"]) * int(r["n"])
            a["w2"] += int(r["n"])
    rows = [{"type": t, "n": v["n"], "enrichment": v["w"] / v["w2"]}
            for t, v in acc.items() if v["w2"] > 0]
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def per_image_enrichment(frame, labels, radius, pool):
    """The per-image rows that r06's ``spatial_by_image`` cell-weight-averages.

    r06 iterates images, skips those with fewer than ``min_cells`` cells or a
    missing centroid, calls MC.neighbourhood_enrichment on the rest, and
    accumulates n and n*enrichment per type.  This performs the same call for
    every image and also returns the two per-image quantities the skip rule
    needs, so the skip can be applied afterwards for any min_cells.

    Returns (n_cells[image], nan_centroid[image], rows) with
    rows = [(image, type, n, observed, expected, enrichment), ...].
    """
    pos_of = {im: i for i, im in enumerate(pool)}
    n_cells = np.zeros(len(pool), dtype=np.int64)
    nan_c = np.zeros(len(pool), dtype=bool)
    rows = []
    tmp = frame[["image", "centroid_x", "centroid_y"]].copy()
    tmp["_lab"] = np.asarray(labels, dtype=object)
    for img, sub in tmp.groupby("image"):
        i = pos_of[str(img)]
        n_cells[i] = len(sub)
        nan_c[i] = bool(sub["centroid_x"].isna().any())
        coords = sub[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
        tab = MC.neighbourhood_enrichment(coords, sub["_lab"].to_numpy(),
                                          sorted(set(sub["_lab"])), radius)
        for _, r in tab.iterrows():
            rows.append((str(img), str(r["type"]), int(r["n"]),
                         float(r["observed"]), float(r["expected"]),
                         float(r["enrichment"])))
    return n_cells, nan_c, rows


def to_matrices(rows, pool_of, voc_of, n_img, n_type):
    """(N, W, n_skipped_nan) from per-image rows; N = counts, W = n*enrichment."""
    N = np.zeros((n_img, n_type), dtype=np.float64)
    W = np.zeros((n_img, n_type), dtype=np.float64)
    n_nan = 0
    for img, t, n, _obs, _exp, e in rows:
        if not np.isfinite(e):
            n_nan += 1
            continue
        N[pool_of[img], voc_of[t]] += n
        W[pool_of[img], voc_of[t]] += e * n
    return N, W, n_nan


def prepare(tabs, min_cells):
    """Fold r06's image rules into the matrices, once per min_cells setting."""
    out = {}
    for name in ("expert_all", "pred_all", "expert_acc", "pred_acc"):
        t = tabs[name]
        incl = ((t["n_cells"] >= min_cells) & (~t["nan"])).astype(np.float64)
        out[name] = (t["N"] * incl[:, None], t["W"] * incl[:, None])
    return out


def strata_components(prep, mult, support=None):
    """Per-type enrichments, dists and weights for one image resample.

    With ``support`` the set of types entering the statistic is frozen at the
    set estimable in the full sample (the observed ``both`` mask) instead of
    being recomputed inside the resample, so a type that loses all its accepted
    cells to the resample drops out only because it has no weight, not because
    the support moved.  This is the fixed-support companion to the pre-specified
    statistic; it exists so that the matched-weight interval can be read
    without the type-dropout artefact of section 5."""
    n_by_t, e_by_t = {}, {}
    for name in ("expert_all", "pred_all", "expert_acc", "pred_acc"):
        n_t = prep[name][0].T @ mult
        w_t = prep[name][1].T @ mult
        e_t = np.divide(w_t, n_t, out=np.full_like(w_t, np.nan), where=n_t > 0)
        n_by_t[name], e_by_t[name] = n_t, e_t
    in_all = np.isfinite(e_by_t["expert_all"]) & np.isfinite(e_by_t["pred_all"])
    in_acc = np.isfinite(e_by_t["expert_acc"]) & np.isfinite(e_by_t["pred_acc"])
    both = in_all & in_acc
    if support is not None:
        both = support & (n_by_t["pred_all"] > 0) & (n_by_t["pred_acc"] > 0)
    return {
        "n": n_by_t, "e": e_by_t,
        "in_all": in_all, "in_acc": in_acc, "both": both,
        "d_all": np.where(in_all, np.abs(e_by_t["pred_all"] - e_by_t["expert_all"]), np.nan),
        "d_acc": np.where(in_acc, np.abs(e_by_t["pred_acc"] - e_by_t["expert_acc"]), np.nan),
        "s_all": np.where(in_all, e_by_t["pred_all"] - e_by_t["expert_all"], np.nan),
        "s_acc": np.where(in_acc, e_by_t["pred_acc"] - e_by_t["expert_acc"], np.nan),
    }


def stratum_stat(comps):
    """delta, paired median, common-weight paired variant and cell counts."""
    T = comps["both"]
    if not T.any():
        return dict(delta=np.nan, med=np.nan, paired=np.nan, paired_acc=np.nan,
                    paired_min=np.nan, cells_all=0.0, cells_acc=0.0, n_types=0)
    wA = comps["n"]["pred_all"][T]
    wC = comps["n"]["pred_acc"][T]
    dA = comps["d_all"][T]
    dC = comps["d_acc"][T]
    wM = np.minimum(wA, wC)
    sa, sc = float(np.sum(wA)), float(np.sum(wC))
    sm = float(np.sum(wM))
    m_all = float(np.sum(wA * dA) / sa) if sa > 0 else np.nan
    m_acc = float(np.sum(wC * dC) / sc) if sc > 0 else np.nan
    return dict(delta=m_all - m_acc, med=float(np.median(dA - dC)),
                paired=float(np.sum(wA * (dA - dC)) / sa) if sa > 0 else np.nan,
                paired_acc=(float(np.sum(wC * (dA - dC)) / sc)
                            if sc > 0 else np.nan),
                paired_min=(float(np.sum(wM * (dA - dC)) / sm)
                            if sm > 0 else np.nan),
                cells_all=sa, cells_acc=sc, n_types=int(T.sum()))


def pool_stats(list_of_comps):
    """Pool strata: one cell-weighted mean over all their T* entries."""
    dA, dC, wA, wC = [], [], [], []
    for comps in list_of_comps:
        T = comps["both"]
        if not T.any():
            continue
        dA.append(comps["d_all"][T])
        dC.append(comps["d_acc"][T])
        wA.append(comps["n"]["pred_all"][T])
        wC.append(comps["n"]["pred_acc"][T])
    if not dA:
        return dict(delta=np.nan, med=np.nan, paired=np.nan, paired_acc=np.nan,
                    paired_min=np.nan, cells_all=0.0, cells_acc=0.0, n_types=0,
                    n_strata=0)
    dA = np.concatenate(dA)
    dC = np.concatenate(dC)
    wA = np.concatenate(wA)
    wC = np.concatenate(wC)
    wM = np.minimum(wA, wC)
    sa, sc = float(np.sum(wA)), float(np.sum(wC))
    sm = float(np.sum(wM))
    m_all = float(np.sum(wA * dA) / sa) if sa > 0 else np.nan
    m_acc = float(np.sum(wC * dC) / sc) if sc > 0 else np.nan
    return dict(delta=m_all - m_acc, med=float(np.median(dA - dC)),
                paired=float(np.sum(wA * (dA - dC)) / sa) if sa > 0 else np.nan,
                paired_acc=(float(np.sum(wC * (dA - dC)) / sc)
                            if sc > 0 else np.nan),
                paired_min=(float(np.sum(wM * (dA - dC)) / sm)
                            if sm > 0 else np.nan),
                cells_all=sa, cells_acc=sc, n_types=int(dA.size),
                n_strata=sum(1 for c in list_of_comps if c["both"].any()))


def entry_table(res, comps_key="comps"):
    """Every (cohort, clf, alpha, type) entry behind the pooled primary mean.

    Purely descriptive: this is what the influence check reads, and it is the
    same set of entries the primary statistic averages over."""
    rows = []
    for st in res["strata"]:
        c = st[comps_key]
        voc = res["data"][st["ds"]]["vocab"]
        for j in np.nonzero(c["both"])[0]:
            w = float(c["n"]["pred_all"][j])
            u = float(c["n"]["pred_acc"][j])
            dA = float(c["d_all"][j])
            dC = float(c["d_acc"][j])
            rows.append(dict(dataset=st["ds"], clf=st["clf"], alpha=st["alpha"],
                             type=voc[j], n_all=w, n_acc=u, dist_all=dA,
                             dist_acc=dC, contrib_all=w * dA, contrib_acc=u * dC,
                             paired_w=w * (dA - dC), paired_u=u * (dA - dC)))
    return pd.DataFrame(rows)


def delta_of_entries(ent):
    """The pooled primary statistic recomputed from a table of entries."""
    if len(ent) == 0:
        return np.nan
    sa = ent["n_all"].sum()
    sc = ent["n_acc"].sum()
    if sa <= 0 or sc <= 0:
        return np.nan
    return float(ent["contrib_all"].sum() / sa - ent["contrib_acc"].sum() / sc)


# ---------------------------------------------------------------- prediction layer
def load_within(ds, clf, gidx, n_cells):
    """Per-fold (cell index, score, fold) and predicted labels.
    Copied from r15_marker_sensitivity._load_within."""
    rows, pred = [], np.full(n_cells, None, dtype=object)
    dup = 0
    for f in range(5):
        p = os.path.join(PRED_DIR, "within_%s_fold%d__%s.npz" % (ds, f, clf))
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        classes = np.asarray(d["classes"], dtype=object)
        probs = np.asarray(d["test_probs"], dtype=np.float64)
        score = 1.0 - probs.max(1)
        lab = classes[probs.argmax(1)]
        for j, (im, cid) in enumerate(zip(d["test_img"], d["test_cellid"])):
            i = gidx.get((str(im), int(cid)), -1)
            if i < 0:
                continue
            if pred[i] is not None:
                dup += 1
            rows.append((i, float(score[j]), f))
            pred[i] = str(lab[j])
    return rows, pred, dup


def lambda_map(met, ds, clf, alpha, repeat=REPEAT):
    """{unit: lambda} for the finite thresholds of one (cohort, clf, alpha)."""
    sub = met[(met["clf"] == clf) & np.isclose(met["alpha"], alpha)
              & met["unit"].str.startswith("within_%s_" % ds)]
    out = {}
    for u, rp, lm in zip(sub["unit"], sub["repeat"], sub["lambda"]):
        if int(rp) == repeat and np.isfinite(lm):
            out[str(u)] = float(lm)
    return out, sub


def repeat_labels(met, ds, clf, alpha):
    """The replicate labels actually present for one (cohort, clf, alpha).

    The metrics table labels repeats 0-9 across all units; a within-split
    (cohort, clf, alpha) cell only ever carries 0-5, so the labels are read off
    the sub-table rather than from the whole file."""
    sub = met[(met["clf"] == clf) & np.isclose(met["alpha"], alpha)
              & met["unit"].str.startswith("within_%s_" % ds)]
    return sorted(int(x) for x in sub["repeat"].unique())


def accepted_mask(idx, sc, fold, lam):
    """r15's acceptance rule: score <= lambda of the cell's own fold."""
    acc = np.zeros(idx.size, dtype=bool)
    for u, l in lam.items():
        f = int(u.rsplit("fold", 1)[1])
        sel = fold == f
        acc[sel] = sc[sel] <= l
    return acc


def build_labeling(frame, labels, radius, pool, pool_of, voc_of, n_type):
    n_cells, nan_c, rows = per_image_enrichment(frame, labels, radius, pool)
    N, W, n_nan = to_matrices(rows, pool_of, voc_of, len(pool), n_type)
    return {"n_cells": n_cells, "nan": nan_c, "N": N, "W": W, "n_nan": n_nan}


def num(x, nd=5):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "na"
    if not np.isfinite(v):
        return "na"
    return ("%%.%df" % nd) % v


# ------------------------------------------------------------------------ driver
def run(alphas, clfs, min_cells, boot, seed, no_figure=False):
    t0 = time.time()
    log("r21_downstream_repair.py")
    log("seeds: image-bootstrap seed=%d (boot=%d), median_nn_radius seed=%d, "
        "MC.positivity seed=%d, lambda repeat=%d"
        % (seed, boot, RADIUS_SEED, POSITIVITY_SEED, REPEAT))
    log("alphas=%s clfs=%s primary min_cells=%d (sensitivity %d)"
        % (list(alphas), list(clfs), min_cells, MIN_CELLS_SENS))

    met = pd.read_csv(os.path.join(RESULTS, "r03_metrics.csv"),
                      usecols=["unit", "clf", "method", "alpha", "repeat",
                               "lambda", "n_test_pool", "n_test_eval"])
    met = met[met["method"] == "crc"]
    log("metrics table experiments/r03_metrics.csv, crc rows: %d" % len(met))
    if int(met["repeat"].max()) + 1 != 6:
        log("NOTE: metrics table repeat count = %d"
            % (int(met["repeat"].max()) + 1))

    data = {}
    for ds in paths.MANUAL_DATASETS:
        g, markers = D.gold(ds, drop_unidentified=True)
        g = g.reset_index(drop=True)
        frame, chans = canonical_frame(g, markers)
        pos, _thr = MC.positivity(frame, chans, seed=POSITIVITY_SEED)
        rules, _failed = rule_table(ds, "or", repair=True)
        groups = group_rules(rules)
        labels = g["coarse_expert"].to_numpy(dtype=object)
        ev, _vi, used, missing = evaluate(labels, groups, pos)
        evl = evaluability(missing, used)
        strict = {c for c in groups if evl.get(c) == "complete"}
        ev_s, _vi_s, _, _ = evaluate(labels, groups, pos, restrict=strict)
        radius = median_nn_radius(g)
        pool = [str(x) for x in sorted(g["image"].astype(str).unique())]
        gidx = {}
        for i, (im, cid) in enumerate(zip(g["image"], g["cellID"])):
            gidx[(str(im), int(cid))] = i
        data[ds] = dict(g=g, pos=pos, groups=groups, labels=labels, ev=ev,
                        strict=strict, ev_s=ev_s, radius=radius, pool=pool,
                        pool_of={im: i for i, im in enumerate(pool)},
                        gidx=gidx, n_clf={})
        dropped = sorted(c for c in groups if c not in strict)
        log("%s: expert cells=%d images=%d radius=%.2f px strict classes=%d"
            % (ds, len(g), len(pool), radius, len(strict)))
        log("    ALL_strict evaluable cells=%d (%.1f%% of expert cells); "
            "classes NOT fully evaluable: %s"
            % (int(ev_s.sum()), 100.0 * ev_s.mean(),
               ", ".join(dropped) if dropped else "-"))

        # --- sanity check: per_image_enrichment aggregation == spatial_by_image
        check = spatial_by_image(g, labels, radius, min_cells)
        nc, nanc, rows = per_image_enrichment(g, labels, radius, pool)
        N, W, _ = to_matrices(rows, data[ds]["pool_of"],
                              {t: i for i, t in enumerate(sorted(set(labels)))},
                              len(pool), len(set(labels)))
        incl = ((nc >= min_cells) & (~nanc)).astype(np.float64)
        nn = N.T @ incl
        ww = W.T @ incl
        ee = np.divide(ww, nn, out=np.full_like(ww, np.nan), where=nn > 0)
        voc = sorted(set(labels))
        worst = 0.0
        for _, r in check.iterrows():
            j = voc.index(r["type"])
            if nn[j] > 0:
                worst = max(worst, abs(ee[j] - float(r["enrichment"])))
        log("    spatial_by_image cross-check (expert labels, all cells): "
            "max |delta enrichment| = %.3e over %d types"
            % (worst, len(check)))

    # ---- predictions, per (cohort, clf)
    for ds in paths.MANUAL_DATASETS:
        dd = data[ds]
        for clf in clfs:
            rows, pred, dup = load_within(ds, clf, dd["gidx"], len(dd["g"]))
            idx = np.array([r[0] for r in rows], dtype=np.int64)
            sc = np.array([r[1] for r in rows], dtype=np.float64)
            fold = np.array([r[2] for r in rows], dtype=np.int64)
            dd["n_clf"][clf] = dict(idx=idx, sc=sc, fold=fold, pred=pred, dup=dup)
        voc = set(str(x) for x in dd["labels"])
        for clf in clfs:
            voc |= set(str(x) for x in dd["n_clf"][clf]["pred"] if x is not None)
        dd["vocab"] = sorted(voc)
        dd["voc_of"] = {t: i for i, t in enumerate(dd["vocab"])}
        dd["n_type"] = len(dd["vocab"])
        log("%s: matched %s expert cells of %d per clf (duplicate matches: %s)"
            % (ds, "/".join(str(dd["n_clf"][c]["idx"].size) for c in clfs),
               len(dd["g"]),
               "/".join(str(dd["n_clf"][c]["dup"]) for c in clfs)))

    # ---- build every stratum
    strata = []
    n_nan_tot = 0
    for ds in paths.MANUAL_DATASETS:
        dd = data[ds]
        g, ev_s, radius = dd["g"], dd["ev_s"], dd["radius"]
        for clf in clfs:
            cc = dd["n_clf"][clf]
            idx, sc, fold, pred = cc["idx"], cc["sc"], cc["fold"], cc["pred"]
            sub = g.iloc[idx].reset_index(drop=True)
            lab_expert = g["coarse_expert"].to_numpy(dtype=object)[idx]
            lab_pred = pred[idx]
            S = ev_s[idx]
            for alpha in alphas:
                lam, _m = lambda_map(met, ds, clf, alpha)
                accm = accepted_mask(idx, sc, fold, lam)
                masks = {"expert_all": S, "pred_all": S,
                         "expert_acc": S & accm, "pred_acc": S & accm}
                labs = {"expert_all": lab_expert, "pred_all": lab_pred,
                        "expert_acc": lab_expert, "pred_acc": lab_pred}
                tabs = {}
                for name in ("expert_all", "pred_all", "expert_acc", "pred_acc"):
                    m = masks[name]
                    fr = sub[m].reset_index(drop=True)
                    tabs[name] = build_labeling(fr, labs[name][m], radius,
                                                dd["pool"], dd["pool_of"],
                                                dd["voc_of"], dd["n_type"])
                    n_nan_tot += tabs[name]["n_nan"]
                st = dict(ds=ds, clf=clf, alpha=alpha, tabs=tabs, S=S, accm=accm,
                          n_S=int(S.sum()), n_acc=int((S & accm).sum()),
                          n_matched=int(idx.size), lam=lam,
                          folds_with_lam=sorted(int(u.rsplit("fold", 1)[1])
                                                for u in lam),
                          n_fold_lam=len(lam))
                st["prep150"] = prepare(tabs, min_cells)
                st["prep_sens"] = prepare(tabs, MIN_CELLS_SENS)
                st["n_img_all"] = int(np.sum((tabs["pred_all"]["n_cells"] >= min_cells)
                                             & ~tabs["pred_all"]["nan"]))
                st["n_img_acc"] = int(np.sum((tabs["pred_acc"]["n_cells"] >= min_cells)
                                             & ~tabs["pred_acc"]["nan"]))
                st["n_img_all_s"] = int(np.sum((tabs["pred_all"]["n_cells"] >= MIN_CELLS_SENS)
                                               & ~tabs["pred_all"]["nan"]))
                st["n_img_acc_s"] = int(np.sum((tabs["pred_acc"]["n_cells"] >= MIN_CELLS_SENS)
                                               & ~tabs["pred_acc"]["nan"]))
                strata.append(st)
    log("built %d strata; NaN-enrichment (image,type) rows skipped: %d"
        % (len(strata), n_nan_tot))

    # ---- observed pass
    ones = {ds: np.ones(len(data[ds]["pool"]), dtype=np.float64)
            for ds in paths.MANUAL_DATASETS}
    for st in strata:
        st["comps"] = strata_components(st["prep150"], ones[st["ds"]])
        st["stat"] = stratum_stat(st["comps"])
        st["comps_s"] = strata_components(st["prep_sens"], ones[st["ds"]])
        st["stat_s"] = stratum_stat(st["comps_s"])
        st["support_prep150"] = st["comps"]["both"].copy()
        st["support_prep_sens"] = st["comps_s"]["both"].copy()

    keys = [(st["ds"], st["clf"], st["alpha"]) for st in strata]
    dss = list(paths.MANUAL_DATASETS)
    by_key = {(st["ds"], st["clf"], st["alpha"]): st for st in strata}

    def observed_levels(prep_key, comps_key):
        pool_c = pool_stats([by_key[k][comps_key] for k in keys])
        ds_c = {ds: pool_stats([by_key[k][comps_key] for k in keys
                                if k[0] == ds]) for ds in dss}
        return pool_c, ds_c

    obs_pool, obs_ds = observed_levels("prep150", "comps")
    obs_pool_s, obs_ds_s = observed_levels("prep_sens", "comps_s")

    # ---- bootstrap over images, nested in cohorts
    def boot_pass(prep_key, mc, seed):
        rng = np.random.default_rng(seed)
        out = {ds: {"delta": np.full(boot, np.nan), "med": np.full(boot, np.nan),
                    "paired": np.full(boot, np.nan),
                    "paired_acc": np.full(boot, np.nan),
                    "paired_min": np.full(boot, np.nan)} for ds in dss}
        for ds in dss:
            out[ds]["paired_acc_fix"] = np.full(boot, np.nan)
        out["__pool__"] = {"delta": np.full(boot, np.nan),
                           "med": np.full(boot, np.nan),
                           "paired": np.full(boot, np.nan),
                           "paired_acc": np.full(boot, np.nan),
                           "paired_min": np.full(boot, np.nan),
                           "paired_acc_fix": np.full(boot, np.nan)}
        for b in range(boot):
            mults = {ds: rng.integers(0, len(data[ds]["pool"]),
                                      len(data[ds]["pool"])).astype(np.float64)
                     for ds in dss}
            all_c, all_f = [], []
            for st in strata:
                all_c.append(strata_components(st[prep_key], mults[st["ds"]]))
                all_f.append(strata_components(st[prep_key], mults[st["ds"]],
                                               support=st["support_" + prep_key]))
            for ds in dss:
                sel = [all_c[i] for i, st in enumerate(strata) if st["ds"] == ds]
                r = pool_stats(sel)
                out[ds]["delta"][b] = r["delta"]
                out[ds]["med"][b] = r["med"]
                out[ds]["paired"][b] = r["paired"]
                out[ds]["paired_acc"][b] = r["paired_acc"]
                out[ds]["paired_min"][b] = r["paired_min"]
            for ds in dss:
                sel_f = [all_f[i] for i, st in enumerate(strata) if st["ds"] == ds]
                out[ds]["paired_acc_fix"][b] = pool_stats(sel_f)["paired_acc"]
            r = pool_stats(all_c)
            out["__pool__"]["delta"][b] = r["delta"]
            out["__pool__"]["med"][b] = r["med"]
            out["__pool__"]["paired"][b] = r["paired"]
            out["__pool__"]["paired_acc"][b] = r["paired_acc"]
            out["__pool__"]["paired_min"][b] = r["paired_min"]
            out["__pool__"]["paired_acc_fix"][b] = pool_stats(all_f)["paired_acc"]
        return out

    log("bootstrapping %d image resamples at min_cells=%d ..." % (boot, min_cells))
    boot_res = boot_pass("prep150", min_cells, seed)
    log("  done (%.1fs elapsed)" % (time.time() - t0))
    log("bootstrapping %d image resamples at min_cells=%d (sensitivity) ..."
        % (boot, MIN_CELLS_SENS))
    boot_res_s = boot_pass("prep_sens", MIN_CELLS_SENS, seed)
    log("  done (%.1fs elapsed)" % (time.time() - t0))

    def ci(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan, np.nan, 0
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)),
                int(v.size))

    # ---- repeat robustness where lambda varies across replicates
    rep_rows = []
    for ds in dss:
        dd = data[ds]
        for clf in clfs:
            cc = dd["n_clf"][clf]
            for alpha in alphas:
                reps = repeat_labels(met, ds, clf, alpha)
                cfgs = {}
                for rp in reps:
                    lam, _ = lambda_map(met, ds, clf, alpha, int(rp))
                    cfgs.setdefault(tuple(sorted((u, round(l, 12))
                                                 for u, l in lam.items())),
                                    int(rp))
                if len(cfgs) <= 1:
                    continue
                g = dd["g"]
                sub = g.iloc[cc["idx"]].reset_index(drop=True)
                lab_expert = g["coarse_expert"].to_numpy(dtype=object)[cc["idx"]]
                lab_pred = cc["pred"][cc["idx"]]
                S = dd["ev_s"][cc["idx"]]
                vals = []
                for key, rp in sorted(cfgs.items(), key=lambda kv: kv[1]):
                    lam, _ = lambda_map(met, ds, clf, alpha, int(rp))
                    accm = accepted_mask(cc["idx"], cc["sc"], cc["fold"], lam)
                    tabs = {}
                    for name, m, lab in (("expert_all", S, lab_expert),
                                         ("pred_all", S, lab_pred),
                                         ("expert_acc", S & accm, lab_expert),
                                         ("pred_acc", S & accm, lab_pred)):
                        fr = sub[m].reset_index(drop=True)
                        tabs[name] = build_labeling(fr, lab[m], dd["radius"],
                                                    dd["pool"], dd["pool_of"],
                                                    dd["voc_of"], dd["n_type"])
                    c = strata_components(prepare(tabs, min_cells),
                                          np.ones(len(dd["pool"])))
                    r = stratum_stat(c)
                    vals.append((int(rp), r["delta"], r["n_types"],
                                 int((S & accm).sum())))
                rep_rows.append(dict(ds=ds, clf=clf, alpha=alpha,
                                     n_configs=len(cfgs), vals=vals))
    log("strata whose lambda varies across repeats: %d" % len(rep_rows))

    # ---- verdict
    lo, hi, nb = ci(boot_res["__pool__"]["delta"])
    if not np.isfinite(obs_pool["delta"]) or nb == 0:
        verdict = ("VERDICT: not estimable (no cohort/classifier/budget had both "
                   "an all-side and an accepted-side type join)")
        vkind = "not_estimable"
    elif lo > 0:
        verdict = "VERDICT: distortion shrinks among accepted cells"
        vkind = "shrinks"
    elif hi < 0:
        verdict = "VERDICT: distortion worsens among accepted cells"
        vkind = "worsens"
    else:
        verdict = "VERDICT: no detectable change"
        vkind = "null"

    res = dict(strata=strata, data=data, alphas=list(alphas), clfs=list(clfs),
               keys=keys, by_key=by_key, dss=dss, obs_pool=obs_pool,
               obs_ds=obs_ds, obs_pool_s=obs_pool_s, obs_ds_s=obs_ds_s,
               boot=boot_res, boot_s=boot_res_s, ci=ci, min_cells=min_cells,
               rep_rows=rep_rows, n_nan_tot=n_nan_tot, verdict=verdict,
               vkind=vkind, boot_lo=lo, boot_hi=hi, boot_nb=nb, seed=seed,
               boot_n=boot)
    res["entries"] = entry_table(res, "comps")
    res["entries_s"] = entry_table(res, "comps_s")
    return res


# ------------------------------------------------------------------------ output
def csv_rows(res):
    rows = []
    for st in res["strata"]:
        c = st["comps"]
        voc = res["data"][st["ds"]]["vocab"]
        nA, nC = c["n"]["pred_all"], c["n"]["pred_acc"]
        for j, t in enumerate(voc):
            if not (c["in_all"][j] or c["in_acc"][j]):
                continue
            if nA[j] <= 0 and nC[j] <= 0:
                continue
            scope = ("both" if (c["in_all"][j] and c["in_acc"][j])
                     else ("all_only" if c["in_all"][j] else "acc_only"))
            rows.append(dict(
                dataset=st["ds"], clf=st["clf"], alpha=st["alpha"], type=t,
                scope=scope, radius=res["data"][st["ds"]]["radius"],
                min_cells=res["min_cells"],
                nS=st["n_S"], nS_acc=st["n_acc"],
                accept_frac=(float(st["n_acc"]) / float(st["n_S"])
                             if st["n_S"] else np.nan),
                n_img_all=st["n_img_all"], n_img_acc=st["n_img_acc"],
                n_folds_with_lambda=st["n_fold_lam"],
                n_all_expert=c["n"]["expert_all"][j],
                n_all_pred=nA[j],
                e_all_expert=c["e"]["expert_all"][j],
                e_all_pred=c["e"]["pred_all"][j],
                dist_all=c["d_all"][j], signed_all=c["s_all"][j],
                n_acc_expert=c["n"]["expert_acc"][j],
                n_acc_pred=nC[j],
                e_acc_expert=c["e"]["expert_acc"][j],
                e_acc_pred=c["e"]["pred_acc"][j],
                dist_acc=c["d_acc"][j], signed_acc=c["s_acc"][j],
                delta=np.nan, median_paired=np.nan,
                paired_common_weight=np.nan, cells_all=np.nan,
                cells_acc=np.nan, n_types_both=np.nan))
        s = st["stat"]
        rows.append(dict(
            dataset=st["ds"], clf=st["clf"], alpha=st["alpha"], type="ALL",
            scope="summary_weighted", radius=res["data"][st["ds"]]["radius"],
            min_cells=res["min_cells"], nS=st["n_S"], nS_acc=st["n_acc"],
            accept_frac=(float(st["n_acc"]) / float(st["n_S"])
                         if st["n_S"] else np.nan),
            n_img_all=st["n_img_all"], n_img_acc=st["n_img_acc"],
            n_folds_with_lambda=st["n_fold_lam"],
            n_all_expert=np.nan, n_all_pred=np.nan, e_all_expert=np.nan,
            e_all_pred=np.nan, dist_all=np.nan, signed_all=np.nan,
            n_acc_expert=np.nan, n_acc_pred=np.nan, e_acc_expert=np.nan,
            e_acc_pred=np.nan, dist_acc=np.nan, signed_acc=np.nan,
            delta=s["delta"], median_paired=s["med"],
            paired_common_weight=s["paired"],
            paired_acc_common_weight=s["paired_acc"],
            paired_min_common_weight=s["paired_min"],
            cells_all=s["cells_all"],
            cells_acc=s["cells_acc"], n_types_both=s["n_types"]))
    return pd.DataFrame(rows)


def _m_all(ent):
    return ent["contrib_all"].sum() / ent["n_all"].sum() if len(ent) else np.nan


def _m_acc(ent):
    return ent["contrib_acc"].sum() / ent["n_acc"].sum() if len(ent) else np.nan


def write_report(res, path):
    dss = res["dss"]
    S = res["strata"]
    ob = res["obs_pool"]
    lo, hi, nb = res["boot_lo"], res["boot_hi"], res["boot_nb"]
    mpa = ob["paired_acc"]
    mpla, mpha, _ = res["ci"](res["boot"]["__pool__"]["paired_acc"])
    mpm = ob["paired_min"]
    mplm, mphm, _ = res["ci"](res["boot"]["__pool__"]["paired_min"])
    R = []
    A = R.append
    A(res["verdict"])
    A("")
    A("=" * 78)
    A("R21 -- downstream spatial distortion: does the certificate's accepted")
    A("        set repair what annotation error does to same-type neighbourhood")
    A("        enrichment?  (supporting analysis, not a theorem)")
    A("=" * 78)
    A("")
    A("HEADLINE")
    A("-" * 78)
    if res["vkind"] == "not_estimable":
        A("The pre-specified statistic could not be formed: no cohort/classifier/")
        A("budget had cell types present in BOTH the all-cell join and the")
        A("accepted-cell join.")
    else:
        A("Pre-specified primary statistic (pooled over cohorts):")
        A("    mean_cellweighted(dist_all) - mean_cellweighted(dist_acc)")
        A("      = %s    95%% CI [%s, %s]  (%d/%d bootstrap replicates finite)"
          % (num(ob["delta"]), num(lo), num(hi), nb, res["boot_n"]))
        A("It rests on %d accepted-side cells and %d all-side cells over %d"
          % (int(round(ob["cells_acc"])), int(round(ob["cells_all"])),
             ob["n_types"]))
        A("(cohort, classifier, budget, type) contributions in %d strata."
          % ob["n_strata"])
        if res["vkind"] == "shrinks":
            A("The interval lies entirely above zero, so on this design the")
            A("downstream distortion is smaller among accepted cells.")
        elif res["vkind"] == "worsens":
            A("The interval lies entirely below zero: the downstream distortion")
            A("is LARGER among the cells the certificate accepts.  The accepted")
            A("set does NOT repair the downstream statistic here.")
        else:
            A("The interval contains zero: no detectable change in the")
            A("downstream distortion among accepted cells.")
        if np.isfinite(mpa):
            A("")
            A("Composition-matched reading (one weight per type shared by BOTH")
            A("scopes, the honest companion to the confound of section 4; see 6):")
            A("    sum_t w_t (dist_all(t) - dist_acc(t)) / sum_t w_t with")
            A("    w_t = n(pred_acc, t)     ->  %s   95%% CI [%s, %s]"
              % (num(mpa), num(mpla), num(mpha)))
            A("    w_t = min(n_all, n_acc)  ->  %s   95%% CI [%s, %s]"
              % (num(mpm), num(mplm), num(mphm)))
    A("")
    A("Read this next to section 3 and section 4.  Where the accepted side is")
    A("estimable at all, it is a much smaller and differently composed sample")
    A("(fewer cells, fewer images, fewer types surviving r06's 150-cell rule),")
    A("so the comparison is not a like-for-like swap of one population for")
    A("another.  The cohort breakdown is in section 5; whether the statistic")
    A("is even defined per cohort is in section 3.")
    A("")
    A("1. WHAT IS TESTED, AND WHAT WAS PRE-SPECIFIED")
    A("-" * 78)
    A("Cohorts: %s" % ", ".join(dss))
    A("Splits: within (cross-field-of-view) test cells only.")
    A("Classifiers: %s      Budgets alpha: %s"
      % (", ".join(res["clfs"]), ", ".join("%.2f" % a for a in res["alphas"])))
    A("For each (cohort, classifier, alpha):")
    A("  S      = matched to a cached prediction AND marker-evaluable under the")
    A("           r15 ALL_strict (repaired, ';' as OR) ruleset.")
    A("  S_acc  = S cells with score 1-max(prob) <= lambda(fold), lambda from")
    A("           experiments/r03_metrics.csv (method crc, repeat %d)." % REPEAT)
    A("  Four labelings, all through r06's spatial_by_image:")
    A("           expert_all / pred_all on S; expert_acc / pred_acc on S_acc.")
    A("  dist_all(t) = |enrichment(pred_all,t) - enrichment(expert_all,t)|")
    A("  dist_acc(t) = |enrichment(pred_acc,t) - enrichment(expert_acc,t)|")
    A("  PRIMARY   delta = (cell-weighted mean of dist_all) - (cell-weighted")
    A("            mean of dist_acc) over types t in BOTH joins, weights =")
    A("            the prediction-side n(t) on each side.")
    A("  CI        95%% percentile interval over %d bootstrap replicates that"
      % res["boot_n"])
    A("            resample WHOLE IMAGES with replacement inside each cohort.")
    A("  SECONDARY median over the same types of (dist_all(t) - dist_acc(t)),")
    A("            same bootstrap; and the composition-matched paired variants")
    A("            sum_t w_t (dist_all - dist_acc) / sum_t w_t with ONE weight")
    A("            w_t shared by both scopes, taken as n(pred_acc, t), as")
    A("            min(n(pred_all,t), n(pred_acc,t)), and -- for continuity with")
    A("            earlier reports -- as n(pred_all, t).  All three are")
    A("            bootstrapped exactly like the primary statistic.")
    A("  Verdict   pre-specified from the pooled CI: entirely above 0 ->")
    A("            'distortion shrinks among accepted cells'; entirely below 0")
    A("            -> 'distortion worsens among accepted cells'; containing 0")
    A("            -> 'no detectable change'.")
    A("Seeds / fixed choices (all printed in work/r21_log.txt):")
    A("  image-bootstrap seed = %d, replicates = %d" % (res["seed"], res["boot_n"]))
    A("  median_nn_radius seed = %d, cap = %d, radius = 2.5 x median NN distance"
      % (RADIUS_SEED, RADIUS_CAP))
    A("  MC.positivity seed = 0 ; MC.neighbourhood_enrichment seed = 0 (its")
    A("  subsampling branch needs >200000 cells per image and never fires here)")
    A("  lambda replicate = %d ; primary min_cells = %d ; sensitivity min_cells = %d"
      % (REPEAT, res["min_cells"], MIN_CELLS_SENS))
    A("")
    A("2. ONE RADIUS PER COHORT (reused from r06's median_nn_radius)")
    A("-" * 78)
    for ds in dss:
        dd = res["data"][ds]
        A("  %-16s %8.2f px  (%d expert cells, %d images)"
          % (ds, dd["radius"], len(dd["g"]), len(dd["pool"])))
    A("")
    A("3. SAMPLE SIZES AT EACH STEP (is a change just a smaller sample?)")
    A("-" * 78)
    A("  cohort  clf     alpha   |S|    |S_acc|  acc%   imgs(S/S_acc)  types(T*)")
    for st in S:
        A("  %-7s %-7s %.2f  %6d  %6d  %5.1f%%   %3d / %3d      %3d"
          % (SHORT[st["ds"]], st["clf"], st["alpha"], st["n_S"], st["n_acc"],
             100.0 * st["n_acc"] / max(st["n_S"], 1), st["n_img_all"],
             st["n_img_acc"], st["stat"]["n_types"]))
    A("")
    A("  Legend: |S| = matched and ALL_strict-evaluable cells; |S_acc| = of those,")
    A("  accepted at the given budget; imgs = images with at least %d cells in the"
      % res["min_cells"])
    A("  restricted set (r06 discards the rest); types(T*) = types present in both")
    A("  joins, i.e. the types the paired distances are computed on.")
    A("")
    first = res["clfs"][0]
    for ds in dss:
        dd = res["data"][ds]
        mine = [st for st in S if st["ds"] == ds and st["clf"] == first]
        A("  %-9s expert cells %6d, matched %6d, |S| %6d, |S_acc| by alpha=%.2f: %s"
          % (SHORT[ds], len(dd["g"]), dd["n_clf"][first]["idx"].size,
             int(mine[0]["n_S"]) if mine else -1,
             res["alphas"][0],
             " ".join("%.2f:%d" % (st["alpha"], st["n_acc"]) for st in mine)))
    A("  (|S| is the same for every classifier, since every expert cell is matched;")
    A("   the numbers above are for classifier '%s' only -- the other classifiers"
      % first)
    A("   are in the per-stratum table and in the CSV.)")
    A("")
    A("  lambda availability (folds with a finite threshold out of 5):")
    for st in S:
        if st["clf"] != res["clfs"][0]:
            continue
        A("    %-9s %-7s alpha=%.2f  folds=%s  lambda=%s"
          % (SHORT[st["ds"]], st["clf"], st["alpha"],
             st["folds_with_lam"],
             ",".join("%.4f" % st["lam"][u] for u in sorted(st["lam"]))
             if st["lam"] else "none finite"))
    A("  (rows shown for one classifier; the other classifiers are in the CSV and")
    A("   in the log.  A fold without a finite lambda is accepted nowhere, exactly")
    A("   as r15 treats it.)")
    A("")
    A("4. THE CONFOUND, STATED PLAINLY")
    A("-" * 78)
    A("Restricting to accepted cells changes cell composition, the effective")
    A("sample size, the number of images surviving the 150-cell rule and the set")
    A("of types that survive into T*.  All of that moves an enrichment estimate")
    A("even for a hypothetical perfect labeller, so dist_all and dist_acc are NOT")
    A("two measurements of the same quantity on the same population.")
    A("")
    A("What the paired construction does control: inside each scope, pred and")
    A("expert labels are scored on exactly the same cells, so any composition or")
    A("sample-size effect that acts through the *cell set* cancels from the")
    A("difference.  For a perfect labeller both distances are exactly zero in both")
    A("scopes.  The comparison is therefore not the classic 'different cells,")
    A("different enrichment' artefact.")
    A("")
    A("What it does NOT control:")
    A("  (i) selection on confidence.  Accepted cells are not a random subsample;")
    A("      they are the cells the classifier already agrees with, which sit in")
    A("      dense homogeneous neighbourhoods.  The damage an error does to a")
    A("      spatial statistic depends on local composition, so the residual can")
    A("      differ between scopes for reasons other than 'errors removed'.")
    A("  (ii) T* is not the same set of types in the two scopes (see section 7);")
    A("      a crop of T* can move the mean even with unchanged per-type errors.")
    A("  (iii) noise.  A smaller sample inflates an estimated absolute difference")
    A("      (Jensen).  dist_acc is therefore biased UPWARD relative to a")
    A("      noiseless reference, which works AGAINST finding shrinkage.")
    A("  (iv) image attrition.  On the accepted side the 150-cell rule can throw")
    A("      away most or all images; cohort-level numbers then rest on a handful")
    A("      of fields of view, and the bootstrap CI is correspondingly wide.")
    A("The control is adequate for (i)-(iii) as an estimate of the *paired*")
    A("quantity; it is NOT adequate to claim that accepting cells makes the")
    A("biological readout comparable across cohorts, because (iv) can make some")
    A("cohorts simply unestimable.")
    A("")
    A("5. PRIMARY STATISTIC (pooled and per cohort)")
    A("-" * 78)
    A("  scope    delta     95% CI              cells(all/acc)  types  strata")
    b = res["boot"]["__pool__"]["delta"]
    blo, bhi, bnb = res["ci"](b)
    A("  %-8s %-9s [%s, %s]   %6d/%-6d  %5d  %5d"
      % ("pooled", num(ob["delta"]), num(blo), num(bhi),
         int(round(ob["cells_all"])), int(round(ob["cells_acc"])),
         ob["n_types"], ob["n_strata"]))
    for ds in dss:
        r = res["obs_ds"][ds]
        v = res["boot"][ds]["delta"]
        dlo, dhi, dnb = res["ci"](v)
        A("  %-8s %-9s [%s, %s]   %6d/%-6d  %5d  %5d"
          % (SHORT[ds], num(r["delta"]), num(dlo), num(dhi),
             int(round(r["cells_all"])), int(round(r["cells_acc"])),
             r["n_types"], r["n_strata"]))
    A("")
    A("  Cells(all/acc) is the total prediction-side cell count the cell-weighted")
    A("  means rest on, summed over T* entries; types is the number of such")
    A("  entries (cohort x clf x alpha x type); strata counts the")
    A("  (clf, alpha) blocks that contributed at least one type.")
    A("")
    A("  The two pooled means separately:")
    A("    mean_cellweighted(dist_all) = %s   (weights: n(pred_all, t))"
      % num(_m_all(res["entries"])))
    A("    mean_cellweighted(dist_acc) = %s   (weights: n(pred_acc, t))"
      % num(_m_acc(res["entries"])))
    A("")
    A("  Bootstrap distribution of the pooled delta (not only its interval):")
    A("    median = %s, 2.5%%/97.5%% = [%s, %s], P(delta <= 0) = %.3f,"
      % (num(np.nanmedian(b)), num(blo), num(bhi),
         float(np.mean(np.asarray(b)[np.isfinite(b)] <= 0))))
    A("    finite replicates = %d of %d." % (bnb, res["boot_n"]))
    if np.isfinite(ob["delta"]) and np.isfinite(blo) and (
            ob["delta"] < blo or ob["delta"] > bhi):
        A("    NOTE: the observed delta lies just outside the percentile interval.")
        A("    The statistic is a cell-weighted mean over a type set that is")
        A("    recomputed in every replicate; types that fall below the 20-cell")
        A("    minimum in a resample drop out and move the mean.  The bootstrap")
        A("    median above and the influence check in 8(d) are the honest")
        A("    companions to the interval.")
    A("")
    A("6. SECONDARY STATISTICS (paired variants)")
    A("-" * 78)
    A("  median over types of (dist_all - dist_acc), pooled and per cohort:")
    A("  scope    median    95% CI")
    pc = res["boot"]["__pool__"]["med"]
    mlo, mhi, _ = res["ci"](pc)
    A("  %-8s %-9s [%s, %s]" % ("pooled", num(ob["med"]), num(mlo), num(mhi)))
    for ds in dss:
        r = res["obs_ds"][ds]
        dlo, dhi, _ = res["ci"](res["boot"][ds]["med"])
        A("  %-8s %-9s [%s, %s]" % (SHORT[ds], num(r["med"]), num(dlo), num(dhi)))
    A("")
    A("  common-weight paired variant sum_t w_t (dist_all - dist_acc) / sum_t w_t")
    A("  with w_t = n(pred_all, t) on both sides:")
    pp = res["boot"]["__pool__"]["paired"]
    plo, phi, _ = res["ci"](pp)
    A("  %-8s %-9s [%s, %s]" % ("pooled", num(ob["paired"]), num(plo), num(phi)))
    for ds in dss:
        r = res["obs_ds"][ds]
        dlo, dhi, _ = res["ci"](res["boot"][ds]["paired"])
        A("  %-8s %-9s [%s, %s]" % (SHORT[ds], num(r["paired"]), num(dlo), num(dhi)))
    A("  (This variant re-weights the accepted side by the ALL-side cell counts, so")
    A("   a type that nearly vanishes among accepted cells can dominate it.  Read")
    A("   it together with the entries table in 8(d); where the point estimate sits")
    A("   outside the percentile interval the same type-dropout effect as in")
    A("   section 5 is at work.)")
    A("")
    A("  Composition-matched variants: ONE weight per type on BOTH sides.  This")
    A("  is the reading that removes the weighting confound of section 4, because")
    A("  a type carries the same weight in dist_all and in dist_acc, so only the")
    A("  error rates of the two scopes can move the number.")
    A("")
    A("  (i) w_t = n(pred_acc, t) on both sides (the accepted-side composition")
    A("      imposed on the all-side mean as well):")
    A("  scope    value     95% CI")
    for ds_, lab_ in [("__pool__", "pooled")] + [(d, SHORT[d]) for d in dss]:
        v_ = (mpa if ds_ == "__pool__" else res["obs_ds"][ds_]["paired_acc"])
        blo_, bhi_, _ = res["ci"](res["boot"][ds_]["paired_acc"])
        A("  %-8s %-9s [%s, %s]" % (lab_, num(v_), num(blo_), num(bhi_)))
    A("")
    A("  (ii) w_t = min(n(pred_all, t), n(pred_acc, t)) on both sides:")
    A("  scope    value     95% CI")
    for ds_, lab_ in [("__pool__", "pooled")] + [(d, SHORT[d]) for d in dss]:
        v_ = (mpm if ds_ == "__pool__" else res["obs_ds"][ds_]["paired_min"])
        blo_, bhi_, _ = res["ci"](res["boot"][ds_]["paired_min"])
        A("  %-8s %-9s [%s, %s]" % (lab_, num(v_), num(blo_), num(bhi_)))
    A("")
    A("  (iii) the same statistic with the TYPE SET frozen at the support estimable")
    A("        in the full sample -- the control for the type-dropout effect of")
    A("        section 5.  A frozen type that a resample empties out keeps its")
    A("        place at weight zero instead of leaving the average:")
    ff = res["boot"]["__pool__"]["paired_acc_fix"]
    flo, fhi, fnb = res["ci"](ff)
    A("  %-8s %-9s [%s, %s]  (%d/%d finite)"
      % ("pooled", num(mpa), num(flo), num(fhi), fnb, res["boot_n"]))
    for ds in dss:
        r = res["obs_ds"][ds]
        dlo, dhi, _ = res["ci"](res["boot"][ds]["paired_acc_fix"])
        A("  %-8s %-9s [%s, %s]" % (SHORT[ds], num(r["paired_acc"]),
                                   num(dlo), num(dhi)))
    A("  The frozen interval moves by %s (lower) and %s (upper) against (i), so"
      % (num(flo - mpla), num(fhi - mpha)))
    A("  the observed point sitting at the edge of its own interval is a property")
    A("  of the ratio of resampled sums, not of type dropout.")
    A("  (These two are the composition-matched reading of the same quantity and")
    A("   are reported next to the primary statistic, not in place of it: the")
    A("   pre-specified primary is the own-weight comparison of section 5 and the")
    A("   verdict in the headline is decided by that interval.  Where the two")
    A("   readings disagree in sign, the disagreement is itself the result -- the")
    A("   own-weight number is carried by the change in weighting between the")
    A("   two scopes.)")
    A("")
    A("7. PER-STRATUM DETAIL")
    A("-" * 78)
    A("  cohort  clf     alpha  delta      med        paired     T*  scope")
    for st in S:
        s = st["stat"]
        A("  %-7s %-7s %.2f   %-9s  %-9s  %-9s  %2d  %s"
          % (SHORT[st["ds"]], st["clf"], st["alpha"], num(s["delta"]),
             num(s["med"]), num(s["paired"]), s["n_types"],
             "estimable" if np.isfinite(s["delta"]) else
             "not estimable (accepted side has no join)"))
    A("")
    A("8. ROBUSTNESS")
    A("-" * 78)
    A("  (a) min_cells = %d instead of r06's %d (the 150-cell rule is what"
      % (MIN_CELLS_SENS, res["min_cells"]))
    A("      discards most accepted cells in GVHD and all of them in CRC):")
    rp = res["obs_pool_s"]
    slo, shi, snb = res["ci"](res["boot_s"]["__pool__"]["delta"])
    A("      pooled delta = %s  95%% CI [%s, %s]  cells(all/acc) %d/%d  types %d"
      % (num(rp["delta"]), num(slo), num(shi), int(round(rp["cells_all"])),
         int(round(rp["cells_acc"])), rp["n_types"]))
    for ds in dss:
        r = res["obs_ds_s"][ds]
        dlo, dhi, _ = res["ci"](res["boot_s"][ds]["delta"])
        A("      %-8s delta = %-9s [%s, %s]  cells(all/acc) %d/%d"
          % (SHORT[ds], num(r["delta"]), num(dlo), num(dhi),
             int(round(r["cells_all"])), int(round(r["cells_acc"]))))
    A("      (images on the accepted side at min_cells=%d: %s)"
      % (MIN_CELLS_SENS, ", ".join(
          "%s %d/%d" % (SHORT[st["ds"]], st["n_img_all_s"], st["n_img_acc_s"])
          for st in S if st["clf"] == res["clfs"][0]
          and abs(st["alpha"] - 0.10) < 1e-9)))
    A("")
    A("  (b) replicate sensitivity.  lambda is identical across the six")
    A("      metrics-table repeats for Breast, GVHD and CRC, so repeat 0 is r15's")
    A("      answer there.  Where lambda does vary, the per-replicate point")
    A("      estimates of delta are:")
    if not res["rep_rows"]:
        A("      none (no stratum had a replicate-varying lambda)")
    for rr in res["rep_rows"]:
        A("      %-9s %-7s alpha=%.2f  delta by repeat: %s  (|S_acc|: %s)"
          % (SHORT[rr["ds"]], rr["clf"], rr["alpha"],
             ", ".join("%s" % num(v[1], 3) for v in rr["vals"]),
             ", ".join("%d" % v[3] for v in rr["vals"])))
    A("")
    A("  (c) NaN handling.  r06 propagates a NaN enrichment into the whole type")
    A("      aggregate.  Here NaN (image, type) rows are dropped and counted:")
    A("      %d dropped over all strata." % res["n_nan_tot"])
    A("")
    A("  (d) influence / fragility (descriptive; the headline statistic is")
    A("      unchanged by this check).  The pooled primary mean is an average of")
    A("      %d (cohort, clf, alpha, type) entries; a handful of them carry most"
      % len(res["entries"]))
    A("      of the weight.  The largest contributions to sum_t w_t (d_all-d_acc),")
    A("      which is what the accepted-vs-all comparison is made of:")
    ent = res["entries"].copy()
    ent["abs_pw"] = ent["paired_w"].abs()
    top = ent.sort_values("abs_pw", ascending=False).head(5)
    A("        cohort    clf     alpha  type            n_all  n_acc  d_all   "
      "d_acc   w*(d_all-d_acc)")
    for _, r in top.iterrows():
        A("        %-9s %-7s %.2f   %-15s %5d  %5d  %-6s  %-6s  %s"
          % (SHORT[r["dataset"]], r["clf"], r["alpha"], r["type"],
             int(r["n_all"]), int(r["n_acc"]), num(r["dist_all"], 3),
             num(r["dist_acc"], 3), num(r["paired_w"], 1)))
    base = delta_of_entries(ent)
    loo = []
    for i in range(len(ent)):
        loo.append(delta_of_entries(ent.drop(ent.index[i])))
    loo = np.asarray(loo, dtype=float)
    fin = loo[np.isfinite(loo)]
    A("      pooled delta (all entries)                      = %s" % num(base))
    A("      worst-case leave-one-entry-out delta            = [%s, %s]"
      % (num(fin.min()) if fin.size else "na",
         num(fin.max()) if fin.size else "na"))
    if len(ent) > 1:
        biggest = ent["abs_pw"].idxmax()
        dd = delta_of_entries(ent.drop(biggest))
        A("      dropping the single largest contributor (%s, %s, alpha=%.2f, %s):"
          % (SHORT[ent.loc[biggest, "dataset"]], ent.loc[biggest, "clf"],
             ent.loc[biggest, "alpha"], ent.loc[biggest, "type"]))
        A("          delta = %s  (it was %s with the entry in)"
          % (num(dd), num(base)))
    A("")
    A("9. LIMITATIONS / WHAT WE COULD NOT DO")
    A("-" * 78)
    A("  * Only the `within` split is analysed; the certificate's behaviour under")
    A("    transfer is not covered here.")
    A("  * Accepted sets are defined at repeat 0 of the metrics table (see 8b).")
    A("  * Where the accepted side has no finite lambda in any fold (parts of")
    A("    CRC) the stratum is simply not estimable; it is reported as such and")
    A("    not dropped from the tables.")
    A("  * The bootstrap resamples images; with 8-48 images per cohort the CI is")
    A("    itself coarse, and for GVHD the accepted side may rest on one image.")
    A("  * No noise-only null distribution was simulated, so 'no detectable")
    A("    change' means the CI contains zero, not that the true difference is")
    A("    exactly zero.")
    A("")
    txt = "\n".join(R) + "\n"
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write(txt)
    return txt


def make_figure(res, csv, path_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
        "legend.frameon": False,
    })

    def panel(ax, letter, title=None):
        ax.set_title(letter, loc="left", fontweight="bold", pad=6)
        if title:
            ax.set_title(title, loc="right", fontsize=7.5, color="0.35")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})
    dss = res["dss"]
    alphas = res["alphas"]

    # ---- panel a: dist_acc vs dist_all per (cohort, clf, alpha, type)
    ax = axes[0]
    pt = csv[(csv["scope"] == "both")]
    shade = {alphas[i]: (0.30 + 0.30 * i) for i in range(len(alphas))}
    for ds in dss:
        for a in alphas:
            sel = pt[(pt["dataset"] == ds) & (np.isclose(pt["alpha"], a))]
            if sel.empty:
                continue
            ax.scatter(np.maximum(sel["dist_all"], FLOOR),
                       np.maximum(sel["dist_acc"], FLOOR), s=15,
                       marker=MK[ds], color=CLR[ds], alpha=shade[a],
                       linewidths=0)
    lim = [FLOOR, 10.0]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("dist_all: |pred - expert| enrichment, all cells")
    ax.set_ylabel("dist_acc: |pred - expert| enrichment, accepted")
    h = [Line2D([], [], marker=MK[ds], ls="", color=CLR[ds], ms=4.5,
                label=SHORT[ds]) for ds in dss]
    h += [Line2D([], [], marker="o", ls="", color="0.35", ms=4.0,
                 alpha=shade[a], label="alpha=%.2f" % a) for a in alphas]
    ax.legend(handles=h, fontsize=6, ncol=2, loc="upper left",
              handletextpad=0.3, columnspacing=0.7)
    ax.grid(False)
    panel(ax, "a", "below the line = distortion smaller when accepted")

    # ---- panel b: primary statistic with bootstrap CI
    ax = axes[1]
    rows = [("pooled", res["obs_pool"], res["boot"]["__pool__"]["delta"])]
    rows += [(SHORT[ds], res["obs_ds"][ds], res["boot"][ds]["delta"])
             for ds in dss]
    ypos = np.arange(len(rows))[::-1]
    for y, (lab, r, bv) in zip(ypos, rows):
        dlo, dhi, _ = res["ci"](bv)
        if not np.isfinite(r["delta"]):
            ax.text(0.02, y, "not estimable (no accepted-side join)",
                    fontsize=6.5, color="0.45", va="center")
            continue
        col = "0.15" if lab == "pooled" else CLR[[k for k in dss
                                                  if SHORT[k] == lab][0]]
        ax.errorbar([r["delta"]], [y],
                    xerr=[[r["delta"] - dlo], [dhi - r["delta"]]],
                    fmt="o", ms=4.5, lw=1.2, capsize=2.5, color=col)
        ax.text(dhi + 0.02, y, "%d cells" % int(round(r["cells_acc"])),
                fontsize=5.8, color="0.4", va="center")
    ax.axvline(0.0, color="k", ls="--", lw=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("dist_all - dist_acc  (cell-weighted mean, accepted cells)\n"
                  "negative = accepted cells distort the spatial readout more")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(False)
    panel(ax, "b", "95% CI, bootstrap over images")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.splitext(path_pdf)[0] + "." + ext,
                    bbox_inches="tight")
    plt.close(fig)
    log("wrote figures/fig9_downstream.pdf and .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alphas", default=",".join("%.2f" % a for a in ALPHAS))
    ap.add_argument("--clfs", default=",".join(CLFS))
    ap.add_argument("--min-cells", type=int, default=MIN_CELLS)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",") if a]
    clfs = [c for c in args.clfs.split(",") if c]

    res = run(alphas, clfs, args.min_cells, args.boot, args.seed)

    csv = csv_rows(res)
    csv.to_csv(CSV_PATH, index=False)
    log("wrote experiments/r21_downstream.csv (%d rows)" % len(csv))

    txt = write_report(res, REPORT_PATH)
    log("wrote work/r21_report.txt (%d bytes)" % len(txt))
    print("")
    print(txt)

    if not args.no_figure:
        make_figure(res, csv, os.path.join(FIGURES, "fig9_downstream.pdf"))

    log("VERDICT LINE: %s" % res["verdict"])
    log("done")


if __name__ == "__main__":
    main()
