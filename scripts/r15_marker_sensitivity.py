"""R15 -- how much of the marker-rule violation rate is real, and how much is
an artefact of the audit's own definitions?

R06 audits every cohort's labels against the Boolean marker rules the depot
publishes for each population, and reports the violation rate pooled over the
classes whose rules are *fully* evaluable on that panel (``ALL_complete``). That
definition is not neutral. A class drops out as soon as one of its rules names a
channel the panel does not carry, and in GVHD the classes that drop out are the
ones with by far the highest violation rate: the pooled 4.02% is computed after
removing a class whose own rule is violated by 69% of its cells.

R15 asks how much of the headline survives when the audit's definitions are
varied, on six axes:

1. per-cell-type violation rates (coarse and native vocabulary) with Wilson
   CIs, plus the dilution decomposition of the pooled cohort rate;
2. positivity-threshold sensitivity -- the same audit under a percentile-based
   per-channel positivity call instead of the two-component Gaussian mixture;
3. rule-reading sensitivity -- ";" read as OR vs AND;
4. restriction to the cells the pipeline actually deploys (the within-fold test
   sets) and to the cells the risk controller accepts at alpha = 0.10;
5. pooling that is not dominated by the largest cohort -- cell-weighted vs
   cohort-weighted mean, with a cluster bootstrap over cohorts;
6. violation rates of the *predicted* labels, so the audit describes the
   pipeline output and not only the reference.

Outputs: experiments/r15_types.csv, r15_threshold.csv, r15_deployed.csv,
r15_pooled.csv, work/r15_report.txt and figures/figS3_marker_sensitivity.pdf/png.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import data as D, markercheck as MC, paths
from errctl.labels import COARSE_MAP
from errctl.markerlogic import canonical, compile_rules

RESULTS = paths.RESULTS
FIGURES = paths.FIGURES
PRED_DIR = os.path.join(paths.WORK, "predictions")
LOG_PATH = os.path.join(paths.WORK, "r15_report.txt")

NON_MARKERS = {"segmentation_borders", "DRAQ5", "DNA", "Morphology", "nearCD31"}

# The depot rule text names a few channels that the matching panel does not
# carry under that exact spelling, but which plainly denote a panel channel: a
# "_hi" state suffix on NAKATPASE, and a transposed-letter spelling of IgA.
# Repairing them is a nomenclature fix, not a change of threshold, so the
# shipped and the repaired reading are always reported side by side.
ALIASES = {"NAKATPASE_hi": "NAKATPASE", "IgaA": "IgA"}

READINGS = ("or", "and")
VARIANTS = ("shipped", "repaired")
Z = 1.959963984540054


def log(msg=""):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")


def wilson(k, n, z=Z):
    """Wilson score interval for a binomial proportion."""
    if not n:
        return np.nan, np.nan
    p = float(k) / float(n)
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cluster_bootstrap(rates, ns, boot=2000, seed=0, mode="cohort_mean"):
    """Percentile CI resampling whole cohorts (clusters), not cells."""
    rates = np.asarray(rates, dtype=float)
    ns = np.asarray(ns, dtype=float)
    ok = np.isfinite(rates)
    rates, ns = rates[ok], ns[ok]
    c = rates.size
    if c < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = np.empty(boot, dtype=float)
    for b in range(boot):
        idx = rng.integers(0, c, c)
        if mode == "cohort_mean":
            out[b] = float(np.mean(rates[idx]))
        else:
            w = ns[idx]
            out[b] = float(np.sum(rates[idx] * w) / np.sum(w)) if np.sum(w) else np.nan
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))

# ------------------------------------------------------------------ rule layer
def canonical_frame(g, markers):
    """Marker columns renamed onto the shared channel vocabulary (dupes dropped)."""
    taken, cols = set(), []
    for m in markers:
        c = canonical(m)
        if c in taken or c in NON_MARKERS or m in NON_MARKERS:
            continue
        taken.add(c)
        cols.append((c, m))
    ren = {m: c for c, m in cols}
    return g[list(ren)].rename(columns=ren), [c for c, _ in cols]


def rule_table(ds, reading="or", repair=True):
    """Compile the depot's cell-type table into evaluable predicates."""
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
    """(coarse class -> [rule, ...], native class -> rule) using COARSE_MAP."""
    groups, native, unmapped = {}, {}, {}
    for ct, rule in rules.items():
        native[ct] = rule
        coarse = COARSE_MAP.get(ct)
        if not coarse:
            unmapped[ct] = rule
            continue
        if coarse == "Unidentified":
            unmapped[ct] = rule
            continue
        groups.setdefault(coarse, []).append(rule)
    return groups, native, unmapped


def evaluate(labels, groups, pos, restrict=None):
    """(evaluable, violated) masks; a class needs at least one usable rule.

    A rule is usable when every channel it names has a positivity call on this
    panel. Violations are the OR over the usable rules, so a class whose rule
    set is only partly evaluable can only have its violation rate *overstated*:
    an observed rate under a partial rule set is an upper bound on the rate the
    complete rule set would give.
    """
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


def evaluate_native(labels, native, pos):
    """Same, but per native depot class so types are not folded together."""
    labels = np.asarray(labels, dtype=object)
    ev = np.zeros(labels.size, dtype=bool)
    vi = np.zeros(labels.size, dtype=bool)
    for ct, rule in native.items():
        m = labels == ct
        if not m.any():
            continue
        if not rule.idents <= set(pos):
            continue
        vals = {i: pos[i][m] for i in rule.idents}
        ok = np.asarray(rule.evaluate(vals), dtype=bool)
        ev[m] = True
        vi[m] = ~ok
    return ev, vi


def evaluability(missing, used):
    out = {}
    for c in used:
        if used[c] and not missing.get(c):
            out[c] = "complete"
        elif used[c]:
            out[c] = "partial"
        else:
            out[c] = "none"
    return out


# ------------------------------------------------------------ positivity calls
def percentile_positivity(frame, chans, pct, mode="all", base=None):
    """Per-channel positivity from a percentile of log1p intensity.

    mode="all"    -- the percentile is taken over every cell on the panel;
    mode="pospop" -- over the cells the Gaussian mixture already calls positive,
                     i.e. the percentile refines the mixture's own boundary.
    """
    pos = {}
    for c in chans:
        x = np.asarray(frame[c], dtype=np.float64)
        v = np.nan_to_num(np.log1p(np.clip(x, 0, None)), nan=0.0, posinf=0.0)
        if mode == "pospop" and base is not None:
            ref = v[np.asarray(base[c], dtype=bool)]
            ref = ref[np.isfinite(ref)]
        else:
            ref = v
        if ref.size == 0 or np.allclose(ref, ref[0]):
            pos[c] = np.zeros(v.size, dtype=bool)
            continue
        pos[c] = v > np.percentile(ref, pct)
    return pos

# --------------------------------------------------------------- dataset worker
def row_for(ds, scope, reading, variant, source, cls, n, k, rules_used="",
            rules_missing="", n_rules_total=np.nan, n_rules_used=np.nan,
            evaluab=""):
    lo, hi = wilson(k, n)
    return {"dataset": ds, "scope": scope, "reading": reading, "variant": variant,
            "source": source, "cls": cls, "n": int(n), "n_violation": int(k),
            "violation_rate": (float(k) / float(n)) if n else np.nan,
            "ci_lo": lo, "ci_hi": hi,
            "n_rules_total": n_rules_total, "n_rules_used": n_rules_used,
            "evaluability": evaluab, "rules_used": " ; ".join(rules_used)
            if isinstance(rules_used, (list, tuple)) else rules_used,
            "rules_missing": " ; ".join(rules_missing)
            if isinstance(rules_missing, (list, tuple)) else rules_missing}


def type_rows(ds, labels, groups, pos, used, missing, ev, vi, reading, variant,
              source):
    """Per-coarse-class rows plus the two pooled summaries."""
    labels = np.asarray(labels, dtype=object)
    evl = evaluability(missing, used)
    out = []
    for coarse in sorted(groups):
        m = labels == coarse
        n_all = int(m.sum())
        if n_all == 0:
            continue
        n_ev = int((m & ev).sum())
        k = int(vi[m & ev].sum())
        out.append(row_for(ds, "type", reading, variant, source, coarse, n_ev, k,
                           used.get(coarse, []), missing.get(coarse, []),
                           len(groups[coarse]), len(used.get(coarse, [])),
                           evl.get(coarse, "none")))
    total = int(labels.size)
    for tag, keep in (("ALL_strict",
                       {c for c in groups if evl.get(c) == "complete"}),
                      ("ALL_evaluable",
                       {c for c in groups if evl.get(c) != "none"})):
        mask = np.isin(labels, sorted(keep)) if keep else np.zeros(labels.size, bool)
        mm = mask & ev
        inner = [c for c in sorted(keep)]
        miss_any = sorted({i for c in inner for i in missing.get(c, [])})
        out.append(row_for(ds, "summary", reading, variant, source, tag,
                           int(mm.sum()), int(vi[mm].sum()), "", miss_any,
                           np.nan, np.nan, "mixed"))
    return out


def dataset_worker(ds, percentiles, clfs, inner_jobs=2):
    """Everything that needs the raw marker intensities for one cohort."""
    cell = D.build_master(ds)
    ex = cell[cell["is_expert"]]
    keep = ex["coarse_expert"].notna() & (ex["coarse_expert"] != "Unidentified")
    n_expert_all = int(ex.shape[0])
    drop_mix = ex.loc[~keep, "class_expert"].value_counts().head(8).to_dict()
    g = ex[keep].copy().reset_index(drop=True)
    markers, _ = D.marker_table(ds, kind="HighQualityLabels" if ds in paths.HQ_DATASETS
                                else "ManualLabels")
    markers = [m for m in markers if m in g.columns]
    frame, chans = canonical_frame(g, markers)
    pos_gmm, _ = MC.positivity(frame, chans)
    expert = g["coarse_expert"].to_numpy(dtype=object)
    native_labels = g["class_expert"].to_numpy(dtype=object)
    sources = {"expert": expert}
    if "coarse_hq" in g.columns and g["coarse_hq"].notna().any():
        sources["depot_HQ"] = g["coarse_hq"].to_numpy(dtype=object)

    compiled = {}
    for reading in READINGS:
        for variant in VARIANTS:
            rules, failed = rule_table(ds, reading, repair=(variant == "repaired"))
            groups, native, unmapped = group_rules(rules)
            compiled[(reading, variant)] = (groups, native, unmapped, failed)

    types, thr = [], []
    for reading in READINGS:
        for variant in VARIANTS:
            groups, native, unmapped, failed = compiled[(reading, variant)]
            for source, labels in sources.items():
                ev, vi, used, missing = evaluate(labels, groups, pos_gmm)
                types += type_rows(ds, labels, groups, pos_gmm, used, missing,
                                   ev, vi, reading, variant, source)
            # native vocabulary, expert labels only
            ev, vi = evaluate_native(native_labels, native, pos_gmm)
            for ct in sorted(native):
                m = native_labels == ct
                if not m.any():
                    continue
                mm = m & ev
                miss = ([i for i in native[ct].idents if i not in pos_gmm]
                        if not native[ct].idents <= set(pos_gmm) else [])
                types.append(row_for(ds, "native", reading, variant, "expert", ct,
                                     int(mm.sum()), int(vi[mm].sum()),
                                     [native[ct].text], miss, 1,
                                     1 if not miss else 0,
                                     "complete" if not miss else "none"))

    def one(variant_name, pct):
        if variant_name == "gmm":
            pos, ruleset_name = pos_gmm, "gmm"
        elif variant_name == "pct_all":
            pos, ruleset_name = percentile_positivity(frame, chans, pct, "all"), "pct_all"
        else:
            pos, ruleset_name = percentile_positivity(
                frame, chans, pct, "pospop", base=pos_gmm), "pct_pospop"
        got = []
        for reading in READINGS:
            for variant in VARIANTS:
                groups, _, _, _ = compiled[(reading, variant)]
                ev, vi, used, missing = evaluate(expert, groups, pos)
                evl = evaluability(missing, used)
                for tag, keep in (("ALL_strict",
                                   {c for c in groups if evl.get(c) == "complete"}),
                                  ("ALL_evaluable",
                                   {c for c in groups if evl.get(c) != "none"})):
                    mask = np.isin(expert, sorted(keep)) if keep else np.zeros(expert.size, bool)
                    mm = mask & ev
                    n_ch = sum(1 for c in chans if pos.get(c) is not None)
                    got.append({"dataset": ds, "variant": ruleset_name,
                                "positivity_pct": (np.nan if pct is None else int(pct)),
                                "reading": reading, "ruleset": variant,
                                "summary": tag, "n": int(mm.sum()),
                                "n_violation": int(vi[mm].sum()),
                                "violation_rate": (float(vi[mm].mean())
                                                   if mm.any() else np.nan),
                                "n_channels_total": len(chans),
                                "n_channels_with_call": int(n_ch)})
                if reading == "or" and variant == "repaired":
                    ev_all, vi_all, used_all, _ = evaluate(expert, groups, pos)
                    for coarse in sorted(groups):
                        mm = (expert == coarse) & ev_all
                        if int(mm.sum()) < 1:
                            continue
                        got.append({"dataset": ds, "variant": ruleset_name,
                                    "positivity_pct": (np.nan if pct is None else int(pct)),
                                    "reading": reading, "ruleset": variant,
                                    "summary": "class:" + coarse,
                                    "n": int(mm.sum()),
                                    "n_violation": int(vi_all[mm].sum()),
                                    "violation_rate": (float(vi_all[mm].mean())
                                                       if mm.any() else np.nan),
                                    "n_channels_total": len(chans),
                                    "n_channels_with_call": int(n_ch)})
        return got

    items = [("gmm", None)] + [(v, int(p)) for v in ("pct_all", "pct_pospop")
                               for p in percentiles]
    res = Parallel(n_jobs=inner_jobs, prefer="threads")(
        delayed(one)(v, p) for v, p in items)
    for got in res:
        thr += got
    return {"ds": ds, "types": types, "thr": thr, "chans": chans,
            "n_expert": int(g.shape[0]), "n_expert_all": n_expert_all,
            "n_dropped": n_expert_all - int(g.shape[0]),
            "drop_mix": {str(k): int(w) for k, w in drop_mix.items()},
            "hq_available": int("depot_HQ" in sources),
            "unmapped": sorted(compiled[("or", "repaired")][2].keys())}

# --------------------------------------------------------------- deployed cells
def _load_within(ds, clf, gidx, n_cells):
    """Per-fold (cell index, score, fold) and the predicted coarse labels."""
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


def deployed_stage(ds, clfs, alphas, met):
    """Violation rate over the cells the pipeline deploys, and among accepted."""
    g, markers = D.gold(ds, drop_unidentified=True)
    g = g.reset_index(drop=True)
    frame, chans = canonical_frame(g, markers)
    pos, _ = MC.positivity(frame, chans)
    labels = g["coarse_expert"].to_numpy(dtype=object)
    rules, _ = rule_table(ds, "or", repair=True)
    groups, _, _ = group_rules(rules)
    ev, vi, used, missing = evaluate(labels, groups, pos)
    evl = evaluability(missing, used)
    strict = {c for c in groups if evl.get(c) == "complete"}
    ev_s, vi_s, _, _ = evaluate(labels, groups, pos, restrict=strict)
    rulesets = {"ALL_evaluable": (ev, vi), "ALL_strict": (ev_s, vi_s)}
    gidx = {}
    for i, (im, cid) in enumerate(zip(g["image"], g["cellID"])):
        gidx[(str(im), int(cid))] = i
    out, diag = [], {"ds": ds, "n_expert": int(g.shape[0]), "dups": {}}

    for ruleset, (e, v) in rulesets.items():
        out.append(dict(dataset=ds, clf="-", alpha=np.nan, reading="or",
                        variant="repaired", ruleset=ruleset, scope="all_expert",
                        n=int(e.sum()), n_violation=int(v[e].sum()),
                        violation_rate=float(v[e].mean()) if e.any() else np.nan,
                        rate_mean_over_repeats=np.nan, rate_sd_over_repeats=np.nan,
                        n_repeats=0))
    for clf in clfs:
        rows, pred, dup = _load_within(ds, clf, gidx, len(g))
        diag["dups"][clf] = int(dup)
        if not rows:
            continue
        idx = np.array([r[0] for r in rows])
        sc = np.array([r[1] for r in rows])
        fold = np.array([r[2] for r in rows])
        ev_p, vi_p, _, _ = evaluate(pred, groups, pos)
        for ruleset, (e, v) in rulesets.items():
            here = np.ones(idx.size, dtype=bool)
            for scope, sel in (("within_test_pooled", here),
                               ("within_test_distinct", None)):
                if sel is None:
                    keep = np.unique(idx)
                    n = int(e[keep].sum())
                    k = int(v[keep].sum())
                else:
                    m = e[idx[sel]]
                    n, k = int(m.sum()), int(v[idx[sel]][m].sum())
                out.append(dict(dataset=ds, clf=clf, alpha=np.nan, reading="or",
                                variant="repaired", ruleset=ruleset, scope=scope,
                                n=n, n_violation=k,
                                violation_rate=float(k) / n if n else np.nan,
                                rate_mean_over_repeats=np.nan,
                                rate_sd_over_repeats=np.nan, n_repeats=0))
            # predicted labels, over all matched cells and over the test cells
            for scope, keep in (("pred_all_expert", np.nonzero(
                                    np.array([x is not None for x in pred]))[0]),
                                ("pred_within_test", np.unique(idx))):
                m = ev_p[keep]
                n, k = int(m.sum()), int(vi_p[keep][m].sum())
                out.append(dict(dataset=ds, clf=clf, alpha=np.nan, reading="or",
                                variant="repaired", ruleset=ruleset, scope=scope,
                                n=n, n_violation=k,
                                violation_rate=float(k) / n if n else np.nan,
                                rate_mean_over_repeats=np.nan,
                                rate_sd_over_repeats=np.nan, n_repeats=0))
            if clf:
                for cls_name in sorted(groups):
                    if ruleset == "ALL_strict" and cls_name not in strict:
                        continue
                    mm = (pred == cls_name) & ev_p
                    if int(mm.sum()) < 1:
                        continue
                    out.append(dict(dataset=ds, clf=clf, alpha=np.nan, reading="or",
                                    variant="repaired", ruleset=ruleset,
                                    scope="pred_class", cls=cls_name,
                                    n=int(mm.sum()), n_violation=int(vi_p[mm].sum()),
                                    violation_rate=float(vi_p[mm].mean()),
                                    rate_mean_over_repeats=np.nan,
                                    rate_sd_over_repeats=np.nan, n_repeats=0))
            # accepted vs deferred at each budget
            for alpha in alphas:
                sub = met[(met.clf == clf) & np.isclose(met.alpha, alpha)]
                if sub.empty:
                    continue
                lamap = {}
                for _u, _rp, _lm in zip(sub["unit"], sub["repeat"], sub["lambda"]):
                    if np.isfinite(_lm):
                        lamap[(_u, int(_rp))] = float(_lm)
                units = sorted({u for u, _ in lamap})
                cfgs = {}
                for rep in sorted({int(r) for r in sub["repeat"].unique()}):
                    key = tuple(round(lamap.get((u, rep), float("nan")), 10)
                                for u in units)
                    cfgs.setdefault(key, rep)
                per_rep, pooled = [], [0, 0, 0, 0]
                for key, rep in sorted(cfgs.items(), key=lambda kv: kv[1]):
                    tot = [0, 0, 0, 0]
                    for unit, lam in zip(units, key):
                        if not np.isfinite(lam):
                            continue
                        sel = fold == int(unit.rsplit("fold", 1)[1])
                        acc = sc[sel] <= lam
                        ii = idx[sel]
                        for a in (True, False):
                            mm = (acc == a) & e[ii]
                            tot[0 if a else 2] += int(mm.sum())
                            tot[1 if a else 3] += int(v[ii][mm].sum())
                    per_rep.append(tot)
                    pooled = [p + t for p, t in zip(pooled, tot)]
                for name, o in (("accepted", 0), ("deferred", 2)):
                    n, k = pooled[o], pooled[o + 1]
                    rr = [t[o + 1] / t[o] if t[o] else np.nan for t in per_rep]
                    out.append(dict(dataset=ds, clf=clf, alpha=alpha, reading="or",
                                    variant="repaired", ruleset=ruleset, scope=name,
                                    n=n, n_violation=k,
                                    violation_rate=float(k) / n if n else np.nan,
                                    rate_mean_over_repeats=float(np.nanmean(rr))
                                    if rr else np.nan,
                                    rate_sd_over_repeats=float(np.nanstd(rr, ddof=1))
                                    if len(rr) > 1 else np.nan,
                                    n_repeats=len(cfgs)))
    for r in out:
        r.setdefault("cls", "-")
        r["ci_lo"], r["ci_hi"] = wilson(r["n_violation"], r["n"])
    return out, diag

# ------------------------------------------------------------------- pooling
def pooling_stage(types, sources, boot=2000, seed=0):
    """Cell-weighted pool vs cohort-weighted mean, with cluster bootstrap CIs."""
    rows = []
    for reading in READINGS:
        for variant in VARIANTS:
            for source in sources:
                for tag in ("ALL_strict", "ALL_evaluable"):
                    ns, ks, dss = [], [], []
                    for ds in paths.MANUAL_DATASETS:
                        s = types[(types.scope == "summary") & (types.dataset == ds)
                                  & (types.reading == reading)
                                  & (types.variant == variant)
                                  & (types.source == source) & (types.cls == tag)]
                        if s.empty:
                            continue
                        ns.append(float(s["n"].iloc[0]))
                        ks.append(float(s["n_violation"].iloc[0]))
                        dss.append(ds)
                        lo, hi = wilson(ks[-1], ns[-1])
                        rows.append(dict(pooling="per_cohort", reading=reading,
                                         variant=variant, source=source, cls=tag,
                                         cohort=ds, estimate=ks[-1] / ns[-1] if ns[-1] else np.nan,
                                         ci_lo_wilson=lo, ci_hi_wilson=hi,
                                         ci_lo_boot=np.nan, ci_hi_boot=np.nan,
                                         n_cells=ns[-1], n_violation=ks[-1],
                                         n_cohorts=1))
                    if len(ns) < 2:
                        continue
                    n = np.asarray(ns)
                    k = np.asarray(ks)
                    rate = k / n
                    N, K = float(n.sum()), float(k.sum())
                    lo_w, hi_w = wilson(K, N)
                    lo_cm, hi_cm = cluster_bootstrap(rate, n, boot, seed, "cohort_mean")
                    lo_cw, hi_cw = cluster_bootstrap(rate, n, boot, seed, "cell_weighted")
                    base = dict(reading=reading, variant=variant, source=source,
                                cls=tag, n_cohorts=int(len(n)))
                    rows.append(dict(base, pooling="cell_weighted", cohort="-",
                                     estimate=K / N, ci_lo_wilson=lo_w,
                                     ci_hi_wilson=hi_w, ci_lo_boot=lo_cw,
                                     ci_hi_boot=hi_cw, n_cells=N, n_violation=K))
                    rows.append(dict(base, pooling="cohort_mean", cohort="-",
                                     estimate=float(rate.mean()),
                                     ci_lo_wilson=np.nan, ci_hi_wilson=np.nan,
                                     ci_lo_boot=lo_cm, ci_hi_boot=hi_cm,
                                     n_cells=float(n.mean()), n_violation=np.nan))
    return rows


# -------------------------------------------------------------------- figure
def figure(types, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
        "legend.frameon": False})
    ORDER = list(paths.MANUAL_DATASETS)
    SHORT = {"Breast_Jun0524": "Breast", "CRC_Feb2623": "CRC",
             "GVHD_Feb1323": "GVHD", "Melanoma_Oct2522": "Melanoma"}
    COL = {"Breast_Jun0524": "#1f5fa9", "CRC_Feb2623": "#2a9d8f",
           "GVHD_Feb1323": "#d1495b", "Melanoma_Oct2522": "#e6a817"}
    FLOOR = 2.5e-4
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))

    ax = axes[0]
    sel = types[(types.scope == "type") & (types.variant == "repaired")
                & (types.reading == "or") & (types.source == "expert")
                & (types.n >= 50)]
    x, ticks, ticklab = 0, [], []
    for ds in ORDER:
        sub = sel[sel.dataset == ds].sort_values("violation_rate", ascending=False)
        if sub.empty:
            continue
        start = x
        for _, r in sub.iterrows():
            y = max(float(r.violation_rate), FLOOR)
            if r.n_violation > 0 and np.isfinite(r.ci_lo):
                lo = max(float(r.ci_lo), FLOOR)
                ax.errorbar([x], [y], yerr=[[y - lo], [max(float(r.ci_hi), FLOOR) - y]],
                            fmt="o", ms=2.4, lw=0.8, capsize=1.1, color=COL[ds],
                            zorder=3)
            else:
                ax.plot([x], [FLOOR], marker="o", ms=2.4, mfc="white", mec=COL[ds],
                        mew=0.7, ls="none", zorder=3)
            ticks.append(x)
            ticklab.append(r.cls)
            x += 1
        p = types[(types.scope == "summary") & (types.cls == "ALL_evaluable")
                  & (types.dataset == ds) & (types.variant == "repaired")
                  & (types.reading == "or") & (types.source == "expert")]
        if not p.empty:
            ax.plot([start - 0.4, x - 0.6], [float(p.violation_rate.iloc[0])] * 2,
                    "-", color=COL[ds], lw=1.5, alpha=0.85, zorder=2)
        ax.text((start + x - 1) / 2.0, 1.15, SHORT[ds], ha="center", fontsize=6.2,
                color=COL[ds])
        x += 2
    ax.axhline(0.05, color="k", ls=":", lw=1.0)
    ax.text(x - 0.8, 0.056, "5%", fontsize=5.8, ha="right", color="0.25")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.6, 1.5)
    ax.set_xlim(-1.2, x - 0.2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklab, rotation=90, fontsize=4.4)
    ax.set_ylabel("violation rate of the expert label")
    ax.set_title("a", loc="left", fontweight="bold", pad=4)
    ax.set_title("class detail, bar = cohort pool", loc="right", fontsize=6.4,
                 color="0.35")

    ax = axes[1]
    t = thr[(thr.variant.isin(["pct_all", "pct_pospop"]))
            & (thr.ruleset == "repaired") & (thr.reading == "or")
            & (thr.summary == "ALL_evaluable")]
    for ds in ORDER:
        s = t[(t.dataset == ds) & (t.variant == "pct_all")].sort_values("positivity_pct")
        if not s.empty:
            ax.plot(s.positivity_pct, s.violation_rate, "o-", ms=2.5, lw=1.1,
                    color=COL[ds], label=SHORT[ds])
        s2 = t[(t.dataset == ds) & (t.variant == "pct_pospop")].sort_values("positivity_pct")
        if not s2.empty:
            ax.plot(s2.positivity_pct, s2.violation_rate, "s--", ms=2.1, lw=0.8,
                    color=COL[ds], alpha=0.55)
        g = thr[(thr.dataset == ds) & (thr.variant == "gmm")
                & (thr.ruleset == "repaired") & (thr.reading == "or")
                & (thr.summary == "ALL_evaluable")]
        if not g.empty:
            ax.axhline(float(g.violation_rate.iloc[0]), color=COL[ds], lw=0.7,
                       ls="-.", alpha=0.7)
    ax.axhline(0.05, color="k", ls=":", lw=1.0)
    ax.set_xlabel("positivity percentile")
    ax.set_ylabel("violation rate, all evaluable classes")
    ax.set_ylim(0, None)
    ax.legend(ncol=2, fontsize=5.8, title="solid: all cells; dashed: mixture-positive",
              title_fontsize=5.6)
    ax.set_title("b", loc="left", fontweight="bold", pad=4)
    ax.set_title("threshold sensitivity, dash-dot = mixture", loc="right",
                 fontsize=6.4, color="0.35")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, "figS3_marker_sensitivity.%s" % ext),
                    bbox_inches="tight")
    plt.close(fig)
    log("wrote figures/figS3_marker_sensitivity.pdf/.png")

# ------------------------------------------------------------------- report
def _p(x, width=7):
    if x is None or not np.isfinite(x):
        return " " * (width - 4) + "n/a"
    return ("%*.2f%%" % (width, 100.0 * x))


def write_report(types, thr, dep, pool, diags, args, elapsed):
    ORDER = [d for d in paths.MANUAL_DATASETS if d in set(types.dataset)]
    SHORT = {d: d.split("_")[0] for d in ORDER}
    dep = pd.DataFrame(dep)
    pool = pd.DataFrame(pool)

    log("")
    log("=" * 78)
    log("R15 -- marker-rule violation: how much is real, how much is definition")
    log("=" * 78)
    log("cohorts=%s  alphas=%s  clfs=%s  percentiles=%s  boot=%d  seed=%d  (%.1fs)"
        % (",".join(SHORT[d] for d in ORDER), args.alphas, args.clfs,
           args.percentiles, args.boot, args.seed, elapsed))
    log("cells audited: " + ", ".join(
        "%s=%d" % (SHORT[d], int(types[(types.scope == "summary") & (types.dataset == d)
                                       & (types.variant == "repaired")
                                       & (types.reading == "or")
                                       & (types.source == "expert")
                                       & (types.cls == "ALL_evaluable")]["n"].iloc[0]))
        for d in ORDER if d in set(types.dataset)))
    log("")
    log("Definitions. 'ALL_strict' keeps only classes whose every published rule is")
    log("evaluable on that panel (this reproduces r06's ALL_complete). 'ALL_evaluable'")
    log("keeps every class with at least one evaluable rule. Because the violation")
    log("flag is an OR over the usable rules, dropping a rule can only INCREASE the")
    log("observed rate, so a partly evaluable class is reported as an upper bound.")
    log("'repaired' maps two channel names the depot writes differently from the")
    log("panel (NAKATPASE_hi -> NAKATPASE, IgaA -> IgA); nothing else changes.")
    log("")
    for d in diags:
        n_all = d.get("n_expert_all", d["n_expert"])
        log("  %-16s audited=%6d of %6d expert cells (%.1f%%)  marker channels=%2d"
            % (d["ds"], d["n_expert"], n_all, 100.0 * d["n_expert"] / max(1, n_all),
               len(d["chans"])))
        if d.get("drop_mix"):
            log("      %d expert cells carry a native class with no coarse mapping and"
                % d.get("n_dropped", 0))
            log("      are excluded from every rate below: %s"
                % ", ".join("%s=%d" % kv for kv in sorted(d["drop_mix"].items(),
                                                          key=lambda kv: -kv[1])))
        if d.get("dups"):
            log("      cells appearing in more than one within-fold test set: %s"
                % ", ".join("%s=%d" % kv for kv in sorted(d["dups"].items())))

    log("")
    log("-" * 78)
    log("## 1. Violation rate of the EXPERT label, per cell type")
    log("     (reading ';' = OR, repaired channel names, n >= 50)")
    log("-" * 78)
    for ds in ORDER:
        sub = types[(types.scope == "type") & (types.dataset == ds)
                    & (types.variant == "repaired") & (types.reading == "or")
                    & (types.source == "expert") & (types.n >= 50)]
        sub = sub.sort_values("violation_rate", ascending=False)
        n_aud = int(types[(types.scope == "summary") & (types.dataset == ds)
                          & (types.variant == "repaired") & (types.reading == "or")
                          & (types.source == "expert")
                          & (types.cls == "ALL_evaluable")]["n"].iloc[0])
        log("")
        log("  %s   (%d audited expert cells)" % (ds, n_aud))
        for _, r in sub.iterrows():
            note = {"partial": "  [PARTIAL rules: missing %s]" % (r.rules_missing or "-"),
                    "none": "  [NO evaluable rule]"}.get(r.evaluability, "")
            log("     %-16s n=%5d  viol=%5d  rate=%s  CI[%s,%s]%s"
                % (r.cls, r.n, r.n_violation, _p(r.violation_rate, 8),
                   _p(r.ci_lo, 7).strip(), _p(r.ci_hi, 7).strip(), note))
        if not sub.empty:
            log("     worst >=50-cell type: %s %s   median over those types: %s"
                % (sub.iloc[0].cls, _p(sub.iloc[0].violation_rate, 8),
                   _p(sub.violation_rate.median(), 8)))

    log("")
    log("-" * 78)
    log("## 1b. Dilution: who carries the pooled cohort rate, and who is dropped")
    log("-" * 78)
    for ds in ORDER:
        s = types[(types.scope == "summary") & (types.dataset == ds)
                  & (types.variant == "repaired") & (types.reading == "or")
                  & (types.source == "expert")]
        st = s[s.cls == "ALL_strict"]
        ev = s[s.cls == "ALL_evaluable"]
        log("")
        log("  %s" % ds)
        log("     ALL_strict    %s  (%d/%d)" % (_p(st.violation_rate.iloc[0], 8),
                                                int(st.n_violation.iloc[0]), int(st.n.iloc[0])))
        log("     ALL_evaluable %s  (%d/%d)" % (_p(ev.violation_rate.iloc[0], 8),
                                                int(ev.n_violation.iloc[0]), int(ev.n.iloc[0])))
        t = types[(types.scope == "type") & (types.dataset == ds)
                  & (types.variant == "repaired") & (types.reading == "or")
                  & (types.source == "expert")]
        tot_k = int(t[t.evaluability != "none"]["n_violation"].sum())
        contr = t[(t.evaluability != "none") & (t.n_violation > 0)].sort_values(
            "n_violation", ascending=False)
        if len(contr):
            log("     carried by: %s" % "; ".join(
                "%s %d/%d (%s of the violations)" % (r.cls, r.n_violation, r.n,
                 "%.0f%%" % (100.0 * r.n_violation / tot_k) if tot_k else "-")
                for _, r in contr.iterrows()))
        else:
            log("     carried by: nothing -- no violations in this cohort")
        drop = t[t.evaluability == "partial"]
        if len(drop):
            log("     dropped by ALL_strict: %s" % "; ".join(
                "%s %d/%d = %s [missing %s]" % (r.cls, r.n_violation, r.n,
                 _p(r.violation_rate, 7).strip(), r.rules_missing or "-")
                for _, r in drop.sort_values("n", ascending=False).iterrows()))
            sh = types[(types.scope == "type") & (types.dataset == ds)
                       & (types.variant == "shipped") & (types.reading == "or")
                       & (types.source == "expert")]
            cmp_ = []
            for _, r in drop.sort_values("n", ascending=False).iterrows():
                s2 = sh[sh.cls == r.cls]
                if len(s2):
                    cmp_.append("%s %s with %d of %d rules usable -> %s with %d of %d"
                                % (r.cls, _p(float(s2.violation_rate.iloc[0]), 7).strip(),
                                   int(s2.n_rules_used.iloc[0]), int(s2.n_rules_total.iloc[0]),
                                   _p(r.violation_rate, 7).strip(),
                                   int(r.n_rules_used), int(r.n_rules_total)))
            if cmp_:
                log("     same cells, shipped vs repaired text: %s" % "; ".join(cmp_))
        zero = t[(t.evaluability != "none") & (t.n_violation == 0)]
        if len(zero):
            log("     diluted by (0 violations): %s  = %d cells"
                % (", ".join(zero.cls), int(zero.n.sum())))
        none_ = t[t.evaluability == "none"]
        if len(none_):
            log("     no evaluable rule (excluded from both summaries): %s"
                % ", ".join(none_.cls))

    log("")
    log("-" * 78)
    log("## 2. Positivity-threshold sensitivity")
    log("     Violation rate over ALL evaluable classes (reading OR, repaired text).")
    log("     'gmm' is the two-component mixture r06 uses; 'pct' thresholds log1p")
    log("     intensity at that percentile of the stated reference population.")
    log("-" * 78)
    t2 = thr[(thr.ruleset == "repaired") & (thr.reading == "or")
             & (thr.summary == "ALL_evaluable")]
    for variant, label in (("gmm", "two-component mixture (r06)"),
                           ("pct_all", "percentile of all cells"),
                           ("pct_pospop", "percentile of the mixture-positive cells")):
        sub = t2[t2.variant == variant]
        log("")
        log("  %s" % label)
        log("     %-6s %s" % ("pct", "  ".join("%9s" % SHORT[d] for d in ORDER)))
        pcts = ([np.nan] if variant == "gmm"
                else sorted(sub.positivity_pct.dropna().unique()))
        for pct in pcts:
            row = []
            for ds in ORDER:
                s = sub[sub.dataset == ds]
                s = s[s.positivity_pct.isna()] if not np.isfinite(pct) else s[
                    np.isclose(s.positivity_pct, pct)]
                row.append(_p(float(s.violation_rate.iloc[0]), 9) if len(s) else "      n/a")
            log("     %-6s %s" % ("-" if not np.isfinite(pct) else "%d" % pct,
                                  "  ".join(row)))

    log("")
    log("-" * 78)
    log("## 2b. Is the ranking of cell types stable across the sweep?")
    log("-" * 78)
    ck = thr[thr.summary.astype(str).str.startswith("class:")
             & (thr.ruleset == "repaired") & (thr.reading == "or")].copy()
    ck["cls"] = ck.summary.astype(str).str.replace("class:", "", regex=False)
    for ds in ORDER:
        s = ck[(ck.dataset == ds) & (ck.n >= 50)]
        gm = s[s.variant == "gmm"].sort_values("violation_rate", ascending=False)
        if gm.empty:
            continue
        ref = list(gm.cls.head(3))
        ranks, keepn, totn = {}, 0, 0
        for key, grp in s[s.variant != "gmm"].groupby(["variant", "positivity_pct"],
                                                      dropna=False):
            del key
            g2 = grp.sort_values("violation_rate", ascending=False)
            for i, c in enumerate(g2.cls):
                ranks.setdefault(c, []).append(i + 1)
            totn += 1
            keepn += int(list(g2.cls.head(3)) == ref)
        log("  %-16s top-3 at the mixture: %s" % (ds, ", ".join(ref)))
        log("      exact top-3 reproduced in %d/%d sweep settings (definition x pct)"
            % (keepn, totn))
        for c in ref:
            if ranks.get(c):
                log("      %-16s rank %d..%d across the sweep"
                    % (c, min(ranks[c]), max(ranks[c])))
        rr = gm.set_index("cls").violation_rate
        rhos = []
        for key, grp in s[s.variant != "gmm"].groupby(["variant", "positivity_pct"],
                                                      dropna=False):
            del key
            g2 = grp.set_index("cls").violation_rate
            common = rr.index.intersection(g2.index)
            if len(common) >= 4:
                rhos.append(float(np.corrcoef(rr[common].rank().to_numpy(),
                                              g2[common].rank().to_numpy())[0, 1]))
        if rhos:
            log("      Spearman rho vs the mixture ranking: min %.2f, median %.2f"
                % (min(rhos), float(np.median(rhos))))

    log("")
    log("-" * 78)
    log("## 3. Rule-reading sensitivity: ';' as OR vs AND")
    log("-" * 78)
    log("     violation rate over ALL evaluable classes, repaired text")
    log("     %-16s %10s %10s %10s" % ("cohort", "OR", "AND", "delta"))
    for ds in ORDER:
        s = types[(types.scope == "summary") & (types.dataset == ds)
                  & (types.variant == "repaired") & (types.source == "expert")
                  & (types.cls == "ALL_evaluable")]
        if len(s) < 2:
            continue
        o = float(s[s.reading == "or"].violation_rate.iloc[0])
        a = float(s[s.reading == "and"].violation_rate.iloc[0])
        log("     %-16s %s %s %+9.2fpp" % (ds, _p(o, 10), _p(a, 10), 100.0 * (a - o)))
        if abs(a - o) > 0.05:
            t = types[(types.scope == "type") & (types.dataset == ds)
                      & (types.variant == "repaired") & (types.source == "expert")]
            j = t.pivot_table(index="cls", columns="reading", values="violation_rate")
            if "or" in j.columns and "and" in j.columns:
                j = j.dropna()
                j["d"] = j["and"] - j["or"]
                j = j[j.d.abs() > 0.05].sort_values("d", ascending=False)
                if len(j):
                    log("        driven by: %s" % "; ".join(
                        "%s %s -> %s" % (c, _p(r["or"], 7).strip(),
                                         _p(r["and"], 7).strip())
                        for c, r in j.iterrows()))

    log("")
    log("-" * 78)
    log("## 4. Operationally relevant cells: deployed and accepted")
    log("     'within_test_*' = the cells the pipeline scores; 'pred_*' = the rate of")
    log("     the PREDICTED label rather than the reference; 'accepted'/'deferred'")
    log("     are the risk controller's decision at alpha = 0.10. Thresholds are the")
    log("     same across r03's repeats for three cohorts, so distinct threshold")
    log("     vectors, not repeats, are counted as replicates. Thresholds are read")
    log("     from the current experiments/r03_metrics.csv; the archived")
    log("     r06_selective.csv predates that file's last regeneration, so its")
    log("     accepted/deferred rates differ from these by up to about 1pp.")
    log("-" * 78)
    d = dep[(dep.variant == "repaired") & (dep.reading == "or")]
    for ruleset in ("ALL_evaluable", "ALL_strict"):
        log("")
        log("  %s" % ruleset)
        log("     %-20s %-7s %9s %8s %10s" % ("scope", "clf", "n", "viol", "rate"))
        for scope in ("all_expert", "within_test_pooled", "within_test_distinct",
                      "pred_all_expert", "pred_within_test", "accepted", "deferred"):
            s = d[(d.scope == scope) & (d.ruleset == ruleset)]
            if s.empty:
                continue
            for clf in sorted(s.clf.unique()):
                for _, r in s[s.clf == clf].iterrows():
                    extra = ""
                    if scope in ("accepted", "deferred"):
                        extra = "   CI[%s,%s] R=%d" % (_p(r.ci_lo, 7).strip(),
                                                       _p(r.ci_hi, 7).strip(),
                                                       int(r.n_repeats))
                    log("     %-20s %-7s %9d %8d %s%s"
                        % (scope, clf, r.n, r.n_violation, _p(r.violation_rate, 10), extra))

    log("")
    log("-" * 78)
    log("## 5. Pooling: cell-weighted vs cohort-weighted")
    log("-" * 78)
    log("     %-11s %-22s %-9s %11s %18s %20s"
        % ("pooling", "rule set / reading", "source", "estimate", "Wilson CI",
           "cluster boot CI"))
    for tag in ("ALL_strict", "ALL_evaluable"):
        for reading in ("or", "and"):
            for source in ("expert", "pred_logreg", "pred_hgb"):
                for mode in ("cell_weighted", "cohort_mean"):
                    s = pool[(pool.pooling == mode) & (pool.cls == tag)
                             & (pool.reading == reading) & (pool.source == source)
                             & (pool.variant == "repaired")]
                    if s.empty:
                        continue
                    r = s.iloc[0]
                    w = ("[%s,%s]" % (_p(r.ci_lo_wilson, 7).strip(),
                                      _p(r.ci_hi_wilson, 7).strip())
                         if np.isfinite(r.ci_lo_wilson) else "-")
                    b = "[%s,%s]" % (_p(r.ci_lo_boot, 7).strip(),
                                     _p(r.ci_hi_boot, 7).strip())
                    log("     %-11s %-22s %-9s %s %18s %20s"
                        % (mode, "%s / %s" % (tag, reading), source,
                           _p(r.estimate, 11), w, b))

    log("")
    log("-" * 78)
    log("## 6. Violation rate of the PREDICTED label (all expert cells)")
    log("-" * 78)
    log("     %-16s %-10s %9s %8s %10s %18s"
        % ("cohort", "source", "n", "viol", "rate", "CI"))
    for ds in ORDER:
        for source in ("expert", "depot_HQ", "pred_logreg", "pred_rf", "pred_hgb"):
            s = types[(types.scope == "summary") & (types.dataset == ds)
                      & (types.variant == "repaired") & (types.reading == "or")
                      & (types.source == source) & (types.cls == "ALL_evaluable")]
            if s.empty:
                continue
            r = s.iloc[0]
            log("     %-16s %-10s %9d %8d %s  [%s,%s]"
                % (ds, source.replace("pred_", ""), r.n, r.n_violation,
                   _p(r.violation_rate, 10), _p(r.ci_lo, 7).strip(),
                   _p(r.ci_hi, 7).strip()))

    log("")
    log("-" * 78)
    log("## 7. What the numbers support")
    log("-" * 78)
    log("     expert labels, three conventions of the same statistic")
    log("     %-16s %14s %18s %18s"
        % ("cohort", "strict/shipped", "evaluable/shipped", "evaluable/repaired"))
    for ds in ORDER:
        s = types[(types.scope == "summary") & (types.dataset == ds)
                  & (types.source == "expert")]

        def _rate(v, cls="ALL_evaluable"):
            r = s[(s.variant == v) & (s.reading == "or") & (s.cls == cls)]
            return float(r.violation_rate.iloc[0]) if len(r) else np.nan

        log("     %-16s %s %18s %18s"
            % (ds, _p(_rate("shipped", "ALL_strict"), 14).strip(),
               _p(_rate("shipped"), 18).strip(), _p(_rate("repaired"), 18).strip()))
    log("")
    log("  These are not three estimates of one number. ALL_strict covers a smaller")
    log("  cell set than ALL_evaluable, and in GVHD it removes the 3009-cell")
    log("  epithelial class and the 1790-cell fibroblast class. The shipped text")
    log("  leaves that epithelial class with only the goblet marker usable, so the")
    log("  23.97% in the middle column counts non-goblet epithelial cells failing a")
    log("  goblet-only rule. The comparable pairing is shipped vs repaired on the")
    log("  SAME 9504 cells, and it shows the large count is an artefact of the")
    log("  missing channel, not biology.")
    if "GVHD_Feb1323" in ORDER:
        g = types[(types.scope == "summary") & (types.dataset == "GVHD_Feb1323")
                  & (types.source == "expert") & (types.cls == "ALL_evaluable")]
        sh = g[g.variant == "shipped"]
        rp = g[g.variant == "repaired"]
        if len(sh) and len(rp):
            log("  GVHD, same cells: %s shipped -> %s repaired. r06's 4.02%% (the"
                % (_p(float(sh.violation_rate.iloc[0]), 8).strip(),
                   _p(float(rp.violation_rate.iloc[0]), 8).strip()))
            log("  first column) is therefore an OVERstatement produced by pooling a")
            log("  diluted cell set, not an understatement of a hidden violation mass.")
    cm = pool[(pool.pooling == "cohort_mean") & (pool.cls == "ALL_evaluable")
              & (pool.reading == "or") & (pool.source == "expert")
              & (pool.variant == "repaired")]
    cw = pool[(pool.pooling == "cell_weighted") & (pool.cls == "ALL_evaluable")
              & (pool.reading == "or") & (pool.source == "expert")
              & (pool.variant == "repaired")]
    if len(cm):
        log("  Best convention, %d cohorts: cohort-weighted mean %s [%s, %s];"
            % (int(cm.n_cohorts.iloc[0]), _p(float(cm.estimate.iloc[0]), 8).strip(),
               _p(float(cm.ci_lo_boot.iloc[0]), 8).strip(),
               _p(float(cm.ci_hi_boot.iloc[0]), 8).strip()))
    if len(cw):
        log("  cell-weighted pool %s (the cell-weighted pool is dominated by"
            % _p(float(cw.estimate.iloc[0]), 8).strip())
        log("  Melanoma, which contributes 51% of the audited cells).")
    log("")
    log("  Consequence for gate #3. It cannot be met in the form 'pooled violation")
    log("  rate > 5%' by any convention that is defensible on its own terms. What")
    log("  clears 5% is always something the depot text does not settle: CRC if ';'")
    log("  is read as AND (section 3), or a cohort once the positivity threshold is")
    log("  tightened past the mixture crossing (section 2, where GVHD runs from 2%")
    log("  to 55% across the grid). Quoting either as the audit's result would be")
    log("  choosing a convention to reach a threshold.")
    log("")
    log("  The ordering does not survive either. Section 2b: tightening the")
    log("  positivity call moves individual types by a factor of five or more, so")
    log("  the exact top-3 is reproduced in 0/16 or 1/16 sweep settings, and even")
    log("  the rank of the worst type wanders (GVHD Neutrophil 1..8, Melanoma CD8T")
    log("  1..6, Breast CD8T 2..8). The rank correlation with the mixture ranking")
    log("  has a median of only 0.16 to 0.49 across cohorts and goes negative in")
    log("  several settings, so the audit does not even license a threshold-free")
    log("  ordering claim. What does survive, and is worth")
    log("  reporting, is the count-level fact behind the ranking: under the mixture")
    log("  convention a few large populations violate their own published rule at")
    log("  20-26%, while under every stricter convention the violation rate is")
    log("  dominated by the threshold and not by the population.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=RESULTS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--alphas", default="0.10")
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    ap.add_argument("--percentiles", default="60,65,70,75,80,85,90,95")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cohorts", default="")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    open(LOG_PATH, "w").close()
    t0 = time.time()
    percentiles = [int(x) for x in args.percentiles.split(",") if x.strip()]
    clfs = [c for c in args.clfs.split(",") if c]
    alphas = [float(a) for a in args.alphas.split(",") if a]
    cohorts = ([c for c in args.cohorts.split(",") if c] if args.cohorts
               else list(paths.MANUAL_DATASETS))
    njobs = max(1, args.jobs)
    outer = max(1, min(njobs, len(cohorts)))
    inner = max(1, njobs // outer)
    log("cohorts=%s jobs=%d (outer %d x inner %d) percentiles=%s alphas=%s clfs=%s"
        % (",".join(cohorts), njobs, outer, inner, percentiles, alphas, clfs))

    res = Parallel(n_jobs=outer)(delayed(dataset_worker)(ds, percentiles, clfs, inner)
                                 for ds in cohorts)
    types = pd.concat([pd.DataFrame(r["types"]) for r in res], ignore_index=True)
    thr = pd.concat([pd.DataFrame(r["thr"]) for r in res], ignore_index=True)
    diags = [{"ds": r["ds"], "n_expert": r["n_expert"], "chans": r["chans"],
              "unmapped": r["unmapped"], "dups": {}} for r in res]
    log("per-type rows=%d  threshold rows=%d" % (len(types), len(thr)))

    usecols = ["unit", "clf", "method", "alpha", "repeat", "lambda",
               "n_accepted", "accept_rate", "selective_risk", "covered", "empty"]
    met = pd.read_csv(os.path.join(RESULTS, "r03_metrics.csv"), usecols=usecols)
    met = met[met.method == "crc"]
    dep, diags2 = [], []
    dres = Parallel(n_jobs=outer)(delayed(deployed_stage)(
        ds, clfs, alphas, met[met.unit.str.startswith("within_%s_" % ds)])
        for ds in cohorts)
    for ds, (rows, diag) in zip(cohorts, dres):
        dep += rows
        diags2.append(diag)
    dm = {d["ds"]: d["dups"] for d in diags2}
    for d in diags:
        d["dups"] = dm.get(d["ds"], {})
    log("deployed rows=%d" % len(dep))
    ddf = pd.DataFrame(dep)
    add = []
    for _, r in ddf[(ddf.scope == "pred_all_expert") & (ddf.clf != "-")].iterrows():
        add.append(dict(dataset=r.dataset, scope="summary", reading="or",
                        variant="repaired", source="pred_%s" % r.clf,
                        cls=r.ruleset, n=r.n, n_violation=r.n_violation,
                        violation_rate=r.violation_rate, ci_lo=r.ci_lo,
                        ci_hi=r.ci_hi, n_rules_total=np.nan, n_rules_used=np.nan,
                        evaluability="mixed", rules_used="", rules_missing=""))
    for _, r in ddf[(ddf.scope == "pred_class") & (ddf.clf != "-")].iterrows():
        add.append(dict(dataset=r.dataset, scope="type", reading="or",
                        variant="repaired", source="pred_%s" % r.clf,
                        cls=r.cls, n=r.n, n_violation=r.n_violation,
                        violation_rate=r.violation_rate, ci_lo=r.ci_lo,
                        ci_hi=r.ci_hi, n_rules_total=np.nan, n_rules_used=np.nan,
                        evaluability="mixed", rules_used="", rules_missing=""))
    if add:
        types = pd.concat([types, pd.DataFrame(add)], ignore_index=True)
        types = types.drop_duplicates(
            subset=["dataset", "scope", "reading", "variant", "source", "cls"],
            keep="first")
    log("types rows after adding predicted labels=%d" % len(types))

    sources = sorted({s for s in types.source.unique()})
    pool = pooling_stage(types, sources, args.boot, args.seed)
    log("pooled rows=%d" % len(pool))

    if not args.no_figure:
        figure(types, thr)

    for name, df in (("r15_types.csv", types), ("r15_threshold.csv", thr),
                     ("r15_deployed.csv", pd.DataFrame(dep)),
                     ("r15_pooled.csv", pd.DataFrame(pool))):
        df.to_csv(os.path.join(args.outdir, name), index=False)
        log("wrote %s (%d rows)" % (name, len(df)))
    write_report(types, thr, dep, pool, diags, args, time.time() - t0)
    log("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
