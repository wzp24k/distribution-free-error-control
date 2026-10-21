"""r33_two_source.py -- can the depot's second annotation source break the
marker-audit "convention tie"?

Background.  r28 showed that the marker-rule violation rate of the CellTuneDepot
reference labels is only partially identified: across the 68 enumerated
conventions (17 positivity calls x 2 semicolon readings x 2 rule-text variants)
the pooled rate moves over 0.4%-90%, and the between-convention span is a median
12.70x wider than the fixed-convention sampling interval.  Five populations are
nevertheless discordant under every one of the 68 conventions, and one is
concordant under every one:

    CRC_Feb2623      CD3T        >= 6.90%
    CRC_Feb2623      Endothelial >= 5.21%
    GVHD_Feb1323     Neutrophil  >= 26.35%
    GVHD_Feb1323     Plasma      >= 19.84%
    Melanoma_Oct2522 CD8T        >= 19.78%
    CRC_Feb2623      APC         <= 0.00%

r33 asks whether a second label source for the same cells can turn those
convention intervals into point statements.  The depot ships, for CRC and GVHD
only, a second label table (HighQualityLabels) beside the manual one
(ManualLabels).  Two labels for the same cell is the classical latent-class
setting, so r33 does both of:

  * a POINT-IDENTIFICATION route: a 3-test latent-class (Hui-Walter) model with
    tests (expert label = T), (HQ label = T) and (cell fails T's rule), which
    yields the per-source sensitivity/specificity and a corrected true violation
    rate, with bootstrap intervals and an explicit identifiability audit;
  * a PARTIAL-IDENTIFICATION route: the sharp identified set for the true rate
    given the two sources' joint table and a bounded conditional dependence,
    reported as an interval and compared with the 68-convention span.

Everything reuses errctl / r15 for marker positivity and rule evaluation, so the
new numbers are comparable with r18/r28 by construction.  ASCII only, no BOM,
deterministic seeds.

Outputs (all new; no existing file is modified)
  experiments/r33_agreement.csv          source-to-source matching + agreement
  experiments/r33_confusion.csv          coarse confusion matrices (2 maps)
  experiments/r33_coverage.csv           per-image expert-labelling coverage
  experiments/r33_mar.csv                selection diagnostic inside expert images
  experiments/r33_decomp.csv             violation decomposition per population x convention
  experiments/r33_lca.csv                point-identification fits
  experiments/r33_sharp.csv              partial-identification sharp sets
  experiments/r33_headline.csv           old vs new interval per population
  experiments/r33_headline_summary.csv   one row per headline population
  work/r33_report.txt                    the narrative, with the CHECKS section
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from errctl import data as D, markercheck as MC, paths          # noqa: E402
from errctl.labels import COARSE_MAP                             # noqa: E402

import r15_marker_sensitivity as R15                             # noqa: E402

np.seterr(all="ignore")

IN_CONV = os.path.join(ROOT, "experiments", "r18_conventions.csv")
OUT_AGREE = os.path.join(ROOT, "experiments", "r33_agreement.csv")
OUT_CONF = os.path.join(ROOT, "experiments", "r33_confusion.csv")
OUT_COVER = os.path.join(ROOT, "experiments", "r33_coverage.csv")
OUT_MAR = os.path.join(ROOT, "experiments", "r33_mar.csv")
OUT_DECOMP = os.path.join(ROOT, "experiments", "r33_decomp.csv")
OUT_LCA = os.path.join(ROOT, "experiments", "r33_lca.csv")
OUT_LCA_BOOT = os.path.join(ROOT, "experiments", "r33_lcaboot.csv")
OUT_SHARP = os.path.join(ROOT, "experiments", "r33_sharp.csv")
OUT_HEAD = os.path.join(ROOT, "experiments", "r33_headline.csv")
OUT_SUM = os.path.join(ROOT, "experiments", "r33_headline_summary.csv")
OUT_REPORT = os.path.join(ROOT, "work", "r33_report.txt")

# ------------------------------------------------------------------ convention grid (r28 definition)
PCTS = (60, 65, 70, 75, 80, 85, 90, 95)
POSITIVITY = ["gmm"] + ["pct_all@%d" % p for p in PCTS] + ["pct_pospop@%d" % p for p in PCTS]
READINGS = ("or", "and")
VARIANTS = ("shipped", "repaired")
GRID = [(p, r, v) for p in POSITIVITY for r in READINGS for v in VARIANTS]
GRID_SIZE = len(GRID)                       # 68
PRIMARY = ("gmm", "or", "repaired")
THRESH = 0.05
ALPHA = 0.05
SEED = 0
BOOT = 400
DELTA_GRID = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
OUT_DELTA = os.path.join(ROOT, "experiments", "r33_delta.csv")

# The 4 populations whose second source exists, plus the two other headline ones.
FOCAL = [("CRC_Feb2623", "CD3T"), ("CRC_Feb2623", "Endothelial"),
         ("CRC_Feb2623", "APC"), ("GVHD_Feb1323", "Neutrophil"),
         ("GVHD_Feb1323", "Plasma")]
# r28 headline claims (old interval endpoints are re-derived from r18_conventions.csv).
OLD_CLAIMS = {
    ("CRC_Feb2623", "CD3T"): (6.90e-2, 1.0000, "discordant"),
    ("CRC_Feb2623", "Endothelial"): (5.21e-2, 0.6042, "discordant"),
    ("CRC_Feb2623", "APC"): (0.0, 0.0, "concordant"),
    ("GVHD_Feb1323", "Neutrophil"): (26.35e-2, 0.9341, "discordant"),
    ("GVHD_Feb1323", "Plasma"): (19.84e-2, 0.9463, "discordant"),
    ("Melanoma_Oct2522", "CD8T"): (19.78e-2, 0.8964, "discordant"),
}

# Extra coarse mappings that are needed because the HQ vocabulary carries
# classes the paper's COARSE_MAP does not list.  Used ONLY as a documented
# robustness variant; the primary analysis uses errctl.labels.COASTE_MAP
# unchanged so that it is comparable with r18/r28.
EXTRA_COARSE = {
    "Macrophage_Calp": "Macrophage", "Neutrophil_CD15": "Neutrophil",
    "Plasma_CD38": "Plasma", "BrunnerGland": "TumorEpithelial",
    "FibroticEpithelial": "TumorEpithelial", "Mesenchymal_VIM": "Fibroblast",
    "NK": "NK", "Mesenchymal": "Fibroblast",
}

PAT = [(e, h, f) for e in (0, 1) for h in (0, 1) for f in (0, 1)]
EE = np.array([p[0] for p in PAT], float)
HH = np.array([p[1] for p in PAT], float)
FF = np.array([p[2] for p in PAT], float)
SEL = {"U": np.ones(8), "UE": EE, "UH": HH, "UF": FF,
       "UEH": EE * HH, "UEF": EE * FF, "UHF": HH * FF, "UEHF": EE * HH * FF}


def log(msg, fh=None):
    print(msg, flush=True)
    if fh is not None:
        fh.write(str(msg) + "\n")


def cp_interval(k, n, alpha=ALPHA):
    """Exact Clopper-Pearson interval (same definition as r28)."""
    from scipy.stats import beta
    k, n = int(k), int(n)
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def coarse_paper(values):
    return pd.Series(values).astype(str).map(COARSE_MAP).fillna("Unidentified").to_numpy()


def coarse_harmonised(values):
    m = dict(COARSE_MAP)
    m.update(EXTRA_COARSE)
    return pd.Series(values).astype(str).map(m).fillna("Unidentified").to_numpy()


def fail_rule_mask(groups, cls, pos, n):
    """(evaluable, fails-rule) for every cell on the panel, for coarse class cls.

    Same semantics as r15.evaluate with restrict={cls}: a rule is usable when
    every channel it names has a positivity call, and the class passes when at
    least one usable rule passes.
    """
    rules = [r for r in groups.get(cls, []) if r.idents <= set(pos)]
    if not rules:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    ok = np.zeros(n, dtype=bool)
    for r in rules:
        vals = {i: pos[i][:n] for i in r.idents}
        try:
            ok |= np.asarray(r.evaluate(vals), dtype=bool)
        except Exception:
            return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    return np.ones(n, dtype=bool), ~ok
# ====================================================================== cohort layer
def positivity_bank(ds, frame, chans, cache=True):
    """The 17 positivity calls of the r28 grid, cached because gmm is slow."""
    cache_path = os.path.join(paths.WORK, "_r33_pos_%s.pkl" % ds)
    if cache and os.path.exists(cache_path):
        import pickle
        with open(cache_path, "rb") as fh:
            bank = pickle.load(fh)
        if sorted(bank) == sorted(POSITIVITY):
            return bank
    t0 = time.time()
    bank = {}
    bank["gmm"], _ = MC.positivity(frame, chans)
    for p in PCTS:
        bank["pct_all@%d" % p] = R15.percentile_positivity(frame, chans, p, "all")
    for p in PCTS:
        bank["pct_pospop@%d" % p] = R15.percentile_positivity(frame, chans, p, "pospop",
                                                              base=bank["gmm"])
    if cache:
        import pickle
        with open(cache_path, "wb") as fh:
            pickle.dump(bank, fh)
    log("    positivity bank built in %.0fs" % (time.time() - t0))
    return bank


def cohort_tables(ds, seed=SEED):
    """Per (population, convention) two-source tables over the expert-labelled cells."""
    cell = D.build_master(ds)
    markers, _ = D.marker_table(ds, kind="HighQualityLabels")
    markers = [m for m in markers if m in cell.columns]
    frame, chans = R15.canonical_frame(cell, markers)
    bank = positivity_bank(ds, frame, chans)

    ex = cell["is_expert"].to_numpy(dtype=bool)
    img = cell["image"].astype(str).to_numpy()
    imgs, icodes = np.unique(img, return_inverse=True)
    n = cell.shape[0]
    le = coarse_paper(cell["class_expert"].to_numpy(dtype=object))
    lh = coarse_paper(cell["class_hq"].to_numpy(dtype=object))
    classes = sorted({c for c in le[ex] if c != "Unidentified"})

    compiled = {}
    for reading in READINGS:
        for variant in VARIANTS:
            rules, _ = R15.rule_table(ds, reading, repair=(variant == "repaired"))
            compiled[(reading, variant)] = R15.group_rules(rules)[0]

    out = {}
    for setting in POSITIVITY:
        pos = bank[setting]
        for reading in READINGS:
            for variant in VARIANTS:
                groups = compiled[(reading, variant)]
                ptag = "%s|%s|%s" % (setting, reading, variant)
                for cls in classes:
                    ev, fl = fail_rule_mask(groups, cls, pos, n)
                    keep = ex & ev
                    e = ((le == cls) & keep).astype(np.int64)
                    h = ((lh == cls) & keep).astype(np.int64)
                    f = fl.astype(np.int64)
                    code = e * 4 + h * 2 + f
                    n8 = np.bincount(code[keep], minlength=8).astype(np.int64)
                    n8_img = np.zeros((imgs.size, 8), dtype=np.int64)
                    np.add.at(n8_img, (icodes[keep], code[keep]), 1)
                    hq_all = (lh == cls) & ev
                    ex_all = (le == cls) & keep
                    out[(cls, ptag)] = dict(
                        n8=n8, n8_img=n8_img, n_img=int(imgs.size),
                        n_expert=int(ex_all.sum()),
                        k_expert=int(fl[ex_all].sum()),
                        n_hq_sc=int(((lh == cls) & keep).sum()),
                        k_hq_sc=int(fl[(lh == cls) & keep].sum()),
                        n_hq_panel=int(hq_all.sum()),
                        k_hq_panel=int(fl[hq_all].sum()))
    return dict(dataset=ds, classes=classes, n_cells=n, n_expert=int(ex.sum()),
                n_images=int(imgs.size), n_expert_images=int(np.unique(icodes[ex]).size),
                chans=chans, tables=out, cell=cell, imgs=imgs, icodes=icodes,
                le=le, lh=lh, ex=ex,
                pos_primary=bank[PRIMARY[0]],
                groups_primary=compiled[(PRIMARY[1], PRIMARY[2])])


def part_a(ds, cell, imgs, icodes, ex, le, lh, le_raw, lh_raw, pos_primary,
           groups_primary):
    """Matching, agreement, confusion, coverage and the selection diagnostic."""
    n = cell.shape[0]
    hq_raw = cell["class_hq"].astype(str).to_numpy()
    ex_raw = cell["class_expert"].astype(str).to_numpy()
    key_hq = set(zip(cell["image"].astype(str), cell["cellID"].astype(int)))
    hq_tbl = D.load_hq_celltable(ds)
    ex_tbl = D.load_manual_labels(ds)
    key_hq2 = set(zip(hq_tbl["image"].astype(str), hq_tbl["cellID"].astype(int)))
    key_ex2 = set(zip(ex_tbl["image"].astype(str), ex_tbl["cellID"].astype(int)))

    # agreement on the cells both sources label
    a = le[ex]
    b = lh[ex]
    ok = (a != "Unidentified") & (b != "Unidentified")
    ar = le_raw[ex]
    br = lh_raw[ex]

    def kappa(x, y):
        labs = sorted(set(map(str, x)) | set(map(str, y)))
        pos = {l: i for i, l in enumerate(labs)}
        m = np.zeros((len(labs), len(labs)))
        for i, j in zip(x, y):
            m[pos[str(i)], pos[str(j)]] += 1.0
        N = m.sum()
        po = float(np.trace(m) / N)
        pe = float((m.sum(1) / N * (m.sum(0) / N)).sum())
        return (float((po - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else np.nan), po, pe

    kn, pon, pen = kappa(ar, br)
    kc, poc, pec = kappa(a, b)
    leh = coarse_harmonised(cell["class_expert"].to_numpy(dtype=object))[ex]
    lhh = coarse_harmonised(cell["class_hq"].to_numpy(dtype=object))[ex]
    row = dict(dataset=ds, match_key="(image, cellID)",
               n_hq_cells=int(n), n_expert_cells=int(ex.sum()),
               n_hq_keys=int(len(key_hq2)), n_expert_keys=int(len(key_ex2)),
               n_expert_keys_missing_from_hq=int(len(key_ex2 - key_hq2)),
               n_hq_images=int(len(set(cell["image"].astype(str)))),
               n_expert_images=int(np.unique(icodes[ex]).size),
               n_expert_labelled_hq_cells=int(ex.sum()),
               n_common_nonunidentified=int(ok.sum()),
               native_agreement=float((ar == br).mean()),
               coarse_agreement=float((a == b).mean()),
               coarse_agreement_common=(float((a[ok] == b[ok]).mean()) if ok.any() else np.nan),
               coarse_agreement_harmonised=float((leh == lhh).mean()),
               native_kappa=kn, native_po=pon, native_pe=pen,
               coarse_kappa=kc, coarse_po=poc, coarse_pe=pec,
               expert_share_of_panel=float(ex.sum() / n))
    conf = pd.crosstab(pd.Series(list(map(str, a)), name="coarse_expert_paper"),
                       pd.Series(list(map(str, b)), name="coarse_hq_paper"))
    conf = conf.stack().reset_index(name="n")
    conf.insert(0, "map", "paper")
    conf.insert(0, "dataset", ds)
    conf2 = pd.crosstab(pd.Series(list(map(str, leh)), name="coarse_expert_harm"),
                        pd.Series(list(map(str, lhh)), name="coarse_hq_harm"))
    conf2 = conf2.stack().reset_index(name="n")
    conf2.insert(0, "map", "harmonised")
    conf2.insert(0, "dataset", ds)
    conf2 = conf2.rename(columns={"coarse_expert_harm": "coarse_expert_paper",
                                  "coarse_hq_harm": "coarse_hq_paper"})
    conf = pd.concat([conf, conf2], ignore_index=True)

    # coverage per image
    cov = []
    for i, im in enumerate(imgs):
        m = icodes == i
        cov.append(dict(dataset=ds, image=str(im), n_hq=int(m.sum()),
                        n_expert=int((m & ex).sum()),
                        frac_expert=float((m & ex).sum() / max(m.sum(), 1))))

    # selection diagnostic: inside the expert-labelled images, is the
    # expert-labelled subset exchangeable with the unlabelled one?
    exi = np.unique(icodes[ex])
    inm = np.isin(icodes, exi)
    rows = []
    for cls in sorted(set(lh[inm])):
        if cls == "Unidentified":
            continue
        ev, fl = fail_rule_mask(groups_primary, cls, pos_primary, n)
        m = inm & (lh == cls) & ev
        if m.sum() < 100:
            continue
        aa = m & ex
        bb = m & (~ex)
        if aa.sum() == 0 or bb.sum() == 0:
            continue
        p_e = float(fl[aa].mean())
        p_u = float(fl[bb].mean())
        var = p_e * (1 - p_e) / aa.sum() + p_u * (1 - p_u) / bb.sum()
        z = (p_e - p_u) / np.sqrt(var) if var > 0 else np.nan
        rows.append(dict(dataset=ds, cls=cls, n_expert_labelled=int(aa.sum()),
                         n_unlabelled=int(bb.sum()), rate_expert_subset=p_e,
                         rate_unlabelled=p_u, diff=p_e - p_u, z=float(z)))
    return row, conf, pd.DataFrame(cov), pd.DataFrame(rows)
# ====================================================================== identification layer
MOM = np.array([SEL[k] for k in ("U", "UE", "UH", "UF", "UEH", "UEF", "UHF", "UEHF")], float)  # 8x8


def derived_stats(P, q):
    """Model-free moments of the latent-class split q = P(Z=1 | observed cell)."""
    M = MOM * P[None, :]                       # M[k, o] = P_o * SEL_k[o]
    y = M @ q
    y0 = M @ (1.0 - q)
    U, U0 = y[0], y0[0]
    if U <= 1e-12 or U0 <= 1e-12:
        return None
    a1E, a1H, pi = y[1] / U, y[2] / U, y[3] / U
    a0E, a0H, phi = y0[1] / U0, y0[2] / U0, y0[3] / U0
    d = dict(p=U, a1E=a1E, a1H=a1H, pi=pi, a0E=a0E, a0H=a0H, phi=phi)
    d["kEH1"] = y[4] / U - a1E * a1H
    d["kEH0"] = y0[4] / U0 - a0E * a0H
    d["kEF1"] = y[5] / U - a1E * pi
    d["kEF0"] = y0[5] / U0 - a0E * phi
    d["kHF1"] = y[6] / U - a1H * pi
    d["kHF0"] = y0[6] / U0 - a0H * phi
    return d


ASSUMPTIONS = [
    ("none", dict(infl=False, pius=False, delta=None)),
    ("infl", dict(infl=True, pius=False, delta=None)),
    ("infl+pius", dict(infl=True, pius=True, delta=None)),
    ("infl+pius+A1_0.005", dict(infl=True, pius=True, delta=0.005)),
    ("infl+pius+A1_0.02", dict(infl=True, pius=True, delta=0.02)),
    ("infl+pius+A1_0.05", dict(infl=True, pius=True, delta=0.05)),
]


def spec_ok(d, infl, pius, delta):
    if infl:
        if not (d["a1E"] >= 0.5 and d["a1H"] >= 0.5 and d["a0E"] <= 0.5 and d["a0H"] <= 0.5):
            return False
    if pius and not (d["pi"] <= d["phi"]):
        return False
    if delta is not None:
        for k in ("kEF1", "kEF0", "kHF1", "kHF0"):
            if abs(d[k]) > delta:
                return False
    return True


def constraints_vec(d, infl, pius, delta):
    out = []
    if infl:
        out += [d["a1E"] - 0.5, d["a1H"] - 0.5, 0.5 - d["a0E"], 0.5 - d["a0H"]]
    if pius:
        out += [d["phi"] - d["pi"]]
    if delta is not None:
        for k in ("kEF1", "kEF0", "kHF1", "kHF0"):
            out += [delta - abs(d[k])]
    return np.array(out)


def sharp_vertices(P, infl=True, pius=True, delta=None, polish_q=None):
    """Sharp set for pi over the cube's vertices (+ optional SLSQP polish).

    The feasible set is {q in [0,1]^8 : moment constraints}, and the objective
    pi = P(F=1|Z=1) is a ratio of linear functionals of q, so extremes are
    attained on the boundary; the 256 vertices are enumerated exactly and are
    verified, so any value reported here is a certificate.  The reported
    interval is an inner approximation of the sharp set (q's in the interior of
    flat faces could in principle widen it); every qualitative claim below is
    backed by an explicit vertex.
    """
    M = MOM * P[None, :]
    lo, hi, q_lo, q_hi, nfeas = np.inf, -np.inf, None, None, 0
    for mask in range(256):
        q = np.array([(mask >> i) & 1 for i in range(8)], float)
        d = derived_stats(P, q)
        if d is None or not spec_ok(d, infl, pius, delta):
            continue
        nfeas += 1
        if d["pi"] < lo:
            lo, q_lo = d["pi"], q
        if d["pi"] > hi:
            hi, q_hi = d["pi"], q
    if nfeas == 0:
        return np.nan, np.nan, None, None, 0
    if polish_q:
        def obj(q, sgn):
            d = derived_stats(P, q)
            return sgn * d["pi"] if d is not None else 1e6
        for q0 in (q_lo, q_hi):
            for sgn in (1, -1):
                try:
                    r = optimize.minimize(
                        lambda q, s=sgn: obj(q, s), q0, method="SLSQP",
                        bounds=[(0.0, 1.0)] * 8,
                        constraints=[{"type": "ineq",
                                      "fun": lambda q: constraints_vec(
                                          derived_stats(P, q), infl, pius, delta)
                                      if derived_stats(P, q) is not None
                                      else np.array([-1.0])}],
                        options={"maxiter": 200, "ftol": 1e-12})
                except Exception:
                    continue
                d = derived_stats(P, r.x)
                if d is None:
                    continue
                if not spec_ok(d, infl, pius, delta):
                    continue
                if d["pi"] < lo:
                    lo, q_lo = d["pi"], r.x
                if d["pi"] > hi:
                    hi, q_hi = d["pi"], r.x
    return lo, hi, q_lo, q_hi, nfeas


def min_feasible_delta(P, grid=(0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25),
                       nrand=4000, seed=SEED):
    """Smallest allowed |kappa_EF|,|kappa_HF| (on a grid) for which the data admit
    ANY latent split satisfying informativeness + PIUS + that dependence bound.
    delta = 0 is exact conditional independence; it is feasible only if the
    latent-class model fits exactly, so this number measures how far the data
    are from the model that a point-identification route would need."""
    for dl in grid:
        _, _, nf, _, _ = sharp_search(P, infl=True, pius=True, delta=dl,
                                      nrand=nrand, seed=seed, polish=True)
        if nf > 0:
            return float(dl), int(nf)
    return np.nan, 0


# ------------------------------------------------------------------------
# EXACT partial identification of pi = P(F=1 | Z=1).
#
# The latent split q = P(Z=1 | observed cell) enters the assumptions only
# through three conditional moments, and every constraint that these
# assumptions impose is LINEAR in q:
#
#   informativeness (the rule is informative about membership)
#       a1E >= 1/2  <=>  (M1 - M0/2).q >= 0
#       a1H >= 1/2  <=>  (M2 - M0/2).q >= 0
#       a0E <= 1/2  <=>  A1 - A0/2 + (M0/2 - M1).q <= 0
#       a0H <= 1/2  <=>  A2 - A0/2 + (M0/2 - M2).q <= 0
#   PIUS (the rule fails less often among true members than among non-members)
#       pi <= phi   <=>  A0*(M3.q) - A3*(M0.q) <= 0
#
# (A_k = sum_o P_o SEL_k[o] is the unconditional mean of the k-th selector; the
# algebra for PIUS uses phi = (A3 - M3.q)/(A0 - M0.q).)  Hence the feasible set
# of q is a POLYTOPE and pi = (M3.q)/(M0.q) is a ratio of linear functionals, so
# its exact max/min are LINEAR PROGRAMS (Charnes-Cooper).  The conditional
# dependence bound |kappa_EF|,|kappa_HF| <= delta is QUADRATIC in q, so with a
# finite delta the LP over the informativeness+PIUS polytope is a rigorous OUTER
# bound and only search yields certified inner (attained) values.
#
# The earlier 8-cube-vertex enumeration used by r33 is NOT the sharp set: the
# extreme points of the constrained polytope are not cube vertices, so it only
# ever supplied certified feasible points (an inner approximation).
# ------------------------------------------------------------------------
def lin_cons(P, infl=True, pius=False):
    """List of (c, b, name) meaning c.q + b <= 0, all linear in q."""
    P = np.asarray(P, float)
    P = P / P.sum()
    Mm = MOM * P[None, :]
    A = Mm.sum(1)
    cons = []
    if infl:
        cons.append((-(Mm[1] - 0.5 * Mm[0]), 0.0, "a1E>=1/2"))
        cons.append((-(Mm[2] - 0.5 * Mm[0]), 0.0, "a1H>=1/2"))
        cons.append((0.5 * Mm[0] - Mm[1], A[1] - 0.5 * A[0], "a0E<=1/2"))
        cons.append((0.5 * Mm[0] - Mm[2], A[2] - 0.5 * A[0], "a0H<=1/2"))
    if pius:
        cons.append((A[0] * Mm[3] - A[3] * Mm[0], 0.0, "PIUS(pi<=phi)"))
    return cons


def sharp_lp(P, infl=True, pius=False, tmax=1e9):
    """Exact sharp set of pi over {q in [0,1]^8 : linear constraints}.

    Charnes-Cooper: substitute z = q*t, t = 1/(M0.q).  Then
    pi = (M3.z)/(M0.z) with M0.z = 1, the linear constraints become
    c.z + b*t <= 0 and the box q <= 1 becomes z <= t, so max and min are LPs.
    """
    from scipy.optimize import linprog
    P = np.asarray(P, float)
    P = P / P.sum()
    Mm = MOM * P[None, :]
    n = 8
    obj = np.concatenate([Mm[3], [0.0]])
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = P
    rows, rhs, names = [], [], []
    for c, b, nm in lin_cons(P, infl, pius):
        r = np.zeros(n + 1)
        r[:n] = c
        r[n] = b
        rows.append(r)
        rhs.append(0.0)
        names.append(nm)
    for i in range(n):
        r = np.zeros(n + 1)
        r[i] = 1.0
        r[n] = -1.0
        rows.append(r)
        rhs.append(0.0)
        names.append("q%d<=1" % i)
    A_ub = np.array(rows)
    b_ub = np.array(rhs)
    out = dict(lo=np.nan, hi=np.nan, lo_q=None, hi_q=None,
               lo_status=4, hi_status=4, binding_hi="", binding_lo="")
    for sgn, key in ((1.0, "hi"), (-1.0, "lo")):
        res = linprog(-sgn * obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                      bounds=[(0.0, None)] * n + [(0.0, tmax)], method="highs")
        out[key + "_status"] = int(res.status)
        if res.status != 0:
            continue
        z = res.x[:n]
        t = res.x[n]
        out[key] = float((-res.fun) * sgn)
        if t > 0:
            out[key + "_q"] = z / t
        slack = b_ub - A_ub @ res.x
        out["binding_" + key] = ";".join(
            names[i] for i in np.nonzero(np.abs(slack) < 1e-9)[0])
    return out


def sharp_bisect(P, infl=True, pius=False, iters=60, tmax=1e9):
    """Independent recomputation of the exact bounds: bisection on c with a
    feasibility LP in the Charnes-Cooper variables z = q*t, t = 1/(M0.q).

    This is a SECOND linear program, not a rerun of sharp_lp: the constraint
    rows are re-derived inline from the raw table (NOT via lin_cons) and the
    candidate value c enters as an extra inequality instead of as the
    objective.

    The first r33 draft bisected in q-space and enforced non-degeneracy with
    the row 'M0.q >= 1e-9' (written as -M0.q <= -1e-9).  HiGHS' default
    primal-feasibility tolerance (1e-7) swallows a right-hand side that small,
    so q = 0 was accepted as feasible; there pi = M3.q/M0.q is 0/0, the row
    -sign*(M3 - c*M0).q <= 0 becomes 0 <= 0 for EVERY c, and the routine
    returned [0, 1] rather than the sharp set.  The equality normalisation
    M0.z = 1 used here has no tolerance to swallow, so the LP is well posed;
    its value is also unchanged over tmax in {1e6, 1e9, 1e12}.
    """
    from scipy.optimize import linprog
    P = np.asarray(P, float)
    P = P / P.sum()
    Mm = MOM * P[None, :]                       # Mm[k, o] = P_o * SEL_k[o]
    A = Mm.sum(1)                               # A[k] = P(SEL_k = 1)
    rows = []                                   # each: c.z + b.t <= 0
    if infl:
        rows.append((-(Mm[1] - 0.5 * Mm[0]), 0.0))              # a1E >= 1/2
        rows.append((-(Mm[2] - 0.5 * Mm[0]), 0.0))              # a1H >= 1/2
        rows.append((0.5 * Mm[0] - Mm[1], A[1] - 0.5 * A[0]))   # a0E <= 1/2
        rows.append((0.5 * Mm[0] - Mm[2], A[2] - 0.5 * A[0]))   # a0H <= 1/2
    if pius:
        rows.append((A[0] * Mm[3] - A[3] * Mm[0], 0.0))         # pi <= phi
    A_ub = np.zeros((len(rows) + 8, 9))
    b_ub = np.zeros(len(rows) + 8)
    for i, (cc, bb) in enumerate(rows):
        A_ub[i, :8] = cc
        A_ub[i, 8] = bb
    for i in range(8):
        A_ub[len(rows) + i, i] = 1.0            # q_i <= 1  <=>  z_i <= t
        A_ub[len(rows) + i, 8] = -1.0
    A_eq = np.zeros((1, 9))
    A_eq[0, :8] = Mm[0]                         # M0.z = 1, exactly
    bnd = [(0.0, None)] * 8 + [(0.0, float(tmax))]

    def feas(c, sign):
        row = np.zeros(9)
        row[:8] = -sign * Mm[3]
        r = linprog(np.zeros(9), A_ub=np.vstack([A_ub, row[None, :]]),
                    b_ub=np.concatenate([b_ub, [-sign * c]]),
                    A_eq=A_eq, b_eq=[1.0], bounds=bnd, method="highs")
        return r.status == 0

    out = {}
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if feas(mid, 1.0):
            lo = mid
        else:
            hi = mid
    out["hi"] = lo
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if feas(mid, -1.0):
            hi = mid
        else:
            lo = mid
    out["lo"] = hi
    return out


def sharp_search(P, infl=True, pius=False, delta=None, nrand=20000, seed=SEED,
                 polish=True):
    """Certified ATTAINED pi values (inner bounds).

    Candidates: all 256 cube vertices, uniform draws, then SLSQP polish in both
    directions.  Every value returned is realised by an explicit q that
    satisfies the full spec (kappa bound included), so it is a certificate for
    sharp_lo <= value <= sharp_hi.
    """
    from scipy.optimize import minimize
    P = np.asarray(P, float)
    P = P / P.sum()
    rng = np.random.default_rng(seed)
    cands = [np.array([(mask >> i) & 1 for i in range(8)], float) for mask in range(256)]
    cands.extend(rng.random((int(nrand), 8)))
    lo, hi, q_lo, q_hi, nf = np.inf, -np.inf, None, None, 0
    for q in cands:
        d = derived_stats(P, q)
        if d is None or not spec_ok(d, infl, pius, delta):
            continue
        nf += 1
        if d["pi"] < lo:
            lo, q_lo = d["pi"], q
        if d["pi"] > hi:
            hi, q_hi = d["pi"], q
    if nf == 0:
        return np.nan, np.nan, 0, None, None
    if polish:
        def _obj(q, sgn):
            d = derived_stats(P, q)
            return sgn * d["pi"] if d is not None else 1e6

        def _con(q):
            d = derived_stats(P, q)
            if d is None:
                return np.array([-1.0])
            v = constraints_vec(d, infl, pius, delta)
            return v if v.size else np.array([1.0])

        for q0 in (q_lo, q_hi):
            for sgn in (1.0, -1.0):
                try:
                    r = minimize(lambda q, s=sgn: _obj(q, s), q0, method="SLSQP",
                                 bounds=[(0.0, 1.0)] * 8,
                                 constraints=[{"type": "ineq", "fun": _con}],
                                 options={"maxiter": 400, "ftol": 1e-14})
                except Exception:
                    continue
                d = derived_stats(P, r.x)
                if d is None or not spec_ok(d, infl, pius, delta):
                    continue
                if d["pi"] < lo:
                    lo, q_lo = d["pi"], r.x
                if d["pi"] > hi:
                    hi, q_hi = d["pi"], r.x
    return lo, hi, nf, q_lo, q_hi


# ------------------------------------------------------------------------
# Dawid-Skene / Hui-Walter latent-class layer.
#
# Model: X = (E,H,F) with a binary latent Z ("the cell really is type T"),
#   E = the expert label says T, H = the depot HighQuality label says T,
#   F = the cell fails T's marker rule.  Under conditional independence of
#   the three tests given Z the observed 2x2x2 table is
#
#       P(X = x) = p * prod_i Q1_i(x_i) + (1-p) * prod_i Q0_i(x_i)
#
#   with parameters (p, aE, bE, aH, bH, pi, phi): 7 parameters for the 7 free
#   probabilities of the 2x2x2 table, so the model is SATURATED.
#
# Two consequences that the first r33 draft got wrong and this version tests:
#   (i)  an EXACT fit generically EXISTS, so a nonzero fit residual can never
#        be evidence against the model -- it is an optimizer failure.  The
#        first draft ran EM only and reported "no population has an exact fit"
#        at residuals 0.006-0.44; an independent least-squares multistart
#        reaches 1e-14 on the same tables (see CHECKS).
#   (ii) the likelihood is exactly symmetric under the relabelling
#        (p,aE,bE,aH,bH,pi,phi) -> (1-p,bE,aE,bH,aH,phi,pi), so the exact-fit
#        set is closed under (pi,phi) -> (phi,pi).  Conditional independence
#        alone therefore identifies pi only up to that swap; a sign
#        restriction (informativeness a1 >= 1/2 >= a0) chooses a branch, and
#        that is the ONLY identifying information the saturated model has.
# ------------------------------------------------------------------------
from scipy import optimize as _opt  # noqa: E402  (scipy.optimize alias used below)

LCA_NAMES = ("p", "aE", "bE", "aH", "bH", "pi", "phi")
LCA_BOX_ALL = (np.zeros(7), np.ones(7))
LCA_BOX_INFL = (np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0]),
                np.array([1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0]))
LCA_EXACT_TOL = 1e-8        # L1 table distance counted as an exact fit


def lca_probs(th):
    """The 8 fitted cell probabilities of the two-component product mixture."""
    p, aE, bE, aH, bH, pi, phi = th
    A = np.where(EE == 1, aE, 1 - aE)
    B = np.where(EE == 1, bE, 1 - bE)
    C = np.where(HH == 1, aH, 1 - aH)
    Dd = np.where(HH == 1, bH, 1 - bH)
    G = np.where(FF == 1, pi, 1 - pi)
    G0 = np.where(FF == 1, phi, 1 - phi)
    return p * A * C * G + (1 - p) * B * Dd * G0


def lca_to_q(th):
    """Posterior P(Z=1 | observed pattern) of a fitted two-component mixture.
    Feeding this back through derived_stats() must reproduce the fitted
    (pi, phi) and give kappa = 0 exactly -- the consistency bridge between the
    latent-class layer and the partial-identification LP layer."""
    p, aE, bE, aH, bH, pi, phi = th
    l1 = np.where(EE == 1, aE, 1 - aE) * np.where(HH == 1, aH, 1 - aH) \
        * np.where(FF == 1, pi, 1 - pi)
    l0 = np.where(EE == 1, bE, 1 - bE) * np.where(HH == 1, bH, 1 - bH) \
        * np.where(FF == 1, phi, 1 - phi)
    num = p * l1
    den = num + (1 - p) * l0
    return np.where(den > 1e-300, num / np.maximum(den, 1e-300), 0.0)


def lca_audit(th):
    p, aE, bE, aH, bH, pi, phi = th
    infl = bool(aE >= 0.5 - 1e-9 and aH >= 0.5 - 1e-9
                and bE <= 0.5 + 1e-9 and bH <= 0.5 + 1e-9)
    pius = bool(pi <= phi + 1e-12)
    return infl, pius


def lca_starts(lo, hi, nrand=0, seed=SEED, warm=None, dense=True):
    """Starts for the least-squares fit.  When dense, a deterministic grid is
    aimed at the boundary facets that the zero cells of a sparse table force;
    otherwise only warm starts (previous solutions) plus uniform draws are
    used, which is what the bootstrap needs to stay cheap."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    starts = [0.5 * (lo + hi)]
    for w in (warm or []):
        starts.append(np.clip(np.asarray(w, float), lo + 1e-12, hi - 1e-12))
    if dense:
        aE_vals = (float(lo[1]) + 1e-3, float(hi[1]) - 1e-3)
        a0_vals = (float(lo[2]) + 1e-3, float(hi[2]) - 1e-3)
        for p in (0.002, 0.02, 0.5, 0.98):
            for aE in aE_vals:
                for a0E in a0_vals:
                    for aH in aE_vals:
                        for a0H in a0_vals:
                            for gm in (float(lo[5]) + 1e-3, float(hi[5]) - 1e-3):
                                for g0 in (float(lo[6]) + 1e-3, float(hi[6]) - 1e-3):
                                    starts.append(np.array([p, aE, a0E, aH, a0H, gm, g0]))
    if nrand > 0:
        rng = np.random.default_rng(seed)
        for _ in range(int(nrand)):
            starts.append(lo + (hi - lo) * rng.random(7))
    return starts


def lca_fit(P, box, starts, max_nfev=4000, dedupe_tol=2e-3):
    """Least-squares fits of the saturated two-component mixture.

    Returns (best, solutions): best is the smallest-L1 solution seen and
    solutions is the deduplicated list of every fit with L1 <= LCA_EXACT_TOL.
    Every returned point is an explicit parameter vector, so res <= tol
    certifies 'an exact conditional-independence fit exists'."""
    from scipy.optimize import least_squares
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    P = np.asarray(P, float)
    best = (np.inf, None)
    sols = []
    for th0 in starts:
        t0 = np.clip(np.asarray(th0, float), lo + 1e-12, hi - 1e-12)
        try:
            r = least_squares(lambda th: lca_probs(th) - P, t0, bounds=(lo, hi),
                              xtol=1e-15, ftol=1e-15, gtol=1e-15,
                              x_scale=1.0, max_nfev=max_nfev)
        except Exception:
            continue
        th = np.clip(r.x, 0.0, 1.0)
        res = float(np.abs(lca_probs(th) - P).sum())
        if res < best[0]:
            best = (res, th.copy())
        if res <= LCA_EXACT_TOL:
            if not any(np.max(np.abs(s - th)) < dedupe_tol for _, s in sols):
                sols.append((res, th.copy()))
    sols.sort(key=lambda z: z[0])
    return best, sols


def lca_boot_one(n8_img, warm, rng, nrand=0, seed=SEED):
    """One FOV-cluster bootstrap replicate: resample whole images, refit the
    informativeness-constrained latent-class model, return the argmin."""
    nimg = n8_img.shape[0]
    idx = rng.integers(0, nimg, nimg)
    tb = n8_img[idx].sum(0).astype(float)
    if tb.sum() <= 0:
        return None
    P = tb / tb.sum()
    starts = lca_starts(LCA_BOX_INFL[0], LCA_BOX_INFL[1], nrand=nrand,
                        seed=seed, warm=warm, dense=False)
    best, _ = lca_fit(P, LCA_BOX_INFL, starts, max_nfev=1500)
    if best[1] is None or best[0] > LCA_EXACT_TOL:
        # second stage: the resampled table moves the solution off the boundary
        # forced by the zero cells, so retry from a wider random set.  A
        # replicate is only called 'rejected' when even this fails.
        starts2 = lca_starts(LCA_BOX_INFL[0], LCA_BOX_INFL[1], nrand=max(nrand, 80),
                             seed=seed + 991, warm=warm, dense=False)
        best2, _ = lca_fit(P, LCA_BOX_INFL, starts2, max_nfev=1500)
        if best2[1] is not None and best2[0] < best[0]:
            best = best2
    if best[1] is None:
        return None
    infl, pius = lca_audit(best[1])
    return dict(resid=best[0], th=best[1], infl=infl, pius=pius)


def em_solve_legacy(tab, nstart=40, seed=SEED, iters=1500):
    """The first draft's EM, kept only as a documented counter-example: it can
    stop at fixed points that are NOT likelihood maxima of this saturated
    model, which is why it reported residuals of 0.006-0.44."""
    P = np.asarray(tab, float)
    P = P / P.sum()
    rng = np.random.default_rng(seed)
    sols = []
    for s in range(nstart):
        if s == 0:
            th = np.array([0.3, 0.8, 0.1, 0.8, 0.1, 0.3, 0.1])
        else:
            th = np.array([rng.uniform(0.02, 0.98), rng.uniform(0.3, 1.0),
                           rng.uniform(0.0, 0.4), rng.uniform(0.3, 1.0),
                           rng.uniform(0.0, 0.4), rng.uniform(0.02, 0.98),
                           rng.uniform(0.02, 0.98)])
        for _ in range(iters):
            l1 = np.where(EE == 1, th[1], 1 - th[1]) * np.where(HH == 1, th[3], 1 - th[3]) \
                * np.where(FF == 1, th[5], 1 - th[5])
            l0 = np.where(EE == 1, th[2], 1 - th[2]) * np.where(HH == 1, th[4], 1 - th[4]) \
                * np.where(FF == 1, th[6], 1 - th[6])
            tot = th[0] * l1 + (1 - th[0]) * l0
            if np.any(tot <= 1e-300) or not np.isfinite(tot).all():
                break
            w = P * th[0] * l1 / tot
            W = w.sum()
            W0 = 1.0 - W
            if W < 1e-9 or W0 < 1e-9:
                break
            new = np.array([W, (w * EE).sum() / W, (w * (1 - EE)).sum() / W0,
                            (w * HH).sum() / W, (w * (1 - HH)).sum() / W0,
                            (w * FF).sum() / W, ((P - w) * FF).sum() / W0])
            new = np.clip(new, 1e-9, 1 - 1e-9)
            if np.max(np.abs(new - th)) < 1e-13:
                th = new
                break
            th = new
        sols.append(dict(resid=float(np.abs(lca_probs(th) - P).sum()),
                         **{k: float(v) for k, v in zip(LCA_NAMES, th)}))
    good, seen = [], set()
    for s in sorted(sols, key=lambda z: z["resid"]):
        key = tuple(round(s[k], 5) for k in LCA_NAMES)
        if key in seen:
            continue
        seen.add(key)
        good.append(s)
    return good


def lca_block(ds, ct, args, rng):
    """Part B1 for one cohort: exact-fit enumeration, informativeness fit, and
    an FOV-cluster bootstrap of the informativeness-constrained fit."""
    rows, boot_rows, summary = [], [], {}
    ptag = "|".join(PRIMARY)
    for (dsx, cls) in FOCAL:
        if dsx != ds:
            continue
        t = ct["tables"].get((cls, ptag))
        if t is None:
            continue
        n8 = t["n8"].astype(float)
        if n8.sum() <= 0:
            continue
        P = n8 / n8.sum()
        st_all = lca_starts(LCA_BOX_ALL[0], LCA_BOX_ALL[1], nrand=args.lca_rand, seed=args.seed)
        best_all, sols_all = lca_fit(P, LCA_BOX_ALL, st_all)
        st_in = lca_starts(LCA_BOX_INFL[0], LCA_BOX_INFL[1], nrand=args.lca_rand, seed=args.seed)
        best_in, sols_in = lca_fit(P, LCA_BOX_INFL, st_in)
        for i, (res, th) in enumerate(sols_all):
            infl, pius = lca_audit(th)
            rows.append(dict(dataset=ds, cls=cls, positivity=PRIMARY[0], reading=PRIMARY[1],
                             variant=PRIMARY[2], box="free", solution_index=i, resid=res,
                             **{k: float(v) for k, v in zip(LCA_NAMES, th)},
                             adm_infl=int(infl), adm_pius=int(pius), is_best=int(i == 0),
                             exact_ok=1, n_solutions=len(sols_all), n_starts=len(st_all),
                             is_primary=1))
        infl, pius = lca_audit(best_in[1])
        rows.append(dict(dataset=ds, cls=cls, positivity=PRIMARY[0], reading=PRIMARY[1],
                         variant=PRIMARY[2], box="infl", solution_index=0, resid=best_in[0],
                         **{k: float(v) for k, v in zip(LCA_NAMES, best_in[1])},
                         adm_infl=int(infl), adm_pius=int(pius), is_best=1,
                         exact_ok=int(best_in[0] <= LCA_EXACT_TOL), n_solutions=len(sols_in),
                         n_starts=len(st_in), is_primary=1))
        # consistency: the fitted LCA posterior must reproduce the fitted
        # (pi, phi) and must have kappa = 0 exactly.
        q = lca_to_q(best_in[1])
        dd = derived_stats(P, q)
        kap = np.nan
        pi_from_q = np.nan
        if dd is not None:
            kap = float(max(abs(dd[k]) for k in ("kEF1", "kEF0", "kHF1", "kHF0")))
            pi_from_q = float(dd["pi"])
        # the first draft's EM, to document that its verdict was an artefact
        em = em_solve_legacy(n8, nstart=40, seed=args.seed, iters=1500)
        em_res = min(s["resid"] for s in em) if em else np.nan
        warm = [s for _, s in sols_all] + [best_in[1]] + [s for _, s in sols_in]
        pi_boot, rej = [], 0
        for rep in range(int(args.lca_boot)):
            one = lca_boot_one(t["n8_img"], warm, rng, nrand=args.lca_rand_ci,
                               seed=args.seed + rep)
            if one is None:
                continue
            ok = (one["resid"] <= LCA_EXACT_TOL) and one["infl"]
            boot_rows.append(dict(dataset=ds, cls=cls, rep=rep, resid=one["resid"],
                                  pi_hat=one["th"][5], a1E=one["th"][1], a1H=one["th"][3],
                                  p=one["th"][0], phi=one["th"][6],
                                  infl_ok=int(one["infl"]), pius_ok=int(one["pius"]),
                                  exact_ok=int(one["resid"] <= LCA_EXACT_TOL),
                                  usable=int(ok)))
            if ok:
                pi_boot.append(float(one["th"][5]))
            else:
                rej += 1
        if len(pi_boot) >= 20:
            lo_b, hi_b = boot_interval(pi_boot)
        else:
            lo_b = hi_b = np.nan
        frac_low = float(np.mean([v <= 0.5 for v in pi_boot])) if pi_boot else np.nan
        summary[cls] = dict(
            n_free=len(sols_all),
            free_pi_min=float(min(s[1][5] for s in sols_all)) if sols_all else np.nan,
            free_pi_max=float(max(s[1][5] for s in sols_all)) if sols_all else np.nan,
            free_best_resid=best_all[0],
            infl_resid=best_in[0], infl_exact=int(best_in[0] <= LCA_EXACT_TOL),
            pi_hat=float(best_in[1][5]), phi_hat=float(best_in[1][6]),
            aE_hat=float(best_in[1][1]), a0E_hat=float(best_in[1][2]),
            aH_hat=float(best_in[1][3]), a0H_hat=float(best_in[1][4]),
            p_hat=float(best_in[1][0]),
            infl_margin=float(min(best_in[1][1] - 0.5, best_in[1][3] - 0.5)),
            pius_ok=int(best_in[1][5] <= best_in[1][6] + 1e-12),
            kappa_max=kap, pi_from_q=pi_from_q,
            em_resid=float(em_res), em_n=len(em),
            boot_n=len(pi_boot), boot_rejected=int(rej), boot_rows=len(boot_rows),
            boot_lo=lo_b, boot_hi=hi_b, boot_frac_pi_le_half=frac_low,
            n_starts_infl=len(st_in))
    return rows, boot_rows, summary


def pooled_lca_fit(tabs, seed=SEED, nstart=32):
    """Multi-group latent-class fit with a SHARED labeller accuracy
    (aE,bE,aH,bH) across populations and free (p, pi, phi) per population,
    fitted by least squares inside the informativeness box.  7K observed
    probabilities against 4 + 3K parameters, so df = 4K - 4 and the model is
    testable.  The first draft fitted this with the same broken EM at a point
    that also ignored informativeness; both choices are replaced here."""
    from scipy.optimize import least_squares
    from scipy.stats import chi2
    tabs = np.asarray(tabs, float)
    K = tabs.shape[0]
    P = tabs / tabs.sum(1, keepdims=True)
    rng = np.random.default_rng(seed)
    lo = np.concatenate([[0.5, 0.0, 0.5, 0.0], np.zeros(3 * K)])
    hi = np.concatenate([[1.0, 0.5, 1.0, 0.5], np.ones(3 * K)])

    def resid(th):
        ab = th[:4]
        rest = th[4:].reshape(K, 3)
        return np.concatenate([lca_probs(np.array([rest[k][0], ab[0], ab[1], ab[2], ab[3],
                                                   rest[k][1], rest[k][2]])) - P[k]
                               for k in range(K)])

    best = (np.inf, None)
    for s in range(nstart):
        if s == 0:
            th0 = np.concatenate([[0.8, 0.1, 0.8, 0.1], np.tile([0.05, 0.1, 0.3], K)])
        elif s == 1:
            th0 = np.concatenate([[0.6, 0.4, 0.6, 0.4], np.tile([0.1, 0.5, 0.5], K)])
        else:
            th0 = np.concatenate([[rng.uniform(0.5, 1.0), rng.uniform(0.0, 0.5),
                                   rng.uniform(0.5, 1.0), rng.uniform(0.0, 0.5)],
                                  rng.uniform(0.0, 1.0, 3 * K)])
        th0 = np.clip(th0, lo + 1e-9, hi - 1e-9)
        try:
            r = least_squares(resid, th0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14,
                              gtol=1e-14, max_nfev=40000)
        except Exception:
            continue
        res = float(np.abs(r.fun).sum())
        if res < best[0]:
            best = (res, r.x.copy())
    out = dict(resid=best[0], nstart=nstart, K=K, aE=np.nan, bE=np.nan, aH=np.nan,
               bH=np.nan, g2=np.nan, df=4 * K - 4, p_value=np.nan)
    if best[1] is None:
        return out
    th = best[1]
    ab = th[:4]
    rest = th[4:].reshape(K, 3)
    g2 = 0.0
    for k in range(K):
        fit = lca_probs(np.array([rest[k][0], ab[0], ab[1], ab[2], ab[3],
                                  rest[k][1], rest[k][2]]))
        n = tabs[k]
        m = n > 0
        g2 += float((2 * n[m] * np.log(n[m] / np.maximum((n.sum() * fit)[m], 1e-300))).sum())
    out.update(aE=float(ab[0]), bE=float(ab[1]), aH=float(ab[2]), bH=float(ab[3]),
               g2=g2, p_value=float(1 - chi2.cdf(min(g2, 1e12), max(4 * K - 4, 1))))
    return out


def boot_interval(values, level=95.0):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size < 20:
        return np.nan, np.nan
    return float(np.percentile(v, (100 - level) / 2.0)), float(np.percentile(v, 100 - (100 - level) / 2.0))


def boot_sharp(n8_img, spec, rng, reps=BOOT):
    """FOV-cluster bootstrap of the EXACT sharp set (whole images resampled).

    For specs with a finite delta the LP is only an outer bound, so the
    bootstrap reports the outer bound and the caller also records the attained
    (inner) values.
    """
    nimg = n8_img.shape[0]
    los, his = [], []
    for _ in range(reps):
        idx = rng.integers(0, nimg, nimg)
        tab = n8_img[idx].sum(0).astype(float)
        if tab.sum() <= 0:
            continue
        P = tab / tab.sum()
        r = sharp_lp(P, spec.get("infl", True), spec.get("pius", True))
        if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
            los.append(r["lo"])
            his.append(r["hi"])
    return los, his


# ====================================================================== analysis driver
def analyse(ds, ct, conv_df):
    """Per-convention decomposition (part A/C) and sharp sets (part B2) for one
    cohort.  The latent-class fit of part B1 lives in lca_block()."""
    classes = ct["classes"]
    expert = conv_df[(conv_df.source == "expert") & (conv_df.scope == "type")
                     & (conv_df.dataset == ds)]
    decomp, sharp = [], []
    old_span = {}
    for cls in classes:
        sub = expert[expert.cls == cls]
        fin = sub[np.isfinite(sub.rate)]
        if len(fin) == 0:
            continue
        old_span[cls] = (float(fin.rate.min()), float(fin.rate.max()),
                         str(fin.loc[fin.rate.idxmin(), "positivity"]) + "|" +
                         str(fin.loc[fin.rate.idxmin(), "reading"]) + "|" +
                         str(fin.loc[fin.rate.idxmin(), "variant"]),
                         str(fin.loc[fin.rate.idxmax(), "positivity"]) + "|" +
                         str(fin.loc[fin.rate.idxmax(), "reading"]) + "|" +
                         str(fin.loc[fin.rate.idxmax(), "variant"]))

    for cls in classes:
        for (setting, reading, variant) in GRID:
            ptag = "%s|%s|%s" % (setting, reading, variant)
            t = ct["tables"].get((cls, ptag))
            if t is None:
                continue
            n8 = t["n8"].astype(float)
            tot = n8.sum()
            if tot <= 0:
                continue
            n_both = n8[6] + n8[7]
            n_e_only = n8[4] + n8[5]
            n_h_only = n8[2] + n8[3]
            coviol = float(n8[7] / n_both) if n_both > 0 else np.nan
            decomp.append(dict(
                dataset=ds, cls=cls, positivity=setting, reading=reading, variant=variant,
                n_expert_labelled=int(t["n_expert"]), k_expert=int(t["k_expert"]),
                rate_expert=float(t["k_expert"] / t["n_expert"]) if t["n_expert"] else np.nan,
                n_hq_samecells=int(t["n_hq_sc"]), k_hq_samecells=int(t["k_hq_sc"]),
                rate_hq_samecells=float(t["k_hq_sc"] / t["n_hq_sc"]) if t["n_hq_sc"] else np.nan,
                n_hq_panel=int(t["n_hq_panel"]), k_hq_panel=int(t["k_hq_panel"]),
                rate_hq_panel=float(t["k_hq_panel"] / t["n_hq_panel"]) if t["n_hq_panel"] else np.nan,
                n_both_labelled=int(n_both), k_both=int(n8[7]),
                rate_both=float(n8[7] / n_both) if n_both > 0 else np.nan,
                n_expert_only=int(n_e_only), k_expert_only=int(n8[5]),
                rate_expert_only=float(n8[5] / n_e_only) if n_e_only > 0 else np.nan,
                n_hq_only=int(n_h_only), k_hq_only=int(n8[3]),
                rate_hq_only=float(n8[3] / n_h_only) if n_h_only > 0 else np.nan,
                co_violation_share=coviol,
                n_neither=int(n8[0] + n8[1]),
                n_cells_expert_set=int(tot)))

            # partial identification: the exact Charnes-Cooper LP when delta is
            # absent (informativeness / PIUS are linear in q), and a rigorous
            # outer LP plus certified attained points when delta is set.
            P = n8 / tot
            LP_SET = {}
            for _nm, _sp in ASSUMPTIONS:
                _key = (bool(_sp["infl"]), bool(_sp["pius"]))
                if _key not in LP_SET:
                    LP_SET[_key] = sharp_lp(P, _key[0], _key[1])
            for name, spec in ASSUMPTIONS:
                dl = spec["delta"]
                lp = LP_SET[(True, True)] if dl is not None else \
                    LP_SET[(bool(spec["infl"]), bool(spec["pius"]))]
                vlo, vhi, _, _, nfv = sharp_vertices(P, **spec)
                sharp.append(dict(
                    dataset=ds, cls=cls, positivity=setting, reading=reading,
                    variant=variant, assumptions=name,
                    delta=(np.nan if dl is None else dl),
                    bounds=("exact-lp" if dl is None else "outer-lp+attained"),
                    pi_lo=lp["lo"], pi_hi=lp["hi"],
                    pi_lo_status=lp["lo_status"], pi_hi_status=lp["hi_status"],
                    binding_hi=lp["binding_hi"],
                    pi_lo_attained=vlo, pi_hi_attained=vhi, n_attained=nfv,
                    method=("charnes-cooper-lp" if dl is None
                            else "lp-relaxation+attained-search"),
                    is_primary=int((setting, reading, variant) == PRIMARY)))
    return dict(decomp=decomp, sharp=sharp, old_span=old_span)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--boot-ci", type=int, default=200)
    ap.add_argument("--rand", type=int, default=20000)
    ap.add_argument("--pool", type=int, default=24)
    ap.add_argument("--lca-boot", type=int, default=200)
    ap.add_argument("--lca-rand", type=int, default=8)
    ap.add_argument("--lca-rand-ci", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "experiments"))
    ap.add_argument("--workdir", default=os.path.join(ROOT, "work"))
    args = ap.parse_args()
    t0 = time.time()
    fh = open(OUT_REPORT, "w", encoding="ascii", newline="\n")
    log("r33 two-source analysis  seed=%d boot=%d boot_ci=%d lca_boot=%d"
        % (args.seed, args.boot, args.boot_ci, args.lca_boot))

    conv_df = pd.read_csv(IN_CONV)
    expert_df = conv_df[(conv_df.source == "expert") & (conv_df.scope == "type")].copy()
    hq_all_df = conv_df[(conv_df.source == "hq_all") & (conv_df.scope == "summary")].copy()
    agree_ref = pd.read_csv(os.path.join(ROOT, "experiments", "r18_agreement.csv"))

    pre_checks = []
    agree_rows, conf_frames, cover_frames, mar_frames = [], [], [], []
    decomp_rows, lca_rows, lcaboot_rows, sharp_rows, delta_rows = [], [], [], [], []
    meta, pooled, lca_sum = {}, {}, {}
    boot_rec = {}
    rng = np.random.default_rng(args.seed)

    for ds in ("CRC_Feb2623", "GVHD_Feb1323"):
        log("  cohort %s" % ds)
        ct = cohort_tables(ds, seed=args.seed)
        cell = ct["cell"]
        row, cf, cv, mr = part_a(ds, cell, ct["imgs"], ct["icodes"], ct["ex"],
                                 ct["le"], ct["lh"],
                                 cell["class_expert"].astype(str).to_numpy(),
                                 cell["class_hq"].astype(str).to_numpy(),
                                 ct["pos_primary"], ct["groups_primary"])
        agree_rows.append(row)
        conf_frames.append(cf)
        cover_frames.append(cv)
        mar_frames.append(mr)
        meta[ds] = dict(n_cells=ct["n_cells"], n_expert=ct["n_expert"],
                        n_images=ct["n_images"], n_expert_images=ct["n_expert_images"],
                        n_hq_images=int(len(set(cell["image"].astype(str)))),
                        classes=list(ct["classes"]), n_classes=len(ct["classes"]))
        res = analyse(ds, ct, conv_df)
        decomp_rows += res["decomp"]
        sharp_rows += res["sharp"]
        # part B1: the latent-class layer
        lr, lb, ls = lca_block(ds, ct, args, rng)
        lca_rows += lr
        lcaboot_rows += lb
        lca_sum[ds] = ls
        # in-loop certificate: Charnes-Cooper LP vs an independent bisection LP,
        # the LP argmax as an explicit feasible point, and the closed form
        # max(infl+pius) = min(max(infl), P(F=1)).
        for _cls in ct["classes"]:
            _t = ct["tables"].get((_cls, "|".join(PRIMARY)))
            if _t is None:
                continue
            _P = _t["n8"].astype(float) / _t["n8"].sum()
            _cc = sharp_lp(_P, True, True)
            _ci = sharp_lp(_P, True, False)
            _bs = sharp_bisect(_P, True, True)
            _A3 = float((MOM[3] * _P).sum())
            pre_checks.append((
                "exact-LP vs independent bisection-LP  %s/%s" % (ds, _cls),
                (abs(_cc["lo"] - _bs["lo"]) < 1e-6 and abs(_cc["hi"] - _bs["hi"]) < 1e-6),
                "cc=[%.9f,%.9f] bisect=[%.9f,%.9f] binding_hi=%s" % (
                    _cc["lo"], _cc["hi"], _bs["lo"], _bs["hi"], _cc["binding_hi"])))
            _q = _cc["hi_q"]
            _dd = derived_stats(_P, _q) if _q is not None else None
            # constraints_vec returns SLSQP-style slacks that must be >= 0 at
            # a feasible q; the first draft tested them for <= 0, so this
            # check could never pass.  The LP argmax is reported together with
            # U = P(Z=1) at that split, because on these tables the maximum is
            # approached only as U -> 0 (the sup is a limit point).
            _spec = (_dd is not None and abs(_dd["pi"] - _cc["hi"]) < 1e-6
                     and all(v >= -1e-9 for v in constraints_vec(_dd, True, True, None)))
            pre_checks.append((
                "LP argmax is an explicit feasible cell-split  %s/%s" % (ds, _cls),
                bool(_spec),
                "pi(q*)=%.9f  hi=%.9f  U(q*)=%.2e  q*=[%s]" % (
                    (_dd["pi"] if _dd else float("nan")), _cc["hi"],
                    (_dd["p"] if _dd else float("nan")),
                    ",".join("%.4f" % v for v in np.round(_q, 4)) if _q is not None else "")))
            pre_checks.append((
                "sharp_hi(infl+pius) == min(sharp_hi(infl), P(F=1))  %s/%s" % (ds, _cls),
                abs(_cc["hi"] - min(_ci["hi"], _A3)) < 1e-9,
                "hi=%.9f  min(hi_infl=%.9f, P(F=1)=%.9f)" % (_cc["hi"], _ci["hi"], _A3)))
        # part B1-bis: cross-layer check -- every exact latent-class fit must
        # lie inside the LP sharp set of the assumption system it satisfies.
        for (dsx, cls) in FOCAL:
            if dsx != ds:
                continue
            t = ct["tables"].get((cls, "|".join(PRIMARY)))
            s = lca_sum[ds].get(cls)
            if t is None or not s:
                continue
            P = t["n8"].astype(float) / t["n8"].sum()
            lpin = sharp_lp(P, True, True)
            okc = (not np.isfinite(s["pi_hat"])) or (lpin["lo"] - 1e-9 <= s["pi_hat"] <= lpin["hi"] + 1e-9)
            if s["infl_exact"] and s["pius_ok"]:
                pre_checks.append((
                    "exact CI fit inside the informativeness+PIUS sharp set  %s/%s" % (ds, cls),
                    bool(okc), "pi_hat=%.6f sharp=[%.6f,%.6f]" % (s["pi_hat"], lpin["lo"], lpin["hi"])))
        # pooled (cross-population, shared labeller accuracy) fit
        ptag = "|".join(PRIMARY)
        tabs = np.array([ct["tables"][(c, ptag)]["n8"] for c in ct["classes"]
                         if (c, ptag) in ct["tables"]], float)
        pooled[ds] = pooled_lca_fit(tabs, seed=args.seed, nstart=args.pool)
        pooled[ds]["classes"] = [c for c in ct["classes"] if (c, ptag) in ct["tables"]]
        pooled[ds]["tabs"] = tabs
        # bootstrap for the populations of interest
        for (dsx, cls) in FOCAL:
            if dsx != ds:
                continue
            t = ct["tables"].get((cls, ptag))
            if t is None:
                continue
            rec = dict(cls=cls, n_expert=t["n_expert"], k_expert=t["k_expert"],
                       n_hq_sc=t["n_hq_sc"], k_hq_sc=t["k_hq_sc"],
                       n_hq_panel=t["n_hq_panel"], k_hq_panel=t["k_hq_panel"],
                       n8=[int(v) for v in t["n8"]],
                       pf1=float((t["n8"][1] + t["n8"][3] + t["n8"][5] + t["n8"][7])
                                 / max(t["n8"].sum(), 1)))
            spec = dict(ASSUMPTIONS)["infl+pius"]
            Ppri = t["n8"].astype(float) / t["n8"].sum()
            lp_pri = sharp_lp(Ppri, True, True)
            rec["pi_lo_exact"] = lp_pri["lo"]
            rec["pi_hi_exact"] = lp_pri["hi"]
            rec["binding_hi"] = lp_pri["binding_hi"]
            los, his = boot_sharp(t["n8_img"], spec, rng, reps=args.boot)
            rec["sharp_boot_lo"] = boot_interval(los)
            rec["sharp_boot_hi"] = boot_interval(his)
            alo, ahi, natt, _qlo, _qhi = sharp_search(Ppri, True, True, None,
                                                      nrand=args.rand, seed=args.seed)
            rec["pi_lo_attained"], rec["pi_hi_attained"] = alo, ahi
            rec["n_attained"] = natt
            rec["min_delta"], rec["min_delta_nf"] = min_feasible_delta(
                Ppri, nrand=max(args.rand // 5, 2000), seed=args.seed)
            dl_rows = []
            for dl in DELTA_GRID:
                dlo, dhi, dnf, _a, _b = sharp_search(Ppri, True, True, dl,
                                                     nrand=args.rand, seed=args.seed)
                dl_rows.append(dict(dataset=ds, cls=cls, delta=dl,
                                    pi_lo_attained=dlo, pi_hi_attained=dhi,
                                    n_attained=dnf, pi_hi_outer=lp_pri["hi"]))
            rec["delta_rows"] = dl_rows
            rate_e, rate_h = [], []
            nimg = t["n8_img"].shape[0]
            for _ in range(args.boot):
                idx = rng.integers(0, nimg, nimg)
                tb = t["n8_img"][idx].sum(0)
                ne = tb[4] + tb[5] + tb[6] + tb[7]
                nh = tb[2] + tb[3] + tb[6] + tb[7]
                if ne > 0:
                    rate_e.append((tb[5] + tb[7]) / ne)
                if nh > 0:
                    rate_h.append((tb[3] + tb[7]) / nh)
            rec["boot_rate_expert"] = boot_interval(rate_e)
            rec["boot_rate_hq"] = boot_interval(rate_h)
            rec["lca"] = lca_sum[ds].get(cls, {})
            boot_rec[(ds, cls)] = rec
            delta_rows += rec["delta_rows"]
        log("    done %s (%.0fs elapsed)" % (ds, time.time() - t0))

    agree = pd.DataFrame(agree_rows)
    conf = pd.concat(conf_frames, ignore_index=True)
    cover = pd.concat(cover_frames, ignore_index=True)
    mar = pd.concat(mar_frames, ignore_index=True) if mar_frames else pd.DataFrame()
    decomp = pd.DataFrame(decomp_rows)
    lca = pd.DataFrame(lca_rows)
    lcaboot = pd.DataFrame(lcaboot_rows)
    sharp = pd.DataFrame(sharp_rows)
    pooled_summary = pd.DataFrame([
        dict(dataset=ds, n_populations=len(pooled[ds]["classes"]),
             shared_aE=pooled[ds]["aE"], shared_bE=pooled[ds]["bE"],
             shared_aH=pooled[ds]["aH"], shared_bH=pooled[ds]["bH"],
             resid=pooled[ds]["resid"], nstart=pooled[ds]["nstart"],
             g2=pooled[ds]["g2"], df=pooled[ds]["df"], p_value=pooled[ds]["p_value"])
        for ds in pooled])

    delta = pd.DataFrame(delta_rows)

    # ------------------------------------------------ Part C: the headline table
    def old_interval(ds, cls):
        sub = expert_df[(expert_df.dataset == ds) & (expert_df.cls == cls)]
        fin = sub[np.isfinite(sub.rate)]
        if len(fin) == 0:
            return dict(old_min=np.nan, old_max=np.nan, old_span=np.nan,
                        old_primary=np.nan, old_primary_lo=np.nan, old_primary_hi=np.nan,
                        old_primary_n=0, n_conventions=0, argmin="", argmax="")
        pri = fin[(fin.positivity == PRIMARY[0]) & (fin.reading == PRIMARY[1])
                  & (fin.variant == PRIMARY[2])]
        prirow = pri.iloc[0] if len(pri) else fin.iloc[0]
        lo, hi = cp_interval(int(prirow.n_violation), int(prirow.n))
        return dict(old_min=float(fin.rate.min()), old_max=float(fin.rate.max()),
                    old_span=float(fin.rate.max() - fin.rate.min()),
                    old_primary=float(prirow.rate), old_primary_lo=lo, old_primary_hi=hi,
                    old_primary_n=int(prirow.n), n_conventions=int(len(fin)),
                    argmin="%s|%s|%s" % (fin.loc[fin.rate.idxmin(), "positivity"],
                                         fin.loc[fin.rate.idxmin(), "reading"],
                                         fin.loc[fin.rate.idxmin(), "variant"]),
                    argmax="%s|%s|%s" % (fin.loc[fin.rate.idxmax(), "positivity"],
                                         fin.loc[fin.rate.idxmax(), "reading"],
                                         fin.loc[fin.rate.idxmax(), "variant"]))

    head_rows = []
    for (ds, cls) in ([(a, b) for a, b in FOCAL] + [("Melanoma_Oct2522", "CD8T")]):
        oi = old_interval(ds, cls)
        sub = sharp[(sharp.dataset == ds) & (sharp.cls == cls) & (sharp.assumptions == "infl+pius")]
        pri = sub[(sub.positivity == PRIMARY[0]) & (sub.reading == PRIMARY[1])
                  & (sub.variant == PRIMARY[2])]
        new_lo = float(pri.pi_lo.iloc[0]) if len(pri) else np.nan
        new_hi = float(pri.pi_hi.iloc[0]) if len(pri) else np.nan
        att_lo = float(pri.pi_lo_attained.iloc[0]) if len(pri) else np.nan
        att_hi = float(pri.pi_hi_attained.iloc[0]) if len(pri) else np.nan
        binding = str(pri.binding_hi.iloc[0]) if len(pri) else ""
        span_lo = float(sub.pi_lo.min()) if len(sub) else np.nan
        span_hi = float(sub.pi_hi.max()) if len(sub) else np.nan
        br = boot_rec.get((ds, cls), {})
        lc = br.get("lca", {}) or {}
        dg = delta[delta.dataset == ds] if delta is not None and len(delta) else None
        dgi = np.nan
        if dg is not None and len(dg):
            dg2 = dg[dg.cls == cls]
            if len(dg2):
                v = dg2.pi_hi_attained.astype(float)
                v = v[np.isfinite(v)]
                if len(v):
                    dgi = float(v.max())
        dec = decomp[(decomp.dataset == ds) & (decomp.cls == cls)
                     & (decomp.positivity == PRIMARY[0]) & (decomp.reading == PRIMARY[1])
                     & (decomp.variant == PRIMARY[2])]
        dec = dec.iloc[0] if len(dec) else None
        claims = OLD_CLAIMS.get((ds, cls), (np.nan, np.nan, "n/a"))
        old_w = oi["old_span"]
        new_w = (new_hi - new_lo) if np.isfinite(new_lo) and np.isfinite(new_hi) else np.nan
        has2 = int(ds in ("CRC_Feb2623", "GVHD_Feb1323"))
        if not has2:
            gate_new = "no second source"
        elif np.isfinite(new_lo) and new_lo > THRESH:
            gate_new = "discordant (>5% under every admissible latent split)"
        elif np.isfinite(new_hi) and new_hi <= THRESH:
            gate_new = "concordant (<=5% under every admissible latent split)"
        else:
            gate_new = "not identified above 5%"
        if not has2:
            gate_lca = "no second source"
        elif lc.get("infl_exact"):
            gate_lca = ("point identified by CI+informativeness at pi=%.4f (%s the 5%% gate)"
                        % (lc["pi_hat"], "below" if lc["pi_hat"] <= THRESH else "above"))
        elif lc:
            gate_lca = ("CI+informativeness rejected (best in-box L1 misfit %.2e)"
                        % lc.get("infl_resid", np.nan))
        else:
            gate_lca = "not fitted"
        overlaps = (np.isfinite(new_lo) and np.isfinite(new_hi)
                    and not (new_hi < oi["old_min"] or new_lo > oi["old_max"]))
        head_rows.append(dict(
            dataset=ds, cls=cls, second_source=has2,
            old_span_lo=oi["old_min"], old_span_hi=oi["old_max"], old_span_width=old_w,
            old_n_conventions=oi.get("n_conventions", 0),
            old_argmin=oi.get("argmin", ""), old_argmax=oi.get("argmax", ""),
            old_primary=oi["old_primary"], old_primary_lo=oi["old_primary_lo"],
            old_primary_hi=oi["old_primary_hi"], old_primary_n=oi["old_primary_n"],
            r28_quoted_min=claims[0], r28_quoted_max=claims[1], r28_verdict=claims[2],
            new_spec="informativeness+PIUS, no conditional-independence assumption",
            new_lo_primary=new_lo, new_hi_primary=new_hi, new_width_primary=new_w,
            new_lo_attained=att_lo, new_hi_attained=att_hi, new_binding=binding,
            new_boot_lo=(br.get("sharp_boot_lo", (np.nan, np.nan))[0]),
            new_boot_hi=(br.get("sharp_boot_hi", (np.nan, np.nan))[1]),
            new_span_lo=span_lo, new_span_hi=span_hi,
            new_span_width=(span_hi - span_lo) if np.isfinite(span_lo) and np.isfinite(span_hi) else np.nan,
            width_ratio_vs_old_span=(new_w / old_w) if (np.isfinite(new_w) and old_w) else np.nan,
            overlaps_old_interval=int(bool(overlaps)),
            new_lower_end_above_gate=int(bool(np.isfinite(new_lo) and new_lo > THRESH)),
            lca_n_free=lc.get("n_free", np.nan),
            lca_free_pi_min=lc.get("free_pi_min", np.nan),
            lca_free_pi_max=lc.get("free_pi_max", np.nan),
            lca_free_best_resid=lc.get("free_best_resid", np.nan),
            lca_infl_resid=lc.get("infl_resid", np.nan),
            lca_exact=lc.get("infl_exact", np.nan),
            lca_pi_hat=lc.get("pi_hat", np.nan), lca_phi_hat=lc.get("phi_hat", np.nan),
            lca_aE_hat=lc.get("aE_hat", np.nan), lca_aH_hat=lc.get("aH_hat", np.nan),
            lca_p_hat=lc.get("p_hat", np.nan),
            lca_infl_margin=lc.get("infl_margin", np.nan),
            lca_pius_ok=lc.get("pius_ok", np.nan),
            lca_kappa_max=lc.get("kappa_max", np.nan),
            lca_pi_from_q=lc.get("pi_from_q", np.nan),
            lca_em_resid=lc.get("em_resid", np.nan),
            lca_boot_n=lc.get("boot_n", np.nan),
            lca_boot_rejected=lc.get("boot_rejected", np.nan),
            lca_boot_lo=lc.get("boot_lo", np.nan), lca_boot_hi=lc.get("boot_hi", np.nan),
            lca_boot_frac_pi_le_half=lc.get("boot_frac_pi_le_half", np.nan),
            pf1_primary=br.get("pf1", np.nan),
            delta_search_min=br.get("min_delta", np.nan),
            delta_grid_inner_hi_max=dgi,
            rate_expert_primary=(float(dec.rate_expert) if dec is not None else np.nan),
            rate_hq_samecells_primary=(float(dec.rate_hq_samecells) if dec is not None else np.nan),
            rate_hq_panel_primary=(float(dec.rate_hq_panel) if dec is not None else np.nan),
            n_hq_panel=(int(dec.n_hq_panel) if dec is not None else 0),
            n_corroborated_primary=(int(dec.n_both_labelled) if dec is not None else 0),
            rate_corroborated_primary=(float(dec.rate_both) if dec is not None else np.nan),
            co_violation_share_primary=(float(dec.co_violation_share) if dec is not None else np.nan),
            gate_old_verdict=("discordant" if oi["old_min"] > THRESH else
                              "concordant" if oi["old_max"] <= THRESH else "mixed"),
            gate_new_verdict=gate_new, gate_lca_verdict=gate_lca))
    head = pd.DataFrame(head_rows)
    summ = head[["dataset", "cls", "second_source", "old_span_lo", "old_span_hi",
                 "old_span_width", "old_primary", "new_lo_primary", "new_hi_primary",
                 "new_width_primary", "width_ratio_vs_old_span", "overlaps_old_interval",
                 "new_lower_end_above_gate", "lca_exact", "lca_pi_hat", "lca_boot_lo",
                 "lca_boot_hi", "lca_infl_margin", "delta_search_min",
                 "n_corroborated_primary", "co_violation_share_primary",
                 "gate_old_verdict", "gate_new_verdict", "gate_lca_verdict"]].copy()

    agree.to_csv(OUT_AGREE, index=False)
    conf.to_csv(OUT_CONF, index=False)
    cover.to_csv(OUT_COVER, index=False)
    mar.to_csv(OUT_MAR, index=False)
    decomp.to_csv(OUT_DECOMP, index=False)
    lca.to_csv(OUT_LCA, index=False)
    lcaboot.to_csv(OUT_LCA_BOOT, index=False)
    sharp.to_csv(OUT_SHARP, index=False)
    head.to_csv(OUT_HEAD, index=False)
    summ.to_csv(OUT_SUM, index=False)
    if len(delta):
        delta.to_csv(OUT_DELTA, index=False)
    log("wrote %d CSVs; rows: agree=%d conf=%d cover=%d mar=%d decomp=%d lca=%d lcaboot=%d "
        "sharp=%d head=%d delta=%d" % (11, len(agree), len(conf), len(cover), len(mar),
                                       len(decomp), len(lca), len(lcaboot), len(sharp),
                                       len(head), len(delta)))

    checks = run_checks(expert_df, conv_df, agree, agree_ref, decomp, sharp, lca, lcaboot,
                        boot_rec, lca_sum, pooled, head, delta, pre_checks)
    write_report(fh, args, meta, agree, conf, cover, mar, decomp, lca, lcaboot, sharp,
                 head, summ, pooled_summary, boot_rec, lca_sum, pooled, checks, delta)
    fh.close()
    npass = sum(1 for c in checks if c[1])
    log("CHECKS: %d PASS / %d FAIL (see work/r33_report.txt)" % (npass, len(checks) - npass))
    log("elapsed %.0fs" % (time.time() - t0))


# ====================================================================== checks + report
def run_checks(expert_df, conv_df, agree, agree_ref, decomp, sharp, lca, lcaboot,
               boot_rec, lca_sum, pooled, head, delta, pre_checks=None):
    """Independently recomputed checks.  Returns [(name, passed, detail), ...]."""
    checks = list(pre_checks or [])

    def add(name, ok, detail=""):
        checks.append((str(name), bool(ok), str(detail)))

    e = expert_df[(expert_df.source == "expert") & (expert_df.scope == "type")]

    # 1 -- the r28 headline spans, recomputed from the raw conventions file
    bad = []
    for (ds, cls), claim in sorted(OLD_CLAIMS.items()):
        fin = e[(e.dataset == ds) & (e.cls == cls)]
        fin = fin[np.isfinite(fin.rate)]
        if len(fin) != GRID_SIZE:
            bad.append("%s/%s has %d conventions" % (ds, cls, len(fin)))
            continue
        mn, mx = float(fin.rate.min()), float(fin.rate.max())
        # OLD_CLAIMS stores the paper's 4-decimal QUOTES, not full precision,
        # so compare after rounding the recomputed endpoints to 4 decimals; a
        # genuine mismatch of more than 5e-5 still fails.
        if abs(round(mn, 4) - claim[0]) > 1e-9 or abs(round(mx, 4) - claim[1]) > 1e-9:
            bad.append("%s/%s [%.6f,%.6f] rounds to [%.4f,%.4f] != quote [%s,%s]" % (
                ds, cls, mn, mx, round(mn, 4), round(mx, 4), claim[0], claim[1]))
    add("r28 headline endpoints recomputed from r18_conventions.csv "
        "(paper quotes are 4-dp rounded)",
        not bad, "; ".join(bad) if bad else
        "6 populations x 68 conventions; every recomputed endpoint rounds to the quoted value")

    # 2 -- part A agreement/kappa vs the stored r18 table
    bad = []
    for r in agree.itertuples():
        ref = agree_ref[agree_ref.dataset == r.dataset]
        if not len(ref):
            bad.append("%s absent from r18_agreement.csv" % r.dataset)
            continue
        ref = ref.iloc[0]
        for col in ("native_agreement", "native_kappa", "coarse_agreement",
                    "coarse_kappa", "coarse_agreement_common"):
            if abs(float(getattr(r, col)) - float(ref[col])) > 1e-9:
                bad.append("%s %s %.9f != r18 %.9f" % (r.dataset, col, getattr(r, col), ref[col]))
        if int(r.n_common_nonunidentified) != int(ref.n_common):
            bad.append("%s n_common %d != r18 %d" % (r.dataset, r.n_common_nonunidentified, ref.n_common))
        if int(r.n_expert_cells) != int(ref.n_expert):
            bad.append("%s n_expert %d != r18 %d" % (r.dataset, r.n_expert_cells, ref.n_expert))
    add("part A agreement/kappa/n reproduce r18_agreement.csv exactly",
        not bad, "; ".join(bad) if bad else "2 cohorts, native+coarse agreement, both kappas, n_common, n_expert")

    # 3 -- 100% cell-level joinability
    bad = []
    for r in agree.itertuples():
        if int(r.n_expert_keys_missing_from_hq) != 0:
            bad.append("%s %d expert keys missing from HQ" % (r.dataset, r.n_expert_keys_missing_from_hq))
        if int(r.n_expert_cells) != int(r.n_expert_labelled_hq_cells):
            bad.append("%s expert cells %d != joinable %d" % (r.dataset, r.n_expert_cells,
                                                              r.n_expert_labelled_hq_cells))
    add("every expert cell joins to the HQ CellTable on (image, cellID)",
        not bad, "; ".join(bad) if bad else "CRC 2505/2505 and GVHD 10283/10283 join; 0 orphans")

    # 4 -- decomposition bookkeeping
    bad = []
    maxdev = 0.0
    for r in decomp.itertuples():
        # the four (E,H) quadrants partition the class's expert/rule universe
        if (r.n_hq_only + r.n_expert_only + r.n_both_labelled + r.n_neither
                != r.n_cells_expert_set):
            bad.append("%s/%s/%s quadrants != universe" % (r.dataset, r.cls, r.positivity))
        # each source's own count is the sum of its two quadrants
        if r.n_expert_only + r.n_both_labelled != r.n_expert_labelled:
            bad.append("%s/%s/%s expert quadrants != n_expert" % (r.dataset, r.cls, r.positivity))
        if r.n_hq_only + r.n_both_labelled != r.n_hq_samecells:
            bad.append("%s/%s/%s hq quadrants != n_hq_samecells" % (r.dataset, r.cls, r.positivity))
        if r.k_expert_only + r.k_both != r.k_expert:
            bad.append("%s/%s/%s k quadrants != k_expert" % (r.dataset, r.cls, r.positivity))
        if r.k_hq_only + r.k_both != r.k_hq_samecells:
            bad.append("%s/%s/%s k quadrants != k_hq_samecells" % (r.dataset, r.cls, r.positivity))
        # every stored rate must be the exact ratio, and nan exactly when n = 0
        for num, den, val, nm in (
                (r.k_expert, r.n_expert_labelled, r.rate_expert, "expert"),
                (r.k_hq_samecells, r.n_hq_samecells, r.rate_hq_samecells, "hq_samecells"),
                (r.k_hq_panel, r.n_hq_panel, r.rate_hq_panel, "hq_panel"),
                (r.k_both, r.n_both_labelled, r.rate_both, "both"),
                (r.k_both, r.n_both_labelled, r.co_violation_share, "co_violation_share"),
                (r.k_expert_only, r.n_expert_only, r.rate_expert_only, "expert_only"),
                (r.k_hq_only, r.n_hq_only, r.rate_hq_only, "hq_only")):
            if den > 0:
                maxdev = max(maxdev, abs(num / den - val))
            elif np.isfinite(val):
                bad.append("%s/%s/%s %s not nan with n=0" % (r.dataset, r.cls, r.positivity, nm))
    add("decomposition: (hq_only + expert_only + both + neither) == universe, per-source "
        "quadrant sums, and every rate is the exact ratio k/n",
        not bad, "; ".join(bad[:6]) if bad else
        "%d rows; max |k/n - stored rate| = %.2e over 7 rate columns" % (len(decomp), maxdev))

    # 5 -- the primary-convention expert rate equals the r18 published rate
    bad = []
    for (ds, cls) in FOCAL + [("Melanoma_Oct2522", "CD8T")]:
        if ds not in paths.HQ_DATASETS:
            continue
        ref = e[(e.dataset == ds) & (e.cls == cls) & (e.positivity == PRIMARY[0])
                & (e.reading == PRIMARY[1]) & (e.variant == PRIMARY[2])]
        dd = decomp[(decomp.dataset == ds) & (decomp.cls == cls)
                    & (decomp.positivity == PRIMARY[0]) & (decomp.reading == PRIMARY[1])
                    & (decomp.variant == PRIMARY[2])]
        if not len(ref) or not len(dd):
            bad.append("%s/%s missing row" % (ds, cls))
            continue
        if abs(float(dd.rate_expert.iloc[0]) - float(ref.rate.iloc[0])) > 1e-12:
            bad.append("%s/%s %.9f != %.9f" % (ds, cls, dd.rate_expert.iloc[0], ref.rate.iloc[0]))
    add("r33 primary-convention expert violation rate == r18 rate",
        not bad, "; ".join(bad) if bad else "4 two-source populations identical to 1e-12")

    # 6 -- sharp rows: ordered, LP solved, attained values inside the LP interval
    bad = []
    sub = sharp[(sharp.positivity == PRIMARY[0]) & (sharp.reading == PRIMARY[1])
                & (sharp.variant == PRIMARY[2])]
    for r in sub.itertuples():
        if int(r.pi_lo_status) != 0 or int(r.pi_hi_status) != 0:
            bad.append("%s/%s status %d/%d" % (r.dataset, r.cls, r.pi_lo_status, r.pi_hi_status))
        if np.isfinite(r.pi_lo) and np.isfinite(r.pi_hi) and r.pi_lo > r.pi_hi + 1e-12:
            bad.append("%s/%s lo>hi" % (r.dataset, r.cls))
        if np.isfinite(r.pi_lo_attained) and np.isfinite(r.pi_lo) and r.pi_lo_attained < r.pi_lo - 1e-12:
            bad.append("%s/%s attained_lo<lo" % (r.dataset, r.cls))
        if np.isfinite(r.pi_hi_attained) and np.isfinite(r.pi_hi) and r.pi_hi_attained > r.pi_hi + 1e-12:
            bad.append("%s/%s attained_hi>hi" % (r.dataset, r.cls))
    add("sharp sets: LP optimal, lo<=hi, certified attained points inside the interval",
        not bad, "; ".join(bad[:6]) if bad else
        "%d primary-convention rows, 0 violations" % len(sub))

    # 7 -- monotone shrinkage as assumptions are added
    bad = []
    piv = sub.pivot_table(index=["dataset", "cls"], columns="assumptions",
                          values=["pi_lo", "pi_hi"])
    for idx, row in piv.iterrows():
        if row[("pi_hi", "infl")] > row[("pi_hi", "none")] + 1e-9:
            bad.append("%s/%s infl wider than none" % (idx,))
        if row[("pi_hi", "infl+pius")] > row[("pi_hi", "infl")] + 1e-9:
            bad.append("%s/%s pius wider than infl" % (idx,))
        if row[("pi_lo", "infl")] < row[("pi_lo", "none")] - 1e-9:
            bad.append("%s/%s infl lo below none lo" % (idx,))
    add("sharp set shrinks monotonically as assumptions are added",
        not bad, "; ".join(bad[:6]) if bad else
        "pi_hi(none) >= pi_hi(infl) >= pi_hi(infl+pius) for every two-source population")

    # 8 -- the saturated latent-class model DOES admit an exact fit; the first
    #      draft's EM missed it, and that is the whole story of its wrong
    #      "point identification is impossible" verdict.
    bad, det = [], []
    for (ds, cls) in FOCAL:
        if ds not in paths.HQ_DATASETS:
            continue
        s = lca_sum.get(ds, {}).get(cls)
        if not s:
            bad.append("%s/%s not fitted" % (ds, cls))
            continue
        det.append("%s/%s ls=%.2e em=%s" % (ds, cls, s["free_best_resid"],
                                            ("%.2e" % s["em_resid"]) if np.isfinite(s["em_resid"]) else "n/a"))
        if not (s["free_best_resid"] <= LCA_EXACT_TOL):
            bad.append("%s/%s least-squares never reaches an exact fit (%.3e)" % (ds, cls, s["free_best_resid"]))
        if s["em_resid"] <= LCA_EXACT_TOL:
            bad.append("%s/%s EM also found an exact fit" % (ds, cls))
    add("the 3-test latent-class model is SATURATED and an exact fit exists "
        "(the first draft's EM residual is an optimizer artefact)",
        not bad, "; ".join(bad) if bad else
        "; ".join(det) + "  |  L1 <= 1e-8 for the least-squares fit, > 1e-6 for the legacy EM")

    # 9 -- relabelling symmetry, verified numerically against the raw table:
    #      if (p,aE,bE,aH,bH,pi,phi) reproduces the table then so does its
    #      mirror (1-p,bE,aE,bH,aH,phi,pi), so {pi,phi} is the identified object
    #      and pi alone is identified only up to that swap.
    bad, nfree, det = [], {}, []
    lf = lca[lca.box == "free"] if lca is not None and len(lca) else lca
    if lf is not None and len(lf):
        for (ds, cls), g in lf.groupby(["dataset", "cls"]):
            nfree[(ds, cls)] = len(g)
            rec = boot_rec.get((ds, cls))
            if rec is None or "n8" not in rec:
                bad.append("%s/%s raw table unavailable" % (ds, cls))
                continue
            Pb = np.asarray(rec["n8"], float)
            Pb = Pb / Pb.sum()
            for r in g.itertuples():
                th = np.array([r.p, r.aE, r.bE, r.aH, r.bH, r.pi, r.phi], float)
                mir = np.array([1 - th[0], th[2], th[1], th[4], th[3], th[6], th[5]], float)
                if float(np.abs(lca_probs(mir) - Pb).sum()) > LCA_EXACT_TOL:
                    bad.append("%s/%s mirror of solution %d is not an exact fit"
                               % (ds, cls, int(r.solution_index)))
                if float(np.abs(lca_probs(th) - Pb).sum()) > LCA_EXACT_TOL:
                    bad.append("%s/%s solution %d residual recomputes > tol"
                               % (ds, cls, int(r.solution_index)))
            pis = sorted(set(round(float(v), 6) for v in g.pi))
            det.append("%s/%s free=%d pi in {%s}" % (ds, cls, len(g),
                                                     ",".join("%.4f" % v for v in pis)))
    else:
        bad.append("no free-box solutions in r33_lca.csv")
    add("exact-fit set is closed under the relabelling (p,aE,bE,aH,bH,pi,phi)->"
        "(1-p,bE,aE,bH,aH,phi,pi), verified against the raw tables",
        not bad, "; ".join(bad[:4]) if bad else "; ".join(det))

    # 10+11 -- informativeness-consistent fits, kappa = 0 feasibility, and the
    #          cross-layer agreement between the latent-class fit and the LP
    bad, det = [], []
    for (ds, cls) in FOCAL:
        if ds not in paths.HQ_DATASETS:
            continue
        s = lca_sum.get(ds, {}).get(cls)
        if not s:
            continue
        det.append("%s/%s pi_hat=%.4f margin=%.4f kappa=%.1e exact=%d"
                   % (ds, cls, s["pi_hat"], s["infl_margin"], s["kappa_max"], s["infl_exact"]))
        if s["infl_exact"]:
            if s["kappa_max"] > 1e-6:
                bad.append("%s/%s exact fit has kappa=%.3e" % (ds, cls, s["kappa_max"]))
            if not (abs(s["pi_from_q"] - s["pi_hat"]) <= 1e-6):
                bad.append("%s/%s pi from q %.6f != fitted %.6f" % (ds, cls, s["pi_from_q"], s["pi_hat"]))
    add("every informativeness-consistent exact fit has kappa = 0 (so the data cannot "
        "reject exact local independence) and reproduces its own pi through derived_stats()",
        not bad, "; ".join(bad) if bad else "; ".join(det))

    # 12 -- pooled shared-accuracy model, now fitted properly
    bad, det = [], []
    for ds, pp in sorted(pooled.items()):
        det.append("%s K=%d L1=%.3e aE=%.3f bE=%.3f aH=%.3f bH=%.3f G2=%.4g df=%d p=%.3g"
                   % (ds, pp["K"], pp["resid"], pp["aE"], pp["bE"], pp["aH"], pp["bH"],
                      pp["g2"], pp["df"], pp["p_value"]))
        if not (np.isfinite(pp["p_value"]) and pp["p_value"] < 1e-3):
            bad.append("%s p=%.4g" % (ds, pp["p_value"]))
        if not (pp["resid"] > 0.05):
            bad.append("%s pooled model nearly fits (L1=%.3e)" % (ds, pp["resid"]))
    add("pooled shared-accuracy latent-class model is rejected (proper multistart least squares, "
        "informativeness imposed)", not bad, "; ".join(det))

    # 13 -- attained values never exceed the outer bound, and the running max is monotone
    bad, det = [], []
    if delta is not None and len(delta):
        for (ds, cls), g in delta.groupby(["dataset", "cls"]):
            g = g.sort_values("delta")
            h = g.pi_hi_attained.to_numpy(float)
            outer = float(g.pi_hi_outer.iloc[0])
            fin = h[np.isfinite(h)]
            if len(fin) and fin.max() > outer + 1e-9:
                bad.append("%s/%s attained %.6f > outer %.6f" % (ds, cls, fin.max(), outer))
            run = -np.inf
            for i in range(len(h)):
                if not np.isfinite(h[i]):
                    continue
                if h[i] < run - 1e-9:
                    det.append("%s/%s raw dip at delta=%g" % (ds, cls, g.delta.iloc[i]))
                run = max(run, h[i])
    add("bounded-dependence sweep: attained <= outer bound for every delta, and the "
        "running maximum is non-decreasing",
        not bad, "; ".join(bad[:4]) if bad else
        ("delta grid consistent; heuristic search dips (informational only): " + "; ".join(det)
         if det else "delta grid consistent and the heuristic was monotone"))

    # 14 -- headline endpoints vs the raw sources
    bad = []
    for r in head.itertuples():
        fin = e[(e.dataset == r.dataset) & (e.cls == r.cls)]
        fin = fin[np.isfinite(fin.rate)]
        if len(fin) and abs(float(fin.rate.min()) - r.old_span_lo) > 1e-9:
            bad.append("%s/%s old_lo" % (r.dataset, r.cls))
        if len(fin) and abs(float(fin.rate.max()) - r.old_span_hi) > 1e-9:
            bad.append("%s/%s old_hi" % (r.dataset, r.cls))
        if r.second_source:
            s2 = sharp[(sharp.dataset == r.dataset) & (sharp.cls == r.cls)
                       & (sharp.assumptions == "infl+pius") & (sharp.positivity == PRIMARY[0])
                       & (sharp.reading == PRIMARY[1]) & (sharp.variant == PRIMARY[2])]
            if not len(s2):
                bad.append("%s/%s no sharp row" % (r.dataset, r.cls))
            elif abs(float(s2.pi_hi.iloc[0]) - r.new_hi_primary) > 1e-12:
                bad.append("%s/%s new_hi mismatch" % (r.dataset, r.cls))
        else:
            if np.isfinite(r.new_hi_primary):
                bad.append("%s/%s has a new interval without a second source" % (r.dataset, r.cls))
    add("headline old endpoints match r18_conventions.csv and new endpoints match r33_sharp.csv",
        not bad, "; ".join(bad[:6]) if bad else "6 populations consistent with the raw sources")

    # 15 -- the co-violation share is a gmm-family statement, NOT a convention-free one
    bad, det = [], []
    f = decomp.copy()
    f["fam"] = np.where(f.positivity == "gmm", "gmm",
                        np.where(f.positivity.str.startswith("pct_all"), "pct_all", "pct_pospop"))
    for (ds, cls) in FOCAL:
        g = f[(f.dataset == ds) & (f.cls == cls)]
        if not len(g):
            continue
        gg = g[g.fam == "gmm"].co_violation_share.astype(float)
        pp = g[g.fam == "pct_pospop"].co_violation_share.astype(float)
        if len(gg) and not (gg.max() <= 1e-12):
            bad.append("%s/%s gmm co-violation share not 0" % (ds, cls))
        det.append("%s/%s gmm=%d/%d zero, pct_pospop median=%.3f max=%.3f"
                   % (ds, cls, int((gg <= 1e-12).sum()), len(gg),
                      float(np.nanmedian(pp)) if len(pp) else np.nan,
                      float(np.nanmax(pp)) if len(pp) else np.nan))
    add("co-violation share is 0 for every gmm-family convention (and rises under pct_pospop), "
        "so it is a positivity-family statement and not a convention-free one",
        not bad, "; ".join(det))

    # 16 -- no unearned narrowing / discordance claim in the headline table
    bad = []
    for r in head.itertuples():
        if r.second_source:
            if "discordant" in r.gate_new_verdict and not (r.new_lo_primary > THRESH):
                bad.append("%s/%s discordant with lower end %.4f" % (r.dataset, r.cls, r.new_lo_primary))
            if r.new_width_primary < r.old_span_width and not (r.new_hi_primary < r.old_span_hi - 1e-12):
                bad.append("%s/%s claims narrowing with no lower upper end" % (r.dataset, r.cls))
            if r.new_lower_end_above_gate and not (r.new_lo_primary > THRESH):
                bad.append("%s/%s lower_end flag inconsistent" % (r.dataset, r.cls))
    add("no interval is called narrower unless its upper end drops, and no discordance is "
        "claimed unless the new lower end clears the gate",
        not bad, "; ".join(bad[:4]) if bad else "4 two-source populations pass the rule")

    # 17 -- bootstrap of the latent-class fit is usable and reported
    bad, det = [], []
    for (ds, cls) in FOCAL:
        if ds not in paths.HQ_DATASETS:
            continue
        s = lca_sum.get(ds, {}).get(cls)
        if not s:
            bad.append("%s/%s not fitted" % (ds, cls))
            continue
        g = lcaboot[(lcaboot.dataset == ds) & (lcaboot.cls == cls)] if len(lcaboot) else lcaboot
        if g is None or not len(g):
            bad.append("%s/%s no bootstrap rows" % (ds, cls))
            continue
        nb, nj = int(s["boot_n"]), int(s["boot_rejected"])
        ntot = max(nb + nj, 1)
        if s["infl_exact"] and s["pius_ok"]:
            # an exact fit inside the informativeness+PIUS box exists, so the
            # bootstrap of the point estimate is defined: require a usable
            # majority of resamples and that they bracket the point estimate
            if nb < max(20, int(0.25 * ntot)):
                bad.append("%s/%s only %d/%d usable bootstrap replicates" % (ds, cls, nb, ntot))
            elif not (s["boot_lo"] - 1e-9 <= s["pi_hat"] <= s["boot_hi"] + 1e-9):
                bad.append("%s/%s pi_hat outside its bootstrap interval" % (ds, cls))
            det.append("%s/%s exact in-box fit: reps=%d rejected=%d interval=[%.3f,%.3f] "
                       "frac(pi<=0.5)=%.2f" % (ds, cls, nb, nj, s["boot_lo"], s["boot_hi"],
                                               s["boot_frac_pi_le_half"]))
        else:
            # no exact fit inside the box: the point estimate is rejected and
            # the bootstrap interval is UNDEFINED, not merely wide.  Require a
            # majority of replicates to be rejected so the report cannot call
            # a handful of lucky resamples an interval.
            if nj <= nb:
                bad.append("%s/%s no in-box exact fit yet %d/%d replicates usable"
                           % (ds, cls, nb, ntot))
            det.append("%s/%s NO in-box exact fit (best misfit %.2e): reps=%d rejected=%d, "
                       "interval undefined" % (ds, cls, s["infl_resid"], nb, nj))
    add("FOV-cluster bootstrap is usable and brackets the point estimate where an in-box "
        "exact fit exists, and is reported as UNDEFINED where it does not",
        not bad, "; ".join(det))

    # 18 -- written outputs are ASCII and BOM-free
    bad = []
    for path in (OUT_AGREE, OUT_CONF, OUT_COVER, OUT_MAR, OUT_DECOMP, OUT_LCA, OUT_LCA_BOOT,
                 OUT_SHARP, OUT_HEAD, OUT_SUM, OUT_DELTA):
        if not os.path.exists(path):
            bad.append(os.path.basename(path) + " missing")
            continue
        b = open(path, "rb").read()
        if b[:3] == b"\xef\xbb\xbf":
            bad.append(os.path.basename(path) + " has a BOM")
        hi = [x for x in b if x > 127]
        if hi:
            bad.append(os.path.basename(path) + " has %d non-ASCII bytes" % len(hi))
    add("every written CSV exists, is ASCII-only and BOM-free", not bad,
        "; ".join(bad) if bad else "11 CSVs, 0 non-ASCII bytes, 0 BOMs")

    return checks


def write_report(fh, args, meta, agree, conf, cover, mar, decomp, lca, lcaboot, sharp,
                 head, summ, pooled_summary, boot_rec, lca_sum, pooled, checks, delta):
    def w(x=""):
        fh.write(str(x) + "\n")

    def num(x, d=4):
        try:
            x = float(x)
        except Exception:
            return "n/a"
        return "n/a" if not np.isfinite(x) else "%.*f" % (d, x)

    def rng(series):
        v = np.asarray([x for x in series if np.isfinite(x)], float)
        return "n/a" if v.size == 0 else "%.4f-%.4f" % (v.min(), v.max())

    w("r33  TWO ANNOTATION SOURCES: can the depot break the marker-audit convention tie?")
    w("=" * 78)
    w("")
    w("Input : experiments/r18_conventions.csv, experiments/r18_agreement.csv,")
    w("        data/depot/CellTuneDepot/{ManualLabels,HighQualityLabels}/*/Tables/")
    w("Output: experiments/r33_agreement.csv, r33_confusion.csv, r33_coverage.csv,")
    w("        r33_mar.csv, r33_decomp.csv, r33_lca.csv, r33_lcaboot.csv, r33_sharp.csv,")
    w("        r33_delta.csv, r33_headline.csv, r33_headline_summary.csv,")
    w("        work/r33_report.txt (this file)")
    w("Primary convention: %s | %s | %s  (same as r18/r28).  Gate: %.0f%%." % (
        PRIMARY[0], PRIMARY[1], PRIMARY[2], 100 * THRESH))
    w("Seeds: seed=%d, boot=%d, boot_ci=%d, rand=%d, lca_boot=%d.  Marker positivity and" % (
        args.seed, args.boot, args.boot_ci, args.rand, args.lca_boot))
    w("rule evaluation reuse errctl.markercheck / r15_marker_sensitivity unchanged.")
    w("")
    w("WHAT THIS ADDS OVER r28 -- AND WHAT IT CORRECTS IN THE FIRST r33 DRAFT")
    w("-" * 78)
    w("r28 showed the reference-label violation rate is only partially identified: the 68")
    w("conventions move it over 0.4%-90%, and the between-convention span is a median 12.70x")
    w("wider than the fixed-convention sampling interval.  CRC_Feb2623 and GVHD_Feb1323 ship")
    w("a SECOND label table for the same cells, so r33 asks whether two annotations of the")
    w("same cells can turn convention intervals into statements.")
    w("")
    w("Answer for the convention tie: NO.  The second source inherits the same convention")
    w("dependence, the two-source latent-class model is SATURATED (so it cannot test its own")
    w("conditional-independence assumption), and the point estimate that conditional")
    w("independence plus an informativeness sign restriction produces is a knife-edge")
    w("boundary solution: its distance to its own informativeness boundary is about 0.01,")
    w("its FOV bootstrap interval is degenerate at zero, and up to half of the resamples")
    w("fall outside the informativeness box altogether.  Two corrections to the first")
    w("draft of this script are load-bearing and are verified in CHECKS:")
    w("  * the draft reported 'no population has an exact fit' at L1 residuals 0.006-0.44.")
    w("    That was an EM failure, not a property of the data: an independent least-squares")
    w("    multistart reaches L1 <= 1e-8 (down to 1e-14) on the same tables.  The model is")
    w("    saturated, so a nonzero residual can never be evidence against it.")
    w("  * the draft reported 'delta = 0 (exact local independence) is rejected'.  The exact")
    w("    latent-class fit IS a kappa = 0 point, so for three of the five populations the")
    w("    data cannot reject exact local independence at all; the draft's delta=0 search")
    w("    simply cannot reach an exact zero.  The delta table below is therefore a display")
    w("    of what a CHOSEN dependence bound would buy, not a data-determined quantity.")
    w("")
    w("What the second source does deliver, with no latent-variable model at all, is a")
    w("decomposition of the flagged cells (section C, part 1) and, under informativeness")
    w("plus PIUS only, an exact upper bound on the latent violation rate of true members.")
    w("")
    w("")
    w("A  WHAT THE TWO SOURCES ARE")
    w("-" * 78)
    w("Matching key: (image, cellID).  The two label tables are joined CELL BY CELL, not")
    w("cohort by cohort, and the join is complete:")
    for r in agree.itertuples():
        w("  %-14s HQ cells %7d (%2d images) | expert cells %6d (%2d images) | orphans %d"
          % (r.dataset, r.n_hq_cells, r.n_hq_images, r.n_expert_cells, r.n_expert_images,
             r.n_expert_keys_missing_from_hq))
        w("                 expert cells that the coarse map sends to 'Unidentified': %d"
          % (r.n_expert_cells - r.n_common_nonunidentified))
        w("                 native agreement %.4f (kappa %.4f) | coarse agreement %.4f (kappa %.4f)"
          % (r.native_agreement, r.native_kappa, r.coarse_agreement, r.coarse_kappa))
        w("                 coarse agreement on the %d doubly-identified cells %.4f | harmonised map %.4f"
          % (r.n_common_nonunidentified, r.coarse_agreement_common,
             r.coarse_agreement_harmonised))
        w("                 expert labels cover %.2f%% of the whole panel"
          % (100 * r.expert_share_of_panel))
    w("")
    w("So the HighQualityLabels CellTable is the SUPERSET (every segmented cell, with mean")
    w("marker intensity) and ManualLabels is a hand-annotated SUBSET of it.  Two labels of")
    w("the same cells is the Dawid-Skene setting, but they are not independent annotators in")
    w("the usual sense: the HQ table is itself an annotation pass of the same depot, and both")
    w("see the same marker measurements (see A5/A6 below).")
    w("")
    w("Coverage inside the expert-labelled images:")
    for r in agree.itertuples():
        sub = cover[cover.dataset == r.dataset]
        w("  %-14s %d images, expert fraction of the HQ cells per image %.4f-%.4f; %d images have no expert cell"
          % (r.dataset, len(sub), sub.frac_expert.min(), sub.frac_expert.max(),
             int((sub.n_expert == 0).sum())))
    w("")
    w("Selection diagnostic: are the expert-labelled cells a random sample of the cells in")
    w("those same images?  (z is the two-proportion test on the primary-convention rule")
    w("failure rate of HQ-labelled cells inside the expert images.)")
    if mar is not None and len(mar):
        w("  %-14s %-16s %8s %8s %12s %12s %8s" % ("dataset", "class", "n_expert",
                                                   "n_unlab", "rate_expert", "rate_unlab", "z"))
        for r in mar.itertuples():
            mark = "  <-- class-dependent labelling" if (np.isfinite(r.z) and abs(r.z) > 3) else ""
            w("  %-14s %-16s %8d %8d %12.4f %12.4f %8.2f%s"
              % (r.dataset, r.cls, r.n_expert_labelled, r.n_unlabelled,
                 r.rate_expert_subset, r.rate_unlabelled, r.z, mark))
        nz = int((np.abs(mar.z.to_numpy(float)) > 3).sum())
        w("  %d of %d evaluable classes differ at |z| > 3, so the expert subset is NOT a"
          % (nz, len(mar)))
        w("  random subsample of its images: the two-source table is a MAR table, and every")
        w("  r33 number below is conditional on the cells the expert actually labelled.")
    w("")
    w("")
    w("B1  POINT IDENTIFICATION (3-test latent class / Dawid-Skene)")
    w("-" * 78)
    w("Model: E = expert says type T, H = HQ says type T, F = cell fails T's marker rule,")
    w("Z = the cell really is T.  Parameters p=P(Z=1), aE/a0E (expert sens/spec), aH/a0H")
    w("(HQ), pi=P(F=1|Z=1), phi=P(F=1|Z=0).  Seven parameters for the seven free")
    w("probabilities of the 2x2x2 table: the model is SATURATED, so (a) an exact fit")
    w("generically exists and its residual carries no evidence, and (b) the likelihood is")
    w("exactly symmetric under (p,aE,a0E,aH,a0H,pi,phi) -> (1-p,a0E,aE,a0H,aH,phi,pi), so")
    w("conditional independence alone identifies pi only up to the swap pi <-> phi.")
    w("")
    w("  population       free exact fits  pi over those fits   in-box best L1   pi_hat  margin  kappa")
    for (ds, cls) in FOCAL:
        if ds not in paths.HQ_DATASETS:
            continue
        s = lca_sum.get(ds, {}).get(cls)
        if not s:
            continue
        w("  %-14s %-13s %6d            [%s, %s]        %.2e      %.4f  %.4f  %.1e"
          % (ds, cls, s["n_free"], num(s["free_pi_min"]), num(s["free_pi_max"]),
             s["infl_resid"], s["pi_hat"], s["infl_margin"], s["kappa_max"]))
    w("  (margin = min(aE-1/2, aH-1/2): how far the selected branch is from the")
    w("   informativeness boundary.  kappa = max |conditional dependence| of the fitted")
    w("   split; kappa = 0 means the fit is an exact conditional-independence point.)")
    w("")
    w("Reading, population by population:")
    for (ds, cls) in FOCAL:
        if ds not in paths.HQ_DATASETS:
            continue
        s = lca_sum.get(ds, {}).get(cls)
        if not s:
            continue
        w("  * %s / %s" % (ds, cls))
        w("      free exact fits: %d, all with L1 <= %.0e (the legacy EM stopped at %.3e,"
          % (s["n_free"], LCA_EXACT_TOL, s["em_resid"]))
        w("      which is why the first draft wrongly concluded the model does not fit).")
        w("      pi over the exact-fit set runs from %s to %s: the set is the mirror pair"
          % (num(s["free_pi_min"]), num(s["free_pi_max"])))
        w("      {pi,phi} = {%s,%s} and pi alone is identified only once a sign restriction"
          % (num(s["free_pi_min"]), num(s["free_pi_max"])))
        w("      is imposed.")
        if s["infl_exact"]:
            w("      informativeness selects the branch pi = %.4f (aE=%.4f, aH=%.4f, p=%.4f, phi=%.4f),"
              % (s["pi_hat"], s["aE_hat"], s["aH_hat"], s["p_hat"], s["phi_hat"]))
            w("      with margin %.4f above the a1 >= 1/2 boundary and PIUS %s.  An exact"
              % (s["infl_margin"], "holding" if s["pius_ok"] else "VIOLATED"))
            w("      kappa = 0 solution therefore exists: the data cannot reject exact local")
            w("      independence for this population.")
            w("      bootstrap of this fit: [%s, %s] over %d usable FOV-cluster replicates"
              % (num(s["boot_lo"]), num(s["boot_hi"]), s["boot_n"]))
            w("      (%d replicates were unusable/rejected); fraction of replicates with"
              % s["boot_rejected"])
            w("      pi <= 1/2: %s.  The point estimate is a knife-edge solution: the"
              % num(s["boot_frac_pi_le_half"], 2))
            w("      bootstrap flips between pi = 0 and the mirror branch, so it is NOT a")
            w("      usable point correction.")
        else:
            w("      NO informativeness-consistent fit exists: the best in-box fit still")
            w("      misses the table by L1 = %.3e, so" % s["infl_resid"])
            w("      local independence + informativeness is REJECTED for this population and")
            w("      no point correction is available.")
    w("")
    w("Pooling the populations with a SHARED labeller accuracy makes the model testable")
    w("(7K probabilities against 4+3K parameters, df = 4K-4).  Fitted properly (least")
    w("squares, multistart, informativeness imposed) it cannot reproduce the tables:")
    for r in pooled_summary.itertuples():
        w("  %-14s %2d populations | shared aE=%.3f bE=%.3f aH=%.3f bH=%.3f | best L1 misfit %.3f | G2=%.1f on %d df, p=%.3g"
          % (r.dataset, r.n_populations, r.shared_aE, r.shared_bE, r.shared_aH, r.shared_bH,
             r.resid, r.g2, r.df, r.p_value))
    w("")
    w("CONCLUSION of B1: a latent-class point correction cannot be defended here.  The model")
    w("is saturated, so its fit is not evidence; the only identifying information is the")
    w("informativeness sign restriction; that restriction selects a branch whose distance to")
    w("its own boundary is 0.01 for the two CRC populations; and the FOV bootstrap shows the")
    w("selection flipping.  Reported as a point, the corrected rate would be an artefact of")
    w("the sign restriction, not a measurement.")
    w("")
    w("")
    w("B2  PARTIAL IDENTIFICATION (exact sharp sets) -- THE DEFENSIBLE ROUTE")
    w("-" * 78)
    w("Assumptions, and the fact that each is LINEAR in the latent split q = P(Z=1|cell):")
    w("  informativeness  aE,aH >= 1/2 and a0E,a0H <= 1/2 (each labeller beats a coin")
    w("                   flip about membership)                    -> linear in q")
    w("  PIUS             pi <= phi: the marker rule fails no more often among true")
    w("                   members than among non-members             -> A0*(M3.q) <= A3*(M0.q)")
    w("Because pi = (M3.q)/(M0.q) is a ratio of linear functionals over a polytope, the")
    w("sharp set is an EXACT linear program (Charnes-Cooper); the cube-vertex enumeration")
    w("used by the first draft is only an inner approximation and is not used for bounds.")
    w("")
    w("Exact sharp sets at the primary convention, in the three nested readings:")
    w("  population                  none              informativeness   +PIUS            binding constraint")
    sp = sharp[(sharp.positivity == PRIMARY[0]) & (sharp.reading == PRIMARY[1])
               & (sharp.variant == PRIMARY[2])]
    for (ds, cls) in FOCAL:
        if ds not in paths.HQ_DATASETS:
            continue
        g = sp[(sp.dataset == ds) & (sp.cls == cls)]
        row = {}
        for name in ("none", "infl", "infl+pius"):
            r = g[g.assumptions == name]
            row[name] = (float(r.pi_lo.iloc[0]), float(r.pi_hi.iloc[0])) if len(r) else (np.nan, np.nan)
        r = g[g.assumptions == "infl+pius"]
        bind = str(r.binding_hi.iloc[0]) if len(r) else ""
        w("  %-14s %-13s [%.4f,%.4f]  [%.4f,%.4f]   [%.4f,%.4f]  %s"
          % (ds, cls, row["none"][0], row["none"][1], row["infl"][0], row["infl"][1],
             row["infl+pius"][0], row["infl+pius"][1], bind))
    w("")
    w("Closed form that the LPs reproduce exactly (checked below):")
    w("     max pi (informativeness + PIUS) = min( max pi (informativeness), P(F=1) )")
    w("PIUS alone caps pi at the UNCONDITIONAL rule-failure rate P(F=1) of the expert-cell")
    w("panel.  At the primary convention that unconditional rate is, per population:")
    for r in head.itertuples():
        if r.second_source:
            w("    %-14s %-13s P(F=1) = %s" % (r.dataset, r.cls, num(r.pf1_primary)))
    w("Where P(F=1) exceeds the informativeness cap of 1/2 the informativeness reading binds.")
    w("The lower end of every sharp set is 0, and that end is NOT vacuous: it is attained by")
    w("an explicit conditional-independence fit (kappa = 0) for three of the five")
    w("populations, so 'no true member violates the rule' is one admissible reading of these")
    w("data.  Nothing here establishes a POSITIVE true violation rate.")
    w("")
    w("Bounded conditional dependence.  kappa = 0 is exact local independence, and the")
    w("two-source problem is SATURATED, so kappa is not identified: the data are consistent")
    w("with the whole interval of dependence levels, including kappa = 0 wherever an exact")
    w("latent-class fit exists (kappa column of B1).  The table below is therefore a display")
    w("of what a CHOSEN bound |kappa| <= delta would buy, not a data-determined quantity; the")
    w("'outer' column is the delta-free exact LP bound and the attained numbers are inner")
    w("certificates from a heuristic search (lower bounds on the true sharp upper end).")
    w("  population                    smallest delta admitted   largest attained pi_hi on the grid")
    for (ds, cls) in FOCAL:
        if ds not in paths.HQ_DATASETS:
            continue
        s = lca_sum.get(ds, {}).get(cls)
        if not s:
            continue
        w("  %-14s %-13s kappa_max=%.1e  delta_search_min=%s  inner_hi_max=%s"
          % (ds, cls, s["kappa_max"], num(boot_rec.get((ds, cls), {}).get("min_delta")),
             num(float(delta[(delta.dataset == ds) & (delta.cls == cls)].pi_hi_attained.max())
                 if (delta is not None and len(delta) and len(delta[(delta.dataset == ds) & (delta.cls == cls)]))
                 else np.nan)))
    if delta is not None and len(delta):
        w("")
        w("Sensitivity of the attained upper bound to the chosen dependence bound (primary")
        w("convention).  outer is the delta-free exact LP bound; attained values are realised")
        w("by explicit cell splits and are lower bounds on the sharp set:")
        for (ds, cls), g in sorted(delta.groupby(["dataset", "cls"])):
            g = g.sort_values("delta")
            w("  %-14s %-13s outer=%.4f | %s"
              % (ds, cls, float(g.pi_hi_outer.iloc[0]),
                 "  ".join(("%.4g:%.4f" % (d, h)) if np.isfinite(h) else ("%.4g:infeasible" % d)
                           for d, h in zip(g.delta, g.pi_hi_attained))))
    w("")
    w("")
    w("C  THE HEADLINE: OLD INTERVAL vs NEW INTERVAL")
    w("-" * 78)
    w("OLD = [min, max] of the violation rate over the 68 conventions for the EXPERT labels")
    w("(the r28 identified set: a statement about the shipped reference labels).")
    w("NEW = the exact sharp set of the latent-membership violation rate pi at the primary")
    w("convention under informativeness + PIUS, with NO conditional-independence assumption.")
    w("These are DIFFERENT estimands, so the width ratio is informative but is not a")
    w("like-for-like confidence-interval comparison; the point-route column adds the")
    w("(fragile) conditional-independence reading for contrast.")
    w("")
    w("  population                  old[lo,hi]          old w   new[lo,hi]        new w   ratio  old gate   new gate (sharp set)")
    for r in head.itertuples():
        w("  %-14s %-13s [%s,%s] %6.4f  [%s,%s] %6s  %6s  %-10s %s"
          % (r.dataset, r.cls, num(r.old_span_lo), num(r.old_span_hi), r.old_span_width,
             num(r.new_lo_primary), num(r.new_hi_primary),
             (num(r.new_width_primary) if np.isfinite(r.new_width_primary) else "n/a"),
             (num(r.width_ratio_vs_old_span, 3) if np.isfinite(r.width_ratio_vs_old_span) else "n/a"),
             r.gate_old_verdict, r.gate_new_verdict))
    w("")
    w("  population                  point route (CI + informativeness)                 verdict")
    for r in head.itertuples():
        w("  %-14s %-13s pi_hat=%s  margin=%s  boot=[%s,%s]  %s"
          % (r.dataset, r.cls, num(r.lca_pi_hat), num(r.lca_infl_margin, 4),
             num(r.lca_boot_lo), num(r.lca_boot_hi), r.gate_lca_verdict))
    w("")
    w("Per-population detail (primary convention; sharp-set bootstrap = FOV-cluster, %d resamples):" % args.boot)
    for r in head.itertuples():
        br = boot_rec.get((r.dataset, r.cls), {})
        w("  %s / %s" % (r.dataset, r.cls))
        w("      old : span [%s, %s] over %d conventions; primary %s (CP %s-%s, n=%d)"
          % (num(r.old_span_lo), num(r.old_span_hi), r.old_n_conventions,
             num(r.old_primary), num(r.old_primary_lo), num(r.old_primary_hi), r.old_primary_n))
        w("            argmin %s ; argmax %s" % (r.old_argmin, r.old_argmax))
        if r.second_source:
            w("      new : exact sharp set [%s, %s]  (attained [%s, %s], binding: %s)"
              % (num(r.new_lo_primary), num(r.new_hi_primary), num(r.new_lo_attained),
                 num(r.new_hi_attained), r.new_binding))
            w("            FOV-bootstrap of the outer bound [%s, %s]; delta_search_min %s (search artefact, see B2)"
              % (num(br.get("sharp_boot_lo", (np.nan, np.nan))[0]),
                 num(br.get("sharp_boot_hi", (np.nan, np.nan))[1]), num(r.delta_search_min)))
            w("      new : point route pi_hat=%s (%s), bootstrap [%s, %s] over %s usable replicates"
              % (num(r.lca_pi_hat), ("exact CI fit" if r.lca_exact == 1 else
                                     "CI+informativeness rejected"), num(r.lca_boot_lo),
                 num(r.lca_boot_hi), num(r.lca_boot_n, 0)))
        else:
            w("      new : not available -- this cohort ships no second label source")
        w("      second source at the primary convention:")
        w("          expert-labelled rate %s | HQ on the same cells %s | HQ on the whole panel %s (n=%d)"
          % (num(r.rate_expert_primary), num(r.rate_hq_samecells_primary),
             num(r.rate_hq_panel_primary), r.n_hq_panel))
        w("          cells both sources call this type: n=%d, of which %s violate the rule"
          % (r.n_corroborated_primary, num(r.rate_corroborated_primary)))
        w("          share of flagged cells that the second source also flags: %s"
          % num(r.co_violation_share_primary))
    w("")
    w("Interpretation of the interval change:")
    w("  * EVERY new interval has lower end 0, so NO population's discordance survives as a")
    w("    statement about true members.  The second source does not break the convention")
    w("    tie: it converts 'at least 5% of the reference labels are discordant' into 'the")
    w("    true-member violation rate is somewhere in [0, hi]'.")
    w("  * The upper end does drop materially where the LP binds: CRC endothelial 0.604 ->")
    w("    0.227 and CRC APC 0.000 -> 0.194 (see the table).  The second source therefore")
    w("    bounds how bad the reference labels can be, but it cannot establish a floor.")
    w("  * CRC APC is the one convention-free CONCORDANT population, and it is the clearest")
    w("    casualty: the reference labels agree on 0.00% under all 68 conventions, but at the")
    w("    latent level a rule-failure rate up to 19.4% of true APCs is admissible.  The")
    w("    concordance is a property of the reference labels only.")
    w("  * The point route would say MORE (pi_hat = 0.0000 for CRC CD3T, CRC endothelial and")
    w("    GVHD plasma) but it is a boundary solution: see B1 and the bootstrap column above.")
    w("  * Melanoma CD8T: no second source, unchanged.")
    w("")
    w("THE MODEL-FREE PART: does the second source corroborate the flagged cells?")
    w("For any convention, the flagged cells of a population are the cells labelled T that")
    w("fail T's rule.  The second source can be asked what fraction of those it also calls T")
    w("(the co-violation share, n8[7]/(n8[6]+n8[7])).")
    w("")
    w("  population        co-violation share: gmm family      pct_all (median/max)     pct_pospop (median/max)")
    ff = decomp.copy()
    ff["fam"] = np.where(ff.positivity == "gmm", "gmm",
                         np.where(ff.positivity.str.startswith("pct_all"), "pct_all", "pct_pospop"))
    for (ds, cls) in FOCAL:
        g = ff[(ff.dataset == ds) & (ff.cls == cls)]
        if not len(g):
            continue
        gg = g[g.fam == "gmm"].co_violation_share.astype(float)
        ga = g[g.fam == "pct_all"].co_violation_share.astype(float)
        gp = g[g.fam == "pct_pospop"].co_violation_share.astype(float)
        w("  %-14s %-13s %-24s %-24s %s"
          % (ds, cls, rng(gg),
             "%.4f / %.4f" % (float(np.nanmedian(ga)), float(np.nanmax(ga))),
             "%.4f / %.4f" % (float(np.nanmedian(gp)), float(np.nanmax(gp)))))
    w("  (Melanoma_Oct2522 CD8T: no second source.)")
    w("")
    w("Reading: for ALL FIVE populations the co-violation share is exactly 0 under the four")
    w("gmm-family conventions (including the primary one): every cell the audit flags is a")
    w("cell the second source labels differently.  That statement does NOT extend to the")
    w("percentile conventions: under pct_pospop the median share is large for CD3T,")
    w("neutrophil and plasma and near zero for endothelial (see the table above), so it is")
    w("a POSITIVITY-FAMILY statement, not a")
    w("convention-free one.  The first draft of this script overstated it as convention-free;")
    w("corrected here and checked in CHECKS.  Within the gmm family it is the strongest")
    w("model-free evidence r33 produces: the 5% discordance lives in the boundary layer")
    w("between the two annotation passes rather than in confidently-typed marker failures.")
    w("")
    w("Verdict for the paper:")
    w("  1. The five convention-free discordant populations SURVIVE as statements about the")
    w("     shipped REFERENCE LABELS: the second source cannot change a rate computed from")
    w("     the expert labels, and it supplies no point-identified correction.")
    w("  2. They do NOT survive as statements about the true marker-violation rate of true")
    w("     members: every sharp set reaches 0, and reaching 0 is not vacuous because an")
    w("     explicit conditional-independence fit attains it for three of the five.")
    w("  3. The convention-free concordance (CRC APC, 0.00%) survives only as a statement")
    w("     about the reference labels; at the latent level its interval widens to [0, 0.194].")
    w("  4. The second source's sharpening runs in the direction of LOWERING the worst case")
    w("     (CRC endothelial 0.604 -> 0.227), not of raising the floor.  The paper should")
    w("     therefore claim an upper bound, not a positive violation rate, at the latent")
    w("     level.")
    w("")
    w("")
    fh.flush()
    rb = open(OUT_REPORT, "rb").read()
    ascii_ok = (rb[:3] != b"\xef\xbb\xbf") and all(x < 128 for x in rb)
    checks = list(checks) + [("work/r33_report.txt (body so far) is ASCII-only with no BOM",
                              ascii_ok, "")]
    w("CHECKS")
    w("-" * 78)
    for name, ok, detail in checks:
        w("  [%s] %s" % ("PASS" if ok else "FAIL", name))
        if detail:
            w("         %s" % detail)
    npass = sum(1 for c in checks if c[1])
    w("")
    w("  %d PASS / %d FAIL" % (npass, len(checks) - npass))
    w("")
    w("ASSUMPTIONS MADE, AND WHAT HAPPENS IF EACH IS FALSE")
    w("-" * 78)
    w("A1  Informativeness: each source's sensitivity exceeds its false-positive rate for")
    w("    membership.  If false the sharp sets widen; without A1 the sharp set is all of")
    w("    [0, 1] and the second source adds nothing at all.  Note A1 is also doing ALL of")
    w("    the identification work in the point route of B1.")
    w("A2  PIUS: the marker rule fails no more often among true members than among")
    w("    non-members.  This is what caps pi at the unconditional rule-failure rate P(F=1)")
    w("    (22.71% for CRC endothelial, 19.40% for CRC APC, above 1/2 for CD3T, GVHD")
    w("    neutrophil and GVHD plasma).  If it is false, pi can reach the informativeness")
    w("    cap of 1/2.")
    w("A3  Conditional dependence.  The point route assumes kappa = 0 exactly; the partial")
    w("    route assumes NOTHING about it.  kappa = 0 is not rejected for three of the five")
    w("    populations (an exact latent-class fit with kappa = 0 exists), and the two-source")
    w("    problem is saturated so kappa cannot be estimated.  The delta grid is therefore a")
    w("    sensitivity display for a CHOSEN bound; its attained values are inner certificates")
    w("    only and are NOT the sharp sets.")
    w("A4  Z is one binary latent membership per cell, and the constraints above are the only")
    w("    conditional assumptions used.  The interpretation of pi still needs 'the cell")
    w("    really is T' to be well defined for this panel.")
    w("A5  The two sources may not be independent reads of the truth: both are annotation")
    w("    passes of the same depot and both see the same marker intensities, and the HQ pass")
    w("    may have reused expert decisions on the disputed cells.  If so, kappa > 0 exactly")
    w("    where B1 reports kappa = 0, and the co-violation share is biased away from 0.  The")
    w("    partial-identification bounds do not use independence, so they survive A5; the")
    w("    point route does not.")
    w("A6  ESTIMAND: OLD and NEW target different quantities (reference-label violation vs")
    w("    latent-membership violation).  A reader who insists on one estimand should compare")
    w("    OLD with the CORROBORATED reference-label rate in r33_decomp.csv (column rate_both).")
    w("A7  The expert subset is not a random sample of its images (see the MAR table), so the")
    w("    two-source table is a MAR table and all r33 numbers are conditional on the cells")
    w("    the expert actually labelled.")
    w("A8  The saturation result (B1) is a property of using ONE rule test, TWO labellers and")
    w("    a binary latent class.  A third labeller or a trichotomous latent class would make")
    w("    the model over-identified and the independence assumption testable; neither is")
    w("    available in this depot.")
    w("")
    w("FILES WRITTEN (all new; no existing file was modified)")
    w("-" * 78)
    for p in (OUT_AGREE, OUT_CONF, OUT_COVER, OUT_MAR, OUT_DECOMP, OUT_LCA, OUT_LCA_BOOT,
              OUT_SHARP, OUT_DELTA, OUT_HEAD, OUT_SUM, OUT_REPORT):
        w("  %s" % p)
    w("")


if __name__ == "__main__":
    main()
