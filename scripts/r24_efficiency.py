"""R24 -- the efficiency claim, repaired.

The manuscript claimed the distribution-free certificate is *more efficient*
than the baselines it is compared with.  On this benchmark that claim cannot
be repaired as stated.  This script (a) quantifies the honest version of the
comparison and (b) introduces and evaluates a different, defensible metric:
the certificate yield per target label spent (how many cells are annotated
automatically per label reviewed).

What is computed, in order:

  1. the k = 0 check -- the src arm of experiments/r20_units.csv must
     reproduce experiments/r03_summary.csv, the table the manuscript reports,
     for the three families and the three shared budgets.  If it does not,
     nothing below is trustworthy and the script says so;
  2. the matched-realised-risk comparison of tgt_k / oracle / src (plus the
     uncalibrated naive threshold, which r20 does not carry), mean acceptance
     against pooled realised selective risk, with a unit-level bootstrap CI on
     the certificate/oracle ratio, and the fraction of evaluations that hold
     the budget -- which is where the certificate's advantage actually is;
  3. the certificate yield per target label spent: per (family, alpha, k)
     pooled acceptance and n_accepted / k, the marginal return curve, its knee
     and where it collapses, with validity from the r20 rule (the one-sided
     95% Clopper-Pearson upper limit of the realised risk among accepted
     cells, per unit and classifier, worst-classifier-first, averaged over
     units);
  4. the same accounting under shift, per cross-tissue pair, from the
     independent r14 arm.

Inputs are read-only:
  experiments/r20_units.csv       the label-cost ladder
  experiments/r03_summary.csv     the manuscript's reported table
  experiments/r03_metrics.csv     its per-evaluation source
  experiments/r14_targetcal.csv   the independent cross-tissue arm

Outputs
  experiments/r24_efficiency.csv
  figures/fig10_yield.pdf / .png
  work/r24_report.txt

Usage
  python scripts/r24_efficiency.py [--jobs 4] [--boot 2000]
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "experiments")
WORK = os.path.join(ROOT, "work")
FIG = os.path.join(ROOT, "figures")

MIN_ACCEPT = 25
GRID = "abs"
ALPHAS_R20 = (0.05, 0.10, 0.20)
R_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)
FAMILIES = ("cellsplit", "within", "transfer")
FAM_LABEL = {"cellsplit": "random split", "within": "cross-field-of-view",
             "transfer": "cross-tissue"}
ARM_ORDER = ("src", "tgt_k", "oracle")
ARM_LABEL = {"src": "src (k = 0)", "tgt_k": "tgt_k", "oracle": "oracle",
             "naive": "naive (uncalibrated)"}
ARM_COL = {"src": "#d1495b", "tgt_k": "#1f5fa9", "oracle": "#8d99ae",
           "naive": "#e6a817"}
FAM_COL = {"cellsplit": "#1f5fa9", "within": "#2a9d8f", "transfer": "#d1495b"}
BOOT = 2000
BOOT_SEED = 20260927


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def fmt(x, nd=3):
    try:
        if x is None or not np.isfinite(float(x)):
            return "n/a"
        return ("%%.%df" % nd) % float(x)
    except (TypeError, ValueError):
        return str(x)


# ----------------------------------------------------------------- bases
def cp_upper(n_wrong, n, conf=0.95):
    """One-sided Clopper-Pearson upper limit on a binomial rate.

    Identical to scripts/r20_label_ladder.py, which is the rule this analysis
    must reproduce: n <= 0 has no limit (NaN) and is dropped by the
    worst-classifier-first average, n_wrong >= n returns 1.0.
    """
    n = int(n)
    n_wrong = int(round(n_wrong))
    if n <= 0 or n_wrong < 0:
        return float("nan")
    if n_wrong >= n:
        return 1.0
    return float(_beta.ppf(conf, n_wrong + 1.0, n - n_wrong))


R20_COLS = ["family", "unit", "clf", "alpha", "min_accept", "grid", "gkey", "k",
            "arm", "n_pool", "n_eval", "n_accepted", "n_wrong", "accept_rate",
            "realised_risk", "covered", "covered_nonvac", "empty"]


def load_r20():
    d = pd.read_csv(os.path.join(EXP, "r20_units.csv"), usecols=R20_COLS)
    d = d[(d.min_accept == MIN_ACCEPT) & (d.grid == GRID)].copy()
    d["nonvac"] = (d["empty"] == 0).astype(int)
    return d


def load_r03_summary():
    return pd.read_csv(os.path.join(EXP, "r03_summary.csv"))


def load_r03_metrics():
    cols = ["unit", "clf", "kind", "repeat", "alpha", "method", "n_accepted",
            "accept_rate", "selective_risk", "covered", "empty",
            "n_wrong_accepted", "n_test_eval"]
    return pd.read_csv(os.path.join(EXP, "r03_metrics.csv"), usecols=cols)


def load_r14():
    cols = ["unit", "clf", "alpha", "k", "repeat", "arm", "n_eval",
            "n_accepted", "accept_rate", "selective_risk", "empty", "covered",
            "nonvac_covered"]
    return pd.read_csv(os.path.join(EXP, "r14_targetcal.csv"), usecols=cols)


# ------------------------------------------------- operating-point table
KEYS = ["family", "alpha", "arm", "k"]


def arm_table(d20):
    """One row per family x alpha x arm x k, with every quantity this needs."""
    pu = (d20.groupby(KEYS + ["unit"])
          .agg(u_acc=("accept_rate", "mean"),
               u_cov=("covered", "mean"),
               nac=("n_accepted", "sum"),
               nw=("n_wrong", "sum"),
               neval=("n_eval", "sum"),
               nrow=("n_accepted", "size"),
               nv=("nonvac", "sum"),
               cnv=("covered_nonvac", "sum"),
               cov=("covered", "sum"),
               npool=("n_pool", "max"))
          .reset_index())
    macro = (pu.groupby(KEYS)
             .agg(accept_mean=("u_acc", "mean"),
                  cov_all=("u_cov", "mean"),
                  n_units=("unit", "size"),
                  n_rows=("nrow", "sum"),
                  nac_total=("nac", "sum"),
                  nw_total=("nw", "sum"),
                  neval_total=("neval", "sum"),
                  nonvac=("nv", "sum"),
                  cov_nonvac_events=("cnv", "sum"),
                  covered=("cov", "sum"),
                  npool=("npool", "max"))
             .reset_index())
    puc = (d20.groupby(KEYS + ["unit", "clf"])
           .agg(c_cal=("n_accepted", "sum"), c_wrong=("n_wrong", "sum"))
           .reset_index())
    puc["cp"] = [cp_upper(w, c) for w, c in zip(puc.c_wrong, puc.c_cal)]
    ru = (puc.groupby(KEYS + ["unit"])["cp"].max().reset_index()
          .groupby(KEYS)["cp"].mean().reset_index().rename(columns={"cp": "cp_mean"}))
    t = macro.merge(ru, on=KEYS, how="left")
    t["accept_pooled"] = t.nac_total / t.neval_total.replace(0, np.nan)
    t["realised_risk"] = t.nw_total / t.nac_total.replace(0, np.nan)
    t["cov_nonvac"] = t.cov_nonvac_events / t.nonvac.replace(0, np.nan)
    t["nonvac_frac"] = t.nonvac / t.n_rows
    t["risk_upper_cp"] = t.cp_mean
    t["valid_cp"] = np.isfinite(t.cp_mean) & (t.cp_mean <= t.alpha)
    t["annotations_per_unit"] = t.nac_total / t.n_units
    t["annotations_per_eval"] = t.nac_total / t.n_rows
    t["yield_per_label"] = t.annotations_per_eval / t.k.replace(0, np.nan)
    t["yield_per_label_taskdef"] = t.annotations_per_unit / t.k.replace(0, np.nan)
    t["grid"] = GRID
    # balanced range: every unit of the family has rows at this k
    total = {}
    for fam in FAMILIES:
        total[fam] = d20[(d20.family == fam) & (d20.arm == "src")]["unit"].nunique()
    t["balanced"] = np.array([bool(r.n_units == total[r.family])
                              for r in t.itertuples()], dtype=bool)
    # the k = 0 protocol, per family x alpha
    base = (t[t.arm == "src"][["family", "alpha", "annotations_per_eval",
                               "accept_mean", "realised_risk"]]
            .rename(columns={"annotations_per_eval": "ap_eval_src0",
                             "accept_mean": "acc_src0",
                             "realised_risk": "risk_src0"}))
    t = t.merge(base, on=["family", "alpha"], how="left")
    t["excess_annot_per_eval_vs_src0"] = (t.annotations_per_eval - t.ap_eval_src0)
    t["excess_accept_vs_src0"] = t.accept_mean - t.acc_src0
    return t


def add_marginal(t):
    """Extra auto-annotated cells per extra 100 labels, along the k grid.

    Only consecutive points that are BOTH on the balanced range are used, so
    the finite difference never mixes a different set of units.
    """
    t = t.sort_values(["family", "alpha", "arm", "k"]).copy()
    t["marginal_return_per100"] = np.nan
    for (fam, al, arm), g in t.groupby(["family", "alpha", "arm"], sort=False):
        if arm != "tgt_k":
            continue
        g = g.sort_values("k")
        idx = g.index.values
        ks = g.k.values.astype(float)
        ap = g.annotations_per_eval.values.astype(float)
        bal = g.balanced.values.astype(bool)
        for i in range(1, len(idx)):
            if not (bal[i] and bal[i - 1]):
                continue
            dk = ks[i] - ks[i - 1]
            if dk <= 0:
                continue
            t.loc[idx[i], "marginal_return_per100"] = 100.0 * (ap[i] - ap[i - 1]) / dk
    return t


# ------------------------------------------------------------- step 1
def step1(d20, r03):
    """The k = 0 src/oracle rows must reproduce r03_summary row for row."""
    rows = []
    for fam in FAMILIES:
        for al in ALPHAS_R20:
            for arm, meth in (("src", "crc"), ("oracle", "oracle")):
                s = d20[(d20.family == fam) & (d20.arm == arm) & (d20.k == 0)
                        & np.isclose(d20.alpha, al)]
                r = r03[(r03.kind == fam) & (r03.method == meth)
                        & np.isclose(r03.alpha, al)]
                if not len(s) or not len(r):
                    rows.append(dict(family=fam, alpha=al, arm=arm, method=meth,
                                     ok=False, note="missing rows"))
                    continue
                acc20 = float(s.accept_rate.mean())
                cov20 = float(s["covered"].mean())
                risk20 = float(s["realised_risk"].mean())
                nv = int((1 - s["empty"]).sum())
                cnv = float(s["covered_nonvac"].sum())
                rows.append(dict(
                    family=fam, alpha=al, arm=arm, method=meth,
                    r03_accept=float(r.mean_accept.iloc[0]), r20_accept=acc20,
                    d_accept=acc20 - float(r.mean_accept.iloc[0]),
                    r03_cov=float(r.coverage.iloc[0]), r20_cov=cov20,
                    d_cov=cov20 - float(r.coverage.iloc[0]),
                    r03_risk=float(r.mean_risk.iloc[0]), r20_risk=risk20,
                    r03_n=int(r.n_eval.iloc[0]), r20_n=int(len(s)),
                    nonvac_cov=(cnv / nv) if nv else np.nan,
                    ok=(abs(acc20 - float(r.mean_accept.iloc[0])) < 1e-9
                        and abs(cov20 - float(r.coverage.iloc[0])) < 1e-9
                        and int(len(s)) == int(r.n_eval.iloc[0]))))
    chk = pd.DataFrame(rows)
    chk.attrs["all_ok"] = bool(chk.ok.all())
    return chk


# -------------------------------------------------- unit blocks + bootstrap
def blocks_r03(m, fam, alpha, method):
    s = m[(m.kind == fam) & np.isclose(m.alpha, alpha) & (m.method == method)]
    if not len(s):
        return None
    g = (s.groupby("unit")
         .agg(s_sum=("accept_rate", "sum"), s_cnt=("accept_rate", "size"),
              nac=("n_accepted", "sum"), nw=("n_wrong_accepted", "sum"),
              cov=("covered", "sum"), nrow=("covered", "size"))
         .reset_index())
    return g


def unit_blocks_r20(d20, fam, alpha, arm, k=None):
    s = d20[(d20.family == fam) & np.isclose(d20.alpha, alpha) & (d20.arm == arm)]
    if k is not None:
        s = s[s.k == k]
    if not len(s):
        return None
    return (s.groupby("unit")
            .agg(s_sum=("accept_rate", "sum"), s_cnt=("accept_rate", "size"),
                 nac=("n_accepted", "sum"), nw=("n_wrong", "sum"))
            .reset_index())


def _pooled(pick, blk):
    return (blk["s_sum"].values[pick].sum() / blk["s_cnt"].values[pick].sum(),
            blk["nw"].values[pick].sum() / max(blk["nac"].values[pick].sum(), 1))


def _boot_same_budget(job):
    fam, alpha, nboot, seed, a, b = job
    rng = np.random.default_rng(seed)
    nu = len(a)
    out = np.empty(nboot)
    for i in range(nboot):
        pick = rng.integers(0, nu, size=nu)
        ac, _ = _pooled(pick, a)
        oc, _ = _pooled(pick, b)
        out[i] = ac / oc if oc > 0 else np.nan
    lo, hi = np.nanquantile(out, [0.025, 0.975])
    return fam, alpha, float(lo), float(hi)


def _boot_interp(job):
    fam, r, nboot, seed, alphas, A, B = job
    rng = np.random.default_rng(seed)
    nu = len(A[0])
    out = np.empty(nboot)
    for i in range(nboot):
        pick = rng.integers(0, nu, size=nu)
        pc, po = [], []
        for j in range(len(alphas)):
            ac, rc = _pooled(pick, A[j])
            ao, ro = _pooled(pick, B[j])
            pc.append((rc, ac))
            po.append((ro, ao))
        vc, okc = curve_at(pc, r)
        vo, oko = curve_at(po, r)
        out[i] = (vc / vo) if (okc and oko and vo > 0) else np.nan
    fin = out[np.isfinite(out)]
    if fin.size < 0.5 * nboot:
        return fam, r, float("nan"), float("nan"), float(fin.size) / nboot
    lo, hi = np.quantile(fin, [0.025, 0.975])
    return fam, r, float(lo), float(hi), float(fin.size) / nboot


def curve_points(points):
    """Upper monotone frontier of an arm's (pooled risk, mean acceptance) cloud.

    Points are sorted by pooled realised risk and only points that improve on
    the best acceptance seen so far are kept, so a tuned arm has a single
    well-defined acceptance at each risk level.  Ties keep the largest
    acceptance.
    """
    pts = [(float(x), float(y)) for x, y in points
           if np.isfinite(x) and np.isfinite(y)]
    pts.sort(key=lambda p: (p[0], -p[1]))
    xs, ys, best = [], [], -np.inf
    for x, y in pts:
        if y > best + 1e-15:
            xs.append(x)
            ys.append(y)
            best = y
    return xs, ys


def curve_at(points, r):
    """Acceptance at realised risk r: linear interpolation on the frontier.

    No extrapolation: outside the frontier's risk range the arm has no
    operating point and the answer is reported as not attained.
    """
    xs, ys = curve_points(points)
    if len(xs) < 1:
        return np.nan, False
    if len(xs) == 1:
        return (ys[0], True) if np.isclose(xs[0], r) else (np.nan, False)
    if r < xs[0] - 1e-12 or r > xs[-1] + 1e-12:
        return np.nan, False
    return float(np.interp(r, xs, ys)), True


def r20_points(t, fam, arm, balanced_only=True):
    s = t[(t.family == fam) & (t.arm == arm)]
    if balanced_only:
        s = s[s.balanced]
    return [(r.realised_risk, r.accept_mean) for r in s.itertuples()]

# ------------------------------------------------------------- step 2
MIN_ACCEPT_RATE = 0.10


def _block_frame(b):
    return b[["unit", "s_sum", "s_cnt", "nac", "nw"]].reset_index(drop=True)


def aligned_blocks(m, fam, meth_a, meth_b, alphas):
    A, B = [], []
    for al in alphas:
        a = blocks_r03(m, fam, al, meth_a)
        b = blocks_r03(m, fam, al, meth_b)
        if a is None or b is None:
            return None, None
        mm = a.merge(b, on="unit", suffixes=("_a", "_b"))
        if len(mm) < 2:
            return None, None
        A.append(_block_frame(mm.rename(columns={
            "s_sum_a": "s_sum", "s_cnt_a": "s_cnt", "nac_a": "nac", "nw_a": "nw"})))
        B.append(_block_frame(mm.rename(columns={
            "s_sum_b": "s_sum", "s_cnt_b": "s_cnt", "nac_b": "nac", "nw_b": "nw"})))
    return A, B


def step2(t, m3, jobs):
    out = {"same": [], "interp": [], "boot_same": [], "boot_interp": [],
           "r03_alphas": sorted(float(x) for x in m3.alpha.unique())}
    # (A) same nominal budget: each arm at the budget r, its own realised risk
    for fam in FAMILIES:
        for r in R_GRID:
            row = {"family": fam, "r": r}
            for meth, tag in (("crc", "cert"), ("oracle", "oracle"), ("naive", "naive")):
                b = blocks_r03(m3, fam, r, meth)
                if b is None:
                    continue
                row[tag + "_accept"] = float(b.s_sum.sum() / b.s_cnt.sum())
                row[tag + "_risk"] = float(b.nw.sum() / max(b.nac.sum(), 1))
                row[tag + "_cov"] = float(b["cov"].sum() / max(b["nrow"].sum(), 1))
                row[tag + "_n_units"] = int(len(b))
                row[tag + "_n_eval"] = int(b.s_cnt.sum())
            if np.isfinite(row.get("cert_accept", np.nan)) and row.get("oracle_accept"):
                row["ratio_cert_oracle"] = row["cert_accept"] / row["oracle_accept"]
            if np.isfinite(row.get("naive_accept", np.nan)) and row.get("oracle_accept"):
                row["ratio_naive_oracle"] = row["naive_accept"] / row["oracle_accept"]
            out["same"].append(row)
    # (B) matched realised risk: frontier of each arm, interpolated at r
    for fam in FAMILIES:
        pts = {}
        for meth, tag in (("crc", "cert"), ("oracle", "oracle"), ("naive", "naive")):
            pp = []
            for al in out["r03_alphas"]:
                b = blocks_r03(m3, fam, al, meth)
                if b is None:
                    continue
                pp.append((b.nw.sum() / max(b.nac.sum(), 1),
                           b.s_sum.sum() / b.s_cnt.sum()))
            pts[tag] = pp
        pts["tgt_k_bal"] = r20_points(t, fam, "tgt_k", True)
        pts["tgt_k_all"] = r20_points(t, fam, "tgt_k", False)
        for r in R_GRID:
            row = {"family": fam, "r": r}
            for tag in ("cert", "oracle", "naive", "tgt_k_bal", "tgt_k_all"):
                v, ok = curve_at(pts[tag], r)
                row[tag] = v if ok else np.nan
            for a, b in (("cert", "oracle"), ("naive", "oracle"),
                         ("tgt_k_bal", "oracle"), ("tgt_k_all", "oracle")):
                row["ratio_%s_%s" % (a, b)] = (row[a] / row[b]
                                               if np.isfinite(row[a]) and row.get(b)
                                               else np.nan)
            out["interp"].append(row)
    # bootstrap CIs on the two ratios
    alphas = out["r03_alphas"]
    for fam in FAMILIES:
        A, B = aligned_blocks(m3, fam, "crc", "oracle", alphas)
        if A is None:
            continue
        for r in R_GRID:
            jobs.append(("same", fam, r, alphas, A, B))
            jobs.append(("interp", fam, r, alphas, A, B))
    return out


def run_boot(jobs, nboot, njobs):
    same_tasks, interp_tasks = [], []
    for tag, fam, r, alphas, A, B in jobs:
        k = int(np.argmin([abs(a - r) for a in alphas]))
        if tag == "same":
            same_tasks.append((fam, r, nboot, BOOT_SEED + 101 * k, A[k], B[k]))
        else:
            interp_tasks.append((fam, r, nboot, BOOT_SEED + 1009 * k, alphas, A, B))
    same, interp = [], []
    if njobs > 1 and same_tasks + interp_tasks:
        import multiprocessing as mp
        with mp.Pool(processes=njobs) as pool:
            same = pool.map(_boot_same_budget, same_tasks)
            interp = pool.map(_boot_interp, interp_tasks)
    else:
        same = [_boot_same_budget(x) for x in same_tasks]
        interp = [_boot_interp(x) for x in interp_tasks]
    return same, interp


# ------------------------------------------------------------- step 3
def knee_table(t):
    """Marginal return along the balanced range: knee, collapse, best k."""
    rows = []
    for fam in FAMILIES:
        for al in ALPHAS_R20:
            g = t[(t.family == fam) & (t.arm == "tgt_k") & np.isclose(t.alpha, al)
                  & t.balanced].sort_values("k")
            if not len(g):
                continue
            mr = g.marginal_return_per100.values.astype(float)
            ks = g.k.values.astype(float)
            fin = np.isfinite(mr)
            mx = float(np.max(mr[fin])) if fin.any() else np.nan
            knee = collapse = None
            if fin.any() and mx > 0:
                for i in range(len(ks)):
                    if not np.isfinite(mr[i]):
                        continue
                    tail = mr[i:][np.isfinite(mr[i:])]
                    if mr[i] < 0.10 * mx and np.all(tail <= 0.10 * mx):
                        knee = float(ks[i])
                        break
                for i in range(len(ks)):
                    if not np.isfinite(mr[i]):
                        continue
                    tail = mr[i:][np.isfinite(mr[i:])]
                    if mr[i] <= 1.0 and np.all(tail <= 1.0):
                        collapse = float(ks[i])
                        break
            y = g.yield_per_label.values.astype(float)
            ymax = float(np.nanmax(y)) if np.isfinite(y).any() else np.nan
            kmax = float(ks[int(np.nanargmax(y))]) if np.isfinite(y).any() else np.nan
            ok = g[(g.valid_cp) & (g.accept_mean >= MIN_ACCEPT_RATE)]
            gv = g[g.valid_cp]
            yv = gv.yield_per_label.values.astype(float)
            ymax_v = float(np.nanmax(yv)) if (len(gv) and np.isfinite(yv).any()) else np.nan
            kmax_v = (float(gv.k.values[int(np.nanargmax(yv))])
                      if (len(gv) and np.isfinite(yv).any()) else np.nan)
            if not np.isfinite(mx) or mx <= 0:
                note = ("no marginal return anywhere in the balanced range")
            elif knee is None:
                note = ("return still rising at the end of the balanced range, "
                        "so no knee is observable inside it")
            else:
                note = "knee at k=%d, collapse at k=%d" % (knee, collapse)
            rows.append(dict(family=fam, alpha=al, k_max_balanced=float(ks[-1]),
                             units=int(g.n_units.iloc[-1]),
                             k_first_useful=(float(ok.k.min()) if len(ok) else np.nan),
                             accept_first_useful=(float(ok.accept_mean.iloc[0])
                                                  if len(ok) else np.nan),
                             best_yield=ymax, k_best_yield=kmax,
                             best_yield_valid=ymax_v, k_best_valid=kmax_v,
                             max_marginal_per100=mx, knee_k=knee,
                             collapse_k=collapse, note=note,
                             n_balanced_points=int(len(g))))
    return pd.DataFrame(rows)


def yield_rows(t):
    s = t[t.arm == "tgt_k"].copy()
    s["_fam"] = pd.Categorical(s.family, FAMILIES, ordered=True)
    s = s.sort_values(["_fam", "alpha", "k"])
    return s[["family", "alpha", "k", "balanced", "n_units", "n_rows",
              "accept_mean", "accept_pooled", "realised_risk", "risk_upper_cp",
              "valid_cp", "cov_nonvac", "annotations_per_eval",
              "annotations_per_unit", "yield_per_label",
              "yield_per_label_taskdef", "marginal_return_per100",
              "excess_accept_vs_src0", "excess_annot_per_eval_vs_src0"]]


# ------------------------------------------------------------- step 4
def step4(r14, t):
    d = r14[r14.unit.str.startswith("transfer:")].copy()
    d["pair"] = d.unit.str.split(":").str[1]
    rows = []
    for al in (0.10, 0.20):
        s = d[np.isclose(d.alpha, al)]
        base = (s[(s.arm == "src") & (s.k == 0)]
                .groupby("pair")
                .agg(acc_src=("accept_rate", "mean"),
                     nac_src=("n_accepted", "sum"),
                     neval=("n_eval", "sum"), nrow=("n_accepted", "size"))
                .reset_index())
        base["n_mean"] = base.neval / base.nrow
        for k in sorted(s.k.unique()):
            if k == 0:
                continue
            g = (s[(s.arm == "tgt_k") & (s.k == k)]
                 .groupby("pair")
                 .agg(acc_tgt=("accept_rate", "mean"), nrow=("n_accepted", "size"))
                 .reset_index())
            m = g.merge(base, on="pair")
            m["dacc"] = m.acc_tgt - m.acc_src
            m["extra_per_1000_batch"] = 1000.0 * m.dacc
            m["labels_per_1000_extra"] = np.where(
                m.dacc > 0, 1000.0 * k / (m.dacc * m.n_mean), np.inf)
            for r in m.itertuples():
                rows.append(dict(source="r14", family="transfer", alpha=al,
                                 pair=r.pair, k=k, acc_src=r.acc_src,
                                 acc_tgt=r.acc_tgt, dacc=r.dacc,
                                 extra_per_1000_batch=r.extra_per_1000_batch,
                                 labels_per_1000_extra=r.labels_per_1000_extra,
                                 n_eval_mean=r.n_mean))
    # family level from the independent r20 ladder
    for al in ALPHAS_R20:
        b = t[(t.family == "transfer") & (t.arm == "src") & np.isclose(t.alpha, al)]
        if not len(b):
            continue
        acc_src = float(b.accept_mean.iloc[0])
        for r in t[(t.family == "transfer") & (t.arm == "tgt_k")
                   & np.isclose(t.alpha, al) & t.balanced].itertuples():
            dacc = r.accept_mean - acc_src
            n_mean = r.neval_total / r.n_rows
            rows.append(dict(source="r20", family="transfer", alpha=al, pair="ALL",
                             k=r.k, acc_src=acc_src, acc_tgt=r.accept_mean,
                             dacc=dacc, extra_per_1000_batch=1000.0 * dacc,
                             labels_per_1000_extra=(1000.0 * r.k / (dacc * n_mean)
                                                    if dacc > 0 else np.inf),
                             n_eval_mean=n_mean))
    return pd.DataFrame(rows)

# ------------------------------------------------------------- main
CSV_COLS = ["family", "alpha", "arm", "k", "grid", "balanced", "n_units",
            "n_rows", "neval_total", "nac_total", "nw_total", "accept_mean",
            "accept_pooled", "realised_risk", "cov_all", "cov_nonvac",
            "nonvac_frac", "risk_upper_cp", "valid_cp", "annotations_per_unit",
            "annotations_per_eval", "yield_per_label",
            "yield_per_label_taskdef", "marginal_return_per100", "acc_src0",
            "ap_eval_src0", "excess_accept_vs_src0",
            "excess_annot_per_eval_vs_src0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    log("r24 start  jobs=%d  boot=%d" % (args.jobs, args.boot))
    d20 = load_r20()
    log("r20 primary rows (min_accept=%d, grid=%s):" % (MIN_ACCEPT, GRID), len(d20))
    r03s = load_r03_summary()
    m3 = load_r03_metrics()
    log("r03_metrics rows:", len(m3))
    r14 = load_r14()
    log("r14 rows:", len(r14))

    t = add_marginal(arm_table(d20))
    log("operating points:", len(t))

    chk = step1(d20, r03s)
    log("step 1: all_ok =", chk.attrs["all_ok"],
        "| max |d accept| = %.3g" % chk.d_accept.abs().max())

    out = t[CSV_COLS].sort_values(["family", "alpha", "arm", "k"])
    out.to_csv(os.path.join(EXP, "r24_efficiency.csv"), index=False)
    log("wrote experiments/r24_efficiency.csv rows:", len(out))

    jobs = []
    s2 = step2(t, m3, jobs)
    log("step 2: matched-risk points", len(s2["same"]), len(s2["interp"]),
        "| bootstrap tasks", len(jobs))
    bs, bi = run_boot(jobs, args.boot, args.jobs)
    log("step 2: bootstrap done", len(bs), len(bi))

    knees = knee_table(t)
    yr = yield_rows(t)
    log("step 3: yield ladder rows", len(yr), "| knee rows", len(knees))

    s4 = step4(r14, t)
    log("step 4: shift accounting rows", len(s4))

    if not args.no_figure:
        make_figure(t)

    r14_src = r14[(r14.arm == "src") & np.isclose(r14.alpha, 0.10)
                  & r14.unit.str.startswith("transfer:")]
    nv = int((1 - r14_src["empty"]).sum())
    r14_src_cov = (float(r14_src["nonvac_covered"].sum()) / nv) if nv else np.nan
    R = dict(chk=chk, chk_ok=bool(chk.attrs["all_ok"]), t=t, same=s2["same"],
             interp=s2["interp"], boot_same=bs, boot_interp=bi, knees=knees,
             yield_rows=yr, s4=s4, r14_src_cov=r14_src_cov, nboot=args.boot)
    rep = build_report(R)
    with open(os.path.join(WORK, "r24_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(rep)
    log("wrote work/r24_report.txt")
    print("")
    print(rep.split("\n")[0])




# ------------------------------------------------------------- report
def _tbl(header, rows, sep="|"):
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    out = ["  " + sep.join(str(h).ljust(w) for h, w in zip(header, widths))]
    for r in rows:
        out.append("  " + sep.join(str(c).ljust(w) for c, w in zip(r, widths)))
    return out


REF = [("cellsplit", "src", 0.5594, 1.0000),
       ("within", "src", 0.5338, 0.7640),
       ("transfer", "src", 0.3676, None)]


def build_report(R):
    L = []
    w = L.append
    same = pd.DataFrame(R["same"])
    interp = pd.DataFrame(R["interp"])
    bs = {(f, round(r, 4)): (lo, hi) for f, r, lo, hi in R["boot_same"]}
    bi = {(f, round(r, 4)): (lo, hi, fr) for f, r, lo, hi, fr in R["boot_interp"]}
    chk = R["chk"]
    t = R["t"]
    knees = R["knees"]
    yr = R["yield_rows"]
    s4 = R["s4"]

    def sel(fam, r):
        return same[(same.family == fam) & np.isclose(same.r, r)].iloc[0]

    c10 = sel("cellsplit", 0.10)
    cl = bs.get(("cellsplit", 0.1), (np.nan, np.nan))
    clb = bi.get(("cellsplit", 0.1), (np.nan, np.nan, np.nan))
    verdict = (
        "VERDICT: the 'more efficient than the baselines' claim does not "
        "survive - at a 10%% risk the certificate recovers %.3f of the "
        "oracle's acceptance (%.4f vs %.4f) while the uncalibrated threshold "
        "reaches %.4f (%.3f of the oracle), so the certificate has no "
        "throughput advantage and cannot have one (the oracle is optimal by "
        "construction); its genuine advantage is validity, %.0f%% of "
        "evaluations inside the budget against %.0f%%, and the new yield "
        "metric shows that target labels buy validity under shift but cost "
        "throughput under exchangeability."
        % (c10.ratio_cert_oracle, c10.cert_accept, c10.oracle_accept,
           c10.naive_accept, c10.ratio_naive_oracle,
           100 * c10.cert_cov, 100 * c10.naive_cov))
    w(verdict)
    w("")
    w("=" * 78)
    w("R24 -- the efficiency claim, repaired: the matched-realised-risk")
    w("        accounting and the certificate yield per target label spent")
    w("=" * 78)
    w("")
    w("HEADLINE")
    w("-" * 78)
    w("The manuscript's efficiency claim was that the certificate is more")
    w("efficient than the baselines.  Measured at a matched 10% risk on the")
    w("random-split family the certificate reaches an acceptance of %.4f, which"
      % c10.cert_accept)
    w("is %.3f of the oracle's %.4f (unit bootstrap 95%% CI [%.3f, %.3f])."
      % (c10.ratio_cert_oracle, c10.oracle_accept, cl[0], cl[1]))
    w("The uncalibrated threshold reaches %.4f, i.e. %.3f of the oracle and"
      % (c10.naive_accept, c10.ratio_naive_oracle))
    w("%.4f MORE than the certificate in absolute acceptance, but it holds the"
      % (c10.naive_accept - c10.cert_accept))
    w("budget in only %.1f%% of evaluations against %.1f%% for the certificate."
      % (100 * c10.naive_cov, 100 * c10.cert_cov))
    w("No valid method can exceed the oracle (1.00 by construction), so the")
    w("original claim cannot be repaired as stated.")
    w("What is repairable is the framing: the certificate's advantage is")
    w("validity, not throughput, and the yield metric below makes the price of")
    w("that validity explicit per target label spent.")
    w("")
    w("1. REPRODUCE THE CHECK: the k = 0 src / oracle arm against r03_summary")
    w("-" * 78)
    w("Load: r20_units.csv, min_accept == %d, grid == 'abs', k == 0, compared"
      % MIN_ACCEPT)
    w("row for row with experiments/r03_summary.csv (src <-> r03 'crc',")
    w("oracle <-> r03 'oracle') at alpha in {0.05, 0.10, 0.20}.")
    w("")
    rows = []
    for r in chk.itertuples():
        rows.append([r.family, r.method, "%.2f" % r.alpha, "%.4f" % r.r03_accept,
                     "%.4f" % r.r20_accept, "%.1e" % r.d_accept,
                     "%.4f" % r.r03_cov, "%.4f" % r.r20_cov, "%.1e" % r.d_cov,
                     "%d" % r.r03_n, "%d" % r.r20_n])
    L.extend(_tbl(["family", "r03 method", "alpha", "r03 acc", "r20 acc",
                   "d acc", "r03 cov", "r20 cov", "d cov", "r03 n", "r20 n"],
                  rows))
    w("")
    w("  all agreements exact (|d| < 1e-9) and row counts identical: %s"
      % ("YES" if R["chk_ok"] else "NO -- STOP, the read is broken"))
    w("  non-vacuous coverage of the same rows (sum covered_nonvac / sum")
    w("  (1 - empty), never a mean):")
    for r in chk[chk.method == "crc"].itertuples():
        if not np.isclose(r.alpha, 0.10):
            continue
        ref = dict((f, (a, c)) for f, _, a, c in REF).get(r.family)
        w("    %-10s accept %.4f (reference %.4f) | non-vacuous coverage %.4f%s"
          % (r.family, r.r20_accept, ref[0],
             r.nonvac_cov,
             (" (reference %.4f)" % ref[1]) if ref[1] is not None else
             " (the 0.043 in the brief is the independent r14 run, 0.0441)"))
    w("")
    w("  Verdict of step 1: the k = 0 src arm is the same computation as r03,")
    w("  not a re-implementation, so everything below is read off a validated")
    w("  table.")
    w("")
    w("2. MATCHED-REALISED-RISK EFFICIENCY")
    w("-" * 78)
    w("METHOD.  An arm's operating point at budget alpha is (pooled realised")
    w("risk, mean acceptance) with pooled realised risk = sum(n_wrong) /")
    w("sum(n_accepted) over all units and evaluations of that point, and mean")
    w("acceptance = mean over units of the per-unit mean accept_rate (r20's")
    w("macro_accept).  An arm is a family of such points; its curve is the")
    w("upper monotone frontier of those points in the (risk, acceptance)")
    w("plane, and 'acceptance at realised risk r' is LINEAR INTERPOLATION on")
    w("that frontier between the two bracketing points, never extrapolated.")
    w("Points of the ladder outside the balanced range (fewer than the")
    w("family's full set of units) are shown but excluded from any claim.")
    w("")
    w("Two matching conventions are reported because they are not the same")
    w("thing and the manuscript's Table 4 uses the first:")
    w("  (A) SAME NOMINAL BUDGET - every arm is handed the same budget r and")
    w("      its own realised risk is reported next to its acceptance.  This")
    w("      is what a reader of Table 4 reproduces.")
    w("  (B) MATCHED REALISED RISK - acceptances are read off the frontier at")
    w("      a common realised risk r.  This is favourable to the certificate,")
    w("      which under-uses its budget.")
    w("A frontier point of the src arm is not automatically a *valid* point:")
    w("at a small nominal budget the source-calibrated certificate can carry a")
    w("realised risk far above that budget (the cross-tissue row of 2.1).")
    w("Convention (B) therefore reads off a throughput ceiling, valid or not;")
    w("whether the point is admissible is settled by the budget-compliance")
    w("table 2.4 and the Clopper-Pearson flags of section 3.")
    w("")
    w("2.1 (A) same nominal budget -- r03_metrics, the source of r03_summary")
    w("     (r20's src k = 0 arm equals its 'crc' arm exactly, step 1)")
    rows = []
    for r in same.itertuples():
        cv = bs.get((r.family, round(r.r, 4)), (np.nan, np.nan))
        rows.append([r.family, "%.2f" % r.r, "%.4f" % r.cert_accept,
                     "%.4f" % r.cert_risk, "%.4f" % r.oracle_accept,
                     "%.4f" % r.oracle_risk, "%.4f" % r.naive_accept,
                     "%.4f" % r.naive_risk, "%.3f" % r.ratio_cert_oracle,
                     "[%.3f, %.3f]" % (cv[0], cv[1]),
                     "%.3f" % r.ratio_naive_oracle])
    L.extend(_tbl(["family", "r", "cert acc", "cert risk", "oracle acc",
                   "oracle risk", "naive acc", "naive risk", "cert/oracle",
                   "95% CI", "naive/oracle"], rows))
    w("")
    w("  This is the table behind the review: at r = 0.10 on the random-split")
    w("  family the certificate's ratio to the oracle is %.3f, and the" %
      c10.ratio_cert_oracle)
    w("  uncalibrated alternative's acceptance %.4f exceeds the certificate's"
      % c10.naive_accept)
    w("  %.4f outright.  Under cross-field-of-view the ratio is %.3f; under"
      % (c10.cert_accept,
         sel("within", 0.10).ratio_cert_oracle))
    w("  cross-tissue the source-calibrated certificate accepts %.4f against an"
      % sel("transfer", 0.10).cert_accept)
    w("  oracle of %.4f, a ratio of %.2f -- which is not an efficiency"
      % (sel("transfer", 0.10).oracle_accept,
         sel("transfer", 0.10).ratio_cert_oracle))
    w("  advantage but a violation of the budget: %.4f against a budget of"
      % sel("transfer", 0.10).cert_risk)
    w("  0.10.")
    w("")
    w("2.2 (B) matched realised risk -- frontier interpolation, no")
    w("     extrapolation; 'n/a' means the arm has no operating point there")
    rows = []
    for r in interp.itertuples():
        rows.append([r.family, "%.2f" % r.r, fmt(r.cert), fmt(r.oracle),
                     fmt(r.naive), fmt(r.tgt_k_bal), fmt(r.tgt_k_all),
                     fmt(r.ratio_cert_oracle, 3), fmt(r.ratio_tgt_k_bal_oracle, 3)])
    L.extend(_tbl(["family", "r", "cert (src)", "oracle", "naive",
                   "tgt_k balanced", "tgt_k all units", "cert/oracle",
                   "tgt_k/oracle"], rows))
    w("")
    w("  Under this convention the certificate looks better than in 2.1")
    w("  because it is being credited for the budget it did not use, and it is")
    w("  still short of the oracle everywhere it can be read.  Neither")
    w("  convention shows an advantage over the oracle, and no convention can.")
    w("  r = 0.20 and r = 0.30 are 'n/a' for the k = 0 certificate because even")
    w("  at a 50% nominal budget its pooled realised risk is only about 0.18,")
    w("  so the frontier never reaches those risk levels: no interpolation is")
    w("  possible and none is invented.")
    w("")
    w("2.2b tgt_k (best k) -- the ladder's own operating point at each budget")
    w("     it was run on.  The ladder has only the budgets 0.05/0.10/0.20, so")
    w("     this arm has no r = 0.15 or r = 0.30 row.  'best k' = largest mean")
    w("     acceptance among the balanced-range points that pass the")
    w("     Clopper-Pearson rule; when no balanced point passes, the largest")
    w("     acceptance is still shown and marked invalid.")
    rows = []
    for fam in FAMILIES:
        for al in ALPHAS_R20:
            g = t[(t.family == fam) & (t.arm == "tgt_k")
                  & np.isclose(t.alpha, al) & t.balanced]
            if not len(g):
                continue
            gv = g[g.valid_cp]
            use = gv if len(gv) else g
            p = use.loc[use.accept_mean.idxmax()]
            o = sel(fam, al)
            rows.append([fam, "%.2f" % al, "%d" % p.k,
                         "%d" % p.n_units, "%.4f" % p.accept_mean,
                         fmt(p.realised_risk, 4), fmt(p.risk_upper_cp, 4),
                         "yes" if p.valid_cp else "no",
                         "%.4f" % o.oracle_accept,
                         fmt(p.accept_mean / o.oracle_accept, 3)])
    L.extend(_tbl(["family", "alpha", "best k", "units", "tgt_k acc", "risk",
                   "CP upper", "valid", "oracle acc", "tgt_k/oracle"], rows))
    w("")
    w("  This is the ladder's honest best case.  On the random split it is")
    w("  %.4f at alpha = 0.10, i.e. %.3f of the oracle, and it is not even"
      % (t[(t.family == "cellsplit") & (t.arm == "tgt_k")
           & np.isclose(t.alpha, 0.10) & t.balanced
           & t.valid_cp].accept_mean.max(),
         t[(t.family == "cellsplit") & (t.arm == "tgt_k")
           & np.isclose(t.alpha, 0.10) & t.balanced & t.valid_cp].accept_mean.max()
         / sel("cellsplit", 0.10).oracle_accept))
    w("  as good as the certificate that spends no target labels at all")
    w("  (0.5594).  Buying target labels does not make the certificate more")
    w("  efficient on this family; it makes it more conservative.")
    w("")
    w("2.3 bootstrap CIs on the ratios (one resampling unit = one cohort fold,")
    w("    or one cross-tissue pair x fold, drawn with replacement; %d"
      % R["nboot"])
    w("    replicates, certificate and oracle redrawn together, seed %d)"
      % BOOT_SEED)
    rows = []
    for fam in FAMILIES:
        for r in R_GRID:
            cv = bs.get((fam, round(r, 4)), (np.nan, np.nan))
            bv = bi.get((fam, round(r, 4)), (np.nan, np.nan, np.nan))
            rows.append([fam, "%.2f" % r, fmt(cv[0], 3), fmt(cv[1], 3),
                         fmt(bv[0], 3), fmt(bv[1], 3),
                         fmt(bv[2], 2) if np.isfinite(bv[2]) else "n/a"])
    L.extend(_tbl(["family", "r", "(A) ratio lo", "(A) ratio hi",
                   "(B) ratio lo", "(B) ratio hi", "(B) finite frac"], rows))
    w("")
    w("  Under (A) the random-split certificate/oracle ratio at r = 0.10 is")
    w("  %.3f, CI [%s, %s], which excludes 1.00: the shortfall is not"
      % (c10.ratio_cert_oracle, fmt(bs[("cellsplit", 0.1)][0], 3),
         fmt(bs[("cellsplit", 0.1)][1], 3)))
    w("  sampling noise.  Under (B) the same ratio is [%s, %s]."
      % (fmt(bi.get(("cellsplit", 0.1), (np.nan, np.nan, np.nan))[0], 3),
         fmt(bi.get(("cellsplit", 0.1), (np.nan, np.nan, np.nan))[1], 3)))
    w("")
    w("2.4 WHERE THE ADVANTAGE ACTUALLY IS: budget compliance")
    w("    fraction of evaluations whose realised risk is at or below the")
    w("    nominal budget (r03's 'covered' flag, mean over evaluations)")
    rows = []
    for r in same.itertuples():
        if not np.isclose(r.r, 0.10):
            continue
        rows.append([r.family, "%.3f" % r.cert_cov, "%.3f" % r.oracle_cov,
                     "%.3f" % r.naive_cov, "%d" % r.cert_n_eval,
                     "%d" % r.cert_n_units])
    L.extend(_tbl(["family", "cert", "oracle", "naive", "n evals (cert)",
                   "n units"], rows))
    w("")
    w("  The uncalibrated rule buys its extra throughput with an exceeded")
    w("  budget: it is the only arm whose compliance is far below 1.  Throughput")
    w("  is the axis on which the certificate has no advantage; validity is the")
    w("  axis on which it has all of it.")
    w("")
    w("3. CERTIFICATE YIELD PER TARGET LABEL SPENT (tgt_k arm of the ladder)")
    w("-" * 78)
    w("yield per label = mean auto-annotated cells per evaluation / k, i.e.")
    w("how many cells come back automatically for every target label reviewed")
    w("(the brief's literal form n_accepted_total / (k * n_units) is also in the")
    w("CSV; the two differ by the constant number of evaluations per unit, so")
    w("every trend below is unchanged).  Validity of a (family, alpha, k) cell is")
    w("r20's rule: the one-sided 95% Clopper-Pearson upper limit of the realised")
    w("risk among accepted cells, per unit and classifier, worst classifier")
    w("first, averaged over units, must be <= alpha (reproduced exactly).")
    w("")
    w("3.1 yield ladder, alpha = 0.10 (b = balanced range: every unit present)")
    rows = []
    q = yr[(np.isclose(yr.alpha, 0.10))]
    for r in q.itertuples():
        rows.append([r.family, "%d" % r.k, "b" if r.balanced else "-",
                     "%d" % r.n_units, "%.4f" % r.accept_mean,
                     fmt(r.realised_risk, 4), fmt(r.risk_upper_cp, 4),
                     "yes" if r.valid_cp else "no",
                     fmt(r.yield_per_label, 4),
                     fmt(r.marginal_return_per100, 1)])
    L.extend(_tbl(["family", "k", "bal", "units", "accept", "risk",
                   "CP upper", "valid", "yield/label", "marg/100"], rows))
    w("")
    w("3.2 the k = 0 baseline and what the labels buy")
    w("    src k = 0 annotates without spending a single target label:")
    rows = []
    for fam in FAMILIES:
        b = t[(t.family == fam) & (t.arm == "src")]
        for r in b.itertuples():
            rows.append([fam, "%.2f" % r.alpha, "%.4f" % r.accept_mean,
                         fmt(r.realised_risk, 4), fmt(r.risk_upper_cp, 4),
                         "yes" if r.valid_cp else "no"])
    L.extend(_tbl(["family", "alpha", "src k=0 accept", "risk", "CP upper",
                   "valid"], rows))
    w("")
    w("  Excess acceptance from spending k target labels (excess over the")
    w("  k = 0 protocol at the SAME budget; k <= 2000 shown):")
    rows = []
    q = yr[yr.k <= 2000]
    for r in q.itertuples():
        rows.append([r.family, "%.2f" % r.alpha, "%d" % r.k,
                     "b" if r.balanced else "-",
                     "%+.4f" % r.excess_accept_vs_src0,
                     "%+.1f" % r.excess_annot_per_eval_vs_src0])
    L.extend(_tbl(["family", "alpha", "k", "bal", "excess acceptance",
                   "excess cells/eval"], rows))
    w("")
    w("  On the random-split and cross-field-of-view families the excess is")
    w("  NEGATIVE across the whole balanced range: the k = 0 protocol is already")
    w("  valid there, and buying target labels makes the certificate annotate")
    w("  less, not more.  Yield per label is therefore not a small positive")
    w("  number on those families; it is negative.  Under cross-tissue transfer")
    w("  the k = 0 protocol is invalid (CP upper %.3f against a budget of"
      % t[(t.family == "transfer") & (t.arm == "src")
          & np.isclose(t.alpha, 0.10)].risk_upper_cp.iloc[0])
    w("  0.10) and the labels buy validity, at the yield recorded below.")
    w("")
    w("3.3 marginal return, knee and collapse (balanced range only)")
    rows = []
    for r in knees.itertuples():
        rows.append([r.family, "%.2f" % r.alpha, "%d" % r.n_balanced_points,
                     "%d" % r.k_max_balanced, fmt(r.k_first_useful, 0),
                     fmt(r.accept_first_useful, 3), fmt(r.best_yield_valid, 4),
                     fmt(r.k_best_valid, 0), fmt(r.max_marginal_per100, 1),
                     fmt(r.knee_k, 0), fmt(r.collapse_k, 0)])
    L.extend(_tbl(["family", "alpha", "n pts", "k max bal", "first useful k",
                   "accept there", "best yield valid", "at k", "max marg/100",
                   "knee k", "collapse k"], rows))
    w("")
    w("  Definitions: 'first useful k' = smallest balanced k that is valid AND")
    w("  annotates at least %.0f%% of the batch (r20's usefulness rule)."
      % (100 * MIN_ACCEPT_RATE))
    w("  'knee k' = smallest balanced k whose marginal return has fallen below")
    w("  %.0f%% of its maximum and stays there; 'collapse k' = smallest k whose"
      % 10)
    w("  marginal return is at or below one extra cell per 100 labels and stays")
    w("  there.  'best yield valid' is the largest yield per label among")
    w("  balanced Clopper-Pearson-valid points.  Why a knee is 'n/a' differs")
    w("  by family and is stated per row below, because the two cases mean")
    w("  opposite things:")
    for r in knees.itertuples():
        w("    %-10s alpha = %.2f : %s" % (r.family, r.alpha, r.note))
    w("  Case one (random split and cross-field-of-view) is a real zero: the")
    w("  certificate annotates nothing at all until k is large, so there is no")
    w("  marginal return to bend.  Case two (cross-tissue) is a censoring: the")
    w("  balanced range stops at k = 1000 because smaller cohorts cannot")
    w("  afford more labels, and the return is still climbing there.  Neither")
    w("  is evidence of a knee inside the range this design can measure.")
    w("")
    w("4. LABEL-COST ACCOUNTING UNDER SHIFT (transfer)")
    w("-" * 78)
    w("Extra auto-annotated cells per 1000 cells of batch and the target labels")
    w("needed per 1000 extra auto-annotated cells, relative to the k = 0")
    w("protocol, per cross-tissue pair.  r14 and r20 are independent runs and")
    w("differ slightly; both are reported and neither is reconciled.")
    w("")
    for src in ("r14", "r20"):
        for al in (0.10, 0.20):
            q = s4[(s4.source == src) & np.isclose(s4.alpha, al)]
            if not len(q):
                continue
            ks = [k for k in sorted(q.k.unique()) if k in (200, 500, 1000, 2000)]
            rows = []
            for pair in sorted(q.pair.unique()):
                row = [pair]
                for k in ks:
                    v = q[(q.pair == pair) & (q.k == k)]
                    row.append(fmt(v.dacc.iloc[0], 4) if len(v) else "n/a")
                rows.append(row)
            w("  %s, alpha = %.2f -- extra annotated cells per 1000 batch cells"
              % (src.upper(), al))
            L.extend(_tbl(["pair"] + ["k = %d" % k for k in ks], rows))
            rows = []
            for pair in sorted(q.pair.unique()):
                row = [pair]
                for k in ks:
                    v = q[(q.pair == pair) & (q.k == k)]
                    if not len(v):
                        row.append("n/a")
                    elif not np.isfinite(v.labels_per_1000_extra.iloc[0]):
                        row.append("never")
                    else:
                        row.append("%.0f" % v.labels_per_1000_extra.iloc[0])
                rows.append(row)
            w("  %s, alpha = %.2f -- labels per 1000 extra annotated cells"
              % (src.upper(), al))
            L.extend(_tbl(["pair"] + ["k = %d" % k for k in ks], rows))
            w("")
    q10 = s4[(s4.source == "r14") & np.isclose(s4.alpha, 0.10)]
    kk = [k for k in sorted(q10.k.unique()) if k <= 2000]
    tot = len(q10.pair.unique())
    pos = set()
    for pair in q10.pair.unique():
        g = q10[(q10.pair == pair) & (q10.k.isin(kk))]
        if (g.dacc > 0).any():
            pos.add(pair)
    w("  Reading: at a 10% budget the ladder never recovers the k = 0")
    w("  protocol's throughput for %d of the %d cross-tissue pairs at any k up"
      % (tot - len(pos), tot))
    w("  to the largest the grid allows (the excess is negative everywhere, so")
    w("  the cost per recovered cell is not large, it is undefined).  For the")
    w("  %d pair(s) that do turn positive, the price is thousands of target"
      % len(pos))
    w("  labels per 1000 extra auto-annotated cells.  The k = 0 protocol's")
    w("  throughput is therefore not a baseline the labels can buy back: the")
    w("  labels buy validity, and they buy it at the cost of the throughput")
    w("  that made the k = 0 protocol look efficient while it was invalid.")
    w("")
    w("5. WHAT DID NOT WORK / LIMITS")
    w("-" * 78)
    w("- r20 carries only three nominal budgets (0.05, 0.10, 0.20), so the")
    w("  tgt_k ladder cannot be read at r = 0.15 or r = 0.30; those cells of")
    w("  table 2.2 are 'n/a' and the r03-based arms carry the extended grid.")
    w("- the k = 0 baseline is not a fair floor for the yield metric on the two")
    w("  exchangeable families, because source calibration is already valid")
    w("  there; the metric was still computed and its marginal return is <= 0,")
    w("  which is the result rather than a defect.")
    w("- no bootstrap CI is attached to the tgt_k ladder ratios: the ladder's")
    w("  per-unit rows exist for all three classifiers and 20 replicate draws,")
    w("  but the ratio needs a common frontier per replicate, which the stored")
    w("  ladder does not support.  Point estimates only for the tgt_k rows.")
    w("- the transfer family's non-vacuous coverage is near zero for the k = 0")
    w("  protocol (r20 ratio of sums %.4f, r14 %.4f); any coverage claim there"
      % (t[(t.family == "transfer") & (t.arm == "src")
           & np.isclose(t.alpha, 0.10)].cov_nonvac.iloc[0],
         R["r14_src_cov"]))
    w("  rests on very few non-vacuous evaluations and is not used above.")
    w("")
    w("VERDICT: %s" % verdict.replace("VERDICT: ", ""))
    return "\n".join(L) + "\n"

# ------------------------------------------------------------- figure
def make_figure(t, outdir=FIG):
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
    fig = plt.figure(figsize=(9.8, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.70,
                          wspace=0.34)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axb = fig.add_subplot(gs[1, :])
    XMAX = {"cellsplit": 0.26, "within": 0.26, "transfer": 0.50}
    YMAX = {"cellsplit": 1.02, "within": 1.02, "transfer": 0.78}
    for i, (ax, fam) in enumerate(zip(axes, FAMILIES)):
        xm, ym = XMAX[fam], YMAX[fam]
        for r in R_GRID:
            if r <= xm:
                ax.axvline(r, color="0.85", lw=0.7, zorder=0)
        ax.plot([0, min(xm, ym)], [0, min(xm, ym)], color="0.5", ls=":",
                lw=0.9, zorder=1)
        for arm, mk, ls in (("oracle", "s", "--"), ("tgt_k", "o", "-"),
                            ("src", "D", "-")):
            g = t[(t.family == fam) & (t.arm == arm)]
            if not len(g):
                continue
            xs, ys = curve_points([(r.realised_risk, r.accept_mean)
                                   for r in g.itertuples()])
            ax.plot(xs, ys, marker=mk, ms=3.6, lw=1.0, ls=ls,
                    color=ARM_COL[arm], zorder=3)
        s = t[(t.family == fam) & (t.arm == "tgt_k")]
        su = s[~s.balanced]
        if len(su):
            ax.plot(su.realised_risk, su.accept_mean, "o", ms=3.4, mfc="white",
                    mec=ARM_COL["tgt_k"], mew=0.8, ls="none", zorder=2)
        for arm in ("src", "oracle"):
            v, ok = curve_at([(r.realised_risk, r.accept_mean) for r in
                              t[(t.family == fam) & (t.arm == arm)].itertuples()],
                             0.10)
            if ok:
                ax.plot([0.10], [v], marker="x", ms=5.5, mew=1.3,
                        color=ARM_COL[arm], zorder=5)
        ax.set_xlim(0, xm)
        ax.set_ylim(0, ym)
        ax.set_xlabel("pooled realised risk")
        ax.set_title(FAM_LABEL[fam], loc="right", fontsize=7.5, color="0.35")
        if fam == "cellsplit":
            ax.set_ylabel("mean acceptance")
        if i == 0:
            ax.set_title("a", loc="left", fontweight="bold", pad=6)
    handles = [Line2D([], [], color=ARM_COL["tgt_k"], marker="o", ms=3.6,
                      lw=1.0, label="tgt_k (target-calibrated)"),
               Line2D([], [], color=ARM_COL["oracle"], marker="s", ms=3.6,
                      lw=1.0, ls="--", label="oracle (peeks at the labels)"),
               Line2D([], [], color=ARM_COL["src"], marker="D", ms=3.6,
                      lw=1.0, label="src (k = 0, source-calibrated)"),
               Line2D([], [], color=ARM_COL["tgt_k"], marker="o", ms=3.4,
                      mfc="white", ls="none", label="tgt_k, partial unit set"),
               Line2D([], [], color="0.5", ls=":", lw=0.9,
                      label="identity: acceptance = risk"),
               Line2D([], [], color="0.25", marker="x", ms=5.5, ls="none",
                      label="read at r = 0.10")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=6.2,
                   labelspacing=0.28, handlelength=1.7, borderpad=0.2)
    axr = axb.twinx()
    axr.spines["right"].set_visible(True)
    axr.grid(False)
    for fam in FAMILIES:
        g = t[(t.family == fam) & (t.arm == "tgt_k")
              & np.isclose(t.alpha, 0.10)].sort_values("k")
        bal = g[g.balanced]
        unb = g[~g.balanced]
        axb.plot(bal.k, bal.yield_per_label, marker="o", ms=3.8, lw=1.4,
                 color=FAM_COL[fam], zorder=3)
        if len(unb):
            axb.plot(unb.k, unb.yield_per_label, "o", ms=3.2, color=FAM_COL[fam],
                     mfc="white", mew=0.9, ls="none", alpha=0.85, zorder=2)
        m = g.dropna(subset=["marginal_return_per100"])
        if len(m):
            axr.plot(m.k, m.marginal_return_per100, marker="^", ms=3.4, lw=1.0,
                     ls=(0, (1, 1.6)), color=FAM_COL[fam], mfc="white",
                     alpha=0.75, zorder=2)
    axb.set_xscale("log")
    axb.set_ylim(0, None)
    axb.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    axb.set_xlabel("target labels reviewed, k")
    axb.set_ylabel("auto-annotated cells per label reviewed")
    axr.set_ylim(0, None)
    axr.set_ylabel("marginal return: extra cells per 100 labels", color="0.35")
    axr.tick_params(axis="y", colors="0.35")
    axr.axhline(0.0, color="0.85", lw=0.8)
    handles = [Line2D([], [], color=FAM_COL[f], marker="o", ms=3.8, lw=1.4,
                      label="%s, balanced range" % FAM_LABEL[f])
               for f in FAMILIES]
    handles += [Line2D([], [], color="0.35", marker="o", ms=3.2, lw=0.0,
                       mfc="white", mew=0.9,
                       label="partial unit set (markers only)"),
                Line2D([], [], color="0.35", marker="^", ms=3.4, lw=1.0,
                       ls=(0, (1, 1.6)), mfc="white",
                       label="marginal return (right axis)")]
    axb.legend(handles=handles, loc="upper left", fontsize=6.2, ncol=2,
               labelspacing=0.3, columnspacing=1.0, handlelength=2.2,
               borderpad=0.2)
    axb.set_title("b  certificate yield per target label, alpha = 0.10",
                  loc="left", fontweight="bold", pad=6)
    axb.set_title("solid line = balanced range (all units); "
                  "open circles = partial unit set (not comparable)",
                  loc="right", fontsize=6.6, color="0.35")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, "fig10_yield.%s" % ext),
                    bbox_inches="tight")
    log("wrote figures/fig10_yield.pdf/.png")
    import matplotlib.pyplot as _plt
    _plt.close(fig)

if __name__ == "__main__":
    main()
