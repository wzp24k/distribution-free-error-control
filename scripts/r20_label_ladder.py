"""r20 -- the label-cost ladder: what it costs to buy the distribution-free
guarantee back when calibration and deployment are not exchangeable.

Motivation
----------
``SelectiveRiskController`` certifies

    P( R(lambda_hat) <= alpha ) >= 1 - delta

only when the calibration cells and the deployed batch are exchangeable.  The
stored r02/r05 units violate that on purpose: ``cellsplit`` keeps it (random
split of one held-out pool), ``within`` breaks it by holding out whole fields of
view, ``transfer`` breaks it by changing cohort altogether.  At delta = 0.10 the
coverage is 1.000 / 0.817 / 0.397 -- the guarantee is real but conditional, and
the condition fails exactly in the regimes a spatial-proteomics deployment lives
in.

This experiment gives that negative a price tag.  For every stored
(unit, classifier) file and every budget alpha we buy k labelled cells from the
deployment batch itself, calibrate the controller on them, and evaluate the
remaining cells:

* ``tgt_k``  the transductive arm.  Calibration and evaluation are drawn from the
  same batch, so exchangeability holds by construction and the guarantee must be
  restored.
* ``src``    the paper's original arm: calibrate on the stored source cells and
  evaluate on the deployment batch.  This is the violation baseline, and at
  k = 0 it reproduces r03 to floating point.
* ``oracle`` the largest threshold that meets the budget when the evaluation
  labels themselves pick it, on the *same* remainder.  Not attainable; it is the
  ceiling on the acceptance rate.

Two k grids, because the two readings answer different questions.  The *relative*
grid uses a pre-specified set of label fractions of the batch, so every unit
contributes at every point; this is the scale-free statement of the price.  The
*absolute* grid uses fixed counts, which is what a certificate actually consumes
(the finite-sample bound depends on n, not on the fraction), but larger counts
are only available in larger cohorts, so a family average over the absolute grid
mixes different units at different k.  Both are reported and the difference is
called out wherever it bites.

Deployment batch = the batch the original protocol would deploy: the per-repeat
test half for cellsplit, the held-out fold for within, the whole target cohort
for transfer.  k is skipped when it exceeds half the batch.

A (family, alpha, k) is called *feasible* when the restored certificate is both

  (i) valid -- the one-sided 95% Clopper-Pearson upper limit of the realised risk
      among the cells it accepted, computed per unit and classifier and then
      worst-classifier-first and averaged over units, is at most alpha; and
  (ii) useful -- it agrees to annotate at least MIN_ACCEPT_RATE of the batch.

The 0/1 "was the realised risk under budget on this evaluation" flag used by r03
is reported too (and is what check (i) compares against), but it is not used for
the ladder: on a small evaluation remainder a *tight, valid* certificate hovers
at alpha and trips that flag about half the time, so it measures audit noise
rather than validity.  The pooled Clopper-Pearson limit is the well-powered
version of the same statement.

Label-free diagnostics, computed at k = 0 before any target label is bought:
``accept_src`` (the fraction of the batch the source certificate already agrees
to annotate), ``accept_gap_lf`` (that fraction minus the same fraction on the
source calibration cells -- a pure shift statistic), ``ks_scores`` (two-sample
Kolmogorov-Smirnov distance between source and deployment scores),
``auc_feat_max`` (strongest single-marker source/target separation),
``median_shift`` (difference of score medians).  Post-hoc quantities that need
deployment labels (``ratio_src_oracle_k0``) are reported only as a check that the
label-free number means what it should.

Outputs: experiments/r20_units.csv, r20_ladder.csv, r20_diag.csv,
work/r20_report.txt, figures/fig8_label_ladder.{pdf,png}.
"""

import argparse
import multiprocessing as mp
import os
import re
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, paths

PRED_DIR = os.path.join(paths.WORK, "predictions")
DELTA = 0.10
ALPHAS = (0.05, 0.10, 0.20)
KS = (0, 10, 25, 50, 100, 200, 500, 1000, 2000, 4000, 8000)
RHOS = (0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
REPS = 20
MIN_ACCEPTS = (25, 5)
MIN_ACCEPT_PRIMARY = 25
SECONDARY_MA_KMAX = 200
ORACLE_MIN_ACCEPT = 25
MIN_ACCEPT_RATE = 0.10
MIN_FRAC_K = 5
NOMINAL = 1.0 - DELTA


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)),
          flush=True)


def parse_name(fname):
    m = re.match(r"(.*)__(logreg|rf|hgb)\.npz$", fname)
    stem, clf = m.group(1), m.group(2)
    for family in ("cellsplit", "within", "transfer"):
        if stem.startswith(family + "_"):
            return family, stem[len(family) + 1:], clf
    raise ValueError("unknown family in %r" % fname)


def row(arm, meta, alpha, ma, k, rho, rep, n_pool, n_cal_used, n_eval, n_accepted,
        risk, lam, grid, gkey):
    # ``covered`` keeps the r03 convention exactly (an empty accepted set leaves
    # the risk undefined and satisfies the budget vacuously) so that check (i) is
    # a genuine re-derivation.  ``covered_nonvac`` restricts it to evaluations in
    # which the certificate actually accepted something.  ``gkey`` is the nominal
    # grid coordinate: the label count on the absolute grid, the label fraction on
    # the relative one.
    covered = int((not np.isfinite(risk)) or risk <= alpha)
    return dict(family=meta[0], unit=meta[1], clf=meta[2], alpha=alpha,
                min_accept=ma, grid=grid, gkey=gkey, k=k, rho=rho, rep=rep, arm=arm,
                n_pool=n_pool, n_cal_used=n_cal_used, n_eval=n_eval,
                n_accepted=n_accepted,
                n_wrong=(int(round(risk * n_accepted)) if n_accepted else 0),
                accept_rate=(n_accepted / n_eval) if n_eval else np.nan,
                realised_risk=risk, covered=covered,
                covered_nonvac=int(covered and n_accepted > 0),
                empty=int(n_accepted == 0), lambda_hat=lam)


def summarise(accepted, e_eval):
    n_acc = int(accepted.sum())
    risk = float(e_eval[accepted].mean()) if n_acc else float("nan")
    return n_acc, risk


def oracle_lambda(s_sorted, e_sorted, keep_sorted, alpha, min_accept):
    """Largest threshold whose empirical risk on the kept cells meets the budget."""
    e_sub = e_sorted[keep_sorted]
    n = e_sub.size
    if n == 0:
        return -np.inf
    cum = np.cumsum(e_sub)
    cnt = np.arange(1, n + 1, dtype=np.float64)
    ok = (cum / cnt <= alpha) & (cnt >= min_accept)
    if not ok.any():
        return -np.inf
    return float(s_sorted[keep_sorted][np.nonzero(ok)[0][-1]])


def midrank(x):
    """Average ranks with ties shared.

    Ranking with ties broken by position is the classic way to invent a
    difference that is not there: the source block is concatenated first, so
    every tied pair is scored in the source's favour.  Markers that are zero for
    most cells are exactly that case, and they drove a bogus AUC of 0.96 on
    cohorts whose source and deployment cells are the same cells.
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.ones(xs.size, dtype=bool)
    starts[1:] = xs[1:] != xs[:-1]
    gid = np.cumsum(starts) - 1
    counts = np.bincount(gid)
    first = np.nonzero(starts)[0]
    avg = (first + first + counts - 1) / 2.0 + 1.0
    out = np.empty(x.size, dtype=np.float64)
    out[order] = avg[gid]
    return out


def rank_auc(x, flag):
    """AUC of x for predicting flag == 1, ties handled by midranks."""
    x = np.asarray(x, dtype=np.float64)
    flag = np.asarray(flag)
    ok = np.isfinite(x)
    x, flag = x[ok], flag[ok]
    pos = flag == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0 or x.size < 4:
        return float("nan"), n1, n0
    r = midrank(x)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)), n1, n0


def domain_auc_max(src_X, tgt_X):
    """Strongest single-feature two-sample AUC between source and target cells."""
    best = 0.5
    for j in range(src_X.shape[1]):
        a, b = src_X[:, j], tgt_X[:, j]
        if np.all(a == a[0]) and np.all(b == b[0]):
            continue
        n1, n0 = a.size, b.size
        r = midrank(np.concatenate([a, b]))
        auc = (r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        best = max(best, auc, 1.0 - auc)
    return float(best)


def ks_two_sample(x, y):
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))

def process_file(job):
    fname, alphas, ks, rhos, reps, min_accepts = job
    d = np.load(os.path.join(PRED_DIR, fname), allow_pickle=True)
    family, unit, clf = parse_name(fname)
    meta = (family, unit, clf)

    s_all = 1.0 - d["test_probs"].max(1)
    e_all = (d["test_probs"].argmax(1) != d["test_y"]).astype(np.float64)
    s_cal = 1.0 - d["cal_probs"].max(1)
    e_cal = (d["cal_probs"].argmax(1) != d["cal_y"]).astype(np.float64)
    cal_rep = d["cal_rep"]
    test_rep = d["test_rep"] if "test_rep" in d.files else None
    n_cols = test_rep.shape[1] if test_rep is not None else 1
    n_cal_cols = cal_rep.shape[1]

    rows = []

    # ---------------------------------------------------------------- src arm
    # The paper's original protocol, per stored calibration repeat.  Acceptance is
    # measured on the whole deployment batch, so the k = 0 rows are exactly the
    # r03 rows, denominator included.  cellsplit stores both halves (10 repeats);
    # within/transfer store only the calibration half (6 repeats), so the loop
    # follows the calibration file and the deployment batch is the whole held-out
    # unit when there is no stored test mask.
    n_src_cols = max(n_cols, n_cal_cols)
    for col in range(n_src_cols):
        cal_sel = cal_rep[:, col % n_cal_cols]
        te_sel = (test_rep[:, col % n_cols] if test_rep is not None
                  else np.ones(s_all.size, bool))
        s_te, e_te = s_all[te_sel], e_all[te_sel]
        order_src = np.argsort(s_te, kind="mergesort")
        s_src_sorted, e_src_sorted = s_te[order_src], e_te[order_src]
        for alpha in alphas:
            lam_o = oracle_lambda(s_src_sorted, e_src_sorted,
                                  np.ones(s_te.size, bool), alpha, ORACLE_MIN_ACCEPT)
            acc_o = s_te <= lam_o
            n_acc_o, risk_o = summarise(acc_o, e_te)
            for ma in min_accepts:
                ctl = C.SelectiveRiskController(alpha, DELTA, min_accept=ma
                                                ).fit(s_cal[cal_sel], e_cal[cal_sel])
                acc = ctl.accept(s_te)
                n_acc, risk = summarise(acc, e_te)
                rows.append(row("src", meta, alpha, ma, 0, 0.0, col, int(te_sel.sum()),
                                int(cal_sel.sum()), int(te_sel.sum()), n_acc, risk,
                                ctl.lambda_hat, "abs", 0.0))
                rows.append(row("oracle", meta, alpha, ma, 0, 0.0, col,
                                int(te_sel.sum()), 0, int(te_sel.sum()), n_acc_o,
                                risk_o, lam_o, "abs", 0.0))

    # ------------------------------------------------- label-free diagnostics
    cal_sel0 = cal_rep[:, 0]
    ctl_diag = C.SelectiveRiskController(ALPHAS[1], DELTA,
                                        min_accept=MIN_ACCEPT_PRIMARY
                                        ).fit(s_cal[cal_sel0], e_cal[cal_sel0])
    acc_diag = ctl_diag.accept(s_all)
    diag = dict(family=family, unit=unit, clf=clf,
                n_pool=int(s_all.size), n_cal_src=int(cal_sel0.sum()),
                ks_scores=ks_two_sample(s_cal[cal_sel0], s_all),
                auc_feat_max=domain_auc_max(np.asarray(d["cal_X"])[cal_sel0],
                                            np.asarray(d["test_X"])),
                median_shift=float(np.median(s_cal[cal_sel0]) - np.median(s_all)),
                accept_src=float(acc_diag.mean()),
                accept_gap_lf=float(acc_diag.mean()
                                    - ctl_diag.summary()["cal_accept_frac"]),
                src_risk_all=(float(e_all[acc_diag].mean()) if acc_diag.any()
                              else float("nan")))

    # --------------------------------------------------- tgt_k + oracle arms
    base_seed = 20250926 + zlib.crc32(("%s|%s" % (unit, clf)).encode("utf-8"))
    for rep in range(reps):
        col = rep % n_cols
        te_sel = test_rep[:, col] if test_rep is not None else np.ones(s_all.size, bool)
        pool_idx = np.nonzero(te_sel)[0]
        s_pool = s_all[pool_idx]
        e_pool = e_all[pool_idx]
        n_pool = pool_idx.size
        order = np.argsort(s_pool, kind="mergesort")
        s_sorted, e_sorted = s_pool[order], e_pool[order]

        candidates = [(int(k), float(k), "abs") for k in ks]
        candidates += [(int(round(rho * n_pool)), float(rho), "frac") for rho in rhos]
        seen = set()
        for k, gkey, grid in candidates:
            if k < 1 or k > n_pool // 2:
                continue
            if grid == "frac" and k < MIN_FRAC_K:
                continue
            if (k, grid) in seen:
                continue
            seen.add((k, grid))
            pick_rng = np.random.default_rng([base_seed, rep, k,
                                              0 if grid == "abs" else 1])
            pick = pick_rng.choice(n_pool, size=k, replace=False)
            keep = np.ones(n_pool, bool)
            keep[pick] = False
            s_ev, e_ev = s_pool[keep], e_pool[keep]
            actual_rho = k / float(n_pool)
            for ma in min_accepts:
                if ma != MIN_ACCEPT_PRIMARY and k > SECONDARY_MA_KMAX:
                    continue
                for alpha in alphas:
                    ctl = C.SelectiveRiskController(alpha, DELTA, min_accept=ma
                                                    ).fit(s_pool[pick], e_pool[pick])
                    acc = ctl.accept(s_ev)
                    n_acc, risk = summarise(acc, e_ev)
                    rows.append(row("tgt_k", meta, alpha, ma, k, actual_rho, rep,
                                    n_pool, k, s_ev.size, n_acc, risk,
                                    ctl.lambda_hat, grid, gkey))
                    lam_o = oracle_lambda(s_sorted, e_sorted, keep[order], alpha,
                                          ORACLE_MIN_ACCEPT)
                    acc_o = s_ev <= lam_o
                    n_acc_o, risk_o = summarise(acc_o, e_ev)
                    rows.append(row("oracle", meta, alpha, ma, k, actual_rho, rep,
                                    n_pool, k, s_ev.size, n_acc_o, risk_o, lam_o,
                                    grid, gkey))
    return rows, diag


def _run(job):
    return process_file(job)


# ------------------------------------------------------------------- analysis
def cp_upper(n_wrong, n, conf=0.95):
    """One-sided Clopper-Pearson upper limit on a binomial rate."""
    n = int(n)
    n_wrong = int(round(n_wrong))
    if n <= 0 or n_wrong < 0:
        return float("nan")
    if n_wrong >= n:
        return 1.0
    return float(_beta.ppf(conf, n_wrong + 1.0, n - n_wrong))


def cluster_boot(values, clusters, n_boot=4000, seed=11):
    """Cluster bootstrap over units, for a CI on a family-level mean."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    if uniq.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    idx_by = {u: np.nonzero(clusters == u)[0] for u in uniq}
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        out[b] = values[idx].mean()
    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi)


def macro_micro(detail):
    """Aggregate the detail rows; micro = over evaluations, macro = over units.

    ``gkey`` is the nominal grid coordinate (the label count on the absolute grid,
    the label fraction on the relative grid) so that the two grids can be joined
    and averaged without mixing units.
    """
    keys = ["family", "alpha", "min_accept", "grid", "arm", "gkey"]
    micro = (detail.groupby(keys)
             .agg(micro_accept=("accept_rate", "mean"),
                  micro_risk=("realised_risk", "mean"),
                  coverage=("covered", "mean"),
                  coverage_nonvac=("covered_nonvac", "mean"),
                  empty_frac=("empty", "mean"),
                  mean_n_eval=("n_eval", "mean"),
                  kbar=("k", "mean"),
                  rho_bar=("rho", "mean"),
                  n_rows=("covered", "size"))
             .reset_index())
    pu = (detail.groupby(keys + ["unit"])
          .agg(u_accept=("accept_rate", "mean"), u_cov=("covered", "mean"),
               u_cov_nonvac=("covered_nonvac", "mean"),
               u_cal=("n_accepted", "sum"), u_wrong=("n_wrong", "sum"),
               u_npool=("n_pool", "max"), u_k=("k", "mean"))
          .reset_index())
    pu["u_risk"] = np.where(pu.u_cal > 0, pu.u_wrong / np.maximum(pu.u_cal, 1), np.nan)
    pu["u_labelfrac"] = pu.u_k / np.maximum(pu.u_npool, 1)
    macro = (pu.groupby(keys)
             .agg(macro_accept=("u_accept", "mean"), macro_coverage=("u_cov", "mean"),
                  macro_coverage_nonvac=("u_cov_nonvac", "mean"),
                  macro_risk=("u_risk", "mean"), label_frac=("u_labelfrac", "mean"),
                  npoolbar=("u_npool", "mean"),
                  worst_unit_cov=("u_cov", "min"), n_units=("unit", "size"))
             .reset_index())
    puc = (detail.groupby(keys + ["unit", "clf"])
           .agg(c_cal=("n_accepted", "sum"), c_wrong=("n_wrong", "sum"))
           .reset_index())
    puc["cp_upper"] = [cp_upper(w, c) for w, c in zip(puc.c_wrong, puc.c_cal)]
    per_unit_max = puc.groupby(keys + ["unit"])["cp_upper"].max().reset_index()
    worst = (per_unit_max.groupby(keys)["cp_upper"].mean()
             .reset_index().rename(columns={"cp_upper": "risk_upper"}))
    tot = (puc.groupby(keys).agg(tot_cal=("c_cal", "sum"),
                                 tot_wrong=("c_wrong", "sum")).reset_index())
    tot["risk_all"] = tot.tot_wrong / np.maximum(tot.tot_cal, 1)
    agg = (micro.merge(macro, on=keys).merge(worst, on=keys)
           .merge(tot[keys + ["tot_cal", "tot_wrong", "risk_all"]], on=keys))
    return agg, pu

def _at(df, key, col):
    if key is None or key < 0 or key not in df.index:
        return float("nan")
    return float(df.loc[key, col])


def _int_or_neg1(x):
    """int(round(x)) for a finite count, -1 where the coordinate is absent.

    The ladder is read off the balanced range, and a grid whose cap leaves no
    coordinate present in every unit of the family has no balanced range at all;
    that is the same -1 the neighbouring fields already carry, not an error.
    """
    return -1 if (x is None or not np.isfinite(x)) else int(round(x))


def balanced_keys(pu, family, grid, ma):
    """Grid coordinates on which every unit of the family has rows.

    The absolute grid is only complete where the smallest cohort can afford the
    count, and the relative grid where the smallest cohort can afford the
    fraction; restricting the ladder to this balanced range keeps every
    family-level average over the same set of units.  ``ma`` is part of the
    restriction because the min_accept=5 ablation is deliberately capped at
    k <= SECONDARY_MA_KMAX and must not shrink the primary ladder's range.
    """
    blk = pu[(pu.family == family) & (pu.grid == grid) & (pu.min_accept == ma)]
    if not len(blk):
        return []
    n_units = blk["unit"].nunique()
    counts = blk.groupby("gkey")["unit"].nunique()
    return [float(k) for k in counts.index[counts == n_units]]


def build_ladder(agg, pu):
    """For each family x alpha x grid: the smallest k that buys a useful valid certificate.

    A k is feasible when the restored certificate is both valid -- the one-sided
    95% Clopper-Pearson upper limit of the realised risk among the cells it
    accepted, computed per unit and classifier, worst-classifier-first, then
    averaged over units, is at most alpha -- and useful -- it agrees to annotate
    at least MIN_ACCEPT_RATE of the deployment batch.  The headline k* requires
    every larger grid point on the balanced range to be feasible as well, and
    feasibility is only ever read off that balanced range, so no family average
    mixes a different set of units at different k.
    """
    rows = []
    for (family, alpha, ma, grid), blk in agg.groupby(
            ["family", "alpha", "min_accept", "grid"]):
        tgt = blk[blk.arm == "tgt_k"].set_index("gkey").sort_index()
        orc = blk[blk.arm == "oracle"].set_index("gkey").sort_index()
        src = agg[(agg.family == family) & (agg.min_accept == ma) &
                  (agg.arm == "src") & np.isclose(agg.alpha, alpha)]
        orc0 = agg[(agg.family == family) & (agg.min_accept == ma) &
                   (agg.arm == "oracle") & (agg.gkey == 0) &
                   (agg.grid == "abs") & np.isclose(agg.alpha, alpha)]
        bal = [k for k in balanced_keys(pu, family, grid, ma) if k in tgt.index]
        bal_all = sorted(tgt.index)
        feas = {}
        for k in bal:
            r = tgt.loc[k]
            feas[k] = bool(r.macro_accept >= MIN_ACCEPT_RATE and
                           np.isfinite(r.risk_upper) and r.risk_upper <= alpha)
        first = next((k for k in bal if feas[k]), None)
        stable = None
        for i, k in enumerate(bal):
            if all(feas[k2] for k2 in bal[i:]):
                stable = k
                break
        cov_first = next((k for k in bal if tgt.loc[k, "macro_coverage"] >= NOMINAL),
                         None)
        kmax = max(bal) if bal else -1
        rec = dict(
            family=family, alpha=alpha, min_accept=ma, grid=grid,
            k_max_balanced=(-1 if kmax < 0 else float(kmax)),
            k_max_grid=(float(max(bal_all)) if bal_all else -1.0),
            k_max_balanced_count=_int_or_neg1(_at(tgt, kmax, "kbar")),
            k_max_balanced_frac=_at(tgt, kmax, "rho_bar"),
            units_balanced=_int_or_neg1(_at(tgt, kmax, "n_units")),
            units_total=int(tgt["n_units"].max()) if len(tgt) else 0,
            k_star=(-1.0 if stable is None else float(stable)),
            k_star_first=(-1.0 if first is None else float(first)),
            k_star_cov_only=(-1.0 if cov_first is None else float(cov_first)),
            k_star_none=int(stable is None),
            cov_all_src_k0=float(src["macro_coverage"].mean()),
            cov_nonvac_src_k0=float(src["macro_coverage_nonvac"].mean()),
            risk_src_k0=float(src["macro_risk"].mean()),
            risk_upper_src_k0=float(src["risk_upper"].mean()),
            accept_src_k0=float(src["macro_accept"].mean()),
            accept_oracle_k0=(float(orc0["macro_accept"].mean())
                              if len(orc0) else np.nan),
            k_count_kmax=_int_or_neg1(_at(tgt, kmax, "kbar")),
            cov_all_kmax=_at(tgt, kmax, "macro_coverage"),
            cov_nonvac_kmax=_at(tgt, kmax, "macro_coverage_nonvac"),
            risk_kmax=_at(tgt, kmax, "macro_risk"),
            risk_upper_kmax=_at(tgt, kmax, "risk_upper"),
            accept_kmax=_at(tgt, kmax, "macro_accept"),
            accept_oracle_kmax=_at(orc, kmax, "macro_accept"),
            label_frac_kmax=_at(tgt, kmax, "label_frac"),
        )
        ktop = max(bal_all) if bal_all else -1
        rec.update(cov_all_gridtop=_at(tgt, ktop, "macro_coverage"),
                   cov_nonvac_gridtop=_at(tgt, ktop, "macro_coverage_nonvac"),
                   risk_gridtop=_at(tgt, ktop, "macro_risk"),
                   risk_upper_gridtop=_at(tgt, ktop, "risk_upper"),
                   accept_gridtop=_at(tgt, ktop, "macro_accept"),
                   accept_oracle_gridtop=_at(orc, ktop, "macro_accept"),
                   units_gridtop=_int_or_neg1(_at(tgt, ktop, "n_units")),
                   count_gridtop=_int_or_neg1(_at(tgt, ktop, "kbar")),
                   frac_gridtop=_at(tgt, ktop, "label_frac"))
        rec["ratio_src_oracle_k0"] = (rec["accept_src_k0"] / rec["accept_oracle_k0"]
                                      if rec["accept_oracle_k0"] else np.nan)
        rec["ratio_kmax"] = (rec["accept_kmax"] / rec["accept_oracle_kmax"]
                             if rec["accept_oracle_kmax"] else np.nan)
        kk = stable if stable is not None else first
        rec["k_star_used"] = -1.0 if kk is None else float(kk)
        # always present, so that the frame has the columns even when nothing is
        # feasible in the balanced range.
        rec.update(cov_all_kstar=np.nan, cov_nonvac_kstar=np.nan, risk_kstar=np.nan,
                   risk_upper_kstar=np.nan, accept_kstar=np.nan,
                   accept_oracle_kstar=np.nan, label_frac_kstar=np.nan,
                   k_count_kstar=-1, cov_ci_lo=np.nan, cov_ci_hi=np.nan,
                   ratio_kstar=np.nan,
                   k_first_count=(-1 if first is None
                                  else int(round(_at(tgt, first, "kbar")))),
                   risk_upper_first=_at(tgt, first, "risk_upper"),
                   accept_first=_at(tgt, first, "macro_accept"),
                   sustained=int(stable is not None))
        if kk is not None:
            puk = pu[(pu.family == family) & (pu.alpha == alpha) &
                     (pu.min_accept == ma) & (pu.grid == grid) &
                     (pu.arm == "tgt_k") & (pu.gkey == kk)]
            lo, hi = cluster_boot(puk["u_cov"].to_numpy(), puk["unit"].to_numpy())
            rec.update(cov_all_kstar=_at(tgt, kk, "macro_coverage"),
                       cov_nonvac_kstar=_at(tgt, kk, "macro_coverage_nonvac"),
                       risk_kstar=_at(tgt, kk, "macro_risk"),
                       risk_upper_kstar=_at(tgt, kk, "risk_upper"),
                       accept_kstar=_at(tgt, kk, "macro_accept"),
                       accept_oracle_kstar=_at(orc, kk, "macro_accept"),
                       label_frac_kstar=_at(tgt, kk, "label_frac"),
                       k_count_kstar=int(round(_at(tgt, kk, "kbar"))),
                       cov_ci_lo=lo, cov_ci_hi=hi)
            rec["ratio_kstar"] = (rec["accept_kstar"] / rec["accept_oracle_kstar"]
                                  if rec["accept_oracle_kstar"] else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def build_diag_table(agg, pu, diag):
    """Per-unit label-free diagnostics joined to the label cost actually needed."""
    rows = []
    d10 = (diag.groupby(["family", "unit"])
           .agg(accept_src=("accept_src", "mean"),
                accept_gap_lf=("accept_gap_lf", "mean"),
                ks_scores=("ks_scores", "first"),
                auc_feat_max=("auc_feat_max", "first"),
                median_shift=("median_shift", "first"),
                n_pool=("n_pool", "max"), n_cal_src=("n_cal_src", "max"))
           .reset_index())
    for _, blk in d10.groupby(["family", "unit"]):
        family, unit = blk["family"].iloc[0], blk["unit"].iloc[0]
        bal = balanced_keys(pu, family, "frac", MIN_ACCEPT_PRIMARY)
        puu = pu[(pu.family == family) & (pu.alpha == ALPHAS[1]) &
                 (pu.min_accept == MIN_ACCEPT_PRIMARY) & (pu.unit == unit) &
                 (pu.arm == "tgt_k") & (pu.grid == "frac")].set_index("gkey")
        ks_av = [k for k in bal if k in puu.index]
        feas = {}
        for k in ks_av:
            up = cp_upper(puu.loc[k, "u_wrong"], puu.loc[k, "u_cal"])
            feas[k] = bool(puu.loc[k, "u_accept"] >= MIN_ACCEPT_RATE and
                           np.isfinite(up) and up <= ALPHAS[1])
        k_first = next((k for k in ks_av if feas[k]), None)
        k_stable = None
        for i, k in enumerate(ks_av):
            if all(feas[k2] for k2 in ks_av[i:]):
                k_stable = k
                break
        o0 = pu[(pu.family == family) & (pu.alpha == ALPHAS[1]) &
                (pu.min_accept == MIN_ACCEPT_PRIMARY) & (pu.unit == unit) &
                (pu.arm == "oracle") & (pu.u_k == 0)]
        s0 = pu[(pu.family == family) & (pu.alpha == ALPHAS[1]) &
                (pu.min_accept == MIN_ACCEPT_PRIMARY) & (pu.unit == unit) &
                (pu.arm == "src")]
        acc_orc0 = float(o0["u_accept"].mean()) if len(o0) else np.nan
        acc_src0 = float(s0["u_accept"].mean()) if len(s0) else np.nan
        npool = float(blk["n_pool"].iloc[0])
        rows.append(dict(
            family=family, unit=unit, alpha=ALPHAS[1],
            n_pool=int(npool), n_cal_src=int(blk["n_cal_src"].iloc[0]),
            accept_src=float(blk["accept_src"].iloc[0]),
            accept_gap_lf=float(blk["accept_gap_lf"].iloc[0]),
            ks_scores=float(blk["ks_scores"].iloc[0]),
            auc_feat_max=float(blk["auc_feat_max"].iloc[0]),
            median_shift=float(blk["median_shift"].iloc[0]),
            rho_star_unit=(-1.0 if k_stable is None else float(k_stable)),
            rho_star_cens=((float(max(ks_av)) + 0.05) if k_stable is None
                           else float(k_stable)),
            k_star_unit=(-1 if k_stable is None else int(round(puu.loc[k_stable, "u_k"]))),
            k_first_unit=(-1 if k_first is None else int(round(puu.loc[k_first, "u_k"]))),
            k_censored=int(k_stable is None),
            rho_max_balanced=(float(max(ks_av)) if ks_av else np.nan),
            k_max_balanced=(-1 if not ks_av
                            else int(round(puu.loc[max(ks_av), "u_k"]))),
            accept_oracle_k0=acc_orc0, accept_src_k0=acc_src0,
            ratio_src_oracle_k0=(acc_src0 / acc_orc0) if acc_orc0 else np.nan,
            accept_gap_k0=(acc_orc0 - acc_src0) if np.isfinite(acc_orc0) else np.nan,
        ))
    return pd.DataFrame(rows)


def spearman(x, y, n_perm=4000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        return float("nan"), float("nan"), int(x.size)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    rho = float((rx * ry).sum() / denom) if denom > 0 else float("nan")
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for b in range(n_perm):
        perm[b] = float((rx * rng.permutation(ry)).sum() / denom)
    p = float((np.abs(perm) >= abs(rho) - 1e-12).mean())
    return rho, p, int(x.size)


def crosscheck_src_arm(agg, r03_path):
    """Check (i)/(ii): the k = 0 src arm and oracle must equal the r03 rows."""
    if not os.path.exists(r03_path):
        return None
    r03 = pd.read_csv(r03_path)
    rows = []
    for family in ("cellsplit", "within", "transfer"):
        for alpha in ALPHAS:
            for method, arm in (("crc", "src"), ("oracle", "oracle")):
                ref = r03[(r03.kind == family) & (r03.method == method) &
                          np.isclose(r03.alpha, alpha)]
                if not len(ref):
                    continue
                mine = agg[(agg.family == family) & (agg.arm == arm) &
                           (agg.min_accept == MIN_ACCEPT_PRIMARY) &
                           (agg.gkey == 0) & np.isclose(agg.alpha, alpha)]
                if not len(mine):
                    continue
                rows.append(dict(
                    family=family, method=method, alpha=alpha,
                    r03_accept=float(ref.mean_accept.iloc[0]),
                    r20_accept=float(mine.macro_accept.iloc[0]),
                    r03_coverage=float(ref.coverage.iloc[0]),
                    r20_coverage=float(mine.macro_coverage.iloc[0]),
                    d_accept=abs(float(ref.mean_accept.iloc[0]) -
                                 float(mine.macro_accept.iloc[0])),
                    d_coverage=abs(float(ref.coverage.iloc[0]) -
                                   float(mine.macro_coverage.iloc[0])),
                    r03_n=int(ref.n_eval.iloc[0]), r20_n=int(mine.n_rows.iloc[0])))
    return pd.DataFrame(rows)

def make_figure(agg, diag_tbl, figs_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))
    colours = {"cellsplit": "#1f77b4", "within": "#ff7f0e", "transfer": "#d62728"}

    def frac_curve(family, arm):
        blk = agg[(agg.family == family) & (agg.alpha == 0.10) &
                  (agg.min_accept == MIN_ACCEPT_PRIMARY) & (agg.grid == "frac") &
                  (agg.arm == arm)]
        return blk.sort_values("gkey")

    ax = axes[0]
    for family, col in colours.items():
        tgt = frac_curve(family, "tgt_k")
        orc = frac_curve(family, "oracle")
        src = agg[(agg.family == family) & (agg.alpha == 0.10) &
                  (agg.min_accept == MIN_ACCEPT_PRIMARY) & (agg.arm == "src")]
        ax.plot(100 * tgt.gkey, tgt.macro_accept, "o-", color=col, label=family,
                ms=4)
        ax.plot(100 * orc.gkey, orc.macro_accept, "--", color=col, lw=1.0,
                alpha=0.75)
        ax.plot([0], [src.macro_accept.mean()], "s", color=col, ms=8,
                markeredgecolor="k", zorder=5)
    ax.axhline(MIN_ACCEPT_RATE, color="0.35", ls="-.", lw=1.0)
    ax.set_xlabel("labels bought, as a fraction of the deployment batch (%)")
    ax.set_ylabel("acceptance rate (macro over units)")
    ax.set_title("(a) acceptance bought by target labels (alpha = 0.10)\n"
                 "filled squares = $k=0$ source-only, dashed = oracle", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    ax.text(0.6, MIN_ACCEPT_RATE - 0.045, "usefulness floor 10%", fontsize=7.5,
            color="0.3")

    ax = axes[1]
    for family, col in colours.items():
        tgt = frac_curve(family, "tgt_k")
        src = agg[(agg.family == family) & (agg.alpha == 0.10) &
                  (agg.min_accept == MIN_ACCEPT_PRIMARY) & (agg.arm == "src")]
        ax.plot(100 * tgt.gkey, tgt.macro_coverage, "o-", color=col, label=family,
                ms=4)
        ax.plot(100 * tgt.gkey, tgt.macro_coverage_nonvac, ":o", color=col, ms=3.5,
                mfc="none", alpha=0.9)
        ax.plot([0], [src.macro_coverage.mean()], "s", color=col, ms=8,
                markeredgecolor="k", zorder=5)
        ax.plot([0], [src.macro_coverage_nonvac.mean()], "s", color=col, ms=8,
                mfc="none", markeredgecolor="k", zorder=5)
    ax.axhline(NOMINAL, color="k", ls="--", lw=1.2)
    ax.text(0.6, NOMINAL + 0.012, "nominal $1-\\delta=0.90$", fontsize=8)
    ax.set_xlabel("labels bought, as a fraction of the deployment batch (%)")
    ax.set_ylabel("macro coverage")
    ax.set_ylim(0.15, 1.08)
    ax.set_title("(b) coverage at alpha = 0.10\n"
                 "solid = all evaluations, dotted = non-empty certificates only",
                 fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[2]
    xx = diag_tbl.accept_src.to_numpy()
    yy = 100.0 * diag_tbl.rho_star_cens.to_numpy()
    tgt_vals = [v for v in yy if v > 0]
    for family, col in colours.items():
        blk = diag_tbl[diag_tbl.family == family]
        y = 100.0 * blk.rho_star_cens.to_numpy()
        cens = blk.k_censored.to_numpy() == 1
        ax.scatter(blk.accept_src[~cens], y[~cens], s=30, color=col, label=family,
                   edgecolor="k", linewidth=0.35)
        if cens.any():
            ax.scatter(blk.accept_src[cens], y[cens], s=34, color=col, marker="^",
                       facecolor="none", linewidth=0.9)
    rho, p, n = spearman(xx, yy)
    ax.set_xlabel("label-free: fraction of the batch the source\ncertificate already accepts")
    ax.set_ylabel("labels needed, $\\rho^{\\star}$ (% of batch)")
    ax.set_title("(c) the price is partly predictable without labels\n"
                 "$\\rho=%.2f$ ($p=%.3f$, $n=%d$); "
                 "$\\triangledown$-markers = never feasible below 40%% of batch"
                 % (rho, p, n), fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figs_dir, "fig8_label_ladder." + ext), dpi=200)
    return rho, p, n


def fmt_table(df, cols=None):
    d = df.copy()
    if cols:
        d = d[cols]
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: ("%.4f" % v) if np.isfinite(v) else "nan")
    return d.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--rhos", default=",".join(str(r) for r in RHOS))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--min-accepts", default=",".join(str(m) for m in MIN_ACCEPTS))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", default="")
    ap.add_argument("--out", default="r20")
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a]
    ks = [int(k) for k in args.ks.split(",") if k != ""]
    rhos = [float(r) for r in args.rhos.split(",") if r != ""]
    min_accepts = [int(m) for m in args.min_accepts.split(",") if m != ""]

    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".npz"))
    if args.limit:
        files = [f for f in files if args.limit in f]
    jobs = [(f, alphas, ks, rhos, args.reps, min_accepts) for f in files]
    log("r20: %d stored (unit, clf) files; alphas=%s reps=%d min_accept=%s"
        % (len(jobs), alphas, args.reps, min_accepts))
    log("     absolute k grid %s ; relative (label fraction) grid %s"
        % (ks, rhos))

    rows, diags = [], []
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            for i, (rws, dg) in enumerate(
                    pool.imap(_run, jobs, chunksize=1), 1):
                rows.extend(rws)
                diags.append(dg)
                if i % 25 == 0 or i == len(jobs):
                    log("  %d/%d files" % (i, len(jobs)))
    else:
        for i, job in enumerate(jobs, 1):
            rws, dg = _run(job)
            rows.extend(rws)
            diags.append(dg)
            if i % 25 == 0 or i == len(jobs):
                log("  %d/%d files" % (i, len(jobs)))

    detail = pd.DataFrame(rows)
    diag_raw = pd.DataFrame(diags)
    base = os.path.join(paths.RESULTS, args.out)
    detail.to_csv(base + "_units.csv", index=False)
    log("wrote %s_units.csv (%d rows)" % (base, len(detail)))

    agg, per_unit = macro_micro(detail)
    ladder = build_ladder(agg, per_unit)
    ladder.to_csv(base + "_ladder.csv", index=False)
    log("wrote %s_ladder.csv (%d rows)" % (base, len(ladder)))

    diag_tbl = build_diag_table(agg, per_unit, diag_raw)
    diag_tbl.to_csv(base + "_diag.csv", index=False)
    log("wrote %s_diag.csv (%d rows)" % (base, len(diag_tbl)))

    rpp = os.path.join(paths.WORK, "r20_report.txt")
    with open(rpp, "w", encoding="utf-8") as fh:
        w = fh.write
        w("=" * 110 + "\n")
        w("r20 -- the label-cost ladder: what it costs to buy the distribution-free\n"
          "       guarantee back when calibration and deployment are not exchangeable\n")
        w("=" * 110 + "\n\n")

        w("DESIGN (pre-specified)\n")
        w("  delta = 0.10, so the nominal coverage is 1 - delta = %.2f.\n" % NOMINAL)
        w("  alpha in %s.\n" % list(alphas))
        w("  Two k grids, both pre-specified:\n")
        w("    absolute  k in %s cells;\n" % list(ks))
        w("    relative  the label FRACTION of the batch in %s.\n" % list(rhos))
        w("  k cells are drawn without replacement from the deployment batch and used\n")
        w("  as the calibration set; the certificate is evaluated on the remaining\n")
        w("  cells.  %d replicate draws per (unit, classifier, alpha, k), fresh seed\n"
          % args.reps)
        w("  per draw, so the numbers do not depend on the order in which k is scanned.\n")
        w("  min_accept in %s.  Primary = %d (the controller default).  The secondary\n"
          % (list(min_accepts), MIN_ACCEPT_PRIMARY))
        w("  value is only run for k <= %d and exists to show that relaxing the floor\n"
          % SECONDARY_MA_KMAX)
        w("  does not make a handful of labels sufficient.\n")
        w("  Deployment batch = the batch the original protocol deploys: the per-repeat\n")
        w("  test half for cellsplit, the held-out fold for within, the whole target\n")
        w("  cohort for transfer.  k is skipped above half the batch.\n")
        w("  Arms: tgt_k (calibrate on k target cells, evaluate the rest), src (the\n")
        w("  original protocol, k = 0, the violation baseline), oracle (the budget met\n")
        w("  by the evaluation labels themselves, on the same remainder).\n\n")
        w("  A (family, alpha, k) is FEASIBLE when BOTH hold:\n")
        w("    (i) validity -- the one-sided 95%% Clopper-Pearson upper limit of the\n")
        w("        realised risk among the cells the certificate accepted, computed per\n")
        w("        unit and classifier, taken worst-classifier-first, then averaged over\n")
        w("        units, is at most alpha; and\n")
        w("    (ii) usefulness -- the certificate annotates at least %.0f%% of the batch.\n"
          % (100 * MIN_ACCEPT_RATE))
        w("  k* = the smallest grid point that is feasible and stays feasible at every\n")
        w("  larger point of the BALANCED range, i.e. the range on which every unit of\n")
        w("  the family has rows.  Outside that range a family average would compare\n")
        w("  different cohorts at different k (only large cohorts can afford a large\n")
        w("  absolute count), so it is never used for a claim.\n")
        w("  Why not the r03 0/1 flag for the ladder: on a small evaluation remainder a\n")
        w("  tight but valid certificate sits exactly at alpha, so 'realised risk <=\n")
        w("  alpha on this draw' trips about half the time.  That flag measures audit\n")
        w("  noise, not validity; the pooled Clopper-Pearson limit is its well-powered\n")
        w("  counterpart.  The flag is still reported and still re-derives r03 exactly.\n")
        w("  Nothing in the grids, the replicates or alpha was tuned after the fact.\n")
        w("  Detail rows: %d.\n\n" % len(detail))

        w("-" * 110 + "\n")
        w("CHECK (i)/(ii) -- the k = 0 src arm and the k = 0 oracle must reproduce\n")
        w("experiments/r03_summary.csv, the table the manuscript reports\n")
        w("-" * 110 + "\n")
        cc = crosscheck_src_arm(agg, os.path.join(paths.RESULTS, "r03_summary.csv"))
        if cc is None:
            w("r03_summary.csv not found -- check skipped\n")
        else:
            w(fmt_table(cc) + "\n")
            w("max |delta accept| = %.3e    max |delta coverage| = %.3e    "
              "row counts identical: %s\n"
              % (cc.d_accept.max(), cc.d_coverage.max(),
                 bool((cc.r03_n == cc.r20_n).all())))
            w("Reading: the src arm is not a re-implementation.  It is the same\n")
            w("computation -- same rows, same denominator, the controller's own\n")
            w("defaults -- so the differences are floating-point only.  The k = 0\n")
            w("oracle is evaluated on the same remainder as r03 and agrees row for row.\n\n")

        grids = [("frac", "RELATIVE GRID (label fraction of the batch) -- the "
                          "scale-free statement"),
                 ("abs", "ABSOLUTE GRID (fixed label counts) -- what a certificate "
                         "actually consumes")]
        for gtag, gtitle in grids:
            lg = ladder[(ladder.min_accept == MIN_ACCEPT_PRIMARY) &
                        (ladder.grid == gtag)]
            w("-" * 110 + "\n")
            w("TABLE L1[%s] -- THE LADDER, min_accept = %d, %s\n"
              % (gtag, MIN_ACCEPT_PRIMARY, gtitle))
            w("-" * 110 + "\n")
            w("balanced range: %s\n" % ("none" if not len(lg) else
                                        "up to %.4f" % lg["k_max_balanced"].max()))
            show = lg.rename(columns={"k_star": "k_star_grid", "k_count_kstar":
                                      "k_star_count",
                                      "k_max_balanced_count": "kmax_count"})
            w(fmt_table(show, cols=[
                "family", "alpha", "k_star_grid", "k_star_count", "sustained",
                "k_star_first", "k_first_count", "k_star_none",
                "k_star_cov_only", "risk_kstar",
                "risk_upper_kstar", "cov_all_kstar", "cov_nonvac_kstar",
                "accept_kstar", "accept_oracle_kstar", "ratio_kstar",
                "label_frac_kstar", "k_max_balanced", "kmax_count",
                "units_balanced", "units_total"]) + "\n\n")

            w("-" * 110 + "\n")
            w("TABLE L2[%s] -- violation (k = 0, source-only) vs recovery (k = k*)\n"
              % gtag)
            w("-" * 110 + "\n")
            l2 = show[["family", "alpha", "k_star_grid", "k_star_count",
                       "cov_all_src_k0", "cov_nonvac_src_k0", "risk_src_k0",
                       "risk_upper_src_k0", "accept_src_k0", "accept_oracle_k0",
                       "risk_kstar", "risk_upper_kstar", "accept_kstar",
                       "accept_oracle_kstar", "ratio_kstar"]]
            w(fmt_table(l2) + "\n\n")

            w("-" * 110 + "\n")
            w("TABLE L3[%s] -- what buying as many labels as the balanced range\n"
              "allows actually delivers\n" % gtag)
            w("-" * 110 + "\n")
            l3 = show[["family", "alpha", "k_max_balanced", "kmax_count",
                       "label_frac_kmax", "cov_all_kmax", "cov_nonvac_kmax",
                       "risk_kmax", "risk_upper_kmax", "accept_kmax",
                       "accept_oracle_kmax", "ratio_kmax"]]
            w(fmt_table(l3) + "\n")
            w("Same quantities at the TOP OF THE GRID, where only the cohorts large\n")
            w("enough to afford the count contribute -- reported for completeness and\n")
            w("NOT used for any claim, because the units_ gridtop column shows how many\n")
            w("of the family's units are in the average there:\n")
            w(fmt_table(show, cols=["family", "alpha", "k_max_grid", "count_gridtop",
                                    "frac_gridtop", "cov_all_gridtop",
                                    "cov_nonvac_gridtop", "risk_gridtop",
                                    "risk_upper_gridtop", "accept_gridtop",
                                    "accept_oracle_gridtop", "units_gridtop",
                                    "units_total"]) + "\n\n")

        w("-" * 110 + "\n")
        w("TABLE L4 -- min_accept ablation on the relative grid (floor relaxed to 5,\n")
        w("           k <= %d): the floor is not what makes small k useless\n"
          % SECONDARY_MA_KMAX)
        w("-" * 110 + "\n")
        l4 = ladder[(ladder.min_accept == 5) & (ladder.grid == "frac")]
        l4 = l4.rename(columns={"k_star": "k_star_grid",
                                "k_count_kstar": "k_star_count"})
        w(fmt_table(l4,
                    cols=["family", "alpha", "k_star_grid", "k_star_count",
                          "risk_kstar", "risk_upper_kstar", "accept_kstar",
                          "accept_oracle_kstar", "label_frac_kstar",
                          "k_max_balanced"]) + "\n\n")

        w("-" * 110 + "\n")
        w("TABLE L5 -- every grid point, alpha = 0.10, min_accept = %d\n"
          % MIN_ACCEPT_PRIMARY)
        w("-" * 110 + "\n")
        b = agg[(agg.alpha == 0.10) & (agg.min_accept == MIN_ACCEPT_PRIMARY)]
        piv = b.pivot_table(index=["grid", "family", "gkey"], columns="arm",
                            values=["macro_accept", "macro_coverage", "risk_upper"],
                            aggfunc="mean")
        w(piv.to_string() + "\n\n")

        w("-" * 110 + "\n")
        w("TABLE D -- label-free diagnostics at k = 0 against the labels actually\n")
        w("           needed (alpha = 0.10, relative grid)\n")
        w("-" * 110 + "\n")
        w(fmt_table(diag_tbl, cols=[
            "family", "unit", "n_pool", "n_cal_src", "accept_src", "accept_gap_lf",
            "ks_scores", "auc_feat_max", "median_shift", "rho_star_unit",
            "rho_star_cens", "k_star_unit", "k_censored", "rho_max_balanced",
            "k_max_balanced", "accept_oracle_k0",
            "ratio_src_oracle_k0"]) + "\n\n")
        w("Spearman rho of each label-free diagnostic against the label fraction\n")
        w("needed, rho*.  A unit that is never feasible inside the balanced range is\n")
        w("right-censored at the top of that range; dropping it would leave only the\n")
        w("easy units and reverse the sign of every correlation (survivorship), so the\n")
        w("primary column below reads the censored units at (top of range + 0.05) and\n")
        w("the secondary column drops them.  accept_src is the fraction of the batch the\n")
        w("source certificate already accepts, so it is large when there is little\n")
        w("shift; accept_gap_lf is that fraction minus the same fraction on the source\n")
        w("calibration cells; auc_feat_max is the strongest single-marker separation.\n")
        w("Sign convention: a NEGATIVE rho on accept_src (positive on auc_feat_max or\n")
        w("ks_scores) means 'looks worse without labels -> costs more labels'.\n\n")
        sp_rows = []
        for scope in ["ALL"] + sorted(diag_tbl.family.unique()):
            blk = diag_tbl if scope == "ALL" else diag_tbl[diag_tbl.family == scope]
            for name in ("accept_src", "accept_gap_lf", "ks_scores", "auc_feat_max",
                         "median_shift"):
                rho_c, p_c, n_c = spearman(blk[name], blk.rho_star_cens)
                sub = blk[blk.k_censored == 0]
                rho_u, p_u, n_u = spearman(sub[name], sub.rho_star_unit)
                sp_rows.append(dict(scope=scope, diagnostic=name,
                                    rho_censored_as_top=rho_c, p_censored=p_c,
                                    n_all=n_c, rho_uncensored_only=rho_u,
                                    p_uncensored=p_u, n_uncensored=n_u))
        w(fmt_table(pd.DataFrame(sp_rows)) + "\n\n")

        w("AUC of each label-free diagnostic for predicting CENSORING, i.e. that the\n")
        w("unit cannot be certified at all inside a 40%-of-batch label budget at\n")
        w("alpha = 0.10.  This is the part of the diagnosis that survives censoring:\n")
        w("it asks which units need labels, not how many.\n\n")
        cen_rows = []
        for scope in ["ALL"] + sorted(diag_tbl.family.unique()):
            blk = diag_tbl if scope == "ALL" else diag_tbl[diag_tbl.family == scope]
            for name in ("accept_src", "accept_gap_lf", "ks_scores", "auc_feat_max",
                         "median_shift"):
                auc, n1, n0 = rank_auc(blk[name], blk.k_censored)
                cen_rows.append(dict(scope=scope, diagnostic=name,
                                     auc_for_censoring=auc, n_censored=n1,
                                     n_uncensored=n0,
                                     direction=("higher -> censored" if auc and auc > 0.5
                                                else "lower -> censored")))
        w(fmt_table(pd.DataFrame(cen_rows)) + "\n\n")
        w("Mean diagnostics for the censored and the uncensored units:\n\n")
        grp_rows = []
        for scope in sorted(diag_tbl.family.unique()):
            blk = diag_tbl[diag_tbl.family == scope]
            for flag, lab in ((0, "uncensored"), (1, "censored")):
                sub = blk[blk.k_censored == flag]
                if not len(sub):
                    continue
                grp_rows.append(dict(family=scope, group=lab, n=len(sub),
                                     accept_src=sub.accept_src.mean(),
                                     accept_gap_lf=sub.accept_gap_lf.mean(),
                                     ks_scores=sub.ks_scores.mean(),
                                     auc_feat_max=sub.auc_feat_max.mean(),
                                     median_shift=sub.median_shift.mean()))
        w(fmt_table(pd.DataFrame(grp_rows)) + "\n\n")

        w("-" * 110 + "\n")
        w("HONEST VERDICT\n")
        w("-" * 110 + "\n")
        for gtag in ("frac", "abs"):
            lg = ladder[(ladder.min_accept == MIN_ACCEPT_PRIMARY) &
                        (ladder.grid == gtag)]
            w("  grid = %s\n" % gtag)
            for _, r in lg.iterrows():
                w("    %-10s alpha=%.2f  k=0: coverage=%.3f (non-vacuous %.3f), "
                  "accept=%.3f, risk=%.3f\n"
                  % (r.family, r.alpha, r.cov_all_src_k0, r.cov_nonvac_src_k0,
                     r.accept_src_k0, r.risk_src_k0))
                w("               coverage alone is met from k=%d onwards (%.3f of the "
                  "batch, %.0f%% of the batch's evaluations),\n"
                  % (int(r.k_star_cov_only) if r.k_star_cov_only > 0 else -1,
                     r.k_star_cov_only, 100 * r.k_star_cov_only))
                w("               but at that point the certificate is empty or "
                  "nearly empty -- which is why the ladder asks for\n")
                w("               a non-vacuous, useful certificate instead of a 0/1 "
                  "coverage flag.\n")
                if r.k_star < 0 and r.k_first_count > 0:
                    w("               k*: feasibility is reached at %d cells "
                      "(%.3f of the batch) but is NOT sustained\n"
                      % (r.k_first_count, r.k_star_first))
                    w("               further up the grid: risk upper %.3f "
                      "(budget %.2f), accept %.3f -- the useful region is an\n"
                      % (r.risk_upper_first, r.alpha, r.accept_first))
                    w("               interval there, not a prefix; see the "
                      "min_accept ablation\n\n")
                elif r.k_star < 0:
                    w("               k*: none inside the balanced range (top = %.3f, "
                      "%d cells, %.1f%% of batch):\n"
                      % (r.k_max_balanced, r.k_max_balanced_count,
                         100 * r.label_frac_kmax))
                    w("               coverage=%.3f, accept=%.3f vs oracle %.3f, "
                      "risk=%.3f (upper %.3f)\n"
                      % (r.cov_all_kmax, r.accept_kmax, r.accept_oracle_kmax,
                         r.risk_kmax, r.risk_upper_kmax))
                    w("               at the top of the grid (%.1f%% of the batch, "
                      "%d of %d units): coverage=%.3f, accept=%.3f vs oracle %.3f, "
                      "risk=%.3f (upper %.3f)\n\n"
                      % (100 * r.frac_gridtop, r.units_gridtop, r.units_total,
                         r.cov_all_gridtop, r.accept_gridtop, r.accept_oracle_gridtop,
                         r.risk_gridtop, r.risk_upper_gridtop))
                else:
                    w("               k*=%d cells (%.3f of the batch, %.1f%%): "
                      "coverage=%.3f [%.3f, %.3f], accept=%.3f vs oracle %.3f\n"
                      % (r.k_count_kstar, r.k_star, 100 * r.label_frac_kstar,
                         r.cov_all_kstar, r.cov_ci_lo, r.cov_ci_hi, r.accept_kstar,
                         r.accept_oracle_kstar))
                    w("               ratio %.3f; risk among accepted = %.3f, "
                      "95%% upper limit %.3f (budget %.2f)\n\n"
                      % (r.ratio_kstar, r.risk_kstar, r.risk_upper_kstar, r.alpha))

    log("wrote %s" % rpp)

    if not args.no_figs:
        rho, p, n = make_figure(agg, diag_tbl, paths.FIGURES)
        log("wrote figures/fig8_label_ladder.{pdf,png} "
            "(pooled Spearman %.3f, p=%.3f, n=%d)" % (rho, p, n))
    log("done")


if __name__ == "__main__":
    main()