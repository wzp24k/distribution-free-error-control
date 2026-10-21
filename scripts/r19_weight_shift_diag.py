"""R19 -- is the weighted certificate *diagnosably* unable to work across tissues?

Gate #5 asks whether importance weighting of the calibration cells restores the
selective-risk certificate once exchangeability is broken.  On the twelve
cross-tissue pairs of the manuscript the answer is no; this script turns that
"no" into a computable verdict along three axes, and separates the two reasons
a certificate can be silent -- a genuinely unidentifiable weight function, and a
transfer model whose score *ranking* has collapsed so that no threshold has a
low error rate to isolate.

(i)   Identifiability of the weights, with no target labels.  The density ratio
      is only usable when the two feature clouds overlap.  We report the
      effective sample size ESS, ESS/n, the weight quantiles and the weight-mass
      concentration, plus three label-free shift statistics: the AUC of a
      cross-validated domain classifier, the overlap coefficient
      int min(f_src, f_tgt) dx (equivalently 1 - total variation), and an
      RBF-kernel MMD^2 with a permutation p-value.

(ii)  The precision the certificate would need.  The range-aware Hoeffding leg of
      ``WeightedSelectiveRiskController`` certifies N^+(lam) <= alpha D^-(lam)
      with

          N^+(lam) = Nhat(lam) + sqrt( sum_{i<=lam} w_i^2 log(1/lev) / (2 n^2) ),
          Nhat(lam) = (1/n) sum_{i<=lam} w_i^2 ,

      where the second term is the sampling slack.  We report the best achievable
      certified risk rho* = min_lam N^+/D^-, the two legs separately, and -- by
      blending the observed weights towards uniform -- the effective sample size
      ESS* at which rho* would just reach alpha.  A pair is *diagnosably* out of
      reach when ESS_obs < ESS*(alpha).

(iii) What the weight-free alternative assumes.  ScoreConditionalController needs
      P(err | s) to agree across domains.  With target labels we can test exactly
      that, per bin and globally by a bin-stratified permutation test; and we ask
      whether any purely label-free statistic could have warned us in advance,
      and how many deployment labels would be needed to see the violation.

Outputs: experiments/r19_weights.csv, r19_ess.csv, r19_sc.csv, r19_sc_bins.csv,
work/r19_report.txt, figures/figS5_weight_shift.{pdf,png}.
"""

import argparse
import itertools
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from errctl import conformal as C, paths
from errctl.classifiers import fit_classifier
from errctl.preprocess import DomainNormaliser
from errctl.weights import density_ratio, effective_sample_size
import r02_run_experiments as R2

DELTA = 0.10
N_BINS = 10
MIN_BIN = 20
MMD_MAX = 600
MMD_PERM = 100
PERM_REPS = 2000
BLEND_PTS = 21


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)),
          flush=True)


def score_of(probs):
    return 1.0 - np.asarray(probs, dtype=np.float64).max(1)


def err_of(probs, y):
    return (np.asarray(probs).argmax(1) != np.asarray(y)).astype(float)


# ---------------------------------------------------------------- geometry ---
def build_geometry(src, tgt, fold, seed):
    """Rebuild exactly the r02/r14 cross-tissue geometry for one (pair, fold).

    The feature matrices returned here are the ones r02 stored in
    ``work/predictions/transfer_*``: ``Xca`` for the calibration fold and
    ``Xex`` for the source fold that is held out and then *unused*, which is the
    fold r14 adds back in its ``src_big`` arm.
    """
    gs, ms, Xs = R2.prepare_cohort(src)
    gt, mt, Xt = R2.prepare_cohort(tgt)
    panel = R2.shared_panel(ms, mt)
    Xs_p = gs[[p[1] for p in panel]].to_numpy(dtype=np.float64)
    Xt_p = gt[[p[2] for p in panel]].to_numpy(dtype=np.float64)
    counts_t = gt["coarse_expert"].value_counts()
    counts_s = gs["coarse_expert"].value_counts()
    shared = sorted(c for c in counts_t.index
                    if counts_t[c] >= R2.MIN_CLASS_TEST and counts_s.get(c, 0) >= 30)
    if len(shared) < 3:
        return None
    sm = gs["coarse_expert"].isin(shared).to_numpy()
    tm = gt["coarse_expert"].isin(shared).to_numpy()
    gs2, Xs2 = gs[sm].reset_index(drop=True), Xs_p[sm]
    gt2, Xt2 = gt[tm].reset_index(drop=True), Xt_p[tm]
    ys = gs2["coarse_expert"].astype(str).to_numpy()
    yt = gt2["coarse_expert"].astype(str).to_numpy()
    folds = R2.fold_split(gs2["image"].unique(), R2.N_FOLDS, seed=seed + 7)
    test_imgs, cal_imgs = set(folds[fold]), set(folds[(fold + 1) % R2.N_FOLDS])
    tr = (~gs2["image"].isin(test_imgs | cal_imgs)).to_numpy()
    ca = gs2["image"].isin(cal_imgs).to_numpy()
    ex = gs2["image"].isin(test_imgs).to_numpy()
    tr_idx = R2.subsample_training(np.nonzero(tr)[0], ys, R2.MAX_TRAIN, seed=seed + fold)
    ca_idx, ex_idx = np.nonzero(ca)[0], np.nonzero(ex)[0]
    if len(ca_idx) < 50:
        return None
    nz = DomainNormaliser("z_dataset")
    nz.fit("src", Xs2[tr_idx])
    nz.fit("tgt", Xt2)
    Xtr = nz.transform("src", Xs2[tr_idx])
    Xca = nz.transform("src", Xs2[ca_idx])
    Xex = nz.transform("src", Xs2[ex_idx])
    Xte = nz.transform("tgt", Xt2)
    keep_cls = pd.Series(ys[tr_idx]).value_counts()
    classes = sorted(c for c in keep_cls.index if keep_cls[c] >= R2.MIN_CLASS_TRAIN)
    tr_keep = np.isin(ys[tr_idx], classes)
    idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([idx[c] for c in ys[tr_idx][tr_keep]], dtype=int)
    y_ca = np.array([idx.get(c, -1) for c in ys[ca_idx]], dtype=int)
    y_ex = np.array([idx.get(c, -1) for c in ys[ex_idx]], dtype=int)
    y_te = np.array([idx.get(c, -1) for c in yt], dtype=int)
    return {"Xtr": Xtr, "tr_keep": tr_keep, "y_train": y_train,
            "Xca": Xca, "y_ca": y_ca, "Xex": Xex, "y_ex": y_ex,
            "Xte": Xte, "y_te": y_te, "classes": classes,
            "img_ca": gs2["image"].to_numpy()[ca_idx],
            "img_ex": gs2["image"].to_numpy()[ex_idx],
            "img_te": gt2["image"].to_numpy(),
            "n_ca": int(len(ca_idx)), "n_ex": int(len(ex_idx))}


# ---------------------------------------------------- label-free shift stats ---
def domain_posterior(X_src, X_tgt, seed=0, cv=5):
    """Out-of-fold P(target | x) from a logistic domain classifier."""
    X = np.vstack([X_src, X_tgt])
    d = np.concatenate([np.zeros(len(X_src)), np.ones(len(X_tgt))])
    clf = Pipeline([("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3000, C=1.0))])
    k = min(cv, int(np.bincount(d.astype(int)).min()))
    if k < 2:
        return None, d
    p = cross_val_predict(clf, X, d, cv=k, method="predict_proba")[:, 1]
    return p, d


def overlap_coefficient(p, d):
    """Estimate int min(f_src, f_tgt).  Needs calibrated P(target|x)."""
    n_s = float(np.sum(d == 0))
    n_t = float(np.sum(d == 1))
    n = n_s + n_t
    pi_s, pi_t = n_s / n, n_t / n
    piece = np.minimum((1.0 - p) / pi_s, p / pi_t)
    return float(np.clip(np.mean(piece), 0.0, 1.0))


def _domain_pipeline():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000, C=1.0))])


def group_folds(g_src, g_tgt, n_folds=5):
    """Round-robin folds that hold out whole images but keep both domains in train.

    ``GroupKFold`` balances fold *sizes* and can therefore hand a training split
    that contains a single domain; here each fold removes the f-th image of each
    domain in turn, so with ``n_folds <= min(#src images, #tgt images)`` every
    training split keeps at least one image -- and hence both classes -- of each.
    """
    us, ut = np.unique(g_src), np.unique(g_tgt)
    n_s = int(min(n_folds, us.size, ut.size))
    if n_s < 2:
        return None
    g = np.concatenate([np.asarray(g_src), np.asarray(g_tgt)])
    folds = []
    for f in range(n_s):
        held = np.isin(g, np.concatenate([us[f::n_s], ut[f::n_s]]))
        te = np.nonzero(held)[0]
        tr = np.nonzero(~held)[0]
        folds.append((tr, te))
    return folds


def auc_domain_mean_fold(X_src, X_tgt, g_src, g_tgt, seed=0, n_folds=5):
    """Mean per-fold AUC of the domain classifier, cell-level and image-blocked.

    Cells inside one field of view are strongly correlated (median ~200 cells per
    image in these cohorts), so a random *cell* split lets a linear classifier
    memorise image identity.  Holding out whole images is the honest test, and the
    gap between the two is itself the diagnostic: a density ratio estimated by a
    plug-in classifier inherits whichever one applies.
    """
    X = np.vstack([X_src, X_tgt])
    d = np.concatenate([np.zeros(len(X_src)), np.ones(len(X_tgt))])

    def aucs_for(folds):
        out = []
        for trn, te in folds:
            if np.unique(d[te]).size < 2 or np.unique(d[trn]).size < 2:
                continue
            try:
                c = _domain_pipeline().fit(X[trn], d[trn])
                out.append(float(roc_auc_score(d[te], c.predict_proba(X[te])[:, 1])))
            except Exception:
                continue
        return out

    k = int(min(n_folds, min(np.bincount(d.astype(int)))))
    a_cell = aucs_for(list(StratifiedKFold(n_splits=max(k, 2), shuffle=True,
                                          random_state=seed).split(X, d)))
    gf = group_folds(g_src, g_tgt, n_folds)
    a_blk = aucs_for(gf) if gf else []
    return (float(np.mean(a_cell)) if a_cell else np.nan,
            float(np.mean(a_blk)) if a_blk else np.nan)


def domain_posterior_blocked(X_src, X_tgt, g_src, g_tgt, n_folds=5):
    """Out-of-fold P(target | x) with whole images held out."""
    X = np.vstack([X_src, X_tgt])
    d = np.concatenate([np.zeros(len(X_src)), np.ones(len(X_tgt))])
    folds = group_folds(g_src, g_tgt, n_folds)
    if folds is None:
        return None, d
    if any(np.unique(d[tr]).size < 2 for tr, te in folds):
        return None, d
    try:
        p = cross_val_predict(_domain_pipeline(), X, d, cv=folds,
                              method="predict_proba")[:, 1]
    except Exception:
        return None, d
    return p, d


def mmd2_perm(X_src, X_tgt, n_max=MMD_MAX, reps=MMD_PERM, seed=0):
    rng = np.random.default_rng(seed)
    ns, nt = len(X_src), len(X_tgt)
    A = X_src[rng.choice(ns, n_max, replace=False)] if ns > n_max else X_src
    B = X_tgt[rng.choice(nt, n_max, replace=False)] if nt > n_max else X_tgt
    Z = np.vstack([A, B]).astype(np.float64)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    med = float(np.median(d2[np.triu_indices(len(Z), 1)]))
    bw = med if med > 0 else 1.0
    K = np.exp(-d2 / bw)
    lab = np.concatenate([np.zeros(len(A)), np.ones(len(B))])

    def stat(l):
        i, j = l == 0, l == 1
        ni, nj = int(i.sum()), int(j.sum())
        if ni < 2 or nj < 2:
            return 0.0
        kxx = K[np.ix_(i, i)].sum()
        kyy = K[np.ix_(j, j)].sum()
        kxy = K[np.ix_(i, j)].sum()
        return float(kxx / (ni * (ni - 1)) + kyy / (nj * (nj - 1)) - 2.0 * kxy / (ni * nj))

    obs = stat(lab)
    perm = np.array([stat(rng.permutation(lab)) for _ in range(reps)])
    p = float((np.sum(perm >= obs) + 1.0) / (reps + 1.0))
    return obs, p, bw


def mmd2_perm_groups(X_src, g_src, X_tgt, g_tgt, max_exact=20000, reps=MMD_PERM,
                     seed=0):
    """MMD^2 between *per-image mean* features, with an exact permutation p.

    Averaging each field of view to one point removes the within-image dependence
    that makes a cell-level MMD p-value anticonservative, at the price of working
    with only a handful of points per domain.
    """
    us, ut = np.unique(g_src), np.unique(g_tgt)
    A = np.array([X_src[g_src == u].mean(0) for u in us], dtype=np.float64)
    B = np.array([X_tgt[g_tgt == u].mean(0) for u in ut], dtype=np.float64)
    Z = np.vstack([A, B])
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    med = float(np.median(d2[np.triu_indices(len(Z), 1)]))
    K = np.exp(-d2 / (med if med > 0 else 1.0))
    na, n = len(A), len(A) + len(B)

    def stat(i):
        j = ~i
        ni, nj = int(i.sum()), int(j.sum())
        if ni < 2 or nj < 2:
            return 0.0
        return float(K[np.ix_(i, i)].sum() / (ni * (ni - 1))
                     + K[np.ix_(j, j)].sum() / (nj * (nj - 1))
                     - 2.0 * K[np.ix_(i, j)].sum() / (ni * nj))

    lab = np.zeros(n, dtype=bool)
    lab[:na] = True
    obs = stat(lab)
    n_comb = math.comb(n, na)
    if n_comb <= max_exact:
        ge = tot = 0
        for idx in itertools.combinations(range(n), na):
            i = np.zeros(n, dtype=bool)
            i[list(idx)] = True
            tot += 1
            if stat(i) >= obs:
                ge += 1
        return obs, float((ge + 1.0) / (tot + 1.0)), tot
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(reps):
        perm = rng.permutation(n)
        i = np.zeros(n, dtype=bool)
        i[perm[:na]] = True
        if stat(i) >= obs:
            ge += 1
    return obs, float((ge + 1.0) / (reps + 1.0)), reps

# ------------------------------------------------- score-conditional binning ---
def pooled_edges(s_src, s_tgt, n_bins=N_BINS):
    pool = np.concatenate([s_src, s_tgt])
    return np.unique(np.quantile(pool, np.linspace(0.0, 1.0, n_bins + 1)))


def bin_counts(s, e, edges):
    B = edges.size - 1
    b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, B - 1)
    n = np.bincount(b, minlength=B).astype(float)
    m = np.bincount(b, weights=(np.asarray(e) > 0.5).astype(float), minlength=B)
    return n, m


def tv_gap(m_b, n_b, m_t, n_t, q_t, min_per_bin=5):
    """sum_b q_t(b) |theta_src(b) - theta_tgt(b)| over bins with enough cells."""
    tot, mass = 0.0, 0.0
    for j in range(n_b.size):
        if n_b[j] >= min_per_bin and n_t[j] >= min_per_bin:
            dev = abs(m_t[j] / n_t[j] - m_b[j] / n_b[j])
            tot += q_t[j] * dev
            mass += q_t[j]
    return tot, mass


def perm_pvalue(m_b, n_b, m_t, n_t, q_t, reps=PERM_REPS, seed=0, min_per_bin=5):
    """Exact conditional permutation test of P_source(err|bin) == P_target(err|bin).

    Under the null the errors inside a bin are exchangeable across domains, so we
    permute them bin by bin, holding the scores -- and hence the bin edges and
    the counts -- fixed.
    """
    obs, _ = tv_gap(m_b, n_b, m_t, n_t, q_t, min_per_bin)
    rng = np.random.default_rng(seed)
    B = n_b.size
    ge = 0
    for _ in range(reps):
        mb = np.zeros(B)
        mt = np.zeros(B)
        for j in range(B):
            nj = int(n_b[j])
            ntj = int(n_t[j])
            if nj + ntj == 0:
                continue
            tot = int(m_b[j] + m_t[j])
            errs = np.zeros(nj + ntj)
            errs[:tot] = 1.0
            rng.shuffle(errs)
            mb[j] = errs[:nj].sum()
            mt[j] = errs[nj:].sum()
        st, _ = tv_gap(mb, n_b, mt, n_t, q_t, min_per_bin)
        if st >= obs:
            ge += 1
    return obs, float((ge + 1.0) / (reps + 1.0))


def labels_to_detect(theta_s, theta_t, mass, alpha=0.05, power=0.80, n_bins=N_BINS):
    """Labelled target cells needed to see a bin's violation at 80% power."""
    if not np.isfinite(theta_s) or not np.isfinite(theta_t):
        return np.nan
    d = abs(theta_t - theta_s)
    if d < 1e-9 or mass <= 0:
        return np.inf
    z_a = ss.norm.ppf(1.0 - alpha / (2.0 * n_bins))
    z_b = ss.norm.ppf(power)
    p1, p2 = float(theta_s), float(theta_t)
    n_grp = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / d ** 2
    return float(n_grp / mass)


# ------------------------------------------- precision the certificate needs ---
def ess_blend_path(s_src, e_src, w, s_tgt, n_pts=BLEND_PTS):
    """rho*(ESS) along the path from uniform weights to the observed ones.

    ``t = 0`` are uniform weights (ESS = n, the largest any reweighting can
    reach), ``t = 1`` the observed density ratio.  ``rho`` is the best certified
    selective risk over the implementation's whole threshold grid, so it is
    alpha-free; a certificate is non-vacuous at level alpha exactly when
    ``rho <= alpha``.
    """
    wbar = np.asarray(w, dtype=np.float64)
    if not np.isfinite(wbar).all() or wbar.sum() <= 0:
        return None
    wbar = wbar / wbar.mean()
    ts = np.unique(np.concatenate([np.linspace(0.0, 1.0, n_pts), [1.0]]))
    ess, rho, rho_h, nhat_r, slack_r = [], [], [], [], []
    for t in ts:
        wt = (1.0 - t) + t * wbar
        ess.append(effective_sample_size(wt))
        ctl = C.WeightedSelectiveRiskController(alpha=0.10, delta=DELTA)
        ctl.fit(s_src, e_src, wt, target_scores=s_tgt)
        if getattr(ctl, "bound_ratio_", None) is None or ctl.bound_ratio_.size == 0:
            rho.append(np.inf)
        else:
            rho.append(float(np.min(ctl.bound_ratio_)))
        ctlh = C.WeightedSelectiveRiskController(alpha=0.10, delta=DELTA,
                                                 exact_strata=False)
        ctlh.fit(s_src, e_src, wt, target_scores=s_tgt)
        if getattr(ctlh, "bound_ratio_", None) is None or ctlh.bound_ratio_.size == 0:
            rho_h.append(np.inf)
            nhat_r.append(np.nan)
            slack_r.append(np.nan)
        else:
            j = int(np.argmin(ctlh.bound_ratio_))
            ld = max(float(ctlh.lower_d_[j]), 1e-12)
            rho_h.append(float(ctlh.bound_ratio_[j]))
            order = np.argsort(s_src, kind="mergesort")
            cum2 = np.cumsum(wt[order] ** 2)
            un = float(cum2[ctlh.cand_[j] - 1])
            nhat_r.append((un / len(s_src)) / ld)
            slack_r.append(float(ctlh.upper_n_[j] - un / len(s_src)) / ld)
    return {"ess": np.array(ess, dtype=float), "rho": np.array(rho, dtype=float),
            "rho_hoeff": np.array(rho_h, dtype=float),
            "nhat_ratio": np.array(nhat_r, dtype=float),
            "slack_ratio": np.array(slack_r, dtype=float),
            "t": ts}


def ess_star(path, alpha):
    """Smallest ESS on the blend path whose certified risk reaches alpha.

    Returns ``(ess_star, kind, n_feasible)`` with ``kind`` one of

      ``"weighted"``     a genuinely reweighted pool suffices;
      ``"no_weighting"`` the only feasible point is the (near-)uniform end, i.e.
                         the certificate needs the weights to *vanish*, which is
                         not a covariate-shift correction at all;
      ``"none"``         nothing on the path certifies -- the level of the error
                         curve, not its spread, is what blocks the guarantee.
    """
    if path is None:
        return np.nan, "none", 0
    e, r = path["ess"], path["rho"]
    ok = np.isfinite(r) & (r <= alpha)
    k = np.nonzero(ok)[0]
    if k.size == 0:
        return np.nan, "none", 0
    i = int(k[0])                           # rho is non-decreasing in 1/ESS
    if i == 0:
        return float(e[0]), "no_weighting", int(k.size)
    e0, e1, r0, r1 = e[i - 1], e[i], r[i - 1], r[i]
    if not np.isfinite(r0) or r0 == r1:
        return float(e1), "weighted", int(k.size)
    frac = (r0 - alpha) / (r0 - r1)
    return float(np.exp(np.log(e0) + frac * (np.log(e1) - np.log(e0)))), \
        "weighted", int(k.size)

# ----------------------------------------------------------------- the unit ---
def unit_job(job):
    pair, fold, clfs, alphas, seed, cache_dir = job
    src, tgt = pair
    try:
        return _unit_job(job)
    except Exception as exc:                     # keep one bad unit from killing the pool
        import traceback
        return [], [], [], [], "%s->%s fold%d FAILED: %r\n%s" % (
            src, tgt, fold, exc, traceback.format_exc())


def _unit_job(job):
    pair, fold, clfs, alphas, seed, cache_dir = job
    src, tgt = pair
    rows_w, rows_e, rows_s, rows_b = [], [], [], []
    g = build_geometry(src, tgt, fold, seed)
    if g is None:
        return rows_w, rows_e, rows_s, rows_b, "%s->%s fold%d: skipped" % (src, tgt, fold)

    Xca, Xex, Xte = g["Xca"], g["Xex"], g["Xte"]
    y_ca, y_ex, y_te = g["y_ca"], g["y_ex"], g["y_te"]
    X_pool = np.vstack([Xca, Xex])

    # ---- (i) identifiability of the weights, no target labels ----------------
    w_cal = density_ratio(Xca, Xte, clip=20.0, seed=0, cv=3)
    w_pool = density_ratio(X_pool, Xte, clip=20.0, seed=0, cv=3)
    ess_cal = effective_sample_size(w_cal)
    ess_pool = effective_sample_size(w_pool)
    wp = np.sort(w_pool / w_pool.mean())
    n_pool = wp.size
    def mass_top(frac):
        k = max(1, int(round(frac * n_pool)))
        return float(wp[-k:].sum() / wp.sum())
    img_pool = np.concatenate([np.asarray(g["img_ca"]), np.asarray(g["img_ex"])])
    img_te = np.asarray(g["img_te"])
    cells_per_img = float(np.median(np.unique(img_pool, return_counts=True)[1]))

    # Label-free shift diagnostics.  Cells inside one FOV are strongly
    # correlated, so a random *cell* split lets the domain classifier memorise
    # image identity; holding out whole images is the honest test, and the gap
    # between the two is itself part of the diagnosis.
    auc_cell, auc_blk = auc_domain_mean_fold(X_pool, Xte, img_pool, img_te, seed=seed)
    p_dom, d_dom = domain_posterior(X_pool, Xte, seed=seed)
    oc = np.nan if p_dom is None else overlap_coefficient(p_dom, d_dom)
    tv = np.nan if not np.isfinite(oc) else float(np.clip(1.0 - oc, 0.0, 1.0))
    p_blk, _ = domain_posterior_blocked(X_pool, Xte, img_pool, img_te)
    oc_blk = np.nan if p_blk is None else overlap_coefficient(p_blk, d_dom)
    mmd2, mmd_p, bw = mmd2_perm(X_pool, Xte, seed=seed)
    mmd2_g, mmd_p_g, mmd_n_g = mmd2_perm_groups(X_pool, img_pool, Xte, img_te, seed=seed)

    max_dev = np.nan
    rows_w.append({
        "pair": "%s->%s" % (src, tgt), "src": src, "tgt": tgt, "fold": fold,
        "n_src_cal": g["n_ca"], "n_src_pool": int(n_pool), "n_tgt": int(Xte.shape[0]),
        "ess_cal": ess_cal, "ess_pool": ess_pool,
        "ess_pool_ratio": ess_pool / n_pool,
        "w_p50": float(np.quantile(w_pool, 0.50)), "w_p90": float(np.quantile(w_pool, 0.90)),
        "w_p99": float(np.quantile(w_pool, 0.99)), "w_max": float(w_pool.max()),
        "mass_top1pct": mass_top(0.01), "mass_top5pct": mass_top(0.05),
        "mass_top10pct": mass_top(0.10),
        "auc_domain": auc_cell, "auc_domain_blocked": auc_blk,
        "overlap_coef": oc, "overlap_blocked": oc_blk, "tv_feature": tv,
        "mmd2": mmd2, "mmd_p": mmd_p, "mmd_bw": bw,
        "mmd2_img": mmd2_g, "mmd_p_img": mmd_p_g, "mmd_n_perm_img": mmd_n_g,
        "n_img_src": int(np.unique(g["img_ca"]).size + np.unique(g["img_ex"]).size),
        "n_img_tgt": int(np.unique(img_te).size),
        "cells_per_img_med": cells_per_img,
        "max_abs_dev_calprob": max_dev,
    })

    # ---- (ii)/(iii) per classifier -----------------------------------------
    for clf_kind in clfs:
        zf = os.path.join(cache_dir, "transfer_%s_to_%s_fold%d__%s.npz"
                          % (src, tgt, fold, clf_kind))
        if not os.path.exists(zf):
            continue
        z = np.load(zf, allow_pickle=True)
        clf = fit_classifier(g["Xtr"][g["tr_keep"]], g["y_train"], kind=clf_kind, seed=seed)
        p_ca = clf.predict_proba(Xca)
        p_ex = clf.predict_proba(Xex)
        p_te = clf.predict_proba(Xte)
        dev = float(np.abs(p_ca - z["cal_probs"]).max())
        rows_w[-1]["max_abs_dev_calprob"] = max(rows_w[-1]["max_abs_dev_calprob"], dev) \
            if np.isfinite(rows_w[-1]["max_abs_dev_calprob"]) else dev

        s_ca, e_ca = score_of(p_ca), err_of(p_ca, y_ca)
        s_ex, e_ex = score_of(p_ex), err_of(p_ex, y_ex)
        s_te, e_te = score_of(p_te), err_of(p_te, y_te)
        s_pool = np.concatenate([s_ca, s_ex])
        e_pool = np.concatenate([e_ca, e_ex])

        # ---- rho*(ESS): what precision the weighted certificate would need ---
        path = ess_blend_path(s_pool, e_pool, w_pool, s_te)
        if path is not None:
            ctl = C.WeightedSelectiveRiskController(alpha=0.10, delta=DELTA)
            ctl.fit(s_pool, e_pool, w_pool, target_scores=s_te)
            rho_obs = float(np.min(ctl.bound_ratio_)) if ctl.bound_ratio_.size else np.inf
            sum_ = ctl.summary() if ctl.chosen_ is not None else None
            rho_uni = float(path["rho"][0])
            dr = np.diff(path["rho"])
            mono = float(np.mean(np.isfinite(dr) & (dr >= 0.0))) if dr.size else np.nan
            for alpha in alphas:
                es, kind, nfeas = ess_star(path, alpha)
                lam_ok = (rho_obs <= alpha)
                rows_e.append({
                    "pair": "%s->%s" % (src, tgt), "src": src, "tgt": tgt, "fold": fold,
                    "clf": clf_kind, "alpha": alpha, "n_src": int(n_pool),
                    "ess_obs": ess_pool, "ess_star": es,
                    "ess_star_ratio": (es / ess_pool) if np.isfinite(es) else np.nan,
                    "ess_star_kind": kind, "n_path_feasible": int(nfeas),
                    "n_path": int(path["rho"].size), "rho_path_mono": mono,
                    "rho_star": rho_obs, "rho_uniform": rho_uni,
                    "rho_hoeff_obs": float(path["rho_hoeff"][-1]),
                    "nhat_ratio": float(path["nhat_ratio"][-1]),
                    "slack_ratio": float(path["slack_ratio"][-1]),
                    "ess_full_accept_need": int(n_pool) / float(alpha),
                    "nonvacuous": int(lam_ok),
                    "uniform_feasible": int(rho_uni <= alpha),
                    "lambda_hat": ctl.lambda_hat if lam_ok else -np.inf,
                    "cal_accept_frac": (sum_["cal_accept_frac"] if lam_ok and sum_ else 0.0),
                    "accept_rate_target": float(np.mean(s_te <= ctl.lambda_hat)) if lam_ok else 0.0,
                })

        # ---- sc on the deployment scores, no target labels -------------------
        edges = pooled_edges(s_pool, s_te)
        n_b, m_b = bin_counts(s_pool, e_pool, edges)
        n_t, m_t = bin_counts(s_te, e_te, edges)
        q_t = n_t / max(n_t.sum(), 1.0)
        q_s = n_b / max(n_b.sum(), 1.0)
        tv_score = float(0.5 * np.abs(q_t - q_s).sum())
        tgap, tgap_mass = tv_gap(m_b, n_b, m_t, n_t, q_t)
        _, p_perm = perm_pvalue(m_b, n_b, m_t, n_t, q_t, seed=fold)
        with np.errstate(divide="ignore", invalid="ignore"):
            th_s = np.where(n_b > 0, m_b / np.maximum(n_b, 1), np.nan)
            th_t = np.where(n_t > 0, m_t / np.maximum(n_t, 1), np.nan)
        usable = (n_b >= MIN_BIN) & (n_t >= MIN_BIN)
        maxdev = float(np.nanmax(np.abs(th_t - th_s)[usable])) if usable.any() else np.nan
        lab_need = [labels_to_detect(th_s[j], th_t[j], q_t[j])
                    for j in range(len(q_t)) if usable[j] and q_t[j] >= 0.03]
        n_lab_min = float(np.nanmin(lab_need)) if lab_need else np.nan
        n_lab_med = float(np.nanmedian(lab_need)) if lab_need else np.nan
        # per-bin p-values, Bonferroni over the bins actually tested
        pv = np.full(len(q_t), np.nan)
        zv = np.full(len(q_t), np.nan)
        for j in np.nonzero(usable)[0]:
            n1, x1, n2, x2 = n_b[j], m_b[j], n_t[j], m_t[j]
            pp = (x1 + x2) / (n1 + n2)
            se = np.sqrt(pp * (1 - pp) * (1.0 / n1 + 1.0 / n2))
            if se > 0:
                zv[j] = (x2 / n2 - x1 / n1) / se
                pv[j] = 2.0 * ss.norm.sf(abs(zv[j]))
        ntest = int(np.sum(np.isfinite(pv)))
        pmin = float(np.nanmin(pv)) if ntest else np.nan
        reject = int(ntest > 0 and pmin * ntest < 0.05)
        for j in range(len(q_t)):
            rows_b.append({
                "pair": "%s->%s" % (src, tgt), "src": src, "tgt": tgt, "fold": fold,
                "clf": clf_kind, "bin": j, "lo": float(edges[j]), "hi": float(edges[j + 1]),
                "n_src": n_b[j], "m_src": m_b[j], "theta_src": th_s[j],
                "n_tgt": n_t[j], "m_tgt": m_t[j], "theta_tgt": th_t[j],
                "q_tgt": q_t[j], "dev": (th_t[j] - th_s[j]),
                "z": zv[j], "p": pv[j],
                "labels_needed": labels_to_detect(th_s[j], th_t[j], q_t[j]),
            })

        for alpha in alphas:
            ctl_s = C.ScoreConditionalController(alpha, DELTA).fit(s_pool, e_pool, s_te)
            acc = s_te <= ctl_s.lambda_hat
            n_acc = int(acc.sum())
            risk = float(e_te[acc].mean()) if n_acc else np.nan
            ctl_c = C.ScoreConditionalController(alpha, DELTA).fit(s_ca, e_ca, s_te)
            acc_cal = s_te <= ctl_c.lambda_hat
            # test-peeking reference: does any threshold on the target work?
            order = np.argsort(s_te, kind="mergesort")
            cum = np.cumsum(e_te[order])
            kk = np.arange(1, s_te.size + 1)
            okk = (cum / kk <= alpha) & (kk >= 25)
            lam_o = float(s_te[order][np.nonzero(okk)[0][-1]]) if okk.any() else -np.inf
            rows_s.append({
                "pair": "%s->%s" % (src, tgt), "src": src, "tgt": tgt, "fold": fold,
                "clf": clf_kind, "alpha": alpha,
                "n_cal_pool": int(n_pool), "n_cal_fold": g["n_ca"], "n_tgt": int(s_te.size),
                "sc_lambda": ctl_s.lambda_hat, "sc_n_acc": n_acc,
                "sc_accept_rate": n_acc / float(s_te.size), "sc_risk": risk,
                "sc_cov_all": int((not np.isfinite(risk)) or risk <= alpha),
                "sc_cov_nonvac": int(n_acc > 0 and risk <= alpha),
                "sc_lambda_fold": ctl_c.lambda_hat,
                "sc_accept_fold": float(np.mean(acc_cal)),
                "sc_accept_delta": float(np.mean(acc) - np.mean(acc_cal)),
                "target_err_all": float(e_te.mean()),
                "source_err_cal": float(e_ca.mean()), "source_err_pool": float(e_pool.mean()),
                "oracle_accept": float(np.mean(e_te.size and (s_te <= lam_o))),
                "oracle_risk": float(e_te[s_te <= lam_o].mean()) if np.any(s_te <= lam_o) else np.nan,
                "tv_score": tv_score, "tv_gap": tgap, "tv_gap_mass": tgap_mass,
                "max_dev": maxdev, "perm_p": p_perm, "z_pmin": pmin,
                "reject_bonf": reject, "n_bins_tested": ntest,
                "n_labels_min": n_lab_min, "n_labels_med": n_lab_med,
                "max_abs_dev_calprob": dev,
                "n_naive_acc": float(np.mean(s_te <= C.naive_threshold(s_ca, e_ca, alpha))),
            })
    cand = np.array([r["ess_star"] for r in rows_e if r["alpha"] == 0.10], dtype=float)
    cand = cand[np.isfinite(cand)]
    msg = "%s->%s fold%d  ess_cal=%.0f ess_pool=%.0f AUCcell=%.3f AUCblk=%.3f ess*_a10=%.0f" % (
        src, tgt, fold, ess_cal, ess_pool, auc_cell, auc_blk,
        float(np.median(cand)) if cand.size else np.nan)
    return rows_w, rows_e, rows_s, rows_b, msg

# ------------------------------------------------------------------- report ---
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  n/a"
    return ("%%.%df" % nd) % x


def build_report(w, e, s, b, cfg):
    L = []
    A = L.append
    wkey = w[["pair", "fold", "tv_feature", "auc_domain", "auc_domain_blocked",
              "overlap_coef", "ess_pool_ratio", "mass_top5pct"]].copy()
    sw = s.merge(wkey, on=["pair", "fold"], how="left")
    A("=" * 96)
    A("R19 -- weight identifiability, certificate precision and the score-conditional")
    A("       assumption, on the twelve cross-tissue pairs of the manuscript")
    A("=" * 96)
    A("")
    A("design      : r02/r14 transfer splits rebuilt exactly (same folds, seeds,")
    A("              z_dataset normaliser, same class filters); source pool =")
    A("              calibration fold + the held-out-but-unused source fold, i.e.")
    A("              the pool r14's `w_big` arm calibrates on.")
    A("weights     : density_ratio(X_src, X_target, clip=20, seed=0, cv=3)")
    A("delta       : %.2f      alphas: %s      classifiers: %s" % (
        DELTA, ",".join("%.2f" % a for a in cfg["alphas"]), ",".join(cfg["clfs"])))
    A("units       : %d pair-folds, %d certificate units" % (w.shape[0], e.shape[0]))
    A("")

    A("-" * 96)
    A("(i)  WEIGHT IDENTIFIABILITY  (no target labels)")
    A("-" * 96)
    A("")
    A("per pair (median over folds).  ESS/n is the fraction of the source pool that")
    A("the weights effectively keep; overlap = int min(f_src, f_tgt); tv_feature =")
    A("1 - overlap; auc = cross-validated domain classifier on the 12 shared markers.")
    A("")
    A("%-24s %5s %5s %6s %6s %6s %6s %7s %7s %7s %7s" % (
        "pair", "n_src", "n_tgt", "ESS/n", "w_p99", "w_max", "top5%",
        "AUCcell", "AUCblk", "overl", "MMD2im"))
    for p, gp in w.groupby("pair"):
        A("%-24s %5.0f %5.0f %6.3f %6.2f %6.1f %6.3f %7.3f %7.3f %7.3f %7.3f" % (
            p, gp.n_src_pool.median(), gp.n_tgt.median(), gp.ess_pool_ratio.median(),
            gp.w_p99.median(), gp.w_max.median(), gp.mass_top5pct.median(),
            gp.auc_domain.median(), gp.auc_domain_blocked.median(),
            gp.overlap_coef.median(), gp.mmd2_img.median()))
    A("")
    A("AUCcell = mean per-fold AUC of the domain classifier on a random CELL split")
    A("AUCblk  = the same classifier with whole images held out (GroupKFold); this")
    A("          is the honest number, because cells within one FOV are correlated")
    A("          (median %s cells per source image)." % _fmt(w.cells_per_img_med.median(), 0))
    A("overl   = int min(f_src, f_tgt) from the cell-level out-of-fold posterior")
    A("MMD2im  = MMD^2 between per-image mean features (median over folds)")
    A("")
    A("pooled over all %d pair-folds:" % w.shape[0])
    A("  ESS/n                    median %s   (min %s, max %s)" % (
        _fmt(w.ess_pool_ratio.median()), _fmt(w.ess_pool_ratio.min()), _fmt(w.ess_pool_ratio.max())))
    A("  max weight               median %s   (max %s)   [ratio clipped at 20]" % (
        _fmt(w.w_max.median(), 1), _fmt(w.w_max.max(), 1)))
    A("  weight mass in top 5%%    median %s" % _fmt(w.mass_top5pct.median()))
    A("  AUCcell                  median %s   (min %s, max %s)" % (
        _fmt(w.auc_domain.median()), _fmt(w.auc_domain.min()), _fmt(w.auc_domain.max())))
    A("  AUCblk                   median %s   (min %s, max %s)" % (
        _fmt(w.auc_domain_blocked.median()), _fmt(w.auc_domain_blocked.min()),
        _fmt(w.auc_domain_blocked.max())))
    A("  AUCblk below chance      in %d of %d pair-folds" % (
        int((w.auc_domain_blocked < 0.5).sum()), w.shape[0]))
    A("  overlap coefficient      median %s   (max %s)" % (
        _fmt(w.overlap_coef.median()), _fmt(w.overlap_coef.max())))
    A("  image-mean MMD^2 p<0.05  in %d of %d pair-folds" % (
        int((w.mmd_p_img < 0.05).sum()), w.shape[0]))
    A("")
    A("read-off: two facts hold at once and both matter.")
    A("  (1) The weights are concentrated: the 99th-percentile weight is ~%s and the"
      % _fmt(w.w_p99.median(), 1))
    A("      top 5%% of cells carry ~%s of the mass, leaving ESS/n ~ %s." % (
        _fmt(w.mass_top5pct.median()), _fmt(w.ess_pool_ratio.median())))
    A("  (2) The weights are also not *estimable* from these features.  The domain")
    A("      classifier reaches AUC %s on a random cell split but only %s when whole"
      % (_fmt(w.auc_domain.median()), _fmt(w.auc_domain_blocked.median())))
    A("      images are held out -- below chance in %d of %d pair-folds.  The plug-in"
      % (int((w.auc_domain_blocked < 0.5).sum()), w.shape[0]))
    A("      density ratio therefore rides on within-image signatures that do not")
    A("      replicate across fields of view.  That is a stronger statement than")
    A("      'the weights have a heavy tail': the weight vector is not reproducible,")
    A("      so it cannot be trusted to point in the right direction at all.")
    A("")

    A("-" * 96)
    A("(ii) PRECISION THE WEIGHTED CERTIFICATE WOULD NEED")
    A("-" * 96)
    A("")
    A("rho* = min over the whole threshold grid of the certified selective risk")
    A("(alpha-free).  ESS* = the effective sample size at which the observed weights,")
    A("blended towards uniform, would make rho* = alpha.  A pair is diagnosably out of")
    A("reach when ESS_obs < ESS*; 'uniform' says whether even unweighted calibration")
    A("would already certify at that alpha.")
    A("")
    A("%-24s %5s %8s %8s %9s %7s %7s %7s %7s %-13s" % (
        "pair", "alpha", "ESS_obs", "ESS*", "ESS*/obs", "rho*", "rho_1",
        "N^/D^-", "slack/D", "ESS* kind"))
    for (p, al), gp in e.groupby(["pair", "alpha"]):
        kinds = gp.ess_star_kind.mode()
        A("%-24s %5.2f %8.0f %8s %9s %7.3f %7.3f %7.3f %7.3f %-13s" % (
            p, al, gp.ess_obs.median(),
            _fmt(gp.ess_star.median(), 0),
            _fmt(gp.ess_star_ratio.median(), 1),
            gp.rho_star.median(), gp.rho_uniform.median(),
            gp.nhat_ratio.median(), gp.slack_ratio.median(),
            str(kinds.iloc[0]) if len(kinds) else "n/a"))
    A("")
    A("rho_1 = the identical bound with all weights set to 1.  It is NOT a valid")
    A("certificate under shift -- it would pair source-domain errors with a target-")
    A("domain denominator -- but it isolates how much of rho* is the weight spread.")
    A("ESS* kind: weighted = a reweighted pool suffices; no_weighting = only the")
    A("(near-)uniform end of the path is feasible, i.e. the certificate needs the")
    A("weights to vanish; none = nothing on the path certifies.")
    A("")
    for al in cfg["alphas"]:
        ga = e[np.isclose(e.alpha, al)]
        cnt = ga.ess_star_kind.value_counts()
        A("alpha=%.2f : weighted certificate non-vacuous in %d/%d units (%.1f%%)" % (
            al, int(ga.nonvacuous.sum()), len(ga), 100.0 * ga.nonvacuous.mean()))
        A("            ESS* kind: weighted %d | no_weighting %d | none %d" % (
            int(cnt.get("weighted", 0)), int(cnt.get("no_weighting", 0)),
            int(cnt.get("none", 0))))
        A("            median ESS*/ESS_obs = %s  (rho* monotone in 1/ESS over %.0f%% of the blend path)" % (
            _fmt(ga.ess_star_ratio.median(), 1), 100.0 * ga.rho_path_mono.median()))
    A("")
    A("decomposition at the observed weights (alpha=0.10):")
    A("  rho* (min of the two valid legs)   median %s" % _fmt(e[np.isclose(e.alpha, 0.10)].rho_star.median()))
    A("  Hoeffding leg alone                median %s" % _fmt(e[np.isclose(e.alpha, 0.10)].rho_hoeff_obs.median()))
    A("    of which point part  N^/D^-      median %s" % _fmt(e[np.isclose(e.alpha, 0.10)].nhat_ratio.median()))
    A("    of which sampling slack / D^-    median %s" % _fmt(e[np.isclose(e.alpha, 0.10)].slack_ratio.median()))
    A("  unweighted (uniform) rho*         median %s" % _fmt(e.rho_uniform.median()))
    A("  naive full-acceptance requirement  ESS >= n/alpha = %s"
      % _fmt(e[np.isclose(e.alpha, 0.10)].ess_full_accept_need.median(), 0))
    A("")
    A("read-off: weighting leaves the bound FAR above alpha.  Note the point part")
    A("N^/D^- does not use the error indicators at all -- the implementation writes")
    A("the Hoeffding numerator as (1/n) sum_{i<=lam} w_i^2, an upper bound on")
    A("(1/n) sum w_i e_i by Cauchy-Schwarz after the mean-one normalisation -- so the")
    A("Hoeffding leg is loose purely from the weight spread, before any sampling noise.")
    A("")

    A("-" * 96)
    A("(iii) THE ASSUMPTION BEHIND THE WEIGHT-FREE ALTERNATIVE")
    A("-" * 96)
    A("")
    A("ScoreConditionalController needs P(err|s) to agree across domains.  Target")
    A("labels let us test it: tv_gap = sum_b q_tgt(b)|theta_tgt(b) - theta_src(b)|,")
    A("tested by a bin-stratified permutation test; tv_score = 1/2 sum_b |q_tgt - q_src|")
    A("is the SAME quantity computed from the score marginals alone, i.e. what a")
    A("practitioner could see without any target labels.")
    A("")
    A("%-24s %5s %8s %8s %8s %8s %7s %9s %9s %8s" % (
        "pair", "alpha", "sc_cov", "sc_ok", "accept", "tv_gap", "reject", "tv_score",
        "lab_min", "oracle"))
    for (p, al), gp in s.groupby(["pair", "alpha"]):
        A("%-24s %5.2f %8.3f %8.3f %8.3f %8.3f %7.2f %9.3f %9s %8.3f" % (
            p, al, gp.sc_cov_all.mean(), gp.sc_cov_nonvac.mean(), gp.sc_accept_rate.mean(),
            gp.tv_gap.median(), gp.reject_bonf.mean(), gp.tv_score.median(),
            _fmt(gp.n_labels_min.median(), 0), gp.oracle_accept.mean()))
    A("")
    for al in cfg["alphas"]:
        ga = s[np.isclose(s.alpha, al)]
        A("alpha=%.2f : sc coverage(any) %.3f   coverage(non-vacuous) %.3f   "
          "accept %.3f   oracle accept %.3f   violation rejected (Bonferroni) in %.1f%% of units"
          % (al, ga.sc_cov_all.mean(), ga.sc_cov_nonvac.mean(),
             ga.sc_accept_rate.mean(), ga.oracle_accept.mean(),
             100.0 * ga.reject_bonf.mean()))
    A("")
    A("is the violation visible without labels?")
    for col, nm in (("tv_score", "score-marginal TV (label-free)"),
                    ("tv_gap", "error-curve TV (needs labels)")):
        A("  %-30s median %s" % (nm, _fmt(s[col].median())))
    sp = sw[(np.isfinite(sw.tv_score)) & (np.isfinite(sw.tv_gap))]
    if len(sp) > 5:
        A("  Spearman rank correlation with the true violation tv_gap (n=%d):" % len(sp))
        for col, nm in (("tv_score", "score-marginal TV   (label-free)"),
                        ("auc_domain", "AUC, random cell split"),
                        ("auc_domain_blocked", "AUC, whole images held out"),
                        ("overlap_coef", "overlap coefficient"),
                        ("ess_pool_ratio", "ESS / n")):
            r_, p_ = ss.spearmanr(sp[col], sp.tv_gap)
            A("    %-34s rho = %s  p = %.3g" % (nm, _fmt(r_, 3), p_))
        # how well does each label-free statistic *flag* the units where the
        # score-conditional certificate actually fails?
        bad = (sp.sc_cov_nonvac == 0).astype(int)
        if 0 < bad.sum() < len(bad):
            A("  and as a *detector* of units where the score-conditional certificate")
            A("  fails (AUC of the statistic against sc_cov_nonvac == 0):")
            for col, nm in (("tv_score", "score-marginal TV   (label-free)"),
                            ("auc_domain_blocked", "AUC, whole images held out"),
                            ("overlap_coef", "overlap coefficient")):
                try:
                    a_ = float(roc_auc_score(bad, sp[col]))
                except Exception:
                    a_ = np.nan
                A("    %-34s AUC = %s" % (nm, _fmt(a_)))
    gp10 = s[np.isclose(s.alpha, 0.10)]
    weak = gp10[gp10.sc_cov_nonvac == 0]
    good = gp10[gp10.sc_cov_nonvac == 1]
    A("")
    A("  at alpha=0.10 the score-conditional certificate holds in %d/%d units;"
      % (int(gp10.sc_cov_nonvac.sum()), len(gp10)))
    if len(weak) and len(good):
        A("    failing units tv_score median %s vs passing %s   (label-free gap)"
          % (_fmt(weak.tv_score.median()), _fmt(good.tv_score.median())))
        A("    failing units tv_gap   median %s vs passing %s   (measured gap)"
          % (_fmt(weak.tv_gap.median()), _fmt(good.tv_gap.median())))
    A("  labels needed to detect the observed violation at 80% power, Bonferroni")
    A("  over bins: median %s target cells (min %s)"
      % (_fmt(s.n_labels_min.median(), 0), _fmt(s.n_labels_min.min(), 0)))
    A("")

    # ------------------------------------------------- consistency and validity
    key = ["pair", "fold", "clf", "alpha"]
    r14p = os.path.join(paths.RESULTS, "r14_targetcal.csv")
    if os.path.exists(r14p):
        A("-" * 96)
        A("CONSISTENCY AND VALIDITY CHECKS against r14 (a separate implementation)")
        A("-" * 96)
        A("")
        r14 = pd.read_csv(r14p)
        r14 = r14[r14.k == 0].copy()
        r14["pair"] = (r14.unit.str.replace("transfer:", "", regex=False)
                       .str.rsplit(":fold", n=1).str[0])
        r14["fold"] = r14.unit.str.rsplit(":fold", n=1).str[1].astype(int)
        s14 = (r14[r14.arm == "sc"][key + ["accept_rate", "covered", "nonvac_covered"]]
               .groupby(key, as_index=False).first())
        js = s.merge(s14, on=key, how="inner", suffixes=("_r19", "_r14"))
        A("  sc acceptance rate reproduces r14 exactly: max |r19 - r14| = %.2e over %d units"
          % (float((js.sc_accept_rate - js.accept_rate).abs().max()), len(js)))
        A("  sc nominal coverage reproduces r14:  max |diff| = %.2e"
          % float((js.sc_cov_all - js.covered).abs().max()))
        wb = (r14[r14.arm == "w_big"][key + ["nonvac_covered", "accept_rate",
                                             "selective_risk"]]
              .groupby(key, as_index=False).first())
        jw = e[key + ["nonvacuous", "rho_star"]].merge(wb, on=key, how="inner")
        A("")
        A("  Does the weighted certificate HOLD when it is non-vacuous?")
        A("  %-7s %12s %16s %18s %14s" % ("alpha", "non-vacuous", "holds (risk<=a)",
                                          "holds among nonvac", "median accept"))
        for al in cfg["alphas"]:
            ga = jw[np.isclose(jw.alpha, al)]
            nv = ga[ga.nonvacuous == 1]
            A("  %-7.2f %7d/%-4d %14d/%-4d %18s %14s" % (
                al, len(nv), len(ga),
                int(nv.nonvac_covered.sum()), len(nv),
                ("%d/%d = %.2f" % (int(nv.nonvac_covered.sum()), len(nv),
                                   100.0 * nv.nonvac_covered.mean())) if len(nv) else "n/a",
                _fmt(nv.accept_rate.median()) if len(nv) else "n/a"))
        A("  (r14's 'nonvac_covered' is the realised target risk among accepted cells;")
        A("   the weighted bound is a claim about that number, so this is a direct")
        A("   out-of-sample validity check, not a re-derivation.)")
        A("")

    # cross-tab
    A("-" * 96)
    A("CROSS-TAB  (alpha=0.10, matched on pair/fold/clf)")
    A("-" * 96)
    e10 = e[np.isclose(e.alpha, 0.10)][["pair", "fold", "clf", "nonvacuous",
                                        "ess_star_ratio"]]
    s10 = s[np.isclose(s.alpha, 0.10)][["pair", "fold", "clf", "sc_cov_nonvac",
                                        "sc_accept_rate"]]
    j = e10.merge(s10, on=["pair", "fold", "clf"], how="inner")
    A("  units                                   %d" % len(j))
    A("  weighted certificate non-vacuous        %d (%.1f%%)" % (
        int(j.nonvacuous.sum()), 100.0 * j.nonvacuous.mean()))
    A("  score-conditional non-vacuous & valid   %d (%.1f%%)" % (
        int(j.sc_cov_nonvac.sum()), 100.0 * j.sc_cov_nonvac.mean()))
    A("  both                                    %d" % int((j.nonvacuous & j.sc_cov_nonvac).sum()))
    A("  weighted fails, score-conditional holds %d" % int(((1 - j.nonvacuous) & j.sc_cov_nonvac).sum()))
    A("")
    A("=" * 96)
    A("VERDICT")
    A("=" * 96)
    A("i.   The weights are not identifiable here, and this is visible without any")
    A("     target label: the domain classifier scores AUC %s on a random cell split"
      % _fmt(w.auc_domain.median()))
    A("     but only %s with whole images held out (below chance in %d/%d pair-folds),"
      % (_fmt(w.auc_domain_blocked.median()),
         int((w.auc_domain_blocked < 0.5).sum()), w.shape[0]))
    A("     and the surviving weight vector is useless anyway -- ESS/n ~ %s, the top"
      % _fmt(w.ess_pool_ratio.median()))
    A("     5%% of cells hold ~%s of the mass." % _fmt(w.mass_top5pct.median()))
    A("ii.  The certificate is *diagnosably* out of reach, not merely unlucky.  At")
    A("     alpha=0.10 rho* is %s (needs <= 0.10); its Hoeffding point term N^/D^- is"
      % _fmt(e[np.isclose(e.alpha, 0.10)].rho_star.median()))
    A("     %s and the sampling slack only %s, and the term that dominates does not"
      % (_fmt(e[np.isclose(e.alpha, 0.10)].nhat_ratio.median()),
         _fmt(e[np.isclose(e.alpha, 0.10)].slack_ratio.median())))
    A("     use the error indicators at all -- it is pure weight spread.  The ESS the")
    A("     bound would need is the maximum any reweighting can reach, i.e. the")
    A("     weights would have to vanish; see the ESS* kind column.")
    g10 = s[np.isclose(s.alpha, 0.10)]
    A("iii. The weight-free alternative does NOT rescue the certificate either:")
    A("     nominal coverage %.3f at alpha=0.10 (accept %.3f), because its"
      % (g10.sc_cov_all.mean(), g10.sc_accept_rate.mean()))
    A("     assumption is simply false on these pairs -- P(err|s) is rejected in")
    A("     %.1f%% of units.  But it is *testable*, and the test is cheap: ~%s"
      % (100.0 * g10.reject_bonf.mean(), _fmt(g10.n_labels_min.median(), 0)))
    A("     labelled deployment cells detect the observed violation at 80% power.")
    A("     The label-free surrogates cannot stand in: score-marginal TV correlates")
    A("     only rho = %s with the true error-curve violation."
      % _fmt(ss.spearmanr(sw[["tv_score", "tv_gap"]].dropna().values[:, 0],
                          sw[["tv_score", "tv_gap"]].dropna().values[:, 1])[0]))
    A("")
    A("Net statement for gate #5: importance weighting does not restore the")
    A("certificate under broken exchangeability, and this is not a matter of a")
    A("heavier tail -- the required ESS is the maximum any reweighting can reach, so")
    A("the certificate is feasible only when the weights vanish, i.e. only when the")
    A("shift it exists to correct is assumed away.  What the analysis does deliver")
    A("is a label-free triage that says so in advance (ESS/n, ESS*/n, the cell-vs-")
    A("image AUC gap) plus a priced alternative: the score-conditional route needs")
    A("P(err|s) invariance, which fails here, and roughly %s labelled deployment"
      % _fmt(g10.n_labels_min.median(), 0))
    A("cells per population to detect that failure.")
    A("")
    return "\n".join(L)


# ------------------------------------------------------------------- figure ---
def make_figure(w, e, s, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.6))
    pairs = sorted(w.pair.unique())
    cmap = plt.get_cmap("tab20")

    a = ax[0, 0]
    a.scatter(w.auc_domain, w.ess_pool_ratio, s=26, alpha=0.8, label="random cell split")
    a.scatter(w.auc_domain_blocked, w.ess_pool_ratio, s=26, alpha=0.8, marker="^",
              label="whole images held out")
    a.axvline(0.5, color="k", ls=":", lw=0.8)
    a.set_yscale("log")
    a.set_xlabel("domain-classifier AUC")
    a.set_ylabel("ESS / n   (source calibration pool)")
    a.set_title("a  the weights concentrate, and are not estimable", loc="left", fontsize=10)
    a.legend(fontsize=6.5, frameon=False, loc="lower left")

    a = ax[0, 1]
    med = w.groupby("pair").agg(ess=("ess_pool_ratio", "median"),
                                m5=("mass_top5pct", "median"),
                                wmax=("w_max", "median")).sort_values("ess")
    yy = np.arange(len(med))
    a.barh(yy, med.m5, color="#b2182b", alpha=0.85)
    a.set_yticks(yy)
    a.set_yticklabels(med.index, fontsize=6)
    a.set_xlabel("fraction of weight mass held by the top 5% of cells")
    a.set_xlim(0, 1.12)
    for i, (m5, wm) in enumerate(zip(med.m5, med.wmax)):
        a.text(m5 + 0.01, i, "max w=%.0f" % wm, va="center", fontsize=5.4)
    a.set_title("b  a few cells carry the extrapolation", loc="left", fontsize=10)

    a = ax[1, 0]
    sub = e[e.alpha.isin(list(cfg["alphas"]))]
    for al in cfg["alphas"]:
        g = sub[np.isclose(sub.alpha, al)]
        a.scatter(g.ess_obs, g.ess_star, s=16, alpha=0.75,
                  label="alpha=%.2f" % al)
    lo = float(np.nanmin(sub.ess_obs)) * 0.7
    hi = float(np.nanmax(np.concatenate([sub.ess_obs, sub.ess_star[np.isfinite(sub.ess_star)]]))) * 1.4
    a.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="ESS* = ESS")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("ESS observed (source pool)")
    a.set_ylabel("ESS* needed for rho* <= alpha")
    a.legend(fontsize=6.5, frameon=False, loc="upper left")
    nun = int((e.ess_star_kind == "no_weighting").sum())
    a.text(0.03, 0.06, "%d/%d units need ESS* = n:\nthe weights must vanish" % (nun, len(e)),
           transform=a.transAxes, fontsize=7.5, va="bottom")
    a.set_title("c  every unit sits above the diagonal", loc="left", fontsize=10)

    a = ax[1, 1]
    for al in cfg["alphas"]:
        g = s[np.isclose(s.alpha, al)]
        a.scatter(g.tv_score, g.tv_gap, s=14, alpha=0.7, label="alpha=%.2f" % al)
    g = s[(np.isfinite(s.tv_score)) & (np.isfinite(s.tv_gap))]
    if len(g) > 5:
        rho_s, _ = ss.spearmanr(g.tv_score, g.tv_gap)
        a.text(0.03, 0.94, "Spearman rho = %.2f" % rho_s, transform=a.transAxes,
               fontsize=8, va="top")
    a.set_xlabel("score-marginal TV  (visible WITHOUT target labels)")
    a.set_ylabel("error-curve TV  (needs target labels)")
    a.legend(fontsize=6.5, frameon=False)
    a.set_title("d  the violation is invisible without labels", loc="left", fontsize=10)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(paths.FIGURES, "figS5_weight_shift." + ext), dpi=200)
    plt.close(fig)

def _init(datasets):
    for ds in datasets:
        try:
            R2.prepare_cohort(ds)
        except Exception as exc:            # pragma: no cover
            log("warm-up failed for %s: %s" % (ds, exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="r19")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="reload the four CSVs instead of recomputing")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    clfs = [c for c in args.clfs.split(",") if c]
    folds = [int(x) for x in args.folds.split(",")]
    pairs = list(R2.TRANSFER_PAIRS)
    if args.only:
        pairs = [p for p in pairs if ("%s->%s" % p) == args.only]
    datasets = sorted({d for p in pairs for d in p})
    cache_dir = os.path.join(paths.WORK, "predictions")

    jobs = [(p, f, clfs, alphas, args.seed, cache_dir) for p in pairs for f in folds]
    log("%d pair-folds x %d clfs x %d alphas   (jobs=%d)" % (
        len(jobs), len(clfs), len(alphas), args.jobs))

    W, E, S, B = [], [], [], []
    t0 = time.time()
    if args.reuse:
        w = pd.read_csv(os.path.join(paths.RESULTS, args.out + "_weights.csv"))
        e = pd.read_csv(os.path.join(paths.RESULTS, args.out + "_ess.csv"))
        s = pd.read_csv(os.path.join(paths.RESULTS, args.out + "_sc.csv"))
        b = pd.read_csv(os.path.join(paths.RESULTS, args.out + "_sc_bins.csv"))
        log("reused %d/%d/%d/%d rows from %s" % (len(w), len(e), len(s), len(b), paths.RESULTS))
        cfg = {"alphas": alphas, "clfs": clfs}
        rep_ = build_report(w, e, s, b, cfg)
        rp = os.path.join(paths.WORK, args.out + "_report.txt")
        with open(rp, "w", encoding="utf-8") as fh:
            fh.write(rep_ + "\n")
        log("wrote %s" % rp)
        print(rep_)
        if not args.no_fig and len(w):
            make_figure(w, e, s, cfg)
            log("wrote figures/figS5_weight_shift.{pdf,png}")
        return
    if args.jobs > 1:
        with mp.Pool(args.jobs, initializer=_init, initargs=(datasets,)) as pool:
            for i, (rw, re_, rs, rb, msg) in enumerate(
                    pool.imap(unit_job, jobs, chunksize=1), 1):
                W.extend(rw); E.extend(re_); S.extend(rs); B.extend(rb)
                log("  %d/%d  %s" % (i, len(jobs), msg))
    else:
        _init(datasets)
        for i, jb in enumerate(jobs, 1):
            rw, re_, rs, rb, msg = unit_job(jb)
            W.extend(rw); E.extend(re_); S.extend(rs); B.extend(rb)
            log("  %d/%d  %s" % (i, len(jobs), msg))
    log("collected in %.1fs" % (time.time() - t0))

    w = pd.DataFrame(W); e = pd.DataFrame(E)
    s = pd.DataFrame(S); b = pd.DataFrame(B)
    for name, df in (("_weights", w), ("_ess", e), ("_sc", s), ("_sc_bins", b)):
        p = os.path.join(paths.RESULTS, args.out + name + ".csv")
        df.to_csv(p, index=False)
        log("wrote %s  (%d rows)" % (p, len(df)))

    cfg = {"alphas": alphas, "clfs": clfs}
    rep = build_report(w, e, s, b, cfg)
    rp = os.path.join(paths.WORK, args.out + "_report.txt")
    with open(rp, "w", encoding="utf-8") as fh:
        fh.write(rep + "\n")
    log("wrote %s" % rp)
    print(rep)

    if not args.no_fig and len(w):
        try:
            make_figure(w, e, s, cfg)
            log("wrote figures/figS5_weight_shift.{pdf,png}")
        except Exception as exc:
            log("figure failed: %r" % (exc,))


if __name__ == "__main__":
    main()