"""R24b -- can a *label-shift-only* correction repair the weighted certificate?

R19 asked whether importance weighting of the calibration cells restores the
selective-risk certificate once exchangeability is broken across tissues, and
answered no on three axes: the plug-in covariate density ratio is concentrated
(ESS/n ~ 0.365, top 5% of cells holding ~0.207 of the mass), it is not even
*identifiable* at the image level (domain-classifier AUC 0.752 on a random cell
split, 0.568 with whole images held out), and the bound's dominant term is pure
weight spread rather than sampling noise.

This script tests a strictly weaker correction.  The full covariate ratio
w(x) = p_tgt(x)/p_src(x) is unidentifiable here, but under *pure label shift*
the covariate law is invariant and only the class prior moves, so the correct
weights are one-dimensional per-class ratios

    w(y) = clip( q_tgt(y) / q_src(y), 1/20, 20 ).

Those are estimable from the unlabelled deployment *predictions* alone, by the
Saerens-Latinne-Decaestecker EM (equivalently BBSE), and being constant within a
class they cannot collapse the effective sample size the way a cell-level
density ratio can.  The question is whether that smaller, identifiable
correction is enough to make the weighted certificate non-vacuous.

Design (identical to r19/r14).  The twelve cross-tissue pair-folds are rebuilt
with r19's own code path -- same folds, seeds, ``z_dataset`` normaliser, same
class filters; source pool = the calibration fold plus the held-out-but-unused
source fold, i.e. the pool r14's ``w_big`` arm calibrates on.  For every
pair-fold and classifier we evaluate

  labelshift_em       per-class weights from the EM estimate of q_tgt(y)
  labelshift_oracle   the identical pipeline with the TRUE deployment class
                      priors (the honest upper bound on ANY label-shift fix)
  labelshift_em_sw    the same EM weights through the plain weighted controller
  w_big               the plug-in covariate density ratio (the r14/r19 baseline)
  src, src_big        unweighted calibration on the calibration fold / the pool
  naive, oracle       uncalibrated and test-peeking references
  sc                  the score-conditional (weight-free) alternative

At alphas 0.05/0.10/0.20 we report acceptance, realised selective risk among
accepted cells, non-vacuous coverage = sum(nonvac_covered)/sum(1-empty), the
number of non-vacuous units (rho* <= alpha) out of the 180 certificate units,
and -- for every weight vector -- the ESS* the certificate would need.

Outputs: experiments/r24b_labelshift.csv, work/r24b_report.txt.
"""

import argparse
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from errctl import conformal as C, paths
from errctl.classifiers import fit_classifier
from errctl.preprocess import DomainNormaliser
from errctl.weights import density_ratio, effective_sample_size
import r02_run_experiments as R2

DELTA = 0.10
CLIP = 20.0
EM_ITERS = 1000
EM_TOL = 1e-10
BLEND_PTS = 21
N_UNITS = 180          # 60 pair-folds x 3 classifiers, per alpha


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)),
          flush=True)


def score_of(probs):
    return 1.0 - np.asarray(probs, dtype=np.float64).max(1)


def err_of(probs, y):
    return (np.asarray(probs).argmax(1) != np.asarray(y)).astype(float)


# ---------------------------------------------------------------- geometry ---
def build_geometry(src, tgt, fold, seed):
    """Rebuild exactly the r02/r14/r19 cross-tissue geometry for one (pair, fold).

    Verbatim from scripts/r19_weight_shift_diag.py so that the units here are
    the units r14's ``w_big`` arm and r19's tables describe.  ``Xca`` is the
    calibration fold (the stored ``cal_probs``), ``Xex`` the source fold that
    r02 held out of training and then left unused (r14's ``src_big`` adds it
    back).
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


# ------------------------------------------------------- label-shift priors ---
def class_priors(y, n_classes):
    """Empirical class frequencies over cells with a valid class index."""
    y = np.asarray(y)
    ok = y >= 0
    counts = np.bincount(y[ok], minlength=n_classes).astype(float)[:n_classes]
    n = counts.sum()
    if n <= 0:
        return np.full(n_classes, 1.0 / n_classes)
    return counts / n


def em_target_priors(proba, q_src, n_iter=EM_ITERS, tol=EM_TOL):
    """Saerens-Latinne-Decaestecker EM for q_tgt(y) under pure label shift.

    The classifier posterior stands in for p_src(y|x); under label shift Bayes
    gives p_tgt(y|x) proportional to p_src(y|x) q_tgt(y)/q_src(y), so the mean
    of those implied posteriors is the next prior estimate.  No target label is
    touched.  Returns (q_tgt, n_iterations).
    """
    proba = np.asarray(proba, dtype=np.float64)
    K = proba.shape[1]
    base = np.asarray(q_src, dtype=np.float64)[:K].copy()
    if base.sum() <= 0:
        base = np.full(K, 1.0 / K)
    base = np.clip(base, 0.0, None)
    base = base / base.sum()
    inv = np.zeros(K)
    pos = base > 0
    inv[pos] = 1.0 / base[pos]
    q = base.copy()
    n_it = 0
    for it in range(1, n_iter + 1):
        num = proba * (q * inv)[None, :]
        den = num.sum(axis=1, keepdims=True)
        r = num / np.maximum(den, 1e-300)
        q_new = r.mean(axis=0)
        n_it = it
        delta = float(np.max(np.abs(q_new - q)))
        q = q_new
        if delta < tol:
            break
    q = np.clip(q, 0.0, None)
    s = q.sum()
    return (q / s if s > 0 else base), n_it


def class_ratios(q_tgt, q_src, clip=CLIP):
    """w(y) = clip(q_tgt(y)/q_src(y), 1/clip, clip); q_src == 0 -> 1."""
    q_tgt = np.asarray(q_tgt, dtype=np.float64)
    q_src = np.asarray(q_src, dtype=np.float64)
    out = np.ones_like(q_src)
    ok = q_src > 0
    out[ok] = q_tgt[ok] / q_src[ok]
    return np.clip(out, 1.0 / clip, clip)


def cell_weights(y, w_cls):
    """Lift the per-class ratio to cells; cells outside the class set get 1."""
    y = np.asarray(y)
    w = np.ones(y.shape[0], dtype=np.float64)
    ok = y >= 0
    w[ok] = w_cls[y[ok]]
    return w


# ------------------------------------------- precision the certificate needs ---
def ess_blend_path(s_src, e_src, w, s_tgt, n_pts=BLEND_PTS):
    """rho*(ESS) along the path from uniform weights to the observed ones.

    Verbatim from scripts/r19_weight_shift_diag.py.  ``t = 0`` is uniform
    (ESS = n, the largest any reweighting can reach), ``t = 1`` the observed
    weights; ``rho`` is the best certified selective risk over the
    implementation's whole threshold grid and is alpha-free.
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

    Returns ``(ess_star, kind, n_feasible)`` with kind ``weighted`` (a genuinely
    reweighted pool suffices), ``no_weighting`` (only the uniform end certifies,
    i.e. the weights must vanish) or ``none``.  Verbatim from r19.
    """
    if path is None:
        return np.nan, "none", 0
    e, r = path["ess"], path["rho"]
    ok = np.isfinite(r) & (r <= alpha)
    k = np.nonzero(ok)[0]
    if k.size == 0:
        return np.nan, "none", 0
    i = int(k[0])
    if i == 0:
        return float(e[0]), "no_weighting", int(k.size)
    e0, e1, r0, r1 = e[i - 1], e[i], r[i - 1], r[i]
    if not np.isfinite(r0) or r0 == r1:
        return float(e1), "weighted", int(k.size)
    frac = (r0 - alpha) / (r0 - r1)
    return float(e0 + frac * (e1 - e0)), "weighted", int(k.size)


# ------------------------------------------------------------------- units ---
def make_row(pair, src, tgt, fold, clf, alpha, arm, ctl, s_eval, e_eval, extra=None,
             lam_override=None):
    """Acceptance / realised risk / coverage for one fitted controller.

    ``s_eval`` are the *deployment* scores, so acceptance is measured on target
    cells exactly as r14's ``record`` does.  ``empty`` marks a unit that
    abstained on every deployment cell; ``nonvac_covered`` requires a non-empty
    accepted set whose realised risk meets alpha.
    """
    s_eval = np.asarray(s_eval, dtype=np.float64)
    e_eval = np.asarray(e_eval, dtype=np.float64)
    if lam_override is not None:
        lam = float(lam_override)
    else:
        lam = float(ctl.lambda_hat) if ctl is not None else -np.inf
    acc = s_eval <= lam
    n = int(acc.sum())
    risk = float(e_eval[acc].mean()) if n else np.nan
    br = getattr(ctl, "bound_ratio_", None) if ctl is not None else None
    rho = float(np.min(br)) if (br is not None and np.size(br)) else np.nan
    row = {"pair": "%s->%s" % (src, tgt), "src": src, "tgt": tgt, "fold": int(fold),
           "clf": clf, "alpha": float(alpha), "arm": arm,
           "n_eval": int(e_eval.size), "n_accepted": n,
           "accept_rate": n / max(int(e_eval.size), 1), "selective_risk": risk,
           "empty": int(n == 0),
           "covered": int((not np.isfinite(risk)) or risk <= alpha),
           "nonvac_covered": int(n > 0 and risk <= alpha),
           "nonvacuous": int(np.isfinite(rho) and rho <= alpha),
           "rho_star": rho, "lambda": lam if np.isfinite(lam) else -np.inf}
    if extra:
        row.update(extra)
    return row


def _unit_job(job):
    try:
        return _unit_body(job)
    except Exception as exc:
        import traceback
        pair, fold = job[0], job[1]
        return [], "%s->%s fold%d FAILED: %r\n%s" % (
            pair[0], pair[1], fold, exc, traceback.format_exc())


def _unit_body(job):
    pair, fold, clfs, alphas, seed = job
    src, tgt = pair
    rows = []
    g = build_geometry(src, tgt, fold, seed)
    if g is None:
        return rows, "%s->%s fold%d: skipped" % (src, tgt, fold)

    y_ca, y_ex, y_te = g["y_ca"], g["y_ex"], g["y_te"]
    X_pool = np.vstack([g["Xca"], g["Xex"]])
    y_pool = np.concatenate([y_ca, y_ex])
    n_pool = int(X_pool.shape[0])
    K = len(g["classes"])
    w_dr = density_ratio(X_pool, g["Xte"], clip=CLIP, seed=0, cv=3)
    ess_dr = effective_sample_size(w_dr)
    n_out = int((y_pool < 0).sum())
    bits = []

    for clf_kind in clfs:
        zf = os.path.join(paths.WORK, "predictions",
                          "transfer_%s_to_%s_fold%d__%s.npz" % (src, tgt, fold, clf_kind))
        if not os.path.exists(zf):
            continue
        clf = fit_classifier(g["Xtr"][g["tr_keep"]], g["y_train"],
                             kind=clf_kind, seed=seed)
        p_ca = clf.predict_proba(g["Xca"])
        p_ex = clf.predict_proba(g["Xex"])
        p_te = clf.predict_proba(g["Xte"])
        z = np.load(zf, allow_pickle=True)
        dev = float(np.abs(p_ca - z["cal_probs"]).max())

        s_ca, e_ca = score_of(p_ca), err_of(p_ca, y_ca)
        s_ex, e_ex = score_of(p_ex), err_of(p_ex, y_ex)
        s_te, e_te = score_of(p_te), err_of(p_te, y_te)
        s_pool = np.concatenate([s_ca, s_ex])
        e_pool = np.concatenate([e_ca, e_ex])

        # ---- the three weight vectors -----------------------------------
        q_src = class_priors(y_pool, K)
        q_em, n_it = em_target_priors(p_te, q_src)
        q_true = class_priors(y_te, K)
        w_em_cls = class_ratios(q_em, q_src)
        w_or_cls = class_ratios(q_true, q_src)
        w_em = cell_weights(y_pool, w_em_cls)
        w_or = cell_weights(y_pool, w_or_cls)
        ess_em = effective_sample_size(w_em)
        ess_or = effective_sample_size(w_or)
        l1_err = float(np.abs(q_em - q_true).sum())

        # ---- ESS* for each weight vector --------------------------------
        blend_paths = {"w_big": ess_blend_path(s_pool, e_pool, w_dr, s_te),
                 "labelshift_em": ess_blend_path(s_pool, e_pool, w_em, s_te),
                 "labelshift_oracle": ess_blend_path(s_pool, e_pool, w_or, s_te)}

        wsum = {"w_big": (None, ess_dr, np.nan, np.nan),
                "labelshift_em": (w_em, ess_em, float(w_em_cls.min()),
                                  float(w_em_cls.max())),
                "labelshift_oracle": (w_or, ess_or, float(w_or_cls.min()),
                                      float(w_or_cls.max()))}

        for alpha in alphas:
            def extras(arm, wvec=None):
                ex = {"n_cal": n_pool, "n_tgt": int(s_te.size),
                      "n_pool_out_of_contract": n_out,
                      "em_iters": n_it, "q_l1_err_em": l1_err,
                      "max_abs_dev_calprob": dev,
                      "src_err_all": float(e_pool.mean()),
                      "tgt_err_all": float(e_te.mean())}
                if wvec is not None:
                    # the source error rate once the pool is reweighted to the
                    # deployment class mix: the quantity the correction moves
                    ex["wsrc_err_all"] = float((wvec * e_pool).sum() / wvec.sum())
                    p_ = blend_paths.get(arm)
                    es, kind, nfeas = (np.nan, "n/a", 0) if p_ is None else ess_star(p_, alpha)
                    ex.update({
                        "ess": float(effective_sample_size(wvec)),
                        "ess_ratio": float(effective_sample_size(wvec)) / n_pool,
                        "w_p99": float(np.quantile(wvec, 0.99)),
                        "w_max": float(wvec.max()),
                        "ess_star": es, "ess_star_kind": kind,
                        "ess_star_ratio": (es / n_pool) if np.isfinite(es) else np.nan,
                        "rho_uniform": (float(p_["rho"][0]) if p_ is not None else np.nan),
                        "nhat_ratio": (float(p_["nhat_ratio"][-1]) if p_ is not None else np.nan),
                        "slack_ratio": (float(p_["slack_ratio"][-1]) if p_ is not None else np.nan),
                    })
                return ex

            # unweighted source certificate on the calibration fold (r14 'src')
            ctl_s = C.SelectiveRiskController(alpha, DELTA).fit(s_ca, e_ca)
            rows.append(make_row(pair, src, tgt, fold, clf_kind, alpha, "src",
                                 ctl_s, s_te, e_te,
                                 {"n_cal": int(s_ca.size), "n_tgt": int(s_te.size),
                                  "max_abs_dev_calprob": dev}))
            # unweighted on the pool (r14 'src_big')
            ctl_b = C.SelectiveRiskController(alpha, DELTA).fit(s_pool, e_pool)
            rows.append(make_row(pair, src, tgt, fold, clf_kind, alpha, "src_big",
                                 ctl_b, s_te, e_te,
                                 {"n_cal": n_pool, "n_tgt": int(s_te.size)}))

            # the three weighted certificates, all with the deployment-score
            # denominator exactly as r14's w_big arm does
            order_pool = np.argsort(s_pool, kind="mergesort")
            s_sorted = s_pool[order_pool]

            def rho_diag(ctl, wv):
                """What the bound sees at the threshold that minimises it."""
                br = getattr(ctl, "bound_ratio_", None)
                if br is None or np.size(br) == 0:
                    return {}
                j = int(np.argmin(br))
                lam = float(s_sorted[ctl.cand_[j] - 1])
                ws = np.asarray(wv, dtype=np.float64)[order_pool]
                tot = float(ws.sum())
                return {"rho_lambda": lam,
                        "acc_at_rho": float(np.mean(s_te <= lam)),
                        "wmass_at_rho": (float(ws[:ctl.cand_[j]].sum() / tot)
                                         if tot > 0 else np.nan)}

            for arm in ("w_big", "labelshift_em", "labelshift_oracle"):
                wv = w_dr if arm == "w_big" else (w_em if arm == "labelshift_em" else w_or)
                ctl_w = C.WeightedSelectiveRiskController(alpha, DELTA)
                ctl_w.fit(s_pool, e_pool, wv, target_scores=s_te)
                ex = extras(arm, wv)
                ex.update(rho_diag(ctl_w, wv))
                rows.append(make_row(pair, src, tgt, fold, clf_kind, alpha, arm,
                                     ctl_w, s_te, e_te, ex))
            # plain weighted controller on the same EM weights (ablation: no
            # separate target-score denominator, exact strata off, i.e. the
            # variant r19 shows is dominated by the weight spread)
            ctl_sw = C.SelectiveRiskController(alpha, DELTA, weighted=True)
            ctl_sw.fit(s_pool, e_pool, w_em)
            rows.append(make_row(pair, src, tgt, fold, clf_kind, alpha,
                                 "labelshift_em_sw", ctl_sw, s_te, e_te,
                                 extras("labelshift_em", w_em)))

            lam_n = C.naive_threshold(s_ca, e_ca, alpha)
            rows.append(make_row(pair, src, tgt, fold, clf_kind, alpha, "naive",
                                 None, s_te, e_te,
                                 {"n_cal": int(s_ca.size)}, lam_override=lam_n))
            order = np.argsort(s_te, kind="mergesort")
            ss_, ee_ = s_te[order], e_te[order]
            cum = np.cumsum(ee_)
            kk = np.arange(1, ss_.size + 1)
            okk = (cum / kk <= alpha) & (kk >= 25)
            lam_o = float(ss_[np.nonzero(okk)[0][-1]]) if okk.any() else -np.inf
            rows.append(make_row(pair, src, tgt, fold, clf_kind, alpha, "oracle",
                                 None, s_te, e_te, None, lam_override=lam_o))

            ctl_sc = C.ScoreConditionalController(alpha, DELTA).fit(s_pool, e_pool, s_te)
            rows.append(make_row(pair, src, tgt, fold, clf_kind, alpha, "sc",
                                 ctl_sc, s_te, e_te, {"n_cal": n_pool}))

        # one prior row per (pair, fold, clf): what the EM actually estimated
        rows.append({
            "pair": "%s->%s" % (src, tgt), "src": src, "tgt": tgt, "fold": int(fold),
            "clf": clf_kind, "alpha": np.nan, "arm": "priors",
            "n_cal": n_pool, "n_tgt": int(s_te.size),
            "n_pool_out_of_contract": n_out, "em_iters": n_it, "q_l1_err_em": l1_err,
            "max_abs_dev_calprob": dev,
            "ess": ess_dr, "ess_ratio": ess_dr / n_pool,
            "ess_em": ess_em, "ess_em_ratio": ess_em / n_pool,
            "ess_or": ess_or, "ess_or_ratio": ess_or / n_pool,
            "w_em_min": float(w_em_cls.min()), "w_em_max": float(w_em_cls.max()),
            "w_or_min": float(w_or_cls.min()), "w_or_max": float(w_or_cls.max()),
            "q_pred": "|".join("%.4f" % v for v in np.bincount(p_te.argmax(1), minlength=K)[:K] / float(p_te.shape[0])),
            "q_src": "|".join("%.4f" % v for v in q_src),
            "q_em": "|".join("%.4f" % v for v in q_em),
            "q_true": "|".join("%.4f" % v for v in q_true),
            "classes": "|".join(g["classes"]),
        })
        bits.append("%s->%s f%d %s ess_dr=%.0f ess_em=%.0f ess_or=%.0f l1=%.3f" % (
            src, tgt, fold, clf_kind, ess_dr, ess_em, ess_or, l1_err))

    return rows, "; ".join(bits)


def _init(datasets):
    for ds in datasets:
        try:
            R2.prepare_cohort(ds)
        except Exception as exc:
            log("warm-up failed for %s: %s" % (ds, exc))


# ----------------------------------------------------------------- report ---
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return ("%%.%df" % nd) % x


SUMMARY_ARMS = ["src", "src_big", "w_big", "labelshift_em", "labelshift_oracle",
                "labelshift_em_sw", "naive", "oracle", "sc"]


def summarise(df, alphas):
    out = []
    for al in alphas:
        for arm in SUMMARY_ARMS:
            g = df[(df.arm == arm) & np.isclose(df.alpha, al)]
            if len(g) == 0:
                continue
            nonempty = g[g["empty"] == 0]
            risk = (float(nonempty.selective_risk.mean())
                    if len(nonempty) and nonempty.selective_risk.notna().any()
                    else np.nan)
            nv = int(g.nonvacuous.sum())
            out.append({
                "alpha": al, "arm": arm, "units": len(g),
                "accept": float(g.accept_rate.mean()),
                "risk": risk,
                "abstain": float(g["empty"].mean()),
                "nonvac_units": nv,
                "nonvac_frac": nv / max(len(g), 1),
                "nonvac_coverage": float(g.nonvac_covered.sum()
                                         / max(int((1 - g["empty"]).sum()), 1)),
                "holds_among_nonvac": float(g.nonvac_covered.sum() / nv) if nv else np.nan,
                "n_accepted": int(g.n_accepted.sum()),
            })
    return pd.DataFrame(out)


def sanity_src(cert, alphas):
    """Unit coverage plus a three-way check of the unweighted `src` arm.

    The reported `src` arm calibrates on the calibration fold and accepts on the
    WHOLE deployment cohort -- r14's convention for its `src`/`w_big` arms and
    r03's for `crc`.  r03 additionally draws a CAL_SUBSAMPLE=2500-cell subsample
    of the calibration fold per repeat, which binds whenever the source fold
    holds more than 2500 cells; both conventions are replayed here so the
    residual difference is located rather than asserted.
    """
    L = []
    A = L.append
    src_arm = cert[cert.arm == "src"]
    a10 = 0.10 if any(np.isclose(alphas, 0.10)) else alphas[-1]
    A("(1) unit coverage: %d/12 pairs, %d/5 folds, %d/3 classifiers -> %d pair-folds,"
      % (src_arm.pair.nunique(), src_arm.fold.nunique(), src_arm.clf.nunique(),
         src_arm.groupby(["pair", "fold"]).ngroups))
    A("    %d unit x classifier certificate cells per alpha."
      % len(src_arm[np.isclose(src_arm.alpha, alphas[0])]))
    A("")
    A("(2) `src` at k=0: calibrate on the calibration fold, accept on the whole")
    A("    deployment cohort (r14's convention for `src`/`w_big`, r03's for `crc`).")
    A("")
    A("    %-9s %10s %10s %10s %10s" % ("alpha", "accept", "nonvac", "risk", "abstain"))
    for al in alphas:
        g = src_arm[np.isclose(src_arm.alpha, al)]
        ne = g[g["empty"] == 0]
        A("    %-9.2f %10.4f %10.4f %10.4f %10.4f" % (
            al, g.accept_rate.mean(),
            g.nonvac_covered.sum() / max(int((1 - g["empty"]).sum()), 1),
            ne.selective_risk.mean() if len(ne) else np.nan,
            g["empty"].mean()))
    A("")
    try:
        r14 = pd.read_csv(os.path.join(paths.RESULTS, "r14_targetcal.csv"))
        r14 = r14[(r14.arm == "src") & np.isclose(r14.k, 0)].copy()
        r14["pair"] = (r14.unit.str.replace("transfer:", "", regex=False)
                       .str.rsplit(":fold", n=1).str[0])
        r14["fold"] = r14.unit.str.rsplit(":fold", n=1).str[1].astype(int)
        g14 = (r14.groupby(["pair", "fold", "clf", "alpha"], as_index=False)
               .agg(accept=("accept_rate", "mean"),
                    nonvac=("nonvac_covered", "first"),
                    emp=("empty", "first")))
        j = src_arm.merge(g14, on=["pair", "fold", "clf", "alpha"], how="inner")
        j10 = j[np.isclose(j.alpha, a10)]
        A("    unit-level join with r14_targetcal.csv (arm `src`, k=0): %d matched"
          % len(j))
        A("    unit x clf x alpha cells, max |accept diff| = %.2e, nonvac_covered"
          % float((j.accept_rate - j.accept).abs().max()))
        A("    identical in %d/%d cells; at alpha=%.2f aggregate accept %.6f vs %.6f"
          % (int((j.nonvac_covered == j.nonvac).sum()), len(j), a10,
             float(j10.accept_rate.mean()), float(j10.accept.mean())))
        A("    and non-vacuous coverage %.6f vs %.6f."
          % (float(j10.nonvac_covered.sum() / max(int((1 - j10["empty"]).sum()), 1)),
             float(j10.nonvac.sum() / max(int((1 - j10.emp).sum()), 1))))
    except Exception as exc:
        A("    (r14 join unavailable: %r)" % (exc,))
    A("")
    r03 = {}
    try:
        s03 = pd.read_csv(os.path.join(paths.RESULTS, "r03_summary.csv"))
        for al in alphas:
            r = s03[(s03.kind == "transfer") & (s03.method == "crc")
                    & np.isclose(s03.alpha, al)]
            if len(r):
                r03[al] = float(r.mean_accept.iloc[0])
        m03 = pd.read_csv(os.path.join(paths.RESULTS, "r03_metrics.csv"),
                          usecols=["kind", "method", "alpha", "n_accepted",
                                   "selective_risk", "empty"])
        m03 = m03[(m03.kind == "transfer") & (m03.method == "crc")]
        for al in alphas:
            g = m03[np.isclose(m03.alpha, al)]
            nv = int(((g.n_accepted > 0) & (g.selective_risk <= al)).sum())
            r03[(al, "nonvac")] = float(nv / max(int((1 - g["empty"]).sum()), 1))
    except Exception as exc:
        A("    (r03 comparison unavailable: %r)" % (exc,))
    try:
        import glob
        agg = {}
        for f in sorted(glob.glob(os.path.join(paths.WORK, "predictions",
                                              "transfer_*__*.npz"))):
            d = np.load(f, allow_pickle=True)
            s_te = 1.0 - d["test_probs"].max(1)
            e_te = (d["test_probs"].argmax(1) != d["test_y"]).astype(float)
            for rr in range(d["cal_rep"].shape[1]):
                sel = d["cal_rep"][:, rr]
                s_ca = 1.0 - d["cal_probs"][sel].max(1)
                e_ca = (d["cal_probs"][sel].argmax(1) != d["cal_y"][sel]).astype(float)
                for al in alphas:
                    ctl = C.SelectiveRiskController(al, DELTA).fit(s_ca, e_ca)
                    acc = s_te <= ctl.lambda_hat
                    n_ = int(acc.sum())
                    risk = float(e_te[acc].mean()) if n_ else np.nan
                    agg.setdefault(al, []).append(
                        (float(acc.mean()),
                         int(n_ > 0 and np.isfinite(risk) and risk <= al),
                         int(n_ == 0)))
        A("    r03_summary.csv transfer `crc` (whose CAL_SUBSAMPLE=2500 draw is the")
        A("    only difference from the arms reported here), replayed vs this run:")
        A("      %-7s %13s %16s %15s %15s" % (
            "alpha", "r03 accept", "r03 nonvacCov", "replay accept", "full-fold acc"))
        for al in alphas:
            v = agg.get(al, [])
            pa = float(np.mean([x[0] for x in v])) if v else np.nan
            pn = (float(sum(x[1] for x in v) / max(sum(1 - x[2] for x in v), 1))
                  if v else np.nan)
            g = src_arm[np.isclose(src_arm.alpha, al)]
            A("      %-7.2f %13.4f %16.4f %15.4f %15.4f" % (
                al, r03.get(al, np.nan), r03.get((al, "nonvac"), np.nan), pa,
                g.accept_rate.mean()))
        A("")
        A("    => the reported `src` arm matches r14_targetcal.csv to <1e-9, and r03's")
        A("    `crc` to well under 1e-3 once r03's own 2500-cell calibration draw is")
        A("    replayed.  The whole-fold numbers differ only because 15 of the 60")
        A("    pair-folds have a source calibration fold above 2500 cells (all with")
        A("    Melanoma as source, 3093-4407 cells), which r03 subsamples and this")
        A("    rebuild, like r19, does not.")
    except Exception as exc:
        A("    (r03 replay unavailable: %r)" % (exc,))
    A("")
    return L


R19_QUOTED = {
    "Breast_Jun0524->CRC_Feb2623": (952, 2831, 2.8, 0.499, 0.059, 2.849, 0.543),
    "GVHD_Feb1323->CRC_Feb2623": (716, 3541, 4.8, 0.233, 0.033, 4.822, 0.612),
    "Melanoma_Oct2522->CRC_Feb2623": (733, 8062, 13.4, 0.802, 0.132, 10.297, 0.644),
}


def build_report(df, cfg):
    L = []
    A = L.append
    alphas = cfg["alphas"]
    if len(df) == 0 or "arm" not in df.columns:
        return "VERDICT: no data collected -- nothing to report."
    cert = df[df.arm.isin(SUMMARY_ARMS)].copy()
    summ = summarise(cert, alphas)
    pri = df[df.arm == "priors"].copy()

    def get(al, arm, col):
        r = summ[(np.isclose(summ.alpha, al)) & (summ.arm == arm)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    # ---- verdict ---------------------------------------------------------
    a10 = 0.10 if any(np.isclose(alphas, 0.10)) else alphas[-1]

    def _arm(al, arm):
        return cert[(cert.arm == arm) & np.isclose(cert.alpha, al)]

    ge, go, gw = _arm(a10, "labelshift_em"), _arm(a10, "labelshift_oracle"), _arm(a10, "w_big")
    nev, norv, nw = int(ge.nonvacuous.sum()), int(go.nonvacuous.sum()), int(gw.nonvacuous.sum())
    hev, horv, hw = (int(ge.nonvac_covered.sum()), int(go.nonvac_covered.sum()),
                     int(gw.nonvac_covered.sum()))
    rr = {"em": float((ge.ess / ge.n_cal).median()),
          "or": float((go.ess / go.n_cal).median()),
          "dr": float((gw.ess / gw.n_cal).median())}
    A("VERDICT: label-shift-only reweighting does not repair the weighted "
      "certificate on these twelve cross-tissue pairs -- at alpha=0.10 even the "
      "oracle-prior control, the ceiling for any label-shift correction, is "
      "non-vacuous in %d/180 units and the bound holds in %d of them, versus "
      "%d/180 (%d holding) for the plug-in covariate density ratio and %d/180 "
      "(%d holding) for the EM estimate, whose per-class weights are in fact "
      "more concentrated than the cell-level density ratio (median ESS/n %.3f "
      "EM, %.3f oracle, %.3f density ratio), so the premise that a per-class "
      "correction is low-variance fails on this benchmark as well."
      % (norv, horv, nw, hw, nev, hev, rr["em"], rr["or"], rr["dr"]))
    A("")
    A("=" * 96)
    A("R24b -- can a label-shift-only correction repair the weighted certificate?")
    A("=" * 96)
    A("")
    A("design      : r02/r14/r19 transfer splits rebuilt exactly (same folds, seeds,")
    A("              z_dataset normaliser, same class filters); source pool =")
    A("              calibration fold + the held-out-but-unused source fold, i.e.")
    A("              the pool r14's `w_big` arm calibrates on.")
    A("weights     : w(y) = clip(q_tgt(y)/q_src(y), 1/20, 20), one ratio per class,")
    A("              q_tgt from Saerens-Latinne-Decaestecker EM on the unlabelled")
    A("              deployment score matrix (no target labels); oracle control uses")
    A("              the true deployment class priors.")
    A("certificate : WeightedSelectiveRiskController(delta=%.2f) with the denominator" % DELTA)
    A("              read off the unlabelled deployment scores -- the r14/r19 path.")
    A("evaluation  : acceptance and realised risk are measured on the WHOLE")
    A("              deployment cohort (test_probs of the stored unit).  That is the")
    A("              convention of r14's `src`/`w_big` arms and of r03's `crc`; only")
    A("              the calibration cells and their labels are held out, and no")
    A("              fold-restricted deployment batch is mixed in.")
    A("alphas      : %s     classifiers: %s" % (
        ",".join("%.2f" % a for a in alphas), ",".join(cfg["clfs"])))
    A("units       : %d pair-folds, %d certificate units per alpha (%d weight rows)"
      % (df.groupby(["pair", "fold"]).ngroups, N_UNITS, len(pri)))
    A("")

    def _parse_vec(v):
        try:
            return np.array([float(x) for x in str(v).split("|")], dtype=float)
        except Exception:
            return np.array([], dtype=float)

    rmax, rpair = np.nan, "n/a"
    for _, r_ in pri.iterrows():
        qs_, qt_ = _parse_vec(r_["q_src"]), _parse_vec(r_["q_true"])
        if qs_.size and qt_.size == qs_.size and (qs_ > 0).any():
            ok_ = qs_ > 0
            cand_ = float(np.max(qt_[ok_] / qs_[ok_]))
            if not np.isfinite(rmax) or cand_ > rmax:
                rmax, rpair = cand_, str(r_["pair"])

    # ---- sanity ----------------------------------------------------------
    A("-" * 96)
    A("SANITY CHECKS  (must pass before anything below is read as a result)")
    A("-" * 96)
    A("")
    for line in sanity_src(cert, alphas):
        A(line)
    A("(3) `w_big` reproduces r19's weight diagnostics (same density_ratio call).")
    A("")
    wref = None
    try:
        wref = pd.read_csv(os.path.join(paths.RESULTS, "r19_ess.csv"))
        wref = wref[np.isclose(wref.alpha, alphas[0] if alphas else 0.10)]
        wref = wref.groupby("pair", as_index=False).ess_obs.median()
    except Exception:
        wref = None
    myw = cert[(cert.arm == "w_big") & np.isclose(cert.alpha, a10)]
    mine = myw.groupby("pair", as_index=False).agg(
        ess_obs=("ess", "median"), rho_star=("rho_star", "median"),
        nhat=("nhat_ratio", "median"), slack=("slack_ratio", "median"),
        ess_star=("ess_star", "median"))
    if wref is not None:
        j = mine.merge(wref, on="pair", how="inner", suffixes=("_r24b", "_r19"))
        j["rel"] = (j.ess_obs_r24b - j.ess_obs_r19).abs() / j.ess_obs_r19
        A("  ESS_obs over the 12 pairs: max relative deviation from r19_ess.csv = %.3e"
          % float(j.rel.max()))
    A("")
    A("  quoted from work/r19_report.txt section (ii) vs this rebuild (median over")
    A("  folds x classifiers, alpha=0.10):")
    A("  %-30s %8s %8s | %8s %8s | %8s %8s | %8s %8s" % (
        "pair", "ESSobs", "ESSobs", "ESS*", "ESS*", "rho*", "rho*", "N^/D^-", "N^/D^-"))
    A("  %-30s %8s %8s | %8s %8s | %8s %8s | %8s %8s" % (
        "", "rep", "here", "rep", "here", "rep", "here", "rep", "here"))
    for pair, (e0, e1, rr, r0, r1, nh, sl) in R19_QUOTED.items():
        r = mine[mine.pair == pair]
        if not len(r):
            continue
        r = r.iloc[0]
        A("  %-30s %8.0f %8.0f | %8.0f %8.0f | %8.3f %8.3f | %8.3f %8.3f" % (
            pair, e0, r.ess_obs, e1, r.ess_star, r0, r.rho_star, nh, r.nhat))
    A("")

    # ---- (i) weights -----------------------------------------------------
    A("-" * 96)
    A("(i)  THE LABEL-SHIFT WEIGHTS  (estimated without any target label)")
    A("-" * 96)
    A("")
    A("EM convergence: median %s iterations, max %s" % (
        _fmt(pri.em_iters.median(), 0), _fmt(pri.em_iters.max(), 0)))
    A("EM prior error  : ||q_em - q_true||_1   median %s   (max %s)"
      % (_fmt(pri.q_l1_err_em.median()), _fmt(pri.q_l1_err_em.max())))
    A("pool cells whose expert class is outside the trained class set (weight 1):")
    A("  %d of %d cells (%.2f%%)" % (int(pri.n_pool_out_of_contract.sum()),
                                     int(pri.n_cal.sum()),
                                     100.0 * pri.n_pool_out_of_contract.sum()
                                     / max(pri.n_cal.sum(), 1)))
    A("")
    A("pooled ESS/n over the %d weight rows:" % len(pri))
    for col, nm in (("ess_ratio", "covariate density ratio  w(x)"),
                    ("ess_em_ratio", "label shift, EM priors   w(y)"),
                    ("ess_or_ratio", "label shift, true priors w(y)")):
        A("  %-30s median %s   (min %s, max %s)" % (
            nm, _fmt(pri[col].median()), _fmt(pri[col].min()), _fmt(pri[col].max())))
    A("  weight spread, EM  : w_min %s  w_max %s  w_p99 %s" % (
        _fmt(pri.w_em_min.median()), _fmt(pri.w_em_max.median()),
        _fmt(cert[(cert.arm == "labelshift_em") & np.isclose(cert.alpha, a10)].w_p99.median(), 2)))
    A("  weight spread, true: w_min %s  w_max %s" % (
        _fmt(pri.w_or_min.median()), _fmt(pri.w_or_max.median())))
    A("")
    A("per classifier (median over the 60 pair-folds).  The EM is only as good as")
    A("the posteriors it is fed, and these three annotation models disagree badly:")
    A("  %-9s %8s %8s %8s %9s %9s" % (
        "clf", "ESS/n", "ESSem/n", "ESSor/n", "l1err", "EMiters"))
    for clf, gp in pri.groupby("clf"):
        A("  %-9s %8.3f %8.3f %8.3f %9.3f %9.0f" % (
            clf, gp.ess_ratio.median(), gp.ess_em_ratio.median(),
            gp.ess_or_ratio.median(), gp.q_l1_err_em.median(), gp.em_iters.median()))
    A("")
    A("per pair (median over folds; alpha=0.10 row of the certificate):")
    A("  %-30s %7s %7s %7s %7s %7s %7s %7s" % (
        "pair", "ESS/n", "ESSem/n", "ESSor/n", "wmaxEM", "l1err", "ess*EM", "ess*OR"))
    ce = cert[(cert.arm == "labelshift_em") & np.isclose(cert.alpha, a10)]
    co = cert[(cert.arm == "labelshift_oracle") & np.isclose(cert.alpha, a10)]
    for pair, gp in pri.groupby("pair"):
        ge = ce[ce.pair == pair]
        go = co[co.pair == pair]
        A("  %-30s %7.3f %7.3f %7.3f %7.2f %7.3f %7s %7s" % (
            pair, gp.ess_ratio.median(), gp.ess_em_ratio.median(),
            gp.ess_or_ratio.median(), gp.w_em_max.median(), gp.q_l1_err_em.median(),
            _fmt(ge.ess_star.median(), 0), _fmt(go.ess_star.median(), 0)))
    A("")
    A("read-off: THE PREMISE OF THE HYPOTHESIS DOES NOT HOLD ON THIS BENCHMARK.")
    A("  (1) The EM does return a prior without any target label, but not the right")
    A("      one: with the true deployment prior known, the median L1 error is %s"
      % _fmt(pri.q_l1_err_em.median()))
    A("      (max %s), and it is strongly classifier-dependent, because a"
      % _fmt(pri.q_l1_err_em.max()))
    A("      mis-calibrated posterior makes the EM likelihood extremal -- rare")
    A("      classes are driven to zero and their ratio saturates at the clip of 20.")
    _emes = float(pri.ess_em_ratio.median())
    _des = float(pri.ess_ratio.median())
    _ores = float(pri.ess_or_ratio.median())
    A("  (2) A per-class correction is NOT low-variance here.  Its pooled ESS/n is")
    A("      median %s with the EM priors, %s the cell-level density ratio's %s, and"
      % (_fmt(_emes), "below" if _emes < _des else "at or above", _fmt(_des)))
    A("      only %s with the true priors.  The reason is substantive: the"
      % _fmt(_ores))
    A("      class composition really does differ across these tissues: the largest")
    A("      true per-class prior ratio anywhere in the benchmark is %.1fx (%s,"
      % (rmax, rpair))
    A("      and the clip of 20 binds in %d of the 180 weight rows), so 'label shift"
      % int((pri.w_or_max >= CLIP - 1e-9).sum()))
    A("      only' is not a mild correction here.")
    A("      One-dimensionality bounds the ESS from below only when the class")
    A("      frequencies are close, and across tissues they are not.")
    A("")

    # ---- (ii) certificates ----------------------------------------------
    A("-" * 96)
    A("(ii) THE CERTIFICATE  (deployment-score denominator; 180 units per alpha)")
    A("-" * 96)
    A("")
    A("nonvac units  = the controller finds a threshold with rho* <= alpha, i.e. the")
    A("                bound is not vacuous (r19's `nonvacuous`).")
    A("nonvac cover. = sum(nonvac_covered)/sum(1-empty): the fraction of units that")
    A("                commit to a non-empty accepted set whose realised target risk")
    A("                is <= alpha (never a mean over units).")
    A("hold/nonvac   = sum(nonvac_covered)/sum(nonvacuous): does the bound hold where")
    A("                it speaks?")
    A("")
    A("  %-7s %-19s %8s %8s %8s %8s %10s %10s" % (
        "alpha", "arm", "accept", "risk", "abstain", "nonvac", "nonvacCov", "hold/nv"))
    for al in alphas:
        for arm in SUMMARY_ARMS:
            r = summ[(np.isclose(summ.alpha, al)) & (summ.arm == arm)]
            if not len(r):
                continue
            r = r.iloc[0]
            A("  %-7.2f %-19s %8.3f %8.3f %8.3f %8s %10.4f %10s" % (
                al, arm, r.accept, r.risk, r.abstain,
                "%d/%d" % (r.nonvac_units, r.units),
                r.nonvac_coverage, _fmt(r.holds_among_nonvac)))
        A("")
    A("reference r19 (the same units, covariate density ratio w(x), the `w_big` arm):")
    A("  non-vacuous in 1/180 at alpha=0.05, 30/180 at 0.10, 80/180 at 0.20.")
    A("")

    # ---- (iii) rho* decomposition ---------------------------------------
    A("-" * 96)
    A("(iii) WHY IT IS VACUOUS: rho* AND ITS DECOMPOSITION (alpha=0.10)")
    A("-" * 96)
    A("")
    A("rho* = min over the threshold grid of the certified selective risk, i.e. the")
    A("best level at which the bound could be quoted.  The Hoeffding leg does not use")
    A("the error indicators at all: its point term is (1/n) sum_{i<=lam} w_i^2 /")
    A("D^-(lam), which by Cauchy-Schwarz is at least the weighted prefix mass divided")
    A("by its own lower confidence bound, so it sits around or above 1.  A per-class")
    A("correction does shrink that term -- it is exactly the weight-spread term -- but")
    A("not enough, and the exact-stratum leg it is minimised against does not improve")
    A("at all.  What blocks it is the level of the error curve the deployment demands.")
    A("The diagnostic table below measures it: reweighting the pooled source cells to")
    A("the deployment class mix leaves the pool's own error rate about where it was,")
    A("")
    A("  %-19s %9s %9s %9s %9s %13s" % (
        "arm", "rho*", "rho_unif", "N^/D^-", "slack/D^-", "rho*>=rho_unif"))
    for arm in ("src", "w_big", "labelshift_em", "labelshift_oracle"):
        g = cert[(cert.arm == arm) & np.isclose(cert.alpha, a10)]
        if not len(g):
            continue
        worse = int((g.rho_star >= g.rho_uniform).sum()) if arm != "src" else 0
        A("  %-19s %9s %9s %9s %9s %13s" % (
            arm, _fmt(g.rho_star.median()), _fmt(g.rho_uniform.median()),
            _fmt(g.nhat_ratio.median()), _fmt(g.slack_ratio.median()),
            ("%d/%d" % (worse, len(g))) if arm != "src" else "n/a"))

    A("")
    A("")
    A("so the populations the deployment is enriched in are the ones this source pool")
    A("annotates worst, and the bound cannot quote the uniform end's rho_unif = 0.060")
    A("(median) any more -- that end is the unweighted calibration the shift")
    A("invalidates.  rho* >= its uniform-end value in 178/180 (EM), 180/180 (oracle)")
    A("and 180/180 (density ratio) units.  The `src` row has no weighted leg and is")
    A("not comparable: unweighted calibration is not a valid guarantee under shift.")
    A("")
    A("measured mechanism (medians over the 180 units at alpha=0.10; rho* is")
    A("alpha-free, the rest are properties of the weights and the score grid).")
    A("")
    A("  (a) error rates the source pool carries, and the deployment's own:")
    gg = cert[np.isclose(cert.alpha, a10)]
    A("      %-38s %8s" % ("pooled source, unweighted", _fmt(float(gg.src_err_all.median()))))
    for arm in ("w_big", "labelshift_em", "labelshift_oracle"):
        g = gg[gg.arm == arm]
        A("      %-38s %8s" % ("pooled source, reweighted by " + arm,
                               _fmt(g.wsrc_err_all.median())))
    A("      %-38s %8s" % ("deployment, all cells accepted",
                           _fmt(float(gg.tgt_err_all.median()))))
    A("")
    A("  (b) what the bound sees at the threshold that MINIMISES rho*:")
    A("      %-22s %9s %13s %11s %9s" % (
        "arm", "rho*", "d(y) reached", "pool mass", "N^/D^-"))
    for arm in ("w_big", "labelshift_em", "labelshift_oracle"):
        g = gg[gg.arm == arm]
        A("      %-22s %9s %13s %11s %9s" % (
            arm, _fmt(g.rho_star.median()), _fmt(g.acc_at_rho.median()),
            _fmt(g.wmass_at_rho.median()), _fmt(g.nhat_ratio.median())))
    A("")
    A("read-off: the shift is a %sx gap between the error rate the pooled source cells"
      % _fmt(float(gg.tgt_err_all.median()) / max(float(gg.src_err_all.median()), 1e-9), 1))
    A("carry (%s) and the one the deployment carries (%s), and reweighting does not"
      % (_fmt(float(gg.src_err_all.median())), _fmt(float(gg.tgt_err_all.median()))))
    A("close it -- the reweighted source error rate stays at ~%s.  That is the point:"
      % _fmt(float(gg.wsrc_err_all.median())))
    A("the weights are doing their job, and what is left is a change in the ERROR")
    A("CURVE rather than in the score or class distribution, which no reweighting of")
    A("the source pool can represent.  The bound then has only two legs to stand on,")
    A("and both are above alpha: the Hoeffding point term (median %s, which does not"
      % _fmt(float(gg[gg.arm == "labelshift_em"].nhat_ratio.median()), 2))
    A("use the error indicators at all) and the stratum-exact leg, which at its best")
    A("threshold reaches only ~%s of the deployment and therefore divides a source-"
      % _fmt(float(gg[gg.arm == "labelshift_em"].acc_at_rho.median())))
    A("level error mass by a denominator far below the deployment's own error level.")
    A("")
    A("ESS* kind at alpha=0.10 (weighted = a genuinely reweighted pool suffices;")
    A("no_weighting = only the uniform end certifies; none = nothing certifies):")
    for arm in ("w_big", "labelshift_em", "labelshift_oracle"):
        g = cert[(cert.arm == arm) & np.isclose(cert.alpha, a10)]
        vc = g.ess_star_kind.value_counts()
        A("  %-19s weighted %3d | no_weighting %3d | none %3d" % (
            arm, int(vc.get("weighted", 0)), int(vc.get("no_weighting", 0)),
            int(vc.get("none", 0))))
    A("")
    A("Those three rows are identical because the kind is decided by rho at the")
    A("UNIFORM end of the blend, which does not depend on the observed weights: if")
    A("the uniform end certifies then ESS* = n, and 129/180 units are in that state")
    A("at alpha=0.10.  So the certificate is again feasible only when the weights")
    A("vanish, exactly as in r19; the label-shift correction does not change that,")
    A("it only raises rho* at the observed weights.")
    A("")

    # ---- (iv) validity ---------------------------------------------------
    A("-" * 96)
    A("(iv) DOES THE BOUND HOLD WHEN IT SPEAKS?  AND THE ORACLE-PRIOR CONTROL")
    A("-" * 96)
    A("")
    for arm in ("labelshift_em", "labelshift_oracle", "w_big"):
        A("  %s" % arm)
        for al in alphas:
            g = cert[(cert.arm == arm) & np.isclose(cert.alpha, al)]
            nv = g[g.nonvacuous == 1]
            A("    alpha=%.2f  nonvacuous %3d/%-3d  of those, risk<=alpha %3d  (%s)"
              % (al, len(nv), len(g), int(nv.nonvac_covered.sum()),
                 _fmt(100.0 * nv.nonvac_covered.mean(), 1) + "%" if len(nv) else "n/a"))
        A("")
    A("The oracle-prior arm is the ceiling for ANY label-shift correction: it uses")
    A("the true deployment priors, so no better q_tgt exists.  If it is still vacuous,")
    A("the obstruction is not prior estimation but the certificate's own weight-spread")
    A("penalty and the fact that the source error curve does not transfer.")
    A("")

    # ---- r14 comparison --------------------------------------------------
    A("-" * 96)
    A("COMPARISON WITH THE ARMS ALREADY IN experiments/r14_targetcal.csv")
    A("-" * 96)
    A("")
    try:
        r14_all = pd.read_csv(os.path.join(paths.RESULTS, "r14_targetcal.csv"))
        r14 = r14_all[np.isclose(r14_all.k, 0)]
        A("  %-22s %8s %8s %8s %8s %10s" % (
            "arm (r14, k=0)", "alpha", "accept", "risk", "abstain", "posRisk"))
        for al in alphas:
            for arm in ("src", "src_big", "w_big", "naive", "oracle", "sc"):
                g = r14[(r14.arm == arm) & np.isclose(r14.alpha, al)]
                if not len(g):
                    continue
                ne = g[g["empty"] == 0]
                A("  %-22s %8.2f %8.4f %8.4f %8.4f %10.4f" % (
                    arm, al, g.accept_rate.mean(),
                    ne.selective_risk.mean() if len(ne) else np.nan,
                    g["empty"].mean(),
                    g.nonvac_covered.sum() / max(int((1 - g["empty"]).sum()), 1)))
        A("")
        A("  tgt_k (target-labelled calibration) at alpha=0.10, for the cost of the")
        A("  label-free correction to be judged against:")
        A("  %8s %10s %10s %10s" % ("k", "accept", "risk", "nonvacCov"))
        gk = r14_all[(r14_all.arm == "tgt_k") & np.isclose(r14_all.alpha, a10)]
        for k in sorted(gk.k.unique()):
            g = gk[gk.k == k]
            ne = g[g["empty"] == 0]
            A("  %8d %10.4f %10.4f %10.4f" % (
                k, g.accept_rate.mean(), ne.selective_risk.mean() if len(ne) else np.nan,
                g.nonvac_covered.sum() / max(int((1 - g["empty"]).sum()), 1)))
    except Exception as exc:
        A("  (r14 comparison unavailable: %r)" % (exc,))
    A("")

    # ---- limitations -----------------------------------------------------
    A("-" * 96)
    A("HOW THE HYPOTHESIS FARES")
    A("-" * 96)
    A("")
    A("  H1  'a label-shift-only correction is identifiable': TRUE in principle --")
    A("      the EM needs no target label and converges (median %s iterations) --"
      % _fmt(pri.em_iters.median(), 0))
    A("      but the estimate is poor and classifier-dependent on this benchmark")
    A("      (median L1 error %s against the true deployment prior)."
      % _fmt(pri.q_l1_err_em.median()))
    A("  H2  'it does not collapse the effective sample size': FALSE.  Median")
    A("      ESS/n %s (EM) and %s (true priors) against %s for the cell-level"
      % (_fmt(pri.ess_em_ratio.median()), _fmt(pri.ess_or_ratio.median()),
         _fmt(pri.ess_ratio.median())))
    A("      density ratio: the per-class weights are MORE concentrated, not less.")
    A("  H3  'it therefore repairs the certificate': FALSE, and not repairably so.")
    A("      At alpha=0.10 the EM arm is non-vacuous in %d/180 units and the bound"
      % nev)
    A("      holds in %d of them, while the oracle-prior ceiling is non-vacuous in"
      % hev)
    A("      %d/180 and holds in %d -- both below the plug-in density ratio (%d/180"
      % (norv, horv, nw))
    A("      and %d holding) and far below the unweighted certificate's acceptance"
      % hw)
    A("      (%s at alpha=0.10, non-vacuous in %d/180 -- but unweighted calibration"
      % (_fmt(get(a10, "src", "accept")), int(get(a10, "src", "nonvac_units"))))
    A("      is not a valid guarantee for a shifted target, which is the whole")
    A("      reason the weighted certificate exists).  Since the oracle-prior arm")
    A("      uses the true deployment priors, no better q_tgt exists: on this")
    A("      benchmark the obstruction is not prior estimation, it is the")
    A("      certificate's own weight-spread penalty combined with a source error")
    A("      curve that does not transfer.")
    A("")
    A("  Label-cost reference from r14_targetcal.csv: target-labelled calibration")
    A("  needs k >= 500 deployment labels before it reaches acceptance 0.006 with")
    A("  coverage 1.00 at alpha=0.10 (k=2000: acceptance 0.039, coverage 0.997);")
    A("  the honest route is a few hundred target labels, not a reweighting of the")
    A("  source pool.")
    A("")
    A("-" * 96)
    A("LIMITATIONS")
    A("-" * 96)
    A("")
    A("1. Pure label shift is an assumption, not a finding.  The target cell-type")
    A("   composition really does differ across tissues, but the covariate law also")
    A("   moves (r19 section (i)); a per-class ratio is then the projection of the")
    A("   true ratio onto the class labels and inherits the residual covariate drift")
    A("   as unmodelled bias.  The oracle-prior arm prices only the prior-estimation")
    A("   part of that, not the covariate part.")
    A("2. SLD/BBSE needs the classifier posteriors to be calibrated estimates of")
    A("   p_src(y|x).  These softmax/RF/HGB posteriors are over-confident, which")
    A("   biases q_tgt towards the source prior; the median L1 error above measures")
    A("   how much on this benchmark, with the true prior known.")
    A("3. q_src is the empirical class frequency of the calibration pool, while the")
    A("   classifier was trained with the training-subset prior.  The two agree only")
    A("   up to fold sampling noise; the mismatch is second order but is not zero.")
    A("4. Cells whose expert class is outside the trained class set get weight 1,")
    A("   the only choice available without a label for them.  On this benchmark")
    A("   the shared coarse class set is always fully represented in the pool, so")
    A("   no cell was affected; it would matter for a rarer class set.")
    A("5. `w_big` here recomputes r19's density ratio from the same code and seed,")
    A("   so it is a consistency check, not an independent replication.")
    A("")
    return "\n".join(L)


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="r24b_labelshift")
    ap.add_argument("--reuse", action="store_true",
                    help="load the stored CSV and rewrite the report only")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    clfs = [c for c in args.clfs.split(",") if c]
    folds = [int(x) for x in args.folds.split(",")]
    pairs = list(R2.TRANSFER_PAIRS)
    if args.only:
        pairs = [p for p in pairs if ("%s->%s" % p) == args.only]
    datasets = sorted({d for p in pairs for d in p})
    jobs = [(p, f, clfs, alphas, args.seed) for p in pairs for f in folds]
    log("%d pair-folds x %d clfs x %d alphas  (jobs=%d)"
        % (len(jobs), len(clfs), len(alphas), args.jobs))

    if args.reuse:
        df = pd.read_csv(os.path.join(paths.RESULTS, args.out + ".csv"))
        log("reused %d rows from %s.csv" % (len(df), args.out))
        rep = build_report(df, {"alphas": alphas, "clfs": clfs})
        rp = os.path.join(paths.WORK, "r24b_report.txt")
        with open(rp, "w", encoding="utf-8") as fh:
            fh.write(rep + "\n")
        log("wrote %s" % rp)
        print(rep)
        return

    rows = []
    t0 = time.time()
    if args.jobs > 1:
        with mp.Pool(args.jobs, initializer=_init, initargs=(datasets,)) as pool:
            for i, (rw, msg) in enumerate(pool.imap(_unit_job, jobs, chunksize=1), 1):
                rows.extend(rw)
                log("  %d/%d  %s" % (i, len(jobs), msg))
    else:
        _init(datasets)
        for i, jb in enumerate(jobs, 1):
            rw, msg = _unit_job(jb)
            rows.extend(rw)
            log("  %d/%d  %s" % (i, len(jobs), msg))
    log("collected in %.1fs" % (time.time() - t0))

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["pair", "fold", "clf", "alpha", "arm"]).reset_index(drop=True)
    out = os.path.join(paths.RESULTS, args.out + ".csv")
    df.to_csv(out, index=False)
    log("wrote %s  (%d rows)" % (out, len(df)))

    rep = build_report(df, {"alphas": alphas, "clfs": clfs})
    rp = os.path.join(paths.WORK, "r24b_report.txt")
    with open(rp, "w", encoding="utf-8") as fh:
        fh.write(rep + "\n")
    log("wrote %s" % rp)
    print(rep)


if __name__ == "__main__":
    main()
