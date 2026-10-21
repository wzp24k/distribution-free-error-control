"""r31 -- targeted repair of the two conformal failures: within and transfer.

Context
-------
The stored ladder (r03, r20) shows the selective-risk certificate
``SelectiveRiskController`` (num="cp", denom="cp", select="bonferroni")
answering at nominal coverage in the exchangeable family only:

    family     alpha  coverage  nonvac-coverage  acceptance
    within     0.05   0.8333    0.7510           0.3566
    within     0.10   0.8167    0.7643           0.5338
    within     0.20   0.9167    0.8990           0.7300
    transfer   0.10   0.3972    0.0083           0.3676   (oracle 0.1205)

and the only accepted cure so far is buying target labels: k = 500 under
transfer makes coverage 1.000 again -- but that 1.000 is *vacuous* at k = 500
because the certificate accepts 0.5% of the batch, and the non-vacuous coverage
is 0.019 there.  This script attacks the two failures on the identical splits,
score (s = 1 - max class proba) and candidate grid (_grid_indices(n, 25, 120))
as the stored runs.

Part A -- can the cross-field-of-view (within) failure be repaired?
  plain        the stored certificate (regression anchor: reproduces r03).
  plain_fs     the same with select="fixed_sequence".
  mond_legacy  the paper's MondrianSelectiveRiskController, which shares delta
               across strata and therefore has no joint 1-delta guarantee.
  mond_class   one certificate per *predicted class*, corrected so the union is
               valid: "max" keeps alpha and charges delta/K per stratum,
               "bonf" charges alpha/K and delta/K.  No pooled fallback, so a
               cell whose class carries no stratum is rejected.
  mond_fov     one certificate per *calibration field of view*, with the most
               conservative of the per-image thresholds as the deployed rule;
               valid if deployment images are exchangeable with calibration
               images.  "nn" assigns each deployment cell to the nearest
               calibration image in label-free feature space and then uses that
               image's threshold (the standard Mondrian reading).
  sc10/sc3     ScoreConditionalController: the denominator read off the
               unlabelled deployment scores, the error curve from calibration.
  w_ess        WeightedSelectiveRiskController with density_ratio weights and a
               deployment-score denominator; the effective sample size is
               reported, because the price of weighting is ESS, not n.
  tgt_k*       label-cost references for within (plain certificate on k target
               labels, evaluated on the remainder).

Part B -- can the cross-tissue label cost come down from 500 to <= 200?
  plain        plain certificate on k random target labels (reproduces r20).
  mond_class   per-predicted-class Mondrian on the k labels.
  iw_sel       importance-weighted calibration of the k labels, weights
               p_eval(x)/p_labelled(x) -- on a uniform draw this should be ~1.
  prop_mond    label k cells of the whole target pool with allocation
               proportional to predicted-class frequency, then per-class
               Mondrian (valid: stratified uniform sampling is exchangeable
               within stratum).
  unc_iw       label the k most uncertain cells of the whole pool (highest
               predicted entropy), then importance-weight them back (the
               labelling is biased, so the weights are what makes it valid).
  twostage_iw  pilot of k/4 uniform labels, then concentrate the remaining
               budget on the predicted classes whose pilot empirical risk is
               highest, then importance-weight.  twostage_nw is the same
               selection *without* the weights, to expose the selection bias.
  pool_iw      the union of the stored source calibration cells, weighted by
               p_target/p_source and pooled with the k target labels, one
               weighted certificate.  (src/w_src stay on a single stored repeat
               so they remain comparable with the stored runs; pool_iw has no
               stored analogue, so it takes the largest available source pool.)
  src          source-only certificate on ONE stored calibration repeat, one
               row per repeat (the r03/r20 protocol; reproduces r20's 'src' rows
               and the paper's Table-1 transfer 'crc' anchor).
  w_src        the same, importance-weighted (k-independent reference).
  oracle       test-peeking ceiling on the same remainder.

Part C -- is the failure exchangeability or bound slack?  For every arm the
realised selective risk among the accepted cells is reported against the budget
alpha, pooled with a Clopper-Pearson upper limit, so a violated bound can be
told apart from genuinely wrong labels.

Part D -- model adaptation (reproduced from experiments/r14b_targetadapt.csv)
and, for one classifier with real refits, whether the new calibration arms
stack on top of it.

Outputs
-------
experiments/r31_within_detail.csv, r31_within_summary.csv,
experiments/r31_within_ref_summary.csv, r31_transfer_detail.csv,
experiments/r31_transfer_summary.csv, r31_ladder_summary.csv, r31_partC.csv,
experiments/r31_paired.csv, r31_partD.csv, r31_stack.csv,
experiments/r31_stack_detail.csv, r31_stack_diag.csv, work/r31_report.txt

Every table carries three coverage conventions side by side: coverage_all (the
certificate's actual nominal guarantee, an empty accepted set counting as a
conservative success), coverage_nonvac = P(covered and non-empty), and
coverage_nonvac_cond = P(covered | non-empty), the manuscript's Table-1
convention.  Acceptance is printed next to each, because a level that is only
'reached' by answering almost nothing is vacuous, not a repair.
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import re
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from errctl import conformal as C, paths
from errctl.classifiers import fit_classifier
from errctl.preprocess import DomainNormaliser
from errctl.weights import density_ratio, effective_sample_size

PRED_DIR = os.path.join(paths.WORK, "predictions")

DELTA = 0.10
ALPHAS = (0.05, 0.10, 0.20)
K_ABS = (25, 50, 100, 200, 500, 1000)
REPS = 20
MIN_ACCEPT = 25                 # the paper's default candidate-count floor
MIN_STRATUM = 15                # mirror MondrianSelectiveRiskController
STRATUM_MIN_ACCEPT = 5          # mirror MondrianSelectiveRiskController
NOMINAL = 1.0 - DELTA
IW_CAP = 2000
BOOT = 4000
BOOT_SEED = 20260927
USEFUL_ACCEPT = 0.10
WITHIN_TGT_KS = (50, 200, 500)
STACK_ALPHA = 0.10
STACK_K = 2000
STACK_REPS = 3

ARM_ORDER_A = ("plain", "plain_fs", "mond_legacy", "mond_class", "mond_class_bonf",
               "mond_fov", "mond_fov_bonf", "mond_fov_nn", "sc10", "sc3", "w_ess")
ARM_ORDER_B = ("plain", "mond_class", "iw_sel", "prop_mond", "unc_iw",
               "twostage_iw", "twostage_nw", "pool_iw")


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)),
          flush=True)


# ------------------------------------------------------------------ primitives
def parse_name(fname):
    m = re.match(r"(.*)__(logreg|rf|hgb)\.npz$", fname)
    if m is None:
        raise ValueError("bad prediction file name %r" % fname)
    stem, clf = m.group(1), m.group(2)
    for family in ("cellsplit", "within", "transfer"):
        if stem.startswith(family + "_"):
            return family, stem[len(family) + 1:], clf
    raise ValueError("unknown family in %r" % fname)


def se_of(probs, y):
    probs = np.asarray(probs, dtype=np.float64)
    s = 1.0 - probs.max(1)
    e = (probs.argmax(1) != np.asarray(y)).astype(np.float64)
    return s, e


def entropy_of(probs):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -(p * np.log(p)).sum(1)


def cal_cols(cal_rep):
    """Distinct calibration repeat masks (within units reuse one mask 6x)."""
    out, seen = [], set()
    for c in range(cal_rep.shape[1]):
        key = cal_rep[:, c].tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def r20_seed(unit, clf):
    return 20250926 + zlib.crc32(("%s|%s" % (unit, clf)).encode("utf-8"))


def r20_pick(unit, clf, rep, k, n_pool, grid_id=0):
    """The r20 draw, reproduced bit for bit so the plain arm is a regression."""
    rng = np.random.default_rng([r20_seed(unit, clf), rep, k, grid_id])
    return rng.choice(n_pool, size=k, replace=False)


def cp_upper(n_wrong, n, conf=0.95):
    n = int(n)
    n_wrong = int(round(n_wrong))
    if n <= 0 or n_wrong < 0:
        return float("nan")
    if n_wrong >= n:
        return 1.0
    return float(_beta.ppf(conf, n_wrong + 1.0, n - n_wrong))


def cp_lower(n_wrong, n, conf=0.95):
    n = int(n)
    n_wrong = int(round(n_wrong))
    if n <= 0:
        return float("nan")
    if n_wrong <= 0:
        return 0.0
    if n_wrong >= n:
        return float(_beta.ppf(1.0 - conf, n_wrong, 1.0))
    return float(_beta.ppf(1.0 - conf, n_wrong, n - n_wrong + 1.0))


def oracle_lambda(s_sorted, e_sorted, keep_sorted, alpha, min_accept):
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


def iw_select(X_lab, X_ev, seed=0, cap=IW_CAP):
    """Weights p_eval(x) / p_labelled(x) for the labelled cells.

    The evaluation remainder is an unlabelled sample of the deployment
    distribution, so a domain classifier between it and the labelled cells
    estimates exactly the correction that turns a biased labelling rule into an
    exchangeable weighted sample.  The evaluation side is capped for speed; the
    ratio model is fitted on at most ``cap`` unlabelled cells.
    """
    X_ev = np.asarray(X_ev, dtype=np.float64)
    n = X_ev.shape[0]
    if n > cap:
        rng = np.random.default_rng(seed)
        X_t = X_ev[rng.choice(n, size=cap, replace=False)]
    else:
        X_t = X_ev
    w = density_ratio(np.asarray(X_lab, dtype=np.float64), X_t,
                      clip=20.0, seed=seed, cv=3)
    return w, effective_sample_size(w)


def select_prop(lab_pool, k, seed):
    """Stratified by predicted class, allocation proportional to frequency."""
    labs, counts = np.unique(lab_pool, return_counts=True)
    rng = np.random.default_rng(seed)
    alloc = counts / float(counts.sum()) * k
    take = np.minimum(np.floor(alloc).astype(int), counts)
    rem = int(k - take.sum())
    if rem > 0:
        frac = alloc - np.floor(alloc)
        order = np.argsort(-frac, kind="mergesort")
        for j in order:
            if rem <= 0:
                break
            if take[j] < counts[j]:
                take[j] += 1
                rem -= 1
    out = []
    for c, n_c in zip(labs, take):
        idx = np.nonzero(lab_pool == c)[0]
        if n_c > 0 and idx.size:
            out.append(rng.choice(idx, size=int(min(n_c, idx.size)), replace=False))
    sel = np.concatenate(out) if out else np.array([], dtype=int)
    sel = np.unique(sel)
    if sel.size < k:
        left = np.setdiff1d(np.arange(lab_pool.size), sel)
        extra = rng.choice(left, size=min(k - sel.size, left.size), replace=False)
        sel = np.union1d(sel, extra)
    return np.sort(sel[:k])


def select_unc(ent_pool, k):
    """The k cells with the least confident prediction (highest entropy)."""
    order = np.lexsort((np.arange(ent_pool.size), -np.asarray(ent_pool, dtype=np.float64)))
    return np.sort(order[:k])


def select_twostage(lab_pool, e_pool, k, seed):
    """Pilot uniformly, then spend the rest on the highest-risk strata.

    The pilot's per-predicted-class smoothed error rate r_c sets the allocation
    weight r_c * sqrt(n_c), so strata that look wrong and are common attract
    labels.  The selection is deliberately biased; the caller importance-weights
    it back.
    """
    rng = np.random.default_rng(seed)
    n_pool = lab_pool.size
    pilot_n = max(5, k // 4)
    pilot = rng.choice(n_pool, size=pilot_n, replace=False)
    rest = np.setdiff1d(np.arange(n_pool), pilot)
    labs = np.unique(lab_pool)
    w = np.ones(labs.size)
    for i, c in enumerate(labs):
        m = pilot[lab_pool[pilot] == c]
        n_c = m.size + 1.0
        wrong = e_pool[m].sum() + 0.5
        w[i] = max((wrong / n_c) * np.sqrt(1.0 + n_c), 1e-3)
    need = k - pilot.size
    alloc = np.minimum(np.floor(w / w.sum() * need).astype(int),
                       np.array([np.sum(lab_pool[rest] == c) for c in labs]))
    rem = int(need - alloc.sum())
    if rem > 0:
        frac = w / w.sum() * need - np.floor(w / w.sum() * need)
        for j in np.argsort(-frac, kind="mergesort"):
            if rem <= 0:
                break
            cap = np.sum(lab_pool[rest] == labs[j])
            if alloc[j] < cap:
                alloc[j] += 1
                rem -= 1
    out = [pilot]
    for c, n_c in zip(labs, alloc):
        pool_c = rest[lab_pool[rest] == c]
        if n_c > 0 and pool_c.size:
            out.append(rng.choice(pool_c, size=int(min(n_c, pool_c.size)),
                                  replace=False))
    sel = np.unique(np.concatenate(out))
    if sel.size < k:
        left = np.setdiff1d(np.arange(n_pool), sel)
        sel = np.union1d(sel, rng.choice(left, size=min(k - sel.size, left.size),
                                         replace=False))
    return np.sort(sel[:k])


def fov_assign(X_cal, g_cal, X_te):
    """Label-free nearest-centroid assignment of deployment cells to images."""
    labs = np.unique(g_cal)
    Z = np.vstack([np.asarray(X_cal, dtype=np.float64),
                   np.asarray(X_te, dtype=np.float64)])
    mu = Z.mean(0)
    sd = Z.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Zc = (np.asarray(X_cal, dtype=np.float64) - mu) / sd
    Zt = (np.asarray(X_te, dtype=np.float64) - mu) / sd
    cent = np.vstack([Zc[g_cal == g].mean(0) for g in labs])
    d2 = ((Zt[:, None, :] - cent[None, :, :]) ** 2).sum(2)
    return labs, labs[np.argmin(d2, 1)]


def detail_row(part, family, unit, clf, alpha, arm, k, rep, n_pool, n_lab, n_ev,
               n_acc, risk, lam, extra=None):
    row = {
        "part": part, "family": family, "unit": unit, "clf": clf,
        "alpha": float(alpha), "arm": arm, "k": int(k), "rep": int(rep),
        "n_pool": int(n_pool), "n_lab": int(n_lab), "n_eval": int(n_ev),
        "n_accepted": int(n_acc),
        "n_wrong": (int(round(risk * n_acc)) if (n_acc and np.isfinite(risk))
                    else 0),
        "accept_rate": (n_acc / float(n_ev) if n_ev else np.nan),
        "selective_risk": risk, "empty": int(n_acc == 0),
        "covered": int((not np.isfinite(risk)) or risk <= alpha),
        "covered_nonvac": int(np.isfinite(risk) and risk <= alpha and n_acc > 0),
        "lambda_hat": float(lam),
        "n_strata": np.nan, "n_fitted": np.nan, "n_unfitted": np.nan,
        "n_empty_strata": np.nan, "ess": np.nan, "w_max": np.nan,
    }
    if extra:
        row.update(extra)
    return row


def empty_detail():
    """A zero-row frame carrying the detail schema (for --only smoke runs)."""
    return pd.DataFrame([detail_row("X", "X", "X", "X", 0.0, "X", 0, 0, 0, 0, 0, 0,
                                    np.nan, -np.inf)]).iloc[0:0]


def summarise(acc, e_ev):
    n = int(np.asarray(acc, dtype=bool).sum())
    if n == 0:
        return 0, float("nan")
    return n, float(np.asarray(e_ev, dtype=np.float64)[np.asarray(acc, dtype=bool)].mean())

def emit(rows, part, family, unit, clf, alpha, arm, k, rep, e_ev, acc, lam,
         n_pool, n_lab, extra=None):
    acc = np.asarray(acc, dtype=bool)
    n_acc, risk = summarise(acc, e_ev)
    rows.append(detail_row(part, family, unit, clf, alpha, arm, k, rep, n_pool,
                           n_lab, np.asarray(e_ev).size, n_acc, risk, lam, extra))


# ------------------------------------------------------------- certifications
class MondrianPred:
    """One certificate per *predicted class*, corrected for multiplicity.

    ``correction="max"``  keeps the target alpha in every stratum and charges
    delta/K per stratum, so a union bound over the K strata still holds at
    delta and each stratum's selective risk is controlled at alpha.
    ``correction="bonf"`` charges alpha/K and delta/K, the conservative reading.

    There is no pooled fallback: a cell whose predicted class carries no fitted
    stratum is rejected.  A pooled fallback would be more efficient but the
    union bound would no longer cover it, so the empty strata are counted and
    reported instead of being papered over.
    """

    def __init__(self, alpha, delta, correction="max", min_stratum=MIN_STRATUM,
                 stratum_min_accept=STRATUM_MIN_ACCEPT):
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.correction = correction
        self.min_stratum = int(min_stratum)
        self.stratum_min_accept = int(stratum_min_accept)
        self.controllers_ = {}
        self.K_ = 0
        self.n_unfitted_ = 0

    def fit(self, s, e, lab):
        s = np.asarray(s, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)
        lab = np.asarray(lab)
        labs = np.unique(lab)
        self.K_ = int(labs.size)
        sh_a = self.alpha / self.K_ if self.correction == "bonf" else self.alpha
        sh_d = self.delta / self.K_
        for c in labs:
            m = lab == c
            if int(m.sum()) < self.min_stratum:
                self.n_unfitted_ += 1
                continue
            ctl = C.SelectiveRiskController(sh_a, sh_d,
                                            min_accept=self.stratum_min_accept)
            self.controllers_[c] = ctl.fit(s[m], e[m])
        return self

    def accept(self, s, lab):
        s = np.asarray(s, dtype=np.float64)
        lab = np.asarray(lab)
        out = np.zeros(s.shape[0], dtype=bool)
        for c, ctl in self.controllers_.items():
            m = lab == c
            if m.any():
                out[m] = ctl.accept(s[m])
        return out

    def report(self, s, lab):
        acc = self.accept(s, lab)
        lab = np.asarray(lab)
        labs = np.unique(lab)
        empty = 0
        for c in labs:
            m = lab == c
            if not acc[m].any():
                empty += 1
        return {"n_strata": int(labs.size), "n_fitted": int(len(self.controllers_)),
                "n_unfitted": int(self.n_unfitted_), "n_empty_strata": int(empty)}


class GroupMondrian:
    """One certificate per calibration *field of view*, min over the strata.

    Deployment field-of-view identity carries no calibration cells (the splits
    are by image), so the only rule that keeps the union bound is the most
    conservative per-image threshold.  The assumption is written down in the
    report: deployment images exchangeable with calibration images.
    """

    def __init__(self, alpha, delta, correction="max", min_stratum=MIN_STRATUM,
                 stratum_min_accept=STRATUM_MIN_ACCEPT):
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.correction = correction
        self.min_stratum = int(min_stratum)
        self.stratum_min_accept = int(stratum_min_accept)
        self.controllers_ = {}
        self.n_unfitted_ = 0
        self.n_infeasible_ = 0
        self.lambda_hat = -np.inf

    def fit(self, s, e, grp):
        s = np.asarray(s, dtype=np.float64)
        e = np.asarray(e, dtype=np.float64)
        grp = np.asarray(grp)
        labs = np.unique(grp)
        K = int(labs.size)
        sh_a = self.alpha / K if self.correction == "bonf" else self.alpha
        sh_d = self.delta / K
        for g in labs:
            m = grp == g
            if int(m.sum()) < self.min_stratum:
                self.n_unfitted_ += 1
                continue
            ctl = C.SelectiveRiskController(sh_a, sh_d,
                                            min_accept=self.stratum_min_accept)
            ctl.fit(s[m], e[m])
            self.controllers_[g] = ctl
            if not np.isfinite(ctl.lambda_hat):
                self.n_infeasible_ += 1
        lams = [c.lambda_hat for c in self.controllers_.values()]
        self.lambda_hat = float(np.min(lams)) if lams else -np.inf
        return self

    def accept(self, s):
        return np.asarray(s, dtype=np.float64) <= self.lambda_hat

    def report(self, s, grp):
        acc = self.accept(s)
        grp = np.asarray(grp)
        labs = np.unique(grp)
        empty = 0
        for g in labs:
            m = grp == g
            if not acc[m].any():
                empty += 1
        return {"n_strata": int(labs.size), "n_fitted": int(len(self.controllers_)),
                "n_unfitted": int(self.n_unfitted_), "n_empty_strata": int(empty)}

# ------------------------------------------------------------------- part A
def within_arms(rows, unit, clf, alpha, col, s_cal, e_cal, p_cal, g_cal, X_cal,
                s_te, e_te, p_te, g_te, X_te, tgt_ks):
    n_pool = int(s_te.size)
    n_cal = int(s_cal.size)

    ctl = C.SelectiveRiskController(alpha, DELTA, min_accept=MIN_ACCEPT
                                    ).fit(s_cal, e_cal)
    emit(rows, "A", "within", unit, clf, alpha, "plain", 0, col, e_te,
         ctl.accept(s_te), ctl.lambda_hat, n_pool, n_cal)

    ctl_fs = C.SelectiveRiskController(alpha, DELTA, min_accept=MIN_ACCEPT,
                                       select="fixed_sequence").fit(s_cal, e_cal)
    emit(rows, "A", "within", unit, clf, alpha, "plain_fs", 0, col, e_te,
         ctl_fs.accept(s_te), ctl_fs.lambda_hat, n_pool, n_cal)

    mon = C.MondrianSelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal, p_cal)
    acc = mon.accept(s_te, p_te)
    labs = np.unique(p_te)
    empty = int(sum(1 for c in labs if not acc[p_te == c].any()))
    emit(rows, "A", "within", unit, clf, alpha, "mond_legacy", 0, col, e_te, acc,
         mon.fallback_.lambda_hat, n_pool, n_cal,
         {"n_strata": int(labs.size), "n_fitted": int(len(mon.controllers_)),
          "n_unfitted": int(labs.size - len(mon.controllers_)),
          "n_empty_strata": empty})

    for tag, corr in (("mond_class", "max"), ("mond_class_bonf", "bonf")):
        md = MondrianPred(alpha, DELTA, corr).fit(s_cal, e_cal, p_cal)
        acc = md.accept(s_te, p_te)
        emit(rows, "A", "within", unit, clf, alpha, tag, 0, col, e_te, acc,
             np.nan, n_pool, n_cal, md.report(s_te, p_te))

    for tag, corr in (("mond_fov", "max"), ("mond_fov_bonf", "bonf")):
        gm = GroupMondrian(alpha, DELTA, corr).fit(s_cal, e_cal, g_cal)
        emit(rows, "A", "within", unit, clf, alpha, tag, 0, col, e_te,
             gm.accept(s_te), gm.lambda_hat, n_pool, n_cal, gm.report(s_te, g_te))

    gl, g_te_nn = fov_assign(X_cal, g_cal, X_te)
    md = MondrianPred(alpha, DELTA, "max").fit(s_cal, e_cal, g_cal)
    acc = md.accept(s_te, g_te_nn)
    emit(rows, "A", "within", unit, clf, alpha, "mond_fov_nn", 0, col, e_te, acc,
         np.nan, n_pool, n_cal, md.report(s_te, g_te_nn))

    for tag, nb in (("sc10", 10), ("sc3", 3)):
        sc = C.ScoreConditionalController(alpha, DELTA, n_bins=nb
                                          ).fit(s_cal, e_cal, s_te)
        emit(rows, "A", "within", unit, clf, alpha, tag, 0, col, e_te,
             sc.accept(s_te), sc.lambda_hat, n_pool, n_cal)

    try:
        w, ess = iw_select(X_cal, X_te, seed=col)
        cw = C.WeightedSelectiveRiskController(alpha, DELTA
                                              ).fit(s_cal, e_cal, w,
                                                    target_scores=s_te)
        emit(rows, "A", "within", unit, clf, alpha, "w_ess", 0, col, e_te,
             cw.accept(s_te), cw.lambda_hat, n_pool, n_cal,
             {"ess": float(ess), "w_max": float(np.max(w))})
    except Exception as exc:
        rows.append(detail_row("A", "within", unit, clf, alpha, "w_ess", 0, col,
                               n_pool, n_cal, n_pool, 0, float("nan"),
                               -np.inf, {"ess": float("nan"),
                                         "w_max": float("nan"),
                                         "arm_note": "failed: %s" % exc}))

    for k in tgt_ks:
        if k >= n_pool:
            continue
        pick = r20_pick(unit, clf, col, k, n_pool)
        keep = np.ones(n_pool, dtype=bool)
        keep[pick] = False
        ev = np.nonzero(keep)[0]
        c = C.SelectiveRiskController(alpha, DELTA, min_accept=MIN_ACCEPT
                                      ).fit(s_te[pick], e_te[pick])
        emit(rows, "A_ref", "within", unit, clf, alpha, "tgt_k%d" % k, k, col,
             e_te[ev], c.accept(s_te[ev]), c.lambda_hat, n_pool, k)


def within_file(job):
    fname, alphas, tgt_ks = job
    family, unit, clf = parse_name(fname)
    if family != "within":
        return []
    d = np.load(os.path.join(PRED_DIR, fname), allow_pickle=True)
    s_te, e_te = se_of(d["test_probs"], d["test_y"])
    s_cal_all, e_cal_all = se_of(d["cal_probs"], d["cal_y"])
    p_cal_all = np.asarray(d["cal_probs"]).argmax(1)
    p_te = np.asarray(d["test_probs"]).argmax(1)
    X_cal_all = np.asarray(d["cal_X"], dtype=np.float64)
    X_te = np.asarray(d["test_X"], dtype=np.float64)
    g_cal_all = np.asarray(d["cal_img"])
    g_te = np.asarray(d["test_img"])
    rows = []
    for col in range(d["cal_rep"].shape[1]):
        sel = d["cal_rep"][:, col]
        for alpha in alphas:
            within_arms(rows, unit, clf, alpha, col, s_cal_all[sel], e_cal_all[sel],
                        p_cal_all[sel], g_cal_all[sel], X_cal_all[sel], s_te, e_te,
                        p_te, g_te, X_te, tgt_ks)
    return rows

# ------------------------------------------------------------------- part B
def transfer_arms(rows, unit, clf, alpha, k, rep, pick, ev, s_all, e_all, p_all,
                  ent, X_all, s_src, e_src, X_src, w_src, ess_src):
    n_pool = int(s_all.size)
    s_lab, e_lab, p_lab = s_all[pick], e_all[pick], p_all[pick]
    X_lab = X_all[pick]
    s_ev, e_ev = s_all[ev], e_all[ev]

    ctl = C.SelectiveRiskController(alpha, DELTA, min_accept=MIN_ACCEPT
                                    ).fit(s_lab, e_lab)
    emit(rows, "B", "transfer", unit, clf, alpha, "plain", k, rep, e_ev,
         ctl.accept(s_ev), ctl.lambda_hat, n_pool, k)

    md = MondrianPred(alpha, DELTA, "max").fit(s_lab, e_lab, p_lab)
    emit(rows, "B", "transfer", unit, clf, alpha, "mond_class", k, rep, e_ev,
         md.accept(s_ev, p_all[ev]), np.nan, n_pool, k, md.report(s_ev, p_all[ev]))

    w, ess = iw_select(X_lab, X_all[ev], seed=1000 * rep + k)
    cw = C.WeightedSelectiveRiskController(alpha, DELTA
                                          ).fit(s_lab, e_lab, w, target_scores=s_ev)
    emit(rows, "B", "transfer", unit, clf, alpha, "iw_sel", k, rep, e_ev,
         cw.accept(s_ev), cw.lambda_hat, n_pool, k,
         {"ess": float(ess), "w_max": float(np.max(w))})

    # The three active arms below choose WHICH cells to label, so they get their
    # own labelled set and their own evaluation remainder.  Selecting from the k
    # already-drawn cells instead would be a no-op (proportional allocation over a
    # size-k sample returns all k of them), which is what an earlier revision did
    # by mistake; the B6 check now guards against that.  Coverage and acceptance
    # are therefore measured on a different remainder from plain's, but the
    # pairing survives at the (unit, classifier, rep, k) level, which is the level
    # at which the two arms share a cohort, a seed family and a batch size.
    pool_idx = np.arange(n_pool, dtype=int)

    def _ov(sel):
        """Overlap of a labelled set with plain's drawn set, as a fraction of k.

        plain labels exactly the k cells drawn by r20_pick.  A no-op
        'active-selection' arm -- one that re-selects among those same k cells,
        which an earlier revision did -- returns sel == pick and therefore
        _ov == 1.0.  A genuine active arm that chooses which cells to label out
        of the whole pool returns a small overlap.  The value is recorded in the
        detail as ``sel_overlap`` and B6 asserts max(sel_overlap) < 1 over the
        four active arms, so a future regression to the no-op cannot pass.
        """
        return float(np.intersect1d(np.asarray(sel, dtype=int), pick).size
                     / float(k))

    selp = select_prop(p_all, k, seed=7 * rep + k)
    evp = np.setdiff1d(pool_idx, selp)
    mdp = MondrianPred(alpha, DELTA, "max").fit(s_all[selp], e_all[selp],
                                                p_all[selp])
    emit(rows, "B", "transfer", unit, clf, alpha, "prop_mond", k, rep,
         e_all[evp], mdp.accept(s_all[evp], p_all[evp]), np.nan, n_pool, k,
         dict(mdp.report(s_all[evp], p_all[evp]), sel_overlap=_ov(selp)))

    selu = select_unc(ent, k)
    evu = np.setdiff1d(pool_idx, selu)
    wu, essu = iw_select(X_all[selu], X_all[evu], seed=13 * rep + k)
    cu = C.WeightedSelectiveRiskController(alpha, DELTA
                                          ).fit(s_all[selu], e_all[selu], wu,
                                                target_scores=s_all[evu])
    emit(rows, "B", "transfer", unit, clf, alpha, "unc_iw", k, rep, e_all[evu],
         cu.accept(s_all[evu]), cu.lambda_hat, n_pool, k,
         {"ess": float(essu), "w_max": float(np.max(wu)),
          "sel_overlap": _ov(selu)})

    # The two-stage rule spends a uniform pilot and then concentrates on the
    # strata whose PILOT realised risk is highest.  That reads errors of pilot
    # cells only, which is legitimate because the pilot is labelled first, but
    # the resulting selection is a function of y and not of x alone, so the
    # density-ratio weights cannot fully undo it: twostage_nw is the uncorrected
    # selection and the pair exposes exactly that residual bias.
    selt = select_twostage(p_all, e_all, k, seed=29 * rep + k)
    evt = np.setdiff1d(pool_idx, selt)
    wt, esst = iw_select(X_all[selt], X_all[evt], seed=29 * rep + k)
    ct = C.WeightedSelectiveRiskController(alpha, DELTA
                                          ).fit(s_all[selt], e_all[selt], wt,
                                                target_scores=s_all[evt])
    emit(rows, "B", "transfer", unit, clf, alpha, "twostage_iw", k, rep,
         e_all[evt], ct.accept(s_all[evt]), ct.lambda_hat, n_pool, k,
         {"ess": float(esst), "w_max": float(np.max(wt)),
          "sel_overlap": _ov(selt)})
    ctn = C.SelectiveRiskController(alpha, DELTA, min_accept=MIN_ACCEPT
                                    ).fit(s_all[selt], e_all[selt])
    emit(rows, "B", "transfer", unit, clf, alpha, "twostage_nw", k, rep,
         e_all[evt], ctn.accept(s_all[evt]), ctn.lambda_hat, n_pool, k,
         {"sel_overlap": _ov(selt)})

    w_cat = np.concatenate([np.asarray(w_src, dtype=np.float64), np.ones(k)])
    s_cat = np.concatenate([s_src, s_lab])
    e_cat = np.concatenate([e_src, e_lab])
    cp_ = C.WeightedSelectiveRiskController(alpha, DELTA
                                           ).fit(s_cat, e_cat, w_cat,
                                                 target_scores=s_ev)
    emit(rows, "B", "transfer", unit, clf, alpha, "pool_iw", k, rep, e_ev,
         cp_.accept(s_ev), cp_.lambda_hat, n_pool, k + int(s_src.size),
         {"ess": float(effective_sample_size(w_cat)),
          "w_max": float(np.max(w_cat))})

    order = np.argsort(s_ev, kind="mergesort")
    lam_o = oracle_lambda(s_ev[order], e_ev[order], np.ones(s_ev.size, bool),
                          alpha, MIN_ACCEPT)
    emit(rows, "B", "transfer", unit, clf, alpha, "oracle", k, rep, e_ev,
         s_ev <= lam_o, lam_o, n_pool, 0)


def transfer_file(job):
    fname, alphas, ks, reps = job
    family, unit, clf = parse_name(fname)
    if family != "transfer":
        return []
    d = np.load(os.path.join(PRED_DIR, fname), allow_pickle=True)
    s_all, e_all = se_of(d["test_probs"], d["test_y"])
    p_all = np.asarray(d["test_probs"]).argmax(1)
    ent = entropy_of(d["test_probs"])
    X_all = np.asarray(d["test_X"], dtype=np.float64)
    s_src, e_src = se_of(d["cal_probs"], d["cal_y"])
    X_src = np.asarray(d["cal_X"], dtype=np.float64)
    cal_rep = np.asarray(d["cal_rep"], dtype=bool)
    n_rep = int(cal_rep.shape[1])
    n_pool = int(s_all.size)
    rows = []

    # The stored protocol (r03/r20) calibrates the source-only reference on ONE
    # stored calibration repeat, not on the union of the repeats: the
    # Melanoma-source pairs store six distinct 2500-cell masks out of a
    # 4101-cell pool, so the union is a different, larger sample.  src and w_src
    # are therefore emitted once per stored repeat, which is exactly what the
    # r20_units 'src' rows are and what the paper's Table-1 transfer 'crc'
    # anchor averages over.  The union is kept only for pool_iw, which has no
    # stored analogue and is strongest with the largest source pool.
    for col in range(n_rep):
        m = cal_rep[:, col]
        s_c, e_c, X_c = s_src[m], e_src[m], X_src[m]
        w_c, ess_c = iw_select(X_c, X_all, seed=100000 + col)
        for alpha in alphas:
            c0 = C.SelectiveRiskController(alpha, DELTA, min_accept=MIN_ACCEPT
                                           ).fit(s_c, e_c)
            emit(rows, "B", "transfer", unit, clf, alpha, "src", 0, col, e_all,
                 c0.accept(s_all), c0.lambda_hat, n_pool, int(s_c.size))
            c0w = C.WeightedSelectiveRiskController(alpha, DELTA
                                                   ).fit(s_c, e_c, w_c,
                                                         target_scores=s_all)
            emit(rows, "B", "transfer", unit, clf, alpha, "w_src", 0, col,
                 e_all, c0w.accept(s_all), c0w.lambda_hat, n_pool,
                 int(s_c.size),
                 {"ess": float(ess_c), "w_max": float(np.max(w_c))})

    w_src, ess_src = iw_select(X_src, X_all, seed=0)

    for rep in range(reps):
        for k in ks:
            if k < 5 or k > n_pool // 2:
                continue
            pick = r20_pick(unit, clf, rep, k, n_pool)
            keep = np.ones(n_pool, dtype=bool)
            keep[pick] = False
            ev = np.nonzero(keep)[0]
            for alpha in alphas:
                transfer_arms(rows, unit, clf, alpha, k, rep, pick, ev, s_all,
                              e_all, p_all, ent, X_all, s_src, e_src, X_src,
                              w_src, ess_src)
    return rows

# ------------------------------------------------------- part D (reproduction)
def part_d():
    """Reproduce the stored target-adaptation result, with the exact filter."""
    p = os.path.join(paths.RESULTS, "r14b_targetadapt.csv")
    q = os.path.join(paths.RESULTS, "r14_targetcal_summary.csv")
    if not os.path.exists(p):
        return None, None
    d = pd.read_csv(p)
    rows = []
    for arm in ("adapt_k", "tgt_k_same", "oracle_adapt", "oracle_base"):
        f = d[(d.arm == arm) & np.isclose(d.alpha, STACK_ALPHA) & (d.k == STACK_K)]
        if not len(f):
            continue
        nv = f[f["empty"] == 0]
        rows.append({
            "arm": arm, "alpha": STACK_ALPHA, "k": STACK_K, "n_rows": int(len(f)),
            "n_units": int(f.unit.nunique()), "n_clf": int(f.clf.nunique()),
            "mean_accept": float(f.accept_rate.mean()),
            "coverage_all": float(f.covered.mean()),
            "nonvac_frac": float(1.0 - f["empty"].mean()),
            "mean_risk": float(f.selective_risk.mean()),
            "coverage_nonvac": (float(nv.covered.mean()) if len(nv) else np.nan),
            "ratio_full_2000_calibration": np.nan,
            "ratio_same_run": np.nan,
        })
    tab = pd.DataFrame(rows)
    ref = np.nan
    if os.path.exists(q):
        s = pd.read_csv(q)
        f = s[(s.arm == "tgt_k") & np.isclose(s.alpha, STACK_ALPHA) & (s.k == STACK_K)]
        if len(f):
            ref = float(f.mean_accept.iloc[0])
    by = {r["arm"]: r for r in rows}
    if "adapt_k" in by and np.isfinite(ref) and ref > 0:
        tab.loc[tab.arm == "adapt_k", "ratio_full_2000_calibration"] = by["adapt_k"]["mean_accept"] / ref
    if "oracle_adapt" in by and "oracle_base" in by and by["oracle_base"]["mean_accept"] > 0:
        tab.loc[tab.arm == "oracle_adapt", "ratio_full_2000_calibration"] = (
            by["oracle_adapt"]["mean_accept"] / by["oracle_base"]["mean_accept"])
    base = by.get("tgt_k_same", {}).get("mean_accept", np.nan)
    if np.isfinite(base) and base > 0:
        tab.loc[tab.arm == "adapt_k", "ratio_same_run"] = by["adapt_k"]["mean_accept"] / base
    info = {"tgt_k_full_2000_mean_accept": ref,
            "n_rows_total": int(len(d)),
            "filter": "arm in (adapt_k,tgt_k_same,oracle_adapt,oracle_base), alpha == 0.10, k == 2000"}
    return tab, info


# --------------------------------------- part D2 (do the new arms stack?)
COHORT = {}
STACK_CFG = {}


def _stack_init(cohort, cfg):
    for k, v in cohort.items():
        COHORT[k] = v
    STACK_CFG.clear()
    STACK_CFG.update(cfg)


def stack_unit(job):
    """Refit on k/2 target labels, then compare calibration arms on top."""
    src, tgt, f, clf_kind, alpha, k, reps, seed = job
    import r02_run_experiments as R2
    rows = []
    gs, ms, Xs = COHORT[src]
    gt, mt, Xt = COHORT[tgt]
    panel = R2.shared_panel(ms, mt)
    Xs_p = gs[[p[1] for p in panel]].to_numpy(dtype=np.float64)
    Xt_p = gt[[p[2] for p in panel]].to_numpy(dtype=np.float64)
    counts_t = gt["coarse_expert"].value_counts()
    counts_s = gs["coarse_expert"].value_counts()
    shared = sorted(c for c in counts_t.index
                    if counts_t[c] >= R2.MIN_CLASS_TEST and counts_s.get(c, 0) >= 30)
    if len(shared) < 3:
        return rows, "%s->%s: too few shared classes" % (src, tgt)
    sm = gs["coarse_expert"].isin(shared).to_numpy()
    tm = gt["coarse_expert"].isin(shared).to_numpy()
    gs2, Xs2 = gs[sm].reset_index(drop=True), Xs_p[sm]
    gt2, Xt2 = gt[tm].reset_index(drop=True), Xt_p[tm]
    ys = gs2["coarse_expert"].astype(str).to_numpy()
    yt = gt2["coarse_expert"].astype(str).to_numpy()
    folds = R2.fold_split(gs2["image"].unique(), R2.N_FOLDS, seed=seed + 7)
    test_imgs, cal_imgs = set(folds[f]), set(folds[(f + 1) % R2.N_FOLDS])
    tr = ~gs2["image"].isin(test_imgs | cal_imgs).to_numpy()
    ca = gs2["image"].isin(cal_imgs).to_numpy()
    tr_idx = R2.subsample_training(np.nonzero(tr)[0], ys, R2.MAX_TRAIN, seed=seed + f)
    ca_idx = np.nonzero(ca)[0]
    if len(ca_idx) < 50:
        return rows, "%s->%s fold%d: calibration fold too small" % (src, tgt, f)
    nz = DomainNormaliser("z_dataset")
    nz.fit("src", Xs2[tr_idx])
    nz.fit("tgt", Xt2)
    Xtr = nz.transform("src", Xs2[tr_idx])
    Xca = nz.transform("src", Xs2[ca_idx])
    Xte = nz.transform("tgt", Xt2)
    keep_cls = pd.Series(ys[tr_idx]).value_counts()
    classes = sorted(c for c in keep_cls.index if keep_cls[c] >= R2.MIN_CLASS_TRAIN)
    tr_keep = np.isin(ys[tr_idx], classes)
    idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([idx[c] for c in ys[tr_idx][tr_keep]], dtype=int)
    y_ca = np.array([idx.get(c, -1) for c in ys[ca_idx]], dtype=int)
    y_te = np.array([idx.get(c, -1) for c in yt], dtype=int)
    unit = "transfer:%s->%s:fold%d" % (src, tgt, f)
    tag = zlib.crc32(unit.encode()) & 0xFFFFFFFF

    clf0 = fit_classifier(Xtr[tr_keep], y_train, kind=clf_kind, seed=seed)
    p_te0 = clf0.predict_proba(Xte)
    s_te0, e_te0 = se_of(p_te0, y_te)
    s_ca0, e_ca0 = se_of(clf0.predict_proba(Xca), y_ca)

    for rep in range(reps):
        rng = np.random.default_rng(100000 * rep + tag + 977)
        perm = rng.permutation(s_te0.size)
        pool, ev = perm[:k], perm[k:]
        n_tr = max(5, k // 2)
        idx_tr, idx_cal = pool[:n_tr], pool[n_tr:]
        if idx_cal.size < 40 or ev.size < 200:
            continue
        X_aug = np.vstack([Xtr[tr_keep], Xte[idx_tr]])
        y_aug = np.concatenate([y_train, y_te[idx_tr]])
        if set(np.unique(y_aug)) != set(range(len(classes))):
            continue
        clf2 = fit_classifier(X_aug, y_aug, kind=clf_kind, seed=seed)
        s_cal2, e_cal2 = se_of(clf2.predict_proba(Xte[idx_cal]), y_te[idx_cal])
        s_ev2, e_ev2 = se_of(clf2.predict_proba(Xte[ev]), y_te[ev])
        p_cal2 = clf2.predict_proba(Xte[idx_cal]).argmax(1)
        p_ev2 = clf2.predict_proba(Xte[ev]).argmax(1)
        s_src2, e_src2 = se_of(clf2.predict_proba(Xca), y_ca)
        X_ev2 = Xte[ev]
        n_pool = int(s_te0.size)
        ctl = C.SelectiveRiskController(alpha, DELTA, min_accept=MIN_ACCEPT
                                        ).fit(s_cal2, e_cal2)
        emit(rows, "D2", "transfer", unit, clf_kind, alpha, "adapt_plain", k, rep,
             e_ev2, ctl.accept(s_ev2), ctl.lambda_hat, n_pool, int(idx_cal.size))
        md = MondrianPred(alpha, DELTA, "max").fit(s_cal2, e_cal2, p_cal2)
        emit(rows, "D2", "transfer", unit, clf_kind, alpha, "adapt_mond", k, rep,
             e_ev2, md.accept(s_ev2, p_ev2), np.nan, n_pool, int(idx_cal.size),
             md.report(s_ev2, p_ev2))
        w, ess = iw_select(Xte[idx_cal], X_ev2, seed=1000 * rep + k)
        cw = C.WeightedSelectiveRiskController(alpha, DELTA
                                              ).fit(s_cal2, e_cal2, w,
                                                    target_scores=s_ev2)
        emit(rows, "D2", "transfer", unit, clf_kind, alpha, "adapt_iw", k, rep,
             e_ev2, cw.accept(s_ev2), cw.lambda_hat, n_pool, int(idx_cal.size),
             {"ess": float(ess), "w_max": float(np.max(w))})
        w_src, ess_src = iw_select(Xca, X_ev2, seed=0)
        w_cat = np.concatenate([w_src, np.ones(idx_cal.size)])
        s_cat = np.concatenate([s_src2, s_cal2])
        e_cat = np.concatenate([e_src2, e_cal2])
        cp_ = C.WeightedSelectiveRiskController(alpha, DELTA
                                               ).fit(s_cat, e_cat, w_cat,
                                                     target_scores=s_ev2)
        emit(rows, "D2", "transfer", unit, clf_kind, alpha, "adapt_pool", k, rep,
             e_ev2, cp_.accept(s_ev2), cp_.lambda_hat, n_pool,
             int(idx_cal.size + s_src2.size),
             {"ess": float(effective_sample_size(w_cat)),
              "w_max": float(np.max(w_cat))})
        order = np.argsort(s_ev2, kind="mergesort")
        lam_o = oracle_lambda(s_ev2[order], e_ev2[order], np.ones(s_ev2.size, bool),
                              alpha, MIN_ACCEPT)
        emit(rows, "D2", "transfer", unit, clf_kind, alpha, "oracle_adapt", k, rep,
             e_ev2, s_ev2 <= lam_o, lam_o, n_pool, 0)
    stub = unit.replace(":", "_").replace("->", "_to_")
    norm_diff = np.nan
    base_prob_diff = np.nan
    try:
        ref = np.load(os.path.join(PRED_DIR, "%s__%s.npz" % (stub, clf_kind)),
                      allow_pickle=True)
        norm_diff = float(np.max(np.abs(Xte - np.asarray(ref["test_X"],
                                                         dtype=np.float64))))
        base_prob_diff = float(np.max(np.abs(p_te0 - np.asarray(ref["test_probs"],
                                                               dtype=np.float64))))
    except Exception:
        pass
    diag = {"unit": unit, "clf": clf_kind, "n_rows": len(rows),
            "norm_diff": norm_diff, "base_prob_diff": base_prob_diff}
    return rows, diag

# ------------------------------------------------------------------- analysis
def agg_detail(detail, keys):
    micro = (detail.groupby(keys)
             .agg(n_rows=("covered", "size"),
                  mean_accept=("accept_rate", "mean"),
                  coverage_all=("covered", "mean"),
                  coverage_nonvac=("covered_nonvac", "mean"),
                  nonvac_frac=("empty", lambda x: 1.0 - float(np.mean(x))),
                  mean_risk=("selective_risk", "mean"),
                  mean_n_accepted=("n_accepted", "mean"),
                  mean_n_eval=("n_eval", "mean"),
                  mean_strata=("n_strata", "mean"),
                  mean_unfitted=("n_unfitted", "mean"),
                  mean_empty_strata=("n_empty_strata", "mean"),
                  mean_ess=("ess", "mean"),
                  n_units=("unit", "nunique"))
             .reset_index())
    pu = (detail.groupby(keys + ["unit"])
          .agg(u_cov=("covered", "mean"), u_cov_nv=("covered_nonvac", "mean"),
               u_acc=("accept_rate", "mean"))
          .reset_index())
    macro = (pu.groupby(keys)
             .agg(macro_coverage=("u_cov", "mean"),
                  macro_coverage_nonvac=("u_cov_nv", "mean"),
                  macro_accept=("u_acc", "mean"))
             .reset_index())
    out = micro.merge(macro, on=keys)
    den = out.nonvac_frac.to_numpy(dtype=float)
    num = out.coverage_nonvac.to_numpy(dtype=float)
    out["coverage_nonvac_cond"] = np.where(den > 0,
                                           num / np.where(den > 0, den, 1.0),
                                           np.nan)
    return out


def part_c(detail, keys):
    tot = (detail.groupby(keys)
           .agg(tot_accepted=("n_accepted", "sum"), tot_wrong=("n_wrong", "sum"),
                n_rows=("n_accepted", "size"), n_empty=("empty", "sum"))
           .reset_index())
    tot["risk_all"] = tot.tot_wrong / np.maximum(tot.tot_accepted, 1)
    tot["risk_upper"] = [cp_upper(w, n) for w, n in zip(tot.tot_wrong, tot.tot_accepted)]
    sub = detail[detail["empty"] == 0]
    if len(sub):
        ov = (sub.assign(over=(sub.selective_risk > sub.alpha).astype(int))
              .groupby(keys)["over"].mean().rename("frac_nonempty_over_budget")
              .reset_index())
        tot = tot.merge(ov, on=keys, how="left")
    else:
        tot["frac_nonempty_over_budget"] = np.nan
    return tot


def pair_stats(df, arm_a, arm_b, join_on, value, n_boot=BOOT, seed=BOOT_SEED):
    a = df.loc[df.arm == arm_a, join_on + [value]].rename(columns={value: "va"})
    b = df.loc[df.arm == arm_b, join_on + [value]].rename(columns={value: "vb"})
    m = a.merge(b, on=join_on, how="inner")
    if not len(m):
        return {"arm": arm_a, "ref": arm_b, "value": value, "n_pairs": 0,
                "n_units": 0, "mean_diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "excludes_zero": 0}
    m["d"] = m.va - m.vb
    per_unit = m.groupby("unit")["d"].mean()
    arr = per_unit.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"arm": arm_a, "ref": arm_b, "value": value, "n_pairs": int(len(m)),
            "n_units": int(arr.size), "mean_diff": float(arr.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": int(lo > 0 or hi < 0)}


def ladder_readout(summ, family, alphas, value="coverage_nonvac"):
    out = []
    for arm in sorted(summ.arm.unique()):
        for alpha in alphas:
            g = summ[(summ.arm == arm) & np.isclose(summ.alpha, alpha)].sort_values("k")
            if not len(g):
                continue
            ok = g[(g[value] >= NOMINAL) & (g.nonvac_frac > 0)]
            okg = g[(g[value] >= NOMINAL) & (g.nonvac_frac > 0)
                    & (g.mean_accept >= USEFUL_ACCEPT)]
            okc = g[(g.coverage_nonvac_cond >= NOMINAL) & (g.nonvac_frac > 0)]
            oka = g[(g.coverage_all >= NOMINAL) & (g.nonvac_frac > 0)]
            i = int(np.argmax(g.mean_accept.to_numpy()))
            at200 = g[g.k == 200]
            ka = int(oka.k.iloc[0]) if len(oka) else -1
            kn = int(ok.k.iloc[0]) if len(ok) else -1
            kc = int(okc.k.iloc[0]) if len(okc) else -1
            out.append({
                "family": family, "arm": arm, "alpha": float(alpha),
                "k_first_nominal_all": ka,
                "k_first_nominal_nonvac": kn,
                "k_first_nominal_useful": (int(okg.k.iloc[0]) if len(okg) else -1),
                "k_first_nominal_cond": kc,
                "accept_at_all_k": (float(g.loc[g.k == ka, "mean_accept"].iloc[0])
                                    if ka > 0 else np.nan),
                "accept_at_nonvac_k": (float(g.loc[g.k == kn, "mean_accept"].iloc[0])
                                       if kn > 0 else np.nan),
                "accept_at_cond_k": (float(g.loc[g.k == kc, "mean_accept"].iloc[0])
                                     if kc > 0 else np.nan),
                "best_accept": float(g.mean_accept.iloc[i]),
                "best_accept_k": int(g.k.iloc[i]),
                "accept_at_200": (float(at200.mean_accept.iloc[0]) if len(at200) else np.nan),
                "cov_nonvac_at_200": (float(at200[value].iloc[0]) if len(at200) else np.nan),
                "risk_upper_at_200": (float(at200.risk_upper.iloc[0]) if len(at200) else np.nan),
            })
    return pd.DataFrame(out)

# ------------------------------------------- independent recomputation (checks)
def indep_within_plain(alphas):
    """A second, straight-line implementation of the plain within certificate."""
    recs = []
    for fname in sorted(f for f in os.listdir(PRED_DIR) if f.startswith("within_")):
        d = np.load(os.path.join(PRED_DIR, fname), allow_pickle=True)
        s_te, e_te = se_of(d["test_probs"], d["test_y"])
        s_ca, e_ca = se_of(d["cal_probs"], d["cal_y"])
        for c in range(d["cal_rep"].shape[1]):
            m = d["cal_rep"][:, c]
            for a in alphas:
                ctl = C.SelectiveRiskController(a, DELTA, min_accept=MIN_ACCEPT
                                                ).fit(s_ca[m], e_ca[m])
                acc = ctl.accept(s_te)
                n = int(acc.sum())
                risk = float(e_te[acc].mean()) if n else float("nan")
                recs.append({"alpha": float(a), "accept_rate": n / float(s_te.size),
                             "covered": int((not np.isfinite(risk)) or risk <= a),
                             "covered_nonvac": int(np.isfinite(risk) and risk <= a
                                                   and n > 0)})
    df = pd.DataFrame(recs)
    g = df.groupby("alpha").agg(mean_accept=("accept_rate", "mean"),
                                coverage_all=("covered", "mean"),
                                coverage_nonvac=("covered_nonvac", "mean"),
                                n_rows=("covered", "size"))
    return g.reset_index()


def indep_transfer(files, alphas, ks, reps):
    """Second implementation of the plain and oracle transfer arms."""
    recs = []
    for fname in files:
        family, unit, clf = parse_name(fname)
        d = np.load(os.path.join(PRED_DIR, fname), allow_pickle=True)
        s_all, e_all = se_of(d["test_probs"], d["test_y"])
        n_pool = s_all.size
        for rep in range(reps):
            for k in ks:
                pick = r20_pick(unit, clf, rep, k, n_pool)
                keep = np.ones(n_pool, dtype=bool)
                keep[pick] = False
                ev = np.nonzero(keep)[0]
                s_ev, e_ev = s_all[ev], e_all[ev]
                order = np.argsort(s_ev, kind="mergesort")
                lam_o = oracle_lambda(s_ev[order], e_ev[order],
                                      np.ones(s_ev.size, bool), alphas[1], MIN_ACCEPT)
                for arm, mask, lam in (
                        ("plain", None, None),
                        ("oracle", s_ev <= lam_o, lam_o)):
                    if arm == "plain":
                        ctl = C.SelectiveRiskController(alphas[1], DELTA,
                                                        min_accept=MIN_ACCEPT
                                                        ).fit(s_all[pick], e_all[pick])
                        mask, lam = ctl.accept(s_ev), ctl.lambda_hat
                    n = int(np.asarray(mask, dtype=bool).sum())
                    risk = float(e_ev[mask].mean()) if n else float("nan")
                    recs.append({"unit": unit, "clf": clf, "alpha": alphas[1], "k": k,
                                 "rep": rep, "arm": arm,
                                 "accept_rate": n / float(ev.size),
                                 "covered": int((not np.isfinite(risk))
                                                or risk <= alphas[1]),
                                 "covered_nonvac": int(np.isfinite(risk)
                                                       and risk <= alphas[1] and n > 0)})
    df = pd.DataFrame(recs)
    g = (df.groupby(["arm", "k"])
         .agg(mean_accept=("accept_rate", "mean"),
              coverage_all=("covered", "mean"),
              coverage_nonvac=("covered_nonvac", "mean"),
              n_rows=("covered", "size")).reset_index())
    return g


def maxabs(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x))) if x.size else float("nan")


def chk(name, ok, detail):
    return {"check": name, "result": ("PASS" if ok else "FAIL"), "detail": detail}


# ------------------------------------------------------------------- reporting
def build_report(res):
    L = []

    def w(s=""):
        L.append(s)

    w("=" * 100)
    w("r31 -- targeted repair of the conformal failures under cross-field-of-view")
    w("       splitting (within) and cross-tissue transfer (transfer)")
    w("=" * 100)
    w("")
    w("scope  : 100 within unit-folds (3 classifiers) + 60 transfer unit-folds")
    w("         (3 classifiers); score s = 1 - max class proba; candidate set")
    w("         _grid_indices(n, min_accept=25, grid_size=120); delta = 0.10 so")
    w("         nominal coverage 1 - delta = 0.90; alphas %s." % (list(ALPHAS),))
    w("seed   : deterministic; r20's draw recipe reproduced bit for bit")
    w("         (default_rng([20250926 + crc32(unit|clf), rep, k, 0]).choice)")
    w("metric : covered = (realised risk among accepted cells <= alpha) or the")
    w("         accepted set is empty; covered_nonvac restricts it to non-empty")
    w("         sets, i.e. P(covered and non-empty).  coverage_nonvac_cond is the")
    w("         manuscript's Table-1 convention, P(covered | non-empty) =")
    w("         coverage_nonvac / nonvac_frac; the two differ only through the")
    w("         vacuity fraction, which is printed beside both.  acceptance =")
    w("         accepted cells / evaluation batch.")
    w("         The certificate's actual nominal guarantee is on coverage_all")
    w("         (outside the empty sets it only asks P(covered) >= 1 - delta,")
    w("         because an empty accepted set is a conservative success), so the")
    w("         primary readout is coverage_all >= 0.90; coverage_nonvac >= 0.90")
    w("         is a strictly stronger ask, and coverage_nonvac_cond is the")
    w("         manuscript's reporting convention.  Every readout below is given")
    w("         under all three with acceptance beside it, so a 'reached' entry at")
    w("         near-zero acceptance is visibly vacuous rather than a repair.")
    w("")
    w("A NOTE ON VACUITY, WHICH DRIVES EVERY PART BELOW")
    w("  The stored ladder says transfer 'recovers' at k = 500 because")
    w("  coverage_all returns to 1.000.  It does so vacuously: on the same runs")
    w("  the certificate accepts 0.5% of the batch at k = 500 and the non-vacuous")
    w("  coverage there is 0.019.  Every table therefore carries acceptance and")
    w("  coverage_nonvac next to coverage_all, and the label-cost readout is")
    w("  'smallest k with coverage >= 0.90 AND a non-empty accepted set', under")
    w("  all three conventions, with the acceptance at that k printed beside it.")
    w("")

    aw = res["within_summary"]
    bw = res["transfer_summary"]
    lad = res["ladder"]
    w("-" * 100)
    w("HEADLINE")
    w("-" * 100)
    pw = aw[aw.arm == "plain"]
    w("within (k = 0, the stored splits), plain certificate:")
    w(pw[["alpha", "coverage_all", "coverage_nonvac", "coverage_nonvac_cond",
          "mean_accept"]].round(4).to_string(index=False))
    oka = aw[aw.coverage_all >= NOMINAL]
    okw = aw[(aw.coverage_nonvac >= NOMINAL) & (aw.nonvac_frac > 0)]
    okc = aw[(aw.coverage_nonvac_cond >= NOMINAL) & (aw.nonvac_frac > 0)]
    for alpha in ALPHAS:
        sa = oka[np.isclose(oka.alpha, alpha)].sort_values("mean_accept",
                                                           ascending=False)
        s = okw[np.isclose(okw.alpha, alpha)].sort_values("mean_accept", ascending=False)
        sc = okc[np.isclose(okc.alpha, alpha)].sort_values("mean_accept",
                                                           ascending=False)
        w("  alpha=%.2f : arms holding the nominal guarantee coverage_all >= %.2f :"
          " %s" % (alpha, NOMINAL,
                   (", ".join("%s (cov_all %.3f, acc %.3f)"
                              % (r.arm, r.coverage_all, r.mean_accept)
                              for r in sa.itertuples()) if len(sa) else "NONE")))
        w("  alpha=%.2f : arms reaching non-vacuous nominal coverage : %s"
          % (alpha, (", ".join("%s (cov %.3f, acc %.3f)"
                               % (r.arm, r.coverage_nonvac, r.mean_accept)
                               for r in s.itertuples()) if len(s) else "NONE")))
        w("             manuscript convention coverage_nonvac_cond >= %.2f : %s"
          % (NOMINAL, (", ".join("%s (cond %.3f, acc %.3f)"
                                 % (r.arm, r.coverage_nonvac_cond, r.mean_accept)
                                 for r in sc.itertuples()) if len(sc) else "NONE")))
    zw = sorted(set(aw[(aw.mean_accept <= 1e-12) & np.isclose(aw.alpha, 0.10)].arm))
    w("  arms that answered nothing at alpha=0.10 (acceptance 0): %s" % (zw,))
    b10 = bw[np.isclose(bw.alpha, 0.10)]
    i = int(np.argmax(b10.mean_accept.to_numpy()))
    w("transfer (alpha = 0.10), best acceptance over the whole grid:")
    w("  %s at k=%d : accept %.4f, non-vacuous coverage %.3f, pooled risk upper %.3f"
      % (b10.arm.iloc[i], int(b10.k.iloc[i]), b10.mean_accept.iloc[i],
         b10.coverage_nonvac.iloc[i], b10.risk_upper.iloc[i]))
    l10 = lad[np.isclose(lad.alpha, 0.10)]
    gota = l10[l10.k_first_nominal_all > 0]
    got = l10[l10.k_first_nominal_nonvac > 0]
    notgot = l10[l10.k_first_nominal_nonvac <= 0]
    gotc = l10[l10.k_first_nominal_cond > 0]
    notgotc = l10[l10.k_first_nominal_cond <= 0]
    w("  smallest k for each readout, every one requiring a non-empty accepted set")
    w("  (acceptance at that k in brackets -- this is what makes a 'reached' entry")
    w("  vacuous or not):")
    w("    nominal guarantee coverage_all >= %.2f : %s"
      % (NOMINAL, (", ".join("%s@%d (acc %.4f)"
                             % (r.arm, r.k_first_nominal_all, r.accept_at_all_k)
                             for r in gota.itertuples()) if len(gota) else "NONE")))
    w("    unconditional P(covered and non-empty) >= %.2f : %s"
      % (NOMINAL, (", ".join("%s@%d (acc %.4f)"
                             % (r.arm, r.k_first_nominal_nonvac, r.accept_at_nonvac_k)
                             for r in got.itertuples()) if len(got) else "NONE")))
    w("      never : %s" % (", ".join(notgot.arm.tolist()) if len(notgot) else "none"))
    w("    manuscript convention P(covered | non-empty) >= %.2f : %s"
      % (NOMINAL, (", ".join("%s@%d (acc %.4f)"
                             % (r.arm, r.k_first_nominal_cond, r.accept_at_cond_k)
                             for r in gotc.itertuples()) if len(gotc) else "NONE")))
    w("      never : %s" % (", ".join(notgotc.arm.tolist()) if len(notgotc)
                             else "none"))
    zt = sorted(set(bw[(bw.mean_accept <= 1e-12) & np.isclose(bw.alpha, 0.10)].arm))
    w("  arms that answered nothing at alpha=0.10 (acceptance 0): %s" % (zt,))
    w("")

    w("-" * 100)
    w("PART A -- within (cross-field-of-view): coverage by arm and alpha")
    w("-" * 100)
    a = res["within_summary"]
    cols = ["arm", "alpha", "coverage_all", "coverage_nonvac", "coverage_nonvac_cond",
            "macro_coverage_nonvac", "nonvac_frac", "mean_accept",
            "mean_empty_strata", "mean_strata", "mean_ess", "n_rows"]
    w(a[cols].round(4).to_string(index=False))
    w("")
    w("  mean_strata / mean_empty_strata: strata considered on the deployment side")
    w("  (predicted classes, or field-of-view assignments) and how many of them")
    w("  ended up with zero accepted cells.  mean_ess is the effective sample size")
    w("  of the importance weights (NaN for unweighted arms).")
    w("")
    for alpha in ALPHAS:
        sub = a[np.isclose(a.alpha, alpha)]
        oka = sub[sub.coverage_all >= NOMINAL]
        ok = sub[(sub.coverage_nonvac >= NOMINAL) & (sub.nonvac_frac > 0)]
        okc = sub[(sub.coverage_nonvac_cond >= NOMINAL) & (sub.nonvac_frac > 0)]
        w("  alpha=%.2f : nominal guarantee coverage_all >= %.2f : %s"
          % (alpha, NOMINAL,
             (", ".join("%s (%.4f, accept %.4f)" % (r.arm, r.coverage_all,
                                                    r.mean_accept)
                        for r in oka.itertuples()) if len(oka) else "none")))
        w("             unconditional non-vacuous >= %.2f : %s"
          % (NOMINAL,
             (", ".join("%s (%.4f, accept %.4f)" % (r.arm, r.coverage_nonvac,
                                                    r.mean_accept)
                        for r in ok.itertuples()) if len(ok) else "none")))
        w("             manuscript convention >= %.2f : %s"
          % (NOMINAL,
             (", ".join("%s (%.4f, accept %.4f)" % (r.arm, r.coverage_nonvac_cond,
                                                    r.mean_accept)
                        for r in okc.itertuples()) if len(okc) else "none")))
    w("")
    if len(res["within_ladder"]):
        w("  within label-cost reference -- the plain certificate recalibrated on")
        w("  k target labels drawn exactly as in part B (the k = 0 row is 'plain'")
        w("  in the table above; these are the tgt_k* rows of the detail CSV):")
        wl = res["within_ladder"]
        w(wl[["arm", "alpha", "k_first_nominal_all", "k_first_nominal_nonvac",
              "k_first_nominal_cond", "accept_at_all_k", "best_accept",
              "best_accept_k"]].round(4).to_string(index=False))
        w("  -1 = never reached the nominal level at any k in %s."
          % (list(WITHIN_TGT_KS),))
        w("")
    if len(res["paired_transfer"]):
        w("  PAIRED DIFFERENCES vs plain on transfer (same 95% CI recipe, paired at")
        w("  the (unit, classifier, rep, k) level; a positive mean_diff with")
        w("  excludes_zero = 1 is the only kind of transfer win claimed).")
        w("  value = covered       : vacuous-inclusive coverage (an empty accepted")
        w("                          set counts as covered)")
        w("  value = covered_nonvac: P(covered and non-empty)")
        w("  value = accept_rate   : fraction of the remainder accepted")
        w(res["paired_transfer"].round(5).to_string(index=False))
        w("")
    w("  PAIRED DIFFERENCES vs plain (95% CI over the 20 units, 4000 bootstrap")
    w("  resamples of units; 'excludes_zero' marks a difference whose CI excludes")
    w("  zero, i.e. the only kind of win claimed anywhere in this report):")
    p = res["paired_within"]
    w(p.round(5).to_string(index=False))
    w("")

    w("-" * 100)
    w("PART B -- transfer: the label-cost ladder, arm by arm")
    w("-" * 100)
    b = res["transfer_summary"]
    for alpha in ALPHAS:
        w("alpha = %.2f" % alpha)
        sub = b[np.isclose(b.alpha, alpha)].sort_values(["arm", "k"])
        w(sub[["arm", "k", "coverage_all", "coverage_nonvac", "coverage_nonvac_cond",
               "nonvac_frac", "mean_accept", "risk_all", "risk_upper"]].round(4)
          .to_string(index=False))
        w("")
    w("  risk_all / risk_upper: realised selective risk among the accepted cells")
    w("  pooled over units and classifiers, and its one-sided 95% Clopper-Pearson")
    w("  upper limit.  A valid certificate keeps risk_upper <= alpha.")
    w("  src / w_src are emitted once per stored calibration repeat (r20's")
    w("  protocol: one stored repeat mask -- 2500 of 4101 source cells for the")
    w("  Melanoma-source pairs -- not the union), so k = 0 averages over the six")
    w("  repeats exactly as the manuscript's Table-1 transfer 'crc' anchor does.")
    w("  pool_iw instead takes the union of the stored source calibration cells.")
    w("  prop_mond / unc_iw / twostage_* choose which cells to label OUT of the")
    w("  whole target pool (proportional to predicted class, highest predicted")
    w("  entropy, and a uniform pilot followed by a risk-concentrated allocation),")
    w("  so their evaluation remainder is not plain's; the pairing is at the")
    w("  (unit, classifier, rep, k) level, which is what the PAIRED section and")
    w("  the B6 duplicate guard use.")
    w("")
    w("SMALLEST k PER ARM  (coverage >= 0.90 with a non-empty accepted set)")
    w("k_first_nominal_all    : the nominal guarantee (coverage_all; an empty set")
    w("                         counts as covered)")
    w("k_first_nominal_nonvac : unconditional P(covered and non-empty)")
    w("k_first_nominal_useful : as nonvac, but also acceptance >= 0.10")
    w("k_first_nominal_cond   : the manuscript's P(covered | non-empty)")
    w("accept_at_*_k          : acceptance at that k, so a tiny value exposes a")
    w("                         vacuously 'reached' level")
    w(res["ladder"].round(4).to_string(index=False))
    w("")
    w("  k_first_nominal_useful additionally requires acceptance >= 0.10 of the")
    w("  batch (the r20 usefulness floor).  -1 means the arm never got there.")
    w("")

    w("-" * 100)
    w("PART C -- realised risk among accepted cells vs the budget")
    w("-" * 100)
    c = res["partC"]
    cw_ = c[c.family == "within"]
    w("within (pooled over units, classifiers and repeats):")
    w(cw_[["arm", "alpha", "tot_accepted", "tot_wrong", "risk_all", "risk_upper",
           "frac_nonempty_over_budget", "n_empty"]].round(4).to_string(index=False))
    w("")
    ct = c[(c.family == "transfer") & np.isclose(c.alpha, 0.10)]
    w("transfer, alpha = 0.10:")
    w(ct[["arm", "k", "tot_accepted", "tot_wrong", "risk_all", "risk_upper",
          "frac_nonempty_over_budget", "n_empty"]].round(4).to_string(index=False))
    w("")
    w("  frac_nonempty_over_budget is the fraction of non-empty evaluations whose")
    w("  realised risk exceeded the budget.  If that is high while the accepted")
    w("  sets are large, the labels are wrong, not merely the bound loose.") 
    w("")

    w("-" * 100)
    w("PART D -- model adaptation, reproduced from experiments/r14b_targetadapt.csv")
    w("-" * 100)
    if res["partD"] is not None:
        w("filter: %s" % res["partD_info"]["filter"])
        w("rows in file: %d ; tgt_k(full 2000 calibration, r14_targetcal_summary) "
          "mean_accept = %.4f"
          % (res["partD_info"]["n_rows_total"],
             res["partD_info"]["tgt_k_full_2000_mean_accept"]))
        w(res["partD"].round(4).to_string(index=False))
        by = {r.arm: r for r in res["partD"].itertuples()}
        if "adapt_k" in by and "tgt_k_same" in by:
            w("")
            nrep = by["adapt_k"].n_rows // max(by["adapt_k"].n_units
                                               * by["adapt_k"].n_clf, 1)
            w("  the stored claim: half of a %d-label budget spent refitting the"
              % STACK_K)
            w("  model (adapt_k, %d rows = %d unit-folds x %d classifiers x %d reps)"
              % (by["adapt_k"].n_rows, by["adapt_k"].n_units, by["adapt_k"].n_clf,
                 nrep))
            w("  reaches acceptance %.4f against %.4f for the same budget spent on"
              % (by["adapt_k"].mean_accept, by["tgt_k_same"].mean_accept))
            w("  calibration of the base model (tgt_k_same), i.e. x%.2f, while the"
              % (by["adapt_k"].mean_accept / by["tgt_k_same"].mean_accept))
            w("  oracle moves %.4f -> %.4f (x%.2f)."
              % (by["oracle_base"].mean_accept, by["oracle_adapt"].mean_accept,
                 by["oracle_adapt"].mean_accept / by["oracle_base"].mean_accept))
            if np.isfinite(by["adapt_k"].ratio_full_2000_calibration):
                w("  The manuscript line 0.039 -> 0.355 (x9.2) pairs the r14b adapt")
                w("  arm with the r14 tgt_k arm that spends all 2000 labels on")
                w("  calibration (0.0388); that pairing is reproduced in the")
                w("  ratio_full_2000_calibration column and is reported as a")
                w("  cross-run comparison, not as a same-split one.")
    else:
        w("r14b_targetadapt.csv not found -- part D not reproduced.")
    w("")

    w("-" * 100)
    w("PART D2 -- do the new calibration arms stack on model adaptation?")
    w("-" * 100)
    if res["stack"] is not None and len(res["stack"]):
        w("scope note: this is a NEW run (%d transfer unit-folds x logreg only,"
          % res["stack_note"]["n_units"])
        w("alpha = %.2f, k = %d target labels, %d repeats), not a reuse of the"
          % (STACK_ALPHA, STACK_K, STACK_REPS))
        w("stored predictions: adaptation requires refitting, so it is re-run here")
        w("with r14b's own seed recipe (100000*rep + crc32(unit) + 977).")
        w(res["stack"].round(4).to_string(index=False))
        w("")
        w("  adapt_plain re-runs r14b's adapt_k recipe here; adapt_mond / adapt_iw")
        w("  / adapt_pool apply this script's remedies to the re-fitted scores.  A")
        w("  remedy 'stacks' only if the paired difference against adapt_plain")
        w("  excludes zero.")
        stored = None
        if res["partD"] is not None and (res["partD"].arm == "adapt_k").any():
            stored = float(res["partD"].loc[res["partD"].arm == "adapt_k",
                                           "mean_accept"].iloc[0])
        mine = float(res["stack"].loc[res["stack"].arm == "adapt_plain",
                                      "mean_accept"].iloc[0])
        if stored is None:
            w("  HONEST CAVEAT: the stored adapt_k reference is unavailable, so the")
            w("  re-run cannot be compared with it.  Only the within-run paired")
            w("  differences in the D4 check are usable.")
        else:
            w("  HONEST CAVEAT: this re-run does NOT reproduce the stored adapt_k")
            w("  acceptance (stored %.4f vs re-run %.4f).  The base classifier refit"
              % (stored, mine))
            w("  is not bit-reproducible in this environment: logreg's iteration")
            w("  count follows the BLAS thread pool (see D3b for the measured")
            w("  drift), and the adapted model amplifies it.  The absolute numbers")
            w("  in this table are therefore not comparable with PART D.  Only the")
            w("  within-run paired differences in the D4 check below are usable;")
            w("  PART D is the reproduction of the stored result.")
    else:
        w("not run (--no-stack).")
    w("")

    w("-" * 100)
    w("CHECKS -- every headline number recomputed independently")
    w("-" * 100)
    ck = pd.DataFrame(res["checks"])
    w(ck.to_string(index=False))
    n_fail = int((ck.result == "FAIL").sum())
    w("")
    w("  %d checks, %d PASS, %d FAIL." % (len(ck), int((ck.result == "PASS").sum()), n_fail))
    if n_fail:
        w("  FAILING CHECKS -- the numbers above must not be quoted until these")
        w("  are resolved; the script exits non-zero.")
    w("")
    return L

def _worker_init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default=",".join("%.2f" % a for a in ALPHAS))
    ap.add_argument("--ks", default=",".join(str(k) for k in K_ABS))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="", help="substring filter on the file name")
    ap.add_argument("--limit", type=int, default=0, help="keep only N files per family")
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--out", default="r31")
    args = ap.parse_args()
    alphas = tuple(float(a) for a in args.alphas.split(",") if a)
    ks = tuple(int(k) for k in args.ks.split(",") if k)
    njobs = max(1, min(int(args.jobs), 4))

    all_files = sorted(os.listdir(PRED_DIR))
    within_files = [f for f in all_files if parse_name(f)[0] == "within"]
    transfer_files = [f for f in all_files if parse_name(f)[0] == "transfer"]
    if args.only:
        within_files = [f for f in within_files if args.only in f]
        transfer_files = [f for f in transfer_files if args.only in f]
    if args.limit:
        within_files = within_files[:args.limit]
        transfer_files = transfer_files[:args.limit]
    log("within files=%d transfer files=%d alphas=%s ks=%s reps=%d jobs=%d"
        % (len(within_files), len(transfer_files), list(alphas), list(ks),
           args.reps, njobs))

    t0 = time.time()
    wrows = []
    if within_files:
        jobs = [(f, alphas, WITHIN_TGT_KS) for f in within_files]
        if njobs > 1:
            with mp.Pool(njobs, initializer=_worker_init) as pool:
                for i, rws in enumerate(pool.imap(within_file, jobs, chunksize=1), 1):
                    wrows.extend(rws)
                    if i % 10 == 0 or i == len(jobs):
                        log("  within %d/%d" % (i, len(jobs)))
        else:
            for j in jobs:
                wrows.extend(within_file(j))
    within_detail = pd.DataFrame(wrows) if wrows else empty_detail()
    log("part A done: %d rows (%.1fs)" % (len(within_detail), time.time() - t0))

    t1 = time.time()
    trows = []
    if transfer_files:
        jobs = [(f, alphas, ks, args.reps) for f in transfer_files]
        if njobs > 1:
            with mp.Pool(njobs, initializer=_worker_init) as pool:
                for i, rws in enumerate(pool.imap(transfer_file, jobs, chunksize=1), 1):
                    trows.extend(rws)
                    if i % 10 == 0 or i == len(jobs):
                        log("  transfer %d/%d  (%.1fs)" % (i, len(jobs), time.time() - t1))
        else:
            for j in jobs:
                trows.extend(transfer_file(j))
    transfer_detail = pd.DataFrame(trows) if trows else empty_detail()
    log("part B done: %d rows (%.1fs)" % (len(transfer_detail), time.time() - t1))

    # ------------------------------------------------------------- part A agg
    a = within_detail[within_detail.part == "A"].copy()
    within_summary = agg_detail(a, ["arm", "alpha"])
    within_summary["arm"] = pd.Categorical(within_summary.arm, list(ARM_ORDER_A),
                                          ordered=True)
    within_summary = within_summary.sort_values(["arm", "alpha"])

    aref = within_detail[within_detail.part == "A_ref"].copy()
    if len(aref):
        within_ref = agg_detail(aref, ["arm", "alpha"])
        cbw = part_c(aref, ["arm", "alpha"])
        within_ref = within_ref.merge(cbw, on=["arm", "alpha"], how="left")
        within_ref["k"] = [int(a.split("tgt_k")[1]) for a in within_ref.arm]
        within_ref = within_ref.sort_values(["arm", "alpha"]).reset_index(drop=True)
        within_ladder = ladder_readout(within_ref, "within", alphas)
    else:
        within_ref = pd.DataFrame()
        within_ladder = pd.DataFrame()

    # ------------------------------------------------------------- part B agg
    b = transfer_detail[transfer_detail.part == "B"].copy()
    transfer_summary = agg_detail(b, ["arm", "alpha", "k"])
    cb = part_c(b, ["arm", "alpha", "k"])
    transfer_summary = transfer_summary.merge(cb, on=["arm", "alpha", "k"], how="left")
    ladder = ladder_readout(transfer_summary, "transfer", alphas)
    ladder = ladder.sort_values(["arm", "alpha"]).reset_index(drop=True)

    # ------------------------------------------------------------- part C
    c_within = part_c(a, ["arm", "alpha"])
    c_within.insert(0, "family", "within")
    c_within["k"] = np.nan
    c_tr = part_c(b, ["arm", "alpha", "k"])
    c_tr.insert(0, "family", "transfer")
    partC = pd.concat([c_within, c_tr], ignore_index=True, sort=False)

    # ------------------------------------------------------------- paired
    pairs = []
    join = ["unit", "clf", "alpha", "k", "rep"]
    for arm in ARM_ORDER_A:
        if arm == "plain":
            continue
        for val in ("covered_nonvac", "accept_rate"):
            pairs.append(pair_stats(a, arm, "plain", join, val))
    paired_within = pd.DataFrame(pairs)

    pairs_b = []
    join_b = ["unit", "clf", "alpha", "k", "rep"]
    for arm in ARM_ORDER_B + ("oracle",):
        if arm == "plain":
            continue
        for val in ("covered", "covered_nonvac", "accept_rate"):
            pairs_b.append(pair_stats(b, arm, "plain", join_b, val))
    paired_transfer = pd.DataFrame(pairs_b)

    # ------------------------------------------------------------- part D
    partD, partD_info = part_d()

    # ------------------------------------------------------------- part D2
    stack = None
    stack_note = None
    stack_diag = None
    stack_detail = None
    if not args.no_stack and transfer_files:
        try:
            import r02_run_experiments as R2
            cohorts = set()
            for s, t in R2.TRANSFER_PAIRS:
                cohorts.add(s)
                cohorts.add(t)
            cohort = {}
            for ds in sorted(cohorts):
                cohort[ds] = R2.prepare_cohort(ds)
                log("  loaded %s (%d cells)" % (ds, len(cohort[ds][0])))
            cfg = {}
            jobs = []
            for fname in transfer_files:
                family, unit, clf = parse_name(fname)
                if clf != "logreg":
                    continue
                hit = None
                for s, t in R2.TRANSFER_PAIRS:
                    pref = "%s_to_%s_fold" % (s, t)
                    if unit.startswith(pref):
                        hit = (s, t, int(unit[len(pref):]))
                        break
                if hit is None:
                    continue
                jobs.append((hit[0], hit[1], hit[2], clf, STACK_ALPHA, STACK_K,
                             STACK_REPS, 0))
            srows, sdiag = [], []
            if jobs and njobs > 1:
                with mp.Pool(njobs, initializer=_stack_init,
                             initargs=(cohort, cfg)) as pool:
                    for i, (rws, dg) in enumerate(pool.imap(stack_unit, jobs,
                                                                      chunksize=1), 1):
                        srows.extend(rws)
                        if isinstance(dg, dict):
                            sdiag.append(dg)
                        if i % 10 == 0 or i == len(jobs):
                            log("  stack %d/%d" % (i, len(jobs)))
            elif jobs:
                _stack_init(cohort, cfg)
                for j in jobs:
                    rws, dg = stack_unit(j)
                    srows.extend(rws)
                    if isinstance(dg, dict):
                        sdiag.append(dg)
            sdf = pd.DataFrame(srows) if srows else empty_detail()
            if len(sdf):
                stack_detail = sdf
                stack = agg_detail(sdf, ["arm", "alpha", "k"])
                stack = stack.sort_values("arm").reset_index(drop=True)
                stack_note = {"n_units": int(sdf.unit.nunique()),
                              "n_rows": int(len(sdf))}
            ddf = pd.DataFrame(sdiag)
            if len(ddf) and "norm_diff" in ddf.columns:
                stack_diag = ddf
        except Exception as exc:
            import traceback
            log("part D2 failed: %s" % exc)
            log(traceback.format_exc())
            stack = None
            stack_note = None

    # ------------------------------------------------------------- write CSVs
    outs = {}
    outs["within_detail"] = within_detail
    outs["within_summary"] = within_summary
    outs["transfer_detail"] = transfer_detail
    outs["transfer_summary"] = transfer_summary
    outs["ladder_summary"] = ladder
    outs["partC"] = partC
    outs["paired"] = paired_within
    outs["paired_transfer"] = paired_transfer
    if len(within_ref):
        outs["within_ref_summary"] = within_ref
    if partD is not None:
        outs["partD"] = partD
    if stack is not None:
        outs["stack"] = stack
    if stack_detail is not None:
        outs["stack_detail"] = stack_detail
    if stack_diag is not None and "norm_diff" in stack_diag.columns:
        outs["stack_diag"] = stack_diag
    for k, v in outs.items():
        p = os.path.join(paths.RESULTS, "%s_%s.csv" % (args.out, k))
        v.to_csv(p, index=False)
        log("wrote %s (%d rows)" % (p, len(v)))

    # ------------------------------------------------------------- checks
    checks = []
    r03 = pd.read_csv(os.path.join(paths.RESULTS, "r03_summary.csv"))
    ref = r03[(r03.kind == "within") & (r03.method == "crc")]
    mine = within_summary[within_summary.arm == "plain"]
    jj = mine.merge(ref, on="alpha", suffixes=("", "_r03"))
    da = maxabs(jj.mean_accept.to_numpy(dtype=float)
                - jj.mean_accept_r03.to_numpy(dtype=float))
    dc = maxabs(jj.coverage_all.to_numpy(dtype=float)
                - jj.coverage.to_numpy(dtype=float))
    checks.append(chk("A1 plain-within reproduces r03_summary crc (mean_accept, coverage)",
                      bool(np.isfinite(da) and np.isfinite(dc)) and max(da, dc) < 1e-9,
                      "max|d accept| = %.3e, max|d coverage| = %.3e, n_rows %d vs 360"
                      % (da, dc, int(mine.n_rows.iloc[0]))))

    iw = indep_within_plain(alphas)
    m = mine.merge(iw, on="alpha", suffixes=("", "_i"))
    d1 = maxabs(m.mean_accept - m.mean_accept_i)
    d2 = maxabs(m.coverage_all - m.coverage_all_i)
    d3 = maxabs(m.coverage_nonvac - m.coverage_nonvac_i)
    checks.append(chk("A2 plain-within vs an independent straight-line reimplementation",
                      bool(np.isfinite(d1) and np.isfinite(d2) and np.isfinite(d3))
                      and max(d1, d2, d3) < 1e-12,
                      "max|d| accept %.2e coverage %.2e nonvac %.2e"
                      % (d1, d2, d3)))

    a05, a10, a20 = [float(within_summary.loc[(within_summary.arm == "plain")
                                              & np.isclose(within_summary.alpha, al),
                                              "coverage_all"].iloc[0])
                     for al in (0.05, 0.10, 0.20)]
    checks.append(chk("A4 plain-within nominal guarantee: 2 of 3 budgets violated",
                      round(a05, 4) == 0.8333 and round(a10, 4) == 0.8167
                      and round(a20, 4) == 0.9167
                      and (a05 < NOMINAL) and (a10 < NOMINAL) and (a20 >= NOMINAL),
                      "coverage_all %.4f / %.4f / %.4f at alpha 0.05 / 0.10 / 0.20"
                      % (a05, a10, a20)))

    msc = {"0.05": 0.751, "0.10": 0.764, "0.20": 0.899}
    got3, bad3 = [], []
    for al, exp in sorted(msc.items()):
        v = float(within_summary.loc[(within_summary.arm == "plain")
                                     & np.isclose(within_summary.alpha, float(al)),
                                     "coverage_nonvac_cond"].iloc[0])
        got3.append(v)
        if round(v, 3) != exp:
            bad3.append((al, v, exp))
    checks.append(chk("A3 plain-within non-vacuous coverage reproduces the manuscript"
                      " 0.751/0.764/0.899",
                      not bad3, "computed %s vs manuscript 0.751/0.764/0.899"
                      % (", ".join("%.4f" % v for v in got3))))

    try:
        r20 = pd.read_csv(os.path.join(paths.RESULTS, "r20_units.csv"),
                          usecols=["family", "unit", "clf", "alpha", "min_accept",
                                   "grid", "k", "rep", "arm", "n_accepted",
                                   "accept_rate", "lambda_hat"])
        rr = r20[(r20.family == "transfer") & (r20.grid == "abs")
                 & (r20.min_accept == MIN_ACCEPT) & (r20.arm == "tgt_k")
                 & (r20.k == 100) & np.isclose(r20.alpha, 0.10)]
        mm = b[(b.arm == "plain") & (b.k == 100) & np.isclose(b.alpha, 0.10)]
        j = rr.merge(mm, on=["unit", "clf", "alpha", "k", "rep"],
                     suffixes=("_r20", "_r31"))
        dacc = maxabs(j.accept_rate_r20 - j.accept_rate_r31)
        dnac = maxabs(j.n_accepted_r20 - j.n_accepted_r31)
        l20 = j.lambda_hat_r20.to_numpy(dtype=float)
        l31 = j.lambda_hat_r31.to_numpy(dtype=float)
        fin = np.isfinite(l20) & np.isfinite(l31)
        dlam = maxabs(l20[fin] - l31[fin])
        nlam = int(fin.sum())
        checks.append(chk("B1 plain-transfer k=100 reproduces r20_units tgt_k",
                          bool(len(rr)) and (len(j) == len(rr))
                          and max(dacc, dnac, dlam) < 1e-9,
                          "matched %d/%d ; max|d accept| %.2e, max|d n_acc| %.2e, "
                          "max|d lambda| %.2e (finite %d/%d)"
                          % (len(j), len(rr), dacc, dnac, dlam, nlam, len(j))))
    except Exception as exc:
        checks.append(chk("B1 plain-transfer k=100 reproduces r20_units tgt_k",
                          False, "could not run: %s" % exc))

    try:
        r20b = pd.read_csv(os.path.join(paths.RESULTS, "r20_units.csv"),
                           usecols=["family", "unit", "clf", "alpha", "min_accept",
                                    "grid", "k", "rep", "arm", "n_accepted",
                                    "accept_rate", "covered", "lambda_hat"])
        def _r20_sel(frame, arm, kk):
            return frame[(frame.family == "transfer") & (frame.grid == "abs")
                         & (frame.min_accept == MIN_ACCEPT) & (frame.arm == arm)
                         & (frame.k == kk) & np.isclose(frame.alpha, 0.10)]
        rr = _r20_sel(r20b, "src", 0)
        mm = b[(b.arm == "src") & (b.k == 0) & np.isclose(b.alpha, 0.10)]
        j = rr.merge(mm, on=["unit", "clf", "alpha", "k", "rep"],
                     suffixes=("_r20", "_r31"))
        dacc = maxabs(j.accept_rate_r20 - j.accept_rate_r31)
        dnac = maxabs(j.n_accepted_r20 - j.n_accepted_r31)
        l20 = j.lambda_hat_r20.to_numpy(dtype=float)
        l31 = j.lambda_hat_r31.to_numpy(dtype=float)
        fin = np.isfinite(l20) & np.isfinite(l31)
        dlam = maxabs(l20[fin] - l31[fin])
        ro = _r20_sel(r20b, "oracle", 200)
        mo = b[(b.arm == "oracle") & (b.k == 200) & np.isclose(b.alpha, 0.10)]
        jo = ro.merge(mo, on=["unit", "clf", "alpha", "k", "rep"],
                      suffixes=("_r20", "_r31"))
        doacc = maxabs(jo.accept_rate_r20 - jo.accept_rate_r31)
        docov = maxabs(jo.covered_r20 - jo.covered_r31)
        # r20 stores lambda_hat computed from float32 probabilities
        # (1 - cal_probs.max(1) with no float64 cast), so the threshold can
        # differ by one float32 epsilon (2.98e-08) while the accepted set is
        # bit-identical; dlam is therefore judged at 1e-6, all else at 1e-9.
        ok4 = (bool(len(rr)) and len(j) == len(rr) and bool(len(ro))
               and len(jo) == len(ro)
               and max(dacc, dnac, doacc, docov) < 1e-9 and dlam < 1e-6)
        checks.append(chk("B4 k=0 src (all stored repeats) and k=200 oracle"
                          " reproduce r20_units",
                          ok4,
                          "src matched %d/%d max|d| accept %.2e n_acc %.2e lambda"
                          " %.2e (float32 epsilon; r20 stores lambda from float32"
                          " scores) ; oracle matched %d/%d max|d| accept %.2e"
                          " covered %.2e"
                          % (len(j), len(rr), dacc, dnac, dlam, len(jo), len(ro),
                             doacc, docov)))
    except Exception as exc:
        checks.append(chk("B4 k=0 src and k=200 oracle reproduce r20_units",
                          False, "could not run: %s" % exc))

    try:
        act = ["prop_mond", "unc_iw", "twostage_iw", "twostage_nw"]
        sd = b[b.arm.isin(act) & b.sel_overlap.notna()]
        per = (sd.groupby("arm")["sel_overlap"]
                 .agg(["min", "max", "size"]).reset_index())
        ok6 = (len(per) == len(act) and float(per["max"].max()) < 1.0
               and float(per["min"].min()) >= 0.0)
        checks.append(chk("B6 the active-selection arms are not no-ops",
                          ok6,
                          "; ".join("%s overlap %.3f-%.3f (n=%d)"
                                    % (r["arm"], r["min"], r["max"], r["size"])
                                    for _, r in per.iterrows())))
    except Exception as exc:
        checks.append(chk("B6 the active-selection arms are not no-ops",
                          False, "could not run: %s" % exc))

    try:
        r16 = pd.read_csv(os.path.join(paths.RESULTS, "r16_review_budget.csv"))
        a16 = r16[(r16.family == "transfer") & (r16.method == "crc")
                  & np.isclose(r16.alpha, 0.10)]
        m16 = transfer_summary[(transfer_summary.arm == "src")
                               & (transfer_summary.k == 0)
                               & np.isclose(transfer_summary.alpha, 0.10)]
        da16 = abs(float(m16.mean_accept.iloc[0]) - float(a16.mean_accept.iloc[0]))
        dc16 = abs(float(m16.coverage_all.iloc[0]) - float(a16.coverage.iloc[0]))
        checks.append(chk("B5 k=0 src reproduces the r16 Table-1 transfer crc anchor",
                          bool(len(a16)) and bool(len(m16)) and max(da16, dc16) < 1e-9,
                          "accept %.4f vs %.4f, coverage %.4f vs %.4f"
                          % (float(m16.mean_accept.iloc[0]),
                             float(a16.mean_accept.iloc[0]),
                             float(m16.coverage_all.iloc[0]),
                             float(a16.coverage.iloc[0]))))
    except Exception as exc:
        checks.append(chk("B5 k=0 src reproduces the r16 Table-1 transfer crc anchor",
                          False, "could not run: %s" % exc))

    try:
        sub_files = transfer_files[:4]
        it = indep_transfer(sub_files, alphas, ks, args.reps)
        pairset = {(parse_name(f)[1], parse_name(f)[2]) for f in sub_files}
        dd = b[(b.arm.isin(["plain", "oracle"])) & np.isclose(b.alpha, alphas[1])]
        dd = dd[[(u, c) in pairset for u, c in zip(dd.unit, dd.clf)]]
        mm = (dd.groupby(["arm", "k"])
              .agg(mean_accept=("accept_rate", "mean"),
                   coverage_nonvac=("covered_nonvac", "mean")).reset_index())
        j = it.merge(mm, on=["arm", "k"], suffixes=("_i", ""))
        dacc = maxabs(j.mean_accept_i - j.mean_accept)
        dcov = maxabs(j.coverage_nonvac_i - j.coverage_nonvac)
        checks.append(chk("B2 plain+oracle transfer vs independent reimplementation (4 files)",
                          bool(len(j)) and max(dacc, dcov) < 1e-12,
                          "matched %d (arm,k) cells, max|d accept| %.2e, max|d nonvac| %.2e"
                          % (len(j), dacc, dcov)))
    except Exception as exc:
        checks.append(chk("B2 plain+oracle transfer vs independent reimplementation",
                          False, "could not run: %s" % exc))

    try:
        lad2 = []
        for arm in sorted(transfer_summary.arm.unique()):
            for al in alphas:
                g = transfer_summary[(transfer_summary.arm == arm)
                                     & np.isclose(transfer_summary.alpha, al)]
                g = g.sort_values("k")

                def _first(frame, col):
                    sel = frame[(frame[col] >= NOMINAL) & (frame.nonvac_frac > 0)]
                    return int(sel.k.iloc[0]) if len(sel) else -1

                lad2.append({"arm": arm, "alpha": al,
                             "k_all": _first(g, "coverage_all"),
                             "k_nonvac": _first(g, "coverage_nonvac"),
                             "k_cond": _first(g, "coverage_nonvac_cond")})
        lad2 = pd.DataFrame(lad2)
        j = ladder.merge(lad2, on=["arm", "alpha"], suffixes=("", "_2"))
        mis = (int(np.sum(j.k_first_nominal_all.to_numpy() != j.k_all.to_numpy()))
               + int(np.sum(j.k_first_nominal_nonvac.to_numpy()
                            != j.k_nonvac.to_numpy()))
               + int(np.sum(j.k_first_nominal_cond.to_numpy()
                            != j.k_cond.to_numpy())))
        checks.append(chk("B3 ladder readout matches an independent recount",
                          mis == 0 and len(j) == len(ladder),
                          "arms compared %d, mismatches %d over all three"
                          " conventions" % (len(j), mis)))
    except Exception as exc:
        checks.append(chk("B3 ladder readout matches an independent recount",
                          False, "could not run: %s" % exc))

    try:
        tot = partC[(partC.family == "transfer") & np.isclose(partC.alpha, 0.10)
                    & (partC.arm.isin(["plain", "pool_iw"]))]
        rec = []
        for r in tot.itertuples():
            sel = b[(b.arm == r.arm) & np.isclose(b.alpha, r.alpha) & (b.k == r.k)]
            rec.append(abs(int(sel.n_accepted.sum()) - int(r.tot_accepted)))
        checks.append(chk("C1 partC accepted counts sum back to the detail rows",
                          max(rec) == 0 if rec else False,
                          "arms x k compared %d, max count mismatch %d"
                          % (len(rec), max(rec) if rec else -1)))
    except Exception as exc:
        checks.append(chk("C1 partC accepted counts sum back to the detail rows",
                          False, "could not run: %s" % exc))

    if partD is not None:
        d = pd.read_csv(os.path.join(paths.RESULTS, "r14b_targetadapt.csv"))
        f = d[(d.arm == "adapt_k") & np.isclose(d.alpha, STACK_ALPHA) & (d.k == STACK_K)]
        manual = float(np.mean([r for r in f.accept_rate.to_numpy()]))
        val = float(partD.loc[partD.arm == "adapt_k", "mean_accept"].iloc[0])
        checks.append(chk("D1 partD adapt_k mean_accept matches a manual re-average",
                          abs(manual - val) < 1e-12,
                          "max|d| %.2e (n=%d)" % (abs(manual - val), len(f))))
        orc = partD[partD.arm == "oracle_adapt"]
        chk_ok = bool(len(orc)) and float(orc.mean_accept.iloc[0]) > float(
            partD.loc[partD.arm == "oracle_base", "mean_accept"].iloc[0])
        checks.append(chk("D2 adaptation raises the oracle ceiling (stored direction)",
                          chk_ok, "oracle_adapt %.4f > oracle_base %.4f"
                          % (float(orc.mean_accept.iloc[0]),
                             float(partD.loc[partD.arm == "oracle_base",
                                            "mean_accept"].iloc[0]))))
    if stack_diag is not None and len(stack_diag):
        nmax = float(np.nanmax(stack_diag.norm_diff.to_numpy(dtype=float)))
        pmax = float(np.nanmax(stack_diag.base_prob_diff.to_numpy(dtype=float)))
        ok_frac = float((stack_diag.base_prob_diff.to_numpy(dtype=float) < 1e-5).mean())
        bad = stack_diag[stack_diag.base_prob_diff.to_numpy(dtype=float) >= 1e-5]
        checks.append(chk("D3a refit pipeline rebuilds the stored target features exactly",
                          nmax < 1e-4,
                          "max|d test_X| %.2e over %d units" % (nmax, len(stack_diag))))
        checks.append(chk("D3b base model reproduced (diagnostic; single-threaded BLAS)",
                          ok_frac >= 0.9,
                          "%.1f%% of %d units within 1e-5 of the stored proba, max %.3f"
                          " (drift on %s); the drift is a thread-dependent logreg"
                          " iteration count, not a pipeline bug -- D3a shows the"
                          " features are rebuilt to %.1e"
                          % (100.0 * ok_frac, len(stack_diag), pmax,
                             ",".join(bad.unit.tolist()) if len(bad) else "none", nmax)))
    if stack is not None and len(stack):
        try:
            sdf = (stack_detail if stack_detail is not None
                   else pd.read_csv(os.path.join(paths.RESULTS,
                                                 "%s_stack_detail.csv" % args.out)))
            rec = []
            for arm in sdf.arm.unique():
                if arm == "adapt_plain":
                    continue
                rec.append(pair_stats(sdf, arm, "adapt_plain", ["unit", "clf", "alpha", "k", "rep"],
                                      "accept_rate"))
            expected = len(rec) == max(len(sdf.arm.unique()) - 1, 0)
            n_excl = int(sum(r.get("excludes_zero", 0) for r in rec))
            n_pos = int(sum(1 for r in rec if r.get("excludes_zero", 0)
                            and r.get("mean_diff", 0) > 0))
            checks.append(chk("D4 stacking comparison computed for every arm",
                              expected, "arms compared %d, paired diffs with CI"
                              " excluding zero %d (of which positive %d)"
                              % (len(rec), n_excl, n_pos)))
        except Exception as exc:
            checks.append(chk("D4 stacking comparison computed for every arm", False,
                              "could not run: %s" % exc))

    res = {"within_summary": within_summary, "transfer_summary": transfer_summary,
           "ladder": ladder, "partC": partC, "paired_within": paired_within,
           "paired_transfer": paired_transfer,
           "partD": partD, "partD_info": partD_info, "stack": stack,
           "stack_note": stack_note, "within_ref": within_ref,
           "within_ladder": within_ladder, "checks": checks}
    lines = build_report(res)
    rp = os.path.join(paths.WORK, "%s_report.txt" % args.out)
    with open(rp, "w", encoding="ascii", errors="replace", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    log("wrote %s" % rp)
    n_fail = sum(1 for c in checks if c["result"] == "FAIL")
    log("CHECKS: %d PASS, %d FAIL" % (len(checks) - n_fail, n_fail))
    log("total %.1fs" % (time.time() - t0))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()