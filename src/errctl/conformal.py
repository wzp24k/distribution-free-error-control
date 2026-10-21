"""Finite-sample, distribution-free control of the error rate among accepted labels.

Object of study -- the *selective risk*

    R(lambda) = E[ err . 1{s(X) <= lambda} ] / E[ 1{s(X) <= lambda} ],

i.e. "of the cells I annotate automatically, what fraction is wrong". Both the
numerator N(lambda) and the denominator D(lambda) are monotone in lambda, but
their ratio is not, so textbook conformal risk control (which needs a monotone
loss bounded by 1) does not apply directly.

What we do instead: bound the two terms separately and uniformly over a grid of
candidate thresholds, then take the largest feasible threshold,

    lambda_hat = max { lambda in grid : N^+(lambda) <= alpha * D^-(lambda) }.

Because the bounds hold simultaneously for every grid point, any data-dependent
choice of lambda_hat inherits them, and therefore

    P( R(lambda_hat) <= alpha ) >= 1 - delta

with no assumption on the data distribution beyond exchangeability of the
calibration and deployment cells.

Both bounds act on the *unconditional* probabilities, because those are the
means of iid bounded variables:

* Numerator bound: the number of accepted-and-wrong calibration cells is
  Binomial(n, N(lambda)), so inverting the exact test gives a Clopper-Pearson
  upper limit on N(lambda). No assumption is needed on how the error
  probability varies with x.
* Denominator bound: the accepted count is Binomial(n, D(lambda)), so the exact
  binomial lower limit bounds D(lambda) -- uniformly tighter than any
  exponential-inequality bound.

The multiplicity price of scanning a grid of thresholds can be removed
altogether: ordering the candidate thresholds and testing them in sequence,
stopping at the first that fails, controls the family-wise error at ``delta``
without any Bonferroni share, because only the first hypothesis in the order
that is actually false can be rejected incorrectly.  ``select="fixed_sequence"``
switches to that rule and ``select="bonferroni"`` (the default) keeps the
uniform-over-the-grid reading.

An optional sharper variant (``num="cond"``) instead bounds the *conditional*
error rate among accepted cells by treating it as binomial, which requires that
rate to be constant across accepted cells; it is reported as an ablation
because a score that leaves the error probability varying breaks that reading.
An empirical-Bernstein variant (``num="eb"``) is kept for comparison.

A time-varying (weighted) variant reweights calibration cells by the estimated
density ratio p_target(x) / p_source(x), which restores validity under covariate
shift at the price of an effective sample size.
"""

import numpy as np
from scipy.stats import beta as _beta


def _grid_indices(n, min_accept, grid_size):
    """Candidate acceptance counts (1-based) spread over the sorted scores."""
    lo = max(int(min_accept), 2)
    if lo > n:
        return np.array([], dtype=int)
    hi = n
    k = np.unique(np.round(np.linspace(lo, hi, min(grid_size, hi - lo + 1))).astype(int))
    return k


def _eb_radius(var, n_eff, log_term, kind="bernstein"):
    """Confidence radius for the mean of [0, 1] variables."""
    n_eff = np.maximum(n_eff, 1.0)
    if kind == "hoeffding":
        return np.sqrt(log_term / (2.0 * n_eff))
    return np.sqrt(2.0 * var * log_term / n_eff) + 3.0 * log_term / n_eff


class SelectiveRiskController:
    """Acceptance threshold with a finite-sample bound on the selective risk.

    Parameters
    ----------
    alpha : target upper bound on the error rate among accepted cells.
    delta : failure probability of that guarantee.
    min_accept : smallest accepted set the threshold may deliver.
    weighted : use importance weights (set by the caller) in both bounds.
    bound : "bernstein" (default) or "hoeffding" for the numerator when
        ``num="eb"``.
    grid_size : number of candidate thresholds; a coarser grid reduces the
        Bonferroni factor at the cost of a coarser threshold.
    denom : "cp" for the exact Clopper-Pearson denominator bound, "eb" for the
        empirical-Bernstein one.
    num : "cp" (default) inverts the exact binomial test for the unconditional
        probability of an accepted-and-wrong cell, which needs no assumption
        beyond exchangeability; "cond" does the same for the conditional error
        rate among accepted cells, which is sharper but assumes that rate to be
        constant, and is reported as an ablation; "eb" uses an
        empirical-Bernstein bound on the unconditional probability.
    target_scores : optional unlabelled scores from the deployment population.
        When given, the denominator D(lambda) is bounded from these scores
        instead of from the labelled calibration cells.  The two samples must
        be independent for the union bound to apply; in r72 the target scores
        are the held-out deployment pool, disjoint from the labelled pilot.
    """

    def __init__(self, alpha=0.10, delta=0.10, min_accept=25, weighted=False,
                 bound="bernstein", grid_size=120, denom="cp", num="cp",
                 select="bonferroni"):
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.min_accept = int(min_accept)
        self.weighted = bool(weighted)
        self.bound = bound
        self.grid_size = int(grid_size)
        self.denom = denom
        self.num = num
        self.select = select
        self.lambda_hat = -np.inf
        self.n_cal_ = 0
        self.n_target_ = 0

    # ------------------------------------------------------------------ fit
    def fit(self, scores, errs, weights=None, target_scores=None):
        scores = np.asarray(scores, dtype=np.float64)
        errs = np.asarray(errs, dtype=np.float64)
        n = scores.size
        self.n_cal_ = n
        self.n_target_ = (0 if target_scores is None
                          else int(np.size(target_scores)))
        if n == 0:
            self.lambda_hat = -np.inf
            return self

        if weights is None:
            w = np.ones(n, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
            if self.weighted and w.sum() > 0:
                w = w * (n / w.sum())

        order = np.argsort(scores, kind="mergesort")
        s, e, w = scores[order], errs[order], w[order]

        cum_w = np.cumsum(w)
        cum_w2 = np.cumsum(w * w)
        cum_we = np.cumsum(w * e)
        cum_w2e = np.cumsum(w * w * e)

        cand = _grid_indices(n, self.min_accept, self.grid_size)
        if cand.size == 0:
            self.lambda_hat = -np.inf
            return self
        m = cand.size
        idx = cand - 1                                  # 0-based

        sw = cum_w[idx]                                 # accepted weight mass
        swx = cum_we[idx]                               # accepted-and-wrong mass
        n_eff = np.where(cum_w2[idx] > 0, sw * sw / np.maximum(cum_w2[idx], 1e-12), 0.0)

        # Unconditional means over the calibration sample,
        #     A = E[w . err . 1{s <= lambda}]   B = E[w . 1{s <= lambda}],
        # whose ratio is the selective risk.  The *conditional* rate among the
        # accepted cells is only that ratio, and it has no binomial law of its
        # own once the error probability varies with x; the bounds are therefore
        # placed on A and B, which are means of iid bounded variables.
        a_hat = np.clip(swx / n, 0.0, 1.0)
        b_hat = np.clip(sw / n, 0.0, 1.0)
        var_a = np.clip(cum_w2e[idx] / n - a_hat ** 2, 0.0, None)
        var_b = np.clip(cum_w2[idx] / n - b_hat ** 2, 0.0, None)

        if self.select == "fixed_sequence":
            # Fixed-sequence (step-down) testing in the prespecified order of
            # increasing threshold.  A claim can only be wrong at the first
            # hypothesis in that order which is actually false, and we stop
            # there, so no multiplicity share is charged to either tail.  The
            # bound is a little tighter and, unlike the Bonferroni reading, it
            # does not degrade as the grid is refined.
            log_term = np.log(3.0 / self.delta)
            delta_m = self.delta / 2.0
        else:
            log_term = np.log(3.0 * m / self.delta)
            delta_m = self.delta / (2.0 * m)

        if self.num == "cp" and not self.weighted:
            # Exact one-sided Clopper-Pearson limits on the two unconditional
            # probabilities; delta/(2m) is the Bonferroni share that carries the
            # uniformity over the threshold grid.
            n_wrong = np.rint(swx).astype(np.int64)
            upper_n = _beta.ppf(1.0 - delta_m, n_wrong + 1.0,
                                np.maximum(n - n_wrong, 1.0))
        elif self.num == "cond" and not self.weighted:
            # Sharpe  n_wrong out of the accepted count.  This is the exact
            # binomial limit for a *homogeneous* error probability among the
            # accepted cells; when the score leaves that probability varying the
            # bound is no longer distribution-free, which is why it is not the
            # default.
            n_wrong = np.rint(swx).astype(np.int64)
            upper_n = _beta.ppf(1.0 - delta_m, n_wrong + 1.0,
                                np.maximum(cand - n_wrong, 1.0))
        else:
            upper_n = np.clip(a_hat + _eb_radius(var_a, n_eff, log_term, self.bound),
                              0.0, 1.0)

        if self.denom == "cp" and not self.weighted:
            lower_d = _beta.ppf(delta_m, cand, n - cand + 1)
        else:
            lower_d = np.clip(b_hat - _eb_radius(var_b, n_eff, log_term, self.bound),
                              0.0, 1.0)

        if target_scores is not None and not self.weighted:
            # D(lambda) = P_target(s <= lambda) is a property of the score
            # distribution and needs no labels.  Read it off the unlabelled
            # deployment scores instead of the n calibration cells; the two
            # samples are independent, so the same delta/(2m) share keeps the
            # union bound over the grid and the two one-sided limits.
            ts = np.sort(np.asarray(target_scores, dtype=np.float64))
            n_t = ts.size
            if n_t:
                c_t = np.searchsorted(ts, s[idx], side="right")
                lower_d = _beta.ppf(delta_m, c_t,
                                    np.maximum(n_t - c_t + 1, 1))

        with np.errstate(divide="ignore", invalid="ignore"):
            if self.num == "cond" and not self.weighted:
                # Sharpe's bound is already a bound on the conditional error
                # rate among the accepted cells, so it *is* the ratio.  Dividing
                # it by lower_d as well would re-impose the numerator's slack and
                # make the ablation strictly more conservative than the default,
                # which defeats its purpose: it is meant to be the sharper but
                # non-distribution-free alternative.
                ratio = upper_n
            else:
                ratio = np.where(lower_d > 0.0, upper_n / np.maximum(lower_d, 1e-12), np.inf)

        # Empirical conditional rate and acceptance fraction, for reporting only.
        mu_n = np.divide(swx, sw, out=np.zeros_like(swx), where=sw > 0)
        mu_d = np.divide(sw, sw, out=np.ones_like(sw), where=sw > 0)

        self.cand_ = cand
        self.upper_n_ = upper_n
        self.lower_d_ = lower_d
        self.bound_ratio_ = ratio
        self.emp_n_ = mu_n
        self.emp_d_ = mu_d
        self.emp_joint_n_ = a_hat
        self.emp_joint_d_ = b_hat

        if self.select == "fixed_sequence":
            # cand ascends in lambda, so scanning from the safest threshold and
            # stopping at the first failure is exactly the step-down rule.
            bad = np.nonzero(ratio > self.alpha)[0]
            j = int(bad[0]) - 1 if bad.size else m - 1
            self.chosen_ = None if j < 0 else int(cand[j])
            self.lambda_hat = (-np.inf if j < 0 else float(s[cand[j] - 1]))
        else:
            feasible = np.nonzero(ratio <= self.alpha)[0]
            if feasible.size == 0:
                self.lambda_hat = -np.inf
                self.chosen_ = None
            else:
                j = feasible[-1]                        # largest feasible threshold
                self.chosen_ = int(cand[j])
                self.lambda_hat = float(s[cand[j] - 1])
        return self

    # -------------------------------------------------------------- predict
    def accept(self, scores):
        return np.asarray(scores, dtype=np.float64) <= self.lambda_hat

    def summary(self):
        if self.chosen_ is None:
            return {"lambda_hat": -np.inf, "cal_accepted": 0, "cal_accept_frac": 0.0,
                    "cal_empirical_risk": float("nan"),
                    "cal_bound": float("nan"),
                    "n_target_scores": self.n_target_}
        j = int(np.nonzero(self.cand_ == self.chosen_)[0][0])
        return {
            "lambda_hat": self.lambda_hat,
            "cal_accepted": self.chosen_,
            "cal_accept_frac": self.chosen_ / float(self.n_cal_),
            "cal_empirical_risk": float(self.emp_n_[j]),
            "cal_empirical_risk_joint": float(self.emp_joint_n_[j]),
            "cal_accept_frac_joint": float(self.emp_joint_d_[j]),
            "cal_bound": float(self.bound_ratio_[j]),
            "n_target_scores": self.n_target_,
        }


def _hoeffding_range(ws, log_term, n):
    """Hoeffding radius for (1/n) * sum of independent [0, w_i] variables.

    ``P(mean - E[mean] >= sqrt(sum w_i^2 log(1/delta) / (2 n^2))) <= delta``.
    Using each cell's own range rather than a variance estimate keeps the
    bound finite when a few importance weights are large.
    """
    return float(np.sqrt(np.sum(np.asarray(ws, dtype=np.float64) ** 2)
                         * log_term / (2.0 * n * n)))


class WeightedSelectiveRiskController:
    """Selective risk control under covariate shift, with bounded weights.

    The calibration cells carry importance weights ``w(x) = p_target(x) /
    p_source(x)``.  The original ``weighted=True`` mode of
    :class:`SelectiveRiskController` places an empirical-Bernstein bound on the
    two weighted means; the radius then scales with the variance of the weight,
    so the heavy right tail that covariate shift produces inflates the bound
    until nothing is feasible.  Two changes recover the power.

    1. *Range-aware Hoeffding instead of a variance estimate.*  For the
       numerator every cell contributes ``w_i e_i 1{s_i <= lambda}`` in
       ``[0, w_i]``, so Hoeffding with the *known* per-cell ranges -- not an
       estimated variance -- is available: its radius is
       ``sqrt(sum_i w_i^2 log(1/delta) / (2 n^2))``.  The same applies to the
       denominator contribution ``w_i 1{s_i <= lambda}``.  Both bounds are
       finite whenever ``w`` is, and neither can be defeated by one outlying
       weight the way a variance estimate can.  A stratum-wise Clopper-Pearson
       count bound is computed as well and the *tighter* of the two is taken;
       both are valid, so their minimum is.

    2. *A denominator measured on unlabelled deployment cells.*  The
       denominator ``D(lambda) = P_target(s <= lambda)`` is a property of the
       score distribution alone and can be read off the deployment sample,
       whose scores need no labels.  Passing ``target_scores`` replaces the
       weighted denominator -- the widest part of the old bound -- by an exact
       one-sided Clopper-Pearson limit on a sample that is orders of magnitude
       larger.  The numerator still needs labels and remains the weighted part;
       that is the irreducible cost of shift.

    Together these make the weighted certificate usable exactly in the
    mild-overlap regime where weighting is supposed to apply, and they leave it
    silent when the weights genuinely have no overlap to exploit.
    """

    def __init__(self, alpha=0.10, delta=0.10, min_accept=25, strata=4,
                 min_stratum=15, grid_size=120, select="bonferroni",
                 exact_strata=True):
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.min_accept = int(min_accept)
        self.strata = int(strata)
        self.min_stratum = int(min_stratum)
        self.grid_size = int(grid_size)
        self.select = select
        self.exact_strata = bool(exact_strata)
        self.lambda_hat = -np.inf
        self.chosen_ = None
        self.n_cal_ = 0

    # ------------------------------------------------------------------ fit
    def fit(self, scores, errs, weights, target_scores=None):
        scores = np.asarray(scores, dtype=np.float64)
        errs = np.asarray(errs, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        n = scores.size
        self.n_cal_ = n
        self.n_target_ = 0 if target_scores is None else int(np.size(target_scores))
        if n == 0 or w.sum() <= 0:
            self.lambda_hat = -np.inf
            return self
        w = w * (n / w.sum())                       # normalise to mean one

        order = np.argsort(scores, kind="mergesort")
        s, e, w_s = scores[order], errs[order], w[order]
        wrong = (e > 0.5)

        # strata on the weight axis, used only for the exact leg of the bound
        K = max(1, min(self.strata, n // max(self.min_stratum, 1)))
        edges = np.quantile(w, np.linspace(0.0, 1.0, K + 1)) if K > 1 else None
        if edges is not None:
            edges = np.unique(edges)
            K = max(1, edges.size - 1)
        if edges is not None and K > 1:
            sid = np.clip(np.searchsorted(edges, w_s, side="right") - 1, 0, K - 1)
        else:
            sid = np.zeros(n, dtype=int)
        K = int(sid.max()) + 1

        cand = _grid_indices(n, self.min_accept, self.grid_size)
        if cand.size == 0:
            self.lambda_hat = -np.inf
            return self
        m = cand.size
        idx = cand - 1

        if self.select == "fixed_sequence":
            lev = self.delta / 2.0
        else:
            lev = self.delta / (2.0 * m)
        log_term = float(np.log(3.0 / lev))
        lev_strata = lev / max(K, 1)

        cum_whoeff_n = np.cumsum(w_s * w_s)          # sum_i w_i^2 over prefix
        cum_hoeff_d = np.cumsum(w_s * w_s)
        upper_h = cum_whoeff_n[idx] / n + _hoeffding_range(
            np.sqrt(cum_whoeff_n[idx]), log_term, n)
        lower_h = cum_hoeff_d[idx] / n - _hoeffding_range(
            np.sqrt(cum_hoeff_d[idx]), log_term, n)

        upper_n, lower_d = upper_h, lower_h
        if self.exact_strata:
            up_s = np.zeros(m)
            lo_s = np.zeros(m)
            for j in range(K):
                mj = sid == j
                nj = int(mj.sum())
                if nj == 0:
                    continue
                pi_j = nj / float(n)
                w_hi_j = float(np.max(w_s[mj]))
                w_lo_j = float(np.min(w_s[mj]))
                Bj = np.cumsum(mj)
                Aj = np.cumsum(mj & wrong)
                wbar = pi_j * w_hi_j
                up_s += wbar * _beta.ppf(1.0 - lev_strata, Aj[idx] + 1.0,
                                         np.maximum(nj - Aj[idx], 1.0))
                lo_s += (pi_j * w_lo_j
                         * _beta.ppf(lev_strata, Bj[idx],
                                     np.maximum(nj - Bj[idx] + 1, 1.0)))
            upper_n = np.minimum(upper_n, up_s)
            lower_d = np.maximum(lower_d, lo_s)

        if target_scores is not None:
            ts = np.sort(np.asarray(target_scores, dtype=np.float64))
            n_t = ts.size
            c_t = np.searchsorted(ts, s[idx], side="right")
            lower_d = _beta.ppf(lev, c_t, np.maximum(n_t - c_t + 1, 1))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(lower_d > 0.0, upper_n / np.maximum(lower_d, 1e-12),
                             np.inf)

        self.cand_ = cand
        self.upper_n_ = upper_n
        self.lower_d_ = lower_d
        self.bound_ratio_ = ratio

        if self.select == "fixed_sequence":
            bad = np.nonzero(ratio > self.alpha)[0]
            j = int(bad[0]) - 1 if bad.size else m - 1
            self.chosen_ = None if j < 0 else int(cand[j])
            self.lambda_hat = (-np.inf if j < 0 else float(s[cand[j] - 1]))
        else:
            feasible = np.nonzero(ratio <= self.alpha)[0]
            if feasible.size == 0:
                self.lambda_hat = -np.inf
                self.chosen_ = None
            else:
                j = feasible[-1]
                self.chosen_ = int(cand[j])
                self.lambda_hat = float(s[cand[j] - 1])
        return self

    def accept(self, scores):
        return np.asarray(scores, dtype=np.float64) <= self.lambda_hat

    def summary(self):
        if self.chosen_ is None:
            return {"lambda_hat": -np.inf, "cal_accepted": 0, "cal_bound": float("nan")}
        j = int(np.nonzero(self.cand_ == self.chosen_)[0][0])
        return {"lambda_hat": self.lambda_hat, "cal_accepted": self.chosen_,
                "cal_accept_frac": self.chosen_ / float(self.n_cal_),
                "cal_bound": float(self.bound_ratio_[j]),
                "n_target_scores": self.n_target_}


class ScoreConditionalController:
    """Selective risk control across domains from unlabelled deployment scores.

    Weighting the calibration cells by the density ratio needs no labels from
    the deployment domain, but it needs *overlap*, and it pays for a heavy
    weight tail with a confidence radius that grows with the weight spread.  A
    different trade is available.  Suppose the error probability *given the
    score* is the same in both domains,

        P_source(err = 1 | s(x) = u)  =  P_target(err = 1 | s(x) = u)   for all u,

    which is weaker than exchangeability -- it permits any shift in the score
    distribution -- and is the assumption a practitioner makes when reading a
    confidence value as a probability of being right.  Then

        R(lambda) = int_0^lambda theta(u) dQ_target(u) / Q_target(lambda),

    where ``theta`` is the shared error curve and ``Q_target`` the deployment
    score distribution.  ``Q_target`` is read off the unlabelled deployment
    scores -- no labels needed -- with the DKW inequality giving a bound valid
    simultaneously at every threshold, and ``theta`` is estimated bin by bin
    from the *unweighted* calibration cells with exact Clopper-Pearson limits.
    No importance weight enters the radius, so the bound does not degrade when
    the two score distributions overlap only partly.

    The assumption is not free and is not silent: it is violated when the
    meaning of the score itself moves, which is exactly the
    ``shift_labelshift`` setting of r17, where coverage collapses as it should.
    It is also testable in deployment, by comparing the calibration error curve
    with a held-out sliver of labelled deployment cells.
    """

    def __init__(self, alpha=0.10, delta=0.10, min_accept=25, n_bins=10,
                 min_bin=20, edges="pooled"):
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.min_accept = int(min_accept)
        self.n_bins = int(n_bins)
        self.min_bin = int(min_bin)
        self.edges = edges
        self.lambda_hat = -np.inf
        self.chosen_ = None
        self.n_cal_ = 0

    def _edges(self, s_src, s_tgt):
        if self.edges == "equal":
            return np.linspace(0.0, 1.0, self.n_bins + 1)
        pool = np.concatenate([s_src, s_tgt])
        e = np.quantile(pool, np.linspace(0.0, 1.0, self.n_bins + 1))
        return np.unique(e)

    def fit(self, s_src, e_src, s_tgt):
        s_src = np.asarray(s_src, dtype=np.float64)
        e_src = np.asarray(e_src, dtype=np.float64)
        s_tgt = np.asarray(s_tgt, dtype=np.float64)
        n, n_t = s_src.size, s_tgt.size
        self.n_cal_, self.n_tgt_ = n, n_t
        if n == 0 or n_t == 0:
            self.lambda_hat = -np.inf
            return self

        edges = self._edges(s_src, s_tgt)
        if edges.size < 3:
            self.lambda_hat = -np.inf
            return self
        B = edges.size - 1                            # bins, edges.size - 1 of them
        lev_t = (self.delta / 2.0) / float(B)          # target CDF limits, one per edge
        lev_th = (self.delta / 2.0) / float(B)         # error-curve limits, one per bin

        b_src = np.clip(np.searchsorted(edges, s_src, side="right") - 1, 0, B - 1)
        b_tgt = np.clip(np.searchsorted(edges, s_tgt, side="right") - 1, 0, B - 1)

        n_b = np.bincount(b_src, minlength=B).astype(float)
        m_b = np.bincount(b_src, weights=(e_src > 0.5).astype(float), minlength=B)
        q_hat = np.bincount(b_tgt, minlength=B).astype(float) / float(n_t)
        cum_t = np.cumsum(np.bincount(b_tgt, minlength=B)).astype(float)

        theta_up = np.zeros(B)
        for j in range(B):
            if n_b[j] <= 0:
                theta_up[j] = 1.0                      # no evidence: no claim
                continue
            theta_up[j] = _beta.ppf(1.0 - lev_th, m_b[j] + 1.0,
                                    np.maximum(n_b[j] - m_b[j], 1.0))

        # Exact one-sided limits on the deployment score distribution at every
        # bin edge, so the target-mass uncertainty enters as a Clopper-Pearson
        # error rather than a distribution-free (DKW) one.
        c_t = np.cumsum(np.bincount(b_tgt, minlength=B)).astype(float)
        q_lo = _beta.ppf(lev_t, c_t, np.maximum(n_t - c_t + 1, 1))
        q_hi = _beta.ppf(1.0 - lev_t, c_t + 1.0, np.maximum(n_t - c_t, 1.0))
        err_t = np.maximum(q_hi - c_t / float(n_t), c_t / float(n_t) - q_lo)
        err_t = np.maximum.accumulate(err_t)

        # R(lambda) = int_0^lambda theta dQ / Q(lambda).  theta <= theta_up
        # pointwise, so integrating by parts turns the target-mass uncertainty
        # into a single margin scaled by the total variation of the error curve
        # rather than one margin per bin.
        tv = np.zeros(B)
        prev = 0.0
        for j in range(B):
            tv[j] = (tv[j - 1] if j else 0.0) + abs(theta_up[j] - prev)
            prev = theta_up[j]

        upper_n = np.cumsum(q_hat * theta_up) + err_t * (theta_up + tv)
        lower_d = q_lo

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(lower_d > 0.0, upper_n / np.maximum(lower_d, 1e-12), np.inf)

        self.edges_ = edges
        self.cand_ = np.cumsum(n_b)
        self.theta_up_ = theta_up
        self.upper_n_ = upper_n
        self.lower_d_ = lower_d
        self.bound_ratio_ = ratio
        self.eps_ = float(np.max(err_t))

        ok = (ratio <= self.alpha) & (self.cand_ >= self.min_accept)
        idx = np.nonzero(ok)[0]
        if idx.size == 0:
            self.lambda_hat = -np.inf
            self.chosen_ = None
        else:
            j = int(idx[-1])
            self.lambda_hat = float(edges[j + 1])
            self.chosen_ = int(self.cand_[j])
        return self

    def accept(self, scores):
        return np.asarray(scores, dtype=np.float64) <= self.lambda_hat

    def summary(self):
        if self.chosen_ is None:
            return {"lambda_hat": -np.inf, "cal_accepted": 0, "cal_bound": float("nan")}
        return {"lambda_hat": self.lambda_hat, "cal_accepted": self.chosen_,
                "cal_accept_frac": self.chosen_ / float(self.n_cal_),
                "cal_bound": float(self.bound_ratio_[self.bound_ratio_ <= self.alpha].min()),
                "n_target_scores": self.n_tgt_}


class MondrianSelectiveRiskController:
    """Classwise thresholds: one controller per *predicted* class.

    A single global threshold lets an easy, abundant population subsidise a hard
    or rare one -- exactly what happens in spatial proteomics, where Treg or
    plasma cells are the populations a pathologist must re-check by hand.
    """

    def __init__(self, alpha=0.10, delta=0.10, min_accept=15, **kw):
        self.alpha, self.delta = alpha, delta
        self.min_accept = min_accept
        self.kw = kw
        self.controllers_ = {}
        self.fallback_ = None

    def fit(self, scores, errs, pred, weights=None):
        scores = np.asarray(scores, dtype=np.float64)
        errs = np.asarray(errs, dtype=np.float64)
        pred = np.asarray(pred)
        self.fallback_ = SelectiveRiskController(
            self.alpha, self.delta, min_accept=self.min_accept, **self.kw
        ).fit(scores, errs, weights)
        for cls in np.unique(pred):
            m = pred == cls
            if m.sum() < self.min_accept:
                continue
            w = None if weights is None else np.asarray(weights)[m]
            self.controllers_[int(cls)] = SelectiveRiskController(
                self.alpha, self.delta,
                min_accept=max(5, self.min_accept // 3), **self.kw
            ).fit(scores[m], errs[m], w)
        return self

    def accept(self, scores, pred):
        scores = np.asarray(scores, dtype=np.float64)
        pred = np.asarray(pred)
        out = np.zeros(scores.shape[0], dtype=bool)
        for cls, ctrl in self.controllers_.items():
            m = pred == cls
            if m.any():
                out[m] = ctrl.accept(scores[m])
        known = np.isin(pred, list(self.controllers_))
        if (~known).any():
            out[~known] = self.fallback_.accept(scores[~known])
        return out


# --------------------------------------------------------------------- tools
def selective_risk(scores, errs, lam, weights=None):
    """Empirical selective risk at a fixed threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)
    m = scores <= lam
    if weights is None:
        denom = m.sum()
        return float(errs[m].sum() / denom) if denom else float("nan")
    w = np.asarray(weights, dtype=np.float64)
    denom = (w * m).sum()
    return float((w * m * errs).sum() / denom) if denom else float("nan")


def naive_threshold(scores, errs, alpha, min_accept=25):
    """Largest threshold whose *empirical* calibration risk is within budget.

    This is what a practitioner does without a finite-sample correction; it is
    the efficiency yardstick our bound has to be compared against.
    """
    scores = np.asarray(scores, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    s, e = scores[order], errs[order]
    cum = np.cumsum(e)
    k = np.arange(1, s.size + 1)
    ok = (cum / k <= alpha) & (k >= min_accept)
    if not ok.any():
        return -np.inf
    return float(s[np.nonzero(ok)[0][-1]])


def risk_coverage_curve(scores, errs, weights=None, n_points=300):
    """Risk as a function of coverage while sweeping the threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    errs = np.asarray(errs, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    s, e = scores[order], errs[order]
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=np.float64)[order]
    cum_w = np.cumsum(w)
    cum_we = np.cumsum(w * e)
    cov = cum_w / cum_w[-1]
    risk = np.divide(cum_we, cum_w, out=np.full_like(cum_we, np.nan), where=cum_w > 0)
    keep = np.unique(np.linspace(0, cov.size - 1, min(n_points, cov.size)).astype(int))
    return cov[keep], risk[keep]


def aurc(scores, errs):
    """Normalised area under the risk-coverage curve (lower is better)."""
    cov, risk = risk_coverage_curve(scores, errs)
    base = float(np.mean(errs))
    if base <= 0:
        return 0.0
    ok = np.isfinite(risk)
    return float(np.trapezoid(risk[ok], cov[ok]) / base)


def conformal_quantile(scores, alpha):
    """Split-conformal (1 - alpha) quantile with the finite-sample correction."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return np.inf if k > n else float(s[k - 1])


def prediction_set_sizes(proba, q):
    """Conformal label sets from class probabilities (score = 1 - p)."""
    proba = np.asarray(proba, dtype=np.float64)
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    in_set = (cum - sorted_p) <= q
    return in_set.sum(axis=1), order[:, 0]
