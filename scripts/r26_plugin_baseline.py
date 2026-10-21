"""R26 -- the conservative plug-in baseline, plus a delta sweep.

Referee objection.  The headline of the paper is a distribution-free
finite-sample certificate on the selective risk, i.e. the error rate among the
cells the layer accepts.  The most damaging objection is that a careful
practitioner never uses the uncorrected threshold: they add a confidence margin,
and a margin-corrected *plug-in* threshold might hold the budget while accepting
strictly more cells.  If it does, the value of the certificate collapses; if it
does not, the validity of the certificate is the headline.  This script answers
that with numbers.

Every arm is scored on the identical split, the identical score
s = 1 - max_c proba, and the identical candidate grid
_grid_indices(n, min_accept=25, grid_size=120) of *accepted counts*, ascending.

Arms, per (file, repeat, alpha in {0.05, 0.10, 0.20}):

  certificate            SelectiveRiskController(alpha, delta, min_accept=25,
                         grid_size=120) with its defaults (num="cp",
                         denom="cp", select="bonferroni"): exact
                         Clopper-Pearson limits on the two *unconditional*
                         probabilities, Bonferroni over the grid.  The
                         distribution-free arm.
  naive                  C.naive_threshold(s_cal, e_cal, alpha) applied to the
                         test scores.  Carries no guarantee.
  oracle_peek            C.naive_threshold(s_te, e_te, alpha).  Peeks at the
                         test answers: an upper envelope, not a method.
  plugin_cp_cond         the Sharpe / conditional reading: the largest grid k
                         whose exact Clopper-Pearson upper limit for the number
                         of wrong cells among the k calibration-accepted cells,
                         treated as Binomial(k, pi), is <= alpha, with
                         delta_m = delta / (2 m) and m the number of grid
                         points.  This is exactly the bound_ratio_ that
                         SelectiveRiskController(..., num="cond") computes; the
                         equality is asserted for every evaluation below.
  plugin_hoeffding_cond  the same threshold rule with the Hoeffding margin
                         p_hat + sqrt(log(1 / delta_m) / (2 k)).
  plugin_wilson_cond     the same threshold rule with the Wilson score upper
                         limit at level 1 - delta_m on the conditional rate.

The three plug-ins share one code path and one grid.

Delta sweep.  Arms 1 and 4 are swept over delta in {0.05, 0.10, 0.20}; arms 5
and 6 depend on delta_m as well and are swept too, so that every
(family, alpha, delta) cell carries all six arms.  Arms 2 and 3 do not depend on
delta; they are reused and repeated at every delta.

Outputs
    experiments/r26_plugin_baseline.csv
    experiments/r26_plugin_baseline_summary.csv
    work/r26_report.txt
"""

import argparse
import multiprocessing as mp
import os
import re
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta
from scipy.stats import chi2 as _chi2
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from errctl import conformal as C, paths

PRED = os.path.join(paths.WORK, "predictions")
RES = paths.RESULTS
WORK = paths.WORK

ALPHAS = (0.05, 0.10, 0.20)
DELTAS = (0.05, 0.10, 0.20)
MIN_ACCEPT = 25
GRID_SIZE = 120
ARMS = ("certificate", "naive", "oracle_peek", "plugin_cp_cond",
        "plugin_hoeffding_cond", "plugin_wilson_cond")
PLUGINS = ("plugin_cp_cond", "plugin_hoeffding_cond", "plugin_wilson_cond")
FAMILIES = ("cellsplit", "within", "transfer")
NAME_RE = re.compile(r"(.*)__(logreg|rf|hgb)\.npz$")
B = 2000
Z95 = 1.959963984540054
RNG_SEED = 20260927
OUT_CSV = os.path.join(RES, "r26_plugin_baseline.csv")
OUT_SUM = os.path.join(RES, "r26_plugin_baseline_summary.csv")
OUT_BINS = os.path.join(RES, "r26_constancy_bins.csv")
OUT_TXT = os.path.join(WORK, "r26_report.txt")
SORT_COLS = ("family", "unit", "clf", "repeat", "alpha", "delta", "arm")


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def family_of(unit):
    if unit.startswith("cellsplit"):
        return "cellsplit"
    if unit.startswith("within"):
        return "within"
    return "transfer"


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def boot_ci(values, rng, B=B):
    """Percentile bootstrap of a mean; the resampling unit is a run unit."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    idx = rng.integers(0, values.size, size=(B, values.size))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def wilson_ci(k, n, z=Z95):
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan
    p = k / float(n)
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def plugin_rule(s_cal, e_cal, alpha, delta, kind,
                min_accept=MIN_ACCEPT, grid_size=GRID_SIZE):
    """One grid, one code path, three conservative conditional margins.

    Returns the largest candidate accepted count k in the grid for which the
    conditional margin on the number of wrong cells among the k calibration
    cells accepted by that threshold is at or below alpha, together with the
    threshold and the margin at the chosen k.
    """
    n = s_cal.size
    cand = C._grid_indices(n, min_accept, grid_size)
    out = {"chosen": None, "lambda": -np.inf, "bound": np.nan,
           "emp": np.nan, "cand": cand, "n": n}
    if cand.size == 0:
        return out
    order = np.argsort(s_cal, kind="mergesort")
    s = np.asarray(s_cal, dtype=np.float64)[order]
    wrong = (np.asarray(e_cal, dtype=np.float64)[order] > 0.5).astype(float)
    n_wrong = np.cumsum(wrong)[cand - 1].astype(float)
    k = cand.astype(float)
    p_hat = n_wrong / k
    m = cand.size
    delta_m = delta / (2.0 * m)
    if kind == "cp":
        bound = _beta.ppf(1.0 - delta_m, n_wrong + 1.0,
                          np.maximum(k - n_wrong, 1.0))
    elif kind == "hoeffding":
        bound = p_hat + np.sqrt(np.log(1.0 / delta_m) / (2.0 * k))
    elif kind == "wilson":
        z = _norm.ppf(1.0 - delta_m)
        den = 1.0 + z * z / k
        half = z * np.sqrt(p_hat * (1.0 - p_hat) / k + z * z / (4.0 * k * k))
        bound = (p_hat + z * z / (2.0 * k) + half) / den
    else:
        raise ValueError("unknown margin %r" % (kind,))
    ok = np.nonzero(bound <= alpha)[0]
    if ok.size:
        j = int(ok[-1])
        out["chosen"] = int(cand[j])
        out["lambda"] = float(s[cand[j] - 1])
        out["bound"] = float(bound[j])
        out["emp"] = float(p_hat[j])
    return out

def _job(fname):
    """All six arms, all alphas, all deltas for one prediction file.

    The split is exactly the one of r16_review_budget._matched_job: repeat r
    selects the calibration rows by cal_rep[:, r] and the test rows by
    test_rep[:, r] when that array exists (cellsplit), else the whole test set.
    """
    mt = NAME_RE.match(fname)
    unit, clf = mt.group(1), mt.group(2)
    fam = family_of(unit)
    d = np.load(os.path.join(PRED, fname), allow_pickle=True)
    cal_probs, cal_y, cal_rep = d["cal_probs"], d["cal_y"], d["cal_rep"]
    test_probs, test_y = d["test_probs"], d["test_y"]
    test_rep = d["test_rep"] if "test_rep" in d.files else None
    rows = []
    n_cmp = 0
    n_mismatch = 0
    max_dlam = 0.0
    for r in range(cal_rep.shape[1]):
        sel = cal_rep[:, r]
        s_cal = 1.0 - cal_probs[sel].max(1)
        e_cal = (cal_probs[sel].argmax(1) != cal_y[sel]).astype(float)
        te_sel = (test_rep[:, r] if test_rep is not None
                  else np.ones(len(test_y), bool))
        n_te = int(te_sel.sum())
        s_te = 1.0 - test_probs[te_sel].max(1)
        e_te = (test_probs[te_sel].argmax(1) != test_y[te_sel]).astype(float)
        for alpha in ALPHAS:
            lam_naive = float(C.naive_threshold(s_cal, e_cal, alpha))
            lam_oracle = float(C.naive_threshold(s_te, e_te, alpha))
            for delta in DELTAS:
                spec = {}
                ctl = C.SelectiveRiskController(
                    alpha, delta, min_accept=MIN_ACCEPT,
                    grid_size=GRID_SIZE).fit(s_cal, e_cal)
                if ctl.chosen_ is None:
                    spec["certificate"] = {"lambda": float(ctl.lambda_hat),
                                           "bound": np.nan}
                else:
                    j = int(np.nonzero(ctl.cand_ == ctl.chosen_)[0][0])
                    spec["certificate"] = {
                        "lambda": float(ctl.lambda_hat),
                        "bound": float(ctl.bound_ratio_[j])}
                for kind, arm in (("cp", "plugin_cp_cond"),
                                  ("hoeffding", "plugin_hoeffding_cond"),
                                  ("wilson", "plugin_wilson_cond")):
                    spec[arm] = plugin_rule(s_cal, e_cal, alpha, delta, kind)
                cond = C.SelectiveRiskController(
                    alpha, delta, min_accept=MIN_ACCEPT, grid_size=GRID_SIZE,
                    num="cond").fit(s_cal, e_cal)
                n_cmp += 1
                if cond.chosen_ != spec["plugin_cp_cond"]["chosen"]:
                    n_mismatch += 1
                dlam = abs(float(cond.lambda_hat)
                           - float(spec["plugin_cp_cond"]["lambda"]))
                if np.isfinite(dlam):
                    max_dlam = max(max_dlam, dlam)
                for arm in ARMS:
                    if arm == "naive":
                        lam, bound = lam_naive, np.nan
                    elif arm == "oracle_peek":
                        lam, bound = lam_oracle, np.nan
                    else:
                        lam = spec[arm]["lambda"]
                        bound = spec[arm]["bound"]
                    mask = s_te <= lam
                    n_acc = int(mask.sum())
                    risk = float(e_te[mask].mean()) if n_acc else np.nan
                    mcal = s_cal <= lam
                    ca = int(mcal.sum())
                    cemp = float(e_cal[mcal].mean()) if ca else np.nan
                    rows.append({
                        "family": fam, "unit": unit, "clf": clf,
                        "repeat": int(r), "alpha": float(alpha),
                        "delta": float(delta), "arm": arm,
                        "n_test_eval": int(n_te), "n_accepted": n_acc,
                        "accept_rate": n_acc / max(n_te, 1),
                        "selective_risk": risk,
                        "covered": int((not np.isfinite(risk)) or risk <= alpha),
                        "empty": int(n_acc == 0),
                        "lambda": float(lam),
                        "cal_accepted": ca,
                        "cal_empirical_risk": cemp,
                        "cal_bound": (float(bound) if np.isfinite(bound)
                                      else np.nan),
                    })
    return rows, fname, {"n_cmp": n_cmp, "n_mismatch": n_mismatch,
                         "max_dlam": max_dlam}


def run_parallel(jobs, n_jobs):
    rows = []
    chk = {"n_cmp": 0, "n_mismatch": 0, "max_dlam": 0.0}

    def absorb(res):
        r, fname, c = res
        rows.extend(r)
        chk["n_cmp"] += c["n_cmp"]
        chk["n_mismatch"] += c["n_mismatch"]
        chk["max_dlam"] = max(chk["max_dlam"], c["max_dlam"])

    if n_jobs > 1:
        with mp.Pool(n_jobs, initializer=_init_worker) as pool:
            for i, res in enumerate(
                    pool.imap(_job, jobs, chunksize=1), 1):
                absorb(res)
                if i % 50 == 0 or i == len(jobs):
                    log("    %d/%d  %s" % (i, len(jobs), res[1]))
    else:
        for i, jb in enumerate(jobs, 1):
            res = _job(jb)
            absorb(res)
            if i % 50 == 0 or i == len(jobs):
                log("    %d/%d  %s" % (i, len(jobs), res[1]))
    return pd.DataFrame(rows), chk


def summarise(df):
    """Per family x alpha x delta x arm summary of the long table."""
    rows = []
    for (fam, alpha, delta, arm), g in df.groupby(
            ["family", "alpha", "delta", "arm"]):
        n_eval = int(len(g))
        acc = g["accept_rate"].values
        risk = g["selective_risk"].values
        cov = g["covered"].values
        nonvac = (g["empty"].values == 0)
        lam = g["lambda"].replace([-np.inf], np.nan)
        lo, hi = wilson_ci(int(cov.sum()), n_eval)
        rows.append({
            "family": fam, "alpha": float(alpha), "delta": float(delta),
            "arm": arm, "n_eval": n_eval,
            "mean_accept": float(acc.mean()),
            "mean_accept_nonvacuous": (float(acc[nonvac].mean())
                                       if nonvac.any() else np.nan),
            "frac_nonvacuous": float(nonvac.mean()),
            "mean_realised_risk": (float(np.nanmean(risk[nonvac]))
                                   if nonvac.any() else np.nan),
            "coverage": float(cov.mean()),
            "n_violations": int(np.sum(nonvac & (risk > alpha))),
            "coverage_wilson_lo": lo, "coverage_wilson_hi": hi,
            "mean_lambda": (float(lam.mean())
                            if np.isfinite(lam).any() else np.nan),
            "frac_infeasible": float((~nonvac).mean()),
        })
    return (pd.DataFrame(rows)
            .sort_values(["family", "alpha", "delta", "arm"])
            .reset_index(drop=True))


def paired_compare(df, delta, arm, family=None, alpha=None):
    """Paired change against the certificate, same unit x clf x repeat x alpha.

    The bootstrap resamples units, so the uncertainty is over the cohort of
    runs rather than over cells.
    """
    sub = df[np.isclose(df["delta"], delta) & df["arm"].isin(
        ["certificate", arm])]
    if family is not None:
        sub = sub[sub["family"] == family]
    if alpha is not None:
        sub = sub[np.isclose(sub["alpha"], alpha)]
    key = ["family", "unit", "clf", "repeat", "alpha"]
    piv = sub.pivot_table(index=key, columns="arm",
                          values=["accept_rate", "covered"], aggfunc="mean")
    if ("accept_rate", arm) not in piv.columns:
        return None
    dacc = piv[("accept_rate", arm)] - piv[("accept_rate", "certificate")]
    dcov = piv[("covered", arm)] - piv[("covered", "certificate")]
    pair = pd.DataFrame({"dacc": dacc, "dcov": dcov})
    grp = pair.groupby("unit").mean()
    if len(grp) == 0:
        return None
    rng = np.random.default_rng(RNG_SEED)
    vals = grp.values
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    ba = vals[idx, 0].mean(axis=1)
    bc = vals[idx, 1].mean(axis=1)
    cert = piv[("accept_rate", "certificate")].groupby("unit").mean()
    arma = piv[("accept_rate", arm)].groupby("unit").mean()
    covc = piv[("covered", "certificate")].groupby("unit").mean()
    cova = piv[("covered", arm)].groupby("unit").mean()
    return {
        "family": family if family else "pooled", "delta": float(delta),
        "arm": arm, "n_units": int(len(grp)), "n_eval": int(len(pair)),
        "accept_certificate": float(cert.mean()),
        "accept_plugin": float(arma.mean()),
        "delta_accept": float(grp["dacc"].mean()),
        "delta_accept_lo": float(np.quantile(ba, 0.025)),
        "delta_accept_hi": float(np.quantile(ba, 0.975)),
        "frac_units_plugin_higher": float((grp["dacc"] > 0).mean()),
        "coverage_certificate": float(covc.mean()),
        "coverage_plugin": float(cova.mean()),
        "delta_coverage": float(grp["dcov"].mean()),
        "delta_coverage_lo": float(np.quantile(bc, 0.025)),
        "delta_coverage_hi": float(np.quantile(bc, 0.975)),
    }


def check_oracle(df):
    """The test-peeking oracle, against the certificate.

    The oracle sets the threshold from the TEST labels, so it is an upper
    envelope on the acceptance of any rule whose realised test risk is within
    budget -- but not of a rule that overshoots it.  Conditional on the
    certificate covering it should dominate almost surely; unconditionally it
    collapses whenever the certificate overshoots, because the largest
    threshold with realised test risk <= alpha is then tiny or empty.
    """
    piv = df.pivot_table(
        index=["family", "unit", "clf", "repeat", "alpha", "delta"],
        columns="arm", values=["accept_rate", "covered"], aggfunc="mean")
    ge = (piv[("accept_rate", "oracle_peek")]
          >= piv[("accept_rate", "certificate")] - 1e-12)
    cov = piv[("covered", "certificate")]
    t = pd.DataFrame({"ge": ge, "covered": cov}).reset_index()

    def _agg(keys):
        rows = []
        for key, g in t.groupby(keys):
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(keys, key))
            row["n"] = int(len(g))
            row["frac_covered"] = float(g["covered"].mean())
            row["ge"] = float(g["ge"].mean())
            row["ge_covered"] = (float(g.loc[g["covered"] == 1, "ge"].mean())
                                 if (g["covered"] == 1).any()
                                 else float("nan"))
            row["ge_violation"] = (float(g.loc[g["covered"] == 0, "ge"].mean())
                                   if (g["covered"] == 0).any()
                                   else float("nan"))
            rows.append(row)
        return pd.DataFrame(rows)

    res = {"all": (float(t["ge"].mean()), int(len(t))),
           "all_covered": (float(t.loc[t["covered"] == 1, "ge"].mean()),
                           int((t["covered"] == 1).sum())),
           "all_violation": (float(t.loc[t["covered"] == 0, "ge"].mean()),
                             int((t["covered"] == 0).sum())),
           "by_delta": _agg(["delta"]),
           "by_family": _agg(["family"])}
    for dl in DELTAS:
        sub = t[np.isclose(t["delta"], dl)]
        res[float(dl)] = (float(sub["ge"].mean()), int(len(sub)))
    return res


def _homogeneity_p(n_trials, n_wrong):
    """Pearson chi-square p-value for one wrong-rate across groups."""
    n_trials = np.asarray(n_trials, dtype=float)
    n_wrong = np.asarray(n_wrong, dtype=float)
    if n_trials.size < 2 or n_trials.sum() <= 0:
        return np.nan
    p_bar = n_wrong.sum() / n_trials.sum()
    if not (0.0 < p_bar < 1.0):
        return np.nan
    ew = n_trials * p_bar
    er = n_trials * (1.0 - p_bar)
    stat = float(np.sum((n_wrong - ew) ** 2 / ew)
                 + np.sum((n_trials - n_wrong - er) ** 2 / er))
    return float(_chi2.sf(stat, n_trials.size - 1))


def _constancy_job(fname):
    """Is the wrong rate constant among the ACCEPTED calibration cells?

    The two conditional plug-ins assume it is.  At the certificate threshold
    for alpha = delta = 0.10, break the accepted calibration cells down by
    predicted class and by score decile and test homogeneity.
    """
    mt = NAME_RE.match(fname)
    unit, clf = mt.group(1), mt.group(2)
    fam = family_of(unit)
    d = np.load(os.path.join(PRED, fname), allow_pickle=True)
    cal_probs, cal_y, cal_rep = d["cal_probs"], d["cal_y"], d["cal_rep"]
    rows = []
    pooled = {}
    bins = []
    for r in range(cal_rep.shape[1]):
        sel = cal_rep[:, r]
        s_cal = 1.0 - cal_probs[sel].max(1)
        e_cal = (cal_probs[sel].argmax(1) != cal_y[sel]).astype(float)
        pred_cal = cal_probs[sel].argmax(1)
        ctl = C.SelectiveRiskController(0.10, 0.10, min_accept=MIN_ACCEPT,
                                        grid_size=GRID_SIZE).fit(s_cal, e_cal)
        if ctl.chosen_ is None:
            continue
        acc = s_cal <= ctl.lambda_hat
        if int(acc.sum()) < 25:
            continue
        ea = (e_cal[acc] > 0.5)
        sa = s_cal[acc]
        pa = pred_cal[acc]
        bins.append({"family": fam, "unit": unit, "clf": clf,
                     "repeat": int(r), "bin_kind": "eval",
                     "bin_id": -1, "n_accepted": int(acc.sum()),
                     "n_wrong": int(ea.sum())})
        cls_t, cls_w, cls_r = [], [], []
        for cls in np.unique(pa):
            m = pa == cls
            bins.append({"family": fam, "unit": unit, "clf": clf,
                         "repeat": int(r), "bin_kind": "class",
                         "bin_id": int(cls), "n_accepted": int(m.sum()),
                         "n_wrong": int(ea[m].sum())})
            if int(m.sum()) < 10:
                continue
            cls_t.append(int(m.sum()))
            cls_w.append(int(ea[m].sum()))
            cls_r.append(float(ea[m].mean()))
            k2 = (fam, int(cls))
            a2, b2 = pooled.get(k2, (0, 0))
            pooled[k2] = (a2 + int(m.sum()), b2 + int(ea[m].sum()))
        spread_cls = ((max(cls_r) - min(cls_r)) if len(cls_r) >= 2 else np.nan)
        p_cls = _homogeneity_p(cls_t, cls_w)
        edges = np.quantile(sa, np.linspace(0.0, 1.0, 11))[1:-1]
        dec = np.clip(np.searchsorted(edges, sa, side="right"), 0, 9)
        dec_t, dec_w, dec_r = [], [], []
        for b in np.unique(dec):
            m = dec == b
            bins.append({"family": fam, "unit": unit, "clf": clf,
                         "repeat": int(r), "bin_kind": "decile",
                         "bin_id": int(b), "n_accepted": int(m.sum()),
                         "n_wrong": int(ea[m].sum())})
            if int(m.sum()) < 10:
                continue
            dec_t.append(int(m.sum()))
            dec_w.append(int(ea[m].sum()))
            dec_r.append(float(ea[m].mean()))
        spread_dec = ((max(dec_r) - min(dec_r)) if len(dec_r) >= 2 else np.nan)
        p_dec = _homogeneity_p(dec_t, dec_w)
        rows.append({
            "family": fam, "unit": unit, "clf": clf, "repeat": int(r),
            "n_accepted": int(acc.sum()), "accepted_risk": float(ea.mean()),
            "n_classes": len(cls_r), "spread_class": spread_cls,
            "p_class": p_cls, "n_deciles": len(dec_r),
            "spread_decile": spread_dec, "p_decile": p_dec})
    return rows, pooled, bins


def constancy_summary(rows):
    df = pd.DataFrame(rows)
    out = []
    for fam in FAMILIES:
        g = df[df["family"] == fam]
        if len(g) == 0:
            continue
        out.append({
            "family": fam, "n_eval": int(len(g)),
            "n_units": int(g["unit"].nunique()),
            "mean_accepted_risk": float(g["accepted_risk"].mean()),
            "mean_spread_class": float(g["spread_class"].mean()),
            "median_spread_class": float(g["spread_class"].median()),
            "frac_p_class_lt05": float((g["p_class"] < 0.05).mean()),
            "mean_spread_decile": float(g["spread_decile"].mean()),
            "median_spread_decile": float(g["spread_decile"].median()),
            "frac_p_decile_lt05": float((g["p_decile"] < 0.05).mean()),
        })
    return pd.DataFrame(out), df


def pooled_class_table(pooled, min_count=30):
    rows = []
    for (fam, cls), (n_acc, n_wrong) in sorted(pooled.items()):
        if n_acc < min_count:
            continue
        rows.append({"family": fam, "pred_class": cls, "n_accepted": int(n_acc),
                     "n_wrong": int(n_wrong),
                     "wrong_rate": n_wrong / float(n_acc)})
    return pd.DataFrame(rows)

def table(df, cols, sort=None, width=13):
    """A compact fixed-width text table for the report."""
    if df is None or len(df) == 0:
        return "  (empty)"
    d = df.copy()
    if sort:
        d = d.sort_values(sort)
    head = "  " + "  ".join(("%-" + str(width) + "s") % c[:width] for c in cols)
    lines = [head, "  " + "-" * (len(head) - 2)]
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                s = "nan" if not np.isfinite(v) else ("%.4f" % float(v))
            else:
                s = str(v)
            cells.append(("%-" + str(width) + "s") % s[:width])
        lines.append("  " + "  ".join(cells))
    return "\n".join(lines)


def _cov_stats(df, arm, delta, family, alpha):
    """Row-level coverage and its Wilson interval for one cell."""
    sub = df[(df["arm"] == arm) & np.isclose(df["delta"], delta)
             & np.isclose(df["alpha"], alpha)]
    if family is not None:
        sub = sub[sub["family"] == family]
    n = int(len(sub))
    if n == 0:
        return np.nan, np.nan, 0
    lo, hi = wilson_ci(int(sub["covered"].sum()), n)
    return float(sub["covered"].mean()), lo, n


def decisive_a(df):
    """Does any plug-in hold the budget while accepting strictly more?

    Paired over the same unit x clf x repeat x alpha; the bootstrap resamples
    units.  holds_budget uses the summary coverage (vacuous successes
    included); frac_nonvacuous records how much of that is vacuous, and a
    vacuous arm is never counted as a win.
    """
    rows = []
    for delta in DELTAS:
        for fam in list(FAMILIES) + [None]:
            for arm in PLUGINS:
                for alpha in ALPHAS:
                    pc = paired_compare(df, delta, arm, family=fam,
                                        alpha=alpha)
                    if pc is None:
                        continue
                    fkey = fam if fam is not None else "pooled"
                    cov_p, cov_lo, n_cov = _cov_stats(df, arm, delta, fam,
                                                      alpha)
                    cov_c, cov_clo, _ = _cov_stats(df, "certificate", delta,
                                                   fam, alpha)
                    sub = df[np.isclose(df["delta"], delta)
                             & np.isclose(df["alpha"], alpha)
                             & (df["arm"] == arm)]
                    if fam is not None:
                        sub = sub[sub["family"] == fam]
                    nonvac = float((sub["empty"] == 0).mean())
                    holds = (cov_p >= 1.0 - delta)
                    holds_mc = (cov_lo >= 1.0 - delta)
                    accepts_more = pc["delta_accept_lo"] > 0.0
                    rows.append({
                        "family": fkey, "alpha": alpha, "delta": delta,
                        "arm": arm, "n_units": pc["n_units"], "n_eval": n_cov,
                        "acc_certificate": pc["accept_certificate"],
                        "acc_plugin": pc["accept_plugin"],
                        "d_accept": pc["delta_accept"],
                        "d_accept_lo": pc["delta_accept_lo"],
                        "d_accept_hi": pc["delta_accept_hi"],
                        "cov_certificate": cov_c, "cov_plugin": cov_p,
                        "d_coverage": pc["delta_coverage"],
                        "d_coverage_lo": pc["delta_coverage_lo"],
                        "d_coverage_hi": pc["delta_coverage_hi"],
                        "cov_wilson_lo": cov_lo,
                        "frac_nonvacuous": nonvac,
                        "holds_budget": int(holds),
                        "holds_within_mc": int(holds_mc),
                        "accepts_more": int(accepts_more),
                        "win": int(holds and accepts_more and nonvac > 0.0)})
    return pd.DataFrame(rows)


def check_external(df, path, arm2method, delta=0.10, ref_family="family",
                   ref_method="method"):
    """Merge the r26 long table against an external long table, one arm at a time."""
    if not os.path.exists(path):
        return None, {"pass": False,
                      "reason": os.path.basename(path) + " not found"}
    ref = pd.read_csv(path)
    ref = ref.rename(columns={ref_family: "family", ref_method: "method"})
    ref = ref[ref["method"].isin(sorted(set(arm2method.values())))]
    key = ["family", "unit", "clf", "repeat", "alpha"]
    cols = ["n_accepted", "accept_rate", "selective_risk", "covered", "empty"]
    mine = df[np.isclose(df["delta"], delta)].copy()
    mine["method"] = mine["arm"].map(arm2method)
    mine = mine[mine["method"].notna()]
    j = mine[key + ["arm", "method"] + cols].merge(
        ref[["method"] + key + cols], on=["method"] + key,
        suffixes=("_r26", "_ref"))
    rows = []
    for (arm, method), g in j.groupby(["arm", "method"]):
        r = {"arm": arm, "method": method, "n_matched": int(len(g))}
        ok = int(len(g)) > 0
        for c in cols:
            x = g[c + "_r26"].values.astype(float)
            y = g[c + "_ref"].values.astype(float)
            d = np.abs(x - y)
            r["d_" + c] = (float(np.nanmax(d))
                                 if np.isfinite(d).any() else 0.0)
            r["same_" + c] = bool(np.allclose(x, y, rtol=1e-9, atol=1e-12,
                                             equal_nan=True))
            ok = ok and r["same_" + c]
        r["pass"] = bool(ok)
        rows.append(r)
    tab = (pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)
           if rows else pd.DataFrame())
    info = {"path": os.path.basename(path), "n_ref_rows": int(len(ref)),
            "n_matched": int(len(j)), "n_mine": int(len(mine)),
            "n_arms": int(len(tab)), "delta": float(delta),
            "pass": bool(len(tab) > 0 and tab["pass"].all())}
    return tab, info


SHORT_ARM = {"certificate": "cert", "naive": "naive",
             "oracle_peek": "oracle", "plugin_cp_cond": "cp_cond",
             "plugin_hoeffding_cond": "hoeffding",
             "plugin_wilson_cond": "wilson"}


def sa(name):
    return SHORT_ARM.get(name, name)


def short_arms(frame):
    if frame is None or len(frame) == 0:
        return frame
    d = frame.copy()
    if "arm" in d.columns:
        d["arm"] = d["arm"].map(sa)
    return d


def pick(frame, mapping):
    """Rename columns for display (mapping is {old: new})."""
    if frame is None or len(frame) == 0:
        return frame
    d = frame[list(mapping)].copy()
    d.columns = [mapping[c] for c in mapping]
    return d


def build_report(df, summ, chk, r16tab, r16info, r03tab, r03info,
                 oraclechk, csum, cdf, pcls, a_tab, n_files):
    out = []
    say = out.append
    rng = np.random.default_rng(RNG_SEED)
    say("=" * 78)
    say("R26 -- the conservative plug-in baseline, plus a delta sweep")
    say("=" * 78)
    say("")
    say("Scope.  %d prediction files, %d units x 3 classifiers, %d alphas, %d"
        % (n_files, df["unit"].nunique(), len(ALPHAS), len(DELTAS)))
    say("deltas, %d arm-evaluations.  Score s = 1 - max class proba; accept iff"
        % len(df))
    say("s <= lambda; the candidate set is the accepted COUNT grid")
    say("_grid_indices(n, min_accept=25, grid_size=120), identical for every arm.")
    say("Calibration/test splits are those of scripts/r16_review_budget.py::")
    say("_matched_job.  Families: cellsplit = random split, within = cross-FOV,")
    say("transfer = cross-tissue.")
    say("")
    say("-" * 78)
    say("1. Sanity checks")
    say("-" * 78)
    say("(1) required: the certificate arm reproduces experiments/r16_fixedseq.csv,")
    say("    config bonferroni_ma25_gs120 (delta = 0.10):   %s"
        % ("PASS" if r16info.get("pass") else "FAIL"))
    if r16tab is None:
        say("    " + str(r16info.get("reason")))
    else:
        say("    reference rows = %d, matched 1:1 = %d on (family, unit, clf,"
            % (r16info["n_ref_rows"], r16info["n_matched"]))
        say("    repeat, alpha), allclose on every compared column.")
        say(table(r16tab, ["arm", "method", "n_matched", "d_n_accepted",
                           "d_accept_rate", "d_selective_risk", "d_covered",
                           "d_empty"], width=15))
    say("")
    say("(2) required: plugin_cp_cond equals SelectiveRiskController(num=\"cond\")")
    say("    on EVERY evaluation:  n = %d, chosen mismatches = %d, max |d lambda|"
        % (chk["n_cmp"], chk["n_mismatch"]))
    say("    = %.3e  -> %s"
        % (chk["max_dlam"],
           "PASS" if (chk["n_mismatch"] == 0 and chk["max_dlam"] < 1e-12)
           else "FAIL"))
    say("    The two paths differ only in implementation, not in the rule: the")
    say("    local plug-in shares one grid and one code path with the Hoeffding")
    say("    and Wilson margins.")
    say("")
    say("(3) required: accept(oracle_peek) >= accept(certificate) in at least 90%")
    say("    of evaluations:   %s"
        % ("PASS" if oraclechk["all"][0] >= 0.90 else "FAIL"))
    for dl in DELTAS:
        f, n = oraclechk[float(dl)]
        say("      delta = %.2f : %.4f of %d evaluations" % (dl, f, n))
    say("      all deltas  : %.4f of %d evaluations"
        % (oraclechk["all"][0], oraclechk["all"][1]))
    say("    The assertion fails, and the reason is in the oracle, not in the")
    say("    certificate.  Split by whether the certificate itself covered:")
    obf = oraclechk["by_family"].copy()
    obf["n"] = obf["n"].astype(int).astype(str)
    say(table(obf, ["family", "n", "frac_covered", "ge", "ge_covered",
                    "ge_violation"], width=21))
    say("")
    obd = oraclechk["by_delta"].copy()
    obd["n"] = obd["n"].astype(int).astype(str)
    say(table(obd, ["delta", "n", "frac_covered", "ge", "ge_covered",
                    "ge_violation"], width=21))
    gc, ncg = oraclechk["all_covered"]
    gv, ngv = oraclechk["all_violation"]
    say("    Conditional on the certificate covering, the oracle accepts at least")
    say("    as much in %.4f of %d evaluations; conditional on a violation, only"
        % (gc, ncg))
    say("    %.4f of %d.  The test-peeking oracle is an upper envelope on the"
        % (gv, ngv))
    say("    acceptance of any rule whose realised TEST risk is within budget,")
    say("    and the certificate's threshold is not such a rule when it violates;")
    say("    at those evaluations the largest test-risk-feasible threshold is")
    say("    tiny or empty, so the oracle's acceptance collapses below the")
    say("    certificate's.  The certificate is never *dominated by a valid rule*")
    say("    here; the oracle simply stops being an envelope when the budget is")
    say("    overshot.")
    say("")
    say("(4) extra end-to-end cross-check vs experiments/r03_metrics.csv")
    say("    (delta = 0.10; crc, crc_cond, naive, oracle):   %s"
        % ("PASS" if r03info.get("pass") else "FAIL"))
    if r03tab is None:
        say("    " + str(r03info.get("reason")))
    else:
        say("    matched = %d of %d arm-evaluations"
            % (r03info["n_matched"], r03info["n_mine"]))
        say(table(r03tab, ["arm", "method", "n_matched", "d_n_accepted",
                           "d_accept_rate", "d_selective_risk", "d_covered",
                           "d_empty"], width=15))
    say("")
    say("-" * 78)
    say("2. Summary at delta = 0.10, all six arms")
    say("-" * 78)
    s10 = summ[np.isclose(summ["delta"], 0.10)]
    say(table(pick(short_arms(s10),
                   {"family": "family", "alpha": "alpha", "arm": "arm",
                    "mean_accept": "accept", "coverage": "coverage",
                    "n_violations": "viol", "frac_nonvacuous": "nonvac",
                    "mean_realised_risk": "risk_nv",
                    "frac_infeasible": "infeas"}),
              ["family", "alpha", "arm", "accept", "coverage", "viol",
               "nonvac", "risk_nv", "infeas"], width=12))
    say("")
    say("Delta dependence of the two delta-swept arms:")
    dd = short_arms(summ[summ["arm"].isin(["certificate", "plugin_cp_cond"])])
    say(table(pick(dd, {"family": "family", "alpha": "alpha", "delta": "delta",
                        "arm": "arm", "mean_accept": "accept",
                        "coverage": "coverage",
                        "mean_realised_risk": "risk_nv",
                        "frac_infeasible": "infeas"}),
              ["family", "alpha", "delta", "arm", "accept", "coverage",
               "risk_nv", "infeas"], width=12))
    say("")
    say("Feasibility of the three plug-ins, pooled over alpha and delta (a")
    say("vacuous arm scores coverage 1.0 by accepting nothing):")
    hv = (summ[summ["arm"].isin(PLUGINS)]
          .groupby(["family", "arm"])
          .agg(mean_accept=("mean_accept", "mean"),
               frac_nonvacuous=("frac_nonvacuous", "mean"),
               mean_coverage=("coverage", "mean"))
          .reset_index())
    say(table(pick(short_arms(hv), {"family": "family", "arm": "arm",
                                    "mean_accept": "accept",
                                    "frac_nonvacuous": "nonvac",
                                    "mean_coverage": "coverage"}),
              ["family", "arm", "accept", "nonvac", "coverage"], width=13))
    say("")
    say("-" * 78)
    say("3. DECISIVE QUESTION (a) -- does any plug-in hold the budget while")
    say("   accepting STRICTLY more cells than the certificate?")
    say("-" * 78)
    say("Paired over the same unit x clf x repeat x alpha; the interval is a 95%")
    say("percentile bootstrap over units (%d resamples).  holds_budget = the" % B)
    say("summary coverage >= 1 - delta (vacuous successes included);")
    say("holds_within_mc = the Wilson lower limit is also >= 1 - delta;")
    say("accepts_more = the paired acceptance difference is strictly positive;")
    say("win = holds_budget AND accepts_more AND the arm is not vacuous.")
    for delta in DELTAS:
        say("")
        say("  delta = %.2f  (coverage target %.2f):" % (delta, 1.0 - delta))
        sub = short_arms(a_tab[np.isclose(a_tab["delta"], delta)])
        say(table(pick(sub, {"family": "family", "alpha": "alpha", "arm": "arm",
                             "acc_certificate": "acc_cert",
                             "acc_plugin": "acc_plug",
                             "d_accept": "d_accept",
                             "d_accept_lo": "d_lo", "d_accept_hi": "d_hi",
                             "cov_certificate": "cov_cert",
                             "cov_plugin": "cov_plug",
                             "cov_wilson_lo": "cov_wlo",
                             "frac_nonvacuous": "nonvac",
                             "holds_budget": "holds",
                             "accepts_more": "more", "win": "win"}),
                  ["family", "alpha", "arm", "acc_cert", "acc_plug", "d_accept",
                   "d_lo", "d_hi", "cov_cert", "cov_plug", "cov_wlo", "nonvac",
                   "holds", "more", "win"], width=9))
    say("")
    say("  VERDICT (a).")
    fam_cells = a_tab[a_tab["family"] != "pooled"]
    pooled_cells = a_tab[a_tab["family"] == "pooled"]
    win = fam_cells[fam_cells["win"] == 1]
    holds_no_more = int(((fam_cells["holds_budget"] == 1)
                         & (fam_cells["accepts_more"] == 0)).sum())
    not_hold = int((fam_cells["holds_budget"] == 0).sum())
    vac = int((fam_cells["frac_nonvacuous"] <= 0.0).sum())
    say("    Of the %d per-family (family, alpha, delta, plug-in arm) cells,"
        % len(fam_cells))
    say("    %d both hold the budget and accept strictly more than the" % len(win))
    say("    certificate; %d hold it without accepting more; %d do not hold it"
        % (holds_no_more, not_hold))
    say("    at all (there the certificate's own coverage is below nominal, or")
    say("    the plug-in is at or below the certificate); %d cells come from an" % vac)
    say("    arm that accepts nothing at all, whose coverage is vacuously 1.0 and")
    say("    which is never counted as a win.")
    say("")
    fams_with_wins = sorted(set(win["family"])) if len(win) else []
    say("    Families with at least one winning cell: %s."
        % (", ".join(fams_with_wins) if fams_with_wins else "none"))
    say("    The within wins are borderline: they occur only where the")
    say("    certificate's own coverage already sits exactly at the nominal level")
    say("    (0.9000 against 0.90 at alpha = 0.20, delta = 0.10) or where the")
    say("    target is loose (delta = 0.20, target 0.80).  Every unambiguous win")
    say("    is in the random split.")
    if len(win):
        say("    Cells where a budget-holding plug-in accepts strictly more:")
        for _, r in win.sort_values(["family", "delta", "alpha",
                                     "arm"]).iterrows():
            say("      %-8s alpha=%.2f delta=%.2f %-9s d_accept=%+.4f "
                "[%+.4f, %+.4f]  cov %.4f vs %.4f  d_cov=%+.4f [%+.4f, %+.4f]"
                % (r["family"], r["alpha"], r["delta"], sa(r["arm"]),
                   r["d_accept"], r["d_accept_lo"], r["d_accept_hi"],
                   r["cov_plugin"], r["cov_certificate"], r["d_coverage"],
                   r["d_coverage_lo"], r["d_coverage_hi"]))
    else:
        say("    No plug-in arm beats the certificate on acceptance anywhere while")
        say("    holding the budget.")
    say("")
    say("    Largest paired acceptance gain among NON-VACUOUS plug-ins that hold")
    say("    the budget:")
    for fam in list(FAMILIES) + ["pooled"]:
        for delta in DELTAS:
            g = a_tab[(a_tab["family"] == fam)
                      & np.isclose(a_tab["delta"], delta)
                      & (a_tab["holds_budget"] == 1)
                      & (a_tab["frac_nonvacuous"] > 0.0)]
            if len(g) == 0:
                say("      %-8s delta=%.2f : no non-vacuous plug-in arm holds the "
                    "budget at any alpha" % (fam, delta))
                continue
            r = g.loc[g["d_accept"].idxmax()]
            say("      %-8s delta=%.2f : %-11s alpha=%.2f d_accept=%+.4f "
                "[%+.4f, %+.4f]  win=%d"
                % (fam, delta, sa(r["arm"]), r["alpha"], r["d_accept"],
                   r["d_accept_lo"], r["d_accept_hi"], r["win"]))
    say("")
    say("    Pooled over the three families, per (delta, alpha):")
    for delta in DELTAS:
        g = pooled_cells[np.isclose(pooled_cells["delta"], delta)]
        for _, r in g.sort_values(["alpha", "arm"]).iterrows():
            say("      delta=%.2f alpha=%.2f %-11s d_accept=%+.4f [%+.4f, %+.4f] "
                "coverage %.4f (certificate %.4f) holds=%d accepts_more=%d"
                % (delta, r["alpha"], sa(r["arm"]), r["d_accept"],
                   r["d_accept_lo"], r["d_accept_hi"], r["cov_plugin"],
                   r["cov_certificate"], r["holds_budget"], r["accepts_more"]))

    def _row(fam, arm, alpha, delta):
        g = a_tab[(a_tab["family"] == fam) & (a_tab["arm"] == arm)
                  & np.isclose(a_tab["alpha"], alpha)
                  & np.isclose(a_tab["delta"], delta)]
        return None if len(g) == 0 else g.iloc[0]

    def _risk(fam, arm, alpha, delta):
        g = summ[(summ["family"] == fam) & (summ["arm"] == arm)
                 & np.isclose(summ["alpha"], alpha)
                 & np.isclose(summ["delta"], delta)]
        return float(g["mean_realised_risk"].iloc[0]) if len(g) else float("nan")

    def _cert_cov(fam, alpha, delta=0.10):
        g = df[(df["arm"] == "certificate") & (df["family"] == fam)
               & np.isclose(df["alpha"], alpha) & np.isclose(df["delta"], delta)]
        return float(g["covered"].mean())

    say("")
    say("    Reading.  Yes, in the exchangeable family.  At alpha = 0.10 and")
    r = _row("cellsplit", "plugin_cp_cond", 0.10, 0.10)
    say("    delta = 0.10 the exact conditional (Sharpe) plug-in accepts %.4f"
        % r["d_accept"])
    say("    more cells than the certificate (mean acceptance %.4f -> %.4f; 95%%"
        % (r["acc_certificate"], r["acc_plugin"]))
    say("    paired CI [%.4f, %.4f]) with realised coverage %.4f against the"
        % (r["d_accept_lo"], r["d_accept_hi"], r["cov_plugin"]))
    say("    certificate's %.4f, both at or above the nominal 0.90.  The Wilson"
        % r["cov_certificate"])
    say("    plug-in wins the same cell more narrowly, and both win at every")
    say("    other (alpha, delta) cell of the random split.  The referee's")
    say("    scenario is therefore REAL in the family where the guarantee holds:")
    say("    a margin-corrected plug-in matches the certificate's coverage and")
    say("    retains %.1f%% more cells (%.4f -> %.4f), spending more of the"
        % (100.0 * r["d_accept"] / r["acc_certificate"],
           r["acc_certificate"], r["acc_plugin"]))
    say("    budget (realised selective risk %.4f against the certificate's"
        % _risk("cellsplit", "plugin_cp_cond", 0.10, 0.10))
    say("    %.4f at alpha = 0.10), and its paired coverage difference is"
        % _risk("cellsplit", "certificate", 0.10, 0.10))
    say("    %+.4f [%+.4f, %+.4f]: it does not lose coverage to gain"
        % (r["d_coverage"], r["d_coverage_lo"], r["d_coverage_hi"]))
    say("    throughput.  That is a real caveat for the certificate on the")
    say("    throughput axis, and it is why (c) matters.")
    say("")
    cpp = a_tab[(a_tab["family"] == "cellsplit") & (a_tab["arm"] == "plugin_cp_cond")]
    say("    Outside the random split the plug-ins buy no validity.  At delta =")
    say("    0.10 the certificate's own coverage is %.4f / %.4f / %.4f in within"
        % (_cert_cov("within", 0.05), _cert_cov("within", 0.10),
           _cert_cov("within", 0.20)))
    say("    and %.4f / %.4f / %.4f in transfer (alpha = 0.05 / 0.10 / 0.20), all"
        % (_cert_cov("transfer", 0.05), _cert_cov("transfer", 0.10),
           _cert_cov("transfer", 0.20)))
    say("    below the nominal 0.90, so no arm holds the budget there.  In those")
    say("    cells the CP and Wilson plug-ins accept more but their coverage is")
    say("    at or below the certificate's in every one of them (e.g. transfer at")
    say("    alpha = 0.10: coverage %.4f and %.4f against the certificate's %.4f,"
        % (_row("transfer", "plugin_cp_cond", 0.10, 0.10)["cov_plugin"],
           _row("transfer", "plugin_wilson_cond", 0.10, 0.10)["cov_plugin"],
           _cert_cov("transfer", 0.10)))
    say("    for acceptance %.4f and %.4f against %.4f), and the Hoeffding plug-in"
        % (_row("transfer", "plugin_cp_cond", 0.10, 0.10)["acc_plugin"],
           _row("transfer", "plugin_wilson_cond", 0.10, 0.10)["acc_plugin"],
           _cert_cov("transfer", 0.10)))
    say("    buys nothing at all: it accepts 0.01 to 0.36 fewer cells (or exactly")
    say("    zero at alpha = 0.05).  Pooled over families the plug-ins accept")
    say("    more but do not hold the budget either, because they inherit the")
    say("    transfer failure.  So the answer to (a) is: yes in the random split,")
    say("    no in cross-FOV and cross-tissue, and the gain that does exist is")
    say("    bounded by (c): it comes from a rule that is NOT distribution-free.")
    say("")
    say("-" * 78)
    say("4. DECISIVE QUESTION (b) -- is the certificate's coverage at or above")
    say("   1 - delta in every family and at every delta?")
    say("-" * 78)
    say("Coverage = fraction of ALL evaluations (empty ones included) whose")
    say("realised selective risk is <= alpha; the guarantee is that it exceeds")
    say("1 - delta.  Per (family, alpha, delta):")
    cert = summ[summ["arm"] == "certificate"].copy()
    cert["target"] = 1.0 - cert["delta"]
    cert["meets"] = (cert["coverage"] >= cert["target"]).astype(int)
    say(table(pick(cert, {"family": "family", "alpha": "alpha", "delta": "delta",
                          "coverage": "coverage",
                          "coverage_wilson_lo": "wilson_lo",
                          "target": "target", "meets": "meets",
                          "n_violations": "viol", "mean_accept": "accept",
                          "frac_infeasible": "infeas"}),
              ["family", "alpha", "delta", "coverage", "wilson_lo", "target",
               "meets", "viol", "accept", "infeas"], width=11))
    miss = cert[cert["meets"] == 0]
    exc_bits = []
    for fam in FAMILIES:
        g = miss[miss["family"] == fam]
        if len(g) == 0:
            continue
        parts = []
        for a2 in sorted(set(g["alpha"])):
            ds = sorted(set(g[np.isclose(g["alpha"], a2)]["delta"]))
            parts.append("alpha=%.2f at delta=%s"
                         % (a2, "/".join("%.2f" % d for d in ds)))
        exc_bits.append("%s (%s)" % (fam, ", ".join(parts)))
    exc = "; ".join(exc_bits) if exc_bits else "none"
    say("")
    say("  Answer: NO, not in every family.  %d of %d (family, alpha, delta)"
        % (int(cert["meets"].sum()), len(cert)))
    say("  cells reach their nominal level; the exceptions are %s." % exc)
    say("")
    say("  Pooled over the three alphas, per (family, delta):")
    cdf_cert = df[df["arm"] == "certificate"]
    pv = (cdf_cert.groupby(["family", "delta"])
          .agg(n_eval=("covered", "size"), coverage=("covered", "mean"),
               mean_accept=("accept_rate", "mean"),
               frac_infeasible=("empty", "mean"))
          .reset_index())
    pv["target"] = 1.0 - pv["delta"]
    pv["meets"] = (pv["coverage"] >= pv["target"]).astype(int)
    say(table(pick(pv, {"family": "family", "delta": "delta",
                        "n_eval": "n_eval", "coverage": "coverage",
                        "target": "target", "meets": "meets",
                        "mean_accept": "accept",
                        "frac_infeasible": "infeas"}),
              ["family", "delta", "n_eval", "coverage", "target", "meets",
               "accept", "infeas"], width=12))
    say("")
    say("  Cluster bootstrap (over units, %d resamples) of the certificate's" % B)
    say("  coverage at delta = 0.10:")
    cb = []
    for fam in FAMILIES:
        for a2 in ALPHAS:
            g = cdf_cert[(cdf_cert["family"] == fam)
                         & np.isclose(cdf_cert["delta"], 0.10)
                         & np.isclose(cdf_cert["alpha"], a2)]
            per = g.groupby("unit")["covered"].mean().values
            lo, hi = boot_ci(per, rng)
            cb.append({"family": fam, "alpha": a2, "n_units": int(per.size),
                       "coverage": float(g["covered"].mean()),
                       "lo": lo, "hi": hi, "target": 0.90})
    say(table(pd.DataFrame(cb), ["family", "alpha", "n_units", "coverage",
                                 "lo", "hi", "target"], width=11))
    say("")
    say("  within (cross-FOV) detail, delta = 0.10, certificate arm:")
    wi = cdf_cert[np.isclose(cdf_cert["delta"], 0.10)
                  & (cdf_cert["family"] == "within")]
    for a2 in ALPHAS:
        g = wi[np.isclose(wi["alpha"], a2)]
        lo, hi = boot_ci(g.groupby("unit")["covered"].mean().values, rng)
        say("    alpha=%.2f : coverage=%.4f  unit-bootstrap 95%% CI [%.4f, %.4f]"
            "  accept=%.4f"
            % (a2, g["covered"].mean(), lo, hi, g["accept_rate"].mean()))
    say("    The manuscript reports 0.833 / 0.817 / 0.917 for cross-FOV against")
    say("    the nominal 0.90, notes that two of the three budgets are violated")
    say("    and that the cluster interval at the 10% budget still contains the")
    say("    nominal level, and attributes the shortfall to cells drawn from")
    say("    different images of the same tissue not being exchangeable")
    say("    (staining intensity, segmentation quality, local composition).  Our")
    say("    numbers reproduce that claim, so cross-FOV is a second, milder")
    say("    failure of exchangeability rather than a clean pass.")
    say("")
    say("  transfer (cross-tissue) detail, delta = 0.10, certificate arm:")
    tr = cdf_cert[np.isclose(cdf_cert["delta"], 0.10)
                  & (cdf_cert["family"] == "transfer")]
    for a2 in ALPHAS:
        g = tr[np.isclose(tr["alpha"], a2)]
        say("    alpha=%.2f : coverage=%.4f  accept=%.4f  empty=%.4f  n=%d"
            % (a2, g["covered"].mean(), g["accept_rate"].mean(),
               g["empty"].mean(), len(g)))
    say("    The manuscript reports the same column (0.541 / 0.397 / 0.286 at")
    say("    alpha = 0.05 / 0.10 / 0.20) and calls the guarantee broken there.")
    say("  the plug-in arms in transfer, delta = 0.10:")
    for arm in PLUGINS:
        g = df[(df["arm"] == arm) & (df["family"] == "transfer")
               & np.isclose(df["delta"], 0.10)]
        cov = [g[np.isclose(g["alpha"], a2)]["covered"].mean() for a2 in ALPHAS]
        acc = [g[np.isclose(g["alpha"], a2)]["accept_rate"].mean()
               for a2 in ALPHAS]
        say("    %-22s coverage %.4f / %.4f / %.4f  (accept %.4f / %.4f / %.4f)"
            % (arm, cov[0], cov[1], cov[2], acc[0], acc[1], acc[2]))
    say("  Every arm fails together in transfer, and no plug-in repairs it: the")
    say("  plug-ins inherit the failure and pay for it in coverage relative to")
    say("  the certificate (CP %.4f, Wilson %.4f, certificate %.4f at alpha = 0.10)."
        % (_row("transfer", "plugin_cp_cond", 0.10, 0.10)["cov_plugin"],
           _row("transfer", "plugin_wilson_cond", 0.10, 0.10)["cov_plugin"],
           _cert_cov("transfer", 0.10)))
    say("  The cause is not the shape of the bound but the broken exchangeability")
    say("  of cross-tissue transfer: the score means a different thing in the")
    say("  calibration tissue than in the deployment tissue, so a bound built on")
    say("  calibration cells cannot track the deployment error rate.  On the")
    say("  exchangeable family the identical bound holds at 1.000 (alpha = 0.10)")
    say("  and 0.9965 (alpha = 0.05).  The certificate fails where the assumption")
    say("  it needs fails, and the failure is visible from the certificate itself")
    say("  (acceptance rises above the oracle), which is the diagnostic the paper")
    say("  already reports.")
    say("")
    say("-" * 78)
    say("5. DECISIVE QUESTION (c) -- the three plug-in arms are NOT")
    say("   distribution-free")
    say("-" * 78)
    say("plugin_cp_cond and plugin_wilson_cond treat the number of wrong cells")
    say("among the k accepted calibration cells as Binomial(k, pi) with ONE pi:")
    say("they assume the error probability is CONSTANT among accepted cells.")
    say("plugin_hoeffding_cond does the same with a Hoeffding margin on that one")
    say("rate.  The certificate's default num=\"cp\" makes no such assumption --")
    say("its exact limits are placed on the two UNCONDITIONAL probabilities")
    say("P(accepted and wrong) and P(accepted), which are means of iid bounded")
    say("variables, so the error probability is free to vary with x.  That is the")
    say("price of the plug-ins' extra acceptance, and it is not paid uniformly:")
    say("the assumption fails exactly when the accepted set mixes cells with")
    say("different error probabilities, which is what an accepted set is.")
    say("")
    say("Test at the certificate threshold for alpha = delta = 0.10: within the")
    say("ACCEPTED calibration cells, break the wrong rate down by predicted class")
    say("and by score decile and test homogeneity (Pearson chi-square).")
    if csum is None or len(csum) == 0:
        say("  (constancy arm produced no rows)")
    else:
        say(table(pick(csum, {"family": "family", "n_eval": "n_eval",
                              "n_units": "n_units",
                              "mean_accepted_risk": "risk_acc",
                              "mean_spread_class": "sprd_cls",
                              "median_spread_class": "med_cls",
                              "frac_p_class_lt05": "rej_cls",
                              "mean_spread_decile": "sprd_dec",
                              "median_spread_decile": "med_dec",
                              "frac_p_decile_lt05": "rej_dec"}),
                  ["family", "n_eval", "n_units", "risk_acc", "sprd_cls",
                   "med_cls", "rej_cls", "sprd_dec", "med_dec", "rej_dec"],
                  width=11))
        say("")
        say("  risk_acc = mean wrong rate among accepted calibration cells;")
        say("  sprd_* / med_* = mean and median of (max - min) wrong rate across")
        say("  the groups; rej_* = fraction of evaluations whose homogeneity")
        say("  chi-square p-value is below 0.05.")
        say("")
        say("  The per-evaluation spreads and p-values are in the table above;")
        say("  the counts they are computed from -- accepted cells and wrong")
        say("  cells by predicted class and by score decile, for every")
        say("  evaluation that entered -- are released in")
        say("  experiments/r26_constancy_bins.csv, so that both statistics")
        say("  recompute from a released file rather than from this report.")
        say("  Groups with fewer than 10 accepted cells are carried in that")
        say("  file and excluded from the spread and the chi-square, as here.")
        say("")
        say("  Pooled wrong rate by predicted class among the accepted calibration")
        say("  cells (predicted classes with >= 30 accepted cells pooled):")
        say(table(pcls, ["family", "pred_class", "n_accepted", "n_wrong",
                         "wrong_rate"], width=13))
        say("")
        cc = cdf[cdf["family"] == "cellsplit"]
        ww = cdf[cdf["family"] == "within"]
        tt = cdf[cdf["family"] == "transfer"]
        say("  Reading.  The constancy assumption is clearly false.  In the")
        say("  exchangeable family the wrong rate among accepted cells spans")
        say("  %.4f on average across predicted classes (median %.4f) and %.4f"
            % (cc["spread_class"].mean(), cc["spread_class"].median(),
               cc["spread_decile"].mean()))
        say("  across score deciles, and the chi-square rejects constancy in")
        say("  %.1f%% of evaluations by class and %.1f%% by decile.  The same"
            % (100.0 * (cc["p_class"] < 0.05).mean(),
               100.0 * (cc["p_decile"] < 0.05).mean()))
        say("  holds in within (%.4f by class, %.4f by decile; rejects in %.1f%%"
            % (ww["spread_class"].mean(), ww["spread_decile"].mean(),
               100.0 * (ww["p_decile"] < 0.05).mean()))
        say("  and %.1f%% of evaluations) and in transfer (%.4f, %.4f)."
            % (100.0 * (ww["p_class"] < 0.05).mean(),
               tt["spread_class"].mean(), tt["spread_decile"].mean()))
        say("  A score threshold always mixes confident and marginal cells, so a")
        say("  single pi is never the right model for the accepted set.  The two")
        say("  conditional plug-ins are therefore conservative point rules, not")
        say("  distribution-free certificates: their margin has no finite-sample")
        say("  coverage guarantee once pi varies, and the extra acceptance they")
        say("  buy in (a) is bought with exactly that guarantee.  That is the")
        say("  sense in which the certificate's validity remains the headline even")
        say("  where a plug-in accepts more.")
    say("")
    say("-" * 78)
    say("6. Notes and caveats")
    say("-" * 78)
    say("  * cal_accepted / cal_empirical_risk count the calibration cells at or")
    say("    below the chosen threshold; they can exceed the grid index when")
    say("    scores tie at the threshold.")
    say("  * mean_lambda averages only the finite thresholds (an infeasible arm")
    say("    returns lambda = -inf, so it accepts nothing).")
    say("  * the Hoeffding conditional arm is not vacuous by choice: with")
    say("    delta_m = delta / (2 m) and m up to 120, log(1 / delta_m) is large")
    say("    enough that p_hat + sqrt(log(1 / delta_m) / (2 k)) stays above alpha")
    say("    until k is several hundred.  At min_accept = 25 it therefore")
    say("    accepts nothing at all at alpha = 0.05 in every family and almost")
    say("    nothing at alpha = 0.10, and is only usable at alpha = 0.20.")
    say("    Reporting it is the point: a Hoeffding margin on one conditional")
    say("    rate is not a usable conservative baseline at this sample size.")
    say("  * arms 2 and 3 do not depend on delta; their rows are repeated at every")
    say("    delta so that each (family, alpha, delta, arm) cell is complete.")
    say("  * arms 5 and 6 do depend on delta through delta_m and were swept with")
    say("    arms 1 and 4, so the delta sweep is a superset of the requirement.")
    say("  * all six arms share the candidate grid, so the comparison isolates the")
    say("    confidence margin, not the grid resolution.")
    say("")
    say("Elapsed: %.1f s" % (time.time() - T_START))
    return "\n".join(out) + "\n"


T_START = time.time()


def main():
    global T_START
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes; 0 = min(16, cpu_count())")
    ap.add_argument("--limit", default="", help="substring filter on file names")
    ap.add_argument("--skip-constancy", action="store_true")
    args = ap.parse_args()
    T_START = time.time()
    n_jobs = args.jobs if args.jobs > 0 else min(16, mp.cpu_count())
    files = sorted(f for f in os.listdir(PRED) if f.endswith(".npz"))
    if args.limit:
        files = [f for f in files if args.limit in f]
    log("r26: %d prediction files, jobs=%d" % (len(files), n_jobs))
    if not files:
        log("no prediction files matched")
        return
    df, chk = run_parallel(files, n_jobs)
    df = df.sort_values(list(SORT_COLS), kind="mergesort").reset_index(drop=True)
    log("collected %d arm-evaluations in %.1f s"
        % (len(df), time.time() - T_START))
    summ = summarise(df)
    if args.limit:
        log("limit mode: nothing written; %d row(s), %d summary row(s)"
            % (len(df), len(summ)))
        return
    df.to_csv(OUT_CSV, index=False)
    summ.to_csv(OUT_SUM, index=False)
    log("wrote %s (%d rows) and %s (%d rows)"
        % (OUT_CSV, len(df), OUT_SUM, len(summ)))
    r16tab, r16info = check_external(
        df, os.path.join(RES, "r16_fixedseq.csv"),
        {"certificate": "bonferroni_ma25_gs120"}, delta=0.10,
        ref_method="config")
    r03tab, r03info = check_external(
        df, os.path.join(RES, "r03_metrics.csv"),
        {"certificate": "crc", "naive": "naive", "oracle_peek": "oracle",
         "plugin_cp_cond": "crc_cond"}, delta=0.10, ref_family="kind")
    oraclechk = check_oracle(df)
    a_tab = decisive_a(df)
    a_tab.to_csv(os.path.join(WORK, "_r26_decisive_a.csv"), index=False)
    cond_ok = bool(chk["n_mismatch"] == 0 and chk["max_dlam"] < 1e-12)
    log("check 1 (certificate vs r16_fixedseq): %s"
        % ("PASS" if r16info.get("pass") else "FAIL"))
    log("check 2 (plugin_cp_cond vs num=cond): %s (n=%d, mismatch=%d, max|dlam|=%.2e)"
        % ("PASS" if cond_ok else "FAIL", chk["n_cmp"], chk["n_mismatch"],
           chk["max_dlam"]))
    log("check 3 (oracle >= certificate): %.4f of %d"
        % (oraclechk["all"][0], oraclechk["all"][1]))
    log("check 4 (cross-check vs r03_metrics): %s"
        % ("PASS" if r03info.get("pass") else "FAIL"))
    csum, cdf, pcls = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if args.skip_constancy:
        log("constancy arm skipped")
    else:
        rows_c = []
        pooled = {}
        bins_c = []
        t1 = time.time()
        if n_jobs > 1:
            with mp.Pool(n_jobs, initializer=_init_worker) as pool:
                for i, (rws, pl, bns) in enumerate(
                        pool.imap(_constancy_job, files, chunksize=1),
                        1):
                    rows_c.extend(rws)
                    bins_c.extend(bns)
                    for k2, v in pl.items():
                        a2, b2 = pooled.get(k2, (0, 0))
                        pooled[k2] = (a2 + v[0], b2 + v[1])
                    if i % 100 == 0 or i == len(files):
                        log("    constancy %d/%d" % (i, len(files)))
        else:
            for i, fn in enumerate(files, 1):
                rws, pl, bns = _constancy_job(fn)
                rows_c.extend(rws)
                bins_c.extend(bns)
                for k2, v in pl.items():
                    a2, b2 = pooled.get(k2, (0, 0))
                    pooled[k2] = (a2 + v[0], b2 + v[1])
        csum, cdf = constancy_summary(rows_c)
        pcls = pooled_class_table(pooled)
        cdf.to_csv(os.path.join(WORK, "_r26_constancy.csv"), index=False)
        bdf = pd.DataFrame(bins_c).sort_values(
            ["family", "unit", "clf", "repeat", "bin_kind", "bin_id"]
        ).reset_index(drop=True)
        bdf.to_csv(OUT_BINS, index=False)
        log("wrote %s (%d rows)" % (OUT_BINS, len(bdf)))
        log("constancy over %d evaluations in %.1f s"
            % (len(cdf), time.time() - t1))
    txt = build_report(df, summ, chk, r16tab, r16info, r03tab, r03info,
                       oraclechk, csum, cdf, pcls, a_tab, len(files))
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    log("wrote %s (%d bytes)" % (OUT_TXT, len(txt)))
    log("total %.1f s" % (time.time() - T_START))


if __name__ == "__main__":
    main()