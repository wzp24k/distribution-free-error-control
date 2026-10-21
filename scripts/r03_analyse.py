"""Turn stored predictions into the paper's headline numbers.

For every stored unit (cohort or transfer pair) x classifier x calibration
repeat x nominal budget alpha we evaluate:

* ``accept_all``        no abstention at all -- the trivial baseline.
* ``naive``             threshold placed where the *empirical* calibration risk
                        equals alpha. What a practitioner does with an
                        uncalibrated softmax. No guarantee.
* ``temp_naive``        same, after temperature scaling on the calibration set.
* ``crc``               our controller (empirical Bernstein numerator,
                        Clopper-Pearson denominator).
* ``crc_hoeffding``     the same controller with a Hoeffding numerator, i.e.
                        what a cruder bound buys.
* ``crc_temp``          our controller applied to temperature-scaled scores.
* ``crc_weighted``      our controller with covariate-shift importance weights.
* ``crc_mondrian``      classwise (Mondrian) thresholds.
* ``conformal_set``     split-conformal prediction sets; accept singletons only.

Output: experiments/r03_metrics.csv (tidy, one row per evaluation).
"""

import argparse
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, paths
from errctl.calibration import TemperatureScaler, expected_calibration_error, nll
from errctl.weights import density_ratio, effective_sample_size

PRED_DIR = os.path.join(paths.WORK, "predictions")
OUT = os.path.join(paths.RESULTS, "r03_metrics.csv")


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)), flush=True)


def temp_scale(cal_probs, cal_y, test_probs):
    ok = cal_y >= 0
    ts = TemperatureScaler().fit(cal_probs[ok], cal_y[ok])
    return ts.transform(cal_probs), ts.transform(test_probs), ts.T


def evaluate(s_cal, e_cal, s_te, e_te, p_cal, p_te, alpha, delta, w_cal=None):
    """All methods for one (unit, clf, repeat, alpha)."""
    out = {}

    def record(name, accepted, extra=None):
        acc = np.asarray(accepted, dtype=bool)
        n_acc = int(acc.sum())
        risk = float(e_te[acc].mean()) if n_acc else np.nan
        row = {
            "method": name, "n_accepted": n_acc,
            "accept_rate": n_acc / max(len(acc), 1),
            "selective_risk": risk,
            # an empty accepted set leaves the risk undefined and satisfies the
            # budget vacuously; counting it as a failure would punish the
            # controller for being conservative rather than for being wrong.
            "covered": int((not np.isfinite(risk)) or risk <= alpha),
            "empty": int(n_acc == 0),
            "n_wrong_accepted": int(e_te[acc].sum()) if n_acc else 0,
        }
        if extra:
            row.update(extra)
        out[name] = row

    record("accept_all", np.ones_like(e_te, dtype=bool))

    acc_o, lam_o = oracle_acceptance(s_te, e_te, alpha)
    record("oracle", acc_o, {"lambda": lam_o})

    lam = C.naive_threshold(s_cal, e_cal, alpha)
    record("naive", s_te <= lam, {"lambda": lam})

    ctl = C.SelectiveRiskController(alpha, delta).fit(s_cal, e_cal)
    record("crc", ctl.accept(s_te), {"lambda": ctl.lambda_hat})

    ctl_c = C.SelectiveRiskController(alpha, delta, num="cond").fit(s_cal, e_cal)
    record("crc_cond", ctl_c.accept(s_te), {"lambda": ctl_c.lambda_hat})

    ctl_e = C.SelectiveRiskController(alpha, delta, num="eb",
                                      bound="bernstein").fit(s_cal, e_cal)
    record("crc_eb", ctl_e.accept(s_te), {"lambda": ctl_e.lambda_hat})

    ctl_h = C.SelectiveRiskController(alpha, delta, num="eb",
                                      bound="hoeffding").fit(s_cal, e_cal)
    record("crc_hoeffding", ctl_h.accept(s_te), {"lambda": ctl_h.lambda_hat})

    if w_cal is not None:
        ctl_w = C.SelectiveRiskController(alpha, delta, weighted=True).fit(s_cal, e_cal, w_cal)
        record("crc_weighted", ctl_w.accept(s_te),
               {"lambda": ctl_w.lambda_hat, "ess": effective_sample_size(w_cal)})

    mon = C.MondrianSelectiveRiskController(alpha, delta).fit(s_cal, e_cal, p_cal)
    record("crc_mondrian", mon.accept(s_te, p_te))

    return out


def conformal_set_acceptance(cal_probs, cal_y, test_probs, alpha):
    ok = cal_y >= 0
    score = 1.0 - cal_probs[ok, cal_y[ok]]
    q = C.conformal_quantile(score, alpha)
    sizes, top = C.prediction_set_sizes(test_probs, q)
    return sizes == 1, q, sizes


def oracle_acceptance(s_te, e_te, alpha, min_accept=25):
    """Efficiency reference: the largest threshold that meets the budget when
    the *test* cells themselves are used to pick it.

    Not attainable in practice -- it peeks at the labels of the cells being
    annotated -- and is reported only as an upper bound on the acceptance rate
    an honest method can reach.
    """
    order = np.argsort(s_te, kind="mergesort")
    s, e = s_te[order], e_te[order]
    cum = np.cumsum(e)
    k = np.arange(1, s.size + 1)
    ok = (cum / k <= alpha) & (k >= min_accept)
    if not ok.any():
        return np.zeros(s_te.size, dtype=bool), -np.inf
    lam = float(s[np.nonzero(ok)[0][-1]])
    return s_te <= lam, lam


def process_file(fname, alphas, delta):
    """Every evaluation for one stored (unit, classifier) file.

    Returns ``(rows, ece_rows)``; the split sizes ride along in each row so that
    downstream scripts can express acceptance as a fraction of the pool.
    """
    rows, ece_rows = [], []
    d = np.load(os.path.join(PRED_DIR, fname), allow_pickle=True)
    m = re.match(r"(.*)__(logreg|rf|hgb)\.npz$", fname)
    unit, clf = m.group(1), m.group(2)
    kind = ("cellsplit" if unit.startswith("cellsplit")
            else "within" if unit.startswith("within") else "transfer")
    classes = list(d["classes"])
    test_probs, test_y = d["test_probs"], d["test_y"]
    cal_probs, cal_y, cal_rep = d["cal_probs"], d["cal_y"], d["cal_rep"]
    cal_X, test_X = d["cal_X"], d["test_X"]
    test_rep = d["test_rep"] if "test_rep" in d.files else None

    t_pred = test_probs.argmax(1)
    e_te_all = (t_pred != test_y).astype(float)
    ece_rows.append({
        "unit": unit, "clf": clf, "kind": kind, "domain": "target",
        "ece": expected_calibration_error(test_probs.max(1), 1 - e_te_all),
        "acc": float(1 - e_te_all.mean()), "n": len(test_y),
    })
    c_pred = cal_probs.argmax(1)
    e_ca_all = (c_pred != cal_y).astype(float)
    ece_rows.append({
        "unit": unit, "clf": clf, "kind": kind, "domain": "source_cal",
        "ece": expected_calibration_error(cal_probs.max(1), 1 - e_ca_all),
        "acc": float(1 - e_ca_all.mean()), "n": len(cal_y),
    })

    # density ratio once per unit (the domain classifier benefits from the
    # whole pool); repeat subsets then just index into it.
    w_pool = density_ratio(cal_X, test_X) if kind == "transfer" else None

    n_rep = cal_rep.shape[1]
    for r in range(n_rep):
        sel = cal_rep[:, r]
        s_cal = 1.0 - cal_probs[sel].max(1)
        p_cal = cal_probs[sel].argmax(1)
        e_cal = (p_cal != cal_y[sel]).astype(float)
        te_sel = test_rep[:, r] if test_rep is not None else np.ones(len(test_y), bool)
        s_te = 1.0 - test_probs[te_sel].max(1)
        p_te = test_probs[te_sel].argmax(1)
        e_te = (p_te != test_y[te_sel]).astype(float)

        w_cal = w_pool[sel] if w_pool is not None else None

        for alpha in alphas:
            res = evaluate(s_cal, e_cal, s_te, e_te, p_cal, p_te, alpha, delta, w_cal)

            acc_mask, q, sizes = conformal_set_acceptance(cal_probs[sel], cal_y[sel],
                                                         test_probs[te_sel], alpha)
            n_acc = int(acc_mask.sum())
            risk = float(e_te[acc_mask].mean()) if n_acc else np.nan
            res["conformal_set"] = {
                "method": "conformal_set", "n_accepted": n_acc,
                "accept_rate": n_acc / max(len(acc_mask), 1),
                "selective_risk": risk,
                "covered": int((not np.isfinite(risk)) or risk <= alpha),
                "empty": int(n_acc == 0),
                "n_wrong_accepted": int(e_te[acc_mask].sum()) if n_acc else 0,
                "lambda": q, "mean_set_size": float(sizes.mean()),
            }

            # temperature-scaled variants
            cp, tp, T = temp_scale(cal_probs[sel], cal_y[sel], test_probs[te_sel])
            s_cal_t = 1.0 - cp.max(1)
            e_cal_t = (cp.argmax(1) != cal_y[sel]).astype(float)
            s_te_t = 1.0 - tp.max(1)
            lam_t = C.naive_threshold(s_cal_t, e_cal_t, alpha)
            acc_t = s_te_t <= lam_t
            n_t = int(acc_t.sum())
            res["temp_naive"] = {
                "method": "temp_naive", "n_accepted": n_t,
                "accept_rate": n_t / max(len(acc_t), 1),
                "selective_risk": float(e_te[acc_t].mean()) if n_t else np.nan,
                "covered": int(n_t == 0 or e_te[acc_t].mean() <= alpha),
                "empty": int(n_t == 0),
                "n_wrong_accepted": int(e_te[acc_t].sum()),
                "lambda": lam_t, "temperature": T,
            }
            ctl_t = C.SelectiveRiskController(alpha, delta).fit(s_cal_t, e_cal_t)
            acc_t2 = ctl_t.accept(s_te_t)
            n_t2 = int(acc_t2.sum())
            res["crc_temp"] = {
                "method": "crc_temp", "n_accepted": n_t2,
                "accept_rate": n_t2 / max(len(acc_t2), 1),
                "selective_risk": float(e_te[acc_t2].mean()) if n_t2 else np.nan,
                "covered": int(n_t2 == 0 or e_te[acc_t2].mean() <= alpha),
                "empty": int(n_t2 == 0),
                "n_wrong_accepted": int(e_te[acc_t2].sum()),
                "lambda": ctl_t.lambda_hat, "temperature": T,
            }

            for name, r_ in res.items():
                rows.append({"unit": unit, "clf": clf, "kind": kind,
                             "repeat": r, "alpha": alpha,
                             "n_cal_pool": int(len(cal_y)),
                             "n_cal_used": int(sel.sum()),
                             "n_test_pool": int(len(test_y)),
                             "n_test_eval": int(te_sel.sum()), **r_})
    return rows, ece_rows


def _run_file(job):
    fname, alphas, delta = job
    rows, ece_rows = process_file(fname, alphas, delta)
    return rows, ece_rows, fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--limit", default="")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes; 1 runs in the caller")
    ap.add_argument("--out", default="r03", help="output basename in experiments/")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    delta = args.delta

    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".npz"))
    if args.limit:
        files = [f for f in files if args.limit in f]
    jobs = [(f, alphas, delta) for f in files]
    log("analysing %d stored (unit, classifier) files with %d worker(s)"
        % (len(jobs), max(1, args.jobs)))

    rows, ece_rows = [], []
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            for i, (rws, ece, fname) in enumerate(
                    pool.imap(_run_file, jobs, chunksize=1), 1):
                rows.extend(rws)
                ece_rows.extend(ece)
                if i % 25 == 0 or i == len(jobs):
                    log("  %d/%d  %s" % (i, len(jobs), fname))
    else:
        for i, job in enumerate(jobs, 1):
            rws, ece, fname = _run_file(job)
            rows.extend(rws)
            ece_rows.extend(ece)
            if i % 25 == 0 or i == len(jobs):
                log("  %d/%d  %s" % (i, len(jobs), fname))

    base = os.path.join(paths.RESULTS, args.out)
    df = pd.DataFrame(rows)
    df.to_csv(base + "_metrics.csv", index=False)
    pd.DataFrame(ece_rows).to_csv(base + "_ece.csv", index=False)
    log("wrote %s_metrics.csv (%d rows)" % (base, len(df)))

    summ = (df.groupby(["kind", "method", "alpha"])
              .agg(mean_accept=("accept_rate", "mean"),
                   mean_risk=("selective_risk", "mean"),
                   coverage=("covered", "mean"),
                   abstain_rate=("empty", "mean"),
                   n_eval=("covered", "size"))
              .reset_index())
    summ.to_csv(base + "_summary.csv", index=False)
    log(summ.to_string(index=False))


if __name__ == "__main__":
    main()


