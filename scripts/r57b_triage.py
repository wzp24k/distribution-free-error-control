"""R57b -- how to spend the review queue the certificate generates.

The certificate commits a fraction of the batch and routes the rest to a human.
This script asks what that human should look at first, using only the released
per-cell probabilities of the random-split family, and it writes the curve into a
table so the manuscript can point at it rather than describe it.

For every evaluation unit and every calibration repeat the deferred set is
recovered from the certificate, the deferred cells are ordered by the
certificate's own score (largest uncertainty first, which is the order the
controller's threshold already defines), and the script records

  * the fraction of the deferred set's errors found in the first q of the queue,
    for q in 0.10, 0.25, 0.50, 0.75, pooled over cells and averaged over units;
  * the error rate inside the first and last quartile of the queue;
  * the same three statistics for an uninformed pass, which is q by construction.

Nothing is refitted. The mean-probability, entropy and cross-learner-disagreement
orderings were also tried and are no better than the certificate's own score, so
the table reports the score ordering alone.

Output: experiments/r57_triage.csv
"""

import glob
import io
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from errctl.conformal import SelectiveRiskController  # noqa: E402

PRED = os.path.join(ROOT, "work", "predictions")
OUT = os.path.join(ROOT, "experiments", "r57_triage.csv")
QS = (0.10, 0.25, 0.50, 0.75)
ALPHAS = (0.05, 0.10, 0.20)


def log(*p):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in p)), flush=True)


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(PRED, "cellsplit_*__hgb.npz")))
    for f in files:
        base = os.path.basename(f)
        d = np.load(f, allow_pickle=True)
        s_all = 1.0 - d["test_probs"].max(1)
        e_all = (d["test_probs"].argmax(1) != d["test_y"]).astype(float)
        sc_all = 1.0 - d["cal_probs"].max(1)
        ec_all = (d["cal_probs"].argmax(1) != d["cal_y"]).astype(float)
        for r in range(d["cal_rep"].shape[1]):
            ci, ti = d["cal_rep"][:, r], d["test_rep"][:, r]
            for a in ALPHAS:
                m = SelectiveRiskController(a, 0.10).fit(sc_all[ci], ec_all[ci]).accept(s_all[ti])
                nd = int((~m).sum())
                if nd == 0:
                    continue
                err = e_all[ti][~m].astype(bool)
                ne = int(err.sum())
                order = np.argsort(-s_all[ti][~m], kind="stable")
                cum = np.cumsum(err[order])
                q25 = max(1, int(round(0.25 * nd)))
                top = err[order[:q25]].mean()
                bot = err[order[-q25:]].mean()
                rec = {"unit": base.replace("cellsplit_", "").replace("__hgb.npz", ""),
                       "rep": r, "alpha": a, "n_deferred": nd, "n_errors": ne,
                       "deferred_error_rate": float(err.mean()),
                       "err_rate_top_quartile": float(top),
                       "err_rate_bottom_quartile": float(bot)}
                for q in QS:
                    k = max(1, int(round(q * nd)))
                    rec["capture_q%.2f" % q] = float(cum[k - 1]) / ne if ne else np.nan
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    out = []
    for a in ALPHAS:
        s = df[np.isclose(df.alpha, a)]
        tot_def, tot_err = int(s.n_deferred.sum()), int(s.n_errors.sum())
        rec = {"alpha": a, "evals": int(len(s)), "pooled_deferred": tot_def, "pooled_errors": tot_err,
               "pooled_deferred_error_rate": float(s.n_errors.sum() / max(s.n_deferred.sum(), 1)),
               "unit_deferred_error_rate": float(s.deferred_error_rate.mean()),
               "unit_err_rate_top_quartile": float(s.err_rate_top_quartile.mean()),
               "unit_err_rate_bottom_quartile": float(s.err_rate_bottom_quartile.mean())}
        for q in QS:
            c = "capture_q%.2f" % q
            w = s.dropna(subset=[c])
            num = float((w[c] * w.n_errors).sum())
            rec["pooled_capture_q%.2f" % q] = num / max(float(w.n_errors.sum()), 1.0)
            rec["unit_capture_q%.2f" % q] = float(w[c].mean())
        out.append(rec)
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(ROOT, "experiments", "r57_triage_summary.csv"), index=False)
    log(summ.round(4).to_string())
    log("wrote %s (%d rows)" % (OUT, len(df)))


if __name__ == "__main__":
    main()
