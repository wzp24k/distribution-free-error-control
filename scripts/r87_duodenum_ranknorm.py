"""R87 -- duodenum cross-panel run with per-marker rank normalisation.

R85 used a per-domain z-score: log1p, then subtract the domain mean and
divide by the domain standard deviation.  That forces the target marker
distribution onto the source location and scale, which is the wrong
assumption when the two panels differ in antibody, titration and detector
gain.  R87 keeps the R85 pipeline but sets R85_NORM=rank, so each marker is
mapped to its within-domain empirical quantile and then to a standard normal
score.  The transform is label-free and is fitted separately inside each
domain.

The run also carries a fourth base model, the equal-weight mean of the three
classifier probabilities, recorded as `ens`.  Summaries are per classifier,
so the ensemble and the three base models are not averaged together.
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import r85_duodenum_crosspanel as r85  # noqa: E402

os.environ["R85_NORM"] = "rank"


def summarise_by_clf(df):
    out = []
    for (arm, clf, alpha, k), g in df.groupby(
            ["arm", "clf", "alpha", "k"], sort=True):
        n = len(g)
        n_cov = int(g["covered"].sum())
        lo, hi = r85.wilson(n_cov, n)
        nv = g[g["empty"] == 0]
        out.append(dict(
            arm=arm, clf=clf, alpha=alpha, k=k, n_eval=n,
            coverage=n_cov / n, coverage_lo=lo, coverage_hi=hi,
            frac_nonvacuous=float((g["empty"] == 0).mean()),
            mean_accept=float(g["accept_rate"].mean()),
            mean_accept_nonvac=(float(nv["accept_rate"].mean())
                                if len(nv) else np.nan),
            risk_nonvac=(float(nv["selective_risk"].mean())
                         if len(nv) else np.nan),
            cp_valid=float(g["valid"].mean()),
            mean_cp_upper=float(g["cp_upper"].mean()),
            acc=float(g["acc"].mean()),
        ))
    return out


def main():
    out_evals = os.path.join(ROOT, "experiments",
                             "r87_duodenum_ranknorm.csv")
    out_summary = os.path.join(ROOT, "experiments",
                               "r87_duodenum_ranknorm_summary.csv")
    out_report = os.path.join(ROOT, "work",
                              "r87_duodenum_ranknorm_report.txt")
    r85.EVALS = out_evals
    r85.SUMMARY = out_summary
    r85.REPORT = out_report
    r85.summarise = lambda df: pd.DataFrame(summarise_by_clf(df))

    argv = ["r85", "--jobs", os.environ.get("R87_JOBS", "3"),
            "--arms", "cross,srcnull,cross_roi,within_target",
            "--clfs", "logreg,rf,hgb,ens"]
    old = sys.argv
    sys.argv = argv
    try:
        r85.main()
    finally:
        sys.argv = old

    sm = pd.read_csv(out_summary)
    A = ["R87 -- duodenum cross-panel, per-marker rank normalisation",
         "=" * 72,
         "Transform: log1p then within-domain per-marker rank -> N(0,1).",
         "Base models: logreg, rf, hgb and their equal-weight ensemble (ens).",
         ""]
    for _, r in sm.iterrows():
        if not np.isclose(r.alpha, 0.10):
            continue
        A.append("%-13s %-6s k=%5d n=%3d cov %.3f [%.3f, %.3f] nonvac %.3f "
                 "accept %.3f risk %.3f acc %.3f"
                 % (r.arm, r.clf, r.k, r.n_eval, r.coverage, r.coverage_lo,
                    r.coverage_hi, r.frac_nonvacuous, r.mean_accept,
                    r.risk_nonvac, r.acc))
    open(out_report, "w", encoding="utf-8").write("\n".join(A) + "\n")
    print("\n".join(A))


if __name__ == "__main__":
    main()