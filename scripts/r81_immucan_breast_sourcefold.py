"""R81 -- source-image-fold robustness for the IMMUcan BREAS repair.

The primary r80 endpoint holds the source model fixed and varies the 1,000-label
target draw. This script holds the target protocol fixed and varies the source
image fold, so the positive result is not a single source split. It is a
post-primary robustness check, not a new confirmatory endpoint.

Outputs
  experiments/r81_immucan_breast_sourcefold.csv
  experiments/r81_immucan_breast_sourcefold_summary.csv
  work/r81_report.txt
"""

import argparse
import multiprocessing as mp
import os
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import r02_run_experiments as R02  # noqa: E402
import r72_panel_repair as R72  # noqa: E402
import r80_immucan_breast_repair as R80  # noqa: E402

CSV = os.path.join(ROOT, "experiments",
                   "r81_immucan_breast_sourcefold.csv")
SUMMARY = os.path.join(ROOT, "experiments",
                       "r81_immucan_breast_sourcefold_summary.csv")
REPORT = os.path.join(ROOT, "work", "r81_report.txt")

N_FOLDS = 5
N_DRAWS = 5
K = R80.K


def job(spec):
    fold, kind = spec
    Xs, ys, imgs, Xt, yt, _ = R80.load()
    folds = R02.fold_split(np.unique(imgs), N_FOLDS, seed=0)
    hold = set(folds[fold].tolist())
    train = ~np.isin(imgs, list(hold))
    Xtr, ytr = Xs[train], ys[train]
    seed = zlib.crc32(("r81|%d|%s" % (fold, kind)).encode()) & 0xFFFFFFFF
    nz = R72.R61.DomainNormaliser("z_dataset").fit("src", Xtr)
    nz.fit("tgt", Xt)
    Zs = nz.transform("src", Xtr)
    Zt = nz.transform("tgt", Xt)
    clf = R72.R61.fit_classifier(Zs, ytr, kind=kind, seed=seed)
    score, err, lab = R72.score_err(clf, Zt, yt)
    acc = float((lab == yt).mean())

    def refit(mask):
        c2 = R72.R61.fit_classifier(
            np.vstack([Zs, Zt[mask]]),
            np.concatenate([ytr, yt[mask]]),
            kind=kind, seed=seed + 7)
        s2, e2, _ = R72.score_err(c2, Zt, yt)
        return s2, e2, None

    R72.KS = (K,)
    R72.N_SPLITS = N_DRAWS
    R72.ALPHAS = R80.ALPHAS
    rows = []
    R72.arms(rows, "G_immucan_breast_sourcefold",
             "depot_Breast->IMMUcan_BREAS", kind, fold, int(train.sum()),
             int(Xt.shape[0]), score, err, yt, seed + 13, refit,
             len(R80.SRC_MARKERS), len(R80.CLASSES))
    for r in rows:
        r["clf_acc"] = acc
        r["n_target"] = int(Xt.shape[0])
        r["source_fold"] = int(fold)
        r["n_train_src"] = int(train.sum())
    print("fold %d %-6s train %d acc %.3f" % (fold, kind, train.sum(), acc),
          flush=True)
    return rows


def summarise(df):
    rows = []
    for (clf, alpha, arm), g in df.groupby(["clf", "alpha", "arm"],
                                            sort=True):
        n_target = int(g["n_target"].iloc[0])
        n_eval = n_target - K
        net = (g["n_accepted"].to_numpy(dtype=float)
               - g["n_labels"].to_numpy(dtype=float)) / n_eval
        fold_means = []
        for fold, h in g.groupby("source_fold"):
            hnet = (h["n_accepted"].to_numpy(dtype=float)
                    - h["n_labels"].to_numpy(dtype=float)) / n_eval
            fold_means.append(float(hnet.mean()))
        rows.append({
            "clf": clf, "alpha": alpha, "arm": arm,
            "n_folds": int(g["source_fold"].nunique()),
            "n_draws": int(len(g)),
            "n_target": n_target, "n_eval": n_eval,
            "n_labels": int(g["n_labels"].iloc[0]),
            "label_frac": float(g["n_labels"].iloc[0] / n_target),
            "accept": float(g["accept_rate"].mean()),
            "risk": float(np.nanmean(g["selective_risk"])),
            "coverage": float(g["covered"].mean()),
            "nonvacuous": float((g["n_accepted"] > 0).mean()),
            "net": float(net.mean()),
            "net_fold_min": float(np.min(fold_means)),
            "net_fold_max": float(np.max(fold_means)),
            "fold_means": ",".join("%.4f" % v for v in fold_means),
            "positive_draws": int((net > 0).sum()),
            "all_folds_positive": int(np.all(np.asarray(fold_means) > 0)),
            "clf_acc": float(g["clf_acc"].mean()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args()
    if args.from_csv and os.path.exists(CSV):
        df = pd.read_csv(CSV)
    else:
        specs = [(f, c) for f in range(N_FOLDS) for c in R80.CLFS]
        rows = []
        with mp.Pool(min(args.jobs, len(specs))) as pool:
            for part in pool.imap(job, specs, chunksize=1):
                rows.extend(part)
        df = pd.DataFrame(rows)
        df.to_csv(CSV, index=False)
    sm = summarise(df)
    sm.to_csv(SUMMARY, index=False, float_format="%.10g")

    A = ["R81 -- source-fold robustness for IMMUcan BREAS repair",
         "=" * 78,
         "Target: 21,731 labelled cells; 1,000 labels (4.60%); "
         "five source-image folds x five target draws.",
         ""]
    z = sm[(np.isclose(sm.alpha, 0.10))
           & (sm.arm == "adapt_k%d" % K)]
    A.append("Repair arm at alpha 0.10:")
    for r in z.itertuples():
        A.append("  %-6s accept %.3f  risk %.3f  cover %.3f  net %+.4f  "
                 "fold means [%s]  positive %d/%d  all folds %s"
                 % (r.clf, r.accept, r.risk, r.coverage, r.net,
                    r.fold_means, r.positive_draws, r.n_draws,
                    bool(r.all_folds_positive)))
    A.append("")
    A.append("Frozen-model arm at alpha 0.10:")
    z = sm[(np.isclose(sm.alpha, 0.10))
           & (sm.arm == "tgt_k%d" % K)]
    for r in z.itertuples():
        A.append("  %-6s accept %.3f  risk %.3f  cover %.3f  net %+.4f  "
                 "fold means [%s]  positive %d/%d  all folds %s"
                 % (r.clf, r.accept, r.risk, r.coverage, r.net,
                    r.fold_means, r.positive_draws, r.n_draws,
                    bool(r.all_folds_positive)))
    txt = "\n".join(A) + "\n"
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
