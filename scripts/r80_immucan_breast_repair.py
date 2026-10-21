"""R80 -- a large target-cohort repair test on independent IMMUcan breast cells.

The existing breast panel-change target has 23,081 cells, so 1,000 labels are
4.3% of the evaluation batch and both tree ensembles repay while logistic
regression is vacuous.  This script asks whether that positive result survives
on a second, independent breast cohort whose target is large enough to make the
same label spend a small fraction of the batch.

Target: IMMUcan panel 1 V3, BREAS indication, six labels with a one-to-one
source class (Tumor, CD4, CD8, B, plasma, MacCD163).  Source: the depot breast
cohort Breast_Jun0524 on the twelve shared channels.  The target analysis set
has 21,731 labelled cells (4.60% for 1,000 labels).  The main split is the same
within-target exchangeable random split used by r72: draw 1,000 target labels,
spend 500 on a refit and 500 on the certificate, and evaluate on the remaining
20,731 cells.

Outputs
  experiments/r80_immucan_breast_repair.csv
  experiments/r80_immucan_breast_repair_summary.csv
  work/r80_report.txt
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

SOURCE = "Breast_Jun0524"
TARGET = os.path.join(ROOT, "data", "external3",
                      "immucan_breas_repair.csv")
CSV = os.path.join(ROOT, "experiments",
                   "r80_immucan_breast_repair.csv")
SUMMARY = os.path.join(ROOT, "experiments",
                       "r80_immucan_breast_repair_summary.csv")
REPORT = os.path.join(ROOT, "work", "r80_report.txt")

CLASS_MAP = {
    "TumorEpithelial": "Tumor",
    "CD4T": "CD4",
    "CD8T": "CD8",
    "Bcell": "B",
    "Plasma": "plasma",
    "Macrophage": "MacCD163",
}
CLASSES = ("Tumor", "CD4", "CD8", "B", "plasma", "MacCD163")
SRC_MARKERS = ("CD3", "CD4", "CD8", "CD20", "CD38", "CD14", "CD15",
               "CD68", "ECAD", "HLADRDPDQ", "Ki67", "SMA")
TGT_MARKERS = ("CD3", "CD4", "CD8a", "CD20", "CD38", "CD14", "CD15",
               "CD68", "Ecad", "HLADR", "Ki67", "SMA")

K = 1000
N_DRAWS = 20
ALPHAS = (0.05, 0.10, 0.20)
CLFS = ("logreg", "rf", "hgb")
PRIMARY_ALPHA = 0.10


def log(*parts):
    print("[%s] %s" % (pd.Timestamp.now().strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def load():
    g, markers, X = R02.prepare_cohort(SOURCE)
    src_lab = g["coarse_expert"].astype(str).map(CLASS_MAP)
    src_ok = src_lab.isin(CLASSES).to_numpy()
    src_cols = []
    for m in SRC_MARKERS:
        if m not in markers:
            raise SystemExit("source marker missing: %s" % m)
        src_cols.append(markers.index(m))
    Xs = X[np.ix_(src_ok, np.asarray(src_cols))]
    ys = np.asarray([CLASSES.index(v) for v in src_lab[src_ok]], dtype=int)
    imgs = g["image"].astype(str).to_numpy()[src_ok]

    t = pd.read_csv(TARGET)
    if not set(TGT_MARKERS).issubset(t.columns):
        raise SystemExit("target markers missing: %s"
                         % sorted(set(TGT_MARKERS) - set(t.columns)))
    tgt_ok = t["label"].isin(CLASSES).to_numpy()
    Xt = t.loc[tgt_ok, list(TGT_MARKERS)].to_numpy(dtype=np.float64)
    yt = np.asarray([CLASSES.index(v) for v in t.loc[tgt_ok, "label"]],
                    dtype=int)
    samples = t.loc[tgt_ok, "sample_id"].astype(str).to_numpy()
    return Xs, ys, imgs, Xt, yt, samples


def job(kind):
    Xs, ys, imgs, Xt, yt, samples = load()
    seed = zlib.crc32(("r80|%s" % kind).encode()) & 0xFFFFFFFF
    nz = R72.R61.DomainNormaliser("z_dataset").fit("src", Xs)
    nz.fit("tgt", Xt)
    Zs = nz.transform("src", Xs)
    Zt = nz.transform("tgt", Xt)
    clf = R72.R61.fit_classifier(Zs, ys, kind=kind, seed=seed)
    score, err, lab = R72.score_err(clf, Zt, yt)
    acc = float((lab == yt).mean())

    def refit(mask):
        c2 = R72.R61.fit_classifier(
            np.vstack([Zs, Zt[mask]]),
            np.concatenate([ys, yt[mask]]),
            kind=kind, seed=seed + 7)
        s2, e2, _ = R72.score_err(c2, Zt, yt)
        return s2, e2, None

    R72.KS = (K,)
    R72.N_SPLITS = N_DRAWS
    R72.ALPHAS = ALPHAS
    rows = []
    R72.arms(rows, "F_immucan_breast", "depot_Breast->IMMUcan_BREAS",
             kind, 0, int(Xs.shape[0]), int(Xt.shape[0]), score, err, yt,
             seed + 13, refit, len(SRC_MARKERS), len(CLASSES))
    for r in rows:
        r["clf_acc"] = acc
        r["n_target"] = int(Xt.shape[0])
        r["n_target_samples"] = int(len(set(samples.tolist())))
        r["source"] = SOURCE
        r["target"] = "IMMUcan_BREAS"
        r["shared_markers"] = ",".join(SRC_MARKERS)
    log("%-6s source %d target %d samples %d acc %.3f"
        % (kind, Xs.shape[0], Xt.shape[0], len(set(samples.tolist())), acc))
    return rows


def boot_ci(values, n_boot=10000, seed=20261019):
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarise(df):
    rows = []
    for (clf, alpha, arm), g in df.groupby(["clf", "alpha", "arm"],
                                            sort=True):
        g = g.sort_values("rep")
        n_target = int(g["n_target"].iloc[0])
        n_eval = n_target - K
        net = (g["n_accepted"].to_numpy(dtype=float)
               - g["n_labels"].to_numpy(dtype=float)) / n_eval
        lo, hi = boot_ci(net)
        risk = g["selective_risk"].to_numpy(dtype=float)
        rows.append({
            "clf": clf, "alpha": alpha, "arm": arm,
            "n_draws": int(len(g)),
            "n_target": n_target,
            "n_eval": n_eval,
            "n_labels": int(g["n_labels"].iloc[0]),
            "label_frac": float(g["n_labels"].iloc[0] / n_target),
            "accept": float(g["accept_rate"].mean()),
            "risk": float(np.nanmean(risk)),
            "coverage": float(g["covered"].mean()),
            "nonvacuous": float((g["n_accepted"] > 0).mean()),
            "cp_valid": float(g["cp_valid"].mean()),
            "oracle": float(g["oracle_accept"].mean()),
            "net": float(net.mean()),
            "net_lo": lo, "net_hi": hi,
            "positive_draws": int((net > 0).sum()),
            "clf_acc": float(g["clf_acc"].iloc[0]),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args()
    if args.from_csv and os.path.exists(CSV):
        df = pd.read_csv(CSV)
    else:
        with mp.Pool(min(args.jobs, len(CLFS))) as pool:
            rows = []
            for part in pool.imap(job, CLFS, chunksize=1):
                rows.extend(part)
        df = pd.DataFrame(rows)
        df.to_csv(CSV, index=False)
    sm = summarise(df)
    sm.to_csv(SUMMARY, index=False, float_format="%.10g")

    A = ["R80 -- large-target repair on independent IMMUcan breast cells",
         "=" * 78,
         "Source: %s; target: IMMUcan BREAS, %d labelled cells, %d samples."
         % (SOURCE, int(df["n_target"].iloc[0]),
            int(df["n_target_samples"].iloc[0])),
         "Shared markers: %s." % ", ".join(SRC_MARKERS),
         "Classes: %s." % ", ".join(CLASSES),
         "Primary: 1,000 labels (%.2f%% of target), 500 refit + 500 calibrate, "
         "20 draws, alpha 0.10."
         % (100.0 * K / int(df["n_target"].iloc[0])),
         ""]
    z = sm[(np.isclose(sm.alpha, PRIMARY_ALPHA))
           & (sm.arm == "adapt_k%d" % K)]
    A.append("Primary repair arm at alpha 0.10:")
    for r in z.itertuples():
        A.append("  %-6s acc %.3f  risk %.3f  cover %.3f  nonvac %.3f  "
                 "net %+.4f [%+.4f, %+.4f]  positive %d/%d  clf acc %.3f"
                 % (r.clf, r.accept, r.risk, r.coverage, r.nonvacuous,
                    r.net, r.net_lo, r.net_hi, r.positive_draws, r.n_draws,
                    r.clf_acc))
    A.append("")
    A.append("Frozen-model comparison at alpha 0.10:")
    z = sm[(np.isclose(sm.alpha, PRIMARY_ALPHA))
           & (sm.arm == "tgt_k%d" % K)]
    for r in z.itertuples():
        A.append("  %-6s acc %.3f  risk %.3f  cover %.3f  nonvac %.3f  "
                 "net %+.4f [%+.4f, %+.4f]  positive %d/%d"
                 % (r.clf, r.accept, r.risk, r.coverage, r.nonvacuous,
                    r.net, r.net_lo, r.net_hi, r.positive_draws, r.n_draws))
    A.append("")
    A.append("All budgets and arms are in %s." % SUMMARY)
    txt = "\n".join(A) + "\n"
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
