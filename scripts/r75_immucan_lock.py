"""R75 -- a locked first evaluation on an external IMC cohort.

This is the pre-registered first evaluation of the certificate on the IMMUcan
panel 1 V3 manual labels.  The protocol is frozen in
``docs/submission/preregistration_immucan.md`` before the endpoint is run.

The cohort is read from ``data/external3/immucan_v3_cells.csv``, produced by
``scripts/r75_immucan_export.R`` from the public Zenodo record.  The primary
endpoint is the vacuity-inclusive coverage at a 10% selective-risk budget,
with 5 sample-grouped folds, 10 calibration draws, and three classifiers.
No threshold, class filter or split parameter is selected after seeing the
endpoint.

Outputs
  experiments/r75_immucan_evals.csv
  experiments/r75_immucan_summary.csv
  work/r75_immucan_report.txt
"""

import argparse
import os
import sys
import multiprocessing as mp

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import r61_shared_panel as R61  # noqa: E402

CSV_IN = os.path.join(ROOT, "data", "external3", "immucan_v3_cells.csv")
EVALS = os.path.join(ROOT, "experiments", "r75_immucan_evals.csv")
SUMMARY = os.path.join(ROOT, "experiments", "r75_immucan_summary.csv")
REPORT = os.path.join(ROOT, "work", "r75_immucan_report.txt")

ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
N_FOLDS = 5
N_SPLITS = 10
CLFS = ("logreg", "rf", "hgb")
MIN_TRAIN = 3
MIN_TARGET = 20
MAX_TRAIN = 60000
POOL_CAP = 20000
SEED = 20261018
PRIMARY_ALPHA = 0.10


def log(*parts):
    print("[%s] %s" % (pd.Timestamp.now().strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def load_cells():
    df = pd.read_csv(CSV_IN, low_memory=False)
    label_col = "label"
    image_col = "sample_id" if "sample_id" in df.columns else "image"
    markers = [c for c in df.columns if c not in ("image", "sample_id", label_col)]
    y = df[label_col].astype(str).to_numpy()
    X = df[markers].to_numpy(dtype=np.float64)
    images = df[image_col].astype(str).to_numpy()
    ok = np.isfinite(X).all(1) & (y != "unlabelled")
    X, y, images = X[ok], y[ok], images[ok]
    return X, y, images, markers


def job(spec):
    fold, clf = spec
    X, y, images, markers = load_cells()
    imgs = np.asarray(sorted(set(images.tolist())), dtype=object)
    rng0 = np.random.default_rng(SEED)
    rng0.shuffle(imgs)
    folds = np.array_split(imgs, N_FOLDS)
    held = np.asarray(sorted(folds[fold].tolist()), dtype=object)
    tr = np.nonzero(~np.isin(images, held))[0]
    pool = np.nonzero(np.isin(images, held))[0]
    if tr.size > MAX_TRAIN:
        tr = np.random.default_rng(SEED + fold).choice(tr, MAX_TRAIN, replace=False)
    counts_tr = pd.Series(y[tr]).value_counts()
    counts_po = pd.Series(y[pool]).value_counts()
    classes = sorted(c for c in set(counts_tr.index) & set(counts_po.index)
                     if counts_tr[c] >= MIN_TRAIN and counts_po[c] >= MIN_TARGET)
    if len(classes) < 2:
        return []
    ix = {c: i for i, c in enumerate(classes)}
    tr = np.asarray([i for i in tr if y[i] in ix], dtype=int)
    pool = np.asarray([i for i in pool if y[i] in ix], dtype=int)
    if pool.size > POOL_CAP:
        pool = np.sort(np.random.default_rng(SEED + 100 + fold).choice(
            pool, POOL_CAP, replace=False))
    ytr = np.asarray([ix[v] for v in y[tr]], dtype=int)
    ypo = np.asarray([ix[v] for v in y[pool]], dtype=int)
    nz = R61.DomainNormaliser("z_dataset").fit("src", X[tr])
    Ztr = nz.transform("src", X[tr])
    Zpo = nz.transform("src", X[pool])
    clf_obj = R61.fit_classifier(Ztr, ytr, kind=clf, seed=SEED + fold)
    p = clf_obj.predict_proba(Zpo).astype(np.float64)
    score = 1.0 - p.max(1)
    err = (p.argmax(1) != ypo).astype(float)
    acc = float((p.argmax(1) == ypo).mean())
    rows = []
    for rep in range(N_SPLITS):
        rng = np.random.default_rng(SEED + 10000 + fold * 100 + rep)
        pick = rng.permutation(pool.size)
        half = pool.size // 2
        cal, ev = pick[:half], pick[half:]
        for alpha in ALPHAS:
            ctl = R61.C.SelectiveRiskController(alpha, DELTA).fit(
                score[cal], err[cal])
            acc_mask = ctl.accept(score[ev])
            n_acc = int(acc_mask.sum())
            risk = float(err[ev][acc_mask].mean()) if n_acc else np.nan
            rows.append({
                "fold": fold, "clf": clf, "rep": rep, "alpha": alpha,
                "n_train": int(tr.size), "n_pool": int(pool.size),
                "n_cal": int(cal.size), "n_eval": int(ev.size),
                "n_accepted": n_acc,
                "accept_rate": float(acc_mask.mean()),
                "selective_risk": risk,
                "covered": int(n_acc == 0 or risk <= alpha),
                "empty": int(n_acc == 0),
                "cp_valid": int(n_acc == 0 or R61.cp_upper(err[ev][acc_mask]) <= alpha),
                "clf_acc": acc,
                "n_classes": len(classes),
            })
    log("fold %d %-6s train %5d pool %5d classes %2d acc %.3f"
        % (fold, clf, tr.size, pool.size, len(classes), acc))
    return rows


def summarise(df):
    rows = []
    for alpha, g in df.groupby("alpha", sort=True):
        n = len(g)
        k = int(g.covered.sum())
        lo, hi = wilson(k, n)
        nonvac = g[g["empty"] == 0]
        rows.append({
            "alpha": alpha, "n_eval": n, "n_units": int(g.fold.nunique()),
            "coverage": k / n, "coverage_lo": lo, "coverage_hi": hi,
            "frac_nonvacuous": float((g["empty"] == 0).mean()),
            "mean_accept": float(g.accept_rate.mean()),
            "mean_accept_nonvac": float(nonvac.accept_rate.mean()) if len(nonvac) else np.nan,
            "risk_nonvac": float(nonvac.selective_risk.mean()) if len(nonvac) else np.nan,
            "n_over_budget": int((g.selective_risk > g.alpha).sum()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-evals", action="store_true",
                    help="rebuild only the summary from saved evaluations")
    args = ap.parse_args()
    if not os.path.exists(CSV_IN):
        raise SystemExit("missing %s; run scripts/r75_immucan_export.R" % CSV_IN)
    if args.from_evals and os.path.exists(EVALS):
        df = pd.read_csv(EVALS)
    else:
        specs = [(f, c) for f in range(N_FOLDS) for c in CLFS]
        rows = []
        with mp.Pool(min(6, len(specs))) as pool:
            for res in pool.imap(job, specs, chunksize=1):
                rows.extend(res)
        df = pd.DataFrame(rows)
        df.to_csv(EVALS, index=False)
    sm = summarise(df)
    sm.to_csv(SUMMARY, index=False)
    raw = pd.read_csv(CSV_IN, low_memory=False)
    label_col = "label"
    image_col = "sample_id" if "sample_id" in raw.columns else "image"
    markers = [c for c in raw.columns
               if c not in ("image", "sample_id", label_col)]
    n_samples = int(raw[image_col].astype(str).nunique())
    n_labels = int((raw[label_col].astype(str) != "unlabelled").sum())
    A = ["R75 -- locked first evaluation on IMMUcan panel 1 V3", "=" * 72]
    A.append("Cohort: %d samples, %s manual labels, %d markers."
             % (n_samples, format(n_labels, ","), len(markers)))
    A.append("Primary endpoint: coverage at alpha = 0.10, 5 sample folds x 10 draws x 3 classifiers.")
    A.append("")
    for _, r in sm.iterrows():
        A.append("alpha %.2f  n=%d  coverage %.3f [%.3f, %.3f]  "
                 "nonvac %.3f  accept %.3f  risk %.3f  over %d"
                 % (r.alpha, r.n_eval, r.coverage, r.coverage_lo, r.coverage_hi,
                    r.frac_nonvacuous, r.mean_accept, r.risk_nonvac, r.n_over_budget))
    A.append("")
    r = sm[np.isclose(sm.alpha, PRIMARY_ALPHA)].iloc[0]
    ok = (r.coverage >= 0.90 and r.coverage_lo >= 0.85
          and r.frac_nonvacuous >= 0.80 and r.mean_accept >= 0.50)
    A.append("PREREGISTERED PRIMARY %s" % ("PASS" if ok else "FAIL"))
    open(REPORT, "w", encoding="utf-8").write("\n".join(A) + "\n")
    print("\n".join(A))


if __name__ == "__main__":
    main()
