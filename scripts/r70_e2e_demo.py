"""R70 -- an end-to-end demonstration on a public cohort, priced in hours.

Why this script exists
----------------------
The paper's review-budget numbers are pooled over the depot's four cohorts and
are quoted as fractions of a batch.  A laboratory planning a run needs the same
claim in the units it buys: a new block arrives, a few hundred of its cells are
annotated to calibrate the certificate, and the rest of the block is either
committed or queued for a human.  This script runs that loop end to end on a
public cohort that the paper uses only as external evidence, and it converts the
review fraction into hours under one explicit and stated per-cell assumption.

Cohort
------
Zenodo 7570242, human liver allograft biopsies by imaging mass cytometry
(Children's Hospital Los Angeles / University of Southern California,
CC-BY-4.0).  109,245 cells, 24 donors, 32 blocks, 9 antibody channels and a
morphology panel.  The cohort is grouped by donor, so a deployment unit is a
disjoint group of whole donors: no cell of a deployment donor enters training.

Arms
----
  review_all     every cell of the block goes to a human.  The reference the
                 hours are counted against.
  strong         the same classifier the layer wraps, with the threshold a
                 laboratory would pick on its own: the largest score cut whose
                 calibration risk meets the budget, fitted on the same k target
                 labels.  It carries no guarantee.
  layer_kK       the certificate of the paper, fitted on the same k labels.
  oracle         the largest cut that meets the budget when the deployment
                 labels themselves are used to pick it.  Not attainable, and
                 reported only as the ceiling.

Outputs
-------
  experiments/r70_e2e_demo.csv           one row per unit, arm, k, budget
  experiments/r70_e2e_demo_summary.csv   pooled and per-budget
  work/r70_report.txt                    the report the manuscript quotes
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from errctl import conformal as C, paths

DATA = os.path.join(ROOT, "data", "external2", "LiverTx_IMC_processed_data.csv")

MARKERS = ["Cell_CD20_Dy161", "Cell_CD3_Er170", "Cell_CD45RA_Gd155",
           "Cell_CD45_Sm152", "Cell_CD66a_Yb171", "Cell_CD68_Tb159",
           "Cell_CD8_Dy162", "Cell_CollagenI_Tm169", "Cell_HLADR_Yb174"]
SHAPE = ["Area", "Eccentricity", "Solidity", "Extent", "Perimeter",
         "MajorAxisLength", "MinorAxisLength", "Percent_Touching",
         "Number_Neighbors"]

ALPHAS = (0.05, 0.10, 0.20)
KS = (0, 100, 250, 500, 1000, 2000)
DELTA = 0.10
N_FOLDS = 4
N_THREADS = 16
SEC_PER_CELL = 20.0          # one minute per three cells, the paper's own anchor
SEC_SENSITIVITY = (5.0, 60.0)

CSV = os.path.join(paths.RESULTS, "r70_e2e_demo.csv")
UNITS = os.path.join(paths.RESULTS, "r70_e2e_units.csv")
REPORT = os.path.join(paths.WORK, "r70_report.txt")


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def cp_upper(flags, delta=DELTA):
    e = np.asarray(flags, dtype=float)
    n = e.size
    if n == 0:
        return float("nan")
    k = int(e.sum())
    if k >= n:
        return 1.0
    return float(_beta.ppf(1.0 - delta, k + 1, n - k))


def load_cohort():
    cols = ["Sample", "Block_ID_long", "ImageID", "Cell_Label"] + MARKERS + SHAPE
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    df = df.dropna(subset=MARKERS + ["Cell_Label"])
    df = df[df["Cell_Label"].astype(str) != "Not Immune Cell"].reset_index(drop=True)
    return df


def build_models(seed, kinds):
    out = []
    if "xgb" in kinds:
        from xgboost import XGBClassifier
        out.append(("xgb", XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=N_THREADS,
            random_state=seed, verbosity=0)))
    if "cat" in kinds:
        from catboost import CatBoostClassifier
        out.append(("cat", CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.1, thread_count=N_THREADS,
            random_seed=seed, verbose=0, loss_function="MultiClass",
            allow_writing_files=False)))
    if "hgb" in kinds:
        from sklearn.ensemble import HistGradientBoostingClassifier
        out.append(("hgb", HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)))
    return out


def vote_proba(models, X):
    acc = None
    for _, m in models:
        p = np.asarray(m.predict_proba(X), dtype=np.float64)
        acc = p if acc is None else acc + p
    return acc / max(len(models), 1)


def naive_lambda(s_cal, e_cal, alpha, min_accept=25):
    """The cut a laboratory would pick with no guarantee: the largest threshold
    whose calibration risk meets the budget."""
    if s_cal.size == 0:
        return -np.inf
    order = np.argsort(s_cal, kind="mergesort")
    s = s_cal[order]
    cum = np.cumsum(e_cal[order])
    k = np.arange(1, s.size + 1)
    ok = (cum / k <= alpha) & (k >= min_accept)
    if not ok.any():
        return -np.inf
    return float(s[np.nonzero(ok)[0][-1]])


def oracle_lambda(s_te, e_te, alpha, min_accept=25):
    order = np.argsort(s_te, kind="mergesort")
    s = s_te[order]
    cum = np.cumsum(e_te[order])
    k = np.arange(1, s.size + 1)
    ok = (cum / k <= alpha) & (k >= min_accept)
    if not ok.any():
        return -np.inf
    return float(s[np.nonzero(ok)[0][-1]])


def emit(rows, unit, arm, k, alpha, accept, err, n_train, n_cal, n_donor):
    a = np.asarray(accept, dtype=bool)
    n_eval = int(a.size)
    n_acc = int(a.sum())
    risk = float(err[a].mean()) if n_acc else float("nan")
    up = cp_upper(err[a]) if n_acc else float("nan")
    rows.append(dict(
        unit=unit, arm=arm, k=k, alpha=alpha, n_train=n_train, n_cal=n_cal,
        n_donor=n_donor, n_eval=n_eval, n_accepted=n_acc,
        accept_rate=n_acc / max(n_eval, 1), realised_risk=risk, cp_upper=up,
        covered=int((not np.isfinite(risk)) or risk <= alpha),
        empty=int(n_acc == 0),
        valid=int(np.isfinite(up) and up <= alpha)))


def unit_job(job):
    unit, tr_idx, pool_idx, X, y, kinds, n_donor = job
    models = build_models(1234, kinds)
    for _, m in models:
        m.fit(X[tr_idx], y[tr_idx])
    Xp = X[pool_idx]
    truth = y[pool_idx]
    classes = np.unique(y)
    proba = vote_proba(models, Xp)
    lab = classes[np.argmax(proba, axis=1)]
    score = 1.0 - proba.max(axis=1)
    err = (lab != truth).astype(float)
    acc_model = float((lab == truth).mean())
    n = score.size
    rows = []
    emit(rows, unit, "review_all", -1, float("nan"), np.ones(n, dtype=bool),
         err, int(tr_idx.size), 0, n_donor)
    for alpha in ALPHAS:
        lam_o = oracle_lambda(score, err, alpha)
        emit(rows, unit, "oracle", -1, alpha, score <= lam_o, err,
             int(tr_idx.size), -1, n_donor)
        for k in KS:
            if k >= n:
                continue
            seed = zlib.crc32(("r70|%s|%d" % (unit, k)).encode()) & 0xFFFFFFFF
            rep = np.random.default_rng(seed).permutation(n)
            cal, dep = rep[:k], rep[k:]
            lam_n = naive_lambda(score[cal], err[cal], alpha)
            emit(rows, unit, "strong", k, alpha, score[dep] <= lam_n, err[dep],
                 int(tr_idx.size), k, n_donor)
            ctl = C.SelectiveRiskController(alpha, DELTA)
            ctl.fit(score[cal], err[cal])
            emit(rows, unit, "layer", k, alpha, ctl.accept(score[dep]), err[dep],
                 int(tr_idx.size), k, n_donor)
    return rows, dict(unit=unit, model_accuracy=acc_model, n_pool=int(n),
                      n_train=int(tr_idx.size), n_donors=int(n_donor))


def hours(rows, sec=SEC_PER_CELL):
    """Net expert seconds saved against reviewing every scored cell.

    ``n_eval`` excludes the cells already spent on calibration, so the full
    manual baseline is ``n_eval + n_cal`` and the work left is
    ``(n_eval - n_accepted) + n_cal``.  Their difference is ``n_accepted``.
    """
    return rows["n_accepted"] * sec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--models", default="xgb,cat,hgb")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    kinds = tuple(args.models.split(","))

    if not args.report_only:
        t0 = time.time()
        log("loading", DATA)
        df = load_cohort()
        log("immune-labelled cells:", len(df))
        donors = df["Sample"].astype(str).to_numpy()
        X = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in MARKERS + SHAPE])
        lab = df["Cell_Label"].astype(str).to_numpy()
        keep = np.isfinite(X).all(axis=1)
        X, lab, donors = X[keep], lab[keep], donors[keep]
        classes = np.unique(lab)
        ix = {c: i for i, c in enumerate(classes)}
        y = np.array([ix[v] for v in lab], dtype=int)
        log("feature matrix", X.shape, "classes", len(classes))

        counts = pd.Series(donors).value_counts()
        fold_of = {d: i % args.folds for i, d in enumerate(counts.index)}
        fd = np.array([fold_of[d] for d in donors])
        jobs = []
        for f in range(args.folds):
            pool = np.nonzero(fd == f)[0]
            tr = np.nonzero(fd != f)[0]
            if pool.size < 500:
                continue
            jobs.append(("fold%d" % f, tr, pool, X, y, kinds,
                         int(len(set(donors[pool])))))
        if args.limit:
            jobs = jobs[:args.limit]
        log("units", len(jobs), "train cells", jobs[0][1].size,
            "pool cells", [int(j[2].size) for j in jobs])
        rows, metas = [], []
        with mp.Pool(args.jobs) as pool:
            for r, meta in pool.imap(unit_job, jobs):
                rows.extend(r)
                metas.append(meta)
                log("done", meta["unit"], "pool", meta["n_pool"],
                    "donors", meta["n_donors"],
                    "model acc %.3f" % meta["model_accuracy"])
        pd.DataFrame(rows).to_csv(CSV, index=False)
        pd.DataFrame(metas).to_csv(UNITS, index=False)
        log("wrote", CSV, "in %.1f s" % (time.time() - t0))

    write_report()
    return 0


def write_report():
    df = pd.read_csv(CSV)
    meta = pd.read_csv(UNITS)
    keep = df["alpha"].notna()
    A = []
    A.append("r70 -- end-to-end demonstration on a public liver IMC cohort")
    A.append("=" * 78)
    A.append("")
    A.append("Cohort: Zenodo 7570242, human liver allograft biopsies, imaging")
    A.append("mass cytometry (CHLA / USC, CC-BY-4.0).  Deployment units are")
    A.append("disjoint groups of whole donors: %d units over %d donors, no cell"
             % (len(meta), 24))
    A.append("of a deployment donor in training.  Pool cells per unit: %d to %d,"
             % (meta.n_pool.min(), meta.n_pool.max()))
    A.append("median %d.  The wrapped model is an XGBoost, CatBoost and"
             % int(meta.n_pool.median()))
    A.append("histogram gradient boosting vote over the 9 antibody channels and")
    A.append("the morphology panel; mean accuracy %.3f."
             % meta.model_accuracy.mean())
    A.append("")
    A.append("Per-cell review time is an assumption, not a measurement: %.0f s,"
             % SEC_PER_CELL)
    A.append("one minute per three cells.  Sensitivity at %.0f s and %.0f s is"
             % SEC_SENSITIVITY)
    A.append("given below.")
    A.append("")
    d = df[keep & df["arm"].isin(["layer", "strong", "oracle"])]
    for alpha in ALPHAS:
        A.append("-- error budget alpha = %.2f" % alpha)
        A.append("  %-8s %-6s %8s %8s %8s %8s %10s"
                 % ("arm", "k", "accept", "risk", "cover", "valid", "h/1e4cells"))
        for arm in ("layer", "strong", "oracle"):
            sel = d[(d.arm == arm) & (d.alpha == alpha)]
            for k in sorted(sel.k.unique()):
                r = sel[sel.k == k]
                hv = hours(r)
                A.append("  %-8s %-6d %8.3f %8.3f %8.3f %8.3f %10.1f"
                         % (arm, k, r.accept_rate.mean(),
                            (r.realised_risk.fillna(0) * r.n_accepted).sum()
                            / max(r.n_accepted.sum(), 1),
                            r.covered.mean(), r.valid.mean(),
                            hv.sum() / max(r.n_eval.sum(), 1) * 10000 / 3600.0))
        A.append("")
    A.append("Hours saved against a full manual pass, k = 500 target labels,")
    A.append("pooled over the deployment units:")
    A.append("  %-6s %8s %9s %10s %12s %10s"
             % ("alpha", "accept", "risk", "reviewed", "hours saved", "of total"))
    sel500 = df[(df.arm == "layer") & (df.k == 500)]
    for alpha in ALPHAS:
        r = sel500[sel500.alpha == alpha]
        hv = hours(r)
        tot = (r.n_eval.sum() + r.n_cal.clip(lower=0).sum()) * SEC_PER_CELL / 3600.0
        A.append("  %-6.2f %8.3f %9.3f %10d %12.1f %10.1f"
                 % (alpha, r.accept_rate.mean(),
                    (r.realised_risk.fillna(0) * r.n_accepted).sum()
                    / max(r.n_accepted.sum(), 1),
                    int((r.n_eval - r.n_accepted).sum()), hv.sum() / 3600.0, tot))
    A.append("")
    A.append("Headline, alpha = 0.20, k = 500 target labels:")
    sel = df[(df.arm == "layer") & (df.alpha == 0.20) & (df.k == 500)]
    hv = hours(sel)
    full = (sel.n_eval.sum() + sel.n_cal.clip(lower=0).sum()) * SEC_PER_CELL
    A.append("  cells in the deployment units   %d"
             % (sel.n_eval.sum() + sel.n_cal.clip(lower=0).sum()))
    A.append("  accepted by the layer           %.3f" % sel.accept_rate.mean())
    A.append("  routed to a human               %d cells"
             % (sel.n_eval - sel.n_accepted).sum())
    A.append("  labels spent on calibration     %d" % sel.n_cal.sum())
    A.append("  expert hours saved              %.1f of %.1f (%.1f%%)"
             % (hv.sum() / 3600.0, full / 3600.0, 100.0 * hv.sum() / full))
    for s in SEC_SENSITIVITY:
        A.append("    at %.0f s per cell             %.1f h"
                 % (s, hours(sel, s).sum() / 3600.0))
    A.append("  realised risk among retained    %.3f"
             % ((sel.realised_risk.fillna(0) * sel.n_accepted).sum()
                / max(sel.n_accepted.sum(), 1)))
    A.append("  coverage of the budget          %.3f" % sel.covered.mean())
    st = df[(df.arm == "strong") & (df.alpha == 0.20) & (df.k == 500)]
    A.append("")
    A.append("The strong baseline on the same labels accepts %.3f against the"
             % st.accept_rate.mean())
    A.append("layer's %.3f, but holds the budget in %.3f of its evaluations"
             % (sel.accept_rate.mean(), st.covered.mean()))
    A.append("against the layer's %.3f, so it removes more review because it is"
             % sel.covered.mean())
    A.append("not guaranteed rather than because it is better.")
    txt = "\n".join(A) + "\n"
    open(REPORT, "w", encoding="utf-8").write(txt)
    print(txt)
    return txt


if __name__ == "__main__":
    sys.exit(main())
