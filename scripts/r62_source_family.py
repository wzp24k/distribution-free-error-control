"""r62 -- the fourth partition family: the annotation source.

Why this exists
---------------
The paper asks its validity question on four partition families: across tissue
(r24/r31), across fields of view (r48), across panels (r61) and across
annotation sources.  The first three are measured.  The fourth was not, because
a cross-source family needs two labels for the *same* cells.

The depot ships exactly that for its two cohorts that carry mean marker
intensities for every segmented cell: ``ManualLabels`` (the expert, source A) and
``HighQualityLabels`` (the depot's own semi-supervised model, source B), both on
the same ``(image, cellID)``.  So one cell has two ground truths and the question
"does the certificate survive a change of ground truth?" is measurable with no
new annotation.

Design
------
The model is trained once per fold on A and then held fixed.  All arms sit on the
same cells, the same folds and the same random calibration/deployment split of
each held-out fold.  Only the label source used to score the calibration half
changes:

  same_expert  calibrate on A      (the exchangeable control)
  same_hq      calibrate on B      (the exchangeable control)
  consmax      one controller fit on err = max(err_A, err_B)
  consint      two controllers fit at delta/2 each, accept iff
               score <= min(lambda_A, lambda_B)

There is no separate cross-source arm, and that is a fact about the design rather
than a simplification: the accepted set does not depend on which source the
deployment will be scored against.  Every arm is therefore read against *both*
sources, and a cross-source result is a second reading of a same-source arm --
the same certificate scoring 1.000 against the labels it was built on and
something else against the other source.

consmax and consint are the repairs, and neither needs to know which source the
deployment will use.  err_A and err_B are pointwise bounded by
max(err_A, err_B), so a certificate for the maximum covers either reading; and
the smaller of two certified thresholds inherits both certificates by a union
bound.  Both pay in acceptance, not in risk.

The failure mechanism is an output too.  err_A and err_B can differ only on the
cells where the two sources disagree, so

    realised risk against B - realised risk against A  <=  disagreement rate

holds cell by cell, hence also in the mean.  Every row carries the disagreement
rate among its accepted cells, so the gap is bounded above by a number reported
beside it, and the report checks the inequality arm by arm.

Outputs: experiments/r62_source_family.csv,
         experiments/r62_source_family_summary.csv, work/r62_report.txt
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from errctl import conformal as C, paths
from errctl.classifiers import fit_classifier
from errctl.preprocess import DomainNormaliser
import r02_run_experiments as R2

DATASETS = ("CRC_Feb2623", "GVHD_Feb1323")
CLFS = ("logreg", "rf", "hgb")
ALPHAS = (0.05, 0.10, 0.20)
ARMS = ("same_expert", "same_hq", "consmax", "consint")
DELTA = 0.10
N_SPLITS = 20
SELECT = "bonferroni"
GRID = 120
BALANCE = True
MIN_CLASS_TEST = 20
MAX_TRAIN = R2.MAX_TRAIN
N_FOLDS = R2.N_FOLDS

CSV = os.path.join(paths.RESULTS, "r62_source_family.csv")
SUMM = os.path.join(paths.RESULTS, "r62_source_family_summary.csv")
REPORT = os.path.join(paths.WORK, "r62_report.txt")
FT = os.path.join(paths.RESULTS, "r62_fold_transfer.csv")

COHORT = {}
VOCAB = {}


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def cp_upper(flags, delta=DELTA):
    """One-sided Clopper-Pearson upper limit of a risk from a 0/1 sample."""
    e = np.asarray(flags, dtype=float)
    n = e.size
    if n == 0:
        return float("nan")
    k = int(e.sum())
    if k >= n:
        return 1.0
    return float(_beta.ppf(1.0 - delta, k + 1, n - k))


def load(ds):
    g, markers, X = R2.prepare_cohort(ds)
    imgs = g["image"].astype(str).to_numpy()
    return dict(ds=ds, g=g, markers=markers, X=X, imgs=imgs,
                yA=g["coarse_expert"].astype(str).to_numpy(),
                yB=g["coarse_hq"].astype(str).to_numpy())


def build_vocab(g):
    vc = g["coarse_expert"].value_counts()
    return [c for c in sorted(vc.index) if vc[c] >= MIN_CLASS_TEST]


def fold_split_balanced(images, weights, n_folds, seed=0):
    """Greedy image folds whose *cell* counts are balanced, not their image counts.

    ``R2.fold_split`` shuffles image names and splits them into equal numbers of
    images, which is not equal numbers of cells: the colorectal cohort has eight
    expert-annotated images holding between 152 and 440 pool cells, so an equal
    image split leaves folds of 756 and 166 cells and the floor drops the small
    one.  Assigning the heaviest image to the lightest fold in turn keeps every
    fold usable, which matters because the certificate's feasibility is close to a
    fixed *count* of calibration cells rather than a fixed fraction.
    """
    img = np.asarray(images, dtype=object)
    w = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    order = sorted(range(len(img)), key=lambda i: (-w[i], rng.random()))
    folds = [[] for _ in range(n_folds)]
    load = np.zeros(n_folds)
    for i in order:
        j = int(np.argmin(load))
        folds[j].append(img[i])
        load[j] += w[i]
    return [np.array(sorted(f), dtype=object) for f in folds]


def _init(ds_list, n_splits=None, select=None, grid=None, balance=None):
    global N_SPLITS, SELECT, GRID, BALANCE
    if n_splits is not None:
        N_SPLITS = int(n_splits)
    if select is not None:
        SELECT = select
    if grid is not None:
        GRID = int(grid)
    if balance is not None:
        BALANCE = bool(balance)
    for ds in ds_list:
        d = load(ds)
        VOCAB[ds] = build_vocab(d["g"])
        inA = np.array([v in set(VOCAB[ds]) for v in d["yA"]])
        pool_imgs = d["imgs"][inA]
        if BALANCE:
            u, cnt = np.unique(pool_imgs, return_counts=True)
            d["folds"] = fold_split_balanced(u, cnt, N_FOLDS, seed=0)
        else:
            d["folds"] = R2.fold_split(np.unique(pool_imgs), N_FOLDS, seed=0)
        d["inA"] = inA
        COHORT[ds] = d


def emit(rows, arm, ds, clf, alpha, rep, n_cal, o, mask, fold=None):
    m = np.asarray(mask, dtype=bool)
    k = int(m.sum())
    n = int(m.size)
    eA, eB = o["errA"][m], o["errB"][m]
    riskA = float(eA.mean()) if k else float("nan")
    riskB = float(eB.mean()) if k else float("nan")
    upA = cp_upper(eA) if k else float("nan")
    upB = cp_upper(eB) if k else float("nan")
    rows.append(dict(
        arm=arm, dataset=ds, clf=clf, alpha=alpha, rep=rep, fold=fold,
        n_cal=int(n_cal), n_pool=n, n_accepted=k,
        accept_rate=k / max(n, 1),
        risk_expert=riskA, risk_hq=riskB,
        covered_expert=int((not np.isfinite(riskA)) or riskA <= alpha),
        covered_hq=int((not np.isfinite(riskB)) or riskB <= alpha),
        cp_upper_expert=upA, cp_upper_hq=upB,
        valid_expert=int(np.isfinite(upA) and upA <= alpha),
        valid_hq=int(np.isfinite(upB) and upB <= alpha),
        disagree_rate=float((o["yA"][m] != o["yB"][m]).mean()) if k else float("nan"),
        gap=float(riskB - riskA) if k else float("nan"),
        hq_oov_rate=float(o["oov"][m].mean()) if k else float("nan"),
        empty=int(k == 0)))


def job(job):
    ds, fold, kind = job
    d = COHORT[ds]
    classes = VOCAB[ds]
    ix = {c: i for i, c in enumerate(classes)}
    yA, yB, imgs = d["yA"], d["yB"], d["imgs"]
    inA = d["inA"]
    in_fold = np.isin(imgs, list(d["folds"][fold].tolist()))
    pool = np.nonzero(in_fold & inA)[0]
    tr_idx = np.nonzero(~in_fold & inA)[0]
    rows = []
    if pool.size < 200 or tr_idx.size < 50:
        return rows, None

    seed = zlib.crc32(("r62|%s|%d|%s" % (ds, fold, kind)).encode()) & 0xFFFFFFFF
    if tr_idx.size > MAX_TRAIN:
        tr_idx = R2.subsample_training(
            tr_idx, np.array([ix[v] for v in yA[tr_idx]]), MAX_TRAIN, seed)
    ytr = np.array([ix[v] for v in yA[tr_idx]], dtype=int)
    Xt = d["X"][tr_idx]
    nz = DomainNormaliser("z_dataset").fit("src", Xt)
    clf = fit_classifier(nz.transform("src", Xt), ytr, kind=kind, seed=seed)
    probs = clf.predict_proba(nz.transform("src", d["X"][pool])).astype(np.float64)
    lab = np.asarray(classes)[probs.argmax(1)]
    score = 1.0 - probs.max(1)
    errA = (lab != yA[pool]).astype(float)
    errB = (lab != yB[pool]).astype(float)
    oov = ~np.isin(yB[pool], list(classes))
    foldinfo = dict(
        dataset=ds, fold=int(fold), clf=kind,
        images=";".join(sorted(str(x) for x in d["folds"][fold])),
        n_images=int(len(d["folds"][fold])),
        pool=int(pool.size), train=int(tr_idx.size),
        acc_expert=float(1.0 - errA.mean()),
        acc_hq=float(1.0 - errB.mean()))

    rng = np.random.default_rng(seed + 13)
    n = pool.size
    half = n // 2
    for rep in range(N_SPLITS):
        pick = rng.permutation(n)
        c, e = pick[:half], pick[half:]
        s_c, s_e = score[c], score[e]
        eA_c, eB_c = errA[c], errB[c]
        o = dict(errA=errA[e], errB=errB[e], yA=yA[pool][e], yB=yB[pool][e],
                 oov=oov[e])
        for alpha in ALPHAS:
            def mk(al, dl):
                return C.SelectiveRiskController(
                    al, dl, grid_size=GRID, select=SELECT)
            ctlA = mk(alpha, DELTA).fit(s_c, eA_c)
            ctlB = mk(alpha, DELTA).fit(s_c, eB_c)
            ctlM = mk(alpha, DELTA).fit(s_c, np.maximum(eA_c, eB_c))
            ctlA2 = mk(alpha, DELTA / 2.0).fit(s_c, eA_c)
            ctlB2 = mk(alpha, DELTA / 2.0).fit(s_c, eB_c)
            lam_int = min(ctlA2.lambda_hat, ctlB2.lambda_hat)
            arms = (
                ("same_expert", ctlA.accept(s_e)),
                ("same_hq", ctlB.accept(s_e)),
                ("consmax", ctlM.accept(s_e)),
                ("consint", s_e <= lam_int),
            )
            for arm, mask in arms:
                emit(rows, arm, ds, kind, alpha, rep, half, o, mask,
                     fold)
    return rows, foldinfo


def _one(df, ds, arm, col, alpha=0.10):
    s = df[(df.arm == arm) & (df.dataset == ds) & np.isclose(df.alpha, alpha)]
    return float(s[col].mean()) if len(s) else float("nan")


def _summary(df):
    return (df.groupby(["dataset", "arm", "alpha"])
              .agg(n_pool=("n_pool", "mean"),
                   nonvac_frac=("empty", lambda x: 1.0 - float(x.mean())),
                   mean_accept=("accept_rate", "mean"),
                   mean_risk_expert=("risk_expert", "mean"),
                   mean_risk_hq=("risk_hq", "mean"),
                   coverage_expert=("covered_expert", "mean"),
                   coverage_hq=("covered_hq", "mean"),
                   cp_valid_expert=("valid_expert", "mean"),
                   cp_valid_hq=("valid_hq", "mean"),
                   mean_disagree=("disagree_rate", "mean"),
                   mean_gap=("gap", "mean"),
                   mean_hq_oov=("hq_oov_rate", "mean"))
              .reset_index())


def write_report(df, summ, txt_path, fold_df=None):
    A = []
    A.append("r62 -- the fourth partition family: the annotation source")
    A.append("=" * 78)
    A.append("")
    A.append("Cohorts: %s (the two with a second label source)." % ", ".join(DATASETS))
    for ds in DATASETS:
        A.append("  %s: vocabulary %d classes" % (ds, len(VOCAB[ds])))
        A.append("    %s" % ", ".join(VOCAB[ds]))
    A.append("")
    A.append("Sources: expert = ManualLabels (A), hq = HighQualityLabels (B),")
    A.append("joined on (image, cellID) so both label the same cells.")
    A.append("Protocol: %d image folds over the expert-labelled images, %d random"
             % (N_FOLDS, N_SPLITS))
    A.append("calibration/deployment splits per fold, delta = %.2f, classifiers %s."
             % (DELTA, ",".join(CLFS)))
    A.append("Controller: select=%s, grid_size=%d, min_accept=%d."
             % (SELECT, GRID,
                C.SelectiveRiskController(0.1, 0.1).min_accept))
    A.append("")
    A.append("The model is trained once per fold on A and held fixed.  Only the")
    A.append("label source used to score the calibration half changes, so the")
    A.append("accepted set does not depend on which source the deployment will be")
    A.append("scored against, and every arm is read against both sources.")
    A.append("'accept' = share of the deployment half the arm annotates; 'risk_x' =")
    A.append("realised error among those cells against source x; 'cov_x' = share of")
    A.append("units whose realised risk against x was within the budget; 'disagg' =")
    A.append("share of the accepted cells the two sources label differently.")
    A.append("")
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        A.append("-" * 78)
        A.append("%s -- arm x alpha (mean over folds, splits and classifiers)" % ds)
        A.append("")
        A.append("  arm           alpha  accept  nonvac  risk_A  cov_A  risk_B  "
                 "cov_B  disagg  gap     hq-oov")
        for arm in ARMS:
            for alpha in ALPHAS:
                s = sub[(sub.arm == arm) & np.isclose(sub.alpha, alpha)]
                if not len(s):
                    continue
                A.append("  %-13s %.2f   %6.3f  %6.3f  %6.3f  %5.3f  %6.3f  %5.3f"
                         "  %6.3f  %+6.3f  %6.3f"
                         % (arm, alpha, s.accept_rate.mean(),
                            1 - s["empty"].mean(), s.risk_expert.mean(),
                            s.covered_expert.mean(), s.risk_hq.mean(),
                            s.covered_hq.mean(), s.disagree_rate.mean(),
                            s.gap.mean(), s.hq_oov_rate.mean()))
            A.append("")
    if fold_df is not None and len(fold_df):
        A.append("-" * 78)
        A.append("Held-out-fold transfer: the accuracy of the frozen model on")
        A.append("the deployment cells of each fold, against each source.  The")
        A.append("model is trained on the other folds only, so this is the")
        A.append("transfer the certificate has to build on, and it is what")
        A.append("limits the colorectal cohort at a 10% budget.")
        A.append("'certified' is the share of units with a non-empty accepted")
        A.append("set at a 10% budget, over the four arms, three classifiers")
        A.append("and twenty splits of that fold.")
        A.append("")
        for ds in DATASETS:
            sub = fold_df[fold_df.dataset == ds]
            if not len(sub):
                continue
            u = df[(df.dataset == ds) & np.isclose(df.alpha, 0.10)]
            A.append("  %s" % ds)
            for f in sorted(sub.fold.unique()):
                g = sub[sub.fold == f]
                r0 = g.iloc[0]
                q = u[u.fold == f]
                cert = 1.0 - float(q["empty"].mean()) if len(q) else float("nan")
                A.append("    fold %d  images %d  pool %4d  certified %.3f"
                         % (f, int(r0.n_images), int(r0["pool"]), cert))
                A.append("      %s" % r0.images)
                for _, r in g.sort_values("clf").iterrows():
                    A.append("        %-7s acc_A %.4f  acc_B %.4f"
                             % (r.clf, r.acc_expert, r.acc_hq))
            A.append("    transfer accuracy spans %.4f to %.4f"
                     % (float(sub.acc_expert.min()),
                        float(sub.acc_expert.max())))
            A.append("")
    A.append("-" * 78)
    A.append("The contrasts, at alpha = 0.10 (mean over folds, splits, classifiers)")
    A.append("")
    A.append(_contrasts(df))
    A.append("")
    txt = "\n".join(A) + "\n"
    with open(txt_path, "w", encoding="ascii", errors="replace") as fh:
        fh.write(txt)
    return txt


def _contrasts(df):
    out = []
    for ds in DATASETS:
        out.append("  %s" % ds)
        ce = _one(df, ds, "same_expert", "covered_expert")
        ch = _one(df, ds, "same_hq", "covered_hq")
        out.append("    controls, same source both sides: expert covers %.3f, hq"
                   " covers %.3f (nominal %.2f)" % (ce, ch, 1 - DELTA))
        for arm in ("same_expert", "same_hq", "consmax", "consint"):
            a = _one(df, ds, arm, "accept_rate")
            rA = _one(df, ds, arm, "risk_expert")
            rB = _one(df, ds, arm, "risk_hq")
            cA = _one(df, ds, arm, "covered_expert")
            cB = _one(df, ds, arm, "covered_hq")
            dis = _one(df, ds, arm, "disagree_rate")
            gap = _one(df, ds, arm, "gap")
            ok = "holds" if (not np.isfinite(gap)) or gap <= dis + 1e-9 else "VIOLATED"
            out.append("    %-13s accept %.3f | vs expert: risk %.3f cov %.3f |"
                       " vs hq: risk %.3f cov %.3f"
                       % (arm, a, rA, cA, rB, cB))
            out.append("                  gap %.4f <= disagg %.4f  [%s]"
                       % (gap, dis, ok))
        out.append("")
    return "\n".join(out)


def main():
    global CSV, SUMM, REPORT, FT, DATASETS
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--clfs", default=",".join(CLFS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--select", default=SELECT,
                    choices=("bonferroni", "fixed_sequence"))
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--no-balance", dest="balance", action="store_false",
                    default=True,
                    help="split by image count instead of balancing by pool cells")
    ap.add_argument("--tag", default="")
    ap.add_argument("--from-csv", action="store_true",
                    help="rebuild the report from the stored CSV instead of running")
    args = ap.parse_args()
    CSV = os.path.join(paths.RESULTS, "r62_source_family%s.csv" % args.tag)
    SUMM = os.path.join(paths.RESULTS, "r62_source_family%s_summary.csv" % args.tag)
    REPORT = os.path.join(paths.WORK, "r62%s_report.txt" % args.tag)
    FT = os.path.join(paths.RESULTS, "r62_fold_transfer%s.csv" % args.tag)
    clfs = [c for c in args.clfs.split(",") if c]
    DATASETS = tuple(x for x in args.datasets.split(",") if x)
    log("r62_source_family.py  tag=%r  jobs=%d  datasets=%s  splits=%d  select=%s"
        "  grid=%d  balance=%s"
        % (args.tag, args.jobs, ",".join(DATASETS), args.splits, args.select,
           args.grid, args.balance))
    _init(list(DATASETS), args.splits, args.select, args.grid, args.balance)
    for ds in DATASETS:
        d = COHORT[ds]
        n_pool = int(d["inA"].sum())
        ag = float((d["yA"] == d["yB"])[d["inA"]].mean())
        log("%s: %d cells, %d classes, pool %d, A-vs-B coarse agreement %.4f"
            % (ds, len(d["g"]), len(VOCAB[ds]), n_pool, ag))
        for f in range(N_FOLDS):
            m = np.isin(d["imgs"], list(d["folds"][f].tolist())) & d["inA"]
            log("   fold %d: %d images, pool %d%s"
                % (f, len(d["folds"][f]), int(m.sum()),
                   "  [SKIPPED: pool < 200]" if m.sum() < 200 else ""))
    if args.from_csv:
        df = pd.read_csv(CSV)
        fdf = pd.read_csv(FT) if os.path.exists(FT) else None
        txt = write_report(df, _summary(df), REPORT, fdf)
        log("wrote %s (%d bytes, from %s)" % (REPORT, len(txt), CSV))
        print(txt)
        return 0

    jobs = [(ds, f, k) for ds in DATASETS for f in range(N_FOLDS) for k in clfs]
    rows = []
    frows = []
    with mp.Pool(args.jobs, initializer=_init,
                 initargs=(list(DATASETS), args.splits, args.select, args.grid,
                           args.balance)) as pool:
        for i, res in enumerate(pool.imap(job, jobs, chunksize=1), 1):
            res_rows, res_fold = res
            rows.extend(res_rows)
            if res_fold:
                frows.append(res_fold)
            log("  job %d/%d -> %d rows" % (i, len(jobs), len(res_rows)))
    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    ft = pd.DataFrame(frows).sort_values(["dataset", "fold", "clf"])
    ft = ft.reset_index(drop=True)
    ft.to_csv(FT, index=False)
    log("wrote %s (%d rows)" % (FT, len(ft)))
    log("wrote %s (%d rows)" % (CSV, len(df)))
    summ = _summary(df)
    summ.to_csv(SUMM, index=False)
    log("wrote %s" % SUMM)
    txt = write_report(df, summ, REPORT, ft)
    log("wrote %s (%d bytes)" % (REPORT, len(txt)))
    print(txt)
    log("done (%.1fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()