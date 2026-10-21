"""r61 -- a real cross-panel deployment on the channels two panels share.

Why this exists
---------------
``scripts/r08_panel.py`` asked the panel question by *simulating* a smaller
panel inside a single cohort: markers are dropped in order of a training-set
F-score, so the surviving channels are chosen by the data.  A real change of
panel is different in the one way that matters -- which channels survive is
fixed by what the two panels happen to share, which is a prior, and no model
gets to choose them.

The depot contains one pair of cohorts whose panels can be matched channel by
channel: colorectal (58 canonical channels) and GVHD (40), which share 27
canonical markers.  This script runs the real deployment on exactly those
channels and holds the panel fixed, so that only the provenance of the cells
changes.

Arms
----
  within_<cohort>_shared   train on four of the five image folds, then split the
                           held-out fold at random into a calibration half and a
                           deployment half.  Exchangeable by construction, so the
                           certificate is valid: this is what the 58 -> 27 (and
                           40 -> 27) restriction costs with no domain shift in it.
  within_<cohort>_full     the same protocol on the cohort's own full panel, so
                           the restriction is measured inside one protocol.
  cross_<src>_<tgt>        train on the source's 27 channels, then calibrate on
                           half of the *target* cohort and deploy on the other
                           half.  The split is drawn at random inside the target,
                           so it is exchangeable and the only violation left is
                           the change of cohort at fixed panel.
  crossk_<src>_<tgt>       the same model, calibrating on k = 500 / 1000 / 2000
                           target labels drawn at random: the paper's label-cost
                           axis, so this arm can be read next to the tgt_k ladder
                           of scripts/r14_targetcal.py.
  srcnull_<src>_<tgt>      the paper's violation baseline: calibrate on a
                           held-out *source* fold and deploy on the whole target.

Together these decompose the panel question into a size term (full vs shared
inside a cohort), a domain term (shared within vs shared across) and the
exchangeability term (srcnull vs cross), all on the same cells, the same
10 coarse classes and the same five-fold protocol.

Validity is read two ways, as elsewhere in the paper: the 0/1 flag "the realised
risk was under budget on this evaluation", and the one-sided 95%
Clopper-Pearson upper limit of the realised risk among the accepted cells (the
well-powered statement; r20's rule).

Outputs: experiments/r61_shared_panel.csv,
         experiments/r61_shared_panel_summary.csv, work/r61_report.txt
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
from errctl.markerlogic import canonical
from errctl.preprocess import DomainNormaliser
import r02_run_experiments as R2

PAIR = ("CRC_Feb2623", "GVHD_Feb1323")
CLFS = ("logreg", "rf", "hgb")
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
N_SPLITS = 5
KS = (500, 1000, 2000)
MIN_CLASS_TEST = 20
MAX_TRAIN = R2.MAX_TRAIN
N_FOLDS = R2.N_FOLDS

CSV = os.path.join(paths.RESULTS, "r61_shared_panel.csv")
SUMM = os.path.join(paths.RESULTS, "r61_shared_panel_summary.csv")
REPORT = os.path.join(paths.WORK, "r61_report.txt")

COHORT = {}
VOCAB = []


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
    fold = np.full(len(g), -1, dtype=int)
    folds = R2.fold_split(np.unique(imgs), N_FOLDS, seed=0)
    for i, part in enumerate(folds):
        for u in part:
            fold[imgs == u] = i
    return dict(ds=ds, g=g, markers=markers, X=X, imgs=imgs, fold=fold,
                uniq=np.unique(imgs), folds=folds)


def build_vocab(g1, g2):
    c1 = g1["coarse_expert"].value_counts()
    c2 = g2["coarse_expert"].value_counts()
    return sorted(c for c in set(c1.index) & set(c2.index)
                  if c1[c] >= MIN_CLASS_TEST and c2[c] >= MIN_CLASS_TEST)


def _init(ds_list, pair=None):
    global VOCAB
    pair = tuple(pair or PAIR)
    for ds in ds_list:
        COHORT[ds] = load(ds)
    VOCAB = build_vocab(COHORT[pair[0]]["g"], COHORT[pair[1]]["g"])
    a = {canonical(m): i for i, m in enumerate(COHORT[pair[0]]["markers"])}
    b = {canonical(m): i for i, m in enumerate(COHORT[pair[1]]["markers"])}
    for ds, dd in ((pair[0], a), (pair[1], b)):
        COHORT[ds]["shared_cols"] = np.array([dd[c] for c in sorted(set(a) & set(b))])
    COHORT["_shared_names"] = sorted(set(a) & set(b))


def fit_predict(gtr, Xtr, cols, classes, tr_idx, gde, Xde, de_idx, kind, seed,
                domain, cols_de=None):
    """Train on (cols) of one cohort, return scores and errors on the deployment
    set.  ``domain`` is the normaliser domain used for the deployment features:
    'src' for within-cohort deployment, 'tgt' for cross-cohort deployment (the
    per-domain standardisation r02 uses, which is label-free)."""
    ix = {c: i for i, c in enumerate(classes)}
    all_lab = gtr["coarse_expert"].to_numpy()
    keep = np.array([v in ix for v in all_lab[tr_idx]])
    train_idx = tr_idx[keep]
    if train_idx.size > MAX_TRAIN:
        num = np.array([ix[v] for v in all_lab[train_idx]])
        train_idx = R2.subsample_training(train_idx, num, MAX_TRAIN, seed)
    ytr = np.array([ix[v] for v in all_lab[train_idx]], dtype=int)
    Xt = Xtr[np.ix_(train_idx, cols)]
    nz = DomainNormaliser("z_dataset").fit("src", Xt)
    if cols_de is None:
        cols_de = cols
    Xd = Xde[np.ix_(de_idx, cols_de)]
    if domain == "tgt":
        nz.fit("tgt", Xd)
    clf = fit_classifier(nz.transform("src", Xt), ytr, kind=kind, seed=seed)
    probs = clf.predict_proba(nz.transform(domain, Xd)).astype(np.float64)
    lab = np.array([classes[i] for i in probs.argmax(1)], dtype=object)
    truth = gde["coarse_expert"].to_numpy()[de_idx]
    return dict(score=1.0 - probs.max(1), err=(lab != truth).astype(float),
                truth=truth, lab=lab, n_train=int(train_idx.size))


def emit(rows, arm, ds, clf, alpha, rep, n_markers, n_train, n_cal, n_pool,
         accept, o):
    err = o["err"]
    n = int(np.asarray(accept, dtype=bool).sum())
    risk = float(err[np.asarray(accept, dtype=bool)].mean()) if n else float("nan")
    cpu = cp_upper(err[np.asarray(accept, dtype=bool)]) if n else float("nan")
    rows.append(dict(
        arm=arm, dataset=ds, clf=clf, alpha=alpha, rep=rep,
        n_markers=n_markers, n_train=n_train, n_cal=n_cal, n_pool=n_pool,
        n_classes=len(VOCAB), acc=float((o["lab"] == o["truth"]).mean()),
        accept_rate=n / max(err.size, 1), n_accepted=n, selective_risk=risk,
        cp_upper=cpu, covered=int((not np.isfinite(risk)) or risk <= alpha),
        empty=int(n == 0),
        valid=int(np.isfinite(cpu) and cpu <= alpha)))


def within_job(job):
    _, ds, fold, kind, tag = job
    d = COHORT[ds]
    g, X = d["g"], d["X"]
    rows = []
    classes = VOCAB
    cols = (np.arange(len(d["markers"])) if tag == "full" else d["shared_cols"])
    tr_idx = np.nonzero(d["fold"] != fold)[0]
    labels = g["coarse_expert"].to_numpy()
    in_v = np.array([v in set(classes) for v in labels])
    pool = np.nonzero((d["fold"] == fold) & in_v)[0]
    if pool.size < 200:
        return rows
    seed = zlib.crc32(("r61|%s|%d|%s|%s" % (ds, fold, kind, tag)).encode()) & 0xFFFFFFFF
    out = fit_predict(g, X, cols, classes, tr_idx, g, X, pool, kind, seed,
                      domain="src")
    rng = np.random.default_rng(seed + 11)
    n = pool.size
    half = n // 2
    for rep in range(N_SPLITS):
        pick = rng.permutation(n)
        s_cal, e_cal = out["score"][pick[:half]], out["err"][pick[:half]]
        o = dict(score=out["score"][pick][half:], err=out["err"][pick][half:],
                 truth=out["truth"][pick][half:], lab=out["lab"][pick][half:])
        for alpha in ALPHAS:
            ctl = C.SelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal)
            emit(rows, "within_%s_%s" % (ds, tag), ds, kind, alpha, rep,
                 int(cols.size), out["n_train"], int(half), int(n - half),
                 ctl.accept(o["score"]), o)
    return rows


def srcnull_job(job):
    src, tgt, fold, kind = job
    ds_, dt_ = COHORT[src], COHORT[tgt]
    classes = VOCAB
    ix = {c: i for i, c in enumerate(classes)}
    cols = ds_["shared_cols"]
    rows = []
    s_lab = ds_["g"]["coarse_expert"].to_numpy()
    tr_idx = np.nonzero(ds_["fold"] != fold)[0]
    cal_idx = np.nonzero(ds_["fold"] == fold)[0]
    cal_idx = cal_idx[np.array([v in ix for v in s_lab[cal_idx]])]
    if cal_idx.size < 40:
        return rows
    seed = zlib.crc32(("r61s|%s|%s|%d|%s" % (src, tgt, fold, kind)).encode()) & 0xFFFFFFFF
    keep = np.array([v in ix for v in s_lab[tr_idx]])
    train_idx = tr_idx[keep]
    ytr = np.array([ix[v] for v in s_lab[train_idx]], dtype=int)
    Xt = ds_["X"][np.ix_(train_idx, cols)]
    nz = DomainNormaliser("z_dataset").fit("src", Xt)
    clf = fit_classifier(nz.transform("src", Xt), ytr, kind=kind, seed=seed)
    p_cal = clf.predict_proba(nz.transform("src", ds_["X"][np.ix_(cal_idx, cols)])).astype(np.float64)
    s_cal = 1.0 - p_cal.max(1)
    e_cal = (np.array([classes[i] for i in p_cal.argmax(1)], dtype=object)
             != s_lab[cal_idx]).astype(float)
    t_lab = dt_["g"]["coarse_expert"].to_numpy()
    pool = np.nonzero(np.array([v in ix for v in t_lab]))[0]
    tcols = dt_["shared_cols"]
    nz.fit("tgt", dt_["X"][np.ix_(pool, tcols)])
    p = clf.predict_proba(nz.transform("tgt", dt_["X"][np.ix_(pool, tcols)])).astype(np.float64)
    lab = np.array([classes[i] for i in p.argmax(1)], dtype=object)
    o = dict(score=1.0 - p.max(1), err=(lab != t_lab[pool]).astype(float),
             truth=t_lab[pool], lab=lab)
    for alpha in ALPHAS:
        ctl = C.SelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal)
        emit(rows, "srcnull_%s_%s" % (src, tgt), "%s->%s" % (src, tgt), kind,
             alpha, -1, int(cols.size), int(train_idx.size), int(cal_idx.size),
             int(pool.size), ctl.accept(o["score"]), o)
    return rows


def cross_job(job):
    src, tgt, fold, kind = job
    ds_, dt_ = COHORT[src], COHORT[tgt]
    classes = VOCAB
    cols = ds_["shared_cols"]
    rows = []
    tr_idx = np.nonzero(ds_["fold"] != fold)[0]
    t_lab = dt_["g"]["coarse_expert"].to_numpy()
    in_v = np.array([v in set(classes) for v in t_lab])
    pool_all = np.nonzero(in_v)[0]
    seed = zlib.crc32(("r61x|%s|%s|%d|%s" % (src, tgt, fold, kind)).encode()) & 0xFFFFFFFF
    out = fit_predict(ds_["g"], ds_["X"], cols, classes, tr_idx,
                      dt_["g"], dt_["X"], pool_all, kind, seed, domain="tgt",
                      cols_de=dt_["shared_cols"])
    ann = "cross_%s_%s" % (src, tgt)
    pname = "%s->%s" % (src, tgt)
    rng = np.random.default_rng(seed + 17)
    n = pool_all.size
    half = n // 2
    for rep in range(N_SPLITS):
        pick = rng.permutation(n)
        s_cal, e_cal = out["score"][pick[:half]], out["err"][pick[:half]]
        o = dict(score=out["score"][pick][half:], err=out["err"][pick][half:],
                 truth=out["truth"][pick][half:], lab=out["lab"][pick][half:])
        for alpha in ALPHAS:
            ctl = C.SelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal)
            emit(rows, ann, pname, kind, alpha, rep, int(cols.size),
                 out["n_train"], int(half), int(n - half),
                 ctl.accept(o["score"]), o)
    for rep in range(N_SPLITS):
        pick = rng.permutation(n)
        for k in KS:
            if k >= n // 2:
                continue
            s_cal, e_cal = out["score"][pick[:k]], out["err"][pick[:k]]
            o = dict(score=out["score"][pick][k:], err=out["err"][pick][k:],
                     truth=out["truth"][pick][k:], lab=out["lab"][pick][k:])
            for alpha in ALPHAS:
                ctl = C.SelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal)
                emit(rows, "crossk_%s_%s" % (src, tgt), pname, kind, alpha, rep,
                     int(cols.size), out["n_train"], int(k), int(n - k),
                     ctl.accept(o["score"]), o)
    return rows


def write_report(df, summ, txt_path):
    A = []
    A.append("r61 -- a real cross-panel deployment on the shared channels")
    A.append("=" * 78)
    A.append("")
    A.append("Pair: %s (58 channels) <-> %s (40 channels)" % PAIR)
    A.append("Shared canonical markers: %d" % len(COHORT.get("_shared_names", [])))
    A.append("  %s" % ", ".join(COHORT.get("_shared_names", [])))
    A.append("Shared coarse classes with >= %d cells in both cohorts: %d"
             % (MIN_CLASS_TEST, len(VOCAB)))
    A.append("  %s" % ", ".join(VOCAB))
    A.append("")
    A.append("Protocol: 5 image folds, %d random calibration/deployment splits of"
             % N_SPLITS)
    A.append("the deployment pool, delta = %.2f, classifiers %s." % (DELTA, ",".join(CLFS)))
    A.append("")
    A.append("  arm                    alpha  markers  acc    accept  nonvac  "
             "risk   cov    cp-valid  cp-upper")
    for _, r in summ.iterrows():
        A.append("  %-22s %.2f   %5.1f    %.3f  %6.3f  %6.3f  %6.3f  %5.3f  "
                 "%6.3f    %6.3f"
                 % (r["arm"], r["alpha"], r["n_markers"], r["acc"],
                    r["mean_accept"], r["nonvac_frac"], r["mean_risk"],
                    r["coverage_all"], r["cp_valid"], r["mean_cp_upper"]))
    A.append("")
    A.append("Legend.  markers = channels the model saw.  accept = share of the")
    A.append("deployment batch it agrees to annotate.  nonvac = share of units that")
    A.append("answered at all.  risk = realised error among accepted cells.  cov =")
    A.append("share of units whose realised risk was within the budget.  cp-valid =")
    A.append("share of units whose one-sided 95%% Clopper-Pearson upper limit on that")
    A.append("risk is within the budget (the well-powered reading).  srcnull is the")
    A.append("source-calibrated baseline: it is the arm the guarantee does not cover.")
    A.append("")
    A.append("The three contrasts, at alpha = 0.10 (mean over folds and classifiers):")
    A.append(_contrasts(df))
    A.append("")
    txt = "\n".join(A) + "\n"
    with open(txt_path, "w", encoding="ascii", errors="replace") as fh:
        fh.write(txt)
    return txt


def _m(df, arm, col, alpha=0.10):
    s = df[(df.arm == arm) & np.isclose(df.alpha, alpha)]
    return float(s[col].mean()) if len(s) else float("nan")


def _contrasts(df):
    out = []
    for ds in PAIR:
        wf = _m(df, "within_%s_full" % ds, "accept_rate")
        ws = _m(df, "within_%s_shared" % ds, "accept_rate")
        out.append("  %s: panel restriction %d -> %d channels costs %.3f of"
                   % (ds, int(_m(df, "within_%s_full" % ds, "n_markers")),
                      int(_m(df, "within_%s_shared" % ds, "n_markers")), wf - ws))
        out.append("      acceptance (%.3f -> %.3f) at fixed cohort and fixed protocol."
                   % (wf, ws))
    for src, tgt in (PAIR, PAIR[::-1]):
        ws = _m(df, "within_%s_shared" % tgt, "accept_rate")
        cx = _m(df, "cross_%s_%s" % (src, tgt), "accept_rate")
        sn = _m(df, "srcnull_%s_%s" % (src, tgt), "accept_rate")
        cc = _m(df, "cross_%s_%s" % (src, tgt), "covered")
        sc = _m(df, "srcnull_%s_%s" % (src, tgt), "covered")
        ns = int(_m(df, "within_%s_shared" % tgt, "n_markers"))
        out.append("  %s -> %s at %d channels: within-target accept %.3f, cross-cohort"
                   % (src, tgt, ns, ws))
        out.append("      accept %.3f (coverage %.3f), source-calibrated accept %.3f"
                   % (cx, cc, sn))
        out.append("      (coverage %.3f).  The domain shift at fixed panel costs %.3f"
                   % (sc, ws - cx))
        out.append("      of acceptance; the source-calibrated arm is the invalid one,"
                   )
        out.append("      so its %.3f is not a gain -- it covers %.3f of its"
                   % (sn, sc))
        out.append("      non-vacuous decisions against a nominal %.2f." % (1 - DELTA))
    return "\n".join(out)


def main():
    global PAIR, CSV, SUMM, REPORT
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--clfs", default=",".join(CLFS))
    ap.add_argument("--from-csv", action="store_true",
                    help="rebuild the report from the stored CSV instead of running")
    ap.add_argument("--pair", default=",".join(PAIR),
                    help="the two cohorts to put on their shared panel")
    ap.add_argument("--tag", default="",
                    help="suffix for the output files (default: none)")
    args = ap.parse_args()
    PAIR = tuple(x for x in args.pair.split(",") if x)
    CSV = os.path.join(paths.RESULTS, "r61_shared_panel%s.csv" % args.tag)
    SUMM = os.path.join(paths.RESULTS, "r61_shared_panel%s_summary.csv" % args.tag)
    REPORT = os.path.join(paths.WORK, "r61%s_report.txt" % args.tag)
    clfs = [c for c in args.clfs.split(",") if c]
    log("r61_shared_panel.py  pair=%s<->%s  tag=%r" % (PAIR + (args.tag,)))
    _init(list(PAIR), PAIR)   # main-process copy: the report needs the shared panel
    log("shared markers=%d  classes=%d  %s"
        % (len(COHORT["_shared_names"]), len(VOCAB), ",".join(VOCAB)))
    if args.from_csv:
        df = pd.read_csv(CSV)
        summ = (df.groupby(["arm", "alpha"])
                  .agg(n_markers=("n_markers", "mean"), n_pool=("n_pool", "mean"),
                       acc=("acc", "mean"), mean_accept=("accept_rate", "mean"),
                       nonvac_frac=("empty", lambda x: 1.0 - float(x.mean())),
                       mean_risk=("selective_risk", "mean"),
                       coverage_all=("covered", "mean"),
                       cp_valid=("valid", "mean"),
                       mean_cp_upper=("cp_upper", "mean"))
                  .reset_index())
        txt = write_report(df, summ, REPORT)
        log("wrote %s (%d bytes, from %s)" % (REPORT, len(txt), CSV))
        print(txt)
        return 0
    wjobs = [("within", ds, f, k, tag) for ds in PAIR for tag in ("full", "shared")
             for f in range(N_FOLDS) for k in clfs]
    xjobs = [(s, t, f, k) for (s, t) in (PAIR, PAIR[::-1]) for f in range(N_FOLDS)
             for k in clfs]
    rows = []
    with mp.Pool(args.jobs, initializer=_init, initargs=(list(PAIR), tuple(PAIR))) as pool:
        for i, res in enumerate(pool.imap(within_job, wjobs, chunksize=1), 1):
            rows.extend(res)
            log("  within %d/%d" % (i, len(wjobs)))
    log("within arms: %d rows (%.1fs)" % (len(rows), time.time() - t0))
    with mp.Pool(args.jobs, initializer=_init, initargs=(list(PAIR), tuple(PAIR))) as pool:
        sjobs = [(s, t, f, k) for (s, t) in (PAIR, PAIR[::-1]) for f in range(N_FOLDS)
                 for k in clfs]
        for i, res in enumerate(pool.imap(srcnull_job, sjobs, chunksize=1), 1):
            rows.extend(res)
        log("  srcnull arms done (%d jobs)" % len(sjobs))
        for i, res in enumerate(pool.imap(cross_job, xjobs, chunksize=1), 1):
            rows.extend(res)
            log("  cross %d/%d" % (i, len(xjobs)))
    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    log("wrote %s (%d rows)" % (CSV, len(df)))
    summ = (df.groupby(["arm", "alpha"])
              .agg(n_markers=("n_markers", "mean"), n_pool=("n_pool", "mean"),
                   acc=("acc", "mean"), mean_accept=("accept_rate", "mean"),
                   nonvac_frac=("empty", lambda x: 1.0 - float(x.mean())),
                   mean_risk=("selective_risk", "mean"),
                   coverage_all=("covered", "mean"),
                   cp_valid=("valid", "mean"),
                   mean_cp_upper=("cp_upper", "mean"))
              .reset_index())
    summ.to_csv(SUMM, index=False)
    log("wrote %s" % SUMM)
    txt = write_report(df, summ, REPORT)
    log("wrote %s (%d bytes)" % (REPORT, len(txt)))
    print(txt)
    log("done (%.1fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
