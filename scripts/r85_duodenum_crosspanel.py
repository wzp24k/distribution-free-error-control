"""R85 -- a real cross-panel deployment: STELLAR small intestine -> duodenum.

The two panels share 14 markers whose row identity in the duodenum matrix is
unambiguous from the published cell-type annotations.  The source is the
STELLAR CODEX small-intestine table (donor B004, 8 regions).  The target is the
adult human proximal small-intestine CODEX atlas (3 donors, 107 ROIs).

Arms
----
cross_random  train on STELLAR, calibrate on a random half of the target
              cells, deploy on the other half.  The calibration/deployment
              split is exchangeable inside the target, so the certificate is
              valid; the model never sees a target label during training.
srcnull       train and calibrate on STELLAR, deploy on the whole target.
              This is the arm the guarantee does not cover: a source-domain
              calibration is not exchangeable with the target.

The primary endpoint is coverage at alpha = 0.10, five target ROI folds,
ten random splits per fold, three classifiers.
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
from scipy.stats import norm as _norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from errctl import conformal as C  # noqa: E402
from errctl.classifiers import make_classifier  # noqa: E402
from errctl.preprocess import DomainNormaliser  # noqa: E402


class RankNormaliser:
    """log1p, then per-marker within-domain rank -> standard normal."""

    def __init__(self, mode="rank"):
        self.mode = mode
        self.ref_ = {}

    @staticmethod
    def _log(X):
        return np.log1p(np.asarray(X, dtype=np.float64))

    def fit(self, domain, X):
        self.ref_[domain] = np.sort(self._log(X), axis=0)
        return self

    def transform(self, domain, X):
        Z = self._log(X)
        ref = self.ref_[domain]
        n = ref.shape[0]
        out = np.empty_like(Z)
        for j in range(Z.shape[1]):
            lo = np.searchsorted(ref[:, j], Z[:, j], side="left")
            hi = np.searchsorted(ref[:, j], Z[:, j], side="right")
            u = (0.5 * (lo + hi) + 0.5) / n
            out[:, j] = _norm.ppf(np.clip(u, 1e-4, 1.0 - 1e-4))
        return out


def make_normaliser():
    if os.environ.get("R85_NORM", "z") == "rank":
        return RankNormaliser()
    return DomainNormaliser("z_dataset")


class EnsembleClassifier:
    """Equal-weight mean of the logreg, rf and hgb probabilities."""

    def __init__(self, seed=0):
        self.seed = seed
        self.models = []
        self.classes_ = None

    def fit(self, X, y):
        for kind in ("logreg", "rf", "hgb"):
            m = make_classifier(kind, seed=self.seed)
            if kind == "rf":
                m.set_params(n_jobs=1)
            m.fit(X, y)
            self.models.append(m)
        self.classes_ = np.asarray(self.models[0].classes_)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def make_model(kind, seed=0):
    if kind == "ens":
        return EnsembleClassifier(seed=seed)
    return make_classifier(kind, seed=seed)


STELLAR = os.path.join(ROOT, "data", "external", "B004_training_dryad.csv")
DUOD_RAW = os.path.join(ROOT, "work", "dbsearch", "duodenum_raw_mat.csv")
DUOD_CELLS = os.path.join(ROOT, "work", "dbsearch", "duodenum_cells.csv")
EVALS = os.path.join(ROOT, "experiments", "r85_duodenum_crosspanel.csv")
SUMMARY = os.path.join(ROOT, "experiments", "r85_duodenum_crosspanel_summary.csv")
REPORT = os.path.join(ROOT, "work", "r85_duodenum_crosspanel_report.txt")

# STELLAR column -> duodenum row (1-based, as published in raw_mat.csv).
SHARED = [
    ("CD31", 25), ("CD4", 17), ("CD8", 16), ("CD44", 6), ("CD3", 15),
    ("CD90", 8), ("aSMA", 33), ("CD206", 23), ("CD138", 19), ("CD19", 18),
    ("CD163", 22), ("HLADR", 11), ("Podoplanin", 5), ("Vimentin", 10),
]

SOURCE_MAP = {
    "Enterocyte": "Epithelial", "Enterocyte_ITLN1p": "Epithelial",
    "Enterocyte_CD57p": "Epithelial", "TA": "Epithelial", "Goblet": "Epithelial",
    "Paneth": "Epithelial", "Neuroendocrine": "Epithelial",
    "CD8T": "T_CD8", "CD4T": "T_CD4", "B": "B", "Plasma": "Plasma",
    "Macrophage": "Macrophage", "DC": "DC", "Endothelial": "Endothelial",
    "Lymphatic": "Lymphatic", "SmoothMuscle": "Stromal", "Stroma": "Stromal",
    "ICC": "Stromal",
}
TARGET_MAP = {
    "epi_NDRG1": "Epithelial", "epi_CA9": "Epithelial", "epi_GLUT1": "Epithelial",
    "Tcell_CD8": "T_CD8", "Tcell_CD4": "T_CD4", "Bcell": "B", "Pcell": "Plasma",
    "mac": "Macrophage", "cDC1": "DC", "cDC2": "DC",
    "endo_arterial": "Endothelial", "endo_venous": "Endothelial",
    "endo_lymph": "Lymphatic", "fib": "Stromal", "fib_BM": "Stromal",
    "telocyte": "Stromal", "telocyte_tip": "Stromal",
}

CLASSES = ["B", "DC", "Endothelial", "Epithelial", "Lymphatic",
           "Macrophage", "Plasma", "Stromal", "T_CD4", "T_CD8"]
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
N_FOLDS = 5
N_SPLITS = 10
CLFS = ("logreg", "rf", "hgb")
KS = (250, 500, 1000, 2000, 5000, 10000)
ROI_K = 10000
MIN_CLASS = 20
MAX_TRAIN = 60000
SEED = 20261020
PRIMARY_ALPHA = 0.10


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


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def load_source():
    cols = [m for m, _ in SHARED] + ["cell_type_A", "unique_region"]
    d = pd.read_csv(STELLAR, usecols=cols, low_memory=False)
    d = d[d["cell_type_A"].isin(SOURCE_MAP)].copy()
    d["label"] = d["cell_type_A"].map(SOURCE_MAP)
    X = d[[m for m, _ in SHARED]].to_numpy(dtype=np.float64)
    y = d["label"].astype(str).to_numpy()
    groups = d["unique_region"].astype(str).to_numpy()
    return X, y, groups


def load_target():
    raw = np.loadtxt(DUOD_RAW, delimiter=",", skiprows=1, dtype=np.float64)
    rows = [r - 1 for _, r in SHARED]
    X = raw[rows].T
    cells = pd.read_csv(DUOD_CELLS, usecols=["cell_name", "sample", "ROI",
                                             "cell_type"], low_memory=False)
    ok = cells["cell_type"].isin(TARGET_MAP).to_numpy()
    X = X[ok]
    cells = cells.loc[ok].reset_index(drop=True)
    y = cells["cell_type"].map(TARGET_MAP).astype(str).to_numpy()
    groups = (cells["sample"].astype(str) + "|" +
              cells["ROI"].astype(str)).to_numpy()
    return X, y, groups


def encode(y, classes):
    ix = {c: i for i, c in enumerate(classes)}
    return ix, np.array([ix[v] for v in y], dtype=int)


def subsample(idx, y, cap, seed):
    if idx.size <= cap:
        return idx
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y[idx]):
        pool = idx[y[idx] == cls]
        take = max(3, int(round(cap * pool.size / idx.size)))
        keep.append(rng.choice(pool, size=min(take, pool.size), replace=False))
    return np.concatenate(keep)


def target_folds(groups, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())), dtype=object)
    rng.shuffle(uniq)
    return np.array_split(uniq, N_FOLDS)


def source_folds(groups, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())), dtype=object)
    rng.shuffle(uniq)
    return np.array_split(uniq, N_FOLDS)


G = {}


def _init(src, tgt):
    G["Xs"], G["ys"], G["gs"] = src
    G["Xt"], G["yt"], G["gt"] = tgt
    G["ix"], G["ysi"] = encode(G["ys"], CLASSES)
    _, G["yti"] = encode(G["yt"], CLASSES)
    G["tfolds"] = target_folds(G["gt"])
    G["sfolds"] = source_folds(G["gs"])


def emit(rows, arm, clf, alpha, rep, fold, n_train, n_cal, n_pool,
         accept, err, lab, truth, k=0):
    n = int(np.asarray(accept, dtype=bool).sum())
    a = np.asarray(accept, dtype=bool)
    risk = float(err[a].mean()) if n else float("nan")
    cpu = cp_upper(err[a]) if n else float("nan")
    rows.append(dict(
        arm=arm, clf=clf, alpha=alpha, rep=rep, fold=fold, k=k,
        n_train=int(n_train), n_cal=int(n_cal), n_pool=int(n_pool),
        acc=float((lab == truth).mean()),
        accept_rate=float(a.mean()), n_accepted=n,
        selective_risk=risk, cp_upper=cpu,
        covered=int((not np.isfinite(risk)) or risk <= alpha),
        empty=int(n == 0),
        valid=int(np.isfinite(cpu) and cpu <= alpha),
    ))


def cross_job(spec):
    fold, kind = spec
    Xs, ysi = G["Xs"], G["ysi"]
    Xt, yti, gt = G["Xt"], G["yti"], G["gt"]
    held = set(G["tfolds"][fold].tolist())
    pool = np.nonzero(np.array([g in held for g in gt]))[0]
    if pool.size < 200:
        return []
    seed = zlib.crc32(("r85x|%d|%s" % (fold, kind)).encode()) & 0xFFFFFFFF
    tr = subsample(np.arange(Xs.shape[0]), ysi, MAX_TRAIN, seed)
    nz = make_normaliser().fit("src", Xs[tr])
    clf = make_model(kind, seed=seed)
    if kind == "rf":
        clf.set_params(n_jobs=4)
    clf.fit(nz.transform("src", Xs[tr]), ysi[tr])
    nz.fit("tgt", Xt[pool])
    p = clf.predict_proba(nz.transform("tgt", Xt[pool]))
    lab = np.array([CLASSES[int(clf.classes_[i])] for i in p.argmax(1)],
                   dtype=object)
    truth = np.array([CLASSES[i] for i in yti[pool]], dtype=object)
    score = 1.0 - p.max(1)
    err = (lab != truth).astype(float)
    out = []
    rows = []
    rng = np.random.default_rng(seed + 17)
    n = pool.size
    half = n // 2
    for rep in range(N_SPLITS):
        pick = rng.permutation(n)
        s_cal, e_cal = score[pick[:half]], err[pick[:half]]
        s_ev, e_ev = score[pick[half:]], err[pick[half:]]
        l_ev, t_ev = lab[pick[half:]], truth[pick[half:]]
        for alpha in ALPHAS:
            ctl = C.SelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal)
            out.append((alpha, rep, ctl.accept(s_ev), e_ev, l_ev, t_ev))
        for k in KS:
            if k >= n - 25:
                continue
            s_k, e_k = score[pick[:k]], err[pick[:k]]
            s_rest, e_rest = score[pick[k:]], err[pick[k:]]
            l_rest, t_rest = lab[pick[k:]], truth[pick[k:]]
            for alpha in ALPHAS:
                ctl = C.SelectiveRiskController(alpha, DELTA).fit(s_k, e_k)
                emit(rows, "crossk", kind, alpha, rep, fold, tr.size, k,
                     n - k, ctl.accept(s_rest), e_rest, l_rest, t_rest, k=k)
    for alpha, rep, acc, e_ev, l_ev, t_ev in out:
        emit(rows, "cross_random", kind, alpha, rep, fold, tr.size,
             half, n - half, acc, e_ev, l_ev, t_ev)
    log("cross fold %d %-6s n=%d acc=%.3f" % (fold, kind, n,
                                              (lab == truth).mean()))
    return rows


def srcnull_job(spec):
    fold, kind = spec
    Xs, ysi, gs = G["Xs"], G["ysi"], G["gs"]
    Xt, yti = G["Xt"], G["yti"]
    held = set(G["sfolds"][fold].tolist())
    tr = np.nonzero(np.array([g not in held for g in gs]))[0]
    cal = np.nonzero(np.array([g in held for g in gs]))[0]
    if cal.size < 40:
        return []
    seed = zlib.crc32(("r85s|%d|%s" % (fold, kind)).encode()) & 0xFFFFFFFF
    tr = subsample(tr, ysi, MAX_TRAIN, seed)
    cal = cal[:10000]
    nz = make_normaliser().fit("src", Xs[tr])
    clf = make_model(kind, seed=seed)
    if kind == "rf":
        clf.set_params(n_jobs=4)
    clf.fit(nz.transform("src", Xs[tr]), ysi[tr])
    p_cal = clf.predict_proba(nz.transform("src", Xs[cal]))
    lab_cal = np.array([CLASSES[int(clf.classes_[i])] for i in p_cal.argmax(1)],
                       dtype=object)
    truth_cal = np.array([CLASSES[i] for i in ysi[cal]], dtype=object)
    s_cal = 1.0 - p_cal.max(1)
    e_cal = (lab_cal != truth_cal).astype(float)
    nz.fit("tgt", Xt)
    p = clf.predict_proba(nz.transform("tgt", Xt))
    lab = np.array([CLASSES[int(clf.classes_[i])] for i in p.argmax(1)],
                   dtype=object)
    truth = np.array([CLASSES[i] for i in yti], dtype=object)
    score = 1.0 - p.max(1)
    err = (lab != truth).astype(float)
    rows = []
    for alpha in ALPHAS:
        ctl = C.SelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal)
        emit(rows, "srcnull", kind, alpha, -1, fold, tr.size, cal.size,
             Xt.shape[0], ctl.accept(score), err, lab, truth)
    log("srcnull fold %d %-6s cal=%d acc=%.3f" % (fold, kind, cal.size,
                                                  (lab == truth).mean()))
    return rows


def within_job(spec):
    arm, fold, kind = spec
    if arm == "within_target":
        X, y, groups, folds, domain = (G["Xt"], G["yti"], G["gt"],
                                       G["tfolds"], "tgt")
    else:
        X, y, groups, folds, domain = (G["Xs"], G["ysi"], G["gs"],
                                       G["sfolds"], "src")
    held = set(folds[fold].tolist())
    tr = np.nonzero(np.array([g not in held for g in groups]))[0]
    pool = np.nonzero(np.array([g in held for g in groups]))[0]
    if tr.size < 200 or pool.size < 200:
        return []
    seed = zlib.crc32(("r85w|%s|%d|%s" % (arm, fold, kind)).encode()) & 0xFFFFFFFF
    tr = subsample(tr, y, MAX_TRAIN, seed)
    nz = make_normaliser().fit(domain, X[tr])
    clf = make_model(kind, seed=seed)
    if kind == "rf":
        clf.set_params(n_jobs=4)
    clf.fit(nz.transform(domain, X[tr]), y[tr])
    p = clf.predict_proba(nz.transform(domain, X[pool]))
    lab = np.array([CLASSES[int(clf.classes_[i])] for i in p.argmax(1)],
                   dtype=object)
    truth = np.array([CLASSES[i] for i in y[pool]], dtype=object)
    score = 1.0 - p.max(1)
    err = (lab != truth).astype(float)
    rows = []
    rng = np.random.default_rng(seed + 23)
    n = pool.size
    half = n // 2
    for rep in range(N_SPLITS):
        pick = rng.permutation(n)
        s_cal, e_cal = score[pick[:half]], err[pick[:half]]
        s_ev, e_ev = score[pick[half:]], err[pick[half:]]
        l_ev, t_ev = lab[pick[half:]], truth[pick[half:]]
        for alpha in ALPHAS:
            ctl = C.SelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal)
            emit(rows, arm, kind, alpha, rep, fold, tr.size, half, n - half,
                 ctl.accept(s_ev), e_ev, l_ev, t_ev)
    log("%s fold %d %-6s n=%d acc=%.3f" % (arm, fold, kind, n,
                                           (lab == truth).mean()))
    return rows


def cross_roi_job(spec):
    fold, kind = spec
    Xs, ysi = G["Xs"], G["ysi"]
    Xt, yti, gt = G["Xt"], G["yti"], G["gt"]
    held = set(G["tfolds"][fold].tolist())
    pool = np.nonzero(np.array([g in held for g in gt]))[0]
    cal_pool = np.nonzero(np.array([g not in held for g in gt]))[0]
    if pool.size < 200 or cal_pool.size < ROI_K:
        return []
    seed = zlib.crc32(("r85r|%d|%s" % (fold, kind)).encode()) & 0xFFFFFFFF
    tr = subsample(np.arange(Xs.shape[0]), ysi, MAX_TRAIN, seed)
    nz = make_normaliser().fit("src", Xs[tr])
    clf = make_model(kind, seed=seed)
    if kind == "rf":
        clf.set_params(n_jobs=4)
    clf.fit(nz.transform("src", Xs[tr]), ysi[tr])
    nz.fit("tgt", Xt)
    p_cal = clf.predict_proba(nz.transform("tgt", Xt[cal_pool]))
    lab_cal = np.array([CLASSES[int(clf.classes_[i])] for i in p_cal.argmax(1)],
                       dtype=object)
    truth_cal = np.array([CLASSES[i] for i in yti[cal_pool]], dtype=object)
    s_cal_all = 1.0 - p_cal.max(1)
    e_cal_all = (lab_cal != truth_cal).astype(float)
    p = clf.predict_proba(nz.transform("tgt", Xt[pool]))
    lab = np.array([CLASSES[int(clf.classes_[i])] for i in p.argmax(1)],
                   dtype=object)
    truth = np.array([CLASSES[i] for i in yti[pool]], dtype=object)
    score = 1.0 - p.max(1)
    err = (lab != truth).astype(float)
    rows = []
    rng = np.random.default_rng(seed + 29)
    for rep in range(N_SPLITS):
        pick = rng.permutation(cal_pool.size)[:ROI_K]
        for alpha in ALPHAS:
            ctl = C.SelectiveRiskController(alpha, DELTA).fit(
                s_cal_all[pick], e_cal_all[pick])
            emit(rows, "cross_roi", kind, alpha, rep, fold, tr.size, ROI_K,
                 pool.size, ctl.accept(score), err, lab, truth, k=ROI_K)
    log("cross_roi fold %d %-6s cal=%d deploy=%d acc=%.3f"
        % (fold, kind, ROI_K, pool.size, (lab == truth).mean()))
    return rows


def summarise(df):
    out = []
    for (arm, alpha, k), g in df.groupby(["arm", "alpha", "k"], sort=True):
        n = len(g)
        n_cov = int(g["covered"].sum())
        lo, hi = wilson(n_cov, n)
        nv = g[g["empty"] == 0]
        out.append(dict(
            arm=arm, alpha=alpha, k=k, n_eval=n,
            coverage=n_cov / n, coverage_lo=lo, coverage_hi=hi,
            frac_nonvacuous=float((g["empty"] == 0).mean()),
            mean_accept=float(g["accept_rate"].mean()),
            mean_accept_nonvac=float(nv["accept_rate"].mean()) if len(nv) else np.nan,
            risk_nonvac=float(nv["selective_risk"].mean()) if len(nv) else np.nan,
            cp_valid=float(g["valid"].mean()),
            mean_cp_upper=float(g["cp_upper"].mean()),
            acc=float(g["acc"].mean()),
        ))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--arms", default="cross,srcnull")
    ap.add_argument("--clfs", default=",".join(CLFS))
    ap.add_argument("--append", action="store_true",
                    help="keep rows for arms not run in this invocation")
    ap.add_argument("--from-evals", action="store_true",
                    help="rebuild the summary and report from saved evaluations")
    args = ap.parse_args()
    log("loading source", STELLAR)
    src = load_source()
    log("source", src[0].shape, "regions", len(set(src[2].tolist())))
    log("loading target", DUOD_CELLS)
    tgt = load_target()
    log("target", tgt[0].shape, "ROIs", len(set(tgt[2].tolist())))
    clfs = [c for c in args.clfs.split(",") if c]
    jobs = []
    if "cross" in args.arms:
        jobs += [("cross", f, k) for f in range(N_FOLDS) for k in clfs]
    if "srcnull" in args.arms:
        jobs += [("srcnull", f, k) for f in range(N_FOLDS) for k in clfs]
    for arm in ("within_target", "within_source"):
        if arm in args.arms:
            jobs += [(arm, f, k) for f in range(N_FOLDS) for k in clfs]
    if "cross_roi" in args.arms:
        jobs += [("cross_roi", f, k) for f in range(N_FOLDS) for k in clfs]
    t0 = time.time()
    if args.from_evals:
        df = pd.read_csv(EVALS)
        if "k" not in df.columns:
            df["k"] = 0
    else:
        rows = []
        with mp.Pool(args.jobs, initializer=_init, initargs=(src, tgt)) as pool:
            for i, res in enumerate(pool.imap(_dispatch, jobs, chunksize=1), 1):
                rows.extend(res)
                log("job %d/%d rows=%d" % (i, len(jobs), len(rows)))
        df = pd.DataFrame(rows)
        if args.append and os.path.exists(EVALS):
            old = pd.read_csv(EVALS)
            if "k" not in old.columns:
                old["k"] = 0
            old = old[~old["arm"].isin(set(df["arm"].unique()))]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(EVALS, index=False)
    sm = summarise(df)
    sm.to_csv(SUMMARY, index=False)
    A = ["R85 -- STELLAR small intestine -> duodenum, 14 shared markers",
         "=" * 72,
         "Source: %s (%d cells, %d regions)" % (STELLAR, src[0].shape[0],
                                                len(set(src[2].tolist()))),
         "Target: %s (%d cells, %d ROIs)" % (DUOD_CELLS, tgt[0].shape[0],
                                             len(set(tgt[2].tolist()))),
         "Markers: %s" % ", ".join(m for m, _ in SHARED),
         "Classes: %s" % ", ".join(CLASSES),
         "Protocol: %d target folds x %d splits x %d classifiers, delta=%.2f"
         % (N_FOLDS, N_SPLITS, len(clfs), DELTA), ""]
    for _, r in sm.iterrows():
        A.append("%-12s k=%5d alpha %.2f  n=%3d  coverage %.3f [%.3f, %.3f]  "
                 "nonvac %.3f  accept %.3f  risk %.3f  cp-valid %.3f  acc %.3f"
                 % (r.arm, r.k, r.alpha, r.n_eval, r.coverage, r.coverage_lo,
                    r.coverage_hi, r.frac_nonvacuous, r.mean_accept,
                    r.risk_nonvac, r.cp_valid, r.acc))
    A.append("")
    pr = sm[(sm.arm == "cross_random") & np.isclose(sm.alpha, PRIMARY_ALPHA)]
    if len(pr):
        r = pr.iloc[0]
        ok = (r.coverage >= 0.90 and r.coverage_lo >= 0.85
              and r.frac_nonvacuous >= 0.80 and r.mean_accept >= 0.50)
        A.append("PRIMARY cross_random alpha=0.10: %s" % ("PASS" if ok else "FAIL"))
    open(REPORT, "w", encoding="utf-8").write("\n".join(A) + "\n")
    print("\n".join(A))
    log("done %.1fs" % (time.time() - t0))


def _dispatch(job):
    arm, fold, kind = job
    if arm == "cross":
        return cross_job((fold, kind))
    if arm == "srcnull":
        return srcnull_job((fold, kind))
    if arm == "cross_roi":
        return cross_roi_job((fold, kind))
    return within_job(job)


if __name__ == "__main__":
    main()
