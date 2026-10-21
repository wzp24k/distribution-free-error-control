"""R72 -- can the throughput of a cross-panel deployment be repaired, and does
a same-tissue panel change behave differently from a cross-tissue one?

Five deployments, one protocol.

Part A -- the repair on a real panel change.  ``scripts/r71_crosspanel_price.py``
spends k labels of the deployment cohort on the *calibrator* alone and leaves the
model frozen; acceptance there reaches 0.03 at a 10% budget and 0.20 at 20%, and
the ladder is still rising at 8,000 labels.  The paper already reports, inside one
cohort, that the same labels spent on the model are worth more than the same
labels spent on the calibrator.  This part asks that question across a real change
of antibody panel: with the model refitted on half of the k labels and the
certificate calibrated on the other half, how much acceptance does the same label
budget buy.

Part B -- a same-tissue panel change.  Every cross-panel pair the paper reports
changes the tissue and the antibody panel together, so the two effects cannot be
separated.  The MIBI-TOF tissue microarray of the Angelo laboratory carries a
colon adenocarcinoma core (R8C1, 8,241 cells; tissue_data.csv) acquired with a
15-marker panel, and the depot carries a colorectal cohort acquired with 58
channels; the two share 13 canonical markers.  That is a change of panel,
laboratory and platform at a fixed tissue, and it is the missing control.

Protocol, identical across all parts: train on the source, score the target with the
target's own unlabelled standardisation, then for each label budget k draw k
target cells at random, and

  tgt_k<K>      calibrate the certificate on those k cells, model frozen; this is
                the r71 arm, so the ladder can be read next to r71 directly;
  tgt_k<K>/2    the frozen model, calibrator on only k/2 cells: the control that
                holds the calibration budget fixed when the other half is spent
                on the model instead;
  adapt_k<K>    split the k cells in half, refit the model on the source training
                set plus one half, calibrate on the other half;
  ora_k<K>      the largest target-calibration cut that meets the budget when the
                deployment labels pick it, the unattainable ceiling.

Outputs: experiments/r72_panel_repair.csv, work/r72_report.txt
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import r61_shared_panel as R61   # noqa: E402
import r69_external2 as R69      # noqa: E402
import r55_stellar_cellsplit as R55   # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)
KS = (250, 500, 1000, 2000, 4000)
DELTA = 0.10
CLFS = ("logreg", "rf", "hgb")
N_FOLDS = 5
N_SPLITS = 3
NEW_PAIR = ("CRC_Feb2623", "GVHD_Feb1323")
COLON_CORE = "R8C1"
DEPOT_COLON = "CRC_Feb2623"
BREAST_CORES = ("R6C10", "R6C11")
DEPOT_BREAST = "Breast_Jun0524"
TONSIL_CORES = ("R1C2", "R1C3", "R1C4", "R1C5")
PART_D = ("D_cross_platform_tonsil", "STELLAR_tonsil->MIBITOF_tonsil")
PART_E = ("E_cross_platform_colon", "STELLAR_colon->MIBITOF_R8C1")
# The STELLAR intestine vocabulary coarsened onto the microarray's own.  The
# absorptive, secretory and stem compartments are one class on the MIBI panel.
HARM_INTESTINE = {
    "CD4T": "Tcell", "CD8T": "Tcell", "Macrophage": "Macrophage", "DC": "DC",
    "Enterocyte": "Epithelial", "Enterocyte_CD57p": "Epithelial",
    "Enterocyte_ITLN1p": "Epithelial", "Goblet": "Epithelial",
    "TA": "Epithelial", "Paneth": "Epithelial",
    "Endothelial": "Endothelial", "Stroma": "Fibroblast",
}
N_FOLDS_E = 4
# The STELLAR tonsil reference vocabulary coarsened onto the microarray's own.
# Only unambiguous equivalences are carried; the remaining STELLAR classes
# (plasma, podoplanin, nerve, smooth muscle) are left out of the shared set.
HARM_STELLAR = {
    "B": "Bcell", "T": "Tcell", "Endothelial": "Endothelial",
    "Squamous_epithelial": "Epithelial", "Innate": "Macrophage",
    "Stroma": "Fibroblast",
}
PARTS_B = [
    ("b", ("B_same_tissue_panel",
           "depot_%s->MIBITOF_%s" % (DEPOT_COLON, COLON_CORE),
           DEPOT_COLON, (COLON_CORE,))),
    ("c", ("C_same_tissue_panel_breast",
           "depot_%s->MIBITOF_breast" % DEPOT_BREAST,
           DEPOT_BREAST, BREAST_CORES)),
]

CSV = os.path.join(R61.paths.RESULTS, "r72_panel_repair.csv")
REPORT = os.path.join(R61.paths.WORK, "r72_report.txt")


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"),
                       " ".join(str(p) for p in parts)), flush=True)


def oracle_lambda(score, err, alpha, min_accept=25):
    order = np.argsort(score, kind="mergesort")
    s, e = score[order], err[order]
    cum = np.cumsum(e)
    k = np.arange(1, s.size + 1)
    ok = (cum / k <= alpha) & (k >= min_accept)
    return float(s[np.nonzero(ok)[0][-1]]) if ok.any() else -np.inf


def score_err(clf, Xd, truth_ix):
    p = clf.predict_proba(Xd)
    lab = p.argmax(1)
    return 1.0 - p.max(1), (lab != truth_ix).astype(float), lab


def pilot_oracle_accept(score, err, alpha, min_accept=25):
    """Largest pilot acceptance whose pilot risk meets the budget.

    This uses only the k labelled pilot cells, so it can be computed before a
    refit is bought.  It is an estimate of the ceiling, not the deployment
    oracle, which sees every target label.
    """
    lam = oracle_lambda(score, err, alpha, min_accept=min_accept)
    if not np.isfinite(lam):
        return 0.0
    return float((score <= lam).mean())


def emit(rows, tag, pair, kind, fold, rep, arm, k, n_train, n_pool, alpha,
         n_cal, n_labels, score_dep, err_dep, acc, n_markers, n_classes,
         pilot_score=None, pilot_err=None, pilot_accept=None,
         acc_ud=None, pilot_accept_ud=None):
    lam = oracle_lambda(score_dep, err_dep, alpha)
    o_acc = score_dep <= lam
    p_acc = (pilot_oracle_accept(pilot_score, pilot_err, alpha)
             if pilot_score is not None else np.nan)
    p_cert = float(pilot_accept) if pilot_accept is not None else np.nan
    row = dict(
        part=tag, pair=pair, clf=kind, fold=fold, rep=rep, arm=arm, k=k,
        n_train=n_train, n_pool=n_pool, alpha=alpha, n_cal=n_cal,
        n_labels=n_labels,
        n_markers=n_markers, n_classes=n_classes,
        n_accepted=int(acc.sum()), accept_rate=float(acc.mean()),
        selective_risk=float(err_dep[acc].mean()) if acc.any() else np.nan,
        covered=int((not acc.any()) or err_dep[acc].mean() <= alpha),
        cp_valid=int((not acc.any()) or R61.cp_upper(err_dep[acc]) <= alpha),
        cp_upper=float(R61.cp_upper(err_dep[acc])) if acc.any() else np.nan,
        ratio=float(acc.mean() / o_acc.mean()) if o_acc.any() else np.nan,
        oracle_accept=float(o_acc.mean()),
        pilot_oracle_accept=p_acc,
        pilot_cert_accept=p_cert,
        pilot_ratio=float(p_cert / p_acc) if p_acc > 0 else np.nan)
    if acc_ud is not None:
        p_cert_ud = (float(pilot_accept_ud)
                     if pilot_accept_ud is not None else np.nan)
        row.update(
            n_accepted_ud=int(acc_ud.sum()),
            accept_rate_ud=float(acc_ud.mean()),
            selective_risk_ud=(float(err_dep[acc_ud].mean())
                               if acc_ud.any() else np.nan),
            covered_ud=int((not acc_ud.any())
                           or err_dep[acc_ud].mean() <= alpha),
            cp_valid_ud=int((not acc_ud.any())
                            or R61.cp_upper(err_dep[acc_ud]) <= alpha),
            cp_upper_ud=(float(R61.cp_upper(err_dep[acc_ud]))
                         if acc_ud.any() else np.nan),
            ratio_ud=(float(acc_ud.mean() / o_acc.mean())
                      if o_acc.any() else np.nan),
            pilot_cert_accept_ud=p_cert_ud,
            pilot_ratio_ud=(float(p_cert_ud / p_acc)
                            if (p_acc > 0 and np.isfinite(p_cert_ud))
                            else np.nan))
    rows.append(row)


def arms(rows, tag, pair, kind, fold, n_src, n_pool, score, err, yt, seed,
         refit, n_markers, n_classes):
    n = score.size
    rng = np.random.default_rng(seed)
    for rep in range(N_SPLITS):
        pick = rng.permutation(n)
        for k in KS:
            if k >= n - 25:
                continue
            cal, dep = pick[:k], pick[k:]
            sd, ed = score[dep], err[dep]
            for alpha in ALPHAS:
                ctl = R61.C.SelectiveRiskController(alpha, DELTA).fit(
                    score[cal], err[cal])
                ctl_ud = R61.C.SelectiveRiskController(alpha, DELTA).fit(
                    score[cal], err[cal], target_scores=sd)
                emit(rows, tag, pair, kind, fold, rep, "tgt_k%d" % k, k,
                     n_src, n, alpha, k, k, sd, ed, ctl.accept(sd),
                     n_markers, n_classes, score[cal], err[cal],
                     pilot_accept=ctl.accept(score[cal]).mean(),
                     acc_ud=ctl_ud.accept(sd),
                     pilot_accept_ud=ctl_ud.accept(score[cal]).mean())
            half = k // 2
            for alpha in ALPHAS:
                ctl = R61.C.SelectiveRiskController(alpha, DELTA).fit(
                    score[cal[:half]], err[cal[:half]])
                ctl_ud = R61.C.SelectiveRiskController(alpha, DELTA).fit(
                    score[cal[:half]], err[cal[:half]], target_scores=sd)
                emit(rows, tag, pair, kind, fold, rep, "tgt_k%d_half" % k, k,
                     n_src, n, alpha, half, half, sd, ed, ctl.accept(sd),
                     n_markers, n_classes, score[cal], err[cal],
                     pilot_accept=ctl.accept(score[cal]).mean(),
                     acc_ud=ctl_ud.accept(sd),
                     pilot_accept_ud=ctl_ud.accept(score[cal]).mean())
            if refit is None:
                continue
            mask = np.zeros(n, dtype=bool)
            mask[cal[:half]] = True
            s2, e2, _ = refit(mask)
            sd2, ed2 = s2[dep], e2[dep]
            for alpha in ALPHAS:
                ctl = R61.C.SelectiveRiskController(alpha, DELTA).fit(
                    s2[cal[half:]], e2[cal[half:]])
                ctl_ud = R61.C.SelectiveRiskController(alpha, DELTA).fit(
                    s2[cal[half:]], e2[cal[half:]], target_scores=sd2)
                emit(rows, tag, pair, kind, fold, rep, "adapt_k%d" % k, k,
                     n_src + half, n, alpha, k - half, k, sd2, ed2,
                     ctl.accept(sd2), n_markers, n_classes,
                     score[cal], err[cal],
                     pilot_accept=ctl.accept(s2[cal[half:]]).mean(),
                     acc_ud=ctl_ud.accept(sd2),
                     pilot_accept_ud=ctl_ud.accept(s2[cal[half:]]).mean())


# ------------------------------------------------------------------ part A
def job_a(spec):
    src, tgt, fold, kind = spec
    ds_, dt_ = R61.COHORT[src], R61.COHORT[tgt]
    classes = R61.VOCAB
    ix = {c: i for i, c in enumerate(classes)}
    cols = ds_["shared_cols"]
    n_markers = int(cols.size)
    s_lab = ds_["g"]["coarse_expert"].to_numpy()
    t_lab = dt_["g"]["coarse_expert"].to_numpy()
    tr_idx = np.nonzero(ds_["fold"] != fold)[0]
    tr_idx = tr_idx[np.array([v in ix for v in s_lab[tr_idx]])]
    pool = np.nonzero(np.array([v in ix for v in t_lab]))[0]
    if tr_idx.size < 100 or pool.size < 300:
        return []
    seed = zlib.crc32(("r72a|%s|%s|%d|%s" % (src, tgt, fold, kind)).encode()) & 0xFFFFFFFF
    # subsample_training indexes the label array with *global* cell indices,
    # so the codes must cover the whole cohort, not just the kept cells
    lab_code = np.array([ix.get(v, -1) for v in s_lab], dtype=int)
    tr_idx = R61.R2.subsample_training(tr_idx, lab_code, R61.MAX_TRAIN, seed)
    ytr = np.array([ix[v] for v in s_lab[tr_idx]], dtype=int)
    yt = np.array([ix[v] for v in t_lab[pool]], dtype=int)
    Xt = ds_["X"][np.ix_(tr_idx, cols)]
    nz = R61.DomainNormaliser("z_dataset").fit("src", Xt)
    Xd = dt_["X"][np.ix_(pool, dt_["shared_cols"])]
    nz.fit("tgt", Xd)
    Ztr = nz.transform("src", Xt)
    Zt = nz.transform("tgt", Xd)
    clf = R61.fit_classifier(Ztr, ytr, kind=kind, seed=seed)
    score, err, _ = score_err(clf, Zt, yt)

    def refit(mask):
        c2 = R61.fit_classifier(np.vstack([Ztr, Zt[mask]]),
                                np.concatenate([ytr, yt[mask]]),
                                kind=kind, seed=seed + 7)
        s2, e2, _ = score_err(c2, Zt, yt)
        return s2, e2, None

    rows = []
    arms(rows, "A_cross_tissue_panel", "%s->%s" % (src, tgt), kind, fold,
         int(tr_idx.size), int(pool.size), score, err, yt, seed + 11, refit,
         n_markers, len(classes))
    return rows


# ------------------------------------------------------------------ part D
def job_d(spec):
    fold, kind = spec
    src = R55.load_cohort("STELLAR_Tonsil")
    ang = R69.load_cohort("EXT2_MIBITOF_Angelo")
    core = np.array([s.rsplit("_", 1)[-1] in TONSIL_CORES
                     for s in ang["img"].tolist()])
    lab_t = R69._harmonise(ang["cls"][ang["y"]],
                           R69.HARM_NEW["EXT2_MIBITOF_Angelo"])
    lab_s = np.asarray([HARM_STELLAR.get(str(v), "Other")
                        for v in src["cls"][src["y"]]])
    pairs = R69.shared_pairs(list(src["markers"]), ang["markers"])
    if len(pairs) < 4:
        return []
    scols = [p[1] for p in pairs]
    acols = [p[2] for p in pairs]
    cnt_s = pd.Series(lab_s).value_counts()
    cnt_t = pd.Series(lab_t[core]).value_counts()
    classes = sorted(c for c in set(cnt_s.index) & set(cnt_t.index)
                     if cnt_s[c] >= 30 and cnt_t[c] >= 30)
    if len(classes) < 3:
        return []
    ix = {c: i for i, c in enumerate(classes)}
    tr_all = np.nonzero(np.array([v in ix for v in lab_s]))[0]
    pool = np.nonzero(core & np.array([v in ix for v in lab_t]))[0]
    if tr_all.size < 500 or pool.size < 300:
        return []
    # the STELLAR tonsil table is one image, so the source fold is a random
    # 80/20 cell split rather than an image group; only the training set moves
    rng = np.random.default_rng(1000 + fold)
    perm = rng.permutation(tr_all.size)
    tr = tr_all[perm[:int(0.8 * tr_all.size)]]
    seed = zlib.crc32(("r72d|%d|%s" % (fold, kind)).encode()) & 0xFFFFFFFF
    lab_code = np.array([ix.get(v, -1) for v in lab_s], dtype=int)
    tr = R61.R2.subsample_training(tr, lab_code, R61.MAX_TRAIN, seed)
    ytr = np.array([ix[v] for v in lab_s[tr]], dtype=int)
    yt = np.array([ix[v] for v in lab_t[pool]], dtype=int)
    Xt = src["X"][np.ix_(tr, scols)]
    nz = R61.DomainNormaliser("z_dataset").fit("src", Xt)
    Xd = ang["X"][np.ix_(pool, acols)]
    nz.fit("tgt", Xd)
    Ztr = nz.transform("src", Xt)
    Zt = nz.transform("tgt", Xd)
    clf = R61.fit_classifier(Ztr, ytr, kind=kind, seed=seed)
    score, err, _ = score_err(clf, Zt, yt)

    def refit(mask):
        c2 = R61.fit_classifier(np.vstack([Ztr, Zt[mask]]),
                                np.concatenate([ytr, yt[mask]]),
                                kind=kind, seed=seed + 7)
        s2, e2, _ = score_err(c2, Zt, yt)
        return s2, e2, None

    rows = []
    arms(rows, PART_D[0], PART_D[1], kind, fold, int(tr.size), int(pool.size),
         score, err, yt, seed + 13, refit, len(pairs), len(classes))
    for r in rows:
        r["n_train_src"] = int(tr.size)
    return rows


# ------------------------------------------------------------------ part B
def job_e(spec):
    """A second cross-platform pair, with a region-grouped source fold.

    Part D trains on a random cell split of one STELLAR tonsil image, so its
    source side is not grouped.  This job uses the four colon regions of the
    STELLAR intestine table: fold f trains on the other three and never sees
    region f, which is how the source heterogeneity enters the fold spread.
    The deployment cohort is the R8C1 colon adenocarcinoma core of the MIBI-TOF
    array, so panel, laboratory and platform all change."""
    fold, kind = spec
    src = R55.load_cohort("STELLAR_Intestine_B004")
    ang = R69.load_cohort("EXT2_MIBITOF_Angelo")
    reg = np.asarray(src["img"])
    colon = np.array(["_CL_" in str(v) for v in reg])
    regions = sorted(set(reg[colon]))
    hold = regions[fold % len(regions)]
    keep_src = colon & (reg != hold)
    core = np.array([str(s).rsplit("_", 1)[-1] == "R8C1"
                     for s in ang["img"].tolist()])
    lab_s = np.array([HARM_INTESTINE.get(str(v), "Other")
                      for v in src["cls"][src["y"]]])
    lab_t = R69._harmonise(ang["cls"][ang["y"]],
                           R69.HARM_NEW["EXT2_MIBITOF_Angelo"])
    pairs = R69.shared_pairs(list(src["markers"]), ang["markers"])
    if len(pairs) < 4:
        return []
    scols = [q[1] for q in pairs]
    acols = [q[2] for q in pairs]
    cnt_s = pd.Series(lab_s[keep_src]).value_counts()
    cnt_t = pd.Series(lab_t[core]).value_counts()
    classes = sorted(c for c in set(cnt_s.index) & set(cnt_t.index)
                     if c != "Other" and cnt_s[c] >= 30 and cnt_t[c] >= 30)
    if len(classes) < 3:
        return []
    ix = {c: i for i, c in enumerate(classes)}
    tr_all = np.nonzero(keep_src & np.array([v in ix for v in lab_s]))[0]
    pool = np.nonzero(core & np.array([v in ix for v in lab_t]))[0]
    if tr_all.size < 500 or pool.size < 300:
        return []
    seed = zlib.crc32(("r72e|%d|%s" % (fold, kind)).encode()) & 0xFFFFFFFF
    lab_code = np.array([ix.get(v, -1) for v in lab_s], dtype=int)
    tr = R61.R2.subsample_training(tr_all, lab_code, R61.MAX_TRAIN, seed)
    ytr = np.array([ix[v] for v in lab_s[tr]], dtype=int)
    yt = np.array([ix[v] for v in lab_t[pool]], dtype=int)
    Xt = src["X"][np.ix_(tr, scols)]
    nz = R61.DomainNormaliser("z_dataset").fit("src", Xt)
    Xd = ang["X"][np.ix_(pool, acols)]
    nz.fit("tgt", Xd)
    Ztr = nz.transform("src", Xt)
    Zt = nz.transform("tgt", Xd)
    clf = R61.fit_classifier(Ztr, ytr, kind=kind, seed=seed)
    score, err, _ = score_err(clf, Zt, yt)

    def refit(mask):
        c2 = R61.fit_classifier(np.vstack([Ztr, Zt[mask]]),
                                np.concatenate([ytr, yt[mask]]),
                                kind=kind, seed=seed + 7)
        s2, e2, _ = score_err(c2, Zt, yt)
        return s2, e2, None

    rows = []
    arms(rows, PART_E[0], PART_E[1], kind, fold, int(tr.size), int(pool.size),
         score, err, yt, seed + 13, refit, len(pairs), len(classes))
    for r in rows:
        r["n_train_src"] = int(tr.size)
    return rows


def job_b(spec):
    tag, pair, depot, cores, fold, kind = spec
    ang = R69.load_cohort("EXT2_MIBITOF_Angelo")
    core = np.array([s.rsplit("_", 1)[-1] in cores for s in ang["img"].tolist()])
    lab_new = R69._harmonise(ang["cls"][ang["y"]],
                             R69.HARM_NEW["EXT2_MIBITOF_Angelo"])
    d = R69.depot_cohort(depot)
    g, markers = d["g"], d["markers"]
    lab_dep = R69._harmonise(g["coarse_expert"].to_numpy(), R69.HARM_DEPOT)
    pairs = R69.shared_pairs(markers, ang["markers"])
    if len(pairs) < 4:
        return []
    dcols = [p[1] for p in pairs]
    acols = [p[2] for p in pairs]
    cnt_d = pd.Series(lab_dep).value_counts()
    cnt_a = pd.Series(lab_new[core]).value_counts()
    classes = sorted(c for c in set(cnt_d.index) & set(cnt_a.index)
                     if cnt_d[c] >= 30 and cnt_a[c] >= 30)
    if len(classes) < 3:
        return []
    ix = {c: i for i, c in enumerate(classes)}
    imgs = g["image"].astype(str).to_numpy()
    folds = R61.R2.fold_split(np.unique(imgs), N_FOLDS, seed=0)
    hold = set(folds[fold].tolist())
    tr = np.nonzero((~np.isin(imgs, list(hold))
                     & np.array([v in ix for v in lab_dep])))[0]
    pool = np.nonzero(core & np.array([v in ix for v in lab_new]))[0]
    if tr.size < 100 or pool.size < 300:
        return []
    seed = zlib.crc32(("r72b|%s|%s|%d|%s" % (tag, depot, fold, kind)).encode()) & 0xFFFFFFFF
    lab_code = np.array([ix.get(v, -1) for v in lab_dep], dtype=int)
    tr = R61.R2.subsample_training(tr, lab_code, R61.MAX_TRAIN, seed)
    ytr = np.array([ix[v] for v in lab_dep[tr]], dtype=int)
    yt = np.array([ix[v] for v in lab_new[pool]], dtype=int)
    Xt = d["X"][np.ix_(tr, dcols)]
    nz = R61.DomainNormaliser("z_dataset").fit("src", Xt)
    Xd = ang["X"][np.ix_(pool, acols)]
    nz.fit("tgt", Xd)
    Ztr = nz.transform("src", Xt)
    Zt = nz.transform("tgt", Xd)
    clf = R61.fit_classifier(Ztr, ytr, kind=kind, seed=seed)
    score, err, _ = score_err(clf, Zt, yt)

    def refit(mask):
        c2 = R61.fit_classifier(np.vstack([Ztr, Zt[mask]]),
                                np.concatenate([ytr, yt[mask]]),
                                kind=kind, seed=seed + 7)
        s2, e2, _ = score_err(c2, Zt, yt)
        return s2, e2, None

    rows = []
    arms(rows, tag, pair, kind, fold, int(tr.size), int(pool.size), score,
         err, yt, seed + 13, refit, len(pairs), len(classes))
    for r in rows:
        r["n_train_src"] = int(tr.size)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", default="abcde")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        t0 = time.time()
        rows = []
        if "a" in args.part:
            R61._init(list(NEW_PAIR), NEW_PAIR)
            log("part A shared markers", len(R61.COHORT["_shared_names"]),
                "classes", len(R61.VOCAB),
                "pool", {d: int(R61.COHORT[d]["g"].shape[0]) for d in NEW_PAIR})
            specs = [(s, t, f, c) for (s, t) in (NEW_PAIR, NEW_PAIR[::-1])
                     for f in range(N_FOLDS) for c in CLFS]
            with mp.Pool(args.jobs, initializer=R61._init,
                         initargs=(list(NEW_PAIR), NEW_PAIR)) as pool:
                for i, res in enumerate(pool.imap(job_a, specs, chunksize=1), 1):
                    rows.extend(res)
                    if i % 5 == 0 or i == len(specs):
                        log("  A %d/%d  rows=%d  %.0fs" % (i, len(specs), len(rows),
                                                           time.time() - t0))
            log("part A done", len(rows))
        for key, part in PARTS_B:
            if key not in args.part:
                continue
            specs = [part + (f, c) for f in range(N_FOLDS) for c in CLFS]
            with mp.Pool(args.jobs) as pool:
                for i, res in enumerate(pool.imap(job_b, specs, chunksize=1), 1):
                    rows.extend(res)
                    log("  %s %d/%d  rows=%d" % (part[0], i, len(specs), len(rows)))
            log("part %s done" % part[0], len(rows))
        if "d" in args.part:
            specs = [(f, c) for f in range(N_FOLDS) for c in CLFS]
            with mp.Pool(args.jobs) as pool:
                for i, res in enumerate(pool.imap(job_d, specs, chunksize=1), 1):
                    rows.extend(res)
                    log("  D %d/%d  rows=%d" % (i, len(specs), len(rows)))
            log("part D done", len(rows))
        if "e" in args.part:
            specs = [(f, c) for f in range(N_FOLDS_E) for c in CLFS]
            with mp.Pool(args.jobs) as pool:
                for i, res in enumerate(pool.imap(job_e, specs, chunksize=1), 1):
                    rows.extend(res)
                    log("  E %d/%d  rows=%d" % (i, len(specs), len(rows)))
            log("part E done", len(rows))
        new = pd.DataFrame(rows)
        if os.path.exists(CSV):
            keep = pd.read_csv(CSV)
            keep = keep[~keep.part.isin(new.part.unique())]
            new = pd.concat([keep, new], ignore_index=True)
        new.to_csv(CSV, index=False)
        log("wrote", CSV, "in %.1f s" % (time.time() - t0))
    write_report()


def write_report():
    SEC = 20.0
    df = pd.read_csv(CSV)
    A = []
    A.append("r72 -- repairing a cross-panel deployment, and the same-tissue control")
    A.append("=" * 78)
    A.append("")
    nmA = df[df.part == "A_cross_tissue_panel"].n_markers
    nmB = df[df.part == "B_same_tissue_panel"].n_markers
    A.append("Part A  A_cross_tissue_panel: %s <-> %s, %s shared canonical markers."
             % (NEW_PAIR[0], NEW_PAIR[1],
                ("%d" % int(nmA.max())) if len(nmA) else "not in this csv"))
    A.append("        Tissue and panel change together; same platform (IMC).")
    A.append("        The frozen arm is r71's arm, so the two reports chain.")
    A.append("Part B  B_same_tissue_panel: colon adenocarcinoma core R8C1 of the")
    A.append("        MIBI-TOF tissue microarray (Angelo laboratory, 15 markers,")
    A.append("        8,241 cells, tissue_data.csv) against the depot colorectal")
    A.append("        cohort %s, on the %s markers the two panels share."
             % (DEPOT_COLON, ("%d" % int(nmB.max())) if len(nmB) else "n"))
    A.append("        Same tissue, different panel, different laboratory, different")
    A.append("        platform (MIBI vs IMC).")
    nmC = df[df.part == "C_same_tissue_panel_breast"].n_markers
    A.append("Part C  C_same_tissue_panel_breast: breast ductal carcinoma cores %s"
             % "+".join(BREAST_CORES))
    A.append("        (%s cells) of the same microarray against the depot breast"
             % ("%d" % int(df[df.part == "C_same_tissue_panel_breast"].n_pool.max())
                if len(nmC) else "n"))
    A.append("        cohort %s, on the %s markers the two panels share.  The"
             % (DEPOT_BREAST, ("%d" % int(nmC.max())) if len(nmC) else "n"))
    A.append("        second same-tissue control, on a different organ.")
    nmD = df[df.part == "D_cross_platform_tonsil"].n_markers
    A.append("Part D  D_cross_platform_tonsil: tonsil cores %s of the same"
             % "+".join(TONSIL_CORES))
    A.append("        microarray against the STELLAR tonsil cohort, on the %s"
             % (("%d" % int(nmD.max())) if len(nmD) else "n"))
    A.append("        markers the two panels share.  Same tissue, CODEX against")
    A.append("        MIBI-TOF, different laboratory; the reference labels are")
    A.append("        coarsened onto one vocabulary.")
    A.append("")
    nmE = df[df.part == "E_cross_platform_colon"].n_markers
    A.append("Part E  E_cross_platform_colon: STELLAR intestine -> MIBI-TOF colon core,")
    A.append("        on the %s markers the two panels share.  Same MIBI-TOF colon"
             % (("%d" % int(nmE.max())) if len(nmE) else "n"))
    A.append("        core as Part B, but an independent STELLAR source and four")
    A.append("        whole-region source folds.  This is a source-side replication")
    A.append("        of the colon repair, not a new target.")
    A.append("")
    A.append("Protocol: 5 image folds of the source, %d random draws of the deployment"
             % N_SPLITS)
    A.append("pool, delta = %.2f, classifiers %s, budgets %s."
             % (DELTA, ",".join(CLFS), ",".join("%.2f" % a for a in ALPHAS)))
    A.append("Review time 20 s per cell, the paper's anchor.")
    A.append("")
    A.append("Arms.  tgt_k=model frozen, calibrator on all k labels (r71's arm).")
    A.append("tgt_k_half=model frozen, calibrator on k/2 only.  adapt_k=model refitted")
    A.append("on the source training set plus k/2 target labels, calibrator on the")
    A.append("other k/2.  Net %% = (accepted - labels read) / pool, the paper's")
    A.append("net-of-calibration accounting, so a rung can be negative.")
    A.append("")
    for part in ("A_cross_tissue_panel", "B_same_tissue_panel",
                 "C_same_tissue_panel_breast", "D_cross_platform_tonsil",
                 "E_cross_platform_colon"):
        d = df[df.part == part]
        if not len(d):
            continue
        for pair in sorted(d.pair.unique()):
            dp = d[d.pair == pair]
            A.append("=" * 78)
            A.append("== %s   %s" % (part, pair))
            nsrc = (int(dp.n_train_src.min())
                    if "n_train_src" in dp.columns and dp.n_train_src.notna().any()
                    else int(dp.n_train.max()))
            A.append("   source training cells %d, deployment pool %d, classes %d"
                     % (nsrc, int(dp.n_pool.max()), int(dp.n_classes.max())))
            A.append("")
            A.append("   %-16s %8s %8s %8s %8s %8s %8s"
                     % ("arm", "accept", "risk", "cover", "cp-ok", "oracle", "net%"))
            for alpha in ALPHAS:
                A.append("   --- budget alpha = %.2f" % alpha)
                for k in KS:
                    for arm in ("tgt_k%d" % k, "tgt_k%d_half" % k, "adapt_k%d" % k):
                        r = dp[(np.isclose(dp.alpha, alpha)) & (dp.arm == arm)]
                        if not len(r):
                            continue
                        net = ((r.n_accepted - r.n_labels) / r.n_pool).mean() * 100.0
                        A.append("   %-16s %8.3f %8.3f %8.3f %8.3f %8.3f %8.1f"
                                 % (arm, r.accept_rate.mean(),
                                    r.selective_risk.fillna(0).mean(), r.covered.mean(),
                                    r.cp_valid.mean(), r.oracle_accept.fillna(0).mean(),
                                    net))
            A.append("")
            A.append("   where the k labels go (mean over folds/classifiers):")
            A.append("   %6s %8s %8s %8s %8s %8s %8s %8s"
                     % ("k", "froz.acc", "froz.net", "refit.acc", "refit.net",
                        "oracle", "refit/fro", "hours"))
            for alpha in ALPHAS:
                A.append("   --- alpha = %.2f" % alpha)
                for k in KS:
                    f = dp[(np.isclose(dp.alpha, alpha)) & (dp.arm == "tgt_k%d" % k)]
                    a = dp[(np.isclose(dp.alpha, alpha)) & (dp.arm == "adapt_k%d" % k)]
                    if not (len(f) and len(a)):
                        continue
                    fn = ((f.n_accepted - f.n_labels) / f.n_pool).mean() * 100.0
                    an = ((a.n_accepted - a.n_labels) / a.n_pool).mean() * 100.0
                    rr = ("%8.2f" % (a.accept_rate.mean() / f.accept_rate.mean())
                          if f.accept_rate.mean() >= 0.005 else "%8s" % "n/a")
                    hv = ((a.n_accepted - a.n_labels).mean() * SEC / 3600.0) / 1000.0
                    A.append("   %6d %8.3f %8.1f %8.3f %8.1f %8.3f %s %8.1f"
                             % (k, f.accept_rate.mean(), fn, a.accept_rate.mean(), an,
                                f.oracle_accept.fillna(0).mean(), rr, hv))
            A.append("")
            A.append("   coverage of the adapted arm across every rung and budget:")
            a = dp[dp.arm.str.startswith("adapt_")]
            A.append("     units %d, realised risk within budget %.3f,"
                     % (len(a), a.covered.mean()))
            A.append("     Clopper-Pearson upper limit within budget %.3f, empty %.3f"
                     % (a.cp_valid.mean(), (a.n_accepted == 0).mean()))
            A.append("")
    A.append("=" * 78)
    A.append("Legend.  accept = share of the deployment batch annotated.  risk =")
    A.append("realised error among accepted cells.  cover = share of units whose")
    A.append("realised risk was within the budget.  cp-ok = share whose one-sided")
    A.append("95%% Clopper-Pearson upper limit on that risk is within the budget.")
    A.append("oracle = what a target-labelled cut would have accepted, unattainable.")
    A.append("net%% = share of the batch removed net of the labels read; hours is")
    A.append("net expert hours per 1,000 cells at 20 s per cell.")
    txt = "\n".join(A) + "\n"
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
