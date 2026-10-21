"""R14 -- what does a cross-tissue certificate actually cost?

The cross-tissue family of r02/r03 calibrates on the *source* cohort and deploys
on the *target*, and its certificate fails: coverage 0.014 among the evaluations
that commit to a label.  Two things could be responsible.  Either the
distributions really have moved -- in which case no amount of source data helps
and the only cure is labels from the target tissue -- or the source calibration
pool is simply far too small, since the design holds out two source folds per
fold and uses only one of them, leaving about 300 cells.

This script separates the two, on the same models and the same splits as r02:

* ``src``        calibrate on the source fold r02 uses, accept on the target.
* ``src_big``    calibrate on *both* source folds held out of training.  Those
                 images never entered the model, so this doubles the calibration
                 pool at no cost and tests the sample-size explanation.
* ``tgt_k``      calibrate on k cells drawn at random from the target, accept on
                 the rest.  Exchangeable by construction, so the certificate is
                 valid; the question is how large k has to be.
* ``w_big``      weighted controller on the source pool, with the density ratio
                 estimated from the features and the denominator read off the
                 unlabelled target scores.
* ``union_k``    the union of ``tgt_k`` and ``w_big``: accept at the *larger* of
                 the two thresholds.  Both certificates hold simultaneously at
                 level delta/2, and the union of two accepted sets that each meet
                 the budget also meets it, so source data can be added to a small
                 target-labelled set without invalidating anything.
* ``naive`` / ``oracle``  the uncalibrated threshold and the test-peeking one.

Output: experiments/r14_targetcal.csv + experiments/r14_targetcal_summary.csv
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from errctl import conformal as C, paths
from errctl.classifiers import fit_classifier
from errctl.preprocess import DomainNormaliser
from errctl.weights import density_ratio, effective_sample_size
import r02_run_experiments as R2


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)), flush=True)


def score_of(probs):
    return 1.0 - np.asarray(probs, dtype=np.float64).max(1)


def err_of(probs, y):
    return (np.asarray(probs).argmax(1) != np.asarray(y)).astype(float)


def record(rows, unit, clf, alpha, k, rep, arm, s_eval, e_eval, accepted, extra=None):
    accepted = np.asarray(accepted, dtype=bool)
    n = int(accepted.sum())
    risk = float(e_eval[accepted].mean()) if n else np.nan
    row = {"unit": unit, "clf": clf, "alpha": alpha, "k": int(k), "repeat": int(rep),
           "arm": arm, "n_eval": int(e_eval.size), "n_accepted": n,
           "accept_rate": n / max(e_eval.size, 1), "selective_risk": risk,
           "empty": int(n == 0),
           "covered": int((not np.isfinite(risk)) or risk <= alpha),
           "nonvac_covered": int(n > 0 and risk <= alpha)}
    if extra:
        row.update(extra)
    rows.append(row)


def unit_job(job):
    src, tgt, f, clfs, alphas, ks, reps, seed = job
    rows = []
    gs, ms, Xs = COHORT[src]
    gt, mt, Xt = COHORT[tgt]
    panel = R2.shared_panel(ms, mt)
    src_cols = [p[1] for p in panel]
    tgt_cols = [p[2] for p in panel]
    Xs_p = gs[src_cols].to_numpy(dtype=np.float64)
    Xt_p = gt[tgt_cols].to_numpy(dtype=np.float64)

    counts_t = gt["coarse_expert"].value_counts()
    counts_s = gs["coarse_expert"].value_counts()
    shared = sorted(c for c in counts_t.index
                    if counts_t[c] >= R2.MIN_CLASS_TEST and counts_s.get(c, 0) >= 30)
    if len(shared) < 3:
        return rows, "%s->%s: too few shared classes" % (src, tgt)

    sm = gs["coarse_expert"].isin(shared).to_numpy()
    tm = gt["coarse_expert"].isin(shared).to_numpy()
    gs2, Xs2 = gs[sm].reset_index(drop=True), Xs_p[sm]
    gt2, Xt2 = gt[tm].reset_index(drop=True), Xt_p[tm]
    ys = gs2["coarse_expert"].astype(str).to_numpy()
    yt = gt2["coarse_expert"].astype(str).to_numpy()

    folds = R2.fold_split(gs2["image"].unique(), R2.N_FOLDS, seed=seed + 7)
    test_imgs, cal_imgs = set(folds[f]), set(folds[(f + 1) % R2.N_FOLDS])
    tr = ~gs2["image"].isin(test_imgs | cal_imgs).to_numpy()
    ca = gs2["image"].isin(cal_imgs).to_numpy()
    ex = gs2["image"].isin(test_imgs).to_numpy()
    tr_idx = R2.subsample_training(np.nonzero(tr)[0], ys, R2.MAX_TRAIN, seed=seed + f)
    ca_idx, ex_idx = np.nonzero(ca)[0], np.nonzero(ex)[0]
    if len(ca_idx) < 50:
        return rows, "%s->%s fold%d: calibration fold too small" % (src, tgt, f)

    nz = DomainNormaliser("z_dataset")
    nz.fit("src", Xs2[tr_idx])
    nz.fit("tgt", Xt2)
    Xtr = nz.transform("src", Xs2[tr_idx])
    Xca = nz.transform("src", Xs2[ca_idx])
    Xex = nz.transform("src", Xs2[ex_idx])
    Xte = nz.transform("tgt", Xt2)

    keep_cls = pd.Series(ys[tr_idx]).value_counts()
    classes = sorted(c for c in keep_cls.index if keep_cls[c] >= R2.MIN_CLASS_TRAIN)
    tr_keep = np.isin(ys[tr_idx], classes)
    idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([idx[c] for c in ys[tr_idx][tr_keep]], dtype=int)
    y_ca = np.array([idx.get(c, -1) for c in ys[ca_idx]], dtype=int)
    y_ex = np.array([idx.get(c, -1) for c in ys[ex_idx]], dtype=int)
    y_te = np.array([idx.get(c, -1) for c in yt], dtype=int)

    unit = "transfer:%s->%s:fold%d" % (src, tgt, f)
    tag = zlib.crc32(unit.encode()) & 0xFFFFFFFF
    if CFG.get("adapt"):
        aks = CFG["adapt_ks"]
        for clf_kind in clfs:
            clf = fit_classifier(Xtr[tr_keep], y_train, kind=clf_kind, seed=seed)
            p_te0 = clf.predict_proba(Xte)
            s_te0, e_te0 = score_of(p_te0), err_of(p_te0, y_te)
            for rep in range(reps):
                rng = np.random.default_rng(100000 * rep + tag + 977)
                perm = rng.permutation(s_te0.size)
                for k in aks:
                    pool, ev = perm[:k], perm[k:]
                    n_tr = max(5, k // 2)
                    idx_tr, idx_cal = pool[:n_tr], pool[n_tr:]
                    if idx_cal.size < 40 or ev.size < 200:
                        continue
                    X_aug = np.vstack([Xtr[tr_keep], Xte[idx_tr]])
                    y_aug = np.concatenate([y_train, y_te[idx_tr]])
                    if set(np.unique(y_aug)) != set(range(len(classes))):
                        continue
                    clf2 = fit_classifier(X_aug, y_aug, kind=clf_kind, seed=seed)
                    p_cal = clf2.predict_proba(Xte[idx_cal])
                    p_ev = clf2.predict_proba(Xte[ev])
                    s_cal, e_cal = score_of(p_cal), err_of(p_cal, y_te[idx_cal])
                    s_ev, e_ev = score_of(p_ev), err_of(p_ev, y_te[ev])
                    for alpha in alphas:
                        ctl = C.SelectiveRiskController(alpha, 0.10).fit(s_cal, e_cal)
                        record(rows, unit, clf_kind, alpha, k, rep, "adapt_k",
                               s_ev, e_ev, ctl.accept(s_ev),
                               {"lambda": ctl.lambda_hat, "n_cal": int(idx_cal.size),
                                "ess": np.nan})
                        ctl_s = C.SelectiveRiskController(alpha, 0.10).fit(s_te0[idx_cal],
                                                                          e_te0[idx_cal])
                        record(rows, unit, clf_kind, alpha, k, rep, "tgt_k_same",
                               s_ev, e_ev, ctl_s.accept(s_ev),
                               {"lambda": ctl_s.lambda_hat, "n_cal": int(idx_cal.size),
                                "ess": np.nan})
                        for arm, (sa, ea) in (("oracle_adapt", (s_ev, e_ev)),
                                              ("oracle_base", (s_te0[ev], e_te0[ev]))):
                            order = np.argsort(sa, kind="mergesort")
                            ss_, ee_ = sa[order], ea[order]
                            cum = np.cumsum(ee_)
                            kk = np.arange(1, ss_.size + 1)
                            ok = (cum / kk <= alpha) & (kk >= 25)
                            lam = float(ss_[np.nonzero(ok)[0][-1]]) if ok.any() else -np.inf
                            record(rows, unit, clf_kind, alpha, k, rep, arm,
                                   s_ev, e_ev, s_ev <= lam,
                                   {"lambda": lam, "n_cal": np.nan, "ess": np.nan})
        return rows, "%s (adapt) n_te=%d" % (unit, len(gt2))

    for clf_kind in clfs:
        clf = fit_classifier(Xtr[tr_keep], y_train, kind=clf_kind, seed=seed)
        p_ca = clf.predict_proba(Xca)
        p_ex = clf.predict_proba(Xex)
        p_te = clf.predict_proba(Xte)
        s_ca, e_ca = score_of(p_ca), err_of(p_ca, y_ca)
        s_ex, e_ex = score_of(p_ex), err_of(p_ex, y_ex)
        s_te, e_te = score_of(p_te), err_of(p_te, y_te)

        s_src, e_src = np.concatenate([s_ca, s_ex]), np.concatenate([e_ca, e_ex])
        X_src = np.vstack([Xca, Xex])
        w = density_ratio(X_src, Xte, clip=20.0, seed=0, cv=3)
        ess = effective_sample_size(w)

        for rep in range(reps):
            rng = np.random.default_rng(100000 * rep + tag)
            perm = rng.permutation(s_te.size)
            lam_cache = {}
            for k in ks:
                cal_t = perm[:k] if k > 0 else np.array([], dtype=int)
                ev = perm[k:]
                s_ev, e_ev = s_te[ev], e_te[ev]
                for alpha in alphas:
                    d = 0.10
                    # ---- source-only, the two calibration pools
                    for arm, (ss, ee) in (("src", (s_ca, e_ca)),
                                          ("src_big", (s_src, e_src))):
                        ctl = C.SelectiveRiskController(alpha, d).fit(ss, ee)
                        record(rows, unit, clf_kind, alpha, k, rep, arm,
                               s_ev, e_ev, ctl.accept(s_ev),
                               {"lambda": ctl.lambda_hat, "n_cal": int(ss.size),
                                "ess": np.nan})
                    # ---- target-only, k labelled cells
                    if k > 0:
                        ctl = C.SelectiveRiskController(alpha, d).fit(s_te[cal_t], e_te[cal_t])
                        record(rows, unit, clf_kind, alpha, k, rep, "tgt_k",
                               s_ev, e_ev, ctl.accept(s_ev),
                               {"lambda": ctl.lambda_hat, "n_cal": int(k), "ess": np.nan})
                    # ---- weighted source certificate, target-score denominator
                    ctl_w = C.WeightedSelectiveRiskController(alpha, d)
                    ctl_w.fit(s_src, e_src, w, target_scores=s_te)
                    acc_w = ctl_w.accept(s_ev)
                    record(rows, unit, clf_kind, alpha, k, rep, "w_big",
                           s_ev, e_ev, acc_w,
                           {"lambda": ctl_w.lambda_hat, "n_cal": int(s_src.size),
                            "ess": ess})
                    # ---- score-conditional certificate: no weights, but the
                    #      error curve given the score must transfer
                    ctl_sc = C.ScoreConditionalController(alpha, d)
                    ctl_sc.fit(s_src, e_src, s_te)
                    record(rows, unit, clf_kind, alpha, k, rep, "sc",
                           s_ev, e_ev, ctl_sc.accept(s_ev),
                           {"lambda": ctl_sc.lambda_hat, "n_cal": int(s_src.size),
                            "ess": np.nan})
                    # ---- union of the target-labelled and weighted certificates
                    if k > 0:
                        ctl_t = C.SelectiveRiskController(alpha, d).fit(s_te[cal_t], e_te[cal_t])
                        lam_u = max(ctl_w.lambda_hat, ctl_t.lambda_hat)
                        record(rows, unit, clf_kind, alpha, k, rep, "union_k",
                               s_ev, e_ev, s_ev <= lam_u,
                               {"lambda": lam_u, "n_cal": int(s_src.size + k), "ess": ess})
                    # ---- uncalibrated and test-peeking references
                    lam_n = C.naive_threshold(s_ca, e_ca, alpha)
                    record(rows, unit, clf_kind, alpha, k, rep, "naive",
                           s_ev, e_ev, s_ev <= lam_n,
                           {"lambda": lam_n, "n_cal": int(s_ca.size), "ess": np.nan})
                    order = np.argsort(s_ev, kind="mergesort")
                    ss_, ee_ = s_ev[order], e_ev[order]
                    cum = np.cumsum(ee_)
                    kk = np.arange(1, ss_.size + 1)
                    ok = (cum / kk <= alpha) & (kk >= 25)
                    lam_o = float(ss_[np.nonzero(ok)[0][-1]]) if ok.any() else -np.inf
                    record(rows, unit, clf_kind, alpha, k, rep, "oracle",
                           s_ev, e_ev, s_ev <= lam_o, {"lambda": lam_o, "n_cal": np.nan,
                                                       "ess": np.nan})
    return rows, "%s  n_ca=%d n_ex=%d n_te=%d" % (unit, len(ca_idx), len(ex_idx), len(gt2))


COHORT = {}
CFG = {}


def _init(cohort, cfg):
    for k, v in cohort.items():
        COHORT[k] = v
    CFG.clear()
    CFG.update(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--ks", default="0,25,50,100,200,500,1000,2000")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="r14_targetcal")
    ap.add_argument("--adapt", action="store_true",
                    help="run only the model-adaptation arm")
    ap.add_argument("--adapt-ks", default="100,200,500,1000,2000")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    clfs = [c for c in args.clfs.split(",") if c]

    cohorts = set()
    for s, t in R2.TRANSFER_PAIRS:
        cohorts.add(s)
        cohorts.add(t)
    cohort = {}
    for ds in sorted(cohorts):
        cohort[ds] = R2.prepare_cohort(ds)
        log("loaded %s: %d cells / %d markers" % (ds, len(cohort[ds][0]), len(cohort[ds][1])))

    pairs = R2.TRANSFER_PAIRS
    if args.only:
        pairs = [p for p in pairs if ("%s->%s" % p) == args.only]
    ks_use = ks
    if args.adapt:
        ks_use = [int(x) for x in args.adapt_ks.split(",")]
    cfg = {"adapt": bool(args.adapt), "adapt_ks": ks_use}
    jobs = [(s, t, f, clfs, alphas, ks_use, args.reps, args.seed)
            for s, t in pairs for f in range(R2.N_FOLDS)]
    log("%d unit-folds x %d classifiers x %d alphas x %d calib sizes x %d repeats"
        % (len(jobs), len(clfs), len(alphas), len(ks), args.reps))

    rows = []
    t0 = time.time()
    if args.jobs > 1:
        with mp.Pool(args.jobs, initializer=_init, initargs=(cohort, cfg)) as pool:
            for i, (rws, msg) in enumerate(pool.imap(unit_job, jobs, chunksize=1), 1):
                rows.extend(rws)
                if i % 5 == 0 or i == len(jobs):
                    log("  %d/%d  %s" % (i, len(jobs), msg))
    else:
        _init(cohort, cfg)
        for i, job in enumerate(jobs, 1):
            rws, msg = unit_job(job)
            rows.extend(rws)
            if i % 5 == 0 or i == len(jobs):
                log("  %d/%d  %s" % (i, len(jobs), msg))

    df = pd.DataFrame(rows)
    p = os.path.join(paths.RESULTS, args.out + ".csv")
    df.to_csv(p, index=False)
    log("wrote %s (%d rows, %.1fs)" % (p, len(df), time.time() - t0))

    summ = (df.groupby(["arm", "alpha", "k"])
              .agg(n_eval=("n_accepted", "size"),
                   mean_accept=("accept_rate", "mean"),
                   coverage_all=("covered", "mean"),
                   nonvac_frac=("empty", lambda x: 1.0 - float(x.mean())),
                   mean_risk=("selective_risk", "mean"))
              .reset_index())
    # coverage restricted to the evaluations that committed to a label
    sub = df[df["empty"] == 0]
    nv = (sub.groupby(["arm", "alpha", "k"])["covered"].mean()
             .rename("coverage_nonvac").reset_index())
    summ = summ.merge(nv, on=["arm", "alpha", "k"], how="left")
    sp = os.path.join(paths.RESULTS, args.out + "_summary.csv")
    summ.to_csv(sp, index=False)
    log("wrote %s" % sp)
    for arm in ("src", "src_big", "tgt_k", "w_big", "union_k", "naive", "oracle",
                "adapt_k", "tgt_k_same", "oracle_adapt", "oracle_base"):
        s = summ[(summ.arm == arm) & np.isclose(summ.alpha, 0.10)]
        if s.empty:
            continue
        log("\n[%s] alpha=0.10" % arm)
        log(s[["k", "mean_accept", "nonvac_frac", "coverage_all",
               "coverage_nonvac", "mean_risk"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
