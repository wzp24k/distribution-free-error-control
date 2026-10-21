"""R71 -- does a real cross-panel deployment have to fail?  A price list.

Why this script exists
----------------------
`scripts/r61_shared_panel.py` puts two real antibody panels on the channels they
share and runs the certificate in three ways: inside a cohort on its own panel,
inside a cohort on the shared channels, and across the two cohorts on the shared
channels.  The cross-cohort arm is the one a laboratory actually faces, and it
is the arm the manuscript reads as nearly silent.  That reading leaves one
question open, and this script answers it with a ladder instead of a verdict:
under a real change of antibody panel, does the certificate ever fail, and what
does acceptance cost as target labels are spent on it.

Arms, all on the identical certificate, folds and classifiers:
  src_cal      calibrate on the source cohort, deploy on the target.  The
               guarantee does not cover this arm; it is the arm the paper
               reports as invalid.
  tgt_k<K>     calibrate on K cells drawn at random from the deployment cohort,
               deploy on the remaining cells.  Exchangeable by construction, so
               this is the arm the guarantee does cover, and the ladder is the
               price list.
  oracle<K>    the largest target-calibration cut that meets the budget when the
               deployment labels pick it.  Not attainable; the ratio of
               acceptance to this ceiling is the diagnostic.

Outputs
  experiments/r71_crosspanel_price.csv
  work/r71_report.txt
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

import r61_shared_panel as R61  # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)
KS = (0, 100, 250, 500, 1000, 2000, 4000, 8000)
DELTA = 0.10
CLFS = ("logreg", "rf", "hgb")
N_FOLDS = 5
N_SPLITS = 3

CSV = os.path.join(R61.paths.RESULTS, "r71_crosspanel_price.csv")
REPORT = os.path.join(R61.paths.WORK, "r71_report.txt")
PAIR = ("CRC_Feb2623", "GVHD_Feb1323")


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


def job(spec):
    src, tgt, fold, kind, rep = spec
    ds_, dt_ = R61.COHORT[src], R61.COHORT[tgt]
    classes = R61.VOCAB
    cols = ds_["shared_cols"]
    tr_idx = np.nonzero(ds_["fold"] != fold)[0]
    t_lab = dt_["g"]["coarse_expert"].to_numpy()
    in_v = np.array([v in set(classes) for v in t_lab])
    pool_all = np.nonzero(in_v)[0]
    seed = zlib.crc32(("r71|%s|%s|%d|%s|%d"
                       % (src, tgt, fold, kind, rep)).encode()) & 0xFFFFFFFF
    out = R61.fit_predict(ds_["g"], ds_["X"], cols, classes, tr_idx,
                          dt_["g"], dt_["X"], pool_all, kind, seed,
                          domain="tgt", cols_de=dt_["shared_cols"])
    rows = []
    n = pool_all.size
    rng = np.random.default_rng(seed + 23)
    pick = rng.permutation(n)
    score, err = out["score"][pick], out["err"][pick]
    pname = "%s->%s" % (src, tgt)

    # the source-calibrated arm: the one the guarantee does not cover
    s_lab = ds_["g"]["coarse_expert"].to_numpy()
    cal_idx = np.nonzero(ds_["fold"] == fold)[0]
    cal_idx = cal_idx[np.array([v in set(classes) for v in s_lab[cal_idx]])]
    if cal_idx.size >= 40:
        keep = np.array([v in set(classes) for v in s_lab[tr_idx]])
        ti = tr_idx[keep]
        ix = {c: i for i, c in enumerate(classes)}
        ytr = np.array([ix[v] for v in s_lab[ti]], dtype=int)
        Xt = ds_["X"][np.ix_(ti, cols)]
        nz = R61.DomainNormaliser("z_dataset").fit("src", Xt)
        clf = R61.fit_classifier(nz.transform("src", Xt), ytr, kind=kind, seed=seed)
        p_cal = clf.predict_proba(
            nz.transform("src", ds_["X"][np.ix_(cal_idx, cols)])).astype(np.float64)
        s_cal = 1.0 - p_cal.max(1)
        e_cal = (np.array([classes[i] for i in p_cal.argmax(1)], dtype=object)
                 != s_lab[cal_idx]).astype(float)
        for alpha in ALPHAS:
            ctl = R61.C.SelectiveRiskController(alpha, DELTA).fit(s_cal, e_cal)
            acc = ctl.accept(score)
            rows.append(dict(arm="src_cal", pair=pname, clf=kind, alpha=alpha,
                             rep=rep, k=-1, n_pool=n, n_cal=int(cal_idx.size),
                             n_accepted=int(acc.sum()),
                             accept_rate=float(acc.mean()),
                             selective_risk=float(err[acc].mean()) if acc.any() else np.nan,
                             covered=int((not acc.any()) or err[acc].mean() <= alpha),
                             cp_valid=int(R61.cp_upper(err[acc]) <= alpha
                                          if acc.any() else True),
                             ratio=np.nan, oracle_accept=np.nan))
    for k in KS:
        if k >= n - 25:
            continue
        ctl_by_a = {}
        for alpha in ALPHAS:
            ctl = R61.C.SelectiveRiskController(alpha, DELTA)
            ctl.fit(score[:k], err[:k])
            ctl_by_a[alpha] = ctl
        for alpha in ALPHAS:
            ctl = ctl_by_a[alpha]
            acc = ctl.accept(score[k:])
            lam = oracle_lambda(score[k:], err[k:], alpha)
            o_acc = score[k:] <= lam
            rows.append(dict(
                arm="tgt_k%d" % k, pair=pname, clf=kind, alpha=alpha, rep=rep,
                k=k, n_pool=n - k, n_cal=k, n_accepted=int(acc.sum()),
                accept_rate=float(acc.mean()),
                selective_risk=float(err[k:][acc].mean()) if acc.any() else np.nan,
                covered=int((not acc.any()) or err[k:][acc].mean() <= alpha),
                cp_valid=int(R61.cp_upper(err[k:][acc]) <= alpha
                             if acc.any() else True),
                ratio=float(acc.mean() / o_acc.mean()) if o_acc.any() else np.nan,
                oracle_accept=float(o_acc.mean())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        t0 = time.time()
        R61._init(list(PAIR), PAIR)
        log("shared markers", len(R61.COHORT["_shared_names"]),
            "classes", len(R61.VOCAB))
        specs = [(s, t, f, c, r)
                 for (s, t) in (PAIR, PAIR[::-1])
                 for f in range(N_FOLDS) for c in CLFS
                 for r in range(N_SPLITS)]
        rows = []
        with mp.Pool(args.jobs, initializer=R61._init,
                     initargs=(list(PAIR), PAIR)) as pool:
            for i, res in enumerate(pool.imap(job, specs, chunksize=1), 1):
                rows.extend(res)
                if i % 10 == 0:
                    log("  %d/%d" % (i, len(specs)))
        pd.DataFrame(rows).to_csv(CSV, index=False)
        log("wrote", CSV, "in %.1f s" % (time.time() - t0))
    else:
        R61._init(list(PAIR), PAIR)
    write_report()


def write_report():
    df = pd.read_csv(CSV)
    A = []
    A.append("r71 -- a real change of antibody panel, priced in target labels")
    A.append("=" * 78)
    A.append("")
    A.append("Pair: %s <-> %s, the %d canonical markers the two panels share."
             % (PAIR[0], PAIR[1], len(R61.COHORT["_shared_names"])))
    A.append("Target labels are drawn at random from the deployment cohort and")
    A.append("the certificate is evaluated on the cells that remain, so the arm")
    A.append("is exchangeable by construction and the guarantee covers it.")
    A.append("")
    d = df[df["arm"] != "src_cal"]
    for alpha in ALPHAS:
        A.append("-- error budget alpha = %.2f" % alpha)
        A.append("  %-10s %8s %8s %8s %8s %7s"
                 % ("arm", "accept", "risk", "cover", "cp-valid", "ratio"))
        for k in KS:
            r = d[(d.alpha == alpha) & (d.k == k)]
            if not len(r):
                continue
            A.append("  %-10s %8.3f %8.3f %8.3f %8.3f %7.2f"
                     % ("tgt_k%d" % k, r.accept_rate.mean(),
                        r.selective_risk.fillna(0).mean(), r.covered.mean(),
                        r.cp_valid.mean(), r.ratio.mean()))
        A.append("")
    A.append("-- the source-calibrated arm, which the guarantee does not cover")
    s = df[df.arm == "src_cal"]
    for alpha in ALPHAS:
        r = s[s.alpha == alpha]
        A.append("  alpha %.2f  accept %.3f  risk %.3f  cover %.3f"
                 % (alpha, r.accept_rate.mean(),
                    r.selective_risk.fillna(0).mean(), r.covered.mean()))
    A.append("")
    A.append("Per direction, at alpha = 0.10:")
    d10 = d[(d.alpha == 0.10) & (d.pair == "CRC_Feb2623->GVHD_Feb1323")]
    for k in KS:
        r = d10[d10.k == k]
        if len(r):
            A.append("  CRC->GVHD k=%-5d accept %.3f  cover %.3f  ratio %.2f"
                     % (k, r.accept_rate.mean(), r.covered.mean(), r.ratio.mean()))
    txt = "\n".join(A) + "\n"
    open(REPORT, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
