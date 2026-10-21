"""R7 -- ablations around the controller.

Five knobs, each swept on the stored predictions so that the classifier is held
fixed and only the calibration layer changes:

* ``n_cal``        calibration-set size: how much a guarantee costs in labels.
* ``grid``         number of candidate thresholds, i.e. the Bonferroni price.
* ``delta``        the failure probability the user is willing to spend.
* ``min_accept``   the smallest accepted set the controller may nominate.
* ``score``        the confidence score fed to the controller (1 - max prob,
                   entropy, or the margin between the top two classes).

Output: experiments/r07_ablations.csv (tidy: ablation, family, clf, alpha,
knob value, coverage, accept rate, abstention rate).
"""

import argparse
import zlib
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from errctl import conformal as C, paths

PRED_DIR = os.path.join(paths.WORK, "predictions")
OUT = os.path.join(paths.RESULTS, "r07_ablations.csv")

N_CAL = (50, 100, 200, 400, 800, 1600, 3200)
GRIDS = (20, 60, 120, 300)
DELTAS = (0.05, 0.10, 0.20)
MIN_ACCEPT = (10, 25, 50, 100)


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)), flush=True)


def scores_of(probs, kind):
    probs = np.asarray(probs, dtype=np.float64)
    if kind == "maxprob":
        return 1.0 - probs.max(1)
    p = np.clip(probs, 1e-12, 1.0)
    if kind == "entropy":
        h = -(p * np.log(p)).sum(1)
        return h / np.log(p.shape[1])
    part = np.sort(p, axis=1)
    return -(part[:, -1] - part[:, -2])          # margin: smaller is safer


def evaluate(unit, clf, family, s_cal, e_cal, s_te, e_te, alpha, delta, **kw):
    ctl = C.SelectiveRiskController(alpha, delta, **kw).fit(s_cal, e_cal)
    acc = ctl.accept(s_te)
    n = int(acc.sum())
    risk = float(e_te[acc].mean()) if n else np.nan
    return {"unit": unit, "clf": clf, "family": family, "alpha": alpha,
            "n_cal": int(s_cal.size),
            "accept_rate": n / max(len(acc), 1),
            "selective_risk": risk,
            "covered": int((not np.isfinite(risk)) or risk <= alpha),
            "empty": int(n == 0)}


def ablations_for_file(fname, alphas, seed=0):
    d = np.load(os.path.join(PRED_DIR, fname), allow_pickle=True)
    m = re.match(r"(.*)__(logreg|rf|hgb)\.npz$", fname)
    unit, clf = m.group(1), m.group(2)
    family = ("cellsplit" if unit.startswith("cellsplit")
              else "within" if unit.startswith("within") else "transfer")
    cal_y, cal_rep = d["cal_y"], d["cal_rep"]
    test_probs, test_y = d["test_probs"], d["test_y"]
    test_rep = d["test_rep"] if "test_rep" in d.files else None

    rows = []
    # `hash` on str is salted per process (PYTHONHASHSEED), which would make the
    # n_cal ablation irreproducible between runs; crc32 is stable.
    rng = np.random.default_rng(seed + zlib.crc32(fname.encode("utf-8")) % 1000)
    for r in range(cal_rep.shape[1]):
        sel = np.nonzero(cal_rep[:, r])[0]
        te_sel = (np.nonzero(test_rep[:, r])[0] if test_rep is not None
                  else np.arange(len(test_y)))
        p_cal, y_cal = d["cal_probs"][sel], cal_y[sel]
        p_te, y_te = test_probs[te_sel], test_y[te_sel]
        e_cal = (p_cal.argmax(1) != y_cal).astype(float)
        e_te = (p_te.argmax(1) != y_te).astype(float)

        # --- calibration-set size
        order = rng.permutation(sel.size)
        for m_ in N_CAL:
            if m_ > sel.size:
                continue
            idx = order[:m_]
            s_cal = scores_of(p_cal[idx], "maxprob")
            for alpha in alphas:
                row = evaluate(unit, clf, family, s_cal, e_cal[idx],
                               scores_of(p_te, "maxprob"), e_te, alpha, 0.10)
                row.update({"ablation": "n_cal", "value": m_})
                rows.append(row)

        s_cal = scores_of(p_cal, "maxprob")
        s_te = scores_of(p_te, "maxprob")
        for alpha in alphas:
            for g in GRIDS:
                row = evaluate(unit, clf, family, s_cal, e_cal, s_te, e_te, alpha, 0.10,
                               grid_size=g)
                row.update({"ablation": "grid", "value": g})
                rows.append(row)
            for dl in DELTAS:
                row = evaluate(unit, clf, family, s_cal, e_cal, s_te, e_te, alpha, dl)
                row.update({"ablation": "delta", "value": dl})
                rows.append(row)
            for ma in MIN_ACCEPT:
                row = evaluate(unit, clf, family, s_cal, e_cal, s_te, e_te, alpha, 0.10,
                               min_accept=ma)
                row.update({"ablation": "min_accept", "value": ma})
                rows.append(row)
            for sk in ("maxprob", "entropy", "margin"):
                row = evaluate(unit, clf, family, scores_of(p_cal, sk), e_cal,
                               scores_of(p_te, sk), e_te, alpha, 0.10)
                row.update({"ablation": "score", "value": sk})
                rows.append(row)
    return rows


def _job(args):
    return ablations_for_file(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.10,0.20")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", default="")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".npz")
                   and ("cellsplit" in f or "within_" in f))
    if args.limit:
        files = [f for f in files if args.limit in f]
    jobs = [(f, alphas) for f in files]
    log("%d files x %d alphas" % (len(files), len(alphas)))

    rows = []
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            for i, res in enumerate(pool.imap(_job, jobs, chunksize=1), 1):
                rows.extend(res)
                if i % 25 == 0:
                    log("  %d/%d" % (i, len(jobs)))
    else:
        for i, job in enumerate(jobs, 1):
            rows.extend(_job(job))
            if i % 25 == 0:
                log("  %d/%d" % (i, len(jobs)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    log("wrote %s (%d rows)" % (OUT, len(df)))
    for ab in ("n_cal", "grid", "delta", "min_accept", "score"):
        s = df[(df.ablation == ab) & (df.family == "cellsplit")]
        t = (s.groupby(["value", "alpha"])
               .agg(accept=("accept_rate", "mean"), coverage=("covered", "mean"),
                    risk=("selective_risk", "mean")).round(3))
        log("\n[cellsplit] %s" % ab)
        log(t.to_string())


if __name__ == "__main__":
    main()


