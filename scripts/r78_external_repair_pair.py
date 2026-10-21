"""R78 -- a second external repair pair: breast and tonsil panel changes.

The B/E pair used one target core (colon R8C1) with two sources.  This script
rebuilds parts C and D of r72 with the current pilot fields, so the same
250-label pilot rule can be tested on a second pair of target tissues:

  C  depot breast -> MIBITOF breast cores, same tissue
  D  STELLAR tonsil -> MIBITOF tonsil cores, cross-platform

The C/D rows in the released r72 table predate the pilot columns, so they are
recomputed here into a separate table; r72 itself is not modified.

Outputs
  experiments/r78_external_repair_pair.csv
  experiments/r78_external_pair_summary.csv
  work/r78_report.txt
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import r72_panel_repair as R72  # noqa: E402

CSV_OUT = os.path.join(ROOT, "experiments", "r78_external_repair_pair.csv")
SUMMARY_OUT = os.path.join(ROOT, "experiments",
                           "r78_external_pair_summary.csv")
REPORT = os.path.join(ROOT, "work", "r78_report.txt")

KEYS = ("part", "fold", "clf", "rep")
PARTS = ("C_same_tissue_panel_breast", "D_cross_platform_tonsil")
ALPHA = 0.20
PILOT = 250
REPAIR = 1000
N_BOOT = 2000
SEED = 20261019
CUT_BE = 0.3725


def _job_c(spec):
    fold, kind = spec
    return R72.job_b(R72.PARTS_B[1][1] + (fold, kind))


def _job_d(spec):
    return R72.job_d(spec)


def _run_cd(jobs):
    specs = [(f, c) for f in range(R72.N_FOLDS) for c in R72.CLFS]
    rows = []
    t0 = time.time()
    with mp.Pool(jobs) as pool:
        for i, res in enumerate(pool.imap(_job_c, specs, chunksize=1),
                                1):
            rows.extend(res)
            print("  C %d/%d rows=%d %.0fs"
                  % (i, len(specs), len(rows), time.time() - t0), flush=True)
    with mp.Pool(jobs) as pool:
        for i, res in enumerate(pool.imap(_job_d, specs, chunksize=1),
                                1):
            rows.extend(res)
            print("  D %d/%d rows=%d %.0fs"
                  % (i, len(specs), len(rows), time.time() - t0), flush=True)
    return pd.DataFrame(rows)


def _frame(df):
    base = df[np.isclose(df.alpha, ALPHA)].copy()
    pilot = base[base.arm == "tgt_k%d" % PILOT][
        list(KEYS) + ["pilot_oracle_accept", "n_pool"]].copy()
    adapt = base[base.arm == "adapt_k%d" % REPAIR][
        list(KEYS) + ["n_accepted", "n_labels", "n_pool"]].copy()
    adapt["net"] = (adapt.n_accepted - adapt.n_labels) / adapt.n_pool
    m = pilot.merge(adapt[list(KEYS) + ["net", "n_pool"]],
                    on=list(KEYS), how="inner",
                    suffixes=("_pilot", "_repair"))
    m["n_pool"] = m.n_pool_repair
    m["outcome"] = m.net > 0
    return m


def _confusion(m, go):
    go = np.asarray(go, dtype=bool)
    y = m.outcome.to_numpy(dtype=bool)
    tp = int((go & y).sum())
    fn = int(((~go) & y).sum())
    tn = int(((~go) & (~y)).sum())
    fp = int((go & (~y)).sum())
    return dict(tp=tp, fn=fn, tn=tn, fp=fp,
                tpr=(tp / (tp + fn)) if (tp + fn) else np.nan,
                tnr=(tn / (tn + fp)) if (tn + fp) else np.nan,
                accuracy=(tp + tn) / max(1, len(m)))


def _policy(m, go):
    go = np.asarray(go, dtype=bool)
    triage = np.where(go, m.net.to_numpy(), -PILOT / m.n_pool.to_numpy())
    always = m.net.to_numpy()
    npool = m.n_pool.to_numpy()
    cm = _confusion(m, go)
    return dict(n=len(m), go=int(go.sum()), **cm,
                net=float(triage.mean()),
                always=float(always.mean()),
                delta=float(triage.mean() - always.mean()),
                net_cell=float((triage * npool).sum() / npool.sum()),
                always_cell=float((always * npool).sum() / npool.sum()),
                delta_cell=float((triage * npool).sum() / npool.sum()
                                 - (always * npool).sum() / npool.sum()))


def _lofo(m):
    """Cross-fitted threshold inside one deployment pair.

    Folds are shared between parts C and D, so a threshold fitted on the pooled
    table would borrow units from the other tissue.  Each part is a separate
    deployment and gets its own threshold.
    """
    grid = np.linspace(0.0, 0.8, 321)
    go = np.zeros(len(m), dtype=bool)
    picks = []
    for fold in sorted(m.fold.unique()):
        train = (m.fold != fold).to_numpy()
        test = ~train
        best = None
        for thr in grid:
            pred = m.pilot_oracle_accept.to_numpy()[train] >= thr
            y = m.outcome.to_numpy()[train]
            tpr = (pred & y).sum() / max(1, y.sum())
            tnr = ((~pred) & (~y)).sum() / max(1, (~y).sum())
            score = 0.5 * (tpr + tnr)
            if best is None or score > best[0] + 1e-12:
                best = (score, float(thr))
        picks.append((int(fold), best[1]))
        go[test] = m.pilot_oracle_accept.to_numpy()[test] >= best[1]
    return go, picks


def _boot_delta(m, go, n_boot=N_BOOT, seed=SEED):
    m = m.copy()
    m["go"] = np.asarray(go, dtype=bool)
    m["triage"] = np.where(m.go, m.net, -PILOT / m.n_pool)
    m["diff"] = m.triage - m.net
    m["diff_cells"] = m["diff"] * m.n_pool
    m["cluster"] = (m.part.astype(str) + "|" + m.fold.astype(str)
                    + "|" + m.clf.astype(str))
    rng = np.random.default_rng(seed)
    clusters = []
    for _, g in m.groupby("part"):
        c = g.groupby("cluster", as_index=False).agg(
            diff=("diff", "sum"), count=("diff", "size"),
            diff_cells=("diff_cells", "sum"), npool=("n_pool", "sum"))
        clusters.append(c)
    mean_d, cell_d = [], []
    for _ in range(n_boot):
        td = tc = tdc = tn = 0.0
        for c in clusters:
            take = rng.choice(len(c), size=len(c), replace=True)
            td += float(c["diff"].to_numpy()[take].sum())
            tc += float(c["count"].to_numpy()[take].sum())
            tdc += float(c.diff_cells.to_numpy()[take].sum())
            tn += float(c.npool.to_numpy()[take].sum())
        mean_d.append(td / tc)
        cell_d.append(tdc / tn)
    return ((float(np.quantile(mean_d, 0.025)),
             float(np.quantile(mean_d, 0.975))),
            (float(np.quantile(cell_d, 0.025)),
             float(np.quantile(cell_d, 0.975))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args()
    if args.from_csv and os.path.exists(CSV_OUT):
        df = pd.read_csv(CSV_OUT)
    else:
        df = _run_cd(args.jobs)
        df.to_csv(CSV_OUT, index=False)
        print("wrote", CSV_OUT)

    m = _frame(df)
    rows = []
    # The cut learned on B/E is applied unchanged to C/D.
    transport_go = m.pilot_oracle_accept.to_numpy() >= CUT_BE
    for part in ("ALL",) + PARTS:
        z = m if part == "ALL" else m[m.part == part]
        g = transport_go if part == "ALL" else \
            transport_go[(m.part == part).to_numpy()]
        out = _policy(z, g)
        out.update(policy="transport_BE_0.3725", part=part)
        rows.append(out)

    # A threshold chosen on four folds of one pair and applied to the fifth.
    lofo_go = np.zeros(len(m), dtype=bool)
    picks = {}
    for part in PARTS:
        sel = (m.part == part).to_numpy()
        g, p = _lofo(m[sel])
        lofo_go[sel] = g
        picks[part] = p
    for part in ("ALL",) + PARTS:
        z = m if part == "ALL" else m[m.part == part]
        g = lofo_go if part == "ALL" else \
            lofo_go[(m.part == part).to_numpy()]
        out = _policy(z, g)
        out.update(policy="lofo_CD", part=part)
        rows.append(out)

    for part in ("ALL", "C_same_tissue_panel_breast",
                 "D_cross_platform_tonsil"):
        z = m if part == "ALL" else m[m.part == part]
        out = _policy(z, np.ones(len(z), dtype=bool))
        out.update(policy="always", part=part)
        rows.append(out)
        out = _policy(z, np.zeros(len(z), dtype=bool))
        out.update(policy="review", part=part)
        rows.append(out)

    # An oracle that knows which units repay is the upper bound on per-unit
    # triage; it shows whether a better pilot could pay at all.
    for part in ("ALL",) + PARTS:
        z = m if part == "ALL" else m[m.part == part]
        out = _policy(z, z.outcome.to_numpy(dtype=bool))
        out.update(policy="oracle", part=part)
        rows.append(out)

    out = pd.DataFrame(rows)
    ci_rows = []
    for part, g in (("ALL", lofo_go),
                    ("C_same_tissue_panel_breast",
                     lofo_go[(m.part == "C_same_tissue_panel_breast").to_numpy()]),
                    ("D_cross_platform_tonsil",
                     lofo_go[(m.part == "D_cross_platform_tonsil").to_numpy()])):
        z = m if part == "ALL" else m[m.part == part]
        mean_ci, cell_ci = _boot_delta(z, g)
        ci_rows.append(dict(policy="lofo_CD", part=part,
                            delta_lo=mean_ci[0], delta_hi=mean_ci[1],
                            delta_cell_lo=cell_ci[0],
                            delta_cell_hi=cell_ci[1]))
    out = out.merge(pd.DataFrame(ci_rows), on=["policy", "part"], how="left")
    out = out[["policy", "part", "n", "go", "tp", "fn", "tn", "fp", "tpr",
               "tnr", "accuracy", "net", "always", "delta", "delta_lo",
               "delta_hi", "net_cell", "always_cell", "delta_cell",
               "delta_cell_lo", "delta_cell_hi"]]
    out.to_csv(SUMMARY_OUT, index=False, float_format="%.10g")

    A = []
    A.append("r78 -- a second external repair pair")
    A.append("=" * 78)
    A.append("Parts C (breast, same tissue) and D (tonsil, cross-platform).")
    A.append("Pilot 250, repair 1,000, alpha 0.20; outcome = positive net.")
    A.append("")
    for part in PARTS:
        z = m[m.part == part]
        A.append("  %-32s units %d  positives %d  pilot-oracle AUC %.3f"
                 % (part, len(z), int(z.outcome.sum()),
                    roc_auc_score(z.outcome, z.pilot_oracle_accept)))
    A.append("")
    A.append("Transport of the B/E cut 0.3725:")
    for r in out[out.policy == "transport_BE_0.3725"].itertuples():
        A.append("  %-24s go %2d/%2d  TPR %s  TNR %s  acc %.3f  net %.3f  "
                 "delta %+.3f"
                 % (r.part, r.go, r.n,
                    "n.a." if not np.isfinite(r.tpr) else "%.3f" % r.tpr,
                    "n.a." if not np.isfinite(r.tnr) else "%.3f" % r.tnr,
                    r.accuracy, r.net, r.delta))
    A.append("")
    for part in PARTS:
        A.append("LOFO thresholds on %s: %s"
                 % (part, ",".join("%.4f" % t for _, t in picks[part])))
    for r in out[out.policy == "lofo_CD"].itertuples():
        A.append("  %-24s go %2d/%2d  TPR %s  TNR %s  acc %.3f  net %.3f  "
                 "delta %+.3f [%.3f, %.3f]"
                 % (r.part, r.go, r.n,
                    "n.a." if not np.isfinite(r.tpr) else "%.3f" % r.tpr,
                    "n.a." if not np.isfinite(r.tnr) else "%.3f" % r.tnr,
                    r.accuracy, r.net, r.delta, r.delta_lo, r.delta_hi))
    A.append("")
    A.append("Oracle upper bound on per-unit triage (knows which units repay):")
    for r in out[out.policy == "oracle"].itertuples():
        A.append("  %-24s net %.3f  delta %+.3f"
                 % (r.part, r.net, r.delta))
    A.append("")
    for part in PARTS:
        z = m[m.part == part]
        go = z.pilot_oracle_accept.to_numpy() >= CUT_BE
        A.append("Transport check on %s: the B/E cut sends %d/%d units to repair."
                 % (part, int(go.sum()), len(z)))
    A.append("A positive within-pair result is a second-pair replication; a")
    A.append("negative one is reported as an external failure, not hidden.")
    txt = "\n".join(A) + "\n"
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
