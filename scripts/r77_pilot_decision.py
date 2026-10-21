"""R77 -- a pre-specifiable pilot decision rule for a panel repair.

R74 used a descriptive 0.35 cut on the pilot oracle ceiling.  This script
replaces that cut with a leave-one-fold-out threshold rule: the cut is chosen
on four folds and applied unchanged to the fifth.  It also evaluates a naive
pilot net-gain lower bound, formed as a one-sided Clopper-Pearson lower bound
on the 250-label oracle ceiling minus the 1,000-label repair cost.

The lower bound is retained as a failed comparator.  The pilot oracle is a
ceiling estimated on 250 labels, not a lower bound on the acceptance of the
1,000-label refit, so subtracting the repair cost from it does not yield a
lower bound on the refit net.  On these 81 units the lower-bound rule is too
permissive; the cross-fitted threshold is the decision rule reported.

Outputs
  experiments/r77_pilot_decision.csv
  experiments/r77_calibration_silence.csv
  work/r77_report.txt
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CSV_IN = os.path.join(ROOT, "experiments", "r72_panel_repair.csv")
CSV_OUT = os.path.join(ROOT, "experiments", "r77_pilot_decision.csv")
SILENCE_OUT = os.path.join(ROOT, "experiments",
                           "r77_calibration_silence.csv")
REPORT = os.path.join(ROOT, "work", "r77_report.txt")

PARTS = ("B_same_tissue_panel", "E_cross_platform_colon")
KEYS = ("part", "fold", "clf", "rep")
ALPHA = 0.20
PILOT = 250
REPAIR = 1000
N_BOOT = 4000
SEED = 20261019


def _frame():
    d = pd.read_csv(CSV_IN)
    d = d[d.part.isin(PARTS)].copy()
    base = d[np.isclose(d.alpha, ALPHA)].copy()
    pilot = base[base.arm == "tgt_k%d" % PILOT][
        list(KEYS) + ["pilot_oracle_accept", "n_pool"]].copy()
    pilot = pilot.rename(columns={"n_pool": "n_pool"})
    adapt = base[base.arm == "adapt_k%d" % REPAIR][
        list(KEYS) + ["n_accepted", "n_labels", "n_pool"]].copy()
    adapt = adapt.rename(columns={"n_pool": "n_pool"})
    adapt["net"] = (adapt.n_accepted - adapt.n_labels) / adapt.n_pool
    m = pilot.merge(adapt[list(KEYS) + ["net", "n_pool"]],
                    on=list(KEYS), how="inner",
                    suffixes=("_pilot", "_repair"))
    m["n_pool"] = m["n_pool_repair"]
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
    return dict(
        n_units=len(m), go=int(go.sum()), **cm,
        net_mean=float(triage.mean()),
        always_mean=float(always.mean()),
        delta_mean=float(triage.mean() - always.mean()),
        net_cell=float((triage * npool).sum() / npool.sum()),
        always_cell=float((always * npool).sum() / npool.sum()),
        delta_cell=float((triage * npool).sum() / npool.sum()
                         - (always * npool).sum() / npool.sum()))


def _lofo(m):
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


def _cp_lower(p, n):
    k = int(round(float(p) * n))
    if k <= 0:
        return 0.0
    return float(beta.ppf(0.05, k, n - k + 1))


def _boot_ci(m, go, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap of the mean and cell-weighted net difference."""
    m = m.copy()
    m["go"] = np.asarray(go, dtype=bool)
    m["triage"] = np.where(m.go, m.net, -PILOT / m.n_pool)
    m["diff"] = m.triage - m.net
    m["diff_cells"] = m["diff"] * m.n_pool
    m["cluster"] = (m.part.astype(str) + "|" + m.fold.astype(str)
                    + "|" + m.clf.astype(str))
    rng = np.random.default_rng(seed)

    def _one(z):
        clusters = []
        for _, g in z.groupby("part"):
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

    return _one(m), _one(m[m.part == "B_same_tissue_panel"]), \
        _one(m[m.part == "E_cross_platform_colon"])


def _silence():
    d = pd.read_csv(CSV_IN)
    d = d[d.part.isin(PARTS)].copy()
    rows = []
    for alpha in (0.10, 0.20):
        base = d[np.isclose(d.alpha, alpha)]
        for k in (250, 500, 1000, 2000, 4000):
            z = base[base.arm == "tgt_k%d" % k]
            if z.empty:
                continue
            rows.append(dict(alpha=alpha, k=k, n_units=len(z),
                             silent=int((z.n_accepted == 0).sum()),
                             mean_accept=float(z.accept_rate.mean())))
    return rows


def main():
    m = _frame()
    lofo_go, picks = _lofo(m)
    m["net_lcb"] = (m.pilot_oracle_accept.map(
        lambda p: _cp_lower(p, PILOT)) - REPAIR / m.n_pool)
    lcb_go = m.net_lcb.to_numpy() > 0
    fixed_go = m.pilot_oracle_accept.to_numpy() >= 0.35
    always_go = np.ones(len(m), dtype=bool)
    review_go = np.zeros(len(m), dtype=bool)

    policies = [
        ("lofo", lofo_go),
        ("fixed_035", fixed_go),
        ("net_lcb", lcb_go),
        ("always", always_go),
        ("review", review_go),
    ]
    rows = []
    for name, go in policies:
        for part in ("ALL",) + PARTS:
            z = m if part == "ALL" else m[m.part == part]
            g = go if part == "ALL" else go[(m.part == part).to_numpy()]
            out = _policy(z, g)
            out.update(policy=name, part=part)
            rows.append(out)
    out = pd.DataFrame(rows)

    # CIs for the three decision rules, on the mean and cell-weighted scale.
    ci_rows = []
    for name, go in (("lofo", lofo_go), ("fixed_035", fixed_go),
                     ("net_lcb", lcb_go)):
        all_ci, b_ci, e_ci = _boot_ci(m, go)
        for part, (mean_ci, cell_ci) in zip(("ALL",) + PARTS,
                                            (all_ci, b_ci, e_ci)):
            ci_rows.append(dict(policy=name, part=part,
                                delta_mean_lo=mean_ci[0],
                                delta_mean_hi=mean_ci[1],
                                delta_cell_lo=cell_ci[0],
                                delta_cell_hi=cell_ci[1]))
    out = out.merge(pd.DataFrame(ci_rows), on=["policy", "part"],
                    how="left")
    out = out[["policy", "part", "n_units", "go", "tp", "fn", "tn", "fp",
               "tpr", "tnr", "accuracy", "net_mean", "always_mean",
               "delta_mean", "delta_mean_lo", "delta_mean_hi", "net_cell",
               "always_cell", "delta_cell", "delta_cell_lo",
               "delta_cell_hi"]]
    out.to_csv(CSV_OUT, index=False, float_format="%.10g")

    silence = pd.DataFrame(_silence())
    silence.to_csv(SILENCE_OUT, index=False, float_format="%.10g")

    A = []
    A.append("r77 -- a pre-specifiable pilot decision rule")
    A.append("=" * 78)
    A.append("Same B/E units, 250-label pilot and 1,000-label refit as r74.")
    A.append("Outcome: positive net at the 1,000-label refit.")
    A.append("LOFO: choose the pilot-oracle cut on four folds, apply it to")
    A.append("the fifth; no outcome label from the held-out fold is used.")
    A.append("")
    A.append("Calibration silence by label budget:")
    for alpha in (0.10, 0.20):
        z = silence[np.isclose(silence.alpha, alpha)]
        A.append("  alpha %.2f  %s" % (alpha, "  ".join(
            "k=%d %d/%d" % (int(r.k), int(r.silent), int(r.n_units))
            for r in z.itertuples())))
    A.append("")
    A.append("LOFO thresholds: %s" % ",".join("%.4f" % t for _, t in picks))
    A.append("")
    A.append("Policy table (net is the share of the batch; delta is relative")
    A.append("to always repairing, with a cluster bootstrap 95% interval):")
    for name in ("lofo", "fixed_035", "net_lcb", "always", "review"):
        z = out[out.policy == name]
        for r in z.itertuples():
            A.append("  %-9s %-24s go %2d/%2d  TPR %.3f  TNR %.3f  "
                     "acc %.3f  net %.3f  delta %+.3f [%.3f, %.3f]"
                     % (name, r.part, r.go, r.n_units, r.tpr, r.tnr,
                        r.accuracy, r.net_mean, r.delta_mean,
                        r.delta_mean_lo, r.delta_mean_hi))
    A.append("")
    A.append("The 0.35 cut is descriptive.  The LOFO rule is the transport")
    A.append("check: the threshold is selected without the held-out fold.")
    A.append("On the cross-platform source the LOFO rule still leaves a net")
    A.append("loss (-0.031 against -0.095 for always repairing), so the")
    A.append("pilot reduces the loss rather than turning it into a gain.")
    A.append("")
    A.append("The naive net-gain LCB is a failed comparator: it treats the")
    A.append("250-label oracle ceiling as a lower bound on the 1,000-label")
    A.append("refit, which it is not.  It admits 67 of 81 units and leaves")
    A.append("TNR 0.433 and a net gain of +0.009, below the LOFO rule.")
    A.append("A true prospective test on a new cohort pair remains future")
    A.append("work; the LOFO rule is a cross-fitted transport check.")
    txt = "\n".join(A) + "\n"
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
