"""R74 -- can a pilot before repair tell a payload from a dead end?

The panel-repair experiment (``r72_panel_repair.py``) contains two deployments
of the same target core from different sources.  The depot colorectal source
(Part B) is repaired by a refit, while the STELLAR intestine source (Part E)
rebuilds acceptance but never repays the labels it spends.  This script asks
whether a small pilot, spent before the repair, can tell those two situations
apart.

The pilot is the frozen model calibrated on k_pilot target labels.  It yields
two candidate statistics:

  ratio          certificate acceptance divided by the target-labelled oracle
  oracle_proxy   the oracle acceptance itself, a ceiling estimated on the pilot

The primary outcome is whether the 1,000-label refit, the rung the policy
actually spends, has a positive net.  A secondary outcome allows any refit
rung to repay.  We report source-level separation, unit-level ROC AUC,
threshold sensitivity, a leave-one-fold-out threshold rule and a conservative
policy that spends the pilot and then repairs only when the proxy clears the
threshold.

Outputs
  experiments/r74_repair_triage.csv
  experiments/r74_repair_triage_summary.csv
  work/r74_report.txt
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

CSV_IN = os.path.join(ROOT, "experiments", "r72_panel_repair.csv")
CSV_OUT = os.path.join(ROOT, "experiments", "r74_repair_triage.csv")
SUMMARY_OUT = os.path.join(ROOT, "experiments", "r74_repair_triage_summary.csv")
REPORT = os.path.join(ROOT, "work", "r74_report.txt")

PARTS = ("B_same_tissue_panel", "E_cross_platform_colon")
ALPHAS = (0.10, 0.20)
PILOTS = (250, 500, 1000)
THRESHOLDS = (0.25, 0.35, 0.45, 0.55)
KEYS = ("part", "fold", "clf", "rep")
PRIMARY_ALPHA = 0.20
PRIMARY_K = 250
PRIMARY_THRESHOLD = 0.35
PRIMARY_ARM = "adapt_k1000"
N_BOOT = 4000
SEED = 20261018


def _pilot_frame(df, alpha, k):
    base = df[np.isclose(df.alpha, alpha)].copy()
    pilot = base[base.arm == "tgt_k%d" % k][
        list(KEYS) + ["accept_rate", "ratio", "oracle_accept",
                      "pilot_oracle_accept", "pilot_ratio", "accept_rate_ud",
                      "ratio_ud", "pilot_ratio_ud",
                      "selective_risk_ud"]].copy()
    adapt = base[base.arm.str.startswith("adapt_")].copy()
    adapt["net"] = (adapt.n_accepted - adapt.n_labels) / adapt.n_pool
    best = adapt.groupby(list(KEYS), as_index=False).net.max()
    best = best.rename(columns={"net": "best_net"})
    return pilot.merge(best, on=list(KEYS), how="inner")


def _pilot_frame_arm(df, alpha, k, arm):
    base = df[np.isclose(df.alpha, alpha)].copy()
    pilot = base[base.arm == "tgt_k%d" % k][
        list(KEYS) + ["accept_rate", "ratio", "oracle_accept",
                      "pilot_oracle_accept", "pilot_cert_accept",
                      "pilot_ratio", "n_pool", "accept_rate_ud", "ratio_ud",
                      "pilot_cert_accept_ud", "pilot_ratio_ud",
                      "selective_risk_ud"]].copy()
    adapt = base[base.arm == arm].copy()
    adapt["net"] = (adapt.n_accepted - adapt.n_labels) / adapt.n_pool
    adapt = adapt[list(KEYS) + ["net"]]
    return pilot.merge(adapt, on=list(KEYS), how="inner")


def _auc(y, x, min_nonzero=1):
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if (good.sum() < 2 or len(np.unique(y[good])) < 2
            or len(np.unique(x[good])) < 2):
        return np.nan
    if min_nonzero > 1 and np.count_nonzero(x[good]) < min_nonzero:
        return np.nan
    return float(roc_auc_score(y[good], x[good]))


def _confusion(m, outcome, threshold):
    y = (m[outcome].to_numpy() > 0)
    pred = m.pilot_oracle_accept.to_numpy() >= threshold
    tp = int((pred & y).sum())
    fn = int(((~pred) & y).sum())
    tn = int(((~pred) & (~y)).sum())
    fp = int((pred & (~y)).sum())
    return dict(tp=tp, fn=fn, tn=tn, fp=fp,
                tpr=tp / max(1, tp + fn), tnr=tn / max(1, tn + fp),
                accuracy=(tp + tn) / max(1, len(m)))


def _cluster_key(m):
    return (m.part.astype(str) + "|" + m.fold.astype(str) + "|"
            + m.clf.astype(str))


def _fmt_auc(value):
    return "n.e." if not np.isfinite(value) else "%.3f" % value


def _bootstrap_metrics(m, outcome, threshold, n_boot=N_BOOT, seed=SEED):
    """Stratified cluster bootstrap over fold x classifier units."""
    rng = np.random.default_rng(seed)
    m = m.copy()
    m["cluster"] = _cluster_key(m)
    parts = {p: sorted(g.cluster.unique()) for p, g in m.groupby("part")}
    vals = {k: [] for k in ("oracle_auc", "ratio_auc", "pilot_ratio_auc",
                            "ratio_ud_auc", "pilot_ratio_ud_auc",
                            "tpr", "tnr", "accuracy")}
    for _ in range(n_boot):
        picks = []
        for part, clusters in parts.items():
            take = rng.choice(clusters, size=len(clusters), replace=True)
            for c in take:
                picks.append(m[m.cluster == c])
        z = pd.concat(picks, ignore_index=True)
        y = (z[outcome].to_numpy() > 0)
        if y.all() or (~y).all():
            continue
        vals["oracle_auc"].append(_auc(y, z.pilot_oracle_accept))
        vals["ratio_auc"].append(_auc(y, z.ratio, min_nonzero=2))
        vals["pilot_ratio_auc"].append(
            _auc(y, z.pilot_ratio, min_nonzero=2))
        vals["ratio_ud_auc"].append(_auc(y, z.ratio_ud, min_nonzero=2))
        vals["pilot_ratio_ud_auc"].append(
            _auc(y, z.pilot_ratio_ud, min_nonzero=2))
        cm = _confusion(z, outcome, threshold)
        for key in ("tpr", "tnr", "accuracy"):
            vals[key].append(cm[key])
    out = {}
    for key, v in vals.items():
        v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
        if v.size:
            out[key] = (float(np.quantile(v, 0.025)),
                        float(np.quantile(v, 0.975)), int(v.size))
        else:
            out[key] = (np.nan, np.nan, 0)
    return out


def _lofo_threshold(m, outcome):
    """Choose a cut on four folds and apply it to the fifth."""
    grid = np.linspace(0.0, 0.8, 321)
    pred = np.zeros(len(m), dtype=int)
    picks = []
    for fold in sorted(m.fold.unique()):
        train = m.fold != fold
        test = ~train
        best = None
        for thr in grid:
            cm = _confusion(m.loc[train], outcome, float(thr))
            score = 0.5 * (cm["tpr"] + cm["tnr"])
            if best is None or score > best[0] + 1e-12:
                best = (score, float(thr))
        picks.append((int(fold), best[1]))
        pred[test.to_numpy()] = (
            m.loc[test, "pilot_oracle_accept"].to_numpy() >= best[1]).astype(int)
    y = m[outcome].to_numpy() > 0
    q = pred.astype(bool)
    tp = int((q & y).sum())
    fn = int(((~q) & y).sum())
    tn = int(((~q) & (~y)).sum())
    fp = int((q & (~y)).sum())
    return dict(picks=picks, tp=tp, fn=fn, tn=tn, fp=fp,
                tpr=tp / max(1, tp + fn), tnr=tn / max(1, tn + fp),
                accuracy=(tp + tn) / max(1, len(m)))


def _policy_frame(df, part):
    m = _pilot_frame_arm(df, PRIMARY_ALPHA, PRIMARY_K, PRIMARY_ARM)
    m = m[m.part == part].copy()
    m["go"] = m.pilot_oracle_accept >= PRIMARY_THRESHOLD
    m["cost"] = PRIMARY_K / m.n_pool
    # The 250 pilot labels are a subset of the 1,000 repair labels, and
    # ``net`` already subtracts all 1,000.  Charge the pilot only to a unit
    # that declines the repair; charging it to both arms double-counts it.
    m["triage"] = np.where(m.go, m.net, -m.cost)
    m["always"] = m.net
    return m


def _bootstrap_policy(m, n_boot=N_BOOT, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    m = m.copy()
    m["cluster"] = m.fold.astype(str) + "|" + m.clf.astype(str)
    clusters = sorted(m.cluster.unique())
    diffs = []
    for _ in range(n_boot):
        take = rng.choice(clusters, size=len(clusters), replace=True)
        z = pd.concat([m[m.cluster == c] for c in take], ignore_index=True)
        diffs.append(float(z.triage.mean() - z.always.mean()))
    return (float(np.quantile(diffs, 0.025)),
            float(np.quantile(diffs, 0.975)), len(diffs))


def _source_separation(df, alpha, k):
    m = _pilot_frame(df, alpha, k)
    out = []
    for part in PARTS:
        x = m[m.part == part].pilot_oracle_accept
        out.append(dict(alpha=alpha, pilot_k=k, part=part, n=len(x),
                        min_oracle=float(x.min()), max_oracle=float(x.max()),
                        median_oracle=float(x.median())))
    return out


def main():
    df = pd.read_csv(CSV_IN)
    df = df[df.part.isin(PARTS)].copy()
    rows = []
    sep = []
    for alpha in ALPHAS:
        for k in PILOTS:
            m = _pilot_frame(df, alpha, k)
            m2 = _pilot_frame_arm(df, alpha, k, PRIMARY_ARM)
            m = m.merge(m2[list(KEYS) + ["net"]], on=list(KEYS), how="inner")
            m["outcome_k1000"] = m.net
            m["outcome_any"] = m.best_net
            sep.extend(_source_separation(df, alpha, k))
            for outcome in ("outcome_k1000", "outcome_any"):
                y = (m[outcome] > 0).to_numpy()
                oracle_auc = _auc(y, m.pilot_oracle_accept)
                ratio_auc = (_auc(y, m.ratio, min_nonzero=2)
                             if m.ratio.notna().any() else np.nan)
                pilot_ratio_auc = (_auc(y, m.pilot_ratio, min_nonzero=2)
                                   if m.pilot_ratio.notna().any() else np.nan)
                ratio_ud_auc = (_auc(y, m.ratio_ud, min_nonzero=2)
                                if m.ratio_ud.notna().any() else np.nan)
                pilot_ratio_ud_auc = (_auc(y, m.pilot_ratio_ud,
                                           min_nonzero=2)
                                      if m.pilot_ratio_ud.notna().any()
                                      else np.nan)
                rho_o = (spearmanr(m.pilot_oracle_accept, m[outcome]).correlation
                         if m.pilot_oracle_accept.nunique() > 1 else np.nan)
                rho_r = (spearmanr(m.ratio, m[outcome]).correlation
                         if m.ratio.nunique() > 1 else np.nan)
                for thr in THRESHOLDS:
                    cm = _confusion(m, outcome, thr)
                    rows.append(dict(
                        alpha=alpha, pilot_k=k, threshold=thr, outcome=outcome,
                        n=len(m), positives=int(y.sum()),
                        oracle_auc=oracle_auc, ratio_auc=ratio_auc,
                        pilot_ratio_auc=pilot_ratio_auc,
                        ratio_ud_auc=ratio_ud_auc,
                        pilot_ratio_ud_auc=pilot_ratio_ud_auc,
                        ratio_support=int((m.ratio > 0).sum()),
                        pilot_ratio_support=int((m.pilot_ratio > 0).sum()),
                        ratio_ud_support=int((m.ratio_ud > 0).sum()),
                        pilot_ratio_ud_support=int(
                            (m.pilot_ratio_ud > 0).sum()),
                        spearman_oracle=rho_o, spearman_ratio=rho_r, **cm))

    # A conservative policy: pilot on 250 labels, then repair with 1,000
    # labels only when the 20% oracle proxy clears 0.35.  The pilot is not
    # charged twice: the 1,000 repair labels include the pilot draw.
    policy = []
    for part in PARTS:
        z = _policy_frame(df, part)
        lo, hi, nb = _bootstrap_policy(z)
        policy.append(dict(part=part, n=len(z), go=int(z.go.sum()),
                           always_repair=float(z.always.mean()),
                           triage_repair=float(z.triage.mean()),
                           review_only=float((-z.cost).mean()),
                           triage_minus_always=float(
                               z.triage.mean() - z.always.mean()),
                           diff_lo=lo, diff_hi=hi, n_boot=nb))

    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

    primary = _pilot_frame_arm(df, PRIMARY_ALPHA, PRIMARY_K, PRIMARY_ARM)
    primary["outcome_k1000"] = primary.net
    primary["outcome_any"] = _pilot_frame(
        df, PRIMARY_ALPHA, PRIMARY_K).best_net
    summary = []
    for outcome in ("outcome_k1000", "outcome_any"):
        y = (primary[outcome] > 0).to_numpy()
        cm = _confusion(primary, outcome, PRIMARY_THRESHOLD)
        bs = _bootstrap_metrics(primary, outcome, PRIMARY_THRESHOLD)
        lofo = _lofo_threshold(primary, outcome)
        for metric, est in (
                ("oracle_auc", _auc(y, primary.pilot_oracle_accept)),
                ("ratio_auc", _auc(y, primary.ratio, min_nonzero=2)),
                ("pilot_ratio_auc",
                 _auc(y, primary.pilot_ratio, min_nonzero=2)),
                ("ratio_ud_auc",
                 _auc(y, primary.ratio_ud, min_nonzero=2)),
                ("pilot_ratio_ud_auc",
                 _auc(y, primary.pilot_ratio_ud, min_nonzero=2)),
                ("tpr", cm["tpr"]), ("tnr", cm["tnr"]),
                ("accuracy", cm["accuracy"])):
            lo, hi, nb = bs[metric]
            if not np.isfinite(est):
                lo = hi = np.nan
                nb = 0
            summary.append(dict(outcome=outcome, metric=metric,
                                estimate=est, lo=lo, hi=hi, n_boot=nb,
                                note="threshold %.2f" % PRIMARY_THRESHOLD))
        summary.append(dict(outcome=outcome, metric="lofo_tpr",
                            estimate=lofo["tpr"], lo=np.nan, hi=np.nan,
                            n_boot=0,
                            note="thresholds " + ",".join(
                                "%.4f" % t for _, t in lofo["picks"])))
        summary.append(dict(outcome=outcome, metric="lofo_tnr",
                            estimate=lofo["tnr"], lo=np.nan, hi=np.nan,
                            n_boot=0, note="fold-wise threshold selection"))
        summary.append(dict(outcome=outcome, metric="lofo_accuracy",
                            estimate=lofo["accuracy"], lo=np.nan, hi=np.nan,
                            n_boot=0, note="fold-wise threshold selection"))
    summary.extend([
        dict(outcome="outcome_k1000", metric="n_units",
             estimate=float(len(primary)), lo=np.nan, hi=np.nan, n_boot=0,
             note="pilot 250 at alpha 0.20"),
        dict(outcome="outcome_k1000", metric="cert_accept_zero",
             estimate=float((primary.accept_rate == 0).sum()), lo=np.nan,
             hi=np.nan, n_boot=0, note="pilot 250 at alpha 0.20"),
        dict(outcome="outcome_k1000", metric="ratio_zero_or_nan",
             estimate=float((primary.ratio.isna() | (primary.ratio == 0)).sum()),
             lo=np.nan, hi=np.nan, n_boot=0, note="deployment oracle ratio"),
        dict(outcome="outcome_k1000", metric="pilot_ratio_zero_or_nan",
             estimate=float((primary.pilot_ratio.isna()
                             | (primary.pilot_ratio == 0)).sum()),
             lo=np.nan, hi=np.nan, n_boot=0, note="pilot-only ratio"),
        dict(outcome="outcome_k1000", metric="ratio_support",
             estimate=float((primary.ratio > 0).sum()), lo=np.nan, hi=np.nan,
             n_boot=0, note="nonzero deployment ratios"),
        dict(outcome="outcome_k1000", metric="pilot_ratio_support",
             estimate=float((primary.pilot_ratio > 0).sum()), lo=np.nan,
             hi=np.nan, n_boot=0, note="nonzero pilot-only ratios"),
        dict(outcome="outcome_k1000", metric="cert_accept_ud_zero",
             estimate=float((primary.accept_rate_ud == 0).sum()),
             lo=np.nan, hi=np.nan, n_boot=0,
             note="unlabelled-denominator certificate"),
        dict(outcome="outcome_k1000", metric="ratio_ud_zero_or_nan",
             estimate=float((primary.ratio_ud.isna()
                             | (primary.ratio_ud == 0)).sum()),
             lo=np.nan, hi=np.nan, n_boot=0,
             note="unlabelled denominator"),
        dict(outcome="outcome_k1000", metric="pilot_ratio_ud_zero_or_nan",
             estimate=float((primary.pilot_ratio_ud.isna()
                             | (primary.pilot_ratio_ud == 0)).sum()),
             lo=np.nan, hi=np.nan, n_boot=0,
             note="pilot-only, unlabelled denominator"),
        dict(outcome="outcome_k1000", metric="ratio_ud_support",
             estimate=float((primary.ratio_ud > 0).sum()), lo=np.nan,
             hi=np.nan, n_boot=0, note="nonzero unlabelled-denominator ratios"),
        dict(outcome="outcome_k1000", metric="pilot_ratio_ud_support",
             estimate=float((primary.pilot_ratio_ud > 0).sum()), lo=np.nan,
             hi=np.nan, n_boot=0,
             note="nonzero pilot-only unlabelled-denominator ratios"),
    ])
    for x in policy:
        summary.append(dict(outcome="outcome_k1000",
                            metric="policy_triage_minus_always_%s" % x["part"],
                            estimate=x["triage_minus_always"], lo=x["diff_lo"],
                            hi=x["diff_hi"], n_boot=x["n_boot"],
                            note="pilot %d, repair %s" % (PRIMARY_K, PRIMARY_ARM)))
    pd.DataFrame(summary).to_csv(SUMMARY_OUT, index=False)

    A = []
    A.append("r74 -- a pilot before the repair")
    A.append("=" * 78)
    A.append("Parts B and E, same target core, different sources.")
    A.append("Pilot = frozen model calibrated on 250/500/1,000 target labels.")
    A.append("Primary outcome = positive net at the 1,000-label refit.")
    A.append("Secondary outcome = positive net at any refit rung.")
    A.append("Confidence intervals are a stratified cluster bootstrap over")
    A.append("fold x classifier units, 4,000 resamples.")
    A.append("")
    for alpha in ALPHAS:
        for k in PILOTS:
            r = [x for x in rows if x["alpha"] == alpha
                 and x["pilot_k"] == k
                 and x["outcome"] == "outcome_k1000"]
            if not r:
                continue
            r0 = r[0]
            A.append("alpha %.2f  pilot k=%-4d n=%d  positives=%d  "
                     "oracle AUC %.3f  deploy ratio %s (sup %d)  "
                     "pilot ratio %s (sup %d)  deploy ratio(UD) %s "
                     "(sup %d)  pilot ratio(UD) %s (sup %d)"
                     % (alpha, k, r0["n"], r0["positives"],
                        r0["oracle_auc"],
                        _fmt_auc(r0["ratio_auc"]), r0["ratio_support"],
                        _fmt_auc(r0["pilot_ratio_auc"]),
                        r0["pilot_ratio_support"],
                        _fmt_auc(r0["ratio_ud_auc"]), r0["ratio_ud_support"],
                        _fmt_auc(r0["pilot_ratio_ud_auc"]),
                        r0["pilot_ratio_ud_support"]))
            A.append("  spearman oracle %.3f  ratio %.3f"
                     % (r0["spearman_oracle"], r0["spearman_ratio"]))
            for rr in r:
                A.append("    threshold %.2f  TPR %.3f  TNR %.3f  acc %.3f  "
                         "tp/fn/tn/fp %d/%d/%d/%d"
                         % (rr["threshold"], rr["tpr"], rr["tnr"],
                            rr["accuracy"], rr["tp"], rr["fn"],
                            rr["tn"], rr["fp"]))
    A.append("")
    A.append("Source-level oracle separation at alpha 0.20 and k=250:")
    for x in sep:
        if x["alpha"] == 0.20 and x["pilot_k"] == 250:
            A.append("  %-28s n=%3d  min %.3f  median %.3f  max %.3f"
                     % (x["part"], x["n"], x["min_oracle"],
                        x["median_oracle"], x["max_oracle"]))
    A.append("")
    A.append("Policy at alpha 0.20: pilot 250, repair 1,000 if proxy >= 0.35.")
    for x in policy:
        A.append("  %-28s go %3d/%3d  always %.3f  triage %.3f  review %.3f  "
                 "triage-always %+.3f [%.3f, %.3f]"
                 % (x["part"], x["go"], x["n"], x["always_repair"],
                    x["triage_repair"], x["review_only"],
                    x["triage_minus_always"], x["diff_lo"], x["diff_hi"]))
    A.append("")
    A.append("Leave-one-fold-out thresholds and held-out performance:")
    for outcome in ("outcome_k1000", "outcome_any"):
        lo = _lofo_threshold(primary, outcome)
        A.append("  %-14s thresholds %s  TPR %.3f  TNR %.3f  acc %.3f"
                 % (outcome, ",".join("%.3f" % t for _, t in lo["picks"]),
                    lo["tpr"], lo["tnr"], lo["accuracy"]))
    A.append("")
    A.append("At a 10% budget the 250-label certificate accepts nothing in all")
    A.append("81 units, so neither the deployment ratio nor the pilot-only")
    A.append("ratio has any support and neither AUC is estimable. At 20% it")
    A.append("accepts nothing in 80 of 81 under the calibration-only")
    A.append("denominator; each ratio has exactly one nonzero value, so the")
    A.append("0.51 figure is the binary non-silence rate, not a ranking AUC.")
    A.append("Reading the denominator off the unlabelled deployment scores")
    A.append("(target_scores) raises support to 6 of 81, but the AUC is 0.53")
    A.append("on six values and remains indistinguishable from chance. The")
    A.append("numerator, not the denominator, is the binding constraint at 250")
    A.append("labels. The 10% silence is expected: the one-sided bound needs")
    A.append("the accepted subset to have near-zero calibration errors, which")
    A.append("a cross-panel model at a 0.10 budget cannot supply. At 500 and")
    A.append("1,000 labels the deployment ratio becomes estimable (AUC 0.55")
    A.append("and 0.73), but the pilot then costs as much as the repair. The")
    A.append("oracle proxy is the usable statistic at 250 labels, and it needs")
    A.append("pilot labels.")
    A.append("The fixed 0.35 cut is descriptive, not pre-registered; the")
    A.append("leave-one-fold-out rule is the prospective check on transport.")
    txt = "\n".join(A) + "\n"
    open(REPORT, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
