"""R48b -- the cross-field-of-view repair arms at field-of-view resolution.

`scripts/r48_fov_units.py` re-estimates the cross-field-of-view family at
field-of-view resolution: one annotated image is held out per unit, so the
family has 96 units instead of the published twenty, and the shortfall is
resolved -- at a 5% and at a 10% budget the interval on the certificate's own
coverage excludes the nominal 0.90.

The repair section of the paper answers that shortfall with three
score-stratified arms and a predicted-class (Mondrian) arm, computed on the
published twenty units.  This script asks whether the answer survives the
sharper design.  It imports the arm implementations from
`scripts/r31_targeted_repair.py` -- the same certificate classes, the same
confidence level, the same minimum-acceptance floor -- and runs the part A
(within-cohort) arms on the field-of-view units, so nothing about the arms can
drift from the released `experiments/r31_within_summary.csv`.

Only the units change.  The `tgt_k` label-ladder references are not produced:
that ladder belongs to the r20 calibration-size study and has no meaning for a
one-image-held-out unit.  The r31 script itself is not modified and its released
tables are not touched.

Outputs
  experiments/r48b_fov_repair_detail.csv
  experiments/r48b_fov_repair_summary.csv   (coverage_all and coverage_nonvac
                                             carry unit-bootstrap intervals; the
                                             conditional readout is the point
                                             estimate agg_detail derives)
  experiments/r48b_fov_repair_paired.csv    (paired against `plain`, pooled)
  experiments/r48b_fov_repair_paired_by_alpha.csv  (paired, per budget)
  work/r48b_report.txt

The bootstrap is the one `scripts/r31_targeted_repair.py` uses: 4,000 resamples
over units with seed 20260927, and `pair_stats` is imported rather than restated,
so a paired difference computed here is the statistic r31 reports.
"""

import argparse
import importlib.util
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from errctl import paths

FOV_DIR = os.path.join(paths.WORK, "predictions_fov")
ALPHAS = (0.05, 0.10, 0.20)
ARMS = ("plain", "sc3", "sc10", "mond_class")
TAG = "r48b_fov_repair"

_R31 = None


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)),
          flush=True)


def _fov_parse_name(orig):
    pat = re.compile("(.*)__(logreg|rf|hgb)[.]npz$")

    def parse(fname):
        m = pat.match(fname)
        if m is not None and m.group(1).startswith("withinFOV_"):
            return "within", m.group(1)[len("withinFOV") + 1:], m.group(2)
        return orig(fname)
    return parse


def r31():
    """r31, pointed at the field-of-view predictions.

    `parse_name` is patched for the `withinFOV_` stems only, so `within_file`
    runs unchanged and the unit label keeps the image name that
    `scripts/r48_fov_units.py` gave it."""
    global _R31
    if _R31 is None:
        spec = importlib.util.spec_from_file_location(
            "r31_targeted_repair", os.path.join(HERE, "r31_targeted_repair.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.PRED_DIR = FOV_DIR
        mod.parse_name = _fov_parse_name(mod.parse_name)
        _R31 = mod
    return _R31


def _init():
    r31()._worker_init()


def _job(args):
    return r31().within_file(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default=",".join("%.2f" % a for a in ALPHAS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    alphas = tuple(float(a) for a in args.alphas.split(",") if a)
    mod = r31()
    files = sorted(f for f in os.listdir(FOV_DIR) if f.endswith(".npz"))
    if args.limit:
        files = files[:args.limit]
    log("field-of-view files=%d alphas=%s jobs=%d" % (len(files), list(alphas), args.jobs))

    t0 = time.time()
    rows = []
    jobs = [(f, alphas, ()) for f in files]
    if args.jobs > 1:
        with mp.Pool(args.jobs, initializer=_init) as pool:
            for i, rws in enumerate(pool.imap(_job, jobs, chunksize=1), 1):
                rows.extend(rws)
                if i % 20 == 0 or i == len(files):
                    log("  %d/%d units (%.1fs)" % (i, len(files), time.time() - t0))
    else:
        _init()
        for i, j in enumerate(jobs, 1):
            rows.extend(_job(j))
            if i % 20 == 0 or i == len(files):
                log("  %d/%d units (%.1fs)" % (i, len(files), time.time() - t0))

    detail = pd.DataFrame(rows)
    detail.to_csv(os.path.join(paths.RESULTS, TAG + "_detail.csv"), index=False)
    log("part A: %d rows" % len(detail))

    d = detail[detail.part == "A"].copy()
    summ = mod.agg_detail(d, ["arm", "alpha"])
    summ = summ.merge(mod.part_c(d, ["arm", "alpha"]), on=["arm", "alpha"], how="left")
    summ = summ[summ.arm.isin(ARMS)].sort_values(["arm", "alpha"]).reset_index(drop=True)

    def _unit_ci(sub, value):
        v = sub.groupby("unit")[value].mean().to_numpy(dtype=float)
        if not v.size:
            return float("nan"), float("nan")
        rng = np.random.default_rng(mod.BOOT_SEED)
        idx = rng.integers(0, v.size, size=(mod.BOOT, v.size))
        b = v[idx].mean(1)
        return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))

    rows_ci = []
    for _, r in summ.iterrows():
        sub = d[(d.arm == r["arm"]) & np.isclose(d.alpha, r["alpha"])]
        ca_lo, ca_hi = _unit_ci(sub, "covered")
        cn_lo, cn_hi = _unit_ci(sub, "covered_nonvac")
        rows_ci.append({"arm": r["arm"], "alpha": r["alpha"],
                        "coverage_all_lo": ca_lo, "coverage_all_hi": ca_hi,
                        "coverage_nonvac_lo": cn_lo, "coverage_nonvac_hi": cn_hi})
    summ = summ.merge(pd.DataFrame(rows_ci), on=["arm", "alpha"], how="left")
    summ.to_csv(os.path.join(paths.RESULTS, TAG + "_summary.csv"), index=False)

    pairs = [mod.pair_stats(d, arm, "plain", ["unit", "clf", "alpha", "k", "rep"], val)
             for arm in ARMS if arm != "plain"
             for val in ("covered_nonvac", "accept_rate")]
    paired = pd.DataFrame(pairs)
    paired.to_csv(os.path.join(paths.RESULTS, TAG + "_paired.csv"), index=False)

    by_a = []
    for a in alphas:
        da = d[np.isclose(d.alpha, a)]
        for arm in ARMS:
            if arm == "plain":
                continue
            for val in ("covered_nonvac", "accept_rate"):
                r = mod.pair_stats(da, arm, "plain", ["unit", "clf", "k", "rep"], val)
                r["alpha"] = float(a)
                by_a.append(r)
    paired_a = pd.DataFrame(by_a)
    paired_a.to_csv(os.path.join(paths.RESULTS, TAG + "_paired_by_alpha.csv"),
                    index=False)

    lines = []

    def w(s=""):
        lines.append(s)
        log(s)

    w("R48b -- the cross-field-of-view repair arms at field-of-view resolution")
    w("=" * 74)
    w("units: %d ; arms: %s ; budgets: %s"
      % (int(summ.n_units.max()), ", ".join(ARMS),
         ", ".join("%.2f" % a for a in alphas)))
    w("arms imported from scripts/r31_targeted_repair.py; only the units change")
    w("")
    w("%-11s %5s %8s %11s %9s %8s %9s" % (
        "arm", "alpha", "coverage", "cov(nonvac)", "cov(cond)", "accept", "risk_up"))
    for _, r in summ.iterrows():
        w("%-11s %5.2f %8.3f %11.3f %9.3f %8.3f %9.3f" % (
            r["arm"], r["alpha"], r["coverage_all"], r["coverage_nonvac"],
            r["coverage_nonvac_cond"], r["mean_accept"], r["risk_upper"]))
    w("")
    w("paired against the plain certificate, unit bootstrap %d" % mod.BOOT)
    for _, r in paired.iterrows():
        w("  %-11s %-15s mean diff %+.4f  [%+.4f, %+.4f]  %s"
          % (r["arm"], r["value"], r["mean_diff"], r["ci_lo"], r["ci_hi"],
             "excludes zero" if r["excludes_zero"] else "includes zero"))
    w("")
    w("paired against the plain certificate, by budget")
    for _, r in paired_a.iterrows():
        w("  %-11s %.2f %-15s mean diff %+.4f  [%+.4f, %+.4f]  %s"
          % (r["arm"], r["alpha"], r["value"], r["mean_diff"], r["ci_lo"], r["ci_hi"],
             "excludes zero" if r["excludes_zero"] else "includes zero"))
    w("")
    w("verdict")
    for a in alphas:
        g = summ[np.isclose(summ.alpha, a)].set_index("arm")
        if "plain" not in g.index:
            continue
        w("  alpha=%.2f plain coverage %.3f [non-vac %.3f, cond %.3f] at accept %.3f"
          % (a, g.loc["plain", "coverage_all"], g.loc["plain", "coverage_nonvac"],
             g.loc["plain", "coverage_nonvac_cond"], g.loc["plain", "mean_accept"]))
        for arm in ARMS:
            if arm == "plain" or arm not in g.index:
                continue
            w("              %-10s coverage %.3f [non-vac %.3f, cond %.3f] at accept %.3f"
              % (arm, g.loc[arm, "coverage_all"], g.loc[arm, "coverage_nonvac"],
                 g.loc[arm, "coverage_nonvac_cond"], g.loc[arm, "mean_accept"]))
    p = os.path.join(paths.WORK, "r48b_report.txt")
    with open(p, "w", encoding="utf-8", newline=chr(10)) as fh:
        fh.write(chr(10).join(lines) + chr(10))
    log("wrote %s" % p)


if __name__ == "__main__":
    main()
