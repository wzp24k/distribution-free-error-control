"""r47 pipeline driver: run every DATA stage in dependency order.

One driver, one place to see the stages, their real flags, their exit codes and
what each stage does to the released tables in experiments/.

  python -u scripts/run_pipeline.py --list
  python -u scripts/run_pipeline.py --mode smoke [--jobs 10] [--only r03,r09] [--skip r34]
  python -u scripts/run_pipeline.py --mode full  [--jobs 10]

For every stage it writes stdout+stderr to work/logs/<stage>.log, appends a row
to work/pipeline_baseline/stage_status.csv, and diffs that stage's experiments/
outputs against work/pipeline_baseline/experiments_manifest.csv
(NEW / MISSING / CHANGED / UNCHANGED).  A missing driver script is reported and
skipped, never fatal; a failing stage is recorded and the driver continues.

ASCII only, LF only.  Figures stages (r10, r22, r35) are excluded on purpose;
--no-figure / --no-fig / --no-figs is passed wherever a stage offers it.
r43_threadpin is owned by a concurrent worker and is not a stage here.
"""
import argparse
import csv
import hashlib
import io
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGDIR = os.path.join(ROOT, "work", "logs")
BASE = os.path.join(ROOT, "work", "pipeline_baseline")
MANIFEST = os.path.join(BASE, "experiments_manifest.csv")
STATUS = os.path.join(BASE, "stage_status.csv")
DIFFS = os.path.join(BASE, "stage_diffs.txt")
FOREIGN = ("r43_",)  # files owned by another worker: excluded from stage diffs

# ---------------------------------------------------------------------------
# Stage table, dependency order.  "args" is the documented full run; "smoke" is
# the reduced, known-good variant.  {jobs} is substituted from --jobs.
# Flags were read from each script's own argparse block; where the README and
# the script disagree the script wins (see work/r47_pipeline_report.txt).  Three
# of them did not reproduce the released tables when the driver was first run
# end to end, because the released artifacts came from longer invocations than
# the script defaults: r03_analyse (eight alphas, not three), r14_targetcal
# (ten replicates, not the script default fifteen) and r14b_targetadapt (the
# k grid 200/1000/2000 at two replicates).  Round 28 of
# docs/manuscript_number_audit.md records the re-run that established this.
# ---------------------------------------------------------------------------
STAGES = [
    dict(name="r01_build_datasets", script="r01_build_datasets.py",
         args=[], smoke=[],
         out=[], side=["work/datasets/*.parquet", "work/r01_summary.csv"]),
    dict(name="r02_run_experiments", script="r02_run_experiments.py",
         args=["--units", "within,transfer", "--clfs", "logreg,rf,hgb",
               "--norm", "z_dataset", "--seed", "0"],
         smoke=["--units", "within", "--clfs", "logreg", "--only", "Breast"],
         out=[], side=["work/r02_runs.csv", "work/predictions/within_*",
                       "work/predictions/transfer_*"],
         note="script has NO --jobs (README says --jobs 8)"),
    dict(name="r05_cellsplit", script="r05_cellsplit.py",
         args=["--clfs", "logreg,rf,hgb", "--norm", "z_dataset", "--seed", "0"],
         smoke=["--clfs", "logreg"],
         out=[], side=["work/r05_runs.csv", "work/predictions/cellsplit_*"],
         note="script has NO --jobs (README says --jobs 8)"),
    dict(name="r03_analyse", script="r03_analyse.py",
         args=["--alphas", "0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50",
               "--jobs", "{jobs}"],
         smoke=["--jobs", "{jobs}", "--limit", "Breast"],
         out=["experiments/r03_*.csv"],
         note="the script default is three alphas; the released table is the "
              "eight-alpha grid (Round 28)"),
    dict(name="r04_report", script="r04_report.py",
         args=[], smoke=[], out=["experiments/r04_*.csv",
                                 "experiments/r04_report.txt"]),
    dict(name="r06_marker_audit", script="r06_marker_audit.py",
         args=["--alphas", "0.05,0.10,0.20", "--clfs", "logreg,rf,hgb"],
         smoke=["--alphas", "0.10", "--clfs", "logreg"],
         out=["experiments/r06_*.csv"],
         note="script has NO --jobs (README says --jobs 8)"),
    dict(name="r07_ablations", script="r07_ablations.py",
         args=["--jobs", "{jobs}"], smoke=["--jobs", "{jobs}", "--limit", "Breast"],
         out=["experiments/r07_*.csv"]),
    dict(name="r08_panel", script="r08_panel.py",
         args=["--jobs", "{jobs}"], smoke=["--jobs", "{jobs}", "--limit", "Breast"],
         out=["experiments/r08_*.csv"]),
    dict(name="r09_simulation", script="r09_simulation.py",
         args=["--reps", "4000", "--jobs", "{jobs}"],
         smoke=["--reps", "20", "--jobs", "{jobs}"],
         out=["experiments/r09_*.csv"]),
    dict(name="r11_ensemble", script="r11_ensemble.py",
         args=["--jobs", "{jobs}"], smoke=["--jobs", "{jobs}", "--kinds", "logreg"],
         out=["experiments/r11_*.csv"]),
    dict(name="r12_labelnoise", script="r12_labelnoise.py",
         args=["--jobs", "{jobs}"], smoke=["--jobs", "{jobs}", "--reps", "2",
                                           "--limit", "Breast"],
         out=["experiments/r12_*.csv"]),
    dict(name="r13_coverage_ci", script="r13_coverage_ci.py",
         args=[], smoke=[], out=["experiments/r13_*.csv"]),
    dict(name="r14_targetcal", script="r14_targetcal.py",
         args=["--reps", "10", "--jobs", "{jobs}"],
         smoke=["--jobs", "{jobs}", "--reps", "1", "--only",
                "Breast_Jun0524->CRC_Feb2623"],
         out=["experiments/r14_targetcal.csv",
              "experiments/r14_targetcal_summary.csv"]),
    dict(name="r14b_targetadapt", script="r14_targetcal.py",
         args=["--adapt", "--adapt-ks", "200,1000,2000", "--reps", "2",
               "--out", "r14b_targetadapt", "--jobs", "{jobs}"],
         smoke=["--adapt", "--adapt-ks", "200,1000,2000", "--out",
                "r14b_targetadapt", "--jobs", "{jobs}",
                "--reps", "1", "--only", "Breast_Jun0524->CRC_Feb2623"],
         out=["experiments/r14b_targetadapt.csv",
              "experiments/r14b_targetadapt_summary.csv"],
         note="r14b outputs come from r14_targetcal.py --adapt --out "
              "r14b_targetadapt; the released k grid is 200/1000/2000 at two "
              "replicates, not the script default (Round 28)"),
    dict(name="r14_targetcal_ext", script="r14_targetcal.py",
         args=["--ks", "0,25,50,100,200,250,500,750,1000,1500,2000",
               "--reps", "10", "--out", "r14_targetcal_ext", "--jobs", "{jobs}"],
         smoke=["--ks", "0,500", "--reps", "1", "--out", "r14_targetcal_ext",
                "--jobs", "{jobs}", "--only", "Breast_Jun0524->CRC_Feb2623"],
         out=["experiments/r14_targetcal_ext.csv",
              "experiments/r14_targetcal_ext_summary.csv"],
         note="the dense ladder behind Table 6 (r14_targetcal.py --ks ... --out "
              "r14_targetcal_ext)"),
    dict(name="r14c_ladder_dense", script="r14c_ladder_dense.py",
         args=[], smoke=[],
         out=["experiments/r14c_ladder_dense.csv"],
         note="reads experiments/r14_targetcal_ext.csv; asserts the coarse grid "
              "against Table 6 before reporting the dense one"),
    dict(name="r15_marker_sensitivity", script="r15_marker_sensitivity.py",
         args=["--jobs", "{jobs}", "--boot", "2000", "--no-figure"],
         smoke=["--jobs", "{jobs}", "--boot", "50", "--clfs", "logreg",
                "--percentiles", "90", "--no-figure"],
         out=["experiments/r15_*.csv"]),
    dict(name="r16_review_budget", script="r16_review_budget.py",
         args=["--jobs", "{jobs}"], smoke=["--jobs", "{jobs}", "--limit", "Breast"],
         out=["experiments/r16_*.csv"]),
    dict(name="r17_simulation", script="r17_simulation.py",
         args=["--reps", "800", "--jobs", "{jobs}"],
         smoke=["--reps", "20", "--jobs", "{jobs}"],
         out=["experiments/r17_*.csv"]),
    dict(name="r18_marker_sources", script="r18_marker_sources.py",
         args=["--jobs", "{jobs}", "--boot", "2000", "--no-figure"],
         smoke=["--jobs", "{jobs}", "--inner-jobs", "1", "--boot", "50",
                "--no-figure"],
         out=["experiments/r18_*.csv"]),
    dict(name="r19_weight_shift_diag", script="r19_weight_shift_diag.py",
         args=["--jobs", "{jobs}", "--no-fig"],
         smoke=["--jobs", "{jobs}", "--no-fig", "--folds", "0", "--only",
                "Breast_Jun0524->CRC_Feb2623"],
         out=["experiments/r19_*.csv"]),
    dict(name="r20_label_ladder", script="r20_label_ladder.py",
         args=["--jobs", "{jobs}", "--no-figs"],
         smoke=["--jobs", "{jobs}", "--no-figs", "--reps", "2",
                "--limit", "Breast"],
         out=["experiments/r20_*.csv"]),
    dict(name="r21_downstream_repair", script="r21_downstream_repair.py",
         args=["--no-figure"], smoke=["--no-figure", "--boot", "50"],
         out=["experiments/r21_*.csv"],
         note="script has NO --jobs (README no longer suggests one)"),
    dict(name="r24_efficiency", script="r24_efficiency.py",
         args=["--jobs", "{jobs}", "--no-figure"],
         smoke=["--jobs", "{jobs}", "--boot", "50", "--no-figure"],
         out=["experiments/r24_efficiency.csv"]),
    dict(name="r24b_labelshift", script="r24b_labelshift.py",
         args=["--jobs", "{jobs}"],
         smoke=["--jobs", "{jobs}", "--folds", "0"],
         out=["experiments/r24b_*.csv"]),
    dict(name="r25_accept_deferred", script="r25_accept_deferred.py",
         args=["--jobs", "{jobs}", "--no-figure"],
         smoke=["--jobs", "{jobs}", "--smoke"],
         out=["experiments/r25_*.csv"]),
    dict(name="r26_plugin_baseline", script="r26_plugin_baseline.py",
         args=["--jobs", "{jobs}"], smoke=["--jobs", "{jobs}", "--limit", "Breast"],
         out=["experiments/r26_*.csv"]),
    dict(name="r27_fixedseq", script="r27_fixedseq.py",
         args=["--jobs", "{jobs}"], smoke=["--jobs", "{jobs}", "--limit", "Breast"],
         out=["experiments/r27_*.csv"]),
    dict(name="r28_marker_bounds", script="r28_marker_bounds.py",
         args=[], smoke=[], out=["experiments/r28_*.csv"]),
    dict(name="r31_targeted_repair", script="r31_targeted_repair.py",
         args=["--jobs", "{jobs}"],
         smoke=["--jobs", "{jobs}", "--reps", "2", "--only", "Breast",
                "--limit", "1"],
         out=["experiments/r31_*.csv"]),
    dict(name="r32_sequential", script="r32_sequential.py",
         args=["--procs", "{jobs}"], smoke=["--procs", "{jobs}", "--limit", "1",
                                            "--quick"],
         out=["experiments/r32_*.csv"],
         note="parallelism flag is --procs, not --jobs"),
    dict(name="r33_two_source", script="r33_two_source.py",
         args=["--boot", "200", "--boot-ci", "120", "--rand", "8000",
               "--lca-boot", "200", "--lca-rand", "8", "--lca-rand-ci", "6",
               "--pool", "24"],
         smoke=["--boot", "8", "--boot-ci", "8", "--rand", "500",
                "--lca-boot", "60", "--lca-rand", "3", "--lca-rand-ci", "3",
                "--pool", "4"],
         out=["experiments/r33_*.csv"],
         note="full flags copied from work/_r33_full.cmd (README lists none)"),
    dict(name="r34_external", script="r34_external.py",
         args=["--stage", "all"], smoke=["--stage", "all", "--skip-runs"],
         out=["experiments/r34_*.csv"],
         note="released r34 came from two runs (default + --stage phase2); "
              "--stage all is the same output set"),
    dict(name="r44_gvhd_delivery", script="r44_gvhd_delivery.py",
         args=[], smoke=[],
         out=["experiments/r44_gvhd_delivery*.csv"],
         side=["work/r44_delivery_report.txt"],
         note="the GVHD delivery defect: do the shipped segmentation masks "
              "cover the shipped labels?  --check exits non-zero if the "
              "recomputed headline moves"),
    dict(name="r61_shared_panel", script="r61_shared_panel.py",
         args=["--jobs", "{jobs}"], smoke=["--jobs", "{jobs}", "--from-csv"],
         out=["experiments/r61_shared_panel*.csv"],
         note="--pair and --tag select the cohort pair; the melanoma/GVHD pair is "
              "--pair Melanoma_Oct2522,GVHD_Feb1323 --tag _mg"),
    dict(name="r62_source_family", script="r62_source_family.py",
         args=["--jobs", "{jobs}"], smoke=["--jobs", "{jobs}", "--from-csv"],
         out=["experiments/r62_source_family*.csv",
              "experiments/r62_fold_transfer*.csv"],
         note="the annotation-source family; only the two cohorts that ship a "
              "second label source are used, and every arm is read against both"),
    dict(name="r68_equal_guarantee", script="r68_equal_guarantee.py",
         args=[], smoke=[],
         out=["experiments/r68_equal_guarantee.csv"],
         note="the matched-guarantee reading of the review budget, the companion "
              "of the matched-realised-risk table; reads the released metrics table "
              "and writes the tidy CSV the manuscript table recomputes from"),
    dict(name="r69_external2", script="r69_external2.py",
         args=["--jobs", "{jobs}"], smoke=[],
         out=["experiments/r69_external2_*.csv"],
         note="two further independent cohorts on two further platforms plus a "
              "real antibody panel swap onto the depot; the per-cell tables are "
              "retrieved from Zenodo into data/external2/, and --report-only "
              "rewrites the checks and the report from the released tidy CSV"),
    dict(name="r63_reproduce_check", script="r63_reproduce_check.py",
         args=[], smoke=[],
         out=[],
         note="reads only the released tables and re-derives the four negative "
              "results, the cross-field-of-view shortfall and the one positive "
              "replication against fixed expected values; writes nothing and "
              "exits non-zero on any mismatch"),
]


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def line_count(path, chunk=1 << 20):
    n = 0
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            n += b.count(b"\n")
    if size > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                n += 1
    return n


def rows_of(path):
    n = line_count(path)
    return max(0, n - 1) if path.lower().endswith(".csv") else n


def load_manifest():
    m = {}
    with io.open(MANIFEST, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m[row["file"]] = row
    return m


def rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def expand(patterns):
    out = []
    for pat in patterns:
        pat = pat.replace("/", os.sep)
        d, base = os.path.split(pat)
        d = os.path.join(ROOT, d) if d else ROOT
        if not os.path.isdir(d):
            continue
        import fnmatch
        for fn in sorted(os.listdir(d)):
            if fnmatch.fnmatch(fn, base):
                out.append(os.path.join(d, fn))
    return out


def stage_manifest_rows(stage, manifest):
    """Manifest rows whose repo-relative path matches this stage's globs."""
    import fnmatch
    hit = {}
    for pat in stage["out"]:
        pp = pat.replace(os.sep, "/").split("/")
        for r, row in manifest.items():
            rr = r.split("/")
            if len(rr) == len(pp) and all(fnmatch.fnmatch(a, b)
                                         for a, b in zip(rr, pp)):
                hit[r] = row
    return hit


def compare(stage, manifest):
    new, missing, changed, unchanged = [], [], [], []
    detail = []
    for p in expand(stage["out"]):
        r = rel(p)
        cur = md5(p)
        if r not in manifest:
            new.append(r)
            detail.append("NEW      %s (%d bytes, %d rows)"
                          % (r, os.path.getsize(p), rows_of(p)))
        elif cur != manifest[r]["md5"]:
            changed.append(r)
            detail.append("CHANGED  %s (md5 %s -> %s, bytes %s -> %d)"
                          % (r, manifest[r]["md5"], cur, manifest[r]["bytes"],
                             os.path.getsize(p)))
        else:
            unchanged.append(r)
    for r, row in stage_manifest_rows(stage, manifest).items():
        full = os.path.join(ROOT, r.replace("/", os.sep))
        if not os.path.exists(full):
            missing.append(r)
            detail.append("MISSING  %s (baseline %s bytes, %s rows)"
                          % (r, row["bytes"], row["rows"]))
    side = expand(stage.get("side", []))
    return new, missing, changed, unchanged, detail, side


def run_stage(stage, mode, jobs, manifest, rows_out):
    script = os.path.join(ROOT, "scripts", stage["script"])
    tpl = stage["args"] if mode == "full" else stage["smoke"]
    args = [a.replace("{jobs}", str(jobs)) for a in tpl]
    rel_script = os.path.join("scripts", stage["script"])
    cmd = [sys.executable, "-u", rel_script] + args
    log_path = os.path.join(LOGDIR, stage["name"] + ".log")
    row = dict(stage=stage["name"], script=stage["script"],
               command=" ".join(["python", "-u",
                                os.path.join("scripts", stage["script"])] + args),
               exit_code="", seconds="", new="", missing="", changed="",
               unchanged="", verdict="")
    if not os.path.exists(script):
        row["verdict"] = "SKIP"
        row["exit_code"] = "n/a"
        row["seconds"] = "0.0"
        row["unchanged"] = "driver script missing: scripts/%s" % stage["script"]
        print("[SKIP ] %-24s driver script does not exist" % stage["name"],
              flush=True)
        rows_out.append(row)
        return row
    t0 = time.time()
    print("[start] %-24s %s" % (stage["name"], " ".join(args) or "(no flags)"),
          flush=True)
    with open(log_path, "wb") as lf:
        lf.write(("CMD: python -u %s\nCWD: %s\n\n"
                  % (" ".join([rel_script] + args), ROOT)).encode("ascii"))
        lf.flush()
        ok = subprocess.Popen(cmd, cwd=ROOT, stdout=lf,
                              stderr=subprocess.STDOUT).wait()
    secs = time.time() - t0
    new, missing, changed, unchanged, detail, side = compare(stage, manifest)
    diff_path = os.path.join(LOGDIR, stage["name"] + ".compare.txt")
    body = ["stage: %s  mode: %s  exit: %s  seconds: %.1f"
            % (stage["name"], mode, ok, secs),
            "command: %s" % row["command"],
            "outputs compared: %d  new %d  missing %d  changed %d  unchanged %d"
            % (len(new) + len(missing) + len(changed) + len(unchanged),
               len(new), len(missing), len(changed), len(unchanged)),
            "side outputs present: %d" % len(side)]
    body += detail
    if stage.get("note"):
        body.insert(1, "note: %s" % stage["note"])
    io.open(diff_path, "w", encoding="utf-8", newline="").write(
        "\n".join(body) + "\n")
    if ok == 0 and not missing:
        verdict = "PASS(diff)" if (new or changed) else "PASS"
    elif ok == 0:
        verdict = "PASS-MISSING"
    else:
        verdict = "FAIL"
    row.update(exit_code=ok, seconds="%.1f" % secs, new=len(new),
               missing=len(missing), changed=len(changed),
               unchanged=len(unchanged), verdict=verdict)
    rows_out.append(row)
    print("[%s] %-24s exit=%s %.1fs  new=%d missing=%d changed=%d unchanged=%d"
          % ("done " if ok == 0 else "FAIL ", stage["name"], ok, secs,
             len(new), len(missing), len(changed), len(unchanged)), flush=True)
    return row


def write_status(rows):
    cols = ["stage", "script", "command", "exit_code", "seconds", "new",
            "missing", "changed", "unchanged", "verdict"]
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = str(r.get(c, ""))
            if "," in v or '"' in v:
                v = '"' + v.replace('"', '""') + '"'
            vals.append(v)
        lines.append(",".join(vals))
    io.open(STATUS, "w", encoding="utf-8", newline="").write("\n".join(lines) + "\n")


def write_diffs(rows, mode, jobs):
    lines = ["stage diff detail (mode=%s jobs=%s)" % (mode, jobs), "=" * 70]
    for r in rows:
        p = os.path.join(LOGDIR, r["stage"] + ".compare.txt")
        if os.path.exists(p):
            lines.append("")
            lines.append(io.open(p, encoding="utf-8").read().rstrip())
    io.open(DIFFS, "w", encoding="utf-8", newline="").write("\n".join(lines) + "\n")


SNAPSHOT = os.path.join(BASE, "exp_snapshot")
DRIFT_CSV = os.path.join(BASE, "drift_report.csv")
DRIFT_TXT = os.path.join(BASE, "drift_report.txt")
KEY_PRIORITY = ["unit", "family", "dataset", "arm", "method", "clf", "alpha",
                "k", "fold", "scope", "source", "cls", "level", "metric",
                "regime", "eta", "rep", "positivity", "reading", "variant",
                "threshold", "pair", "row", "setting", "name", "id"]


def snap_path(r):
    """Snapshot copy of a manifest entry: strip the top-level dir (experiments/)."""
    rel2 = r.split("/", 1)[1] if "/" in r else r
    return os.path.join(SNAPSHOT, rel2.replace("/", os.sep))


def _rowkey(row, cols):
    parts = []
    for c in cols:
        v = row.get(c, "")
        parts.append("%s=%s" % (c, v))
    return ";".join(parts) or "(no key columns)"


def numeric_drift(path_a, path_b):
    """Worst absolute / relative numeric difference between two released CSVs."""
    import pandas as pd
    import numpy as np
    info = dict(rows_a="", rows_b="", cols_a="", cols_b="", n_cells_compared="",
                n_cells_differ="", order_same="", worst_abs="", worst_abs_col="",
                worst_abs_key="", worst_rel="", worst_rel_col="",
                worst_rel_key="", note="")
    try:
        a = pd.read_csv(path_a)
        b = pd.read_csv(path_b)
    except Exception as exc:
        info["note"] = "not a readable CSV pair (%r)" % (exc,)
        return info
    info["rows_a"] = len(a)
    info["rows_b"] = len(b)
    info["cols_a"] = a.shape[1]
    info["cols_b"] = b.shape[1]
    ka = [c for c in KEY_PRIORITY if c in a.columns]
    if sorted(a.columns.tolist()) != sorted(b.columns.tolist()):
        info["note"] = "column sets differ"
        return info
    # A released table's content does not include the order of its rows.  The
    # snapshot was taken before every stage collected its workers in submission
    # order, so comparing positionally pairs unrelated rows and reports a
    # difference the size of a whole n_accepted column.  Both frames are
    # canonicalised on their non-numeric columns first, and whether the raw
    # order matched is recorded rather than assumed.
    _cats = [c for c in a.columns
             if not (pd.api.types.is_numeric_dtype(a[c])
                     and pd.api.types.is_numeric_dtype(b[c]))]
    _order = _cats + [c for c in a.columns if c not in _cats]
    _aa = a.sort_values(_order, kind="mergesort").reset_index(drop=True)
    _bb = b.sort_values(_order, kind="mergesort").reset_index(drop=True)
    info["order_same"] = "yes" if (a.equals(_aa) and b.equals(_bb)) else "no"
    a, b = _aa, _bb
    if len(a) != len(b):
        info["note"] = "row counts differ; the shorter table was truncated"
    n = min(len(a), len(b))
    a = a.iloc[:n].reset_index(drop=True)
    b = b.iloc[:n].reset_index(drop=True)
    worst_abs, wac, wak = -1.0, "", ""
    worst_rel, wrc, wrk = -1.0, "", ""
    nc = nd = 0
    for c in a.columns:
        if not (pd.api.types.is_numeric_dtype(a[c])
                and pd.api.types.is_numeric_dtype(b[c])):
            continue
        x = a[c].to_numpy(dtype="float64")
        y = b[c].to_numpy(dtype="float64")
        d = np.abs(x - y)
        both_nan = np.isnan(x) & np.isnan(y)
        d = np.where(both_nan, 0.0, d)
        nc += int(np.isfinite(d).sum())
        nd += int((d > 0).sum())
        if len(d):
            with np.errstate(invalid="ignore"):
                j = int(np.nanargmax(np.where(np.isnan(d), -1.0, d)))
            if np.isfinite(d[j]) and d[j] > worst_abs:
                worst_abs, wac = float(d[j]), c
                wak = _rowkey(a.iloc[j], ka)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(np.abs(x) > 0, d / np.abs(x), np.nan)
        if np.isfinite(rel).any():
            j = int(np.nanargmax(np.where(np.isnan(rel), -1.0, rel)))
            if rel[j] > worst_rel:
                worst_rel, wrc = float(rel[j]), c
                wrk = _rowkey(a.iloc[j], ka)
    info["n_cells_compared"] = nc
    info["n_cells_differ"] = nd
    info["worst_abs"] = ("%.17g" % worst_abs) if worst_abs >= 0 else ""
    info["worst_abs_col"] = wac
    info["worst_abs_key"] = wak
    info["worst_rel"] = ("%.17g" % worst_rel) if worst_rel >= 0 else ""
    info["worst_rel_col"] = wrc
    info["worst_rel_key"] = wrk
    return info


def _pid_alive(pid):
    """Windows-safe liveness check (os.kill(pid, 0) would TERMINATE on Windows)."""
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong(0)
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == STILL_ACTIVE
    except Exception:
        return False


def reconcile(dry=False, wait_pid=0):
    """Drift discipline: the audited released CSVs must survive a re-run.

    For every file in the baseline manifest compare md5, then (for changed
    files) row counts and every numeric column against the snapshot.  A file
    that is byte-identical is left alone; a CHANGED file is restored from the
    snapshot and its drift is recorded.  experiments/r43_* is never touched
    (owned by a concurrent worker) and is reported as foreign.
    """
    if wait_pid:
        print("[reconcile] waiting for pid %d to exit ..." % wait_pid, flush=True)
        while _pid_alive(wait_pid):
            time.sleep(10)
        print("[reconcile] pid %d has exited" % wait_pid, flush=True)
    man = load_manifest()
    ok, changed, restored, gone = 0, 0, 0, 0
    rows_out = []
    print("[reconcile] %d baseline files, dry=%s" % (len(man), dry), flush=True)
    for r, row in man.items():
        full = os.path.join(ROOT, r.replace("/", os.sep))
        snap = snap_path(r)
        if r.startswith("experiments/r43_"):
            rows_out.append(dict(file=r, status="FOREIGN-SKIPPED",
                                 restored="no", note="owned by another worker"))
            continue
        if not os.path.exists(full):
            d = dict(file=r, status="MISSING", restored="no" if dry else "yes",
                     note="file was deleted; restored from snapshot")
            rows_out.append(d)
            gone += 1
            if not dry and os.path.exists(snap):
                import shutil
                shutil.copy2(snap, full)
            continue
        cur = md5(full)
        if cur == row["md5"]:
            ok += 1
            continue
        changed += 1
        info = numeric_drift(snap, full)
        d = dict(file=r, status="CHANGED",
                 bytes_snapshot=row["bytes"], bytes_now=os.path.getsize(full),
                 md5_snapshot=row["md5"], md5_now=cur,
                 restored="no" if dry else "yes")
        d.update({k: info[k] for k in info})
        if not dry and os.path.exists(snap):
            import shutil
            shutil.copy2(snap, full)
            restored += 1
        rows_out.append(d)
        print("[reconcile] %s: CHANGED order_same=%s (worst abs %s at %s; "
              "worst rel %s at %s)"
              % (r, info["order_same"], info["worst_abs"], info["worst_abs_col"],
                 info["worst_rel"], info["worst_rel_col"]), flush=True)
    cols = ["file", "status", "restored", "bytes_snapshot", "bytes_now",
            "md5_snapshot", "md5_now", "rows_a", "rows_b", "cols_a", "cols_b",
            "n_cells_compared", "n_cells_differ", "order_same", "worst_abs",
            "worst_abs_col", "worst_abs_key", "worst_rel", "worst_rel_col",
            "worst_rel_key", "note"]
    lines = [",".join(cols)]
    for d in rows_out:
        vals = []
        for c in cols:
            v = str(d.get(c, ""))
            if "," in v or '"' in v:
                v = '"' + v.replace('"', '""') + '"'
            vals.append(v)
        lines.append(",".join(vals))
    io.open(DRIFT_CSV, "w", encoding="utf-8", newline="").write("\n".join(lines) + "\n")
    txt = ["drift reconciliation (dry=%s)" % dry,
           "method                         : each changed file is re-read, its rows "
           "are aligned on a canonical sort of the non-numeric columns (a released "
           "table has no canonical row order), and every numeric cell is then "
           "compared position by position -- so a reported difference is a value "
           "difference, never an artefact of which worker finished first",
           "byte-identical, left in place : %d" % ok,
           "CHANGED                        : %d" % changed,
           "restored from snapshot         : %d" % restored,
           "missing (restored)             : %d" % gone,
           "foreign r43_* (never touched)  : %d"
           % sum(1 for d in rows_out if d["status"] == "FOREIGN-SKIPPED"),
           ""]
    for d in rows_out:
        if d["status"] in ("CHANGED", "MISSING"):
            txt.append("%-40s %-8s restored=%s" % (d["file"], d["status"],
                                                   d.get("restored")))
            if d["status"] == "CHANGED":
                txt.append("    rows %s -> %s   numeric cells compared %s, differing %s"
                           % (d.get("rows_a"), d.get("rows_b"),
                              d.get("n_cells_compared"), d.get("n_cells_differ")))
                txt.append("    row order %s (canonical sort applied before the "
                           "comparison)"
                           % ("unchanged"
                              if d.get("order_same") == "yes" else "DIFFERED"))
                txt.append("    worst ABS diff %s at column %r  (row key: %s)"
                           % (d.get("worst_abs"), d.get("worst_abs_col"),
                              d.get("worst_abs_key")))
                txt.append("    worst REL diff %s at column %r  (row key: %s)"
                           % (d.get("worst_rel"), d.get("worst_rel_col"),
                              d.get("worst_rel_key")))
    io.open(DRIFT_TXT, "w", encoding="utf-8", newline="").write("\n".join(txt) + "\n")
    print("[reconcile] identical %d, changed %d, restored %d, missing %d"
          % (ok, changed, restored, gone), flush=True)
    print("[reconcile] wrote %s and %s" % (DRIFT_CSV, DRIFT_TXT), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--only", default="",
                    help="comma list of stages (name or rNN short form)")
    ap.add_argument("--skip", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop starting new stages once this many seconds have "
                         "elapsed (the running stage is never interrupted), then "
                         "reconcile")
    ap.add_argument("--reconcile-only", action="store_true",
                    help="only run the drift reconciliation, then exit")
    ap.add_argument("--reconcile-dry", action="store_true",
                    help="compare against the snapshot but restore nothing")
    ap.add_argument("--wait-pid", type=int, default=0,
                    help="wait for this pid to exit before reconciling")
    args = ap.parse_args()

    if args.list:
        for s in STAGES:
            print("%-24s %-26s %s" % (s["name"], s["script"],
                                      " ".join(s["args"]) or "(no flags)"))
            if s.get("note"):
                print("%-24s NOTE: %s" % ("", s["note"]))
        return 0

    if args.reconcile_only:
        return reconcile(dry=args.reconcile_dry, wait_pid=args.wait_pid)

    def toks(s):
        return [t.strip() for t in s.split(",") if t.strip()]

    def selected(stage):
        short = stage["name"].split("_")[0]
        if toks(args.only) and not any(t == stage["name"] or t == short or
                                       stage["name"].startswith(t + "_")
                                       for t in toks(args.only)):
            return False
        if any(t == stage["name"] or t == short or
               stage["name"].startswith(t + "_") for t in toks(args.skip)):
            return False
        return True

    for d in (LOGDIR, BASE):
        if not os.path.isdir(d):
            os.makedirs(d)
    manifest = load_manifest()
    print("driver: %s" % os.path.abspath(__file__))
    print("root  : %s" % ROOT)
    print("mode  : %s   jobs: %d   baseline manifest: %d files"
          % (args.mode, args.jobs, len(manifest)), flush=True)
    rows, t0 = [], time.time()
    capped = False
    for stage in STAGES:
        if not selected(stage):
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            capped = True
            print("[cap  ] time budget of %.0fs reached before %s - not starting "
                  "any further stage" % (args.max_seconds, stage["name"]),
                  flush=True)
            break
        run_stage(stage, args.mode, args.jobs, manifest, rows)
        write_status(rows)
    write_diffs(rows, args.mode, args.jobs)
    npass = sum(1 for r in rows if str(r["verdict"]).startswith("PASS"))
    nfail = sum(1 for r in rows if r["verdict"] == "FAIL")
    nskip = sum(1 for r in rows if r["verdict"] == "SKIP")
    if capped:
        print("\n[cap  ] stopped by --max-seconds; stages after %s were not reached"
              % (rows[-1]["stage"] if rows else "(none)"), flush=True)
        io.open(os.path.join(BASE, "not_reached.txt"), "w", encoding="utf-8",
                newline="").write("\n".join(
                    s2["name"] for s2 in STAGES
                    if s2["name"] not in [r["stage"] for r in rows]) + "\n")
    print("\nsummary: %d stages, PASS %d, FAIL %d, SKIP %d, total %.1fs"
          % (len(rows), npass, nfail, nskip, time.time() - t0), flush=True)
    print("status : %s" % STATUS)
    print("diffs  : %s" % DIFFS)
    if args.mode == "full" and not toks(args.only) and not args.skip:
        print("", flush=True)
        print("[reconcile] full run complete - enforcing drift discipline",
              flush=True)
        reconcile(dry=args.reconcile_dry)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
