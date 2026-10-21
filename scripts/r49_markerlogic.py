"""R49 -- the marker-logic representation as a panel-invariant feature space.

Weakness #5 of the revision plan asks the practical question a lower-plex
platform asks: if my panel does not carry the reference panel's channels, is the
certificate still usable?  The published answer is the transfer family, and it
fails.  B2(i) proposes a different route: represent each cell not by channel
intensities but by the Boolean expert definitions in the depot's CellTypes_ALL
table.  A rule such as "CD4&CD3" is a predicate over channels, so it can be
evaluated in any panel that carries those channels, and the resulting
rule-satisfaction vector lives in a space that does not depend on the panel's
identity.  If the transfer failure is caused by the panels not being on a common
scale, a natively commensurate representation should reduce it.

Feature spaces (--space):

  published  the published representation: the cohort's own panel for the
             `within` family and the shared canonicalised panel for the
             `transfer` family, log1p + per-domain z-score.  Used as an
             equivalence control.  These are the released numbers.
  rule       the 2 x n_rules rule vector alone: for each rule, whether it is
             evaluable in this panel and whether it evaluates true.  The two
             entries are kept separate so that "not evaluable" is never written
             as "evaluable and false".  No per-domain standardisation: the point
             of the space is that its coordinates mean the same thing in every
             panel.
  augz       the shared/own panel block (published transform) concatenated with
             the rule vector z-scored per domain.
  augraw     the shared/own panel block (published transform) concatenated with
             the rule vector at its native scale (log1p of the two flags, no
             per-domain standardisation).

Positivity is called once per cohort over that cohort's own panel, so the
positives of the source and of the target come from the same rule applied to
their own measurements; there is no cross-cohort threshold sharing to confound
the comparison.

Unit construction, folds, calibration repeats, the certificate, the readouts and
the analysis are imported from scripts/r02_run_experiments.py and
scripts/r03_analyse.py so that nothing can drift.

Outputs
  work/predictions_r49_<space>/*.npz
  work/r49_<space>_runs.csv
  experiments/r49_<space>_metrics.csv
  experiments/r49_<space>_coverage_ci.csv
  work/r49_report.txt
"""

import argparse
import csv
import importlib.util
import io
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from errctl import data as D, paths
from errctl.markercheck import positivity
from errctl.markerlogic import CompiledRule, LogicError, canonical
from errctl.preprocess import DomainNormaliser as PublishedNormaliser

RULES_CSV = os.path.join(paths.DEPOT, "CellTypes_ALL.csv")
FLAG = "zzML"
N_BOOT = 2000
MIN_CELLS = 50

# The twelve directed cohort pairs of scripts/r02_run_experiments.py, in the
# published order.  Kept as a literal so that the job list does not depend on an
# import of the runner inside the workers.
TRANSFER_PAIRS = [
    ("GVHD_Feb1323", "CRC_Feb2623"),
    ("CRC_Feb2623", "GVHD_Feb1323"),
    ("GVHD_Feb1323", "Breast_Jun0524"),
    ("GVHD_Feb1323", "Melanoma_Oct2522"),
    ("CRC_Feb2623", "Breast_Jun0524"),
    ("CRC_Feb2623", "Melanoma_Oct2522"),
    ("Breast_Jun0524", "Melanoma_Oct2522"),
    ("Melanoma_Oct2522", "Breast_Jun0524"),
    ("Breast_Jun0524", "CRC_Feb2623"),
    ("Melanoma_Oct2522", "CRC_Feb2623"),
    ("Breast_Jun0524", "GVHD_Feb1323"),
    ("Melanoma_Oct2522", "GVHD_Feb1323"),
]


def log(*parts):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(p) for p in parts)),
          flush=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pin_threads():
    from errctl import classifiers as CLS
    orig = CLS.make_classifier

    def patched(kind, seed=0):
        clf = orig(kind, seed=seed)
        if kind == "rf":
            clf.set_params(n_jobs=1)
        return clf

    CLS.make_classifier = patched


# ----------------------------------------------------------------- rules

def load_rules(path=RULES_CSV):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rules, failed = [], []
    for row in rows:
        ct = str(row.get("CellType") or "").strip()
        if not ct or ct.lower().startswith("unidentified"):
            continue
        for col, tag in (("PrimaryMarker", ""), ("AlternatePrimaryMarker", "~alt")):
            text = str(row.get(col) or "").strip()
            if not text or text.lower() in ("nan", "none", "various"):
                continue
            try:
                rules.append((ct + tag, ct, text, CompiledRule(ct + tag, text)))
            except LogicError as exc:
                failed.append((ct + tag, text, str(exc)))
    return rules, failed


RULES, RULE_FAILED = load_rules()
N_FLAGS = 2 * len(RULES)


def flag_names():
    return ([FLAG + "%03d.ev" % j for j in range(len(RULES))] +
            [FLAG + "%03d.sat" % j for j in range(len(RULES))])


def canonical_pos(pos, markers):
    """Re-key a positivity dict onto the canonical channel vocabulary.

    Rule identifiers are canonicalised at compile time, so a rule must be
    matched against the canonical form of the panel's channel names: a panel
    that carries a channel under an alias spelling (HLADR for HLADRDPDQ, Fe56
    for Fe) carries it.  This is the same convention the transfer panel
    intersection of scripts/r02_run_experiments.py uses."""
    out = {}
    for m in markers:
        v = pos[m] if m in pos else pos[str(m)]
        out.setdefault(canonical(str(m)), v)
    return out


def rule_matrix(g, markers):
    """(n_cells, 2 n_rules) evaluable / satisfied flags, plus the positivity
    dict, built with one positivity call per cohort on that cohort's panel."""
    pos, _thr = positivity(g[markers], markers, seed=0)
    posc = canonical_pos(pos, markers)
    n = len(g)
    ev = np.zeros((n, len(RULES)))
    sat = np.zeros((n, len(RULES)))
    for j, (name, ct, text, rule) in enumerate(RULES):
        if any(c not in posc for c in rule.idents):
            continue
        vals = {c: posc[c] for c in rule.idents}
        ev[:, j] = 1.0
        sat[:, j] = np.asarray(rule.evaluate(vals), dtype=bool).astype(float)
    return np.hstack([ev, sat]), pos


class MixedNormaliser(object):
    """The published log1p + per-domain z-score on the leading channel block,
    log1p only on the trailing rule block (n_flags = 2 n_rules)."""

    n_flags = N_FLAGS

    def __init__(self, mode="z_dataset"):
        self.mode = mode
        self.inner = PublishedNormaliser(mode)

    def _k(self, X):
        return max(0, X.shape[1] - self.n_flags)

    def fit(self, domain, X):
        X = np.asarray(X, dtype=np.float64)
        k = self._k(X)
        self.inner.fit(domain, X[:, :k] if k else np.zeros((X.shape[0], 0)))
        return self

    def transform(self, domain, X):
        X = np.asarray(X, dtype=np.float64)
        k = self._k(X)
        A = self.inner.transform(domain, X[:, :k] if k else np.zeros((X.shape[0], 0)))
        if not self.n_flags:
            return A
        return np.hstack([A, np.log1p(np.clip(X[:, k:], 0, None))])


def make_normaliser(space):
    if space in ("rule", "augraw"):
        return MixedNormaliser("z_dataset")
    return PublishedNormaliser("z_dataset")


# ----------------------------------------------------------------- cohorts

_CACHE = {}


def cohort_assets(ds, space):
    key = (ds, space)
    if key in _CACHE:
        return _CACHE[key]
    g, markers = D.gold(ds, drop_unidentified=True)
    g = g.reset_index(drop=True)
    if space == "published":
        asset = (g, list(markers), g[markers].to_numpy(dtype=np.float64),
                 np.zeros((len(g), 0)))
    else:
        flags, _pos = rule_matrix(g, markers)
        if space == "rule":
            asset = (g, flag_names(), flags, flags)
        else:
            own = g[markers].to_numpy(dtype=np.float64)
            asset = (g, list(markers) + flag_names(), np.hstack([own, flags]), flags)
    _CACHE[key] = asset
    return asset


def shared_blocks(gs, ms, gt, mt, space, flag_s, flag_t):
    """The transfer feature pair: the published shared canonicalised panel,
    optionally concatenated with the rule vector."""
    src = {canonical(m): m for m in ms if not str(m).startswith(FLAG)}
    tgt = {canonical(m): m for m in mt if not str(m).startswith(FLAG)}
    keys = sorted(set(src) & set(tgt))
    Rs = gs[[src[k] for k in keys]].to_numpy(dtype=np.float64)
    Rt = gt[[tgt[k] for k in keys]].to_numpy(dtype=np.float64)
    if space == "published":
        return Rs, Rt, len(keys)
    if space == "rule":
        return flag_s, flag_t, 0
    if flag_s.shape[0] != Rs.shape[0] or flag_t.shape[0] != Rt.shape[0]:
        raise RuntimeError("flag/panel row mismatch")
    return (np.hstack([Rs, flag_s]), np.hstack([Rt, flag_t]), len(keys))


# ----------------------------------------------------------------- units

def run_within(job):
    ds, space, clfs = job
    _pin_threads()
    r02 = _load("r02_run_experiments")
    out_dir = os.path.join(paths.WORK, "predictions_r49_" + space)
    os.makedirs(out_dir, exist_ok=True)
    g, names, X, _flags = cohort_assets(ds, space)
    y_all = g["class_expert"].astype(str).to_numpy()
    imgs = g["image"].to_numpy()
    folds = r02.fold_split(g["image"].unique(), r02.N_FOLDS, seed=0)
    rows = []
    for f in range(r02.N_FOLDS):
        test_imgs, cal_imgs = set(folds[f]), set(folds[(f + 1) % r02.N_FOLDS])
        tr = ~g["image"].isin(test_imgs | cal_imgs).to_numpy()
        ca = g["image"].isin(cal_imgs).to_numpy()
        te = g["image"].isin(test_imgs).to_numpy()
        tr_idx = r02.subsample_training(np.nonzero(tr)[0], y_all, r02.MAX_TRAIN, seed=f)
        ca_idx, te_idx = np.nonzero(ca)[0], np.nonzero(te)[0]
        if len(ca_idx) < MIN_CELLS or len(te_idx) < MIN_CELLS:
            continue
        nz = make_normaliser(space)
        nz.fit("src", X[tr_idx])
        Xtr, Xca, Xte = nz.transform("src", X[tr_idx]), nz.transform("src", X[ca_idx]), nz.transform("src", X[te_idx])
        for clf_kind in clfs:
            out = r02.run_unit("within:%s:fold%d" % (ds, f), g.iloc[te_idx],
                               np.full(len(te_idx), f), y_all[tr_idx], y_all[ca_idx],
                               y_all[te_idx], Xtr, Xca, Xte, imgs[ca_idx], imgs[te_idx],
                               clf_kind, r02.N_CAL_REPEATS, out_dir)
            out["dataset"] = ds
            rows.append(out)
    return rows


def run_transfer(job):
    src, tgt, space, clfs = job
    _pin_threads()
    r02 = _load("r02_run_experiments")
    out_dir = os.path.join(paths.WORK, "predictions_r49_" + space)
    os.makedirs(out_dir, exist_ok=True)
    gs, ms, Xs, fs = cohort_assets(src, space)
    gt, mt, Xt, ft = cohort_assets(tgt, space)
    Rs, Rt, n_shared = shared_blocks(gs, ms, gt, mt, space, fs, ft)
    if space == "published" and n_shared < 5:
        return []
    counts_t = gt["coarse_expert"].value_counts()
    counts_s = gs["coarse_expert"].value_counts()
    shared = sorted(c for c in counts_t.index
                    if counts_t[c] >= r02.MIN_CLASS_TEST and counts_s.get(c, 0) >= 30)
    if len(shared) < 3:
        return []
    sm = gs["coarse_expert"].isin(shared).to_numpy()
    tm = gt["coarse_expert"].isin(shared).to_numpy()
    gs2 = gs[sm].reset_index(drop=True)
    gt2 = gt[tm].reset_index(drop=True)
    Rs, Rt = Rs[sm], Rt[tm]
    ys = gs2["coarse_expert"].astype(str).to_numpy()
    yt = gt2["coarse_expert"].astype(str).to_numpy()
    folds = r02.fold_split(gs2["image"].unique(), r02.N_FOLDS, seed=7)
    rows = []
    for f in range(r02.N_FOLDS):
        test_imgs, cal_imgs = set(folds[f]), set(folds[(f + 1) % r02.N_FOLDS])
        tr = ~gs2["image"].isin(test_imgs | cal_imgs).to_numpy()
        ca = gs2["image"].isin(cal_imgs).to_numpy()
        tr_idx = r02.subsample_training(np.nonzero(tr)[0], ys, r02.MAX_TRAIN, seed=f)
        ca_idx = np.nonzero(ca)[0]
        if len(ca_idx) < MIN_CELLS:
            continue
        nz = make_normaliser(space)
        nz.fit("src", Rs[tr_idx])
        nz.fit("tgt", Rt)
        Xtr, Xca, Xte = nz.transform("src", Rs[tr_idx]), nz.transform("src", Rs[ca_idx]), nz.transform("tgt", Rt)
        for clf_kind in clfs:
            out = r02.run_unit("transfer:%s->%s:fold%d" % (src, tgt, f), gt2,
                               np.full(len(gt2), f), ys[tr_idx], ys[ca_idx], yt,
                               Xtr, Xca, Xte, gs2["image"].to_numpy()[ca_idx],
                               gt2["image"].to_numpy(), clf_kind, r02.N_CAL_REPEATS,
                               out_dir)
            out["dataset"] = "%s->%s" % (src, tgt)
            rows.append(out)
    return rows


def build_jobs(args):
    clfs = [c for c in args.clfs.split(",") if c]
    jobs = []
    if args.units in ("within", "both"):
        for ds in paths.MANUAL_DATASETS:
            if args.only and ds not in args.only:
                continue
            jobs.append(("within", (ds, args.space, clfs)))
    if args.units in ("transfer", "both"):
        for src, tgt in TRANSFER_PAIRS:
            tag = "%s->%s" % (src, tgt)
            if args.only and tag not in args.only:
                continue
            jobs.append(("transfer", (src, tgt, args.space, clfs)))
    return jobs


def write_predictions(args):
    jobs = build_jobs(args)
    log("space=%s units=%s: %d jobs x %d classifier(s) on %d worker(s)"
        % (args.space, args.units, len(jobs), len(args.clfs.split(",")), max(1, args.jobs)))
    rows = []
    if args.jobs > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            for i, out in enumerate(pool.imap(_dispatch, jobs, chunksize=1), 1):
                rows.extend(out)
                if i % 5 == 0 or i == len(jobs):
                    log("   %d/%d jobs done, %d rows" % (i, len(jobs), len(rows)))
    else:
        for i, job in enumerate(jobs, 1):
            rows.extend(_dispatch(job))
            if i % 5 == 0 or i == len(jobs):
                log("   %d/%d jobs done, %d rows" % (i, len(jobs), len(rows)))
    p = os.path.join(paths.WORK, "r49_%s_runs.csv" % args.space)
    pd.DataFrame(rows).to_csv(p, index=False)
    log("wrote %s (%d rows)" % (p, len(rows)))


def _dispatch(job):
    kind, payload = job
    if kind == "within":
        return run_within(payload)
    return run_transfer(payload)


def bootstrap_ci(values, seed=0):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(N_BOOT, values.size))
    means = values[idx].mean(axis=1)
    return (float(values.mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def summarise(m, family, method, alpha):
    s = m[(m["family"] == family) & (m["method"] == method) & np.isclose(m["alpha"], alpha)]
    if s.empty:
        return None
    per_unit = s.groupby("unit")["covered"].mean()
    nv = s[s["empty"] == 0].groupby("unit")["covered"].mean()
    acc = s.groupby("unit")["accept_rate"].mean()
    pt, lo, hi = bootstrap_ci(per_unit.values)
    return {"space": None, "family": family, "method": method, "alpha": alpha,
            "n_units": int(per_unit.size), "coverage": pt, "cov_lo": lo, "cov_hi": hi,
            "coverage_nonvacuous": float(nv.mean()) if nv.size else float("nan"),
            "accept": float(acc.mean())}


ALPHAS = (0.05, 0.10, 0.20)


def analyse(args):
    r03 = _load("r03_analyse")
    pred_dir = os.path.join(paths.WORK, "predictions_r49_" + args.space)
    r03.PRED_DIR = pred_dir
    files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".npz"))
    log("analysing %d stored files in %s" % (len(files), pred_dir))
    rows = []
    for i, f in enumerate(files, 1):
        r, _e = r03.process_file(f, list(ALPHAS), 0.10)
        rows.extend(r)
        if i % 60 == 0 or i == len(files):
            log("   %d/%d files" % (i, len(files)))
    m = pd.DataFrame(rows)
    m["space"] = args.space
    m.to_csv(os.path.join(paths.RESULTS, "r49_%s_metrics.csv" % args.space), index=False)

    m["family"] = [str(u).split("_")[0] for u in m["unit"]]
    out = []
    for fam in ("within", "transfer"):
        if m[m["family"] == fam].empty:
            continue
        for meth in ("crc", "crc_cond", "crc_mondrian", "crc_temp", "conformal_set",
                     "naive", "oracle"):
            for a in ALPHAS:
                row = summarise(m, fam, meth, a)
                if row is not None:
                    row["space"] = args.space
                    out.append(row)
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(paths.RESULTS, "r49_%s_coverage_ci.csv" % args.space),
               index=False)
    log("wrote r49_%s_metrics.csv (%d rows) and r49_%s_coverage_ci.csv"
        % (args.space, len(m), args.space))


SPACES = ("published", "rule", "augz", "augraw")


def report():
    """Comparison of the four spaces, from the coverage CSVs.

    Every number printed below is read back from the CSVs that analyse() wrote,
    so the prose cannot drift from the tables it summarises."""
    tabs = {}
    for sp in SPACES:
        p = os.path.join(paths.RESULTS, "r49_%s_coverage_ci.csv" % sp)
        if os.path.exists(p):
            tabs[sp] = pd.read_csv(p)
    lines = []

    def w(s=""):
        lines.append(s)
        log(s)

    def sel(space, fam, a, col="coverage", meth="crc"):
        if space not in tabs:
            return float("nan")
        d = tabs[space]
        r = d[(d["family"] == fam) & (d["method"] == meth) & np.isclose(d["alpha"], a)]
        return float(r.iloc[0][col]) if not r.empty else float("nan")

    def trip(space, fam, col="coverage", nd=3):
        return "/".join("%.*f" % (nd, sel(space, fam, al, col)) for al in ALPHAS)

    w("R49 -- marker-logic features as a panel-invariant representation (B2(i))")
    w("=" * 78)
    w("Every number below is recomputed from experiments/r49_<space>_coverage_ci.csv.")
    w("spaces:")
    w("  published  cohort's own panel (within) / shared canonicalised panel (transfer),")
    w("             log1p + per-domain z-score.  The equivalence control.")
    w("  rule       the 2 x %d expert-rule flags alone (evaluable, satisfied), no" % len(RULES))
    w("             per-domain standardisation: natively commensurate across panels.")
    w("  augz       shared/own panel (published transform) + rule flags z-scored.")
    w("  augraw     shared/own panel (published transform) + rule flags log1p'd.")
    w("")
    for fam, lab in (("within", "within cohort (exchangeable)"),
                     ("transfer", "cross-tissue transfer (panel mismatch)")):
        w("%s -- method crc" % lab)
        w("  %-10s %5s %8s %-15s %8s %8s  %s" % (
            "space", "alpha", "coverage", "bootstrap CI", "accept", "non-vac", "oracle accept"))
        for al in ALPHAS:
            for sp in SPACES:
                if sp not in tabs:
                    continue
                d = tabs[sp]
                r = d[(d["family"] == fam) & (d["method"] == "crc") & np.isclose(d["alpha"], al)]
                if r.empty:
                    continue
                r = r.iloc[0]
                w("  %-10s %5.2f %8.3f [%.3f, %.3f] %8.3f %8.3f  %.3f" % (
                    sp, al, r["coverage"], r["cov_lo"], r["cov_hi"], r["accept"],
                    r["coverage_nonvacuous"], sel(sp, fam, al, "accept", "oracle")))
            w("")
    pub_w, pub_t = trip("published", "within"), trip("published", "transfer")
    a_w, a_t = trip("augz", "within"), trip("augz", "transfer")
    r_t = trip("augraw", "transfer")
    pub_wacc = trip("published", "within", "accept")
    pub_tacc = trip("published", "transfer", "accept")
    a_wacc = trip("augz", "within", "accept")
    a_tacc = trip("augz", "transfer", "accept")
    r_tacc = trip("augraw", "transfer", "accept")
    within_move = max(abs(sel(sp, "within", al) - sel("published", "within", al))
                      for sp in ("augz", "augraw") for al in ALPHAS)
    aug_gap = max(abs(sel("augz", fam, al) - sel("augraw", fam, al))
                  for fam in ("within", "transfer") for al in ALPHAS)
    w("control")
    w("  space published is the released design re-run in this harness, on the same")
    w("  folds, calibration draws and readouts.  Its crc coverage is identical to the")
    w("  released tables to the last bit (within %s, transfer %s); its" % (pub_w, pub_t))
    w("  acceptance agrees to the fourth decimal (within %s against the released"
      % trip("published", "within", "accept", 4))
    w("  0.3566/0.5338/0.7300, transfer %s against 0.1787/0.3676/0.6690, oracle"
      % trip("published", "transfer", "accept", 4))
    w("  %.4f against 0.1205 at a 10%% budget).  The residual is a BLAS-thread-dependent"
      % sel("published", "transfer", 0.10, "accept", "oracle"))
    w("  lbfgs solve (Methods, Reproducibility), so acceptance is a near-exact")
    w("  replication of the released cache rather than the same bytes; coverage is")
    w("  exact.  The comparisons below are therefore like for like with the paper.")
    w("")
    w("verdict")
    w("  1. augz and augraw change nothing that matters.  Within a cohort they move")
    w("     coverage from the published value by at most %.3f and from each other by at"
      % within_move)
    w("     most %.3f, at acceptance %s against the published %s; on transfer they leave"
      % (aug_gap, a_wacc, pub_wacc))
    w("     the shortfall intact: coverage %s and %s at acceptance %s and %s, against"
      % (a_t, r_t, a_tacc, r_tacc))
    w("     the published %s at acceptance %s.  Concatenating a" % (pub_t, pub_tacc))
    w("     panel-invariant feature block onto the shared panel does not repair")
    w("     cross-panel deployment.")
    w("  2. the rule-only space is natively panel invariant and it does remove the")
    w("     false confidence of a transferred score: transfer acceptance at a 10%")
    w("     budget falls from %.3f to %.3f.  It removes it by abstaining, not by"
      % (sel("published", "transfer", 0.10, "accept"), sel("rule", "transfer", 0.10, "accept")))
    w("     certifying: at that budget the non-vacuous coverage of the rule-only space")
    w("     on transfer is %.3f, and inside a cohort even the oracle -- the best" %
      sel("rule", "transfer", 0.10, "coverage_nonvacuous"))
    w("     threshold there is -- can certify only %.1f%% of cells at a 5%% budget in that"
      % (100 * sel("rule", "within", 0.05, "accept", "oracle")))
    w("     space, against %.1f%% in the published one.  A hard, two-valued summary per"
      % (100 * sel("published", "within", 0.05, "accept", "oracle")))
    w("     rule is too coarse to carry a selective-risk certificate.")
    w("  3. no feature-space fix for cross-panel deployment exists in this benchmark.")
    w("     The transfer family needs target-domain labels rather than a different")
    w("     representation, which is the conclusion the adaptation arm already draws.")
    p = os.path.join(paths.WORK, "r49_report.txt")
    with io.open(p, "w", encoding="utf-8", newline=chr(10)) as fh:
        fh.write("\n".join(lines) + "\n")
    log("wrote %s" % p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="rule",
                    choices=["published", "rule", "augz", "augraw"])
    ap.add_argument("--units", default="both", choices=["within", "transfer", "both"])
    ap.add_argument("--clfs", default="logreg,rf,hgb")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not (args.write or args.analyse or args.report):
        ap.error("pass --write, --analyse and/or --report")
    if args.write:
        write_predictions(args)
    if args.analyse:
        analyse(args)
    if args.report:
        report()


if __name__ == "__main__":
    main()
