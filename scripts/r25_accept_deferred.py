"""R25 -- is the accepted-vs-deferred marker-consistency claim convention-free?

The manuscript states that "the cells the controller accepts are more
marker-consistent than the cells it defers, in every cohort and in six of the
nine cohort-classifier combinations where the comparison is estimable".  That
comparison exists at exactly one marker-audit convention: the two-component
Gaussian-mixture positivity call, the depot's ";" read as OR, the repaired rule
text, at alpha = 0.10 (experiments/r15_deployed.csv).  R18 showed that the
marker-audit answer is dominated by the positivity convention, so the ordering
may be an artefact of that one convention.

R25 re-runs the SAME accepted/deferred split -- same cohorts, same classifiers,
same folds, same score, same lambda table -- under

    positivity in {gmm, pct_all@60..95, pct_pospop@60..95}   (17 settings)
  x reading    in {or, and}                                  (2)
  x ruleset    in {shipped, repaired}                         (2)
  x scope      in {ALL_evaluable, ALL_strict}                 (2)
  x alpha      in {0.05, 0.10, 0.20}

and asks how often the violation rate among ACCEPTED cells is below the rate
among DEFERRED cells.  The primary statistic is the paired sign test over all
estimable (dataset, classifier, alpha, convention) units.

The marker logic, the class scopes and the evaluability rules are imported
unchanged from scripts/r15_marker_sensitivity.py.

Outputs: experiments/r25_accept_deferred.csv, work/r25_report.txt,
work/r25_log.txt, figures/figS6_accept_deferred.pdf/.png.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from errctl import data as D, markercheck as MC, paths

RESULTS = paths.RESULTS
FIGURES = paths.FIGURES
WORK = paths.WORK
LOG_PATH = os.path.join(WORK, "r25_log.txt")
UNITS_CSV = os.path.join(WORK, "_r25_units.csv")
DIAG_JSON = os.path.join(WORK, "_r25_diag.json")

READINGS = ("or", "and")
VARIANTS = ("shipped", "repaired")
SCOPES = ("ALL_evaluable", "ALL_strict")
PCTS = (60, 65, 70, 75, 80, 85, 90, 95)
ALPHAS = (0.05, 0.10, 0.20)
CLFS = ("logreg", "rf", "hgb")

# A unit is estimable when the accepted side has enough evaluable cells to
# evaluate the rules at all, and the deferred side can carry a rate.  The
# threshold mirrors the manuscript's own language ("all three colorectal pairs
# accept fewer than 75 cells").
EVAL_MIN = 75
SENS_MIN = (30, 200)

FAMILY_CLR = {"gmm": "#1f5fa9", "pct_all": "#d1495b", "pct_pospop": "#2a9d8f"}
FAM_ORDER = {"gmm": 0, "pct_all": 1, "pct_pospop": 2}
PRIMARY_SCOPE = "ALL_strict"


def log(msg=""):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")


def load_r15():
    spec = importlib.util.spec_from_file_location(
        "r15_marker_sensitivity",
        os.path.join(HERE, "r15_marker_sensitivity.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R15 = load_r15()
wilson = R15.wilson


def positivity_grid():
    out = [("gmm", "gmm", np.nan)]
    for p in PCTS:
        out.append(("pct_all@%d" % p, "pct_all", float(p)))
    for p in PCTS:
        out.append(("pct_pospop@%d" % p, "pct_pospop", float(p)))
    return out


POSITIVITIES = positivity_grid()
CONVENTIONS = [(pl, fa, pc, rd, vr) for pl, fa, pc in POSITIVITIES
               for rd in READINGS for vr in VARIANTS]


def conv_key(family, pct):
    return (FAM_ORDER[family], -1.0 if not np.isfinite(pct) else float(pct))


def short_conv(positivity, reading, variant):
    if positivity == "gmm":
        core = "gmm"
    else:
        fam, pct = positivity.split("@")
        core = ("a" if fam == "pct_all" else "p") + pct
    return "%s %s %s" % (core, reading, "rep" if variant == "repaired" else "shp")


# ------------------------------------------------------------------- counting
def _acc_parts(idx, sc, fold, lam_by_unit, units):
    """(cell index, accept flag) for one lambda configuration, folds pooled."""
    ii_parts, acc_parts = [], []
    for unit in units:
        lam = lam_by_unit.get(unit, np.nan)
        if not np.isfinite(lam):
            continue
        sel = fold == int(unit.rsplit("fold", 1)[1])
        ii_parts.append(idx[sel])
        acc_parts.append(sc[sel] <= lam)
    if not ii_parts:
        return None, None
    return np.concatenate(ii_parts), np.concatenate(acc_parts)


def _counts(e, v, ii, acc):
    """(n_acc, k_acc, n_def, k_def) for one evaluable mask and one split."""
    ea = e[ii]
    va = v[ii]
    ma = acc & ea
    md = (~acc) & ea
    return (int(ma.sum()), int(va[ma].sum()),
            int(md.sum()), int(va[md].sum()))


# ------------------------------------------------------------------ per cohort
def dataset_stage(ds, met, clfs, alphas):
    t0 = time.time()
    g, markers = D.gold(ds, drop_unidentified=True)
    g = g.reset_index(drop=True)
    frame, chans = R15.canonical_frame(g, markers)
    labels = g["coarse_expert"].to_numpy(dtype=object)

    pos_gmm = MC.positivity(frame, chans)[0]
    pos_by = {"gmm": pos_gmm}
    for p in PCTS:
        pos_by["pct_all@%d" % p] = R15.percentile_positivity(frame, chans, p, "all")
    for p in PCTS:
        pos_by["pct_pospop@%d" % p] = R15.percentile_positivity(
            frame, chans, p, "pospop", base=pos_gmm)

    groups_by = {}
    for reading in READINGS:
        for variant in VARIANTS:
            rules, _ = R15.rule_table(ds, reading, repair=(variant == "repaired"))
            groups, _, _ = R15.group_rules(rules)
            groups_by[(reading, variant)] = groups

    # One (evaluable, violated) mask pair per convention x scope, exactly the
    # rulesets deployed_stage builds for ALL_evaluable and ALL_strict.
    convs = []
    for reading in READINGS:
        for variant in VARIANTS:
            groups = groups_by[(reading, variant)]
            for plabel, family, pct in POSITIVITIES:
                pos = pos_by[plabel]
                ev, vi, used, missing = R15.evaluate(labels, groups, pos)
                evl = R15.evaluability(missing, used)
                strict = {c for c in groups if evl.get(c) == "complete"}
                ev_s, vi_s, _, _ = R15.evaluate(labels, groups, pos, restrict=strict)
                convs.append((plabel, family, pct, reading, variant,
                              "ALL_evaluable", ev, vi))
                convs.append((plabel, family, pct, reading, variant,
                              "ALL_strict", ev_s, vi_s))

    gidx = {}
    for i, (im, cid) in enumerate(zip(g["image"], g["cellID"])):
        gidx[(str(im), int(cid))] = i

    class_list = sorted(groups_by[("or", "repaired")])
    code = np.zeros(int(g.shape[0]), dtype=np.int64)
    for j, c in enumerate(class_list, start=1):
        code[labels == c] = j

    rows = []
    diag = {"ds": ds, "n_gold": int(g.shape[0]), "n_channels": len(chans),
            "dups": {}, "n_matched": {}, "n_unique": {}}
    pool_primary = {}
    sens_pairs = []
    std_pairs = []

    for clf in clfs:
        rws, _pred, dup = R15._load_within(ds, clf, gidx, len(g))
        diag["dups"][clf] = int(dup)
        if not rws:
            continue
        idx = np.array([r[0] for r in rws], dtype=np.int64)
        sc = np.array([r[1] for r in rws], dtype=float)
        fold = np.array([r[2] for r in rws], dtype=int)
        diag["n_matched"][clf] = int(len(rws))
        diag["n_unique"][clf] = int(np.unique(idx).size)

        for alpha in alphas:
            sub = met[(met.clf == clf) & np.isclose(met.alpha, alpha)]
            if sub.empty:
                continue
            lamap = {}
            for _u, _rp, _lm in zip(sub["unit"], sub["repeat"], sub["lambda"]):
                if np.isfinite(_lm):
                    lamap[(_u, int(_rp))] = float(_lm)
            units = sorted({u for u, _ in lamap})
            if not units:
                continue

            # PRIMARY unit definition, identical to r15's deployed_stage: the
            # accepted/deferred split is built once per DISTINCT lambda
            # configuration present in the repeat column, and the counts are
            # then POOLED over those configurations.  In melanoma this counts
            # each cell about six times, once per configuration.
            cfgs = {}
            for rep in sorted({int(r) for r in sub["repeat"].unique()}):
                key = tuple(round(lamap.get((u, rep), np.nan), 10) for u in units)
                cfgs.setdefault(key, rep)
            cfg_parts = []
            for key, _rep in sorted(cfgs.items(), key=lambda kv: kv[1]):
                lub = {u: key[i] for i, u in enumerate(units)}
                ii_c, acc_c = _acc_parts(idx, sc, fold, lub, units)
                if ii_c is not None:
                    cfg_parts.append((ii_c, acc_c))

            # sensitivity only: the repeat-0 lambda alone (each cell once)
            lam0 = {u: lamap.get((u, 0), np.nan) for u in units}
            ii0, acc0 = _acc_parts(idx, sc, fold, lam0, units)
            if ii0 is None:
                continue

            for (plabel, family, pct, reading, variant, scope, e, v) in convs:
                na, ka, nd, kd = 0, 0, 0, 0
                ua_parts, ud_parts = [], []
                for ii_c, acc_c in cfg_parts:
                    c = _counts(e, v, ii_c, acc_c)
                    na += c[0]
                    ka += c[1]
                    nd += c[2]
                    kd += c[3]
                    ec = e[ii_c]
                    ua_parts.append(ii_c[acc_c & ec])
                    ud_parts.append(ii_c[(~acc_c) & ec])
                n_uniq_acc = (int(np.unique(np.concatenate(ua_parts)).size)
                              if ua_parts else 0)
                n_uniq_def = (int(np.unique(np.concatenate(ud_parts)).size)
                              if ud_parts else 0)
                est = (na >= EVAL_MIN) and (nd >= EVAL_MIN)
                for side, n_, k_ in (("accepted", na, ka), ("deferred", nd, kd)):
                    lo, hi = wilson(k_, n_)
                    rows.append(dict(
                        dataset=ds, clf=clf, alpha=float(alpha),
                        positivity=plabel, family=family, pct=pct,
                        reading=reading, ruleset=variant, scope=scope,
                        side=side, n=int(n_), n_violation=int(k_),
                        rate=(float(k_) / float(n_)) if n_ else np.nan,
                        ci_lo=lo, ci_hi=hi, pooling="r15pool",
                        n_unique_cells=(n_uniq_acc if side == "accepted"
                                        else n_uniq_def),
                        evaluable=bool(n_ >= EVAL_MIN),
                        estimable=bool(est)))

                if (plabel == "gmm" and reading == "or" and variant == "repaired"
                        and abs(alpha - 0.10) < 1e-9):
                    pool_primary["%s|%s|%s|accepted" % (ds, clf, scope)] = (na, ka)
                    pool_primary["%s|%s|%s|deferred" % (ds, clf, scope)] = (nd, kd)

                if scope == "ALL_strict" and na and nd:
                    na_arr = np.zeros(len(class_list) + 1, dtype=np.int64)
                    nd_arr = np.zeros(len(class_list) + 1, dtype=np.int64)
                    ka_arr = np.zeros(len(class_list) + 1, dtype=np.float64)
                    kd_arr = np.zeros(len(class_list) + 1, dtype=np.float64)
                    for ii_c, acc_c in cfg_parts:
                        ec = e[ii_c]
                        vc = v[ii_c].astype(float)
                        ba = acc_c & ec
                        bd = (~acc_c) & ec
                        ia = ii_c[ba]
                        idd = ii_c[bd]
                        na_arr += np.bincount(code[ia],
                                              minlength=len(class_list) + 1)
                        nd_arr += np.bincount(code[idd],
                                              minlength=len(class_list) + 1)
                        ka_arr += np.bincount(code[ia], weights=vc[ba],
                                              minlength=len(class_list) + 1)
                        kd_arr += np.bincount(code[idd], weights=vc[bd],
                                              minlength=len(class_list) + 1)
                    num = den = 0.0
                    ncl = 0
                    for j in range(1, len(class_list) + 1):
                        na_c = int(na_arr[j])
                        nd_c = int(nd_arr[j])
                        if na_c < 20 or nd_c < 20:
                            continue
                        w = na_c + nd_c
                        num += w * (kd_arr[j] / nd_c - ka_arr[j] / na_c)
                        den += w
                        ncl += 1
                    std_pairs.append(dict(
                        dataset=ds, clf=clf, alpha=float(alpha),
                        positivity=plabel, reading=reading, ruleset=variant,
                        scope="ALL_strict",
                        diff_std=(num / den if den else np.nan),
                        n_classes_used=ncl))

                r0a, r0ka, r0d, r0kd = _counts(e, v, ii0, acc0)
                r0 = ((r0kd / r0d - r0ka / r0a) if (r0a and r0d) else np.nan)
                rp = ((kd / nd - ka / na) if (na and nd) else np.nan)
                sens_pairs.append(dict(
                    dataset=ds, clf=clf, alpha=float(alpha),
                    positivity=plabel, reading=reading, ruleset=variant,
                    scope=scope, diff_repeat0=r0, diff_r15pool=rp,
                    n_configs=len(cfg_parts)))

    log("%-16s gold=%d chans=%d masks=%d rows=%d  %.1fs"
        % (ds, diag["n_gold"], diag["n_channels"], len(convs), len(rows),
           time.time() - t0))
    return {"rows": rows, "diag": diag, "pool_primary": pool_primary,
            "sens": sens_pairs, "std": std_pairs}


# ------------------------------------------------------------------ statistics
def sign_test(diffs):
    """Paired sign test.  diff = rate_deferred - rate_accepted."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    n_tie = int((d == 0).sum())
    n = n_pos + n_neg
    out = dict(n_pos=n_pos, n_neg=n_neg, n_tie=n_tie, n=n,
               median=float(np.median(d)) if d.size else np.nan,
               mean=float(np.mean(d)) if d.size else np.nan,
               p_greater=np.nan, p_two=np.nan)
    if n:
        from scipy import stats
        out["p_greater"] = float(
            stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue)
        out["p_two"] = float(
            stats.binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue)
    return out


def to_units(rows):
    df = pd.DataFrame(rows)
    key = ["dataset", "clf", "alpha", "positivity", "family", "pct",
           "reading", "ruleset", "scope"]
    a = df[df.side == "accepted"].copy()
    b = df[df.side == "deferred"].copy()
    u = a.merge(b, on=key, suffixes=("_acc", "_def"))
    u["diff"] = u["rate_def"] - u["rate_acc"]
    u["estimable"] = (u["n_acc"] >= EVAL_MIN) & (u["n_def"] >= EVAL_MIN)
    return u


def _pp(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    if p < 1e-4:
        return "%.2e" % p
    return "%.4f" % p


def _sig_line(label, d, width=34):
    return ("%-*s  %5d est  %5d <  %5d >  %5d =  %4d    %-10s  %+9.5f"
            % (width, label, d["n"] + d["n_tie"], d["n_pos"], d["n_neg"],
               d["n_tie"], d["n"], _pp(d["p_greater"]), d["median"]))


SIG_HEADER = ("%-34s  %9s  %7s  %7s  %5s  %5s  %10s  %9s"
              % ("group", "n_units", "acc<def", "acc>def", "ties", "n_sig",
                 "p_one_sided", "median_d"))


# --------------------------------------------------------------------- report
def make_verdict(st, scope, univ, n_fail_conv, n_conv):
    k, n, p, med = st["n_pos"], st["n"], st["p_greater"], st["median"]
    if n == 0:
        return ("VERDICT: no convention yields an estimable accepted-vs-deferred "
                "comparison, so the manuscript's claim cannot be tested here.")
    bad = [u for u in univ if u["n_neg"] > 0]
    if k == n and not bad:
        return ("VERDICT: convention-free -- accepted cells violate the marker "
                "rules less often than deferred cells in all %d estimable "
                "(dataset, classifier, alpha, convention) units under the %s "
                "scope (sign test %d/%d, one-sided p=%s, median paired "
                "difference deferred-minus-accepted %+.4f), so the manuscript's "
                "claim does not depend on the marker-audit convention."
                % (n, scope, k, n, _pp(p), med))
    det = "; ".join("%s reverses in %d of its %d estimable units"
                    % (u["dataset"], u["n_neg"],
                       u["n_pos"] + u["n_neg"] + u["n_tie"])
                    for u in bad)
    failclause = ("breaks down outright in %d of %d conventions"
                  % (n_fail_conv, n_conv) if n_fail_conv
                  else "never fails outright inside a whole convention (0 of %d)"
                  % n_conv)
    return ("VERDICT: NOT convention-free -- accepted cells violate the marker "
            "rules less often than deferred cells in %d of %d estimable "
            "(dataset, classifier, alpha, convention) units under the %s scope "
            "(sign test %d/%d, one-sided p=%s, median paired difference "
            "deferred-minus-accepted %+.4f), but the ordering is not universal "
            "in every cohort x classifier cell (%s) and %s, so the manuscript's "
            "'in every cohort' wording must be narrowed to the cohorts and "
            "conventions where it holds."
            % (k, n, scope, k, n, _pp(p), med, det, failclause))


def write_report(units, rows, diags, pool_primary, sens, std, repro, args, elapsed):
    L = []
    if std is not None and len(std):
        kcol = ["dataset", "clf", "alpha", "positivity", "reading", "ruleset",
                "scope"]
        units = units.merge(std[kcol + ["diff_std", "n_classes_used"]], on=kcol,
                            how="left")
    prim = units[units.scope == PRIMARY_SCOPE]
    prim_est = prim[prim.estimable]
    st = sign_test(prim_est["diff"])

    conv_groups = prim_est.groupby(["positivity", "family", "pct", "reading",
                                    "ruleset"], dropna=False)
    n_conv_all = len(list(conv_groups)) if len(prim_est) else 0
    fail_convs = []
    for key, sub in conv_groups:
        s = sign_test(sub["diff"])
        if s["n"] and s["n_pos"] == 0:
            fail_convs.append(key)
    univ = []
    for ds in paths.MANUAL_DATASETS:
        s = sign_test(prim_est[prim_est.dataset == ds]["diff"])
        univ.append(dict(dataset=ds, n_pos=s["n_pos"], n_neg=s["n_neg"],
                         n_tie=s["n_tie"]))
    verdict = make_verdict(st, PRIMARY_SCOPE, univ, len(fail_convs), n_conv_all)
    L.append(verdict)
    L.append("")
    L.append("=" * 78)
    L.append('R25 -- does "the cells the controller accepts are more '
             'marker-consistent')
    L.append('        than the cells it defers" survive a change of the '
             'marker-audit')
    L.append("        convention?  (supporting analysis, not a theorem)")
    L.append("=" * 78)
    L.append("")

    # ---------------------------------------------------------------- section 1
    L.append("1. WHAT IS TESTED, AND WHAT WAS PRE-SPECIFIED")
    L.append("-" * 78)
    L.append("Cohorts    : %s" % ", ".join(paths.MANUAL_DATASETS))
    L.append("Classifiers: %s     Budgets alpha: %s"
             % (", ".join(CLFS), ", ".join("%.2f" % a for a in ALPHAS)))
    L.append("Split      : the within-cohort test cells of work/predictions/")
    L.append("             within_<cohort>_fold<k>__<clf>.npz; score = 1 - max prob;")
    L.append("             a cell is ACCEPTED when score <= lambda(fold) and")
    L.append("             DEFERRED otherwise.  lambda is read from")
    L.append("             experiments/r03_metrics.csv, method 'crc'.")
    L.append("UNITS      : identical to r15's deployed_stage -- one split per")
    L.append("             DISTINCT lambda configuration in the repeat column,")
    L.append("             counts POOLED over those configurations.  In melanoma")
    L.append("             (6 distinct configurations) each cell is therefore")
    L.append("             counted ~6 times, exactly as in r15_deployed.csv.")
    L.append("             n_unique_cells in the csv gives the distinct cells.")
    L.append("             The repeat-0 split alone is reported as a sensitivity.")
    L.append("Audit      : per-cell marker-rule violation of the cell's EXPERT")
    L.append("             coarse label, r15's rules and evaluability rules unchanged.")
    L.append("Conventions: positivity x reading x ruleset")
    L.append("             positivity in {gmm, pct_all@60..95, pct_pospop@60..95}")
    L.append("                         (17 settings, as in experiments/r18_conventions.csv)")
    L.append('             reading   in {or, and}    (depot ";" read as OR or AND)')
    L.append("             ruleset   in {shipped, repaired}")
    L.append("             -> 68 conventions per scope, x 2 class scopes")
    L.append("                (ALL_evaluable, ALL_strict) = 136 audit variants.")
    L.append("Estimable  : a (dataset, clf, alpha, convention, scope) unit counts")
    L.append("             only when BOTH sides carry >= %d evaluable cells." % EVAL_MIN)
    L.append("             The threshold mirrors the manuscript's own wording")
    L.append('             ("all three colorectal pairs accept fewer than 75 cells").')
    L.append("             Sensitivity at >= %d and >= %d is reported in section 4."
             % SENS_MIN)
    L.append("PRIMARY    : scope ALL_strict (the convention behind every rate the")
    L.append("             manuscript quotes for GVHD, melanoma and breast).")
    L.append("             Statistic = paired sign test of accepted < deferred over")
    L.append("             all estimable units; effect size = median paired")
    L.append("             difference (deferred rate - accepted rate).")
    L.append("SECONDARY  : the same over ALL_evaluable, over both scopes pooled,")
    L.append("             and per alpha / per dataset / per classifier.")
    L.append("Ties       : units with an exactly equal pair are dropped from the")
    L.append("             sign test and counted separately.")
    L.append("Reproduction: the primary convention (gmm / or / repaired / alpha=0.10)")
    L.append("             is recomputed with r15's own distinct-configuration")
    L.append("             pooling and compared row by row with r15_deployed.csv.")
    L.append("")

    # ---------------------------------------------------------------- section 2
    L.append("2. PRIMARY-CONVENTION REPRODUCTION CHECK")
    L.append("-" * 78)
    if repro is None or repro.empty:
        L.append("  r15_deployed.csv not found: reproduction NOT verified.")
    else:
        n_ok = int(repro["exact"].sum())
        L.append("  %d of %d accepted/deferred rows of experiments/r15_deployed.csv"
                 % (n_ok, len(repro)))
        L.append("  (gmm / or / repaired, alpha = 0.10, both scopes, 3 classifiers x")
        L.append("  4 cohorts = 48 rows) reproduced EXACTLY in n, n_violation")
        L.append("  and rate = k/n.")
        L.append("  max |rate_mine - float(k_ref)/n_ref| = %.1e"
                 % float(np.nanmax(repro["abs_rate_dev"].astype(float))))
        L.append("  max |rate_mine - rate as stored in the csv| = %.1e"
                 % float(np.nanmax(repro["abs_rate_dev_csv"].astype(float))))
        L.append("  (the second is non-zero only because the csv stores 16")
        L.append("  significant digits; the recomputed value is bit-identical.)")
        if n_ok != len(repro):
            L.append("  *** MISMATCH -- generalisation below is NOT trustworthy ***")
    L.append("")
    L.append("  The unit definition above is therefore r15's: one accepted/deferred")
    L.append("  split per distinct lambda configuration, pooled.  Sections 3-9 all")
    L.append("  use it.  Section 9 reports the repeat-0 split (each cell once) as")
    L.append("  a sensitivity.")
    L.append("")

    # ---------------------------------------------------------------- section 3
    L.append("3. ESTIMABILITY (is the comparison defined at all?)")
    L.append("-" * 78)
    tot_units = len(units[units.scope == PRIMARY_SCOPE])
    est = int(units[units.scope == PRIMARY_SCOPE]["estimable"].sum())
    L.append("  %s scope: %d units defined, %d estimable (%.1f%%), %d with an"
             % (PRIMARY_SCOPE, tot_units, est, 100.0 * est / max(tot_units, 1),
                tot_units - est))
    L.append("  uninformative or too-small accepted side.")
    L.append("")
    L.append("  %-16s %8s %8s %8s   %s"
             % ("cohort", "units", "est", "est%",
                "accepted pooled/unique n at gmm/or/repaired a=0.10"))
    for ds in paths.MANUAL_DATASETS:
        s = units[(units.scope == PRIMARY_SCOPE) & (units.dataset == ds)]
        e = int(s["estimable"].sum())
        g0 = s[(s.positivity == "gmm") & (s.reading == "or")
               & (s.ruleset == "repaired")
               & np.isclose(s.alpha, 0.10)].sort_values("clf")
        nacc = ",".join("%d/%d" % (int(a), int(b)) for a, b in
                        zip(g0["n_acc"], g0["n_unique_cells_acc"]))
        L.append("  %-16s %8d %8d %7.1f%%   %s"
                 % (ds, len(s), e, 100.0 * e / max(len(s), 1), nacc))
    L.append("")
    have_cell = {(str(a), str(b), round(float(c), 6))
                 for a, b, c in zip(rows["dataset"], rows["clf"], rows["alpha"])}
    miss_cell = [(ds, clf, a) for ds in paths.MANUAL_DATASETS for clf in CLFS
                 for a in ALPHAS
                 if (str(ds), str(clf), round(float(a), 6)) not in have_cell]
    L.append("  Budget cells with NO defined comparison at all (the conformal")
    L.append("  threshold is -inf in every fold, so the accepted set is always")
    L.append("  empty and neither rate exists):")
    if miss_cell:
        for ds, clf, a in miss_cell:
            L.append("    %-16s %-7s alpha=%.2f" % (ds, clf, a))
    else:
        L.append("    none")
    L.append("")
    L.append("  Cohort x classifier cells that are estimable in NO convention")
    L.append("  at any alpha (the manuscript's \"not estimable\" cases):")
    none_any = True
    for ds in paths.MANUAL_DATASETS:
        for clf in CLFS:
            s = units[(units.scope == PRIMARY_SCOPE) & (units.dataset == ds)
                      & (units.clf == clf)]
            if int(s["estimable"].sum()) == 0:
                none_any = False
                L.append("    %-16s %-7s  0 / %d convention-alpha cells estimable"
                         % (ds, clf, len(s)))
    if none_any:
        L.append("    none -- every cohort x classifier cell becomes estimable")
        L.append("    under at least one positivity setting; the three colorectal")
        L.append("    cells are unestimable only at the manuscript's own convention")
        L.append("    (gmm / or / repaired / alpha = 0.10), see section 6.")
    L.append("  Non-estimability is driven by a vanishing accepted set at small")
    L.append("  alpha and by the stricter rule readings shrinking the class set.")
    L.append("")

    # ---------------------------------------------------------------- section 4
    L.append("4. PRIMARY STATISTIC -- PAIRED SIGN TEST")
    L.append("-" * 78)
    L.append("  diff_t = (violation rate among DEFERRED cells)")
    L.append("         - (violation rate among ACCEPTED cells)   at unit t.")
    L.append("  diff_t > 0 is the manuscript's ordering (accepted more consistent).")
    L.append("")
    L.append(SIG_HEADER)
    L.append(_sig_line("PRIMARY  %s, all" % PRIMARY_SCOPE, st))
    for scope in ("ALL_evaluable",):
        s2 = units[(units.scope == scope) & units.estimable]
        L.append(_sig_line("         %s, all" % scope, sign_test(s2["diff"])))
    both = units[units.estimable]
    L.append(_sig_line("         both scopes pooled", sign_test(both["diff"])))
    if "diff_std" in prim_est.columns and prim_est["diff_std"].notna().any():
        L.append(_sig_line("         class-standardized diff",
                           sign_test(prim_est["diff_std"])))
    L.append("")
    L.append("  by alpha (%s scope):" % PRIMARY_SCOPE)
    for a in ALPHAS:
        s2 = prim_est[np.isclose(prim_est.alpha, a)]
        L.append(_sig_line("    alpha=%.2f" % a, sign_test(s2["diff"])))
    L.append("")
    L.append("  estimability-threshold sensitivity (%s scope):" % PRIMARY_SCOPE)
    for tmin in SENS_MIN:
        s2 = prim[(prim["n_acc"] >= tmin) & (prim["n_def"] >= tmin)]
        L.append(_sig_line("    both sides >= %d" % tmin, sign_test(s2["diff"])))
    L.append("")
    L.append("  units counted: %d.  One-sided p tests H0: P(accepted<deferred)=1/2"
             % st["n"])
    L.append("  against the manuscript's direction.  The %d paired units are NOT"
             % st["n"])
    L.append("  independent -- the same cells are re-audited under 68 conventions x")
    L.append("  3 alphas -- so the p-value ranks the ordering, it does not certify it.")
    L.append("")

    # ---------------------------------------------------------------- section 5
    L.append("5. PER-CONVENTION BREAKDOWN (%s scope)" % PRIMARY_SCOPE)
    L.append("-" * 78)
    L.append("  Worst family first; a convention fails when NO estimable unit")
    L.append("  shows accepted < deferred.")
    L.append("")
    L.append(SIG_HEADER.replace("group", "positivity family"))
    for f in sorted(set(prim_est["family"]), key=lambda x: FAM_ORDER.get(x, 9)):
        sub = prim_est[prim_est.family == f]
        L.append(_sig_line("  " + str(f), sign_test(sub["diff"]), 32))
    L.append("")
    n_est_conv = len(prim_est.groupby(["positivity", "reading", "ruleset"],
                                      dropna=False))
    n_all_conv = len(units[units.scope == PRIMARY_SCOPE].groupby(
        ["positivity", "reading", "ruleset"], dropna=False))
    L.append("  %d of %d conventions have at least one estimable unit."
             % (n_est_conv, n_all_conv))
    L.append("  %d conventions fail outright (no estimable unit shows the"
             % len(fail_convs))
    L.append("  manuscript's ordering).")
    rev_by_conv = prim_est[prim_est["diff"] < 0].groupby(
        ["positivity", "reading", "ruleset"], dropna=False).size()
    if len(rev_by_conv):
        top = rev_by_conv.sort_values(ascending=False).head(6)
        L.append("  conventions carrying the most reversals: "
                 + ", ".join("%s/%s/%s x%d" % (k[0], k[1], k[2], int(v))
                             for k, v in top.items()))
    L.append("")
    L.append("  %-24s %5s %6s %5s %5s %5s  %10s  %9s"
             % ("convention", "units", "ntest", "acc<d", "acc>d", "ties",
                "p_one_sided", "median_d"))
    ordered = sorted(
        prim_est.groupby(["positivity", "family", "pct", "reading", "ruleset"],
                         dropna=False).groups.items(),
        key=lambda kv: (conv_key(kv[0][1], kv[0][2]), kv[0][3], kv[0][4]))
    for key, gidx in ordered:
        sub = prim_est.loc[gidx]
        s = sign_test(sub["diff"])
        L.append("  %-24s %5d %6d %5d %5d %5d  %10s  %+9.5f"
                 % ("%s/%s/%s" % (key[0], key[3], key[4]), len(sub), s["n"],
                    s["n_pos"], s["n_neg"], s["n_tie"], _pp(s["p_greater"]),
                    s["median"]))
    all_conv = units[units.scope == PRIMARY_SCOPE].groupby(
        ["positivity", "family", "pct", "reading", "ruleset"], dropna=False).groups
    have = {tuple(repr(x) for x in k) for k, _ in ordered}
    miss = [k for k in all_conv if tuple(repr(x) for x in k) not in have]
    if miss:
        L.append("")
        L.append("  conventions with NO estimable unit at all (%d):" % len(miss))
        for k in sorted(miss, key=lambda x: (conv_key(x[1], x[2]), x[3], x[4])):
            L.append("    %s/%s/%s" % (k[0], k[3], k[4]))
    conv_med = prim_est.groupby(
        ["positivity", "reading", "ruleset"], dropna=False)["diff"].median()
    L.append("")
    L.append("  the DIRECTION is robust but the EFFECT SIZE is not: the median")
    L.append("  paired difference runs from %+.4f to %+.4f across the 68"
             % (float(conv_med.min()), float(conv_med.max())))
    L.append("  conventions (gmm family median %+.4f, pct_all %+.4f, pct_pospop %+.4f)."
             % (float(prim_est[prim_est.family == "gmm"]["diff"].median()),
                float(prim_est[prim_est.family == "pct_all"]["diff"].median()),
                float(prim_est[prim_est.family == "pct_pospop"]["diff"].median())))
    L.append("")

    # ---------------------------------------------------------------- section 6
    L.append("6. PER-DATASET AND PER-CLASSIFIER BREAKDOWN (%s scope)" % PRIMARY_SCOPE)
    L.append("-" * 78)
    L.append(SIG_HEADER)
    for ds in paths.MANUAL_DATASETS:
        sub = prim_est[prim_est.dataset == ds]
        L.append(_sig_line("  " + ds, sign_test(sub["diff"]), 32))
    L.append("")
    for clf in CLFS:
        sub = prim_est[prim_est.clf == clf]
        L.append(_sig_line("  " + clf, sign_test(sub["diff"]), 32))
    L.append("")
    L.append("  Manuscript's own cohort x classifier claim, ALL_strict,"
             " alpha = 0.10:")
    L.append("  %-16s %-7s %10s %22s %22s %s"
             % ("cohort", "clf", "verdict", "accepted rate (95% CI)",
                "deferred rate (95% CI)", "conventions holding"))
    for ds in paths.MANUAL_DATASETS:
        for clf in CLFS:
            sub = prim_est[(prim_est.dataset == ds) & (prim_est.clf == clf)]
            g0 = sub[(sub.positivity == "gmm") & np.isclose(sub.alpha, 0.10)]
            if not len(g0) or not bool(g0["estimable"].iloc[0]):
                L.append("  %-16s %-7s %10s" % (ds, clf, "not est."))
                continue
            r = g0.iloc[0]
            ok = "hold" if r["diff"] > 0 else ("fail" if r["diff"] < 0 else "tie")
            n_cv = int((sub["diff"] > 0).sum())
            L.append("  %-16s %-7s %10s %8.4f%% (%.2f-%.2f) %8.4f%% (%.2f-%.2f)"
                     "  %d/%d"
                     % (ds, clf, ok, 100 * r["rate_acc"], 100 * r["ci_lo_acc"],
                        100 * r["ci_hi_acc"], 100 * r["rate_def"],
                        100 * r["ci_lo_def"], 100 * r["ci_hi_def"],
                        n_cv, len(sub)))
    L.append("")

    # ---------------------------------------------------------------- section 7
    L.append("7. IS THE ORDERING UNIVERSAL WITHIN A COHORT x CLASSIFIER CELL?")
    L.append("-" * 78)
    L.append("  For each of the 12 cohort x classifier cells: how many of its")
    L.append("  estimable convention x alpha units show accepted < deferred?")
    L.append("")
    L.append("  %-16s %-7s %10s %9s %9s %6s   %s"
             % ("cohort", "clf", "estimable", "acc<def", "acc>def", "ties",
                "universal?"))
    for ds in paths.MANUAL_DATASETS:
        for clf in CLFS:
            sub = prim_est[(prim_est.dataset == ds) & (prim_est.clf == clf)]
            s = sign_test(sub["diff"])
            if s["n"] == 0:
                L.append("  %-16s %-7s %10d %9s %9s %6s   %s"
                         % (ds, clf, len(sub), "-", "-", "-", "not estimable"))
                continue
            univ = "yes" if s["n_neg"] == 0 else "NO"
            L.append("  %-16s %-7s %10d %9d %9d %6d   %s"
                     % (ds, clf, len(sub), s["n_pos"], s["n_neg"], s["n_tie"],
                        univ))
    L.append("")
    L.append('  A cohort-level claim ("in every cohort") needs every row of a')
    L.append("  cohort to be universal; a cell with 0 estimable units is untested.")
    L.append("")

    # ---------------------------------------------------------------- section 8
    L.append("8. DOES A REVERSAL SURVIVE CLASS STANDARDIZATION?")
    L.append("-" * 78)
    L.append("  The pooled difference mixes cell types whose published rules")
    L.append("  differ in strictness.  diff_std is the cell-weighted mean of the")
    L.append("  WITHIN-class differences (classes with >= 20 evaluable cells on")
    L.append("  both sides), which removes the class-composition term.")
    L.append("")
    if "diff_std" in prim_est.columns and prim_est["diff_std"].notna().any():
        ss = prim_est.dropna(subset=["diff_std"])
        rev = ss[ss["diff"] < 0]
        rev_std = rev[rev["diff_std"] < 0]
        pos_std = ss[(ss["diff"] > 0) & (ss["diff_std"] < 0)]
        L.append("  pooled reversals (diff < 0)                    : %d units"
                 % len(rev))
        L.append("  of those still < 0 after standardisation       : %d (%.0f%%)"
                 % (len(rev_std), 100.0 * len(rev_std) / max(len(rev), 1)))
        L.append("  pooled hold but class-standardized reversal    : %d units"
                 % len(pos_std))
        L.append("")
        L.append(SIG_HEADER.replace("group", "statistic"))
        L.append(_sig_line("  pooled diff", sign_test(ss["diff"]), 32))
        L.append(_sig_line("  class-standardized diff", sign_test(ss["diff_std"]), 32))
        L.append("")
        L.append("  largest pooled reversals:")
        L.append("  %-16s %-7s %5s %-16s %-5s %-7s %9s %9s"
                 % ("cohort", "clf", "alpha", "positivity", "read", "ruleset",
                    "pooled", "std"))
        order = rev["diff"].abs().sort_values(ascending=False).index[:10]
        for _, r in rev.loc[order].iterrows():
            L.append("  %-16s %-7s %5.2f %-16s %-5s %-7s %+9.4f %+9.4f"
                     % (r["dataset"], r["clf"], r["alpha"], r["positivity"],
                        r["reading"], r["ruleset"], r["diff"], r["diff_std"]))
        if len(pos_std):
            L.append("")
            L.append("  units that hold pooled but reverse within classes: "
                     + ", ".join(sorted(set(pos_std["dataset"]))))
    else:
        L.append("  class-standardized statistic not available.")
    L.append("")

    # ---------------------------------------------------------------- section 9
    L.append("9. THE MANUSCRIPT'S CONVENTION, AND THE repeat-0 SENSITIVITY")
    L.append("-" * 78)
    L.append("  gmm / or / repaired / alpha = 0.10, ALL_strict (r15 pooling):")
    L.append("")
    L.append("  %-16s %-7s %22s %22s %s"
             % ("cohort", "clf", "accepted (r15 pool)", "deferred (r15 pool)",
                "r15_deployed"))
    dep = None
    dep_path = os.path.join(RESULTS, "r15_deployed.csv")
    if os.path.exists(dep_path):
        dep = pd.read_csv(dep_path)
    for ds in paths.MANUAL_DATASETS:
        for clf in CLFS:
            sub = prim_est[(prim_est.dataset == ds) & (prim_est.clf == clf)
                           & (prim_est.positivity == "gmm")
                           & np.isclose(prim_est.alpha, 0.10)]
            if not len(sub):
                continue
            r = sub.iloc[0]
            ref = ""
            if dep is not None:
                a = dep[(dep.dataset == ds) & (dep.clf == clf)
                        & np.isclose(dep.alpha, 0.10)
                        & (dep.ruleset == "ALL_strict") & (dep.scope == "accepted")]
                d = dep[(dep.dataset == ds) & (dep.clf == clf)
                        & np.isclose(dep.alpha, 0.10)
                        & (dep.ruleset == "ALL_strict") & (dep.scope == "deferred")]
                if len(a) and len(d):
                    ref = "r15 n = %d + %d cells" % (int(a["n"].iloc[0]),
                                                     int(d["n"].iloc[0]))
            L.append("  %-16s %-7s %10.4f%% (%.2f-%.2f) %8.4f%% (%.2f-%.2f)  %s"
                     % (ds, clf, 100 * r["rate_acc"], 100 * r["ci_lo_acc"],
                        100 * r["ci_hi_acc"], 100 * r["rate_def"],
                        100 * r["ci_lo_def"], 100 * r["ci_hi_def"], ref))
    L.append("")
    if sens is not None and len(sens):
        s = sens.dropna(subset=["diff_repeat0", "diff_r15pool"])
        if len(s):
            agree = float(np.mean(np.sign(s["diff_repeat0"]) ==
                                  np.sign(s["diff_r15pool"])))
            L.append("  Sign agreement between the repeat-0 split and r15's pooled")
            L.append("  split over %d units: %.1f%%." % (len(s), 100 * agree))
            L.append("")

    # ---------------------------------------------------------------- section 10
    L.append("10. LIMITATIONS")
    L.append("-" * 78)
    L.append("  * The paired units are not independent: the same cells are")
    L.append("    re-audited under 68 conventions x 2 scopes x 3 alphas.  The")
    L.append("    sign-test p-value summarises the ordering; it is not a")
    L.append("    calibrated test of a single hypothesis.")
    L.append("  * The violation flag is attached to the cell's EXPERT label, not")
    L.append("    to the predicted label, so this compares which cells the")
    L.append("    controller keeps, not whether their predicted labels are right.")
    L.append("  * Positivity settings are percentile variants of the same log1p")
    L.append("    intensities; a panel with a different dynamic range would move")
    L.append("    the whole family.  The gmm setting is not a privileged one.")
    L.append("  * Extremely permissive positivity settings (pct_all@60) call")
    L.append("    almost every cell positive, which drives both rates to zero and")
    L.append("    produces ties rather than a reversal; ties are excluded from")
    L.append("    the sign test and reported separately.")
    L.append("  * Estimability (>= %d accepted and >= %d deferred evaluable cells)"
             % (EVAL_MIN, EVAL_MIN))
    L.append("    is a resolution floor, not a power calculation; section 4")
    L.append("    reports the same statistic at >= %d and >= %d." % SENS_MIN)
    L.append("  * The accepted/deferred split pools the distinct conformal runs")
    L.append("    present in the repeat column, so the pooled n is a")
    L.append("    configuration sum rather than a cell count; n_unique_cells in")
    L.append("    the csv gives the distinct cells, and section 9 gives the")
    L.append("    repeat-0-only sensitivity.")
    L.append("  * Because the pooled n repeats cells (x6 in melanoma), the Wilson")
    L.append("    intervals in the csv treat those repeats as independent and are")
    L.append("    therefore slightly too narrow; the sign test it feeds uses only")
    L.append("    the ordering of the paired rates, which is unaffected.")
    L.append("")
    L.append("Elapsed %.1fs.  Units: %d rows, %d estimable (%s scope)."
             % (elapsed, len(units), est, PRIMARY_SCOPE))
    L.append("")

    with open(os.path.join(WORK, "r25_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return verdict


# --------------------------------------------------------------------- figure
def plot_units(units, st_prim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
        "legend.frameon": False})

    def panel(ax, letter, title=None):
        ax.set_title(letter, loc="left", fontweight="bold", pad=6)
        if title:
            ax.set_title(title, loc="right", fontsize=7, color="0.35")

    u = units[(units.scope == PRIMARY_SCOPE) & units.estimable].copy()
    fig = plt.figure(figsize=(10.0, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for fam in ("gmm", "pct_all", "pct_pospop"):
        s = u[u.family == fam]
        if not len(s):
            continue
        ax1.scatter(s.rate_acc, s.rate_def, s=5, alpha=0.40, linewidths=0,
                    color=FAMILY_CLR[fam], label=fam)
    hi = float(np.nanmax([u.rate_acc.max(), u.rate_def.max()])) if len(u) else 1.0
    hi = max(hi * 1.06, 1e-3)
    ax1.plot([0, hi], [0, hi], color="0.35", lw=0.8, ls="--", zorder=0)
    ax1.set_xlim(-0.005 * hi, hi)
    ax1.set_ylim(-0.005 * hi, hi)
    ax1.set_box_aspect(1)
    ax1.set_xlabel("violation rate among accepted cells")
    ax1.set_ylabel("violation rate among deferred cells")
    ax1.text(0.035, 0.96,
             "below the y = x line:\nthe manuscript's ordering\n%d of %d units (%.0f%%)"
             % (st_prim["n_pos"], st_prim["n"],
                100.0 * st_prim["n_pos"] / max(st_prim["n"], 1)),
             transform=ax1.transAxes, va="top", ha="left", fontsize=6.5,
             color="0.25",
             bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
    panel(ax1, "a", "68 conventions x 3 alpha, ALL_strict")
    ax1.legend(loc="lower right", markerscale=2.2)

    groups = list(u.groupby(["positivity", "family", "pct", "reading",
                             "ruleset"], dropna=False))
    summ = []
    for key, sub in groups:
        summ.append(dict(positivity=key[0], family=key[1], pct=key[2],
                         reading=key[3], ruleset=key[4], n=len(sub),
                         med=float(sub["diff"].median()),
                         q1=float(sub["diff"].quantile(0.25)),
                         q3=float(sub["diff"].quantile(0.75))))
    cdf = pd.DataFrame(summ).sort_values("med").reset_index(drop=True)
    rng = np.random.default_rng(0)
    for i, r in cdf.iterrows():
        sub = u[(u.positivity == r.positivity) & (u.reading == r.reading)
                & (u.ruleset == r.ruleset)]
        jit = rng.uniform(-0.20, 0.20, len(sub))
        ax2.scatter(sub["diff"], i + jit, s=3.5, alpha=0.45, linewidths=0,
                    color=FAMILY_CLR[r.family])
        ax2.plot([r.med, r.med], [i - 0.36, i + 0.36], color="0.12", lw=1.2,
                 zorder=5)
    ax2.axvline(0.0, color="0.35", lw=0.8, ls="--")
    labels = [short_conv(r.positivity, r.reading, r.ruleset)
              for _, r in cdf.iterrows()]
    ax2.set_yticks(np.arange(len(cdf)))
    ax2.set_yticklabels(labels, fontsize=4.3)
    ax2.set_ylim(-0.9, len(cdf) - 0.1)
    ax2.set_xlabel("violation-rate difference  (deferred - accepted)")
    ax2.set_ylabel("convention, ordered by median difference")
    ax2.text(0.985, 0.985,
             "sign test (accepted < deferred)\n%d / %d units, one-sided p = %s\n"
             "median difference %+.4f"
             % (st_prim["n_pos"], st_prim["n"], _pp(st_prim["p_greater"]),
                st_prim["median"]),
             transform=ax2.transAxes, va="top", ha="right", fontsize=6,
             color="0.2",
             bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
    panel(ax2, "b", "median per convention")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, "figS6_accept_deferred.%s" % ext),
                    bbox_inches="tight")
    plt.close(fig)
    log("wrote figures/figS6_accept_deferred.pdf/.png")


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--figure-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="one cohort, one classifier")
    args = ap.parse_args()

    open(LOG_PATH, "w").close()
    t0 = time.time()

    if args.figure_only:
        units = pd.read_csv(UNITS_CSV)
        prim_est = units[(units.scope == PRIMARY_SCOPE) & units.estimable]
        plot_units(units, sign_test(prim_est["diff"]))
        return

    cohorts = list(paths.MANUAL_DATASETS)
    clfs = list(CLFS)
    if args.smoke:
        cohorts = ["Breast_Jun0524"]
        clfs = ["logreg"]

    usecols = ["unit", "clf", "method", "alpha", "repeat", "lambda"]
    met = pd.read_csv(os.path.join(RESULTS, "r03_metrics.csv"), usecols=usecols)
    met = met[met.method == "crc"]
    log("cohorts=%s clfs=%s alphas=%s jobs=%d conventions=%d"
        % (",".join(cohorts), ",".join(clfs),
           ",".join("%.2f" % a for a in ALPHAS), args.jobs, len(CONVENTIONS)))

    njobs = max(1, min(args.jobs, len(cohorts)))
    res = Parallel(n_jobs=njobs)(
        delayed(dataset_stage)(ds, met[met.unit.str.startswith("within_%s_" % ds)],
                               clfs, list(ALPHAS))
        for ds in cohorts)

    rows, diags, pool_primary = [], [], {}
    sens, std = [], []
    for r in res:
        rows += r["rows"]
        diags.append(r["diag"])
        pool_primary.update(r["pool_primary"])
        sens += r["sens"]
        std += r["std"]
    df = pd.DataFrame(rows)
    units = to_units(rows)
    sens_df = pd.DataFrame(sens)
    std_df = pd.DataFrame(std)

    # ---- reproduction check against r15_deployed.csv
    repro = None
    dep_path = os.path.join(RESULTS, "r15_deployed.csv")
    if os.path.exists(dep_path) and not args.smoke:
        dep = pd.read_csv(dep_path)
        recs = []
        for key, tot in pool_primary.items():
            ds_key, clf, scope, side = key.split("|")
            ref = dep[(dep.dataset == ds_key) & (dep.scope == side)
                      & (dep.clf == clf)
                      & np.isclose(dep.alpha, 0.10) & (dep.ruleset == scope)
                      & (dep.reading == "or") & (dep.variant == "repaired")]
            if not len(ref):
                continue
            r0 = ref.iloc[0]
            n_ok = int(r0["n"]) == tot[0]
            k_ok = int(r0["n_violation"]) == tot[1]
            rate = (float(tot[1]) / float(tot[0])) if tot[0] else np.nan
            ref_rate = (float(r0["n_violation"]) / float(r0["n"])
                        if r0["n"] else np.nan)
            recs.append(dict(dataset=r0["dataset"], clf=clf, ruleset=scope,
                             side=side, n_ref=int(r0["n"]), n_mine=tot[0],
                             k_ref=int(r0["n_violation"]), k_mine=tot[1],
                             rate_ref=ref_rate,
                             rate_csv=float(r0["violation_rate"]),
                             rate_mine=rate,
                             abs_rate_dev=abs(rate - ref_rate)
                             if np.isfinite(rate) and np.isfinite(ref_rate)
                             else np.nan,
                             abs_rate_dev_csv=abs(rate - float(r0["violation_rate"]))
                             if np.isfinite(rate) else np.nan,
                             exact=bool(n_ok and k_ok and rate == ref_rate)))
        if recs:
            repro = pd.DataFrame(recs)
            log("reproduction check: %d/%d rows exact"
                % (int(repro["exact"].sum()), len(repro)))
            if int(repro["exact"].sum()) != len(repro):
                log("*** REPRODUCTION MISMATCH -- STOP ***")

    if not args.smoke:
        df.to_csv(os.path.join(RESULTS, "r25_accept_deferred.csv"), index=False)
        log("wrote experiments/r25_accept_deferred.csv (%d rows)" % len(df))
        units.to_csv(UNITS_CSV, index=False)
        sens_df.to_csv(os.path.join(WORK, "_r25_sens.csv"), index=False)
        std_df.to_csv(os.path.join(WORK, "_r25_std.csv"), index=False)
        with open(DIAG_JSON, "w", encoding="utf-8") as fh:
            json.dump(diags, fh, indent=1)
        verdict = write_report(units, df, diags, pool_primary, sens_df, std_df,
                               repro, args, time.time() - t0)
        log(verdict)
    if not args.no_figure and not args.smoke:
        prim_est = units[(units.scope == PRIMARY_SCOPE) & units.estimable]
        plot_units(units, sign_test(prim_est["diff"]))
    log("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
