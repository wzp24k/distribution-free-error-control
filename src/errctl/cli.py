"""errctl -- the calibration layer of the paper, as a command line tool.

Four things a user needs and this tool returns:

  certify   a labelled table, split at random into a calibration half and a
            deployment half, so the certificate is valid by construction.
  deploy    a labelled source table and a new batch, of which a chosen number
            of cells is annotated and spent on calibration, so the certificate
            is valid on the batch that is actually being annotated.
  advise    one run of the three practical products: a per-cell commit/review
            budget with a finite-sample certificate, a target-label price list
            for repairing a panel or cohort change, and a label-free audit of
            the batch (with the labelled pilot ratio when labels are present).
            A target table may contain labels for only a pilot subset; the
            remaining cells are still scored, committed or deferred, and their
            missing labels are never imputed.
  demo      a synthetic two batch problem, so the tool can be tried without
            any data of your own.

Every run writes three files into the output directory:

  decisions.csv   one row per deployment cell: predicted class, score, and
                  whether the certificate commits the label or defers it to a
                  human, with the reason.
  report.txt      the numbers a laboratory plans with: acceptance, realised
                  risk, coverage of the budget, the review the certificate
                  still routes, the diagnostic ratio against an oracle, and
                  the same review converted into hours under a stated
                  per-cell assumption.
  budget.svg      the review budget as a figure, text as real text so the
                  labels stay editable.

``advise`` additionally writes:

  certificate.md    the guarantee, its scope, the audit verdict and the next
                    action, in plain language.
  certificate.json  the same object for a script or a laboratory information
                    system.
  price_list.csv    the target-label ladder: labels spent, acceptance, realised
                    risk, coverage, net review removed and expert hours.
  audit.csv         one row per audit check, with its interval where available.

Usage
-----
  python -m errctl.cli demo --out-dir out
  python -m errctl.cli advise --demo --out-dir out
  python -m errctl.cli certify --csv cells.csv --label Cell_Label \
      --markers Cell_CD3 Cell_CD20 --alpha 0.10 --out-dir out
  python -m errctl.cli deploy --source train.csv --target new_batch.csv \
      --label Cell_Label --k 500 --alpha 0.10 --out-dir out
  python -m errctl.cli advise --source train.csv --target new_batch.csv \
      --label Cell_Label --k 500 --scan-k 250,500,1000,2000 --out-dir out
  python -m errctl.cli advise --source train.csv --target unlabelled_batch.csv \
      --calibration-csv pilot_labels.csv --label Cell_Label \
      --calibration-id cell --k 500 --out-dir out

The score is the paper's convention, 1 - maximum class probability, and the
controller is the paper's own, imported from errctl.conformal. Everything is
CPU only. Only numpy, pandas and scikit-learn are required; CatBoost and
XGBoost are used when they are installed and asked for by name.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta
from scipy.stats import norm as _norm

from . import conformal as C
from . import markercheck as MC
from . import markerlogic as ML

SEC_PER_CELL = 20.0      # one minute per three cells; an assumption, stated
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10


# --------------------------------------------------------------------- data
def _numeric_columns(df, exclude):
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def load_table(path, label_col, markers=None, group_col=None, id_col=None):
    df = pd.read_csv(path, low_memory=False)
    if label_col not in df.columns:
        raise SystemExit("label column %r not in %s" % (label_col, path))
    if markers:
        missing = [m for m in markers if m not in df.columns]
        if missing:
            raise SystemExit("marker columns missing: %s" % ", ".join(missing))
    else:
        markers = _numeric_columns(df, {label_col, group_col, id_col})
    df = df.dropna(subset=list(markers) + [label_col]).reset_index(drop=True)
    X = df[list(markers)].to_numpy(dtype=np.float64)
    y = df[label_col].astype(str).to_numpy()
    groups = df[group_col].astype(str).to_numpy() if group_col else None
    ids = df[id_col].astype(str).to_numpy() if id_col else np.arange(len(df)).astype(str)
    return dict(df=df, X=X, y=y, markers=list(markers), groups=groups, ids=ids)


def load_table_unlabelled(path, label_col, markers=None, group_col=None,
                          id_col=None):
    """Read a batch whose labels may be present for only a pilot subset.

    Rows with a missing label are kept: they are exactly the cells that a
    production run has to route without ever reading a label.  A missing label
    column makes the whole table unlabelled; calibration labels can then be
    supplied separately with ``--calibration-csv``.
    """
    df = pd.read_csv(path, low_memory=False)
    if markers:
        missing = [m for m in markers if m not in df.columns]
        if missing:
            raise SystemExit("marker columns missing: %s" % ", ".join(missing))
    else:
        markers = _numeric_columns(df, {label_col, group_col, id_col})
    missing_markers = [m for m in markers if m not in df.columns]
    if missing_markers:
        raise SystemExit("marker columns missing: %s" % ", ".join(missing_markers))
    if label_col in df.columns:
        raw = df[label_col]
        present = raw.notna() & (raw.astype(str).str.strip() != "")
        labels = np.array([str(v).strip() if ok else None
                           for v, ok in zip(raw.to_numpy(), present.to_numpy())],
                          dtype=object)
    else:
        present = np.zeros(len(df), dtype=bool)
        labels = np.array([None] * len(df), dtype=object)
    keep = df[list(markers)].notna().all(axis=1).to_numpy()
    df = df.loc[keep].reset_index(drop=True)
    labels = labels[keep]
    present = present[keep]
    X = df[list(markers)].to_numpy(dtype=np.float64)
    groups = df[group_col].astype(str).to_numpy() if group_col else None
    ids = (df[id_col].astype(str).to_numpy() if id_col
           else np.arange(len(df)).astype(str))
    return dict(df=df, X=X, y=labels, labelled=present,
                markers=list(markers), groups=groups, ids=ids)


def attach_calibration_labels(tgt, path, label_col, id_col=None):
    """Attach a separate id,label table to an unlabelled deployment batch."""
    cal = pd.read_csv(path, low_memory=False)
    if label_col not in cal.columns:
        raise SystemExit("calibration label column %r not in %s"
                         % (label_col, path))
    if id_col:
        if id_col not in cal.columns:
            raise SystemExit("calibration id column %r not in %s"
                             % (id_col, path))
        id_name = id_col
    else:
        id_name = next((c for c in ("cell", "cell_id", "id", "ID", "index")
                        if c in cal.columns), None)
        if id_name is None:
            if len(cal) != len(tgt["ids"]):
                raise SystemExit("calibration table needs an id column")
            # A label-only table is accepted only when it is aligned row for row.
            id_name = None
    if id_name is None:
        pairs = zip(tgt["ids"], cal[label_col].to_numpy())
    else:
        pairs = zip(cal[id_name].astype(str).to_numpy(),
                    cal[label_col].to_numpy())
    if len(set(tgt["ids"])) != len(tgt["ids"]):
        raise SystemExit("target cell ids are not unique; cannot attach "
                         "calibration labels safely")
    lookup = {}
    for cid, lab in pairs:
        if pd.isna(lab) or str(lab).strip() == "":
            continue
        cid, lab = str(cid), str(lab).strip()
        if cid in lookup and lookup[cid] != lab:
            raise SystemExit("conflicting calibration labels for cell %s" % cid)
        lookup[cid] = lab
    if not lookup:
        raise SystemExit("calibration table has no non-empty labels")

    y = tgt["y"].copy()
    present = tgt["labelled"].copy()
    matched = 0
    for i, cid in enumerate(tgt["ids"]):
        if cid not in lookup:
            continue
        lab = lookup[cid]
        if present[i] and y[i] != lab:
            raise SystemExit("conflicting label for cell %s in %s"
                             % (cid, path))
        y[i] = lab
        present[i] = True
        matched += 1
    if matched == 0:
        raise SystemExit("no calibration id matched the target table")
    return y, present, matched, sorted(set(lookup) - set(tgt["ids"]))


def standardise(Xtr, Xs):
    """log1p then z score, statistics from the training cells only.

    Marker intensities are non negative, but a table that carries a background
    subtracted or centred channel can hold negative values; the shift is taken
    from the training cells alone, so it stays label free."""
    shift = np.minimum(0.0, np.asarray(Xtr, dtype=np.float64).min(0))
    tr = np.log1p(np.maximum(Xtr - shift, 0.0))
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [(np.log1p(np.maximum(X - shift, 0.0)) - mu) / sd for X in Xs]


def fit_model(X, y, kind, seed=0):
    classes = np.unique(y)
    ix = {c: i for i, c in enumerate(classes)}
    yy = np.array([ix[v] for v in y], dtype=int)
    models = []
    if kind in ("hgb", "vote"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        models.append(HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed))
    if kind in ("logreg", "vote"):
        from sklearn.linear_model import LogisticRegression
        models.append(LogisticRegression(max_iter=5000, C=1.0))
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        models = [RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                         random_state=seed, n_jobs=-1)]
    if kind == "cat":
        from catboost import CatBoostClassifier
        models = [CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                                     verbose=0, loss_function="MultiClass",
                                     allow_writing_files=False)]
    if kind == "xgb":
        from xgboost import XGBClassifier
        models = [XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", random_state=seed, verbosity=0)]
    for m in models:
        m.fit(X, yy)
    return dict(models=models, classes=classes)


def predict(model, X):
    p = None
    for m in model["models"]:
        q = np.asarray(m.predict_proba(X), dtype=np.float64)
        p = q if p is None else p + q
    p = p / len(model["models"])
    lab = model["classes"][np.argmax(p, axis=1)]
    return lab, 1.0 - p.max(axis=1)


# ------------------------------------------------------------------ oracle
def oracle_lambda(score, err, alpha, min_accept=25):
    order = np.argsort(score, kind="mergesort")
    s, e = score[order], err[order]
    cum = np.cumsum(e)
    k = np.arange(1, s.size + 1)
    ok = (cum / k <= alpha) & (k >= min_accept)
    return float(s[np.nonzero(ok)[0][-1]]) if ok.any() else -np.inf


# ------------------------------------------------------------------ report
def summarise(decisions, alphas=ALPHAS, delta=DELTA, sec=SEC_PER_CELL):
    rows = []
    n = len(decisions)
    err = decisions["error"].to_numpy(dtype=float)
    evaluable = np.isfinite(err)
    for alpha in alphas:
        acc = decisions["accepted_%s" % _tag(alpha)].to_numpy(dtype=bool)
        n_acc = int(acc.sum())
        acc_eval = acc & evaluable
        n_acc_eval = int(acc_eval.sum())
        risk = float(err[acc_eval].mean()) if n_acc_eval else float("nan")
        score = decisions["score"].to_numpy(dtype=float)
        if evaluable.any():
            lam = oracle_lambda(score[evaluable], err[evaluable], alpha)
            o_eval = (score[evaluable] <= lam) if np.isfinite(lam) else np.zeros(
                int(evaluable.sum()), dtype=bool)
            oracle_acceptance = float(o_eval.mean())
            n_oracle_eval = int(o_eval.sum())
            ratio = (n_acc_eval / max(n_oracle_eval, 1)) if n_oracle_eval else (
                float("inf") if n_acc_eval else float("nan"))
        else:
            oracle_acceptance = float("nan")
            ratio = float("nan")
        review = n - n_acc
        covered = (float("nan") if not evaluable.any()
                   else int((not np.isfinite(risk)) or risk <= alpha))
        rows.append(dict(
            alpha=alpha, n=n, n_accepted=n_acc,
            acceptance=n_acc / max(n, 1), realised_risk=risk,
            review_fraction=review / max(n, 1),
            covered=covered,
            oracle_acceptance=oracle_acceptance, ratio=ratio,
            n_evaluated=int(evaluable.sum()),
            n_evaluated_committed=n_acc_eval,
            evaluated_acceptance=(n_acc_eval / max(int(evaluable.sum()), 1)
                                  if evaluable.any() else float("nan")),
            review_hours=review * sec / 3600.0,
            saved_hours=(n - review) * sec / 3600.0,
            full_hours=n * sec / 3600.0))
    return pd.DataFrame(rows)


def _tag(alpha):
    return ("%g" % alpha).replace(".", "")


def write_report(path, title, data, table, extra=(), sec=SEC_PER_CELL):
    A = [title, "=" * len(title), ""]
    A.append("input                  %s" % data)
    A.append("cells                  %d" % int(table.n.iloc[0]))
    A.append("per-cell review time   %.0f s (an assumption, change with --sec-per-cell)"
             % sec)
    A.append("")
    A.append("  budget  accept  risk   cover  review  oracle  ratio  review_h  saved_h")
    for _, r in table.iterrows():
        A.append("  %6.2f  %6.3f  %5.3f  %5.3f  %6.3f  %6.3f  %5.2f  %8.1f  %7.1f"
                 % (r.alpha, r.acceptance, r.realised_risk, r.covered,
                    r.review_fraction, r.oracle_acceptance, r.ratio,
                    r.review_hours, r.saved_hours))
    A.append("")
    A.append("Read the table as: at this error budget the certificate commits to")
    A.append("acceptance x of the batch with a realised error of risk among the")
    A.append("labels it commits, and routes review of the rest to a human, which")
    A.append("is review_h hours of work against full_h of a manual pass.")
    A.append("cover is the share of evaluations whose realised risk met the")
    A.append("budget; ratio above one means the calibration sample does not")
    A.append("represent the deployment batch and the certificate must not be")
    A.append("quoted on it.")
    for line in extra:
        A.append(line)
    txt = "\n".join(A) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)
    return txt


# -------------------------------------------------------------------- plot
def write_svg(path, table, title):
    w, h = 720, 300
    pad_l, pad_b, pad_t = 70, 50, 40
    pw, ph = w - pad_l - 30, h - pad_b - pad_t
    xs = list(table.alpha)
    def px(a):
        lo, hi = min(xs), max(xs)
        return pad_l + (0 if hi == lo else (a - lo) / (hi - lo)) * pw
    def py(v, vmax=1.0):
        return pad_t + ph * (1.0 - v / vmax)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
           'viewBox="0 0 %d %d" font-family="Times New Roman, serif">' % (w, h, w, h),
           '<rect width="%d" height="%d" fill="white"/>' % (w, h),
           '<text x="%d" y="24" font-size="14" font-weight="bold">%s</text>'
           % (pad_l, title)]
    for frac in (0, .25, .5, .75, 1.0):
        y = py(frac)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#dddddd" '
                   'stroke-width="0.5"/>' % (pad_l, y, pad_l + pw, y))
        out.append('<text x="%d" y="%.1f" font-size="10" text-anchor="end">%.2f</text>'
                   % (pad_l - 6, y + 3, frac))
    out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#333333" '
               'stroke-width="1"/>' % (pad_l, py(0), pad_l + pw, py(0)))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%.1f" stroke="#333333" '
               'stroke-width="1"/>' % (pad_l, pad_t, pad_l, py(0)))
    for label, col, key in (("acceptance", "#1f77b4", "acceptance"),
                            ("review", "#d62728", "review_fraction"),
                            ("realised risk", "#2ca02c", "realised_risk"),
                            ("oracle acceptance", "#888888", "oracle_acceptance")):
        pts = []
        for _, r in table.iterrows():
            v = r[key]
            v = 0.0 if not np.isfinite(v) else v
            pts.append("%.1f,%.1f" % (px(r.alpha), py(v)))
        out.append('<polyline fill="none" stroke="%s" stroke-width="2" '
                   'points="%s"/>' % (col, " ".join(pts)))
        for _, r in table.iterrows():
            v = r[key]
            v = 0.0 if not np.isfinite(v) else v
            out.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s"/>'
                       % (px(r.alpha), py(v), col))
    lx = pad_l + 10
    for label, col in (("acceptance", "#1f77b4"), ("review", "#d62728"),
                       ("realised risk", "#2ca02c"), ("oracle", "#888888")):
        out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
                   'stroke-width="2"/>' % (lx, pad_t - 14, lx + 18, pad_t - 14, col))
        out.append('<text x="%d" y="%d" font-size="11">%s</text>'
                   % (lx + 22, pad_t - 10, label))
        lx += 30 + 7 * len(label) + 30
    for a in xs:
        out.append('<text x="%.1f" y="%d" font-size="11" text-anchor="middle">'
                   '%.2f</text>' % (px(a), h - pad_b + 16, a))
    out.append('<text x="%d" y="%d" font-size="12" text-anchor="middle">'
               'error budget (alpha)</text>' % (pad_l + pw // 2, h - 12))
    out.append('</svg>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def write_decisions(path, ids, lab, score, err, truth, ctl, alphas):
    d = pd.DataFrame({"cell": ids, "predicted": lab, "score": score,
                      "error": err, "truth": truth})
    d["evaluated"] = np.isfinite(np.asarray(err, dtype=np.float64))
    for alpha in alphas:
        acc = ctl[alpha].accept(score)
        d["accepted_%s" % _tag(alpha)] = acc
        d["decision_%s" % _tag(alpha)] = np.where(acc, "commit", "review")
    d.to_csv(path, index=False)
    return d


# ------------------------------------------------------- advise utilities
class RankNormaliser:
    """log1p, then a per-marker within-domain rank mapped to N(0, 1).

    The transform is fitted separately in each domain and uses no labels.  It
    is the safer default for a real change of antibody panel because it does
    not force the target's location and scale onto the source's.
    """

    def __init__(self):
        self.ref_ = {}

    @staticmethod
    def _log(X):
        return np.log1p(np.asarray(X, dtype=np.float64))

    def fit(self, domain, X):
        self.ref_[domain] = np.sort(self._log(X), axis=0)
        return self

    def transform(self, domain, X):
        Z = self._log(X)
        ref = self.ref_[domain]
        n = ref.shape[0]
        out = np.empty_like(Z)
        for j in range(Z.shape[1]):
            lo = np.searchsorted(ref[:, j], Z[:, j], side="left")
            hi = np.searchsorted(ref[:, j], Z[:, j], side="right")
            u = (0.5 * (lo + hi) + 0.5) / max(n, 1)
            out[:, j] = _norm.ppf(np.clip(u, 1e-4, 1.0 - 1e-4))
        return out


def _cp_upper(err, level=0.95):
    e = np.asarray(err, dtype=np.float64)
    n = int(e.size)
    if n == 0:
        return float("nan")
    k = int(np.rint(e).sum())
    if k >= n:
        return 1.0
    return float(_beta.ppf(level, k + 1.0, max(n - k, 1)))


def _ks_distance(a, b):
    x = np.sort(np.asarray(a, dtype=np.float64))
    y = np.sort(np.asarray(b, dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return float("nan")
    grid = np.unique(np.concatenate([x, y]))
    cx = np.searchsorted(x, grid, side="right") / float(x.size)
    cy = np.searchsorted(y, grid, side="right") / float(y.size)
    return float(np.max(np.abs(cx - cy)))


def _boot_diff(a, b, seed=0, reps=600):
    """Bootstrap mean(a) - mean(b), with a 95% percentile interval."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(int(reps), dtype=np.float64)
    for i in range(int(reps)):
        out[i] = (a[rng.integers(0, a.size, a.size)].mean()
                  - b[rng.integers(0, b.size, b.size)].mean())
    return (float(a.mean() - b.mean()), float(np.quantile(out, 0.025)),
            float(np.quantile(out, 0.975)))


def _class_centroids(Z, y):
    return {str(c): np.asarray(Z[np.asarray(y) == c].mean(0), dtype=np.float64)
            for c in np.unique(y)}


def _centroid_distances(Z, labels, centroids):
    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    out = np.empty(Z.shape[0], dtype=np.float64)
    names = list(centroids)
    for i, lab in enumerate(labels):
        if lab in centroids:
            out[i] = float(np.linalg.norm(Z[i] - centroids[lab]))
        else:
            out[i] = min(float(np.linalg.norm(Z[i] - centroids[c]))
                         for c in names)
    return out


def _review_rank(score, accepted):
    rank = np.full(score.size, np.nan, dtype=np.float64)
    idx = np.nonzero(~accepted)[0]
    if idx.size:
        order = idx[np.argsort(-np.asarray(score)[idx], kind="stable")]
        rank[order] = np.arange(1, order.size + 1, dtype=np.float64)
    return rank


def _canonical_positivity(pos, markers):
    out = {}
    for m in markers:
        if m in pos:
            out[ML.canonical(m)] = pos[m]
    return out


def _normalise_rule_text(text):
    text = str(text).strip()
    if not text or text.lower() in ("nan", "none", "various"):
        return ""
    # Some depot tables write a conjunction as CD3;CD4.
    return text.replace(";", "&")


def _load_rules(path):
    if not path:
        return {}, []
    df = pd.read_csv(path, encoding="utf-8-sig")
    cell_col = next((c for c in ("CellType", "cell_type", "label", "Label")
                     if c in df.columns), None)
    mark_col = next((c for c in ("PrimaryMarker", "primary_marker",
                                 "markers", "Marker")
                     if c in df.columns), None)
    if cell_col is None or mark_col is None:
        raise SystemExit("rules table needs CellType and PrimaryMarker columns")
    rules, failed = {}, []
    for _, row in df.iterrows():
        ct = str(row[cell_col]).strip()
        text = _normalise_rule_text(row[mark_col])
        if not ct or ct.lower().startswith("unidentified") or not text:
            continue
        try:
            rules[ct] = ML.CompiledRule(ct, text)
        except ML.LogicError as exc:
            failed.append({"cell_type": ct, "rule": text, "error": str(exc)})
    return rules, failed


def _audit_marker_rules(path, frame, markers, pred, accepted, seed=0):
    if not path:
        return {"status": "not_run",
                "message": "No --rules table supplied; the optional marker-logic audit was not run."}
    rules, failed = _load_rules(path)
    if not rules:
        return {"status": "unavailable", "message": "No evaluable rule in the rules table.",
                "compile_failures": failed}
    pos, thresholds = MC.positivity(frame, markers, seed=seed)
    posc = _canonical_positivity(pos, markers)
    evaluable, violated, reason = MC.evaluate_labels(pred, posc, rules)
    use = evaluable
    acc = use & np.asarray(accepted, dtype=bool)
    defer = use & ~np.asarray(accepted, dtype=bool)
    if acc.sum() < 10 or defer.sum() < 10:
        return {"status": "unavailable",
                "message": "At least 10 evaluable accepted and 10 evaluable "
                           "deferred cells are needed for a comparison.",
                "n_evaluable": int(use.sum()), "n_rules": len(rules)}
    va = violated[acc].astype(np.float64)
    vd = violated[defer].astype(np.float64)
    gap, lo, hi = _boot_diff(va, vd, seed=seed + 11)
    status = "warn" if (gap > 0 and lo > 0) else ("check" if gap > 0 else "pass")
    return {
        "status": status,
        "n_rules": int(len(rules)),
        "n_evaluable": int(use.sum()),
        "accepted_evaluable": int(acc.sum()),
        "deferred_evaluable": int(defer.sum()),
        "accepted_violation_rate": float(va.mean()),
        "deferred_violation_rate": float(vd.mean()),
        "violation_gap_accepted_minus_deferred": float(gap),
        "violation_gap_ci95": [float(lo), float(hi)],
        "compile_failures": failed,
        "message": ("Accepted cells violate the supplied marker logic at a "
                    "higher rate than deferred cells." if gap > 0
                    else "Accepted cells are no worse than deferred cells under "
                         "the supplied marker logic."),
    }


def _audit_centroid(Ztr, ytr, Zdep, pred, accepted, seed=0):
    centroids = _class_centroids(Ztr, ytr)
    dist = _centroid_distances(Zdep, pred, centroids)
    d_acc = dist[np.asarray(accepted, dtype=bool)]
    d_def = dist[~np.asarray(accepted, dtype=bool)]
    if d_acc.size < 10 or d_def.size < 10:
        return {"status": "not_testable",
                "message": "At least 10 committed and 10 deferred cells are "
                           "needed for this comparison.",
                "accepted_mean_distance": (float(d_acc.mean()) if d_acc.size else None),
                "deferred_mean_distance": (float(d_def.mean()) if d_def.size else None)}
    gap, lo, hi = _boot_diff(d_def, d_acc, seed=seed + 21)
    # Positive gap means deferred cells are farther from their predicted-class
    # centroid than committed cells, which is the expected direction.
    status = "pass" if (gap > 0 and lo > 0) else ("check" if gap > 0 else "warn")
    return {"status": status,
            "distance_advantage_deferred_minus_accepted": float(gap),
            "distance_advantage_ci95": [float(lo), float(hi)],
            "accepted_mean_distance": float(d_acc.mean()) if d_acc.size else None,
            "deferred_mean_distance": float(d_def.mean()) if d_def.size else None,
            "message": ("Committed cells sit closer to their predicted-class "
                        "centroid than deferred cells." if gap > 0
                        else "Committed cells are not closer to their predicted "
                             "class than deferred cells; inspect the queue.")}


def _audit_stability(Ztr, ytr, Zdep, pred, accepted, model, seed=0):
    if Ztr.shape[0] < 30:
        return {"status": "unavailable", "message": "Fewer than 30 training cells."}
    try:
        rng = np.random.default_rng(seed + 31)
        idx = rng.integers(0, Ztr.shape[0], Ztr.shape[0])
        alt = fit_model(Ztr[idx], ytr[idx], model, seed + 101)
        pred_alt, _ = predict(alt, Zdep)
    except Exception as exc:
        return {"status": "unavailable", "message": "Refit failed: %s" % exc}
    agree = (pred_alt == np.asarray(pred)).astype(np.float64)
    a_acc = agree[np.asarray(accepted, dtype=bool)]
    a_def = agree[~np.asarray(accepted, dtype=bool)]
    if a_acc.size < 10 or a_def.size < 10:
        return {"status": "not_testable",
                "message": "At least 10 committed and 10 deferred cells are "
                           "needed for this comparison.",
                "accepted_agreement": (float(a_acc.mean()) if a_acc.size else None),
                "deferred_agreement": (float(a_def.mean()) if a_def.size else None)}
    gap, lo, hi = _boot_diff(a_acc, a_def, seed=seed + 41)
    status = "pass" if (gap > 0 and lo > 0) else ("check" if gap >= 0 else "warn")
    return {"status": status,
            "agreement_gap_accepted_minus_deferred": float(gap),
            "agreement_gap_ci95": [float(lo), float(hi)],
            "accepted_agreement": float(a_acc.mean()) if a_acc.size else None,
            "deferred_agreement": float(a_def.mean()) if a_def.size else None,
            "message": ("Committed labels are more stable under a refit than "
                        "deferred labels." if gap > 0
                        else "Committed labels are not more stable under a refit.")}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(obj), fh, indent=2, sort_keys=True)
        fh.write("\n")


def _parse_scan(text, fallback, n, min_leave=25):
    vals = []
    for token in str(text or "").replace(",", " ").split():
        try:
            vals.append(int(token))
        except ValueError:
            pass
    if not vals:
        vals = list(fallback)
    vals = sorted(set(v for v in vals if 4 <= v < n - min_leave))
    return vals


def _net_share(n_accepted, n_labels, n_eval):
    return float(n_accepted - n_labels) / max(n_eval, 1)


def _audit_verdict(audit):
    statuses = [v.get("status") for v in audit.values() if isinstance(v, dict)]
    if "fail" in statuses:
        return "fail"
    if any(s in ("warn", "check") for s in statuses):
        return "warn"
    if any(s in ("unavailable", "not_testable") for s in statuses):
        return "partial"
    return "pass"


def _write_audit_csv(path, audit):
    rows = []
    for name, val in audit.items():
        if not isinstance(val, dict):
            continue
        metric, value, ci = "", None, [None, None]
        if name == "ratio":
            metric, value = "certified_over_oracle", val.get("ratio")
        elif name == "score_shift":
            metric, value = "score_ks", val.get("ks")
        elif name == "centroid_distance":
            metric = "distance_advantage_deferred_minus_accepted"
            value = val.get("distance_advantage_deferred_minus_accepted")
            ci = val.get("distance_advantage_ci95", [None, None])
        elif name == "refit_stability":
            metric = "agreement_gap_accepted_minus_deferred"
            value = val.get("agreement_gap_accepted_minus_deferred")
            ci = val.get("agreement_gap_ci95", [None, None])
        elif name == "marker_rules":
            metric = "violation_gap_accepted_minus_deferred"
            value = val.get("violation_gap_accepted_minus_deferred")
            ci = val.get("violation_gap_ci95", [None, None])
        rows.append({
            "check": name, "status": val.get("status"), "metric": metric,
            "value": value, "ci_low": ci[0], "ci_high": ci[1],
            "message": val.get("message", ""),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_certificate(path, payload):
    b = payload["budget"]
    a = payload["audit"]
    p = payload["repair"]
    lines = [
        "# errctl certificate",
        "",
        "Run: %s" % payload["run"]["mode"],
        "Batch: %s" % payload["input"]["batch"],
        "Generated: %s" % payload["run"]["created_utc"],
        "",
        "## Guarantee",
        "",
        ("If the labelled calibration cells are exchangeable with the "
         "deployment cells, then with probability at least %.0f%% the "
         "selective error among committed labels is at most %.2f."
         % (100.0 * (1.0 - payload["alpha_delta"][1]),
            payload["alpha_delta"][0])),
        ("This is a finite-sample, distribution-free statement about the "
         "committed set. It is not a statement about the cells sent to review."),
        ("The guarantee does not require labels on the deployment cells. It "
         "does require that the calibration labels are a uniform pilot from "
         "the same exchangeable batch."),
        ("The calibration labels must be drawn uniformly from the batch. "
         "Drawing them from the review queue breaks exchangeability and voids "
         "the statement."),
        "",
        "## Budget",
        "",
        "| item | value |",
        "|---|---:|",
        "| error budget alpha | %.2f |" % b["alpha"],
        "| failure probability delta | %.2f |" % b["delta"],
        "| deployment cells | %d |" % b["n_deployment"],
        "| committed automatically | %d (%.1f%%) |"
        % (b["n_accepted"], 100.0 * b["acceptance"]),
        "| routed to human review | %d (%.1f%%) |"
        % (b["n_review"], 100.0 * b["review_fraction"]),
        "| realised error among committed | %s |"
        % ("not estimable" if b["realised_risk"] is None
           else "%.3f" % b["realised_risk"]),
        "| one-sided CP upper on that error | %s |"
        % ("not estimable" if b["cp_upper"] is None else "%.3f" % b["cp_upper"]),
        "| deployment cells with labels | %d |" % b.get("n_evaluated", 0),
        "| committed cells with labels | %d |"
        % b.get("n_evaluated_committed", 0),
        "| calibration labels already read | %d |" % b["n_calibration"],
        "| gross review hours | %.1f |" % b["review_hours"],
        "| net hours saved vs a manual pass | %.1f |" % b["net_saved_hours"],
        "",
        ("Hours assume %.0f seconds per cell; change this with "
         "`--sec-per-cell`." % payload["run"]["sec_per_cell"]),
        "",
        "## Audit",
        "",
        ("Score shift, distance to the predicted-class centroid, refit "
         "stability and the optional marker rules use no held-out labels. "
         "When deployment labels are available, the certified-to-oracle ratio "
         "uses them as a labelled-pilot audit."),
        "",
        "| check | status | reading |",
        "|---|---|---|",
    ]
    for key, val in a.items():
        if not isinstance(val, dict):
            continue
        reading = val.get("message", "")
        if key == "ratio" and val.get("ratio") is not None:
            reading = ("certified/oracle ratio %.3f; %s"
                       % (val["ratio"], "< 1 is consistent with a representative calibration sample"
                          if val["ratio"] <= 1 else "> 1 means the calibration sample does not represent this batch"))
        if key == "score_shift" and val.get("ks") is not None:
            reading = ("KS distance %.3f between calibration and deployment scores; "
                       "it screens for shift and does not prove validity" % val["ks"])
        lines.append("| %s | %s | %s |"
                     % (key.replace("_", " "), val.get("status", ""),
                        str(reading).replace("|", "/")))
    lines += [
        "",
        "## Repair price",
        "",
        p.get("message", ""),
        "",
    ]
    if p.get("recommended_k") is not None:
        lines += [
            "| recommended target labels | %d |" % p["recommended_k"],
            "| label / batch ratio | %.3f |" % p["recommended_k_over_batch"],
            "| net review removed | %.3f of the evaluation pool |"
            % p["net_share"],
            "| net expert hours saved | %.1f |" % p["net_hours"],
            "",
        ]
    lines += [
        "## Quotation",
        "",
        ("This certificate may be quoted on this batch, with the audit caveats "
         "above." if payload["quotation_allowed"] else
         "DO NOT quote this certificate as valid on this batch. The audit found "
         "that the calibration sample does not represent deployment; collect "
         "fresh labels uniformly from the batch, or repair and re-certify."),
        "",
        ("The deployment cells were unlabelled, so no realised error or ratio "
         "audit is reported." if b.get("n_evaluated", 0) == 0 else
         "Realised error and the ratio audit use only the %d deployment cells "
         "that carry a label." % b.get("n_evaluated", 0)),
        "",
        "## Files",
        "",
        "- `budget.csv`: one row per budget.",
        "- `decisions.csv`: one row per deployment cell, with commit/review and queue rank.",
        "- `price_list.csv`: the label ladder behind the repair recommendation.",
        "- `audit.csv`: one row per audit check, with its interval where available.",
        "- `certificate.json`: the same record for software integration.",
        "",
    ]
    if payload.get("actions"):
        lines += ["## Next action", ""]
        lines += ["- %s" % action for action in payload["actions"]]
        lines += [""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ------------------------------------------------------------- subcommands
def cmd_certify(args):
    data = load_table(args.csv, args.label, args.markers, args.group, args.id)
    n = len(data["y"])
    perm = np.random.default_rng(args.seed).permutation(n)
    a, b = int(0.50 * n), int(0.75 * n)
    tr, cal, dep = perm[:a], perm[a:b], perm[b:]
    Ztr, Zcal, Zdep = standardise(
        data["X"][tr], [data["X"][tr], data["X"][cal], data["X"][dep]])
    model = fit_model(Ztr, data["y"][tr], args.model, args.seed)
    lab_c, score_c = predict(model, Zcal)
    err_c = (lab_c != data["y"][cal]).astype(float)
    lab, score = predict(model, Zdep)
    truth = data["y"][dep]
    err = (lab != truth).astype(float)
    ctl = {a: C.SelectiveRiskController(a, args.delta).fit(score_c, err_c)
           for a in ALPHAS}
    os.makedirs(args.out_dir, exist_ok=True)
    d = write_decisions(os.path.join(args.out_dir, "decisions.csv"),
                        data["ids"][dep], lab, score, err, truth, ctl, ALPHAS)
    tab = summarise(d, sec=args.sec_per_cell)
    tab.to_csv(os.path.join(args.out_dir, "budget.csv"), index=False)
    write_report(os.path.join(args.out_dir, "report.txt"),
                 "errctl certify -- random exchangeable split",
                 args.csv, tab,
                 extra=["", "Model accuracy on the deployment half: %.3f"
                        % float((lab == truth).mean()),
                        "Classifiers: %s, seed %d" % (args.model, args.seed)],
                 sec=args.sec_per_cell)
    write_svg(os.path.join(args.out_dir, "budget.svg"), tab,
              "Review budget, random split")
    print(open(os.path.join(args.out_dir, "report.txt"), encoding="utf-8").read())
    return 0


def cmd_deploy(args):
    src = load_table(args.source, args.label, args.markers, args.group, args.id)
    tgt = load_table(args.target, args.label, args.markers, args.group, args.id)
    shared = [m for m in src["markers"] if m in set(tgt["markers"])]
    if not shared:
        raise SystemExit("the two tables share no marker columns")
    isrc = [src["markers"].index(m) for m in shared]
    itgt = [tgt["markers"].index(m) for m in shared]
    Xs, Xt = standardise(src["X"][:, isrc], [src["X"][:, isrc], tgt["X"][:, itgt]])
    model = fit_model(Xs, src["y"], args.model, args.seed)
    lab_all, score_all = predict(model, Xt)
    n = len(lab_all)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    k = int(args.k)
    cal, dep = perm[:k], perm[k:]
    err_all = (lab_all != tgt["y"]).astype(float)
    ctl = {a: C.SelectiveRiskController(a, args.delta).fit(
        score_all[cal], err_all[cal]) for a in ALPHAS}
    os.makedirs(args.out_dir, exist_ok=True)
    d = write_decisions(os.path.join(args.out_dir, "decisions.csv"),
                        tgt["ids"][dep], lab_all[dep], score_all[dep],
                        err_all[dep], tgt["y"][dep], ctl, ALPHAS)
    tab = summarise(d, sec=args.sec_per_cell)
    tab.to_csv(os.path.join(args.out_dir, "budget.csv"), index=False)
    write_report(os.path.join(args.out_dir, "report.txt"),
                 "errctl deploy -- source trained, target calibrated",
                 "%s -> %s" % (args.source, args.target), tab,
                 extra=["", "Source cells %d, target cells %d, shared markers %d"
                        % (len(src["y"]), n, len(shared)),
                        "Target labels spent on calibration: %d" % k,
                        "Model accuracy on the target batch: %.3f"
                        % float((lab_all == tgt["y"]).mean())],
                 sec=args.sec_per_cell)
    write_svg(os.path.join(args.out_dir, "budget.svg"), tab,
              "Review budget, target calibrated")
    print(open(os.path.join(args.out_dir, "report.txt"), encoding="utf-8").read())
    return 0


def _arm(ids, lab, score, err, truth, dep, cal_idx, alphas, delta,
         sec, out_dir, name):
    """Calibrate one arm on its own labels, write its decisions and its table.

    cal_idx indexes the whole batch, so the calibrator is fit before the
    deployment frame is sliced; dep carries which cells the report covers."""
    ctl = {a: C.SelectiveRiskController(a, delta).fit(score[cal_idx], err[cal_idx])
           for a in alphas}
    d = write_decisions(os.path.join(out_dir, "decisions_%s.csv" % name),
                        ids[dep], lab[dep], score[dep], err[dep], truth[dep],
                        ctl, alphas)
    tab = summarise(d, alphas=alphas, delta=delta, sec=sec)
    return tab


def cmd_repair(args):
    """Where the labels go: a frozen model against one refitted on them.

    Both arms draw the same k cells from the deployment batch at random.  The
    frozen arm spends all k on the calibrator and leaves the model alone.  The
    repaired arm refits on the source training set plus the first half and
    calibrates on the second half, so the half that carries the labels into the
    model is never also counted by the certificate."""
    src = load_table(args.source, args.label, args.markers, args.group, args.id)
    tgt = load_table(args.target, args.label, args.markers, args.group, args.id)
    shared = [m for m in src["markers"] if m in set(tgt["markers"])]
    if not shared:
        raise SystemExit("the two tables share no marker columns")
    isrc = [src["markers"].index(m) for m in shared]
    itgt = [tgt["markers"].index(m) for m in shared]
    Xs, Xt = standardise(src["X"][:, isrc], [src["X"][:, isrc], tgt["X"][:, itgt]])
    n = len(tgt["y"])
    k = int(args.k)
    if k < 4 or k >= n:
        raise SystemExit("--k must be at least 4 and smaller than the batch (%d)" % n)
    os.makedirs(args.out_dir, exist_ok=True)
    perm = np.random.default_rng(args.seed).permutation(n)
    half = k // 2
    fit_cells, cal, dep = perm[:half], perm[half:k], perm[k:]
    alphas = (args.alpha,) if args.alpha else ALPHAS

    frozen = fit_model(Xs, src["y"], args.model, args.seed)
    lab_f, score_f = predict(frozen, Xt)
    err_f = (lab_f != tgt["y"]).astype(float)
    repaired = fit_model(np.vstack([Xs, Xt[fit_cells]]),
                         np.concatenate([src["y"], tgt["y"][fit_cells]]),
                         args.model, args.seed + 7)
    lab_r, score_r = predict(repaired, Xt)
    err_r = (lab_r != tgt["y"]).astype(float)

    out = {}
    for name, lab, score, err, cal_idx, nlab in (
            ("frozen", lab_f, score_f, err_f, perm[:k], k),
            ("repaired", lab_r, score_r, err_r, cal, k)):
        tab = _arm(tgt["ids"], lab, score, err, tgt["y"], dep, cal_idx,
                   alphas, args.delta, args.sec_per_cell, args.out_dir, name)
        tab["labels_read"] = nlab
        tab["net_removed"] = (tab["n_accepted"] - nlab) / tab["n"]
        tab["net_hours"] = tab["net_removed"] * tab["n"] * args.sec_per_cell / 3600.0
        tab.to_csv(os.path.join(args.out_dir, "budget_%s.csv" % name), index=False)
        out[name] = tab

    extra = ["",
             "Source cells %d, batch %d, shared markers %d, labels drawn %d"
             % (len(src["y"]), n, len(shared), k),
             "frozen   : model on the source, all %d labels on the calibrator" % k,
             "repaired : model on the source plus %d labels, calibrator on %d"
             % (half, k - half),
             "",
             "  budget  frozen accept  frozen net  repaired accept  repaired net",
             "         net = accepted minus the labels read, over the batch"]
    for a in alphas:
        f = out["frozen"][np.isclose(out["frozen"].alpha, a)].iloc[0]
        r = out["repaired"][np.isclose(out["repaired"].alpha, a)].iloc[0]
        extra.append("  %6.2f  %13.3f  %10.3f  %15.3f  %12.3f"
                     % (a, f.acceptance, f.net_removed, r.acceptance,
                        r.net_removed))
    best = out["repaired"].sort_values("net_removed", ascending=False).iloc[0]
    extra += ["",
              "The repaired arm removes the most net review at a %.2f budget:"
              % best.alpha,
              "%.3f of the batch, %.1f expert hours per 1,000 cells, at a"
              % (best.net_removed, best.net_hours / max(best.n, 1) * 1000.0),
              "realised error of %.3f among the labels it commits."
              % (best.realised_risk if np.isfinite(best.realised_risk) else 0.0),
              "A negative net means the batch is too small to buy this repair:",
              "review it by hand instead."]

    if args.scan_k:
        ks = [int(v) for v in str(args.scan_k).replace(",", " ").split() if v]
        rows = []
        for kk in ks:
            if kk < 4 or kk >= n:
                continue
            h = kk // 2
            fit_c, cal_c = perm[:h], perm[h:kk]
            m = fit_model(np.vstack([Xs, Xt[fit_c]]),
                          np.concatenate([src["y"], tgt["y"][fit_c]]),
                          args.model, args.seed + 7)
            lab_s, score_s = predict(m, Xt)
            err_s = (lab_s != tgt["y"]).astype(float)
            for a in alphas:
                ctl = C.SelectiveRiskController(a, args.delta).fit(
                    score_s[cal_c], err_s[cal_c])
                accs = ctl.accept(score_s[perm[kk:]])
                nacc = int(accs.sum())
                risk = (float(err_s[perm[kk:]][accs].mean()) if nacc
                        else float("nan"))
                rows.append(dict(
                    k=kk, alpha=a, n_batch=n - kk, n_accepted=nacc,
                    acceptance=nacc / max(n - kk, 1), realised_risk=risk,
                    covered=int((not np.isfinite(risk)) or risk <= a),
                    labels_read=kk,
                    net_removed=(nacc - kk) / max(n - kk, 1)))
        pl = pd.DataFrame(rows)
        pl.to_csv(os.path.join(args.out_dir, "price_list.csv"), index=False)
        extra += ["", "price list, repaired arm, one row per label count and budget:",
                  "  written to price_list.csv (%d rows, k from %d to %d)"
                  % (len(pl), min(ks), max(ks))]

    write_report(os.path.join(args.out_dir, "report.txt"),
                 "errctl repair -- the same labels on the model instead",
                 "%s -> %s" % (args.source, args.target),
                 out["repaired"], extra=extra, sec=args.sec_per_cell)
    write_svg(os.path.join(args.out_dir, "budget.svg"), out["repaired"],
              "Review budget, repaired model")
    print(open(os.path.join(args.out_dir, "report.txt"), encoding="utf-8").read())
    return 0


def _advise_demo(out_dir, seed=0):
    """A two-domain synthetic table, so all three products can be tried."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    markers = ["m1", "m2", "m3", "m4", "m5", "m6"]
    classes = ["A", "B", "C", "D"]
    centres = np.array([
        [2.4, 0.4, 0.3, 0.2, 0.9, 0.4],
        [0.3, 2.3, 0.4, 0.3, 0.8, 0.3],
        [0.2, 0.4, 2.2, 0.3, 1.0, 0.4],
        [0.3, 0.2, 0.4, 2.4, 0.9, 0.3],
    ])

    def make(n, prefix, shift, probs):
        y = rng.choice(classes, size=n, p=probs)
        X = np.empty((n, len(markers)), dtype=np.float64)
        for j, c in enumerate(classes):
            m = y == c
            X[m] = rng.normal(centres[j] + shift, 0.48, size=(int(m.sum()), len(markers)))
        X = np.exp(np.clip(X, -3.0, 4.0))
        # A small amount of reference-label noise keeps the demonstration from
        # looking like a classifier fitted and scored on the same clean rule.
        y_obs = y.copy()
        flip = rng.random(n) < 0.03
        y_obs[flip] = rng.choice(classes, size=int(flip.sum()))
        df = pd.DataFrame(X, columns=markers)
        df.insert(0, "cell", ["%s%05d" % (prefix, i) for i in range(n)])
        df.insert(0, "sample", prefix)
        df["Cell_Label"] = y_obs
        return df

    src = make(2400, "src", np.zeros(len(markers)), [0.30, 0.30, 0.20, 0.20])
    tgt = make(3000, "tgt", np.array([0.25, -0.10, 0.35, 0.10, -0.20, 0.15]),
               [0.40, 0.25, 0.20, 0.15])
    src_path = os.path.join(out_dir, "demo_source.csv")
    tgt_path = os.path.join(out_dir, "demo_target.csv")
    rules_path = os.path.join(out_dir, "demo_rules.csv")
    src.to_csv(src_path, index=False)
    tgt.to_csv(tgt_path, index=False)
    pd.DataFrame({
        "CellType": classes,
        "PrimaryMarker": ["m1", "m2", "m3", "m4"],
    }).to_csv(rules_path, index=False)
    return src_path, tgt_path, rules_path, markers


def _group_split(data, seed, frac=(0.50, 0.25, 0.25)):
    """Return train/calibration/deployment indices, group-disjoint if possible."""
    n = len(data["y"])
    groups = data["groups"]
    warnings = []
    if groups is None:
        perm = np.random.default_rng(seed).permutation(n)
        a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
        return perm[:a], perm[a:b], perm[b:], warnings

    g = np.asarray(groups).astype(str)
    uniq = np.unique(g)
    if uniq.size < 3:
        warnings.append("Fewer than three groups; used a cell-level split.")
        perm = np.random.default_rng(seed).permutation(n)
        a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
        return perm[:a], perm[a:b], perm[b:], warnings

    order = np.random.default_rng(seed).permutation(uniq)
    counts = np.array([np.sum(g == u) for u in order], dtype=float)
    cum = np.cumsum(counts) / max(n, 1)
    j = int(np.searchsorted(cum, frac[0]))
    k = int(np.searchsorted(cum, frac[0] + frac[1]))
    j = min(max(j, 0), uniq.size - 3)
    k = min(max(k, j + 1), uniq.size - 2)
    tr_g, cal_g, dep_g = set(order[:j + 1]), set(order[j + 1:k + 1]), set(order[k + 1:])
    tr = np.nonzero(np.isin(g, list(tr_g)))[0]
    cal = np.nonzero(np.isin(g, list(cal_g)))[0]
    dep = np.nonzero(np.isin(g, list(dep_g)))[0]
    if min(tr.size, cal.size, dep.size) < 20:
        warnings.append("The group-disjoint split left a part below 20 cells; "
                        "used a cell-level split instead.")
        perm = np.random.default_rng(seed).permutation(n)
        a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
        return perm[:a], perm[a:b], perm[b:], warnings
    return tr, cal, dep, warnings


def _ratio_audit(score, err, accepted, alpha):
    """Compare the certificate with an oracle that sees deployment labels.

    The comparison is made on the deployment cells that carry a label.  A
    production batch may have none; in that case the ratio is not testable and
    the certificate remains a label-free statement under exchangeability.
    """
    score = np.asarray(score, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    accepted = np.asarray(accepted, dtype=bool)
    evaluable = np.isfinite(err) & np.isfinite(score)
    n_eval = int(evaluable.sum())
    if n_eval == 0:
        return {"status": "not_testable", "ratio": None,
                "acceptance": float(accepted.mean()) if accepted.size else 0.0,
                "oracle_acceptance": None, "n_evaluated": 0,
                "n_evaluated_committed": 0, "ratio_infinite": False,
                "message": ("No deployment labels are available, so the "
                            "certified-to-oracle ratio cannot be computed. "
                            "This is a label-free certificate, not a failed audit.")}
    s, e, a = score[evaluable], err[evaluable], accepted[evaluable]
    n_acc = int(a.sum())
    lam = oracle_lambda(s, e, alpha)
    oracle_accept = float(np.mean(s <= lam)) if np.isfinite(lam) else 0.0
    if oracle_accept > 0:
        ratio = float((n_acc / max(n_eval, 1)) / oracle_accept)
        status = "fail" if ratio > 1.0 + 1e-9 else "pass"
        msg = ("The certificate accepts more than the best threshold chosen "
               "with deployment labels; calibration does not represent this batch."
               if ratio > 1.0 + 1e-9 else
               "The certificate stays inside the best threshold chosen with "
               "deployment labels, as validity requires.")
        return {"status": status, "ratio": ratio,
                "acceptance": n_acc / max(n_eval, 1),
                "oracle_acceptance": oracle_accept,
                "n_evaluated": n_eval, "n_evaluated_committed": n_acc,
                "ratio_infinite": False, "message": msg}
    if n_acc == 0:
        return {"status": "pass", "ratio": None,
                "acceptance": 0.0, "oracle_acceptance": 0.0,
                "n_evaluated": n_eval, "n_evaluated_committed": 0,
                "ratio_infinite": False,
                "message": "Both the certificate and the oracle are silent; "
                           "the ratio is not informative."}
    return {"status": "fail", "ratio": None,
            "acceptance": n_acc / max(n_eval, 1), "oracle_acceptance": 0.0,
            "n_evaluated": n_eval, "n_evaluated_committed": n_acc,
            "ratio_infinite": True,
            "message": "The certificate commits labels while the oracle commits "
                       "none; calibration does not represent this batch."}


def _scan_price_single(args, score_cal, err_cal, score_dep, err_dep,
                       n_batch, sec, alpha, delta):
    rows = []
    n_total = int(n_batch)
    ks = _parse_scan(args.scan_k, (100, 250, 500, 1000, 2000), n_total)
    for kk in ks:
        if kk > len(score_cal) or kk < 25:
            continue
        ctl = C.SelectiveRiskController(alpha, delta).fit(
            score_cal[:kk], err_cal[:kk])
        acc = ctl.accept(score_dep)
        nacc = int(acc.sum())
        risk = float(err_dep[acc].mean()) if nacc else float("nan")
        cp = _cp_upper(err_dep[acc], 1.0 - delta) if nacc else float("nan")
        net = _net_share(nacc, kk, len(score_dep))
        rows.append({
            "kind": "calibration_ladder",
            "k": int(kk), "k_over_batch": float(kk) / max(n_total, 1),
            "n_cal": int(kk), "n_fit": 0, "n_deploy": int(len(score_dep)),
            "frozen_acceptance": nacc / max(len(score_dep), 1),
            "frozen_risk": risk, "frozen_cp_upper": cp,
            "frozen_covered": int((not np.isfinite(risk)) or risk <= alpha),
            "frozen_net_share": net,
            "frozen_net_hours": (nacc - kk) * sec / 3600.0,
            "refit_acceptance": None, "refit_risk": None,
            "refit_cp_upper": None, "refit_covered": None,
            "refit_net_share": None, "refit_net_hours": None,
        })
    pl = pd.DataFrame(rows)
    eligible = pl[(pl.frozen_covered == 1) & (pl.frozen_acceptance > 0)
                  & (pl.frozen_net_share > 0)
                  & (pl.frozen_cp_upper <= alpha)] if len(pl) else pl
    if len(eligible):
        r = eligible.sort_values("k").iloc[0]
        summary = {
            "kind": "calibration_ladder", "status": "ok",
            "recommended_k": int(r.k), "recommended_k_over_batch": float(r.k_over_batch),
            "net_share": float(r.frozen_net_share),
            "net_hours": float(r.frozen_net_hours),
            "message": ("Use at least %d calibration labels: the certificate "
                        "then commits %d more cells than the labels it consumes."
                        % (int(r.k), int(r.frozen_net_share * r.n_deploy))),
        }
    else:
        summary = {
            "kind": "calibration_ladder", "status": "no_repair",
            "recommended_k": None, "recommended_k_over_batch": None,
            "net_share": None, "net_hours": None,
            "message": ("No calibration rung both holds the budget and repays "
                        "its labels. This mode has no source/target panel change; "
                        "rerun with --source and --target to price a real repair."),
        }
    return pl, summary


def _scan_price_target(args, Zs, ys, Zt, yt, truth, score, err, perm,
                       sec, alpha, delta):
    rows = []
    n = int(len(err))
    default = (250, 500, 1000, 2000, 4000)
    ks = _parse_scan(args.scan_k, default, n)
    if int(args.k) not in ks and 4 <= int(args.k) < n - 25:
        ks = sorted(set(ks + [int(args.k)]))
    for kk in ks:
        half = kk // 2
        if half < 2 or kk >= n - 25:
            continue
        fit_c, cal_c, dep_c = perm[:half], perm[half:kk], perm[kk:]
        # The frozen arm spends every target label on calibration.
        frozen_ctl = C.SelectiveRiskController(alpha, delta).fit(
            score[perm[:kk]], err[perm[:kk]])
        f_acc = frozen_ctl.accept(score[dep_c])
        fn = int(f_acc.sum())
        f_risk = float(err[dep_c][f_acc].mean()) if fn else float("nan")
        f_cp = _cp_upper(err[dep_c][f_acc], 1.0 - delta) if fn else float("nan")
        # The repaired arm spends half on the model and half on calibration.
        refit = fit_model(np.vstack([Zs, Zt[fit_c]]),
                          np.concatenate([ys, np.asarray(yt, dtype=object)[fit_c]]),
                          args.model, args.seed + 7)
        r_lab, r_score = predict(refit, Zt)
        r_err = (r_lab != np.asarray(truth)).astype(float)
        r_ctl = C.SelectiveRiskController(alpha, delta).fit(
            r_score[cal_c], r_err[cal_c])
        r_acc = r_ctl.accept(r_score[dep_c])
        rn = int(r_acc.sum())
        r_risk = float(r_err[dep_c][r_acc].mean()) if rn else float("nan")
        r_cp = _cp_upper(r_err[dep_c][r_acc], 1.0 - delta) if rn else float("nan")
        rows.append({
            "kind": "panel_or_cohort_repair",
            "k": int(kk), "k_over_batch": float(kk) / max(n, 1),
            "n_cal": int(kk - half), "n_fit": int(half), "n_deploy": int(len(dep_c)),
            "frozen_acceptance": fn / max(len(dep_c), 1),
            "frozen_risk": f_risk, "frozen_cp_upper": f_cp,
            "frozen_covered": int((not np.isfinite(f_risk)) or f_risk <= alpha),
            "frozen_net_share": _net_share(fn, kk, len(dep_c)),
            "frozen_net_hours": (fn - kk) * sec / 3600.0,
            "refit_acceptance": rn / max(len(dep_c), 1),
            "refit_risk": r_risk, "refit_cp_upper": r_cp,
            "refit_covered": int((not np.isfinite(r_risk)) or r_risk <= alpha),
            "refit_net_share": _net_share(rn, kk, len(dep_c)),
            "refit_net_hours": (rn - kk) * sec / 3600.0,
        })
    pl = pd.DataFrame(rows)
    if not len(pl):
        return pl, {
            "kind": "panel_or_cohort_repair", "status": "unavailable",
            "recommended_k": None, "recommended_k_over_batch": None,
            "net_share": None, "net_hours": None,
            "message": "The target batch is too small for the requested ladder.",
        }
    eligible = pl[(pl.refit_covered == 1) & (pl.refit_acceptance > 0)
                  & (pl.refit_net_share > 0)
                  & (pl.refit_cp_upper <= alpha)]
    if len(eligible):
        r = eligible.sort_values("k").iloc[0]
        summary = {
            "kind": "panel_or_cohort_repair", "status": "ok",
            "recommended_k": int(r.k),
            "recommended_k_over_batch": float(r.k_over_batch),
            "net_share": float(r.refit_net_share),
            "net_hours": float(r.refit_net_hours),
            "message": ("Spend %d target labels (%.1f%% of the batch). The "
                        "refitted arm commits %d more cells than the labels it "
                        "consumes, a net %.1f expert hours at %.0f s per cell."
                        % (int(r.k), 100.0 * float(r.k_over_batch),
                           int(r.refit_net_share * r.n_deploy),
                           float(r.refit_net_hours), sec)),
        }
    else:
        best = pl.sort_values("refit_net_share", ascending=False).iloc[0]
        summary = {
            "kind": "panel_or_cohort_repair", "status": "no_repair",
            "recommended_k": None, "recommended_k_over_batch": None,
            "net_share": float(best.refit_net_share),
            "net_hours": float(best.refit_net_hours),
            "best_k": int(best.k),
            "best_acceptance": float(best.refit_acceptance),
            "message": ("No rung both holds the budget under the repaired arm "
                        "and repays its labels. The best net at k=%d is %.3f of "
                        "the pool. Collect a larger batch, use a stronger score, "
                        "or review this batch without quoting a certificate."
                        % (int(best.k), float(best.refit_net_share))),
        }
    return pl, summary


def _scan_price_pilot(args, Zs, ys, Zt, tgt, lab_all, score_all, truth_all,
                      err_all, sec, alpha, delta):
    """Price a repair from a labelled pilot when most of the batch is unlabelled.

    The pilot is split into a fit part, a calibration part and a held-out
    evaluation part.  The evaluation labels are read only after the threshold
    is fixed, so the risk and CP columns are a descriptive cross-fit check, not
    a second certificate.  Acceptance and net review are still counted on the
    unlabelled deployment pool, which is the quantity a laboratory needs.
    """
    rows = []
    n_batch = int(len(tgt["y"]))
    pilot = np.nonzero(tgt["labelled"])[0]
    n_pilot = int(pilot.size)
    ks = _parse_scan(args.scan_k, (100, 250, 500, 1000, 2000), n_pilot)
    if int(args.k) not in ks and 4 <= int(args.k) <= n_pilot:
        ks = sorted(set(ks + [int(args.k)]))
    for kk in ks:
        if kk > n_pilot or kk < 50:
            continue
        n_eval = max(25, int(round(0.25 * kk)))
        n_rest = kk - n_eval
        n_fit = n_rest // 2
        n_cal = n_rest - n_fit
        if n_fit < 2 or n_cal < 2 or n_eval < 10:
            continue
        rng = np.random.default_rng(args.seed + 1000 + int(kk))
        order = rng.permutation(pilot)
        fit_c = order[:n_fit]
        cal_c = order[n_fit:n_rest]
        eval_c = order[n_rest:kk]
        spent = np.concatenate([fit_c, cal_c])
        deploy_c = np.setdiff1d(np.arange(n_batch), spent, assume_unique=True)
        target_scores = score_all[deploy_c]

        frozen_ctl = C.SelectiveRiskController(alpha, delta).fit(
            score_all[spent], err_all[spent], target_scores=target_scores)
        f_acc = frozen_ctl.accept(score_all[deploy_c])
        f_eval = frozen_ctl.accept(score_all[eval_c])
        f_err = (lab_all[eval_c] != truth_all[eval_c]).astype(float)
        f_risk = float(f_err[f_eval].mean()) if f_eval.any() else float("nan")
        f_cp = _cp_upper(f_err[f_eval], 1.0 - delta) if f_eval.any() else float("nan")

        refit = fit_model(np.vstack([Zs, Zt[fit_c]]),
                          np.concatenate([ys, truth_all[fit_c]]),
                          args.model, args.seed + 7)
        r_lab, r_score = predict(refit, Zt)
        r_err_all = np.full(n_batch, np.nan, dtype=np.float64)
        known = np.asarray(tgt["labelled"], dtype=bool)
        r_err_all[known] = (r_lab[known] != truth_all[known]).astype(float)
        r_ctl = C.SelectiveRiskController(alpha, delta).fit(
            r_score[cal_c], r_err_all[cal_c], target_scores=r_score[deploy_c])
        r_acc = r_ctl.accept(r_score[deploy_c])
        r_eval = r_ctl.accept(r_score[eval_c])
        r_err = r_err_all[eval_c]
        r_risk = float(r_err[r_eval].mean()) if r_eval.any() else float("nan")
        r_cp = _cp_upper(r_err[r_eval], 1.0 - delta) if r_eval.any() else float("nan")
        fn, rn = int(f_acc.sum()), int(r_acc.sum())
        rows.append({
            "kind": "pilot_crossfit",
            "k": int(kk), "k_over_batch": float(kk) / max(n_batch, 1),
            "n_cal": int(n_cal), "n_fit": int(n_fit), "n_eval": int(n_eval),
            "n_deploy": int(deploy_c.size),
            "frozen_acceptance": fn / max(deploy_c.size, 1),
            "frozen_risk": f_risk, "frozen_cp_upper": f_cp,
            "frozen_covered": int((not np.isfinite(f_risk)) or f_risk <= alpha),
            "frozen_net_share": _net_share(fn, kk, deploy_c.size),
            "frozen_net_hours": (fn - kk) * sec / 3600.0,
            "refit_acceptance": rn / max(deploy_c.size, 1),
            "refit_risk": r_risk, "refit_cp_upper": r_cp,
            "refit_covered": int((not np.isfinite(r_risk)) or r_risk <= alpha),
            "refit_net_share": _net_share(rn, kk, deploy_c.size),
            "refit_net_hours": (rn - kk) * sec / 3600.0,
        })
    pl = pd.DataFrame(rows)
    if not len(pl):
        return pl, {
            "kind": "pilot_crossfit", "status": "unavailable",
            "basis": "pilot_crossfit_descriptive",
            "recommended_k": None, "recommended_k_over_batch": None,
            "net_share": None, "net_hours": None,
            "message": ("Pricing a repair needs at least 50 labelled target "
                        "cells and a 25% held-out reserve for a cross-fit "
                        "check. The available pilot is too small."),
        }
    eligible = pl[(pl.refit_covered == 1) & (pl.refit_acceptance > 0)
                  & (pl.refit_net_share > 0)
                  & (pl.refit_cp_upper <= alpha)]
    if len(eligible):
        r = eligible.sort_values("k").iloc[0]
        summary = {
            "kind": "pilot_crossfit", "status": "ok",
            "basis": "pilot_crossfit_descriptive",
            "recommended_k": int(r.k),
            "recommended_k_over_batch": float(r.k_over_batch),
            "net_share": float(r.refit_net_share),
            "net_hours": float(r.refit_net_hours),
            "message": ("Pilot cross-fit estimate, not a guarantee: spend %d "
                        "labelled target cells (%.1f%% of the batch, with %d "
                        "reserved for evaluation). The refitted arm commits %d "
                        "more cells than it consumes."
                        % (int(r.k), 100.0 * float(r.k_over_batch),
                           int(r.n_eval),
                           int(r.refit_net_share * r.n_deploy))),
        }
    else:
        best = pl.sort_values("refit_net_share", ascending=False).iloc[0]
        summary = {
            "kind": "pilot_crossfit", "status": "no_repair",
            "basis": "pilot_crossfit_descriptive",
            "recommended_k": None, "recommended_k_over_batch": None,
            "net_share": float(best.refit_net_share),
            "net_hours": float(best.refit_net_hours),
            "best_k": int(best.k),
            "best_acceptance": float(best.refit_acceptance),
            "message": ("The pilot cross-fit finds no rung that both holds the "
                        "budget and repays its labels. The best net at k=%d is "
                        "%.3f of the pool. Collect a larger pilot, use a stronger "
                        "score, or review this batch without a repair."
                        % (int(best.k), float(best.refit_net_share))),
        }
    return pl, summary


def cmd_advise(args):
    """One run, three deliverables: budget, repair price, audit."""
    warnings = []
    if args.demo:
        p_src, p_tgt, p_rules, markers = _advise_demo(args.out_dir, args.seed)
        args.source, args.target = p_src, p_tgt
        args.rules = args.rules or p_rules
        args.label, args.markers = "Cell_Label", markers
        args.group, args.id = "sample", "cell"
        print("wrote demo source, target and rules under %s\n" % args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    mode = "source_target" if args.source and args.target else "random_split"
    if mode == "source_target":
        src = load_table(args.source, args.label, args.markers, args.group, args.id)
        tgt = load_table_unlabelled(args.target, args.label, args.markers,
                                    args.group, args.id)
        if args.calibration_csv:
            y_new, present_new, matched, unmatched = attach_calibration_labels(
                tgt, args.calibration_csv, args.calibration_label or args.label,
                args.calibration_id or args.id)
            tgt["y"], tgt["labelled"] = y_new, present_new
            warnings.append("Attached %d calibration labels from %s."
                            % (matched, args.calibration_csv))
            if unmatched:
                warnings.append("%d calibration ids did not match the target "
                                "table." % len(unmatched))
        if int(np.asarray(tgt["labelled"], dtype=bool).sum()) < 4:
            raise SystemExit(
                "the target has %d labelled cells; supply at least 4 "
                "calibration labels with --calibration-csv or fill the label "
                "column for a pilot"
                % int(np.asarray(tgt["labelled"], dtype=bool).sum()))
        shared = [m for m in src["markers"] if m in set(tgt["markers"])]
        if not shared:
            raise SystemExit("the two tables share no marker columns")
        isrc = [src["markers"].index(m) for m in shared]
        itgt = [tgt["markers"].index(m) for m in shared]
        if args.norm == "rank":
            rn = RankNormaliser().fit("source", src["X"][:, isrc])
            rn.fit("target", tgt["X"][:, itgt])
            Zs = rn.transform("source", src["X"][:, isrc])
            Zt = rn.transform("target", tgt["X"][:, itgt])
        else:
            Zs, Zt = standardise(src["X"][:, isrc],
                                 [src["X"][:, isrc], tgt["X"][:, itgt]])
        model = fit_model(Zs, src["y"], args.model, args.seed)
        lab_all, score_all = predict(model, Zt)
        truth_all = np.asarray(tgt["y"], dtype=object)
        labelled_all = np.asarray(tgt["labelled"], dtype=bool)
        n = len(truth_all)
        n_input = n
        err_all = np.full(n, np.nan, dtype=np.float64)
        if labelled_all.any():
            err_all[labelled_all] = (
                lab_all[labelled_all] != truth_all[labelled_all]).astype(float)
        if labelled_all.all():
            if not (4 <= int(args.k) < n - 25):
                raise SystemExit("--k must be between 4 and %d for this batch"
                                 % (n - 26))
            perm = np.random.default_rng(args.seed + 3).permutation(n)
            cal, dep = perm[:int(args.k)], perm[int(args.k):]
            target_scores = None
            calibration_kind = "target_labels"
            price_labelled = True
        else:
            pilot = np.nonzero(labelled_all)[0]
            if pilot.size < 4:
                raise SystemExit(
                    "the target has %d labelled cells; supply at least 4 "
                    "calibration labels with --calibration-csv or fill the "
                    "label column for a pilot" % pilot.size)
            n_cal_want = min(int(args.k), int(pilot.size))
            if n_cal_want < 4:
                raise SystemExit("--k must be at least 4")
            if n_cal_want < int(args.k):
                warnings.append("Only %d target labels are available; all of "
                                "them are used for calibration." % pilot.size)
            rng = np.random.default_rng(args.seed + 3)
            cal = rng.choice(pilot, size=n_cal_want, replace=False)
            dep = np.setdiff1d(np.arange(n), cal, assume_unique=True)
            if dep.size <= 25:
                raise SystemExit("too few deployment cells after calibration")
            target_scores = score_all[dep]
            calibration_kind = ("calibration_csv" if args.calibration_csv
                                else "target_pilot")
            price_labelled = False
            if int((~labelled_all[dep]).sum()):
                warnings.append(
                    "%d deployment cells have no label; realised risk and the "
                    "ratio audit are reported only on the labelled subset."
                    % int((~labelled_all[dep]).sum()))
        n_cal, n_dep = int(len(cal)), int(len(dep))
        lab, score, truth, err = lab_all[dep], score_all[dep], truth_all[dep], err_all[dep]
        ids = tgt["ids"][dep]
        dep_frame = tgt["df"].iloc[dep].reset_index(drop=True)
        Zdep, pred_dep = Zt[dep], lab_all[dep]
        ys = src["y"]
        source_n = int(len(src["y"]))
        source_label = args.source
        target_label = args.target
        cal_score, cal_err = score_all[cal], err_all[cal]
    else:
        if not args.csv:
            raise SystemExit("give --csv, or --source and --target, or --demo")
        data = load_table(args.csv, args.label, args.markers, args.group, args.id)
        tr, cal, dep, w = _group_split(data, args.seed)
        warnings.extend(w)
        if args.norm == "rank":
            rn = RankNormaliser().fit("all", data["X"])
            Z = rn.transform("all", data["X"])
            Ztr, Zcal, Zdep = Z[tr], Z[cal], Z[dep]
        else:
            Ztr, Zcal, Zdep = standardise(
                data["X"][tr], [data["X"][tr], data["X"][cal], data["X"][dep]])
        model = fit_model(Ztr, data["y"][tr], args.model, args.seed)
        lab_c, score_c = predict(model, Zcal)
        err_c = (lab_c != data["y"][cal]).astype(float)
        lab, score = predict(model, Zdep)
        truth = data["y"][dep]
        err = (lab != truth).astype(float)
        ids = data["ids"][dep]
        dep_frame = data["df"].iloc[dep].reset_index(drop=True)
        Zs, ys = Ztr, data["y"][tr]
        pred_dep = lab
        shared = list(data["markers"])
        n_input = int(len(data["y"]))
        n_cal, n_dep = int(len(cal)), int(len(dep))
        source_n = 0
        source_label = None
        target_label = args.csv
        target_scores = None
        calibration_kind = "random_split"
        price_labelled = True
        labelled_all = np.ones(n_input, dtype=bool)
        cal_score, cal_err = score_c, err_c

    alphas = tuple(sorted(set(ALPHAS + (float(args.alpha),))))
    ctl = {a: C.SelectiveRiskController(a, args.delta).fit(
        cal_score, cal_err, target_scores=target_scores) for a in alphas}
    os.makedirs(args.out_dir, exist_ok=True)
    d = write_decisions(os.path.join(args.out_dir, "decisions.csv"),
                        ids, lab, score, err, truth, ctl, alphas)
    tab = summarise(d, alphas=alphas, delta=args.delta, sec=args.sec_per_cell)
    tag = _tag(float(args.alpha))
    accepted_sel = d["accepted_%s" % tag].to_numpy(dtype=bool)
    d["decision"] = np.where(accepted_sel, "commit", "review")
    d["review_rank"] = _review_rank(score, accepted_sel)
    d["review_minutes"] = np.where(accepted_sel, 0.0, args.sec_per_cell / 60.0)
    d.to_csv(os.path.join(args.out_dir, "decisions.csv"), index=False)
    tab["delta"] = args.delta
    tab["n_batch_input"] = n_input
    tab["n_calibration"] = n_cal
    tab["n_deployment"] = n_dep
    tab["calibration_hours"] = n_cal * args.sec_per_cell / 3600.0
    tab["manual_hours"] = n_dep * args.sec_per_cell / 3600.0
    tab["net_saved_hours"] = ((tab["n_accepted"] - n_cal)
                              * args.sec_per_cell / 3600.0)
    tab["requested"] = np.isclose(tab["alpha"], float(args.alpha)).astype(int)
    tab["n_labelled_deployment"] = int(np.isfinite(err).sum())
    tab["n_unlabelled_deployment"] = int((~np.isfinite(err)).sum())
    tab["calibration_source"] = calibration_kind
    # One-sided Clopper-Pearson descriptive check on the committed set.
    err_vec = d["error"].to_numpy(dtype=float)
    tab["cp_upper"] = [
        _cp_upper(err_vec[d["accepted_%s" % _tag(a)].to_numpy(dtype=bool)
                          & np.isfinite(err_vec)], 1.0 - args.delta)
        if (d["accepted_%s" % _tag(a)].to_numpy(dtype=bool)
            & np.isfinite(err_vec)).any() else float("nan")
        for a in tab["alpha"]
    ]
    tab.to_csv(os.path.join(args.out_dir, "budget.csv"), index=False)
    row = tab[tab.requested == 1].iloc[0]
    summary_row = ctl[float(args.alpha)].summary()

    # The audit has two label-free legs (score shift and centroid distance),
    # one refit-stability leg, and one optional marker-logic leg.  The ratio
    # needs deployment labels and is therefore a labelled-pilot check.
    ratio = _ratio_audit(score, err, accepted_sel, float(args.alpha))
    ks = _ks_distance(score_all[cal] if mode == "source_target" else score_c, score)
    score_shift = {
        "status": "warn" if np.isfinite(ks) and ks >= 0.25 else "pass",
        "ks": ks,
        "message": ("Large separation between calibration and deployment "
                    "scores; screen the batch before quoting the certificate."
                    if np.isfinite(ks) and ks >= 0.25 else
                    "Calibration and deployment scores are not visibly separated."),
    }
    centroid = _audit_centroid(Zs, ys, Zdep, pred_dep, accepted_sel, seed=args.seed)
    stability = _audit_stability(Zs, ys, Zdep, pred_dep, accepted_sel,
                                 args.model, seed=args.seed)
    marker_rules = _audit_marker_rules(
        args.rules, dep_frame, shared,
        pred_dep, accepted_sel, seed=args.seed)
    audit = {"ratio": ratio, "score_shift": score_shift,
             "centroid_distance": centroid, "refit_stability": stability,
             "marker_rules": marker_rules}
    verdict = _audit_verdict(audit)
    quotation_allowed = ratio.get("status") != "fail"

    if mode == "source_target":
        if price_labelled:
            price, price_summary = _scan_price_target(
                args, Zs, np.asarray(ys, dtype=object), Zt, tgt["y"], truth_all,
                score_all, err_all, perm, args.sec_per_cell,
                float(args.alpha), args.delta)
        else:
            price, price_summary = _scan_price_pilot(
                args, Zs, np.asarray(ys, dtype=object), Zt, tgt, lab_all,
                score_all, truth_all, err_all, args.sec_per_cell,
                float(args.alpha), args.delta)
    else:
        price, price_summary = _scan_price_single(
            args, score_c, err_c, score, err, n_input, args.sec_per_cell,
            float(args.alpha), args.delta)
    price.to_csv(os.path.join(args.out_dir, "price_list.csv"), index=False)

    # Budget plot over the three standard budgets, not only the requested one.
    write_svg(os.path.join(args.out_dir, "budget.svg"), tab,
              "Review budget, errctl advise")
    actions = []
    if not quotation_allowed:
        actions.append("Do not release these labels as certified. Draw fresh "
                       "labels uniformly from the batch, then rerun advice.")
    if ratio.get("status") == "not_testable":
        actions.append("No deployment labels were available. The certificate "
                       "is label-free; keep a small uniformly drawn pilot if "
                       "you also need the certified-to-oracle ratio audit.")
    if price_summary.get("status") == "ok":
        actions.append("If a repair is planned, spend %d target labels "
                       "(%.1f%% of the batch) and re-certify the refitted model."
                       % (int(price_summary["recommended_k"]),
                          100.0 * float(price_summary["recommended_k_over_batch"])))
        if price_summary.get("basis") == "pilot_crossfit_descriptive":
            actions.append("Treat that price as a pilot cross-fit estimate, "
                           "not a guarantee; re-certify after the repair.")
    elif price_summary.get("status") == "no_repair":
        actions.append("Do not buy this repair on this batch. A larger batch, "
                       "a stronger score or a different source panel is needed.")
    elif price_summary.get("status") == "unavailable":
        actions.append("Do not price a repair from this pilot. Collect at "
                       "least 50 labelled target cells with a held-out reserve.")
    if centroid.get("status") in ("warn", "check"):
        actions.append("Read the review queue in descending score "
                       "(review_rank = 1 first); the distance audit did not "
                       "confirm that committed cells are the easier ones.")
    if score_shift.get("status") == "warn":
        actions.append("Check for a batch or panel change before quoting the "
                       "certificate; score distributions differ visibly.")
    if not actions:
        actions.append("Release the committed labels with the stated caveats "
                       "and read the deferred queue in review_rank order.")
    payload = {
        "tool": "errctl advise",
        "run": {
            "mode": mode,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": args.model, "seed": args.seed, "norm": args.norm,
            "sec_per_cell": args.sec_per_cell,
        },
        "input": {
            "batch": target_label,
            "source": source_label,
            "n_source": source_n,
            "n_batch": int(n_input),
            "n_calibration": n_cal,
            "n_deployment": n_dep,
            "n_labelled_target": int(labelled_all.sum()),
            "n_unlabelled_target": int((~labelled_all).sum()),
            "n_evaluated_deployment": int(np.isfinite(err).sum()),
            "n_evaluated_committed": int((np.isfinite(err)
                                          & accepted_sel).sum()),
            "calibration_source": calibration_kind,
            "markers": shared,
        },
        "alpha_delta": [float(args.alpha), float(args.delta)],
        "budget": {
            "alpha": float(args.alpha), "delta": float(args.delta),
            "lambda_hat": float(summary_row["lambda_hat"]),
            "n_deployment": n_dep, "n_accepted": int(row.n_accepted),
            "n_review": int(n_dep - row.n_accepted),
            "acceptance": float(row.acceptance),
            "review_fraction": float(row.review_fraction),
            "realised_risk": (None if not np.isfinite(row.realised_risk)
                              else float(row.realised_risk)),
            "cp_upper": (None if not np.isfinite(row.cp_upper)
                         else float(row.cp_upper)),
            "covered": (None if not np.isfinite(row.covered)
                        else int(row.covered)),
            "oracle_acceptance": (None if not np.isfinite(row.oracle_acceptance)
                                  else float(row.oracle_acceptance)),
            "ratio": (None if not np.isfinite(row.ratio) else float(row.ratio)),
            "n_calibration": n_cal,
            "n_evaluated": int(np.isfinite(err).sum()),
            "n_evaluated_committed": int((np.isfinite(err)
                                          & accepted_sel).sum()),
            "review_hours": float(row.review_hours),
            "net_saved_hours": float(row.net_saved_hours),
        },
        "audit": audit,
        "audit_verdict": verdict,
        "quotation_allowed": bool(quotation_allowed),
        "sampling_rule": ("draw calibration labels uniformly from the batch; "
                          "never from the review queue"),
        "repair": price_summary,
        "warnings": warnings,
        "actions": actions,
    }
    _write_json(os.path.join(args.out_dir, "certificate.json"), payload)
    _write_audit_csv(os.path.join(args.out_dir, "audit.csv"), audit)
    _write_certificate(os.path.join(args.out_dir, "certificate.md"), payload)

    print(open(os.path.join(args.out_dir, "certificate.md"),
               encoding="utf-8").read())
    return 0


def cmd_demo(args):
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.choice(["A", "B", "C", "D"], size=n, p=[.4, .3, .2, .1])
    centres = {"A": 0.0, "B": 1.5, "C": 3.0, "D": 4.5}
    X = np.array([[np.exp(centres[v] + rng.normal(0, 0.7)) + rng.normal(0, 0.3)
                   for _ in range(4)] for v in y])
    X = np.clip(X, 0.0, None)
    conf = np.abs(rng.normal(0, 1, n))
    hard = conf > np.quantile(conf, 0.75)
    flip = hard & (rng.random(n) < 0.5)
    y_obs = y.copy()
    y_obs[flip] = rng.choice(["A", "B", "C", "D"], size=int(flip.sum()))
    df = pd.DataFrame(X, columns=["m%d" % (i + 1) for i in range(4)])
    df["sample"] = "s1"
    df["cell"] = ["c%d" % i for i in range(n)]
    df["Cell_Label"] = y_obs
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, "demo_cells.csv")
    df.to_csv(p, index=False)
    args.csv = p
    args.label = "Cell_Label"
    args.markers = ["m1", "m2", "m3", "m4"]
    args.group = "sample"
    args.id = "cell"
    print("wrote %s; running certify on it\n" % p)
    return cmd_certify(args)


def build_parser():
    ap = argparse.ArgumentParser(prog="errctl", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    HELP = {"certify": "certificate on a random exchangeable split of one table",
            "deploy": "certificate on a new batch, calibrated with k target labels",
            "repair": "the same k labels on the model instead of the calibrator",
            "advise": "budget, repair price and audit in one run",
            "demo": "run the whole procedure on a synthetic two-batch table"}
    for name, fn in (("certify", cmd_certify), ("deploy", cmd_deploy),
                     ("repair", cmd_repair), ("advise", cmd_advise),
                     ("demo", cmd_demo)):
        s = sub.add_parser(name, help=HELP[name])
        if name in ("deploy", "repair", "advise"):
            if name == "advise":
                s.add_argument("--csv", help="labelled table for a random split")
                s.add_argument("--source", help="source training table")
                s.add_argument("--target", help="new batch to be certified")
                s.add_argument("--demo", action="store_true",
                               help="write and run a synthetic source/target pair")
                s.add_argument("--rules", default=None,
                               help="optional CellTypes table with CellType,PrimaryMarker")
                s.add_argument("--calibration-csv", default=None,
                               help="id,label table for a partly or wholly "
                                    "unlabelled target batch")
                s.add_argument("--calibration-label", default=None,
                               help="label column in --calibration-csv "
                                    "(default: --label)")
                s.add_argument("--calibration-id", default=None,
                               help="id column in --calibration-csv "
                                    "(default: --id or a cell/id column)")
            else:
                s.add_argument("--source", required=True)
                s.add_argument("--target", required=True)
            s.add_argument("--k", type=int, default=500,
                           help="target labels drawn from the batch")
        else:
            s.add_argument("--csv", help="labelled table")
        if name in ("repair", "advise"):
            s.add_argument("--scan-k", default="",
                           help="label counts for the price list, e.g. 250,500,1000")
        s.add_argument("--label", default="Cell_Label")
        s.add_argument("--markers", nargs="*", default=None)
        s.add_argument("--group", default=None,
                       help="column to keep the split group disjoint (optional)")
        s.add_argument("--id", default=None, help="cell identifier column")
        s.add_argument("--model", default="hgb",
                       choices=("hgb", "logreg", "rf", "vote", "cat", "xgb"))
        s.add_argument("--norm", default="z", choices=("z", "rank"),
                       help="z: log1p then source z-score; rank: within-domain rank")
        s.add_argument("--alpha", type=float, default=0.10)
        s.add_argument("--delta", type=float, default=DELTA)
        s.add_argument("--sec-per-cell", type=float, default=SEC_PER_CELL)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--out-dir", default="errctl_out")
        s.set_defaults(func=fn)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
