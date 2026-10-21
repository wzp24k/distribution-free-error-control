"""Evaluation utilities: splits, uncertainty, and risk-coverage summaries."""

import numpy as np
import pandas as pd


def leave_one_fov_out(df, image_col="image"):
    """Every FOV is held out once; training uses the rest."""
    for img in sorted(df[image_col].unique()):
        test = df[image_col] == img
        yield img, np.nonzero(~test.to_numpy())[0], np.nonzero(test.to_numpy())[0]


def kfold_by_group(df, group_col="image", k=5, seed=0):
    """Grouped k-fold: no group appears in both train and test."""
    groups = np.array(sorted(df[group_col].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    folds = np.array_split(groups, k)
    values = df[group_col].to_numpy()
    for f in folds:
        test = np.isin(values, f)
        yield np.nonzero(~test)[0], np.nonzero(test)[0]


def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = stat(values[idx], axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(stat(values)), float(lo), float(hi)


def cluster_bootstrap_ci(values, clusters, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap over clusters (e.g. FOVs) rather than cells."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    point = float(stat(values))
    if uniq.size < 2:
        return point, float("nan"), float("nan")
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.nonzero(clusters == u)[0] for u in pick])
        boot[b] = stat(values[idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


class AcceptanceReport:
    """Bookkeeping for one (split, method) evaluation."""

    def __init__(self, name):
        self.name = name
        self.rows = []

    def add(self, **kw):
        self.rows.append(kw)

    def frame(self):
        return pd.DataFrame(self.rows)


def selective_metrics(accepted, errs, coverage_total=None):
    """Return dict of headline selective-classification numbers."""
    accepted = np.asarray(accepted, dtype=bool)
    errs = np.asarray(errs, dtype=np.float64)
    n = accepted.size
    n_acc = int(accepted.sum())
    n_err = int(errs[accepted].sum()) if n_acc else 0
    return {
        "n": n,
        "n_accepted": n_acc,
        "accept_rate": n_acc / n if n else float("nan"),
        "n_wrong_accepted": n_err,
        "selective_risk": (n_err / n_acc) if n_acc else float("nan"),
        "coverage_total": coverage_total if coverage_total is not None else n_acc / n,
    }


def risk_violations(nominal, empirical, tol=0.0):
    """Fraction of splits whose empirical selective risk exceeds the budget."""
    empirical = np.asarray(empirical, dtype=np.float64)
    return float(np.mean(empirical > nominal + tol))
