"""Confidence calibration diagnostics and post-hoc recalibration."""

import numpy as np
from scipy.optimize import minimize_scalar


def _bin_edges(conf, scheme, n_bins):
    if scheme == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(conf, qs))
        if edges.size < 3:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def reliability_curve(conf, correct, n_bins=15, scheme="quantile"):
    """Return (bin_confidence, bin_accuracy, bin_count) for a reliability plot."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = _bin_edges(conf, scheme, n_bins)
    idx = np.digitize(conf, edges[1:-1])
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean())
        ys.append(correct[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def expected_calibration_error(conf, correct, n_bins=15, scheme="quantile"):
    """Weighted mean |accuracy - confidence| over confidence bins (ECE)."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    _x, _y, n = reliability_curve(conf, correct, n_bins, scheme)
    if n.sum() == 0:
        return float("nan")
    xs, ys, ns = reliability_curve(conf, correct, n_bins, scheme)
    return float((ns / ns.sum() * np.abs(ys - xs)).sum())


def maximum_calibration_error(conf, correct, n_bins=15, scheme="quantile"):
    xs, ys, ns = reliability_curve(conf, correct, n_bins, scheme)
    if ns.sum() == 0:
        return float("nan")
    return float(np.abs(ys - xs).max())


def brier(proba, y_idx):
    proba = np.asarray(proba, dtype=np.float64)
    oh = np.zeros_like(proba)
    oh[np.arange(len(y_idx)), np.asarray(y_idx)] = 1.0
    return float(((proba - oh) ** 2).sum(axis=1).mean())


def nll(proba, y_idx, eps=1e-12):
    proba = np.asarray(proba, dtype=np.float64)
    p = np.clip(proba[np.arange(len(y_idx)), np.asarray(y_idx)], eps, 1.0)
    return float(-np.log(p).mean())


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureScaler:
    """Single-parameter post-hoc recalibration fitted by minimising NLL."""

    def __init__(self):
        self.T = 1.0

    def fit(self, proba, y_idx, eps=1e-12):
        proba = np.asarray(proba, dtype=np.float64)
        y_idx = np.asarray(y_idx)
        logits = np.log(np.clip(proba, eps, 1.0))

        def objective(log_T):
            T = np.exp(log_T)
            return nll(_softmax(logits / T), y_idx)

        res = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        self.T = float(np.exp(res.x))
        return self

    def transform(self, proba, eps=1e-12):
        logits = np.log(np.clip(np.asarray(proba, dtype=np.float64), eps, 1.0))
        return _softmax(logits / self.T)
