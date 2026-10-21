"""Turn marker Boolean definitions into per-cell consistency checks.

A cell whose assigned label contradicts the expert definition of that label --
a "CD8T" with no CD8 -- is a factual error that no amount of classifier
accuracy reporting will surface, because the ground truth used to score the
classifier came from the same annotation process.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .markerlogic import canonical


def positivity(raw, markers, seed=0, min_component=20):
    """Per-marker Boolean positivity via a two-component 1D mixture on log1p.

    Returns (positives dict, thresholds dict). Markers whose values cannot be
    split (e.g. an all-zero channel) are reported as all-negative.
    """
    pos, thr = {}, {}
    for m in markers:
        x = np.asarray(raw[m], dtype=np.float64)
        x = x[np.isfinite(x)]
        v = np.log1p(np.clip(np.asarray(raw[m], dtype=np.float64), 0, None))
        if x.size == 0 or np.nanmax(x) <= 0 or np.nanstd(v) < 1e-9:
            pos[m] = np.zeros(len(raw), dtype=bool)
            thr[m] = np.inf
            continue
        gm = GaussianMixture(n_components=2, random_state=seed, n_init=3)
        gm.fit(v.reshape(-1, 1))
        means = gm.means_.ravel()
        order = np.argsort(means)
        lo, hi = order[0], order[1]
        if np.sum(gm.predict(v.reshape(-1, 1)) == hi) < min_component:
            # degenerate split: fall back to the upper tertile
            t = np.quantile(v, 2.0 / 3.0)
        else:
            grid = np.linspace(means[lo], means[hi], 256)
            w = gm.weights_
            dens = np.zeros_like(grid)
            for k in (lo, hi):
                dens += w[k] * np.exp(-0.5 * ((grid - means[k]) / gm.covariances_[k, 0, 0] ** 0.5) ** 2)
            # crossing point of the two weighted components
            comp = []
            for k in (lo, hi):
                comp.append(w[k] * np.exp(-0.5 * ((grid - means[k]) / gm.covariances_[k, 0, 0] ** 0.5) ** 2))
            diff = comp[0] - comp[1]
            sign_change = np.nonzero(np.diff(np.sign(diff)))[0]
            t = grid[sign_change[0] + 1] if sign_change.size else np.median(grid)
        thr[m] = float(t)
        pos[m] = v > t
    return pos, thr


def evaluate_labels(labels, pos, rules):
    """Violation flags for cells whose label has an evaluable rule.

    Returns (evaluable mask, violated mask, rule-availability table).
    """
    n = len(labels)
    evaluable = np.zeros(n, dtype=bool)
    violated = np.zeros(n, dtype=bool)
    reason = np.empty(n, dtype=object)
    reason[:] = "no_rule"
    for ct, rule in rules.items():
        m = (np.asarray(labels) == ct)
        if not m.any():
            continue
        missing = [c for c in rule.idents if c not in pos]
        if missing:
            reason[m] = "missing_channel"
            continue
        values = {c: pos[c][m] for c in rule.idents}
        try:
            res = np.asarray(rule.evaluate(values), dtype=bool)
        except Exception:
            reason[m] = "error"
            continue
        evaluable[m] = True
        violated[m] = ~res
        reason[m] = "checked"
    return evaluable, violated, reason


def neighbourhood_enrichment(coords, labels, types, radius, max_cells=200000, seed=0):
    """Observed / expected same-type neighbour counts within `radius`.

    Expected counts come from the global type frequencies, so a value above 1
    means the type clusters with itself.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels)
    if coords.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(coords.shape[0], max_cells, replace=False)
        coords, labels = coords[idx], labels[idx]
    tree = cKDTree(coords)
    pairs = tree.query_ball_point(coords, radius, return_sorted=False)
    freq = pd.Series(labels).value_counts(normalize=True)
    out = []
    for t in types:
        self_idx = np.nonzero(labels == t)[0]
        if self_idx.size < 20:
            continue
        obs = np.mean([np.sum(labels[np.asarray(p)] == t) - 1 for p in pairs[self_idx]])
        exp = freq.get(t, 0.0) * (np.mean([len(p) for p in pairs[self_idx]]) - 1)
        out.append({"type": t, "n": int(self_idx.size), "observed": float(obs),
                    "expected": float(exp),
                    "enrichment": float(obs / exp) if exp > 0 else np.nan})
    return pd.DataFrame(out)
