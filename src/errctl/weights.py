"""Density-ratio estimation, used to make conformal bounds survive shift.

Under covariate shift p_target(x) != p_source(x) the calibration cells are not
exchangeable with deployment cells. Reweighting them by w(x) = p_target(x) /
p_source(x) restores the needed symmetry, at the cost of a wider confidence
bound driven by the effective sample size.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def density_ratio(source_X, target_X, clip=20.0, seed=0, cv=3):
    """Estimate w(x) = p_target(x) / p_source(x) with a domain classifier.

    Returns weights for the *source* (calibration) points, normalised to mean 1.
    """
    source_X = np.asarray(source_X, dtype=np.float64)
    target_X = np.asarray(target_X, dtype=np.float64)
    X = np.vstack([source_X, target_X])
    d = np.concatenate([np.zeros(len(source_X)), np.ones(len(target_X))])
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    n_splits = min(cv, np.bincount(d.astype(int)).min())
    if n_splits < 2:
        return np.ones(len(source_X))
    p = cross_val_predict(clf, X, d, cv=n_splits, method="predict_proba")[:, 1]
    p_src = np.clip(p[:len(source_X)], 1e-6, 1 - 1e-6)
    w = p_src / (1.0 - p_src)
    w = np.clip(w, 0.0, clip)
    if not np.isfinite(w).all() or w.sum() == 0:
        return np.ones(len(source_X))
    return w * (len(w) / w.sum())


def effective_sample_size(w):
    w = np.asarray(w, dtype=np.float64)
    return float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12))
