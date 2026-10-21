"""Base classifiers used as the annotation models under study.

These are deliberately plain, off-the-shelf models: the paper's contribution is
the calibration layer wrapped around them, not the models themselves. Input is
expected to be already log1p-transformed and standardised (see
:mod:`errctl.preprocess`).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

CLASSIFIERS = ("logreg", "rf", "hgb")


def make_classifier(kind, seed=0):
    if kind == "logreg":
        return LogisticRegression(max_iter=5000, C=1.0)
    if kind == "logreg_l2":
        return LogisticRegression(max_iter=5000, C=0.1)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                      random_state=seed, n_jobs=-1)
    if kind == "hgb":
        # Plain HistGradientBoosting defaults under-fit badly on these small
        # tabular problems (13 classes, ~5k training cells); this configuration
        # was selected on training data only.
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0, random_state=seed,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)
    raise ValueError("unknown classifier %r" % kind)


class FittedClassifier:
    """Keeps the class order next to the fitted estimator."""

    def __init__(self, estimator, classes):
        self.estimator = estimator
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict_index(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict(self, X):
        return self.classes_[self.predict_index(X)]

    def confidence(self, X):
        return self.predict_proba(X).max(axis=1)


def fit_classifier(X, y, kind="logreg", seed=0):
    clf = make_classifier(kind, seed=seed)
    y = np.asarray(y)
    if isinstance(clf, HistGradientBoostingClassifier):
        _, counts = np.unique(y, return_counts=True)
        if counts.size and counts.min() < 2:
            # HistGradientBoosting stratifies its internal early-stopping split
            # and raises when the least populated class holds a single member --
            # which happens inside bootstrap resamples of rare populations.
            clf.set_params(early_stopping=False, max_iter=200)
    clf.fit(X, y)
    return FittedClassifier(clf, clf.classes_)
