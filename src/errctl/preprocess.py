"""Feature preprocessing shared by every experiment.

Spatial-proteomics channels span orders of magnitude, and two cohorts profiled
with different panels / detectors are not on a common scale at all. We therefore

1. apply log1p, and
2. standardise each marker.

Standardisation statistics are computed *per domain* from unlabelled cells only
(source statistics from the training cells, target statistics from the target
cohort itself). That is a label-free transformation, so it cannot leak the
labels whose error rate we are trying to control.
"""

import numpy as np


def stats(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


class DomainNormaliser:
    """log1p + z-score, with statistics held per domain."""

    def __init__(self, mode="z_dataset"):
        self.mode = mode
        self.mu_, self.sd_ = {}, {}

    def fit(self, domain, X):
        Z = np.log1p(np.asarray(X, dtype=np.float64))
        if self.mode == "none":
            self.mu_[domain] = np.zeros(Z.shape[1])
            self.sd_[domain] = np.ones(Z.shape[1])
        else:
            self.mu_[domain], self.sd_[domain] = stats(Z)
        return self

    def transform(self, domain, X):
        Z = np.log1p(np.asarray(X, dtype=np.float64))
        return (Z - self.mu_[domain]) / self.sd_[domain]
