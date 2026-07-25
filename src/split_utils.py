"""Stratified 60/20/20 data splitting for conformal prediction.

Splits data into three disjoint subsets:
  - Reference set (60%): compute per-genre prototype centroids
  - Calibration set (20%): compute nonconformity score distribution
  - Test set (20%): held-out evaluation

The split is stratified by class to preserve per-genre proportions in all subsets.
"""

import numpy as np


def stratified_split(labels, n_classes, ref_ratio=0.6, cal_ratio=0.2, seed=42):
    n = len(labels)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)

    ref_indices = []
    cal_indices = []
    test_indices = []

    for c in range(n_classes):
        c_idx = indices[labels == c]
        rng.shuffle(c_idx)
        n_ref = int(len(c_idx) * ref_ratio)
        n_cal = int(len(c_idx) * cal_ratio)
        ref_indices.append(c_idx[:n_ref])
        cal_indices.append(c_idx[n_ref:n_ref + n_cal])
        test_indices.append(c_idx[n_ref + n_cal:])

    ref_idx = np.concatenate(ref_indices)
    cal_idx = np.concatenate(cal_indices)
    test_idx = np.concatenate(test_indices)

    return ref_idx, cal_idx, test_idx
