"""Conformal prediction and baseline methods for classification.

Implements:
  - CP (inductive conformal prediction) with empirical p-values
  - Bootstrap resampling baseline
  - Gaussian approximation baseline
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats


def compute_centroids(emb, labels, ref_idx, n_classes):
    centroids = np.vstack([
        emb[ref_idx][labels[ref_idx] == c].mean(axis=0)
        for c in range(n_classes)
    ])
    return centroids


def compute_nonconformity_scores(emb, labels, idx, centroids):
    return np.array([
        1.0 - cosine_similarity(emb[i:i + 1], centroids[labels[i]:labels[i] + 1])[0, 0]
        for i in idx
    ])


def compute_alpha_scores_test(emb, test_idx, centroids, n_classes):
    return np.array([
        [1.0 - cosine_similarity(emb[i:i + 1], centroids[k:k + 1])[0, 0]
         for k in range(n_classes)]
        for i in test_idx
    ])


def cp_prediction(alpha_scores_test, cal_scores, alpha):
    n_cal = len(cal_scores)
    pred_sets = []
    for scores in alpha_scores_test:
        s = []
        for k, v in enumerate(scores):
            p = (np.sum(cal_scores >= v) + 1) / (n_cal + 1)
            if p > alpha:
                s.append(k)
        pred_sets.append(s)
    return pred_sets


def bootstrap_prediction(alpha_scores_test, cal_scores, alpha, n_boot=1000):
    n_cal = len(cal_scores)
    rng = np.random.RandomState(42)
    thresholds = []
    for _ in range(n_boot):
        boot = cal_scores[rng.choice(n_cal, size=n_cal, replace=True)]
        thresholds.append(np.quantile(boot, 1.0 - alpha))
    threshold = np.mean(thresholds)
    pred_sets = []
    for scores in alpha_scores_test:
        s = [k for k, v in enumerate(scores) if v <= threshold]
        pred_sets.append(s)
    return pred_sets


def gaussian_prediction(alpha_scores_test, cal_scores, alpha):
    mu, sigma = np.mean(cal_scores), np.std(cal_scores, ddof=1)
    z = stats.norm.ppf(1.0 - alpha)
    threshold = mu + z * sigma
    pred_sets = []
    for scores in alpha_scores_test:
        s = [k for k, v in enumerate(scores) if v <= threshold]
        pred_sets.append(s)
    return pred_sets
