"""Conformal prediction and baseline methods for classification.

Implements:
  - CP (inductive conformal prediction) with empirical p-values
  - Bootstrap resampling baseline
  - Gaussian approximation baseline
  - Linear Probe with learned prototypes (main experiment)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def cp_prediction(alpha_scores_test, cal_scores, cal_labels, alpha):
    pred_sets = []
    for scores in alpha_scores_test:
        s = []
        for k, v in enumerate(scores):
            class_cal_scores = cal_scores[cal_labels == k]
            n_cal_k = len(class_cal_scores)
            if n_cal_k == 0:
                s.append(k)
                continue
            p = (np.sum(class_cal_scores >= v) + 1) / (n_cal_k + 1)
            if p > alpha:
                s.append(k)
        pred_sets.append(s)
    return pred_sets


def bootstrap_prediction(alpha_scores_test, cal_scores, cal_labels, alpha, n_boot=1000):
    rng = np.random.RandomState(42)
    n_classes = alpha_scores_test.shape[1]
    class_thresholds = {}
    for k in range(n_classes):
        class_cal_scores = cal_scores[cal_labels == k]
        n_cal_k = len(class_cal_scores)
        if n_cal_k == 0:
            class_thresholds[k] = np.inf
            continue
        thresholds = []
        for _ in range(n_boot):
            boot = class_cal_scores[rng.choice(n_cal_k, size=n_cal_k, replace=True)]
            thresholds.append(np.quantile(boot, 1.0 - alpha))
        class_thresholds[k] = np.mean(thresholds)
        
    pred_sets = []
    for scores in alpha_scores_test:
        s = [k for k, v in enumerate(scores) if v <= class_thresholds[k]]
        pred_sets.append(s)
    return pred_sets


def gaussian_prediction(alpha_scores_test, cal_scores, cal_labels, alpha):
    n_classes = alpha_scores_test.shape[1]
    class_thresholds = {}
    z = stats.norm.ppf(1.0 - alpha)
    for k in range(n_classes):
        class_cal_scores = cal_scores[cal_labels == k]
        if len(class_cal_scores) < 2:
            class_thresholds[k] = np.inf
            continue
        mu, sigma = np.mean(class_cal_scores), np.std(class_cal_scores, ddof=1)
        class_thresholds[k] = mu + z * sigma
        
    pred_sets = []
    for scores in alpha_scores_test:
        s = [k for k, v in enumerate(scores) if v <= class_thresholds[k]]
        pred_sets.append(s)
    return pred_sets


class LinearProbe(nn.Module):
    def __init__(self, input_dim=1024, n_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


def _train_linear_probe(model, X, y, device, epochs=100, lr=1e-3, weight_decay=1e-2):
    model.train()
    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def linear_probe_pipeline(emb, labels, ref_idx, cal_idx, test_idx, n_classes,
                          epochs=100, lr=1e-3, weight_decay=1e-2, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LinearProbe(input_dim=emb.shape[1], n_classes=n_classes)
    X_ref = emb[ref_idx]
    y_ref = labels[ref_idx]
    _train_linear_probe(model, X_ref, y_ref, device, epochs=epochs, lr=lr, weight_decay=weight_decay)

    prototypes = model.linear.weight.detach().cpu().numpy()

    cos_cal_scores = compute_nonconformity_scores(emb, labels, cal_idx, prototypes)
    cos_alpha_test = compute_alpha_scores_test(emb, test_idx, prototypes, n_classes)

    cal_emb = torch.tensor(emb[cal_idx], dtype=torch.float32).to(device)
    test_emb = torch.tensor(emb[test_idx], dtype=torch.float32).to(device)
    with torch.no_grad():
        cal_logits = model(cal_emb)
        cal_probs = F.softmax(cal_logits, dim=-1).cpu().numpy()
        test_logits = model(test_emb)
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()

    smx_cal_scores = np.array([
        1.0 - cal_probs[i, labels[cal_idx][i]]
        for i in range(len(cal_idx))
    ])
    smx_alpha_test = 1.0 - test_probs
    return cos_cal_scores, cos_alpha_test, smx_cal_scores, smx_alpha_test, labels[cal_idx]
