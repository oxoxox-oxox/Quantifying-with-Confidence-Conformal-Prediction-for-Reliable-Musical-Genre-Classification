"""Evaluation metrics for prediction sets."""

import numpy as np


def evaluate(pred_sets, true_labels):
    n = len(true_labels)
    covered = sum(1 for i in range(n) if true_labels[i] in pred_sets[i])
    sizes = [len(s) for s in pred_sets]
    empty = sum(1 for s in pred_sets if len(s) == 0)
    singleton = sum(1 for s in pred_sets if len(s) == 1)
    return covered / n, np.mean(sizes), sizes, empty, singleton
