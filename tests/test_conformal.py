"""Tests for conformal prediction and baseline methods."""

import numpy as np
import pytest
from src.conformal import (
    compute_centroids,
    compute_nonconformity_scores,
    cp_prediction,
    bootstrap_prediction,
    gaussian_prediction,
    compute_alpha_scores_test,
)
from src.evaluation import evaluate


def make_dummy_data(n_samples=300, n_classes=3, n_features=16, seed=42):
    rng = np.random.RandomState(seed)
    emb = rng.randn(n_samples, n_features)
    labels = np.concatenate([np.full(100, c) for c in range(n_classes)])
    return emb, labels


class TestComputeCentroids:

    def test_centroid_shape(self):
        emb, labels = make_dummy_data()
        ref_idx = np.arange(250)  # includes all 3 classes
        centroids = compute_centroids(emb, labels, ref_idx, n_classes=3)
        assert centroids.shape == (3, 16)

    def test_centroid_values(self):
        emb, labels = make_dummy_data()
        ref_idx = np.arange(250)  # includes all 3 classes
        centroids = compute_centroids(emb, labels, ref_idx, n_classes=3)
        expected_c0 = emb[:100][labels[:100] == 0].mean(axis=0)
        assert np.allclose(centroids[0], expected_c0)


class TestConformalPrediction:

    def test_cp_prediction_alpha_zero_gives_all(self):
        rng = np.random.RandomState(42)
        n_cal, n_test, n_classes = 50, 20, 3
        cal_scores = rng.rand(n_cal)
        alpha_scores_test = rng.rand(n_test, n_classes)

        sets = cp_prediction(alpha_scores_test, cal_scores, alpha=0.0)
        assert len(sets) == n_test
        for s in sets:
            assert set(s) == set(range(n_classes))

    def test_cp_prediction_alpha_one_gives_empty(self):
        rng = np.random.RandomState(42)
        n_cal, n_test, n_classes = 50, 20, 3
        cal_scores = rng.rand(n_cal)
        alpha_scores_test = rng.rand(n_test, n_classes)

        sets = cp_prediction(alpha_scores_test, cal_scores, alpha=1.0)
        for s in sets:
            assert len(s) == 0, "At alpha=1.0 all sets should be empty"

    def test_cp_prediction_output_format(self):
        rng = np.random.RandomState(42)
        n_cal, n_test, n_classes = 50, 20, 3
        cal_scores = rng.rand(n_cal)
        alpha_scores_test = rng.rand(n_test, n_classes)

        sets = cp_prediction(alpha_scores_test, cal_scores, alpha=0.1)
        assert len(sets) == n_test
        for s in sets:
            assert isinstance(s, list)
            assert all(isinstance(k, int) for k in s)
            assert all(0 <= k < n_classes for k in s)

    def test_cp_stronger_signal_smaller_sets(self):
        n_cal, n_classes = 100, 5
        cal_scores = np.linspace(0, 1, n_cal)
        easy = np.array([[0.1 if k == 0 else 0.9 for k in range(n_classes)]])
        hard = np.array([[0.5 for _ in range(n_classes)]])

        easy_sets = cp_prediction(easy, cal_scores, alpha=0.1)
        hard_sets = cp_prediction(hard, cal_scores, alpha=0.1)
        assert len(easy_sets[0]) <= len(hard_sets[0])


class TestBootstrapPrediction:

    def test_bootstrap_output_format(self):
        rng = np.random.RandomState(42)
        n_cal, n_test, n_classes = 50, 10, 3
        cal_scores = rng.rand(n_cal)
        alpha_scores_test = rng.rand(n_test, n_classes)

        sets = bootstrap_prediction(alpha_scores_test, cal_scores, alpha=0.1, n_boot=200)
        assert len(sets) == n_test
        for s in sets:
            assert isinstance(s, list)


class TestGaussianPrediction:

    def test_gaussian_output_format(self):
        rng = np.random.RandomState(42)
        n_cal, n_test, n_classes = 50, 10, 3
        cal_scores = rng.rand(n_cal)
        alpha_scores_test = rng.rand(n_test, n_classes)

        sets = gaussian_prediction(alpha_scores_test, cal_scores, alpha=0.1)
        assert len(sets) == n_test
        for s in sets:
            assert isinstance(s, list)


class TestEvaluate:

    def test_perfect_prediction(self):
        pred_sets = [[0], [1], [2], [0]]
        true_labels = np.array([0, 1, 2, 0])
        coverage, avg_size, sizes, empty, sing = evaluate(pred_sets, true_labels)
        assert coverage == 1.0
        assert avg_size == 1.0
        assert empty == 0
        assert sing == 4

    def test_empty_sets_no_coverage(self):
        pred_sets = [[], [], []]
        true_labels = np.array([0, 1, 2])
        coverage, avg_size, sizes, empty, sing = evaluate(pred_sets, true_labels)
        assert coverage == 0.0
        assert avg_size == 0.0
        assert empty == 3
        assert sing == 0

    def test_all_classes_in_set(self):
        pred_sets = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        true_labels = np.array([0, 1, 2])
        coverage, avg_size, sizes, empty, sing = evaluate(pred_sets, true_labels)
        assert coverage == 1.0
        assert avg_size == 3.0
        assert empty == 0
        assert sing == 0


class TestNonconformityScores:

    def test_compute_nonconformity_shape(self):
        emb, labels = make_dummy_data(n_features=16)
        ref_idx = np.arange(250)
        centroids = compute_centroids(emb, labels, ref_idx, n_classes=3)
        cal_idx = np.arange(60, 80)
        scores = compute_nonconformity_scores(emb, labels, cal_idx, centroids)
        assert scores.shape == (len(cal_idx),)
        assert np.all(scores >= 0)
        assert np.all(scores <= 2)

    def test_self_distance_near_zero(self):
        rng = np.random.RandomState(42)
        centroid = rng.randn(1, 16)
        centroid = centroid / np.linalg.norm(centroid)
        emb = centroid + 0.001 * rng.randn(30, 16)
        labels = np.zeros(30, dtype=int)
        ref_idx = np.arange(30)
        centroids = compute_centroids(emb, labels, ref_idx, n_classes=1)
        scores = compute_nonconformity_scores(emb, labels, ref_idx, centroids)
        assert np.mean(scores) < 0.01
