"""Tests for stratified 60/20/20 data splitting."""

import numpy as np
from src.split_utils import stratified_split


def make_labels(n_classes=3, n_per_class=100):
    labels = np.concatenate([
        np.full(n_per_class, c) for c in range(n_classes)
    ])
    return labels


class TestStratifiedSplit:

    def test_split_sizes(self):
        labels = make_labels(n_classes=3, n_per_class=100)
        ref, cal, test = stratified_split(labels, n_classes=3, seed=42)

        total = len(labels)
        assert abs(len(ref) / total - 0.6) < 0.05
        assert abs(len(cal) / total - 0.2) < 0.05
        assert abs(len(test) / total - 0.2) < 0.05
        assert len(ref) + len(cal) + len(test) == total

    def test_no_overlap(self):
        labels = make_labels(n_classes=3, n_per_class=100)
        ref, cal, test = stratified_split(labels, n_classes=3, seed=42)

        ref_set = set(ref)
        cal_set = set(cal)
        test_set = set(test)
        assert ref_set.isdisjoint(cal_set)
        assert ref_set.isdisjoint(test_set)
        assert cal_set.isdisjoint(test_set)

    def test_all_indices_covered(self):
        labels = make_labels(n_classes=3, n_per_class=100)
        ref, cal, test = stratified_split(labels, n_classes=3, seed=42)

        all_idx = set(np.concatenate([ref, cal, test]))
        assert all_idx == set(range(len(labels)))

    def test_stratification_preserves_class_counts(self):
        labels = make_labels(n_classes=3, n_per_class=100)
        ref, cal, test = stratified_split(labels, n_classes=3, seed=42)

        for c in range(3):
            c_total = np.sum(labels == c)
            c_ref = np.sum(labels[ref] == c)
            c_cal = np.sum(labels[cal] == c)
            c_test = np.sum(labels[test] == c)
            assert c_ref + c_cal + c_test == c_total
            assert c_ref == int(c_total * 0.6)
            assert c_cal == int(c_total * 0.2)

    def test_reproducibility(self):
        labels = make_labels(n_classes=3, n_per_class=100)
        ref1, cal1, test1 = stratified_split(labels, n_classes=3, seed=42)
        ref2, cal2, test2 = stratified_split(labels, n_classes=3, seed=42)

        assert np.array_equal(ref1, ref2)
        assert np.array_equal(cal1, cal2)
        assert np.array_equal(test1, test2)

    def test_different_seeds_produce_different_splits(self):
        labels = make_labels(n_classes=3, n_per_class=100)
        ref1, cal1, test1 = stratified_split(labels, n_classes=3, seed=42)
        ref2, cal2, test2 = stratified_split(labels, n_classes=3, seed=43)

        assert not np.array_equal(ref1, ref2)

    def test_unequal_class_sizes(self):
        labels = np.concatenate([
            np.zeros(50, dtype=int),
            np.ones(100, dtype=int),
            np.full(150, 2, dtype=int),
        ])
        ref, cal, test = stratified_split(labels, n_classes=3, seed=42)

        for c in range(3):
            c_total = np.sum(labels == c)
            c_ref = np.sum(labels[ref] == c)
            c_cal = np.sum(labels[cal] == c)
            c_test = np.sum(labels[test] == c)
            assert c_ref + c_cal + c_test == c_total
            assert c_ref == int(c_total * 0.6)
            assert c_cal == int(c_total * 0.2)

    def test_no_empty_subsets(self):
        labels = make_labels(n_classes=3, n_per_class=10)
        ref, cal, test = stratified_split(labels, n_classes=3, seed=42)

        for c in range(3):
            assert np.sum(labels[ref] == c) > 0, f"Class {c} empty in ref"
            assert np.sum(labels[cal] == c) > 0, f"Class {c} empty in cal"
            assert np.sum(labels[test] == c) > 0, f"Class {c} empty in test"
