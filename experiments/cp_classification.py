"""Step 8-9: Conformal Prediction for musical style classification.

Scheme A: CP classification with genre prototype centroids.
- Reference set (60%): compute per-genre prototype embeddings
- Calibration set (20%): compute nonconformity scores distribution
- Test set (20%): build prediction sets, evaluate coverage & set size
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "outputs" / "embeddings"
OUT_DIR = ROOT / "outputs" / "cp_results"

ALPHAS = [0.01, 0.05, 0.10]
N_SPLITS = 5
REF_RATIO = 0.6
CAL_RATIO = 0.2
RNG = np.random.RandomState(42)


def load_data():
    emb = np.load(EMB_DIR / "embeddings.npy")
    ids = np.load(EMB_DIR / "label_ids.npy")
    meta = pd.read_csv(EMB_DIR / "metadata.csv")
    le = LabelEncoder()
    le.fit(meta["genre"].unique())
    return emb, ids, le


def build_prediction_sets(alpha_scores_test, cal_scores, alpha_level):
    n_cal = len(cal_scores)
    prediction_sets = []
    p_values_all = []

    for i in range(len(alpha_scores_test)):
        alpha_x = alpha_scores_test[i]
        p_vals = []
        for k in range(len(alpha_x)):
            p = (np.sum(cal_scores >= alpha_x[k]) + 1) / (n_cal + 1)
            p_vals.append(p)
        p_values_all.append(p_vals)
        pred_set = [k for k, p in enumerate(p_vals) if p > alpha_level]
        prediction_sets.append(pred_set)

    return prediction_sets, p_values_all


def evaluate(prediction_sets, true_labels, n_classes):
    n = len(true_labels)
    covered = 0
    set_sizes = []

    for i in range(n):
        sz = len(prediction_sets[i])
        set_sizes.append(sz)
        if true_labels[i] in prediction_sets[i]:
            covered += 1

    coverage = covered / n
    avg_size = np.mean(set_sizes)
    empty = sum(1 for s in prediction_sets if len(s) == 0)
    singleton = sum(1 for s in prediction_sets if len(s) == 1)
    return coverage, avg_size, set_sizes, empty, singleton


def run_split(emb, labels, le, split_idx):
    n = len(labels)
    n_classes = len(le.classes_)
    indices = np.arange(n)
    rng = np.random.RandomState(42 + split_idx)

    ref_indices = []
    cal_indices = []
    test_indices = []

    for c in range(n_classes):
        c_idx = indices[labels == c]
        rng.shuffle(c_idx)
        n_ref = int(len(c_idx) * REF_RATIO)
        n_cal = int(len(c_idx) * CAL_RATIO)
        ref_indices.append(c_idx[:n_ref])
        cal_indices.append(c_idx[n_ref:n_ref + n_cal])
        test_indices.append(c_idx[n_ref + n_cal:])

    ref_idx = np.concatenate(ref_indices)
    cal_idx = np.concatenate(cal_indices)
    test_idx = np.concatenate(test_indices)

    centroids = np.zeros((n_classes, emb.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = labels[ref_idx] == c
        centroids[c] = emb[ref_idx][mask].mean(axis=0)

    cal_scores = np.array([
        1.0 - cosine_similarity(emb[i:i+1], centroids[labels[i]:labels[i]+1])[0, 0]
        for i in cal_idx
    ])

    alpha_scores_test = np.array([
        [1.0 - cosine_similarity(emb[i:i+1], centroids[k:k+1])[0, 0]
         for k in range(n_classes)]
        for i in test_idx
    ])

    results = {}
    for alpha in ALPHAS:
        pred_sets, p_values = build_prediction_sets(alpha_scores_test, cal_scores, alpha)
        cov, avg_sz, sizes, empty, sing = evaluate(
            pred_sets, labels[test_idx], n_classes
        )
        results[alpha] = {
            "coverage": cov,
            "avg_set_size": avg_sz,
            "set_sizes": sizes,
            "empty_sets": empty,
            "singleton_sets": sing,
            "test_count": len(test_idx),
        }

    return results, centroids, cal_scores


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    emb, labels, le = load_data()
    n_classes = len(le.classes_)
    print(f"  embeddings: {emb.shape}")
    print(f"  classes: {n_classes} ({list(le.classes_)})")

    all_results = []

    for split in range(N_SPLITS):
        print(f"\nSplit {split+1}/{N_SPLITS} ...")
        results, centroids, cal_scores = run_split(emb, labels, le, split)

        for alpha in ALPHAS:
            r = results[alpha]
            print(f"  α={alpha:.2f}: coverage={r['coverage']:.3f}, "
                  f"avg_set_size={r['avg_set_size']:.2f}, "
                  f"empty={r['empty_sets']}/{r['test_count']}, "
                  f"singletons={r['singleton_sets']}/{r['test_count']}")
            all_results.append({
                "split": split,
                "alpha": alpha,
                "coverage": r["coverage"],
                "avg_set_size": r["avg_set_size"],
                "empty_sets": r["empty_sets"],
                "singleton_sets": r["singleton_sets"],
                "test_count": r["test_count"],
                "set_sizes": r["set_sizes"],
            })

    df = pd.DataFrame(all_results)

    print("\n=== AGGREGATED RESULTS ===")
    for alpha in ALPHAS:
        sub = df[df["alpha"] == alpha]
        cov_mean = sub["coverage"].mean()
        cov_std = sub["coverage"].std()
        size_mean = sub["avg_set_size"].mean()
        size_std = sub["avg_set_size"].std()
        target = 1.0 - alpha
        print(f"  α={alpha:.2f}: coverage={cov_mean:.3f}±{cov_std:.3f} "
              f"(target: {target:.2f}), set_size={size_mean:.2f}±{size_std:.2f}")

    df_path = OUT_DIR / "cp_results.csv"
    df.to_csv(df_path, index=False)
    print(f"\nSaved results: {df_path}")

    np.savez(
        OUT_DIR / "cp_details.npz",
        set_sizes=np.array([r["set_sizes"] for _, r in df.iterrows()], dtype=object),
        alphas=ALPHAS,
        n_splits=N_SPLITS,
    )
    print("Saved details: cp_details.npz")
    print("\n=== CP CLASSIFICATION DONE ===")


if __name__ == "__main__":
    main()
