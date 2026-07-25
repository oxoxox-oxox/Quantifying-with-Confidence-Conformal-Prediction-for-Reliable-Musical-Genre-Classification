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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import (
    load_data,
    stratified_split,
    compute_centroids,
    compute_alpha_scores_test,
    cp_prediction,
    evaluate,
)

EMB_DIR = ROOT / "outputs" / "embeddings"
OUT_DIR = ROOT / "outputs" / "cp_results"

ALPHAS = [0.01, 0.05, 0.10]
N_SPLITS = 5
REF_RATIO = 0.6
CAL_RATIO = 0.2


def run_split(emb, labels, n_classes, split_idx):
    ref_idx, cal_idx, test_idx = stratified_split(
        labels, n_classes,
        ref_ratio=REF_RATIO, cal_ratio=CAL_RATIO,
        seed=42 + split_idx,
    )

    centroids = compute_centroids(emb, labels, ref_idx, n_classes)

    cal_scores = np.array([
        1.0 - np.dot(
            emb[i] / np.linalg.norm(emb[i]),
            centroids[labels[i]] / np.linalg.norm(centroids[labels[i]])
        )
        for i in cal_idx
    ])

    alpha_scores_test = compute_alpha_scores_test(emb, test_idx, centroids, n_classes)

    results = {}
    for alpha in ALPHAS:
        pred_sets = cp_prediction(alpha_scores_test, cal_scores, alpha)
        cov, avg_sz, sizes, empty, sing = evaluate(pred_sets, labels[test_idx])
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
    emb, labels, le = load_data(EMB_DIR)
    n_classes = len(le.classes_)
    print(f"  embeddings: {emb.shape}")
    print(f"  classes: {n_classes} ({list(le.classes_)})")

    all_results = []

    for split in range(N_SPLITS):
        print(f"\nSplit {split+1}/{N_SPLITS} ...")
        results, centroids, cal_scores = run_split(emb, labels, n_classes, split)

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
