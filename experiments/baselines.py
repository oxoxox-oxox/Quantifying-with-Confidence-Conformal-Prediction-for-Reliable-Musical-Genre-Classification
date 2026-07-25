"""Step 10: Baseline comparisons — Bootstrap & Gaussian vs Conformal Prediction.

Evaluates whether Bootstrap/Gaussian achieve valid coverage (they don't, by
design, since they assume normality), highlighting CP's distribution-free guarantee.
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
    compute_nonconformity_scores,
    cp_prediction,
    bootstrap_prediction,
    gaussian_prediction,
    evaluate,
)

EMB_DIR = ROOT / "outputs" / "embeddings"
OUT_DIR = ROOT / "outputs" / "cp_results"

ALPHAS = [0.01, 0.05, 0.10]
N_SPLITS = 5
REF_RATIO = 0.6
CAL_RATIO = 0.2
N_BOOTSTRAP = 1000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb, labels, le = load_data(EMB_DIR)
    n_classes = len(le.classes_)

    methods = ["CP", "Bootstrap", "Gaussian"]
    all_rows = []

    for split in range(N_SPLITS):
        ref_idx, cal_idx, test_idx = stratified_split(
            labels, n_classes,
            ref_ratio=REF_RATIO, cal_ratio=CAL_RATIO,
            seed=42 + split,
        )

        centroids = compute_centroids(emb, labels, ref_idx, n_classes)

        cal_scores = compute_nonconformity_scores(emb, labels, cal_idx, centroids)

        alpha_test = compute_alpha_scores_test(emb, test_idx, centroids, n_classes)

        true_test = labels[test_idx]

        for alpha in ALPHAS:
            cp_sets = cp_prediction(alpha_test, cal_scores, alpha)
            boot_sets = bootstrap_prediction(alpha_test, cal_scores, alpha, N_BOOTSTRAP)
            gauss_sets = gaussian_prediction(alpha_test, cal_scores, alpha)

            for method, pred_sets in zip(methods, [cp_sets, boot_sets, gauss_sets]):
                cov, avg_sz, sizes, empty, sing = evaluate(pred_sets, true_test)
                all_rows.append({
                    "split": split, "alpha": alpha, "method": method,
                    "coverage": cov, "avg_set_size": avg_sz,
                    "empty_sets": empty, "singleton_sets": sing,
                    "n_test": len(test_idx),
                })

    df = pd.DataFrame(all_rows)

    print("=== BASELINE COMPARISON ===\n")
    for alpha in ALPHAS:
        print(f"--- α = {alpha:.2f} (target coverage = {1-alpha:.2f}) ---")
        for method in methods:
            sub = df[(df["alpha"] == alpha) & (df["method"] == method)]
            cov_m, cov_s = sub["coverage"].mean(), sub["coverage"].std()
            sz_m, sz_s = sub["avg_set_size"].mean(), sub["avg_set_size"].std()
            cov_ok = "[OK]" if abs(cov_m - (1 - alpha)) < 0.03 else "[FAIL]"
            print(f"  {method:12s}: coverage={cov_m:.3f}+/-{cov_s:.3f} {cov_ok}  "
                  f"set_size={sz_m:.2f}+/-{sz_s:.2f}")
        print()

    df.to_csv(OUT_DIR / "baseline_comparison.csv", index=False)
    print(f"Saved: {OUT_DIR / 'baseline_comparison.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
