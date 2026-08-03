"""Main experiment: Linear Probe + Conformal Prediction with learned prototypes.

Main experiment:
  - LP-Cos-CP: learned prototypes via linear classification, cosine-based
    nonconformity score, calibrated with empirical p-values.

Baselines (all share the same Linear Probe prediction model):
  - LP-Softmax-Bootstrap: softmax residual score, bootstrap quantile calibration
  - LP-Softmax-Gaussian: softmax residual score, Gaussian quantile calibration
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
    cp_prediction,
    bootstrap_prediction,
    gaussian_prediction,
    linear_probe_pipeline,
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

    methods = ["LP-Cos-CP", "LP-Softmax-Bootstrap", "LP-Softmax-Gaussian"]
    all_rows = []

    for split in range(N_SPLITS):
        print(f"\nSplit {split+1}/{N_SPLITS} ...")
        ref_idx, cal_idx, test_idx = stratified_split(
            labels, n_classes,
            ref_ratio=REF_RATIO, cal_ratio=CAL_RATIO,
            seed=42 + split,
        )

        cos_cal, cos_alpha, smx_cal, smx_alpha = linear_probe_pipeline(
            emb, labels, ref_idx, cal_idx, test_idx, n_classes,
            seed=42 + split,
        )

        true_test = labels[test_idx]

        for alpha in ALPHAS:
            lp_cos_sets = cp_prediction(cos_alpha, cos_cal, alpha)
            lp_smx_boot_sets = bootstrap_prediction(smx_alpha, smx_cal, alpha, N_BOOTSTRAP)
            lp_smx_gauss_sets = gaussian_prediction(smx_alpha, smx_cal, alpha)

            for method, pred_sets in zip(
                methods,
                [lp_cos_sets, lp_smx_boot_sets, lp_smx_gauss_sets],
            ):
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
        print(f"--- a = {alpha:.2f} (target coverage = {1-alpha:.2f}) ---")
        for method in methods:
            sub = df[(df["alpha"] == alpha) & (df["method"] == method)]
            cov_m, cov_s = sub["coverage"].mean(), sub["coverage"].std()
            sz_m, sz_s = sub["avg_set_size"].mean(), sub["avg_set_size"].std()
            cov_ok = "[OK]" if abs(cov_m - (1 - alpha)) < 0.03 else "[FAIL]"
            print(f"  {method:22s}: coverage={cov_m:.3f}+/-{cov_s:.3f} {cov_ok}  "
                  f"set_size={sz_m:.2f}+/-{sz_s:.2f}")
        print()

    df.to_csv(OUT_DIR / "baseline_comparison.csv", index=False)
    print(f"Saved: {OUT_DIR / 'baseline_comparison.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
