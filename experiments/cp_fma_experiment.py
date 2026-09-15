"""Conformal Prediction experiment on FMA Small dataset.

Main experiment:
  - LP-Cos-CP: learned prototypes (cosine score) + CP p-value

Baselines (all share the same Linear Probe prediction model):
  - LP-Softmax-Bootstrap: softmax residual score + bootstrap quantile
  - LP-Softmax-Gaussian: softmax residual score + Gaussian quantile
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

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

EMB_DIR = ROOT / "outputs" / "embeddings_fma"
OUT_DIR = ROOT / "outputs" / "cp_results_fma"

ALPHAS = [0.05, 0.10]
N_SPLITS = 5
REF_RATIO = 0.6
CAL_RATIO = 0.2
N_BOOTSTRAP = 1000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FMA data ...")
    emb, labels, le = load_data(EMB_DIR)
    n_classes = len(le.classes_)
    n = len(labels)
    print(f"  embeddings: {emb.shape}")
    print(f"  classes: {n_classes} ({list(le.classes_)})")
    print(f"  total samples: {n}")

    print(f"\n  Checking embedding quality ...")
    intra_sims = []
    for c in range(n_classes):
        c_mask = labels == c
        c_embs = emb[c_mask]
        if len(c_embs) >= 2:
            sims = cos_sim(c_embs, c_embs)
            mask = ~np.eye(len(c_embs), dtype=bool)
            intra_sims.extend(sims[mask].tolist())
    intra_mean = np.mean(intra_sims) if intra_sims else 0

    inter_sims = []
    n_sample = min(5000, n)
    rng = np.random.RandomState(42)
    for _ in range(n_sample):
        i, j = rng.choice(n, 2, replace=False)
        if labels[i] != labels[j]:
            inter_sims.append(cos_sim(emb[i:i+1], emb[j:j+1])[0, 0])
    inter_mean = np.mean(inter_sims) if inter_sims else 0
    delta = intra_mean - inter_mean
    print(f"  intra-class cos: {intra_mean:.4f}")
    print(f"  inter-class cos: {inter_mean:.4f}")
    print(f"  separation delta: {delta:.4f}")

    methods = ["LP-Cos-CP", "LP-Softmax-Bootstrap", "LP-Softmax-Gaussian"]
    all_rows = []

    for split in range(N_SPLITS):
        print(f"\nSplit {split+1}/{N_SPLITS} ...")
        ref_idx, cal_idx, test_idx = stratified_split(
            labels, n_classes,
            ref_ratio=REF_RATIO, cal_ratio=CAL_RATIO,
            seed=42 + split,
        )

        cos_cal, cos_alpha, smx_cal, smx_alpha, cal_labels = linear_probe_pipeline(
            emb, labels, ref_idx, cal_idx, test_idx, n_classes,
            seed=42 + split,
        )

        true_test = labels[test_idx]

        top1_preds = np.argmin(smx_alpha, axis=1)
        top1_acc = (top1_preds == true_test).mean()

        for alpha in ALPHAS:
            lp_cos_sets = cp_prediction(cos_alpha, cos_cal, cal_labels, alpha)
            lp_smx_boot_sets = bootstrap_prediction(smx_alpha, smx_cal, cal_labels, alpha, N_BOOTSTRAP)
            lp_smx_gauss_sets = gaussian_prediction(smx_alpha, smx_cal, cal_labels, alpha)

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
                    "top1_acc": top1_acc,
                })

    df = pd.DataFrame(all_rows)

    print("\n" + "=" * 60)
    print("FMA SMALL -- LP + CP EXPERIMENT RESULTS")
    print("=" * 60)

    mean_top1 = df["top1_acc"].mean()
    std_top1 = df["top1_acc"].std()
    print(f"\nBase Model (Linear Probe) Top-1 Accuracy: {mean_top1:.3f} +/- {std_top1:.3f}")

    for alpha in ALPHAS:
        print(f"\n--- a = {alpha:.2f} (target coverage = {1-alpha:.2f}) ---")
        for method in methods:
            sub = df[(df["alpha"] == alpha) & (df["method"] == method)]
            cov_m, cov_s = sub["coverage"].mean(), sub["coverage"].std()
            sz_m, sz_s = sub["avg_set_size"].mean(), sub["avg_set_size"].std()
            cov_ok = "[OK]" if abs(cov_m - (1 - alpha)) < 0.03 else "[BELOW]"
            print(f"  {method:22s}: coverage={cov_m:.3f}+/-{cov_s:.3f} {cov_ok}  "
                  f"set_size={sz_m:.2f}+/-{sz_s:.2f}")

    df.to_csv(OUT_DIR / "baseline_comparison_fma.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'baseline_comparison_fma.csv'}")

    print("\n=== COMPARISON: GTZAN vs FMA ===")
    print(f"{'Dataset':<10s} {'#Tracks':>8s} {'Classes':>8s} {'Intra-Cos':>10s} {'Delta':>8s}")
    print(f"{'GTZAN':<10s} {'999':>8s} {'10':>8s} {'0.917':>10s} {'0.044':>8s}")
    print(f"{'FMA':<10s} {str(n):>8s} {str(n_classes):>8s} "
          f"{intra_mean:.3f}:>10s {delta:.3f}:>8s")

    print("\nDone.")


if __name__ == "__main__":
    main()
