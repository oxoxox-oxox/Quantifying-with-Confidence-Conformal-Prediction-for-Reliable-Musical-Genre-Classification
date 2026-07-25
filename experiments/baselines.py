"""Step 10: Baseline comparisons — Bootstrap & Gaussian vs Conformal Prediction.

Evaluates whether Bootstrap/Gaussian achieve valid coverage (they don't, by
design, since they assume normality), highlighting CP's distribution-free guarantee.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "outputs" / "embeddings"
OUT_DIR = ROOT / "outputs" / "cp_results"

ALPHAS = [0.01, 0.05, 0.10]
N_SPLITS = 5
REF_RATIO = 0.6
CAL_RATIO = 0.2
N_BOOTSTRAP = 1000


def load_data():
    emb = np.load(EMB_DIR / "embeddings.npy")
    ids = np.load(EMB_DIR / "label_ids.npy")
    meta = pd.read_csv(EMB_DIR / "metadata.csv")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(meta["genre"].unique())
    return emb, ids, le


def cp_prediction(alpha_scores_test, cal_scores, alpha):
    n_cal = len(cal_scores)
    pred_sets = []
    for i in range(len(alpha_scores_test)):
        s = []
        for k in range(len(alpha_scores_test[i])):
            p = (np.sum(cal_scores >= alpha_scores_test[i][k]) + 1) / (n_cal + 1)
            if p > alpha:
                s.append(k)
        pred_sets.append(s)
    return pred_sets


def bootstrap_prediction(alpha_scores_test, cal_scores, alpha, n_boot):
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


def evaluate(pred_sets, true_labels):
    n = len(true_labels)
    covered = sum(1 for i in range(n) if true_labels[i] in pred_sets[i])
    sizes = [len(s) for s in pred_sets]
    return covered / n, np.mean(sizes), sizes


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb, labels, le = load_data()
    n_classes = len(le.classes_)
    n = len(labels)

    methods = ["CP", "Bootstrap", "Gaussian"]
    all_rows = []

    for split in range(N_SPLITS):
        rng = np.random.RandomState(42 + split)
        indices = np.arange(n)
        ref_idx, cal_idx, test_idx = [], [], []

        for c in range(n_classes):
            c_idx = indices[labels == c]
            rng.shuffle(c_idx)
            n_ref = int(len(c_idx) * REF_RATIO)
            n_cal = int(len(c_idx) * CAL_RATIO)
            ref_idx.append(c_idx[:n_ref])
            cal_idx.append(c_idx[n_ref:n_ref + n_cal])
            test_idx.append(c_idx[n_ref + n_cal:])

        ref_idx = np.concatenate(ref_idx)
        cal_idx = np.concatenate(cal_idx)
        test_idx = np.concatenate(test_idx)

        centroids = np.array([
            emb[ref_idx][labels[ref_idx] == c].mean(axis=0)
            for c in range(n_classes)
        ])

        cal_scores = np.array([
            1.0 - cosine_similarity(emb[i:i+1], centroids[labels[i]:labels[i]+1])[0, 0]
            for i in cal_idx
        ])

        alpha_test = np.array([
            [1.0 - cosine_similarity(emb[i:i+1], centroids[k:k+1])[0, 0]
             for k in range(n_classes)]
            for i in test_idx
        ])

        true_test = labels[test_idx]

        for alpha in ALPHAS:
            cp_sets = cp_prediction(alpha_test, cal_scores, alpha)
            boot_sets = bootstrap_prediction(alpha_test, cal_scores, alpha, N_BOOTSTRAP)
            gauss_sets = gaussian_prediction(alpha_test, cal_scores, alpha)

            for method, pred_sets in zip(methods, [cp_sets, boot_sets, gauss_sets]):
                cov, avg_sz, sizes = evaluate(pred_sets, true_test)
                all_rows.append({
                    "split": split, "alpha": alpha, "method": method,
                    "coverage": cov, "avg_set_size": avg_sz,
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
