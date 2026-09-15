"""Steps 6-7: Verify MERT embeddings quality + generate style annotations.

Checks:
  - Shape validity
  - Intra-class vs inter-class cosine similarity
  - 2D PCA projection for genre separability
  - Class distribution summary

Generates labels.csv for downstream CP experiments.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "outputs" / "embeddings"
FIG_DIR = ROOT / "outputs" / "figures"

GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]


def load_data():
    embeddings = np.load(EMB_DIR / "embeddings.npy")
    metadata = pd.read_csv(EMB_DIR / "metadata.csv")
    labels_raw = metadata["genre"].values
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    return embeddings, metadata, labels, le


def compute_similarity_stats(embeddings, labels):
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(embeddings)
    n = len(labels)

    intra_sims = []
    inter_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_sims.append(sim_matrix[i, j])
            else:
                inter_sims.append(sim_matrix[i, j])

    intra_sims = np.array(intra_sims)
    inter_sims = np.array(inter_sims)

    sep = intra_sims.mean() - inter_sims.mean()
    return intra_sims, inter_sims, sep


def plot_pca(embeddings, labels, label_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, name in enumerate(label_names):
        mask = labels == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], label=name, alpha=0.6, s=12, edgecolors="none"
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("GTZAN MERT Embeddings — PCA Projection")
    ax.legend(markerscale=2, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()

    out_path = FIG_DIR / "pca_embeddings.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved PCA plot: {out_path}")


def main():
    print("Loading embeddings ...")
    embeddings, metadata, labels, le = load_data()
    print(f"  shape: {embeddings.shape}")
    print(f"  dtype: {embeddings.dtype}")
    print(f"  classes: {list(le.classes_)}")
    print(f"  NaN count: {np.isnan(embeddings).sum()}")
    print(f"  Inf count: {np.isinf(embeddings).sum()}")

    print("\nClass distribution:")
    for cls_name in le.classes_:
        count = (metadata["genre"] == cls_name).sum()
        print(f"  {cls_name:12s}: {count}")

    print("\nComputing intra/inter-class cosine similarity ...")
    intra, inter, sep = compute_similarity_stats(embeddings, labels)
    print(f"  Intra-class (mean):  {intra.mean():.4f}  (std: {intra.std():.4f})")
    print(f"  Inter-class (mean):  {inter.mean():.4f}  (std: {inter.std():.4f})")
    print(f"  Separation (delta):  {sep:.4f}")

    if sep <= 0:
        print("  WARNING: intra-class similarity is NOT higher than inter-class!")

    print("\nGenerating PCA plot ...")
    plot_pca(embeddings, labels, list(le.classes_))

    labels_path = EMB_DIR / "labels.csv"
    metadata[["genre"]].to_csv(labels_path, index_label="index")
    print(f"Saved labels: {labels_path}")

    label_ids_path = EMB_DIR / "label_ids.npy"
    np.save(label_ids_path, labels)
    print(f"Saved label IDs: {label_ids_path}")

    print("\n=== EMBEDDINGS VERIFIED ===")


if __name__ == "__main__":
    main()
