# Quantifying with Confidence: Conformal Prediction for Reliable Musical Style Similarity Ranking

> MERT + Conformal Prediction framework for uncertainty-aware musical style similarity assessment.

---

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

### GTZAN (Primary, 1.2 GB)

1. Go to: <https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification>
2. Click **Download** (requires Kaggle account)
3. Extract the zip into `data/` so it looks like:

```
data/gtzan/
├── genres_original/
│   ├── blues/        (100 x .wav)
│   ├── classical/
│   ├── country/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/         (99 valid + 1 corrupt: jazz.00054.wav)
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/         (100 x .wav, 999 total valid)
├── images_original/
├── features_30_sec.csv
└── features_3_sec.csv
```

4. Verify:

```
python scripts/prepare_gtzan.py
```

### FMA (Supplementary, optional)

1. <https://github.com/mdeff/fma>
2. Download `fma_small.zip` (8000 tracks, 8 genres)

---

## Experiments

| Step | Script | Description |
|------|--------|-------------|
| Verify MERT | `scripts/verify_mert.py` | Load model + GPU inference test |
| Data prep | `scripts/prepare_gtzan.py` | Verify GTZAN structure |
| Extract embeddings | `experiments/extract_embeddings.py` | MERT hidden states → (999, 1024) |
| Verify embeddings | `experiments/verify_embeddings.py` | Intra/inter-class similarity + PCA |
| CP classification | `experiments/cp_classification.py` | CP scheme A: genre centroid classification |
| Baselines | `experiments/baselines.py` | Bootstrap + Gaussian vs CP |
| Visualization | `experiments/plot_results.py` | 3 publication figures |

### Key Results (GTZAN, 5-fold stratified, 60/20/20 split)

| α | Nominal Coverage | CP | Bootstrap | Gaussian |
|---|-------------------|-----|-----------|----------|
| 0.01 | 0.99 | **0.993** ± 0.010 | 0.987 ± 0.012 | 0.965 ± 0.020 |
| 0.05 | 0.95 | **0.955** ± 0.021 | 0.950 ± 0.023 | 0.932 ± 0.033 |
| 0.10 | 0.90 | **0.910** ± 0.028 | 0.910 ± 0.029 | 0.907 ± 0.031 |

- CP coverage tightly matches nominal levels (distribution-free guarantee)
- Gaussian undercovers at high confidence (α=0.01: 0.965 vs target 0.99) due to heavy-tailed score distribution
- Embedding quality: intra-class cos=0.917, inter-class=0.873, separation δ=0.044

---

## Project Structure

```
├── data/                     # Datasets (git-ignored)
│   └── gtzan/genres_original/  # 10 genres x 100 WAV
├── experiments/
│   ├── extract_embeddings.py   # MERT embedding extraction
│   ├── verify_embeddings.py    # Embedding quality check + PCA
│   ├── cp_classification.py    # CP scheme A (classification)
│   ├── baselines.py            # Bootstrap + Gaussian baselines
│   └── plot_results.py         # Publication figures
├── notebooks/                # Exploratory notebooks
├── outputs/
│   ├── embeddings/           # embeddings.npy, metadata.csv, labels.csv
│   ├── cp_results/           # cp_results.csv, baseline_comparison.csv
│   └── figures/              # pca_embeddings, coverage plots, set_size plots
├── paper/
│   ├── figures/
│   └── sections/
│       └── 01_introduction.tex    # Introduction draft
├── references/               # Literature .bib
├── scripts/
│   ├── verify_mert.py        # MERT model verification
│   └── prepare_gtzan.py      # GTZAN validation
├── src/                      # Core library code
├── tests/                    # Unit tests
├── Guide.md                  # Full writing guide (Chinese)
├── SKILL.md                  # Agent skill reference
└── requirements.txt
```

---

## Tech Stack

- Python 3.11
- PyTorch 2.7+ (CUDA 12.8)
- HuggingFace Transformers
- MERT-v1-330M (`m-a-p/MERT-v1-330M`)
- MAPIE (Conformal Prediction)
