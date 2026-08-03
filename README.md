# Quantifying with Confidence: Conformal Prediction for Reliable Musical Genre Classification

> Linear Probe + Conformal Prediction: learned prototypes with distribution-free coverage guarantees for uncertainty-aware genre classification.

---

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

### GTZAN (Primary, 1.2 GB)

1. Go to: <https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification>
2. Click **Download** (requires Kaggle account)
3. Extract the zip into `data/`:

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
```

1. Verify: `python scripts/prepare_gtzan.py`

### FMA (Supplementary, 7.2 GB)

```bash
python scripts/prepare_fma.py --download
python scripts/extract_embeddings_fma.py
python experiments/cp_fma_experiment.py
python scripts/plot_fma_comparison.py
```

FMA Small: 7994 valid tracks across 8 top-level genres.

---

## Experiments

### Full Pipeline

```bash
# 1. Verify MERT model
python scripts/verify_mert.py

# 2. GTZAN: extract embeddings + run experiments
python scripts/extract_embeddings.py
python scripts/verify_embeddings.py
python experiments/baselines.py

# 3. FMA: extract embeddings + run experiments
python scripts/extract_embeddings_fma.py
python experiments/cp_fma_experiment.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Results

### GTZAN (999 tracks, 10 genres, 5-fold stratified, 60/20/20 split)

| α | Target | LP-Cos-CP (Main) | LP-Softmax-Bootstrap | LP-Softmax-Gaussian |
| --- | -------- | :---: | :---: | :---: |
| 0.01 | 0.99 | **0.988** ± 0.014 | 0.986 ± 0.013 | 1.000 ± 0.000 |
| 0.05 | 0.95 | **0.963** ± 0.011 | 0.958 ± 0.016 | 0.991 ± 0.010 |
| 0.10 | 0.90 | **0.905** ± 0.020 | 0.899 ± 0.017 | 0.947 ± 0.022 |

| α | LP-Cos-CP (Main) | LP-Softmax-Bootstrap | LP-Softmax-Gaussian |
| --- | :---: | :---: | :---: |
| 0.01 | **6.13** | 5.26 | 9.96 |
| 0.05 | **3.62** | 3.11 | 5.39 |
| 0.10 | **2.42** | 2.13 | 2.84 |

- **LP-Cos-CP (main)**: learned prototypes satisfy coverage at all alpha levels with compact prediction sets
- **Bootstrap**: slightly undercovers at alpha=0.10 (0.899 vs target 0.90); no finite-sample guarantee
- **Gaussian**: massively overcovers (1.000 at alpha=0.01, set size 9.96/10), failing the tightness requirement

### FMA (7994 tracks, 8 genres, 5-fold)

| α | Target | LP-Cos-CP | LP-Softmax-Bootstrap | LP-Softmax-Gaussian |
|---|--------|:---:|:---:|:---:|:---:|

(see outputs/cp_results_fma/baseline_comparison_fma.csv for full results)

---

## Project Structure

```
├── src/                        # Core shared library
│   ├── data_utils.py           # Data loading
│   ├── split_utils.py          # Stratified 60/20/20 split
│   ├── conformal.py            # CP, Bootstrap, Gaussian, Linear Probe
│   └── evaluation.py           # Coverage & set size metrics
├── tests/
│   ├── test_split_utils.py     # Split correctness tests
│   └── test_conformal.py       # CP & baseline method tests
├── experiments/
│   ├── __init__.py
│   ├── baselines.py                # Main experiment + 3 baselines (GTZAN)
│   └── cp_fma_experiment.py        # Main experiment + 3 baselines (FMA)
├── scripts/
│   ├── verify_mert.py              # MERT model verification
│   ├── prepare_gtzan.py            # GTZAN download validation
│   ├── prepare_fma.py              # FMA download + extract
│   ├── extract_embeddings.py       # MERT → (999, 1024) for GTZAN
│   ├── verify_embeddings.py        # Embedding quality + PCA
│   ├── extract_embeddings_fma.py   # MERT → (7994, 1024) for FMA
│   ├── plot_results.py             # GTZAN publication figures
│   └── plot_fma_comparison.py      # GTZAN vs FMA comparison figures
├── outputs/
│   ├── embeddings/                 # GTZAN: embeddings.npy (999, 1024)
│   ├── embeddings_fma/             # FMA: embeddings.npy (7994, 1024)
│   ├── cp_results/                 # GTZAN results CSV
│   ├── cp_results_fma/             # FMA results CSV
│   └── figures/                    # All plots (PCA, coverage, comparison)
├── paper/
│   ├── main.tex                    # IEEE format, 22 pages, 22 refs
│   ├── sections/                   # Intro / Related / Methodology / Experiments / Discussion / Conclusion
│   └── refs.bib                    # Bibliography
├── data/                           # Datasets (git-ignored)
└── requirements.txt
```

---

## Tech Stack

- Python 3.11
- PyTorch 2.7+ (CUDA)
- HuggingFace Transformers — MERT-v1-330M
- scikit-learn, scipy, MAPIE (Conformal Prediction)
- matplotlib, seaborn, pandas
