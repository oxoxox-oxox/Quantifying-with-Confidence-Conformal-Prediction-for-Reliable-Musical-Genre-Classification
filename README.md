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

### FMA (Supplementary, 7.2 GB)

Full pipeline (download + extraction + experiments):

```
python scripts/prepare_fma.py --download
python experiments/extract_embeddings_fma.py
python experiments/cp_fma_experiment.py
python experiments/plot_fma_comparison.py
```

FMA Small: 7994 valid tracks across 8 top-level genres (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock). Genres are evenly distributed (~1000 per class).

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
| **FMA data prep** | `scripts/prepare_fma.py --download` | Download + verify FMA Small |
| **FMA embeddings** | `experiments/extract_embeddings_fma.py` | MERT → (7994, 1024) |
| **FMA experiments** | `experiments/cp_fma_experiment.py` | CP + baselines on FMA |
| **FMA comparison** | `experiments/plot_fma_comparison.py` | GTZAN vs FMA figures |

## Results

### Key Results (GTZAN, 5-fold stratified, 60/20/20 split)

| α | Nominal Coverage | CP | Bootstrap | Gaussian |
|---|-------------------|-----|-----------|----------|
| 0.01 | 0.99 | **0.993** ± 0.010 | 0.987 ± 0.012 | 0.965 ± 0.020 |
| 0.05 | 0.95 | **0.955** ± 0.021 | 0.950 ± 0.023 | 0.932 ± 0.033 |
| 0.10 | 0.90 | **0.910** ± 0.028 | 0.910 ± 0.029 | 0.907 ± 0.031 |

- CP coverage tightly matches nominal levels (distribution-free guarantee)
- Gaussian undercovers at high confidence (α=0.01: 0.965 vs target 0.99) due to heavy-tailed score distribution
- Embedding quality: intra-class cos=0.917, inter-class=0.873, separation δ=0.044

### FMA Results (7994 tracks, 8 genres, 5-fold)

| α | Nominal Coverage | CP | Bootstrap | Gaussian |
|---|-------------------|-----|-----------|----------|
| 0.01 | 0.99 | **0.988** ± 0.004 | 0.987 ± 0.004 | 0.966 ± 0.003 |
| 0.05 | 0.95 | **0.948** ± 0.004 | 0.947 ± 0.004 | 0.934 ± 0.005 |
| 0.10 | 0.90 | **0.898** ± 0.006 | 0.896 ± 0.006 | 0.903 ± 0.005 |

- CP coverage guarantee holds across datasets (distribution-free property verified)
- FMA embedding quality lower: intra=0.847, inter=0.821, δ=0.026 (broad genres overlap more)
- Larger prediction sets (6-8/8) reflect higher label ambiguity in real-world data

## Paper

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Output: `paper/main.pdf` (19 pages, ~630 KB, IEEE format)

---

## Project Structure

```
├── data/                     # Datasets (git-ignored)
│   ├── gtzan/genres_original/  # 10 genres x 100 WAV
│   └── fma/                    # FMA Small (8 genres, 8000 MP3)
├── experiments/
│   ├── extract_embeddings.py   # MERT embedding extraction (GTZAN)
│   ├── verify_embeddings.py    # Embedding quality check + PCA
│   ├── cp_classification.py    # CP scheme A (classification)
│   ├── baselines.py            # Bootstrap + Gaussian baselines
│   ├── plot_results.py         # Publication figures
│   ├── extract_embeddings_fma.py   # MERT embedding extraction (FMA)
│   ├── cp_fma_experiment.py        # CP + baselines on FMA
│   └── plot_fma_comparison.py      # GTZAN vs FMA comparison plots
├── notebooks/                # Exploratory notebooks
├── outputs/
│   ├── embeddings/           # GTZAN: embeddings.npy (999, 1024)
│   ├── embeddings_fma/       # FMA: embeddings.npy (7994, 1024)
│   ├── cp_results/           # GTZAN: cp_results, baseline_comparison
│   ├── cp_results_fma/       # FMA: baseline_comparison_fma
│   └── figures/              # All figures (PCA, coverage, FMA comparison)
├── paper/
│   ├── main.tex                 # Main LaTeX file (IEEE, 21 references)
│   ├── main.pdf                 # Compiled PDF (19 pages)
│   └── sections/
│       ├── 01_introduction.tex  # ~1000 words, 3 contributions
│       ├── 02_related_work.tex  # SSL/MERT/CP/UQ/evaluation review
│       ├── 03_methodology.tex   # 6 equations, CP framework
│       ├── 04_experiments.tex   # Results table, coverage analysis
│       ├── 05_discussion.tex    # Interpretations, limitations, future work
│       └── 06_conclusion.tex    # Summary and outlook
├── references/               # Literature .bib
├── scripts/
│   ├── verify_mert.py        # MERT model verification
│   ├── prepare_gtzan.py      # GTZAN validation
│   └── prepare_fma.py        # FMA download + validation
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
