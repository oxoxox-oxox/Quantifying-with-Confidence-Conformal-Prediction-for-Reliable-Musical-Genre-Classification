"""Shared data loading utilities for embedding-based experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(emb_dir):
    emb = np.load(Path(emb_dir) / "embeddings.npy")
    ids = np.load(Path(emb_dir) / "label_ids.npy")
    meta = pd.read_csv(Path(emb_dir) / "metadata.csv")
    le = LabelEncoder()
    le.fit(meta["genre"].unique())
    return emb, ids, le
