"""Extract MERT-v1-330M embeddings from FMA Small dataset.

Processes MP3 files (30s clips, 8 genres, ~8000 tracks).
Uses librosa for MP3 loading (cross-platform) + same MERT extraction pipeline as GTZAN.
Saves embeddings + metadata + labels to outputs/embeddings_fma/.
"""
import sys
import time
import warnings
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
FMA_DIR = ROOT / "data" / "fma"
OUTPUT_DIR = ROOT / "outputs" / "embeddings_fma"

MODEL_NAME = "m-a-p/MERT-v1-330M"
SAMPLE_RATE = 24000
SEGMENT_S = 10
OVERLAP_S = 5
MAX_DURATION_S = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FMA_GENRES = [
    "electronic", "experimental", "folk", "hip-hop",
    "instrumental", "international", "pop", "rock"
]


def load_file_list():
    mapping_path = FMA_DIR / "genre_mapping.csv"
    if not mapping_path.exists():
        print(f"ERROR: genre_mapping.csv not found at {mapping_path}")
        print("Run 'python scripts/prepare_fma.py --download' first.")
        sys.exit(1)
    df = pd.read_csv(mapping_path)
    files = []
    for _, row in df.iterrows():
        files.append((row["genre"], row["path"]))
    return files


def load_model():
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    return processor, model


@torch.no_grad()
def extract_embedding(audio_path, processor, model):
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True,
                                 duration=MAX_DURATION_S)
    except Exception as e:
        print(f"\n  SKIP: {Path(audio_path).name} ({e})")
        return None

    if len(audio) < SAMPLE_RATE * 0.5:
        return None

    total_samples = len(audio)
    segment_samples = SEGMENT_S * SAMPLE_RATE
    hop_samples = (SEGMENT_S - OVERLAP_S) * SAMPLE_RATE

    if total_samples <= segment_samples:
        segments = [audio]
    else:
        segments = []
        start = 0
        while start + segment_samples <= total_samples:
            segments.append(audio[start:start + segment_samples])
            start += hop_samples

    if len(segments) == 0:
        return None

    segment_embeddings = []
    for seg in segments:
        inputs = processor(
            seg, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        emb = hidden.mean(dim=1).cpu()
        segment_embeddings.append(emb)

    track_embedding = torch.stack(segment_embeddings).mean(dim=0)
    return track_embedding.squeeze(0).numpy()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = load_file_list()
    print(f"Found {len(files)} audio files across {len(FMA_GENRES)} genres")
    print(f"Device: {DEVICE}")

    print(f"Loading MERT model ({MODEL_NAME}) ...")
    processor, model = load_model()
    vram = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
    print(f"Model loaded. VRAM: {vram:.2f} GB")

    emb_list = []
    metadata = []
    labels_list = []
    batch_times = []
    skipped = []

    label_to_id = {g: i for i, g in enumerate(FMA_GENRES)}

    for _, (genre, path) in enumerate(tqdm(files, desc="Extracting FMA", unit="file")):
        t0 = time.time()
        emb = extract_embedding(path, processor, model)
        elapsed = time.time() - t0
        batch_times.append(elapsed)

        if emb is None:
            skipped.append(Path(path).name)
            continue

        emb_list.append(emb)
        metadata.append({
            "genre": genre,
            "filename": Path(path).name,
            "path": path,
        })
        labels_list.append(label_to_id.get(genre, -1))

    if len(emb_list) == 0:
        print("ERROR: No embeddings extracted. Check audio files.")
        return

    embeddings = np.stack(emb_list, axis=0).astype(np.float32)
    label_ids = np.array(labels_list, dtype=np.int32)

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    print(f"Saved embeddings: {OUTPUT_DIR / 'embeddings.npy'}  shape={embeddings.shape}")

    np.save(OUTPUT_DIR / "label_ids.npy", label_ids)
    print(f"Saved labels: {OUTPUT_DIR / 'label_ids.npy'}  shape={label_ids.shape}")

    with open(OUTPUT_DIR / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("index,genre,filename,path\n")
        for idx, m in enumerate(metadata):
            f.write(f"{idx},{m['genre']},{m['filename']},{m['path']}\n")
    print(f"Saved metadata: {OUTPUT_DIR / 'metadata.csv'}")

    with open(OUTPUT_DIR / "labels.csv", "w", encoding="utf-8") as f:
        f.write("index,genre,label_id\n")
        for idx, m in enumerate(metadata):
            f.write(f"{idx},{m['genre']},{label_to_id[m['genre']]}\n")

    if skipped:
        print(f"\nSkipped {len(skipped)} files: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    avg_time = np.mean(batch_times) if batch_times else 0
    total_time = sum(batch_times)
    eta_8000 = avg_time * 8000 / 60
    print(f"\nExtraction complete. {len(emb_list)}/{len(files)} files in {total_time:.1f}s "
          f"(avg {avg_time:.2f}s/file, est. {eta_8000:.0f} min for 8000 tracks)")
    print(f"Genre distribution:")
    for genre in FMA_GENRES:
        count = sum(1 for m in metadata if m["genre"] == genre)
        if count > 0:
            print(f"  {genre:15s}: {count:4d}")
    print("Done.")


if __name__ == "__main__":
    main()
