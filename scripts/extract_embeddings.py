"""Step 5: Extract MERT-v1-330M embeddings from GTZAN audio files.

Processes 1000 × 30s WAV files in chunked segments (10s windows, 5s overlap).
Saves (1000, 1024) float32 embeddings + metadata CSV.
"""
import time
import warnings
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "gtzan" / "genres_original"
OUTPUT_DIR = ROOT / "outputs" / "embeddings"

MODEL_NAME = "m-a-p/MERT-v1-330M"
SAMPLE_RATE = 24000
SEGMENT_S = 10
OVERLAP_S = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]


def collect_files():
    files = []
    for genre in GENRES:
        genre_dir = DATA_DIR / genre
        wavs = sorted(genre_dir.glob("*.wav"))
        for p in wavs:
            files.append((genre, str(p)))
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
        audio, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"\n  SKIP: {Path(audio_path).name} ({e})")
        return None

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

    audio = audio.squeeze(0)
    total_samples = audio.shape[0]
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
            seg.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
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

    files = collect_files()
    print(f"Found {len(files)} audio files across {len(GENRES)} genres")
    print(f"Device: {DEVICE}")

    print(f"Loading MERT model ({MODEL_NAME}) ...")
    processor, model = load_model()
    vram = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
    print(f"Model loaded. VRAM: {vram:.2f} GB")

    emb_list = []
    metadata = []
    batch_times = []
    skipped = []

    for i, (genre, path) in enumerate(tqdm(files, desc="Extracting", unit="file")):
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

    embeddings = np.stack(emb_list, axis=0).astype(np.float32)

    np_path = OUTPUT_DIR / "embeddings.npy"
    np.save(np_path, embeddings)
    print(f"Saved embeddings: {np_path}  shape={embeddings.shape}")

    csv_path = OUTPUT_DIR / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,genre,filename,path\n")
        for idx, m in enumerate(metadata):
            f.write(f"{idx},{m['genre']},{m['filename']},{m['path']}\n")
    print(f"Saved metadata: {csv_path}")

    if skipped:
        print(f"Skipped {len(skipped)} corrupted files: {skipped}")

    avg_time = np.mean(batch_times) if batch_times else 0
    total_time = sum(batch_times)
    print(f"\nExtraction complete. {len(emb_list)}/{len(files)} files in {total_time:.1f}s "
          f"(avg {avg_time:.2f}s/file)")
    print("Done.")


if __name__ == "__main__":
    main()
