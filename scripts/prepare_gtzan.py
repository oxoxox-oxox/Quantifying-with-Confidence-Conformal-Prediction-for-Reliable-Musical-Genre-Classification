"""Step 3: Data preparation - download GTZAN via torchaudio."""
import torchaudio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GTZAN_ROOT = DATA_DIR / "gtzan"

GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]


def download():
    genres_dir = GTZAN_ROOT / "genres_original"
    if genres_dir.exists() and len(list(genres_dir.iterdir())) >= 10:
        print(f"[SKIP] GTZAN already exists at {GTZAN_ROOT}")
        return
    print("[DOWNLOAD] GTZAN via torchaudio (may take 2-5 min for ~1.2GB)...")
    ds = torchaudio.datasets.GTZAN(
        root=str(DATA_DIR),
        url="http://opihi.cs.uvic.ca/sound/genres.tar.gz",
        folder_in_archive="genres_original",
        download=True
    )
    print(f"[DONE] Downloaded to {ds._path}")


def verify():
    total = 0
    for genre in GENRES:
        genre_dir = GTZAN_ROOT / "genres_original" / genre
        if not genre_dir.exists():
            print(f"  MISSING: {genre_dir}")
            continue
        n = len(list(genre_dir.glob("*.wav")))
        status = "OK" if n == 100 else f"WARN: {n}/100"
        print(f"  {genre:12s}: {status}")
        total += n
    print(f"  {'TOTAL':12s}: {total} / 1000")
    return total == 1000


if __name__ == "__main__":
    download()
    ok = verify()
    print("\n=== GTZAN READY ===" if ok else "\n!!! FIX ISSUES ABOVE !!!")
