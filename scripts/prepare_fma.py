"""Download and prepare FMA Small dataset.

FMA Small: 8000 tracks, 8 genres, ~7.2 GB audio + 342 MB metadata.
Downloads from University of Lausanne servers.
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FMA_DIR = DATA_DIR / "fma"

FMA_AUDIO_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_AUDIO_SIZE = 7565311302  # ~7.2 GB
FMA_META_SIZE = 341839560    # ~342 MB

FMA_GENRES = [
    "electronic", "experimental", "folk", "hip-hop",
    "instrumental", "international", "pop", "rock"
]


def _download_with_progress(url, dest, expected_size=None):
    import urllib.request

    dest = Path(dest)
    if dest.exists() and expected_size and abs(dest.stat().st_size - expected_size) < 1024:
        print(f"  [SKIP] {dest.name} already exists ({dest.stat().st_size / 1024**2:.0f} MB)")
        return

    print(f"  Downloading {dest.name} ...")
    print(f"    URL: {url}")
    if expected_size:
        print(f"    Expected size: {expected_size / 1024**3:.1f} GB")

    def _report(count, block_size, total_size):
        pct = min(count * block_size / total_size * 100, 100) if total_size > 0 else 0
        downloaded = count * block_size
        sys.stdout.write(f"\r    {downloaded / 1024**3:.2f} GB / {total_size / 1024**3:.2f} GB ({pct:.0f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=_report)
    print()
    actual = dest.stat().st_size
    print(f"    Downloaded: {actual / 1024**3:.2f} GB")


def _extract_zip(zip_path, extract_to):
    extract_to = Path(extract_to)
    first_file = None
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        first_file = names[0].split("/")[0] if names else None
    marker = extract_to / first_file
    if marker and marker.exists():
        print(f"  [SKIP] Already extracted: {marker}")
        return

    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Extraction complete.")


def build_genre_mapping(tracks_csv_path):
    print(f"  Loading tracks metadata: {tracks_csv_path}")
    tracks = pd.read_csv(tracks_csv_path, index_col=0, header=[0, 1])

    track_ids = tracks.index.values.astype(int)
    top_level = tracks[("track", "genre_top")]
    genres = top_level.values

    valid_mask = pd.notna(genres)
    track_ids = track_ids[valid_mask]
    genres = genres[valid_mask]
    genres = np.array([g.lower().replace(" ", "-").replace("_", "-") for g in genres])

    valid_genre_mask = np.isin(genres, FMA_GENRES)
    track_ids = track_ids[valid_genre_mask]
    genres = genres[valid_genre_mask]

    print(f"  Valid tracks with top-level genre: {len(track_ids)}")
    return track_ids, genres


def main(download=False):
    FMA_DIR.mkdir(parents=True, exist_ok=True)

    audio_zip = FMA_DIR / "fma_small.zip"
    meta_zip = FMA_DIR / "fma_metadata.zip"
    audio_dir = FMA_DIR / "fma_small"
    meta_dir = FMA_DIR / "fma_metadata"

    if download:
        print("=== DOWNLOADING FMA DATASET ===\n")
        _download_with_progress(FMA_AUDIO_URL, audio_zip, FMA_AUDIO_SIZE)
        _download_with_progress(FMA_META_URL, meta_zip, FMA_META_SIZE)

        print("\n=== EXTRACTING ===")
        _extract_zip(audio_zip, FMA_DIR)
        _extract_zip(meta_zip, FMA_DIR)
    else:
        if not audio_dir.exists() and not audio_zip.exists():
            print("FMA data not found. Download URLs:\n")
            print(f"  Audio:     {FMA_AUDIO_URL}")
            print(f"  Metadata:  {FMA_META_URL}\n")
            print("Place files in data/fma/ and re-run, or use --download flag.\n")
            print(f"  Expected location:")
            print(f"    {audio_zip}")
            print(f"    {meta_zip}")
            return

        if audio_zip.exists() and not audio_dir.exists():
            _extract_zip(audio_zip, FMA_DIR)
        if meta_zip.exists() and not meta_dir.exists():
            _extract_zip(meta_zip, FMA_DIR)

    print("\n=== VERIFYING DATA ===")

    tracks_csv = meta_dir / "tracks.csv"
    if not tracks_csv.exists():
        print(f"  ERROR: tracks.csv not found at {tracks_csv}")
        print("  Make sure fma_metadata.zip is extracted correctly.")
        return

    track_ids, genres = build_genre_mapping(tracks_csv)

    print("\n  Checking audio files ...")
    found = 0
    missing = 0
    file_paths = []
    file_genres = []
    for tid, genre in zip(track_ids, genres):
        subdir = f"{tid:06d}"[:3]
        fname = f"{tid:06d}.mp3"
        fpath = audio_dir / subdir / fname
        if fpath.exists():
            found += 1
            file_paths.append(str(fpath))
            file_genres.append(genre)
        else:
            missing += 1

    print(f"  Found: {found}, Missing: {missing}")

    mapping_df = pd.DataFrame({"path": file_paths, "genre": file_genres})
    mapping_path = FMA_DIR / "genre_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"\n  Saved genre mapping: {mapping_path}")

    print("\n=== GENRE DISTRIBUTION ===")
    for genre in FMA_GENRES:
        count = sum(1 for g in file_genres if g == genre)
        print(f"  {genre:15s}: {count:4d}")

    total = sum(1 for g in file_genres if g in FMA_GENRES)
    print(f"  {'TOTAL':15s}: {total:4d}")
    print(f"\n=== FMA DATA READY ===")


if __name__ == "__main__":
    download_flag = "--download" in sys.argv or "-d" in sys.argv
    main(download=download_flag)
