"""Step 1: look at the data BEFORE training anything.

Prints split sizes, genre balance, whether any artist appears in more than one
split (leakage), and whether the audio files are actually on disk.
Run:  python check_split.py
"""
import pandas as pd
from data_utils import load_metadata, audio_path
from config import SPLITS


def main():
    meta = load_metadata()
    print(f"Tracks in fma_small metadata: {len(meta)}\n")

    print("Tracks per split:")
    print(meta["split"].value_counts().reindex(SPLITS), "\n")

    print("Tracks per genre per split:")
    print(pd.crosstab(meta["genre"], meta["split"])[SPLITS], "\n")

    # Leakage check: does any artist appear in more than one split?
    splits_per_artist = meta.groupby("artist_id")["split"].nunique()
    leaking = int((splits_per_artist > 1).sum())
    print(f"Artists appearing in more than one split: {leaking} "
          f"(out of {len(splits_per_artist)} artists)")
    if leaking == 0:
        print("-> splits are artist-disjoint\n")
    else:
        print("-> NOT artist-disjoint; do not claim that on the resume\n")

    # Are the audio files present?
    missing = [t for t in meta["track_id"] if not audio_path(t).exists()]
    print(f"Audio files missing on disk: {len(missing)} of {len(meta)}")


if __name__ == "__main__":
    main()
