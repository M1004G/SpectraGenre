"""Helpers for reading the FMA metadata and finding audio files."""
import pandas as pd
from config import AUDIO_DIR, METADATA_CSV


def load_metadata(csv_path=METADATA_CSV):
    """Return one row per fma_small track: track_id, split, genre, artist_id."""
    # tracks.csv has a two-row header (a "group" row and a "field" row)
    tracks = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    small = tracks[tracks[("set", "subset")] == "small"]
    meta = pd.DataFrame(
        {
            "track_id": small.index.astype(int),
            "split": small[("set", "split")].values,
            "genre": small[("track", "genre_top")].values,
            "artist_id": small[("artist", "id")].values,
        }
    )
    return meta.reset_index(drop=True)


def audio_path(track_id, audio_dir=AUDIO_DIR):
    """FMA stores track 2 as fma_small/000/000002.mp3 (folder = first 3 digits)."""
    tid = f"{int(track_id):06d}"
    return audio_dir / tid[:3] / f"{tid}.mp3"
