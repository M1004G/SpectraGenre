"""Step 2: decode every clip once, compute features, cache them as .npy files.
Run:  python prepare_data.py            (full run, can take a while)
      python prepare_data.py --limit 20 (quick test: 20 clips per split)
"""
import argparse
import numpy as np
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm

from config import SR, CLIP_SECONDS, CACHE_DIR, SPLITS, N_MELS, N_FRAMES
from data_utils import load_metadata, audio_path
from features import extract


def process_track(track_id):
    """Load one mp3 and return (log_mel, mfcc) or None if the file is unreadable."""
    try:
        y, _ = librosa.load(audio_path(track_id), sr=SR, mono=True, duration=CLIP_SECONDS)
        if len(y) < SR:  # under 1 second: treat as broken
            return None
        log_mel, mfcc = extract(y)
        return log_mel.astype(np.float16), mfcc  # float16 halves memory
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="max clips per split")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel workers")
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    meta = load_metadata()

    for split in SPLITS:
        rows = meta[meta["split"] == split]
        if args.limit:
            rows = rows.head(args.limit)
        ids = rows["track_id"].tolist()

        results = Parallel(n_jobs=args.jobs)(
            delayed(process_track)(t) for t in tqdm(ids, desc=split)
        )

        keep = [i for i, r in enumerate(results) if r is not None]
        skipped = len(ids) - len(keep)
        mels = np.stack([results[i][0] for i in keep])
        mfccs = np.stack([results[i][1] for i in keep])
        kept_rows = rows.iloc[keep]

        np.save(CACHE_DIR / f"mel_{split}.npy", mels)
        np.save(CACHE_DIR / f"mfcc_{split}.npy", mfccs)
        kept_rows.to_csv(CACHE_DIR / f"meta_{split}.csv", index=False)
        print(f"{split}: saved {len(keep)} clips, skipped {skipped}, "
              f"mel shape {mels.shape} (expected ({len(keep)}, {N_MELS}, {N_FRAMES}))")


if __name__ == "__main__":
    main()