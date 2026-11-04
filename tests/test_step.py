import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from config import SR, N_SAMPLES, N_MELS, N_FRAMES, N_MFCC
from features import fix_length, extract
from data_utils import load_metadata, audio_path


def sine(freq, seconds=30):
    t = np.arange(int(SR * seconds)) / SR
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_fix_length_pads_and_crops():
    assert len(fix_length(np.zeros(10))) == N_SAMPLES
    assert len(fix_length(np.zeros(N_SAMPLES + 500))) == N_SAMPLES


def test_extract_shapes():
    log_mel, mfcc = extract(sine(440))
    assert log_mel.shape == (N_MELS, N_FRAMES)
    assert mfcc.shape == (2 * N_MFCC,)


def test_different_sounds_give_different_features():
    _, a = extract(sine(220))
    _, b = extract(sine(3000))
    assert not np.allclose(a, b)


def test_short_clip_is_padded_not_crashing():
    log_mel, _ = extract(sine(440, seconds=5))
    assert log_mel.shape == (N_MELS, N_FRAMES)


def test_audio_path_format():
    assert audio_path(2).as_posix().endswith("fma_small/000/000002.mp3")
    assert audio_path(123456).as_posix().endswith("fma_small/123/123456.mp3")


def test_load_metadata_keeps_only_small(tmp_path):
    cols = pd.MultiIndex.from_tuples(
        [("set", "split"), ("set", "subset"), ("track", "genre_top"), ("artist", "id")]
    )
    df = pd.DataFrame(
        [["training", "small", "Rock", 1],
         ["test", "small", "Hip-Hop", 2],
         ["training", "medium", "Pop", 3]],
        index=pd.Index([2, 5, 10], name="track_id"), columns=cols,
    )
    p = tmp_path / "tracks.csv"
    df.to_csv(p)
    meta = load_metadata(p)
    assert list(meta["track_id"]) == [2, 5]
    assert list(meta["genre"]) == ["Rock", "Hip-Hop"]
    assert list(meta["split"]) == ["training", "test"]
