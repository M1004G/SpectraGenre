"""All constants live here so every script uses the same values."""
from pathlib import Path

# Folder layout (you create data/ by downloading FMA, see README)
DATA_DIR = Path("data")
METADATA_CSV = DATA_DIR / "fma_metadata" / "tracks.csv"
AUDIO_DIR = DATA_DIR / "fma_small"
CACHE_DIR = Path("cache")

# Audio settings
SR = 22050            # samples per second we resample every clip to
CLIP_SECONDS = 30     # FMA-small clips are 30 s long
N_FFT = 2048          # window size for the spectrogram (samples)
HOP = 1024            # step between windows (samples)
N_MELS = 64           # number of mel frequency bands
N_MFCC = 20           # number of MFCC coefficients

N_SAMPLES = SR * CLIP_SECONDS
# With librosa's default centered windows: 1 + N_SAMPLES // HOP = 646 time frames
N_FRAMES = 1 + N_SAMPLES // HOP

SPLITS = ["training", "validation", "test"]
