"""Turn one audio waveform into the two inputs our models need."""
import numpy as np
import librosa
from config import SR, N_SAMPLES, N_FFT, HOP, N_MELS, N_MFCC


def fix_length(y, n_samples=N_SAMPLES):
    """Crop to n_samples, or pad with silence (zeros) if the clip is shorter."""
    if len(y) >= n_samples:
        return y[:n_samples]
    return np.pad(y, (0, n_samples - len(y)))


def log_mel_spectrogram(y):
    """Shape (N_MELS, N_FRAMES): loudness per mel band per time step, in dB."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def mfcc_summary(log_mel):
    """Shape (2*N_MFCC,): mean and std over time of each MFCC coefficient.
    This fixed-size vector is the input for the SVM baseline."""
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def extract(y):
    """Waveform -> (log-mel spectrogram, MFCC summary vector)."""
    y = fix_length(y)
    log_mel = log_mel_spectrogram(y)
    return log_mel, mfcc_summary(log_mel)
