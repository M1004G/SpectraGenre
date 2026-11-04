"""The CNN, plus the function that turns spectrograms into class probabilities."""
import numpy as np
import torch
import torch.nn as nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class GenreCNN(nn.Module):
    """Input (batch, 1, mel bands, frames). Works for any number of frames
    because the global average pool collapses the time axis."""

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def predict_proba(model, mel, mean, std, device="cpu", batch_size=64):
    """mel: array (N, mel bands, frames) -> probabilities (N, n_classes)."""
    model.eval()
    out = []
    for i in range(0, len(mel), batch_size):
        x = (np.asarray(mel[i : i + batch_size], dtype=np.float32) - mean) / std
        logits = model(torch.from_numpy(x).unsqueeze(1).to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)
