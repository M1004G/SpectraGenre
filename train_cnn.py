"""Step 3: train the CNN on the cached log-mel spectrograms.
Run:  python train_cnn.py --seeds 1 --epochs 3     (quick check)
      python train_cnn.py --seeds 3 --epochs 30    (the real run)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evaluate import compute_metrics, load_cached, save_result
from model import GenreCNN, predict_proba

CROP = 256        # frames (about 12 s): each training step sees a random crop of the clip
BATCH = 32
MODELS_DIR = Path("models")


def train_one(seed, epochs, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mel_tr, _, y_tr, classes = load_cached("training")
    mel_va, _, y_va, _ = load_cached("validation")
    mel_te, _, y_te, _ = load_cached("test")

    # Normalisation constants come from the training split only
    sample = mel_tr[::10]
    mean, std = float(sample.mean(dtype=np.float32)), float(sample.std(dtype=np.float32))

    model = GenreCNN(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_val, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        start, total = time.time(), 0.0
        model.train()
        order = rng.permutation(len(mel_tr))
        for i in range(0, len(order), BATCH):
            idx = np.sort(order[i : i + BATCH])
            s = int(rng.integers(0, mel_tr.shape[2] - CROP + 1))
            x = (mel_tr[idx][:, :, s : s + CROP].astype(np.float32) - mean) / std
            x = torch.from_numpy(x).unsqueeze(1).to(device)
            y = torch.from_numpy(y_tr[idx]).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        val_acc = float((predict_proba(model, mel_va, mean, std, device).argmax(1) == y_va).mean())
        print(f"seed {seed} epoch {epoch:>2}  loss {total / len(order):.3f}  "
              f"val acc {val_acc:.3f}  ({time.time() - start:.0f}s)")
        if val_acc > best_val:  # keep the weights from the best validation epoch
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_pred = predict_proba(model, mel_te, mean, std, device).argmax(1)
    metrics = compute_metrics(y_te, test_pred, classes)
    metrics.update(seed=seed, epochs=epochs, best_val_accuracy=best_val)
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std, "classes": classes},
               MODELS_DIR / f"cnn_seed{seed}.pt")
    save_result(f"cnn_seed{seed}", metrics)
    print(f"seed {seed}: test accuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}\n")
    for seed in range(args.seeds):
        train_one(seed, args.epochs, device)


if __name__ == "__main__":
    main()
