"""Step 4: measuring. Holds the helpers the other scripts share, and prints the report.
Run:  python evaluate.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from config import CACHE_DIR

RESULTS_DIR = Path("results")


def load_cached(split):
    """Return (log-mel array, MFCC array, integer labels, class names) for one split."""
    meta = pd.read_csv(CACHE_DIR / f"meta_{split}.csv")
    classes = sorted(pd.read_csv(CACHE_DIR / "meta_training.csv")["genre"].unique())
    y = meta["genre"].map({g: i for i, g in enumerate(classes)}).to_numpy()
    mel = np.load(CACHE_DIR / f"mel_{split}.npy", mmap_mode="r")  # read from disk lazily
    mfcc = np.load(CACHE_DIR / f"mfcc_{split}.npy")
    return mel, mfcc, y, classes


def compute_metrics(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "per_genre_f1": dict(zip(classes, map(float, f1_score(y_true, y_pred, labels=labels, average=None)))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": list(classes),
    }


def save_result(name, metrics):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(metrics, indent=2))


def load_results(prefix):
    return [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob(f"{prefix}*.json"))]


def mean_std(values):
    """'0.612' for one run, '0.612 ± 0.008' (sample std) for several runs."""
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def main():
    svm, cnn = load_results("svm"), load_results("cnn_seed")
    if not svm or not cnn:
        print("Run baseline_svm.py and train_cnn.py first.")
        return
    classes = svm[0]["classes"]
    print(f"Chance level (balanced classes): {1 / len(classes):.3f}\n")
    print(f"{'Model':<14}{'Accuracy':>18}{'Macro-F1':>18}")
    for name, runs in [("MFCC + SVM", svm), ("Mel CNN", cnn)]:
        acc = mean_std([r["accuracy"] for r in runs])
        f1 = mean_std([r["macro_f1"] for r in runs])
        print(f"{name:<14}{acc:>18}{f1:>18}")
    print(f"\n(CNN: {len(cnn)} seed(s); all numbers on the held-out test split)\n")
    print(f"{'Genre':<16}{'SVM F1':>10}{'CNN F1':>10}")
    for c in classes:
        cnn_f1 = np.mean([r["per_genre_f1"][c] for r in cnn])
        print(f"{c:<16}{svm[0]['per_genre_f1'][c]:>10.3f}{cnn_f1:>10.3f}")


if __name__ == "__main__":
    main()
