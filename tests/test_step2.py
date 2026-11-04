import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from evaluate import compute_metrics, mean_std
from model import GenreCNN, predict_proba


def test_perfect_predictions():
    m = compute_metrics(np.array([0, 1, 2, 0]), np.array([0, 1, 2, 0]), ["a", "b", "c"])
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0


def test_imperfect_predictions():
    m = compute_metrics(np.array([0, 1, 2, 0]), np.array([1, 1, 2, 2]), ["a", "b", "c"])
    assert m["accuracy"] == 0.5 and m["macro_f1"] < 1.0


def test_cnn_handles_any_clip_width():
    net = GenreCNN(8)
    for frames in (256, 646):
        x = np.random.randn(3, 64, frames).astype(np.float32)
        p = predict_proba(net, x, 0.0, 1.0)
        assert p.shape == (3, 8) and np.allclose(p.sum(axis=1), 1, atol=1e-5)


def test_mean_std_format():
    assert mean_std([0.5]) == "0.500"
    assert "±" in mean_std([0.5, 0.6])
