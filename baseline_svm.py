"""Step 2: baseline. MFCC summary vector -> standardise -> SVM.
Run:  python baseline_svm.py
"""
from pathlib import Path

import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate import compute_metrics, load_cached, save_result


def main():
    _, X_tr, y_tr, classes = load_cached("training")
    _, X_va, y_va, _ = load_cached("validation")
    _, X_te, y_te, _ = load_cached("test")

    # Choose the SVM's C on the validation split only
    best_acc, best_model, best_C = -1, None, None
    for C in [0.1, 1, 10, 100]:
        model = make_pipeline(StandardScaler(), SVC(C=C))
        model.fit(X_tr, y_tr)
        acc = model.score(X_va, y_va)
        print(f"C={C}: validation accuracy {acc:.3f}")
        if acc > best_acc:
            best_acc, best_model, best_C = acc, model, C

    # The test split is used exactly once, for the chosen model
    metrics = compute_metrics(y_te, best_model.predict(X_te), classes)
    metrics.update(C=best_C, best_val_accuracy=float(best_acc))
    save_result("svm", metrics)
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "classes": classes}, "models/svm.joblib")
    print(f"\nTest accuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f} (C={best_C})")


if __name__ == "__main__":
    main()
