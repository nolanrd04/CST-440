from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_classifier(model, x_test: np.ndarray, y_test: np.ndarray, class_names: list[str]) -> dict:
    """Step 7: Compute accuracy, confusion matrix, and per-class metrics."""
    probs = model.predict(x_test, batch_size=32, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    summary = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=class_names, output_dict=True),
    }
    return summary


def save_evaluation(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
