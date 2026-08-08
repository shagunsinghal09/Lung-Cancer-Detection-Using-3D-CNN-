"""Example training entry-point for research experiments."""

from pathlib import Path

import numpy as np

from ml.evaluation.metrics import evaluate_predictions


def run_dummy_training(output_dir: str = "artifacts") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.8, 0.6, 0.2])
    y_pred = (y_score >= 0.5).astype(int)
    metrics = evaluate_predictions(y_true, y_pred, y_score)
    return metrics


if __name__ == "__main__":
    print(run_dummy_training())
