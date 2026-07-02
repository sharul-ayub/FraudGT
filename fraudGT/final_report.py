"""Generate the final BTC binary-classification evaluation report."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _as_fraud_probability(pred_score: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return class-1 probabilities and labels using FraudGT's prediction rules."""
    scores = pred_score.detach().cpu().numpy()

    if scores.ndim == 2:
        if scores.shape[1] != 2:
            raise ValueError(
                "Final report requires exactly two output classes; "
                f"received shape {scores.shape}."
            )
        predicted_label = scores.argmax(axis=1).astype(np.int64)
        probability_rows = np.exp(scores)
        if not np.allclose(probability_rows.sum(axis=1), 1.0, atol=1e-4):
            shifted = scores - scores.max(axis=1, keepdims=True)
            probability_rows = np.exp(shifted)
            probability_rows /= probability_rows.sum(axis=1, keepdims=True)
        return probability_rows[:, 1], predicted_label

    scores = scores.reshape(-1)
    if np.all((scores >= 0.0) & (scores <= 1.0)):
        probabilities = scores
    else:
        probabilities = 1.0 / (1.0 + np.exp(-scores))
    return probabilities, (probabilities > 0.5).astype(np.int64)


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _save_confusion_matrix(matrix: np.ndarray, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Test Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_roc_curve(
    true: np.ndarray,
    probability: np.ndarray,
    roc_auc: float,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
    if math.isfinite(roc_auc):
        false_positive_rate, true_positive_rate, _ = roc_curve(true, probability)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"FraudGT (ROC-AUC = {roc_auc:.5f})",
        )
    else:
        axis.text(0.5, 0.5, "ROC curve unavailable: test labels contain one class", ha="center")
    axis.set(
        title="Test ROC Curve",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_precision_recall_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    prevalence: float,
    pr_auc: float,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence = {prevalence:.5f}")
    axis.plot(recall, precision, label=f"FraudGT (PR-AUC = {pr_auc:.5f})")
    axis.set(
        title="Test Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def generate_final_report(
    *,
    true: torch.Tensor,
    pred_score: torch.Tensor,
    output_dir: str | Path,
    epoch: int,
    validation_f1: float,
    test_loss: float,
) -> dict[str, Any]:
    """Write numerical and visual test results for the best validation-F1 epoch."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    labels = true.detach().cpu().numpy().reshape(-1)
    fraud_probability, predicted_label = _as_fraud_probability(pred_score)
    known = np.isin(labels, [0, 1])
    labels = labels[known].astype(np.int64)
    fraud_probability = fraud_probability[known]
    predicted_label = predicted_label[known]
    if labels.size == 0:
        raise ValueError("Final report received no known binary test labels.")

    matrix = confusion_matrix(labels, predicted_label, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(labels, fraud_probability)
    pr_auc = auc(recall_curve, precision_curve)
    average_precision = average_precision_score(labels, fraud_probability)
    try:
        roc_auc = roc_auc_score(labels, fraud_probability)
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "best_epoch": int(epoch),
        "selection_metric": "validation_f1",
        "validation_f1": float(validation_f1),
        "positive_class": 1,
        "positive_class_name": "fraud",
        "prediction_rule": "argmax_class_score",
        "score_name": "class_1_fraud_probability",
        "num_test_examples": int(labels.size),
        "fraud_prevalence": float(labels.mean()),
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(labels, predicted_label)),
        "precision": float(precision_score(labels, predicted_label, zero_division=0)),
        "recall": float(recall_score(labels, predicted_label, zero_division=0)),
        "f1": float(f1_score(labels, predicted_label, zero_division=0)),
        "roc_auc": _json_number(roc_auc),
        "pr_auc": _json_number(pr_auc),
        "average_precision": _json_number(average_precision),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["true_negative", "false_positive", "false_negative", "true_positive"])
        writer.writeheader()
        writer.writerow(
            {
                "true_negative": int(true_negative),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_positive": int(true_positive),
            }
        )

    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", "predicted_label", "fraud_probability"])
        writer.writerows(
            (int(target), int(prediction), float(probability))
            for target, prediction, probability in zip(labels, predicted_label, fraud_probability)
        )

    _save_confusion_matrix(matrix, output / "confusion_matrix.png")
    _save_roc_curve(labels, fraud_probability, roc_auc, output / "roc_curve.png")
    _save_precision_recall_curve(
        recall_curve,
        precision_curve,
        float(labels.mean()),
        pr_auc,
        output / "precision_recall_curve.png",
    )
    return metrics
