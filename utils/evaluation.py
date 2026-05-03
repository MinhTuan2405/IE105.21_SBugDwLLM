"""
Evaluation utilities for computing classification metrics
and generating confusion matrices for defect detection models.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred):
    """
    Compute all required classification metrics.

    Returns:
        Dict with accuracy, precision, recall, f1 scores.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        title: Plot title.
        save_path: If provided, saves the figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Safe (0)", "Defective (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
    return cm


def print_classification_report(y_true, y_pred):
    """Print a detailed sklearn classification report."""
    target_names = ["Safe (0)", "Defective (1)"]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    return report


def save_results(metrics, model_name, strategy, path="results"):
    """
    Save evaluation results to a JSON file.

    Args:
        metrics: Dict of metric name -> value.
        model_name: Name of the model (e.g. 'codellama').
        strategy: Evaluation strategy (e.g. 'finetuning', 'zero_shot').
        path: Directory to save results in.
    """
    os.makedirs(path, exist_ok=True)

    result = {
        "model": model_name,
        "strategy": strategy,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    filename = f"{model_name}_{strategy}.json"
    filepath = os.path.join(path, filename)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {filepath}")
    return filepath


def parse_model_output(output_text):
    """
    Extract a binary prediction (0 or 1) from model output text.

    Handles common output formats: plain "0"/"1", wrapped in text, etc.
    Returns -1 if parsing fails.
    """
    text = output_text.strip()

    if text in ("0", "1"):
        return int(text)

    for char in text:
        if char in ("0", "1"):
            return int(char)

    return -1


def evaluate_predictions(y_true, y_pred, model_name, strategy, save_dir=None):
    """
    Full evaluation pipeline: compute metrics, print report, plot confusion matrix,
    and optionally save results.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name: Model identifier.
        strategy: Prompting/finetuning strategy name.
        save_dir: Directory to save results and plots. None to skip saving.

    Returns:
        Dict of computed metrics.
    """
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} - {strategy}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.4f}")
    print()

    print_classification_report(y_true, y_pred)

    cm_path = None
    if save_dir:
        cm_path = os.path.join(save_dir, f"{model_name}_{strategy}_cm.png")

    plot_confusion_matrix(
        y_true, y_pred,
        title=f"{model_name} - {strategy}",
        save_path=cm_path,
    )

    if save_dir:
        save_results(metrics, model_name, strategy, path=save_dir)

    return metrics
