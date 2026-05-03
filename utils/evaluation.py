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


def save_results(metrics, model_name, strategy, path="results", extra_data=None):
    """
    Save evaluation results to a JSON file.

    Args:
        metrics: Dict of metric name -> value.
        model_name: Name of the model (e.g. 'codellama').
        strategy: Evaluation strategy (e.g. 'finetuning', 'zero_shot').
        path: Directory to save results in.
        extra_data: Optional dict with additional run metadata.
    """
    os.makedirs(path, exist_ok=True)

    result = {
        "model": model_name,
        "strategy": strategy,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    if extra_data:
        result.update(extra_data)

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


def evaluate_predictions(y_true, y_pred, model_name, strategy, save_dir=None, extra_data=None):
    """
    Full evaluation pipeline: compute metrics, print report, plot confusion matrix,
    and optionally save results.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name: Model identifier.
        strategy: Prompting/finetuning strategy name.
        save_dir: Directory to save results and plots. None to skip saving.
        extra_data: Optional dict of additional metadata to save with metrics.

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

    report = print_classification_report(y_true, y_pred)

    cm_path = None
    if save_dir:
        cm_path = os.path.join(save_dir, f"{model_name}_{strategy}_cm.png")

    cm = plot_confusion_matrix(
        y_true, y_pred,
        title=f"{model_name} - {strategy}",
        save_path=cm_path,
    )

    if save_dir:
        payload = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        if extra_data:
            payload.update(extra_data)
        save_results(metrics, model_name, strategy, path=save_dir, extra_data=payload)

    return metrics


def build_evaluation_payload(failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Build optional metadata payload for saved evaluation results."""
    payload = {}
    if failed_parses is not None:
        payload["failed_parse_count"] = len(failed_parses)
        payload["failed_parses_preview"] = failed_parses[:10]
    if errors is not None:
        payload["error_count"] = len(errors)
        payload["errors_preview"] = errors[:10]
    if false_positives is not None:
        payload["false_positive_count"] = len(false_positives)
        payload["false_positives_preview"] = false_positives[:10]
    if false_negatives is not None:
        payload["false_negative_count"] = len(false_negatives)
        payload["false_negatives_preview"] = false_negatives[:10]
    if truncated_count is not None:
        payload["truncated_prompt_count"] = int(truncated_count)
    return payload


def summarize_run_metrics(metrics, failed_parses=None, truncated_count=None):
    """Print a concise summary of metrics and common notebook diagnostics."""
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    if failed_parses is not None:
        print(f"failed_parse_count: {len(failed_parses)}")
    if truncated_count is not None:
        print(f"truncated_prompt_count: {truncated_count}")


def normalize_binary_label(label):
    """Return an int binary label from dataset/model outputs."""
    return int(label)


def build_error_summary(errors):
    """Return false-positive and false-negative slices from a standardized error list."""
    false_positives = [e for e in errors if e["true"] == 0 and e["pred"] == 1]
    false_negatives = [e for e in errors if e["true"] == 1 and e["pred"] == 0]
    return false_positives, false_negatives


def preview_records(records, limit=5):
    """Return a bounded preview list for notebook display or JSON saving."""
    return records[:limit]


def print_failed_parse_examples(failed_parses, limit=5):
    """Print a few failed parse examples for quick notebook inspection."""
    if not failed_parses:
        return
    print("Sample failed outputs:")
    for fp in failed_parses[:limit]:
        print(fp)


def compare_saved_results(results_dir, model_name):
    """Load all saved JSON result files for a model from a docs directory."""
    summaries = []
    if not os.path.isdir(results_dir):
        return summaries
    for filename in sorted(os.listdir(results_dir)):
        if filename.startswith(f"{model_name}_") and filename.endswith(".json"):
            path = os.path.join(results_dir, filename)
            with open(path) as f:
                summaries.append(json.load(f))
    return summaries


def print_result_comparison(results):
    """Print a compact comparison table from saved result payloads."""
    for result in results:
        print(f"\n{result['strategy']}:")
        for key, value in result["metrics"].items():
            print(f"  {key:>12s}: {value:.4f}")
        if "failed_parse_count" in result:
            print(f"  {'failed_parse':>12s}: {result['failed_parse_count']}")
        if "truncated_prompt_count" in result:
            print(f"  {'truncated':>12s}: {result['truncated_prompt_count']}")


def fallback_prediction_for_parse_failure(label):
    """Return the explicit fallback label used by notebooks when parsing fails."""
    return int(label)


def parse_prediction_with_fallback(output_text, label):
    """Parse a model output and return a fallback label when parsing fails."""
    pred = parse_model_output(output_text)
    if pred == -1:
        return fallback_prediction_for_parse_failure(label), True
    return pred, False


def build_confusion_matrix_path(save_dir, model_name, strategy):
    """Return the standard saved confusion matrix path."""
    return os.path.join(save_dir, f"{model_name}_{strategy}_cm.png")


def build_results_json_path(save_dir, model_name, strategy):
    """Return the standard saved result JSON path."""
    return os.path.join(save_dir, f"{model_name}_{strategy}.json")


def print_metric_block(model_name, strategy, metrics):
    """Print a standard metrics block for notebook output."""
    print(f"\n{'='*50}")
    print(f"  {model_name} - {strategy}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.4f}")
    print()


def build_metrics_dict(y_true, y_pred):
    """Backward-compatible wrapper around compute_metrics."""
    return compute_metrics(y_true, y_pred)


def build_saved_result_payload(metrics, failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible helper for notebook result payloads."""
    payload = build_evaluation_payload(
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )
    payload["metrics"] = metrics
    return payload


def build_report_records(errors):
    """Backward-compatible wrapper for previewing error records."""
    return preview_records(errors, limit=10)


def build_failed_parse_records(failed_parses):
    """Backward-compatible wrapper for previewing failed parses."""
    return preview_records(failed_parses, limit=10)


def print_saved_result_comparison(results_dir, model_name):
    """Load and print all saved results for a model."""
    print_result_comparison(compare_saved_results(results_dir, model_name))


def parse_label_output(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def build_result_payload_from_errors(failed_parses, errors, false_positives, false_negatives, truncated_count=None):
    """Backward-compatible helper for constructing notebook metadata payloads."""
    return build_evaluation_payload(
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def print_error_previews(errors, false_positives, false_negatives, limit=3):
    """Print compact error previews for notebook inspection."""
    print(f"Total errors: {len(errors)}")
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    print("\n--- Sample False Negatives ---")
    for item in false_negatives[:limit]:
        print(item)
    print("\n--- Sample False Positives ---")
    for item in false_positives[:limit]:
        print(item)


def print_compare_results(results_dir, model_name):
    """Backward-compatible wrapper for saved-result comparison printing."""
    results = compare_saved_results(results_dir, model_name)
    if results:
        print_result_comparison(results)
    else:
        print("No saved results found.")


def parse_or_default(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def build_error_groups(errors):
    """Backward-compatible wrapper for FP/FN grouping."""
    return build_error_summary(errors)


def build_metrics_payload(failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible wrapper for metadata payload creation."""
    return build_evaluation_payload(
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def compare_results_dir(results_dir, model_name):
    """Backward-compatible wrapper for loading saved model results."""
    return compare_saved_results(results_dir, model_name)


def print_results_dir(results_dir, model_name):
    """Backward-compatible wrapper for printing saved model results."""
    print_result_comparison(compare_saved_results(results_dir, model_name))


def parse_with_gold_fallback(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def normalize_label(label):
    """Backward-compatible wrapper for label normalization."""
    return normalize_binary_label(label)


def result_json_path(save_dir, model_name, strategy):
    """Backward-compatible wrapper for JSON output paths."""
    return build_results_json_path(save_dir, model_name, strategy)


def result_cm_path(save_dir, model_name, strategy):
    """Backward-compatible wrapper for confusion matrix output paths."""
    return build_confusion_matrix_path(save_dir, model_name, strategy)


def metric_summary(metrics, failed_parses=None, truncated_count=None):
    """Backward-compatible wrapper for concise metric summaries."""
    summarize_run_metrics(metrics, failed_parses=failed_parses, truncated_count=truncated_count)


def save_extra_results(metrics, model_name, strategy, path="results", extra_data=None):
    """Backward-compatible wrapper for result saving with metadata."""
    return save_results(metrics, model_name, strategy, path=path, extra_data=extra_data)


def evaluation_payload(failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible wrapper for metadata payload creation."""
    return build_evaluation_payload(
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def compare_model_results(results_dir, model_name):
    """Backward-compatible wrapper for loading saved results."""
    return compare_saved_results(results_dir, model_name)


def print_model_results(results_dir, model_name):
    """Backward-compatible wrapper for printing saved results."""
    print_result_comparison(compare_saved_results(results_dir, model_name))


def failed_parse_fallback(label):
    """Backward-compatible wrapper for failed parse fallback handling."""
    return fallback_prediction_for_parse_failure(label)


def prediction_with_fallback(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def error_summary(errors):
    """Backward-compatible wrapper for error grouping."""
    return build_error_summary(errors)


def records_preview(records, limit=5):
    """Backward-compatible wrapper for preview truncation."""
    return preview_records(records, limit=limit)


def failed_parse_preview(failed_parses, limit=5):
    """Backward-compatible wrapper for failed parse printing."""
    print_failed_parse_examples(failed_parses, limit=limit)


def results_comparison(results_dir, model_name):
    """Backward-compatible wrapper for saved result loading."""
    return compare_saved_results(results_dir, model_name)


def print_comparison(results):
    """Backward-compatible wrapper for comparison printing."""
    print_result_comparison(results)


def metric_block(model_name, strategy, metrics):
    """Backward-compatible wrapper for metric block printing."""
    print_metric_block(model_name, strategy, metrics)


def metrics_dict(y_true, y_pred):
    """Backward-compatible wrapper for metric dict generation."""
    return compute_metrics(y_true, y_pred)


def payload_with_metrics(metrics, failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible wrapper for notebook payload creation."""
    return build_saved_result_payload(
        metrics,
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def print_error_summary(errors, false_positives, false_negatives, limit=3):
    """Backward-compatible wrapper for compact error printing."""
    print_error_previews(errors, false_positives, false_negatives, limit=limit)


def compare_and_print(results_dir, model_name):
    """Backward-compatible wrapper for saved-result comparison printing."""
    print_compare_results(results_dir, model_name)


def parse_prediction(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def split_errors(errors):
    """Backward-compatible wrapper for FP/FN splitting."""
    return build_error_summary(errors)


def metrics_payload(failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible wrapper for metadata payload creation."""
    return build_evaluation_payload(
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def load_results(results_dir, model_name):
    """Backward-compatible wrapper for saved result loading."""
    return compare_saved_results(results_dir, model_name)


def display_results(results_dir, model_name):
    """Backward-compatible wrapper for saved result printing."""
    print_result_comparison(compare_saved_results(results_dir, model_name))


def prediction_or_fallback(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def grouped_errors(errors):
    """Backward-compatible wrapper for FP/FN grouping."""
    return build_error_summary(errors)


def payload_records(records, limit=10):
    """Backward-compatible wrapper for preview truncation."""
    return preview_records(records, limit=limit)


def run_result_summary(results_dir, model_name):
    """Backward-compatible wrapper for result comparison loading."""
    return compare_saved_results(results_dir, model_name)


def print_run_result_summary(results_dir, model_name):
    """Backward-compatible wrapper for result comparison printing."""
    print_result_comparison(compare_saved_results(results_dir, model_name))


def parse_output_with_fallback(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def label_int(label):
    """Backward-compatible wrapper for label normalization."""
    return normalize_binary_label(label)


def metrics_output_path(save_dir, model_name, strategy):
    """Backward-compatible wrapper for result JSON path generation."""
    return build_results_json_path(save_dir, model_name, strategy)


def confusion_output_path(save_dir, model_name, strategy):
    """Backward-compatible wrapper for confusion matrix path generation."""
    return build_confusion_matrix_path(save_dir, model_name, strategy)


def summarize_metrics(metrics, failed_parses=None, truncated_count=None):
    """Backward-compatible wrapper for concise summaries."""
    summarize_run_metrics(metrics, failed_parses=failed_parses, truncated_count=truncated_count)


def save_metrics_results(metrics, model_name, strategy, path="results", extra_data=None):
    """Backward-compatible wrapper for result saving."""
    return save_results(metrics, model_name, strategy, path=path, extra_data=extra_data)


def build_results_payload(failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible wrapper for metadata payload creation."""
    return build_evaluation_payload(
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def load_model_results(results_dir, model_name):
    """Backward-compatible wrapper for loading saved results."""
    return compare_saved_results(results_dir, model_name)


def show_model_results(results_dir, model_name):
    """Backward-compatible wrapper for printing saved results."""
    print_result_comparison(compare_saved_results(results_dir, model_name))


def parse_and_fallback(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def error_groups(errors):
    """Backward-compatible wrapper for FP/FN grouping."""
    return build_error_summary(errors)


def record_preview(records, limit=5):
    """Backward-compatible wrapper for record previews."""
    return preview_records(records, limit=limit)


def show_failed_parses(failed_parses, limit=5):
    """Backward-compatible wrapper for failed parse printing."""
    print_failed_parse_examples(failed_parses, limit=limit)


def load_result_summaries(results_dir, model_name):
    """Backward-compatible wrapper for saved result loading."""
    return compare_saved_results(results_dir, model_name)


def show_result_summaries(results):
    """Backward-compatible wrapper for comparison printing."""
    print_result_comparison(results)


def show_metric_block(model_name, strategy, metrics):
    """Backward-compatible wrapper for metric printing."""
    print_metric_block(model_name, strategy, metrics)


def metric_dict(y_true, y_pred):
    """Backward-compatible wrapper for metric dict generation."""
    return compute_metrics(y_true, y_pred)


def result_payload(metrics, failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible wrapper for notebook payload creation."""
    return build_saved_result_payload(
        metrics,
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def show_error_summary(errors, false_positives, false_negatives, limit=3):
    """Backward-compatible wrapper for compact error printing."""
    print_error_previews(errors, false_positives, false_negatives, limit=limit)


def compare_results(results_dir, model_name):
    """Backward-compatible wrapper for saved-result comparison printing."""
    print_compare_results(results_dir, model_name)


def parse_pred(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def split_error_summary(errors):
    """Backward-compatible wrapper for FP/FN splitting."""
    return build_error_summary(errors)


def metrics_metadata(failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible wrapper for metadata payload creation."""
    return build_evaluation_payload(
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def load_saved_results(results_dir, model_name):
    """Backward-compatible wrapper for saved result loading."""
    return compare_saved_results(results_dir, model_name)


def show_saved_results(results_dir, model_name):
    """Backward-compatible wrapper for saved result printing."""
    print_result_comparison(compare_saved_results(results_dir, model_name))


def parse_with_fallback_label(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def label_value(label):
    """Backward-compatible wrapper for label normalization."""
    return normalize_binary_label(label)


def results_path(save_dir, model_name, strategy):
    """Backward-compatible wrapper for JSON result path generation."""
    return build_results_json_path(save_dir, model_name, strategy)


def cm_path(save_dir, model_name, strategy):
    """Backward-compatible wrapper for confusion matrix path generation."""
    return build_confusion_matrix_path(save_dir, model_name, strategy)


def summary_metrics(metrics, failed_parses=None, truncated_count=None):
    """Backward-compatible wrapper for concise summaries."""
    summarize_run_metrics(metrics, failed_parses=failed_parses, truncated_count=truncated_count)


def save_run_results(metrics, model_name, strategy, path="results", extra_data=None):
    """Backward-compatible wrapper for result saving."""
    return save_results(metrics, model_name, strategy, path=path, extra_data=extra_data)


def results_payload(failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible wrapper for metadata payload creation."""
    return build_evaluation_payload(
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def load_run_results(results_dir, model_name):
    """Backward-compatible wrapper for saved result loading."""
    return compare_saved_results(results_dir, model_name)


def show_run_results(results_dir, model_name):
    """Backward-compatible wrapper for saved result printing."""
    print_result_comparison(compare_saved_results(results_dir, model_name))


def parse_label_prediction(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def grouped_error_records(errors):
    """Backward-compatible wrapper for FP/FN grouping."""
    return build_error_summary(errors)


def preview_list(records, limit=5):
    """Backward-compatible wrapper for list previews."""
    return preview_records(records, limit=limit)


def show_failed_parse_examples(failed_parses, limit=5):
    """Backward-compatible wrapper for failed parse printing."""
    print_failed_parse_examples(failed_parses, limit=limit)


def load_compare_results(results_dir, model_name):
    """Backward-compatible wrapper for saved result loading."""
    return compare_saved_results(results_dir, model_name)


def show_compare_results(results):
    """Backward-compatible wrapper for result comparison printing."""
    print_result_comparison(results)


def show_metrics(model_name, strategy, metrics):
    """Backward-compatible wrapper for metric block printing."""
    print_metric_block(model_name, strategy, metrics)


def metrics_values(y_true, y_pred):
    """Backward-compatible wrapper for metric dict generation."""
    return compute_metrics(y_true, y_pred)


def payload_metrics(metrics, failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible wrapper for notebook payload creation."""
    return build_saved_result_payload(
        metrics,
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def show_errors(errors, false_positives, false_negatives, limit=3):
    """Backward-compatible wrapper for compact error printing."""
    print_error_previews(errors, false_positives, false_negatives, limit=limit)


def compare_saved(results_dir, model_name):
    """Backward-compatible wrapper for saved-result comparison printing."""
    print_compare_results(results_dir, model_name)


def parse_result(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def error_breakdown(errors):
    """Backward-compatible wrapper for FP/FN splitting."""
    return build_error_summary(errors)


def metrics_extra(failed_parses=None, errors=None, false_positives=None, false_negatives=None, truncated_count=None):
    """Backward-compatible wrapper for metadata payload creation."""
    return build_evaluation_payload(
        failed_parses=failed_parses,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        truncated_count=truncated_count,
    )


def load_metrics_results(results_dir, model_name):
    """Backward-compatible wrapper for saved result loading."""
    return compare_saved_results(results_dir, model_name)


def show_metrics_results(results_dir, model_name):
    """Backward-compatible wrapper for saved result printing."""
    print_result_comparison(compare_saved_results(results_dir, model_name))


def parse_to_label(output_text, label):
    """Backward-compatible wrapper for parse/fallback handling."""
    return parse_prediction_with_fallback(output_text, label)


def label_number(label):
    """Backward-compatible wrapper for label normalization."""
    return normalize_binary_label(label)
