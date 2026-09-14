#!/usr/bin/env python3
"""Reproduce the metrics reported in the paper from raw per-case JSON results.

Reads the `*_detailed_*.json` files written by `run.py` under `results/` and
prints, for every model and every task:

  * MNV subtype classification (four-class): balanced accuracy, macro-F1,
    per-class recall and the confusion matrix.
  * Treatment response (four-class): balanced accuracy, macro-F1, per-class
    recall and the confusion matrix.
  * Treatment response (binary, Good/Partial vs Poor/Non-response): balanced
    accuracy and macro-F1.

Balanced accuracy is the mean of the per-class recalls and macro-F1 the mean of
the per-class F1 scores; both are used unchanged for the four-class and the
binary task so that the two parts of the paper share one metric definition.

Label mapping and metric definitions are imported from `run.py` and `utils/`
so that this script and the in-run evaluation in `run.py` always agree.

Usage:
    python scripts/reproduce_metrics.py [RESULTS_DIR]

`RESULTS_DIR` defaults to `results/`. Run `run.py` first to generate the JSON
files, or point this script at a directory containing previously produced ones.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from run import TASKS
from utils import (
    MAX_TOKENS,
    compute_average_metrics,
    compute_confusion_matrix,
    compute_metrics_from_confusion,
    get_model_config,
)

MODEL_KEYS = ["GPT", "Gemini", "Qwen"]

# Response tasks, ordered as they appear in the paper.
RESPONSE_TASKS = ["response_direct", "given_biomarker_noimg", "given_biomarker"]


def load_run_files(results_dir, prefix, model_short):
    """Return the sorted list of per-run JSON files for one task and model."""
    pattern = os.path.join(
        results_dir, f"{prefix}_detailed_{model_short}_tokens{MAX_TOKENS}_run[0-9].json"
    )
    return sorted(glob.glob(pattern))


def evaluate_four_class(files, task):
    """Per-run four-class metrics: balanced accuracy, macro-F1, per-class recall, CM."""
    labels = task["eval_labels"]
    map_label = task["map_label"]
    runs = []
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        y_true, y_pred = [], []
        n_unrecognized = 0
        for d in data:
            gt = map_label(d.get("ground_truth"))
            if gt is None or gt == "":
                continue
            pred = map_label(d.get("extracted_prediction"))
            if pred is None or pred == "":
                # Missing or unrecognized prediction: excluded from all metrics.
                n_unrecognized += 1
                continue
            y_true.append(gt)
            y_pred.append(pred)

        if not y_true:
            continue

        cm, _ = compute_confusion_matrix(y_true, y_pred, labels=labels)
        metrics = compute_metrics_from_confusion(cm)
        for lb in metrics:
            metrics[lb]["support"] = sum(1 for t in y_true if t == lb)

        balanced = np.mean([metrics[lb]["recall"] for lb in labels]) * 100
        macro = compute_average_metrics(metrics, "macro")

        runs.append({
            "balanced_acc": balanced,
            "macro_f1": macro["f1"] * 100,
            "per_class": metrics,
            "cm": cm,
            "n": len(y_true),
            "excluded": n_unrecognized,
        })
    return runs


def evaluate_binary(files, task):
    """Per-run binary metrics: Good/Partial vs Poor/Non-response.

    Uses the same metric definitions as the four-class tasks -- balanced
    accuracy (mean per-class recall) and macro-F1 -- so that both parts of the
    paper are reported on a common scale.
    """
    map_label = task["map_label"]
    # Column order: class '1' = positive (Good/Partial), class '0' = negative.
    labels = ["1", "0"]
    runs = []
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        y_true, y_pred = [], []
        n_unrecognized = 0
        for d in data:
            gt = map_label(d.get("ground_truth"))
            if gt is None or gt == "":
                continue
            pred = map_label(d.get("extracted_prediction"))
            if pred is None or pred == "":
                # Missing or unrecognized prediction: excluded from all metrics.
                n_unrecognized += 1
                continue
            # Labels '1'/'2' are Good/Partial; '3'/'4' are Poor/Non-response.
            y_true.append("1" if gt in ("1", "2") else "0")
            y_pred.append("1" if pred in ("1", "2") else "0")

        if not y_true:
            continue

        cm, _ = compute_confusion_matrix(y_true, y_pred, labels=labels)
        metrics = compute_metrics_from_confusion(cm)
        for lb in metrics:
            metrics[lb]["support"] = sum(1 for t in y_true if t == lb)

        balanced = np.mean([metrics[lb]["recall"] for lb in labels]) * 100
        macro = compute_average_metrics(metrics, "macro")

        runs.append({
            "balanced_acc": balanced,
            "macro_f1": macro["f1"] * 100,
            "per_class": metrics,
            "cm": cm,
            "n": len(y_true),
            "excluded": n_unrecognized,
        })
    return runs


def _ms(values):
    """Format mean ± std (population std when only one run is available)."""
    arr = np.array(values, dtype=float)
    if len(arr) == 1:
        return f"{arr[0]:.1f}"
    return f"{arr.mean():.1f}±{arr.std(ddof=1):.1f}"


def report_four_class(model_key, model_short, task_name, results_dir):
    task = TASKS[task_name]
    files = load_run_files(results_dir, task["file_prefix"], model_short)
    if not files:
        return None

    runs = evaluate_four_class(files, task)
    if not runs:
        return None

    labels = task["eval_labels"]
    names = task["label_names"]

    print(f"\n{'=' * 78}")
    print(f"  {model_key} — {task['name']}  ({len(runs)} run(s), n={runs[0]['n']})")
    print(f"{'=' * 78}")
    print(f"  Balanced accuracy : {_ms([r['balanced_acc'] for r in runs])}")
    print(f"  Macro F1 (%)      : {_ms([r['macro_f1'] for r in runs])}")
    excluded = sum(r["excluded"] for r in runs)
    if excluded:
        per_run = ", ".join(str(r["excluded"]) for r in runs)
        print(f"  Excluded (missing/unrecognized prediction): {excluded} total ({per_run} per run)")

    print(f"\n  Per-class recall (%):")
    for lb in labels:
        rec = [r["per_class"][lb]["recall"] * 100 for r in runs]
        print(f"    {names.get(lb, lb):<14} {_ms(rec)}")

    print(f"\n  Confusion matrix (rows = ground truth, cols = prediction), run 1:")
    cell = max(12, max(len(names.get(lb, lb)) for lb in labels) + 2)
    header = "    " + " " * 12 + "".join(f"{names.get(lb, lb):>{cell}}" for lb in labels)
    print(header)
    cm = runs[0]["cm"]
    for lb in labels:
        row = "".join(f"{cm[lb][c]:>{cell}}" for c in labels)
        print(f"    {names.get(lb, lb):<12}{row}")

    return {
        "model": model_key,
        "task": task_name,
        "balanced_acc": float(np.mean([r["balanced_acc"] for r in runs])),
        "macro_f1": float(np.mean([r["macro_f1"] for r in runs])),
        "excluded": int(sum(r["excluded"] for r in runs)),
    }


def report_binary(model_key, model_short, task_name, results_dir):
    task = TASKS[task_name]
    files = load_run_files(results_dir, task["file_prefix"], model_short)
    if not files:
        return None

    runs = evaluate_binary(files, task)
    if not runs:
        return None

    print(f"\n{'=' * 78}")
    print(f"  {model_key} — {task['name']} (binary: Good/Partial vs Poor/Non-response)")
    print(f"  {len(runs)} run(s), n={runs[0]['n']}")
    print(f"{'=' * 78}")
    print(f"  Balanced accuracy : {_ms([r['balanced_acc'] for r in runs])}")
    print(f"  Macro F1 (%)      : {_ms([r['macro_f1'] for r in runs])}")
    excluded = sum(r["excluded"] for r in runs)
    if excluded:
        per_run = ", ".join(str(r["excluded"]) for r in runs)
        print(f"  Excluded (missing/unrecognized prediction): {excluded} total ({per_run} per run)")

    return {
        "model": model_key,
        "task": task_name,
        "balanced_acc": float(np.mean([r["balanced_acc"] for r in runs])),
        "macro_f1": float(np.mean([r["macro_f1"] for r in runs])),
        "excluded": int(sum(r["excluded"] for r in runs)),
    }


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        print("Run `python run.py -m <MODEL> --task <TASK> --run <N>` first.")
        return

    print("=" * 78)
    print("  Reproduction of paper metrics")
    print(f"  results directory: {os.path.abspath(results_dir)}")
    print("=" * 78)

    four_class_summary = []
    binary_summary = []
    for model_key in MODEL_KEYS:
        try:
            model_short = get_model_config(model_key)[1].replace("/", "_").replace(":", "_")
        except Exception as exc:  # model key absent from config/config.json
            print(f"\nSkipping {model_key}: {exc}")
            continue

        for task_name in ("mnv", "mnv_cfp_only", "mnv_oct_only", *RESPONSE_TASKS):
            res = report_four_class(model_key, model_short, task_name, results_dir)
            if res:
                four_class_summary.append(res)

        for task_name in RESPONSE_TASKS:
            res = report_binary(model_key, model_short, task_name, results_dir)
            if res:
                binary_summary.append(res)

    had_output = bool(four_class_summary or binary_summary)

    print(f"\n\n{'=' * 78}")
    print("  Summary — four-class tasks (mean over runs)")
    print(f"{'=' * 78}")
    print(f"  {'Model':<8} {'Task':<24} {'BalAcc':>8} {'MacroF1':>9}")
    for r in four_class_summary:
        print(f"  {r['model']:<8} {r['task']:<24} "
              f"{r['balanced_acc']:>8.1f} {r['macro_f1']:>9.1f}")

    total_excluded = sum(r["excluded"] for r in four_class_summary)
    total_excluded_bin = sum(r["excluded"] for r in binary_summary)
    if total_excluded or total_excluded_bin:
        print(f"\n  Cases excluded for a missing or unrecognized prediction:")
        print(f"    four-class tasks : {total_excluded}")
        print(f"    binary task      : {total_excluded_bin}")

    print(f"\n\n{'=' * 78}")
    print("  Summary — binary response task (mean over runs)")
    print(f"{'=' * 78}")
    if binary_summary:
        print(f"  {'Model':<8} {'Task':<24} {'BalAcc':>8} {'MacroF1':>9}")
        for r in binary_summary:
            print(f"  {r['model']:<8} {r['task']:<24} "
                  f"{r['balanced_acc']:>8.1f} {r['macro_f1']:>9.1f}")
    else:
        print("  (no binary results found)")

    if not had_output:
        print("\nNo result files matched. Expected files such as:")
        print(f"  {results_dir}/predictions_detailed_<model>_tokens{MAX_TOKENS}_run1.json")


if __name__ == "__main__":
    main()
