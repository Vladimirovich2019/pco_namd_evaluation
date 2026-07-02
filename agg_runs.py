#!/usr/bin/env python3
"""Aggregate metrics from multiple runs and print comparison table."""
import json, sys, os
import numpy as np
from run import TASKS
from utils import compute_confusion_matrix, compute_metrics_from_confusion, compute_average_metrics, get_model_config

runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
model_key = sys.argv[2] if len(sys.argv) > 2 else "GPT"
tasks = sys.argv[3:] if len(sys.argv) > 3 else ["mnv", "mnv_cfp_only", "mnv_oct_only", "response_direct", "given_biomarker_noimg", "given_biomarker"]
model_name = get_model_config(model_key)[1]
model_short = model_name.replace('/', '_').replace(':', '_')
disp_names = {
    "mnv": "MNV (CFP+OCT)",
    "mnv_cfp_only": "MNV (CFP)",
    "mnv_oct_only": "MNV (OCT)",
    "response_direct": "Response (Images)",
    "given_biomarker_noimg": "Response (w/o Images)",
    "given_biomarker": "Response (Combination)",
}

def _vis_len(s):
    """Display width: CJK chars count as 2, ASCII as 1."""
    return sum(2 if ord(c) > 0x2e80 else 1 for c in s)

def _pad(s, w, align='L'):
    """Pad string s to display width w. align: L=left, R=right."""
    n = _vis_len(s)
    space = ' ' * max(0, w - n)
    return s + space if align == 'L' else space + s

COL_W = 20

summary = {}

for task in tasks:
    cfg = TASKS[task]
    prefix = cfg["file_prefix"]
    label_fn = cfg["map_label"]
    labels = cfg["eval_labels"]
    names = cfg["label_names"]
    base = f"results/{prefix}_detailed_{model_short}_tokens16384"

    all_metrics = []
    for r in range(1, runs + 1):
        path = f"{base}_run{r}.json"
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        y_true, y_pred = [], []
        for d in data:
            gt = label_fn(d.get("ground_truth"))
            pred = label_fn(d.get("extracted_prediction", ""))
            if gt is None:
                continue
            y_true.append(gt)
            y_pred.append(pred)

        cm = compute_confusion_matrix(y_true, y_pred, labels=labels)
        metrics = compute_metrics_from_confusion(cm)
        for lb in metrics:
            metrics[lb]['support'] = sum(1 for t in y_true if t == lb)

        weighted = compute_average_metrics(metrics, 'weighted')
        macro = compute_average_metrics(metrics, 'macro')
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) * 100
        all_metrics.append({
            "acc": acc, "w_prec": weighted['precision'], "w_rec": weighted['recall'],
            "w_f1": weighted['f1'], "m_f1": macro['f1'], "per_class": metrics,
        })

    summary[task] = all_metrics


# ── Overall Comparison Table ──
rows = [
    ("Accuracy", "acc", "{:.2f}"),
    ("Weighted Precision", "w_prec", "{:.2f}"),
    ("Weighted Recall", "w_rec", "{:.2f}"),
    ("Weighted F1", "w_f1", "{:.2f}"),
    ("Macro F1", "m_f1", "{:.2f}"),
]

print(f"\n{'=' * 70}")
print(f"  Comparison — {runs} runs each  (mean ± std)")
print(f"{'=' * 70}")
header = f"  {_pad('Metric', COL_W, 'L')}"
for t in tasks:
    header += f"  {_pad(disp_names.get(t, t), COL_W, 'R')}"
print(header)
print(f"  {'-' * (22 + 22 * len(tasks))}")
for label, key, fmt in rows:
    line = f"  {_pad(label, COL_W, 'L')}"
    for t in tasks:
        vals = [m[key] for m in summary[t]]
        mu = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0
        pair = f"{fmt.format(mu)} ± {fmt.format(sd)}"
        line += f"  {_pad(pair, COL_W, 'R')}"
    print(line)

# ── Per-class table ──
for t in tasks:
    cfg = TASKS[t]
    names = cfg["label_names"]
    labels = cfg["eval_labels"]
    print(f"\n  ─── {disp_names.get(t, t)} (per-class) ───")
    print(f"  {'Class':<15}  {'Prec':>15}  {'Rec':>15}  {'F1':>15}")
    for lb in labels:
        precs = [m["per_class"][lb]["precision"] * 100 for m in summary[t]]
        recs  = [m["per_class"][lb]["recall"] * 100 for m in summary[t]]
        f1s   = [m["per_class"][lb]["f1"] * 100 for m in summary[t]]
        supp  = summary[t][0]["per_class"][lb]["support"]
        def _fmt(vals):
            mu, sd = np.mean(vals), np.std(vals, ddof=1)
            return f"{mu:.1f}±{sd:.1f}" if len(vals) > 1 else f"{mu:.1f}"
        print(f"  {names.get(lb, lb):<15}  {_fmt(precs):>15}  {_fmt(recs):>15}  {_fmt(f1s):>15}")
