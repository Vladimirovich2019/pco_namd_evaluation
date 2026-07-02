#!/usr/bin/env python3
"""Reproduce binary treatment response metrics from raw JSON result files."""
import json, glob, numpy as np
from sklearn.metrics import confusion_matrix

TASKS = {
    "Images Only": "predictions_response_direct",
    "w/o Images": "predictions_givenBiomarker_noimg",
    "Combination": "predictions_givenBiomarker",
}
MODELS = {
    "gpt-5": "GPT-5",
    "gemini-3.1-pro-preview-thinking": "Gemini",
    "qwen-vl-max": "Qwen",
}

for mk, ml in MODELS.items():
    for tl, prefix in TASKS.items():
        files = sorted(glob.glob(f"results/{prefix}_detailed_{mk}_tokens16384_run[0-9].json"))
        if not files: continue
        accs, senss, specs, f1s = [], [], [], []
        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            yt, yp = [], []
            for d in data:
                gt = d.get("ground_truth")
                pred = d.get("extracted_prediction")
                if not gt or not pred:
                    continue
                gt_bin = 1 if gt in ("1", "2") else 0
                p = pred.lower()
                pred_bin = 1 if "good" in p or "partial" in p else (0 if "poor" in p or "non" in p else None)
                if pred_bin is None:
                    continue
                yt.append(gt_bin)
                yp.append(pred_bin)
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            N = len(yt)
            accs.append((tp + tn) / N * 100)
            sn = tp / (tp + fn) * 100 if tp + fn else 0
            sp = tn / (tn + fp) * 100 if tn + fp else 0
            pr = tp / (tp + fp) * 100 if tp + fp else 0
            senss.append(sn)
            specs.append(sp)
            f1s.append(2 * pr * sn / (pr + sn) if pr + sn else 0)
        a, s, p_arr, f = np.array(accs), np.array(senss), np.array(specs), np.array(f1s)
        print(f"{ml:<8} {tl:<18} "
              f"Acc={a.mean():.1f}±{a.std(ddof=1):.1f}  "
              f"Sens={s.mean():.1f}±{s.std(ddof=1):.1f}  "
              f"Spec={p_arr.mean():.1f}±{p_arr.std(ddof=1):.1f}  "
              f"F1={f.mean():.1f}±{f.std(ddof=1):.1f}")
