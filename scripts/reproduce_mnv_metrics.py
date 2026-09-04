#!/usr/bin/env python3
"""Reproduce MNV classification metrics from raw JSON result files."""
import json, glob, numpy as np
from sklearn.metrics import precision_score, f1_score
from collections import Counter

def map_mnv(x):
    if not x:
        return None
    m = {'Type 1 MNV (Occult)':'1','Type 2 MNV (Classic)':'2','Type 3 MNV':'3','Mix MNV':'Mix'}
    for k, v in m.items():
        if k in str(x):
            return v
    return None

LABELS = ['1','2','3','Mix']
NAMES = {'1':'Type 1','2':'Type 2','3':'Type 3','Mix':'Mix'}

EXPERIMENTS = [
    ("GPT-5  CFP Only",     "predictions_mnv_cfp", "gpt-5"),
    ("GPT-5  OCT Only",     "predictions_mnv_oct", "gpt-5"),
    ("GPT-5  CFP+OCT",      "predictions",         "gpt-5"),
    ("Gemini 3.1 Pro  CFP Only", "predictions_mnv_cfp", "gemini-3.1-pro-preview-thinking"),
    ("Gemini 3.1 Pro  OCT Only", "predictions_mnv_oct", "gemini-3.1-pro-preview-thinking"),
    ("Gemini 3.1 Pro  CFP+OCT",  "predictions",         "gemini-3.1-pro-preview-thinking"),
    ("Qwen VL Max  CFP Only",    "predictions_mnv_cfp", "qwen-vl-max"),
    ("Qwen VL Max  OCT Only",    "predictions_mnv_oct", "qwen-vl-max"),
    ("Qwen VL Max  CFP+OCT",     "predictions",         "qwen-vl-max"),
]

for label, prefix, mk in EXPERIMENTS:
    files = sorted(glob.glob(f"results/{prefix}_detailed_{mk}_tokens16384_run[0-9].json"))
    if not files:
        continue
    accs, wps, wfs = [], [], []
    per_cls = {lb:{'prec':[],'rec':[]} for lb in LABELS}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        yt, yp = [], []
        for d in data:
            gt = d.get("ground_truth")
            pred = map_mnv(d.get("extracted_prediction"))
            if gt is None or pred is None:
                continue
            yt.append(gt)
            yp.append(pred)
        accs.append(sum(1 for a,b in zip(yt,yp) if a==b)/len(yt)*100)
        wps.append(precision_score(yt,yp,labels=LABELS,average='weighted',zero_division=0)*100)
        wfs.append(f1_score(yt,yp,labels=LABELS,average='weighted',zero_division=0)*100)
        for lb in LABELS:
            tp = sum(1 for a,b in zip(yt,yp) if a==lb and b==lb)
            gt_t = sum(1 for a in yt if a==lb)
            pd_t = sum(1 for b in yp if b==lb)
            per_cls[lb]['prec'].append(tp/pd_t*100 if pd_t else 0)
            per_cls[lb]['rec'].append(tp/gt_t*100 if gt_t else 0)
    a = np.array(accs)
    wp = np.mean(wps)
    wf = np.mean(wfs)
    print(f"\n{label}: Acc={a.mean():.1f}±{a.std(ddof=1):.1f}%  W-Prec={wp:.1f}  W-F1={wf:.1f}")
    print(f"  Individual: {[f'{x:.1f}' for x in accs]}")
    for lb in LABELS:
        pr = np.mean(per_cls[lb]['prec'])
        rc = np.mean(per_cls[lb]['rec'])
        print(f"  {NAMES[lb]:<8}: Prec={pr:.1f}%  Rec={rc:.1f}%")
