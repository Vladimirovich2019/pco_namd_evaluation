import base64
import json
import os
import re
import requests
import numpy as np
from typing import Dict, Optional, List, Tuple


MAX_TOKENS = 16384
MAX_WORKERS = 32
DATA_DIR = "./data"
MNV_EXCEL = os.path.join(DATA_DIR, "PCO-nAMD_MNV.xlsx")
RESP_EXCEL = os.path.join(DATA_DIR, "PCO-nAMD_Response.xlsx")
MNV_IMG_DIR = os.path.join(DATA_DIR, "PCO-nAMD_MNV")
RESP_IMG_DIR = os.path.join(DATA_DIR, "PCO-nAMD_Response")
OUTPUT_DIR = "./results"

with open("config/config.json", "r") as f:
    MODELS = json.load(f)


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _image_mime(path: str) -> str:
    with open(path, 'rb') as f:
        magic = f.read(4)
    if magic[:4] == b'\x89PNG':
        return 'image/png'
    if magic[:2] == b'\xff\xd8':
        return 'image/jpeg'
    return 'image/jpeg'


def prepare_case_data(case_id: str, img_dir: str) -> Optional[Dict]:
    case_dir = os.path.join(img_dir, case_id)
    cfp_path = os.path.join(case_dir, "cfp.jpg")
    oct_path = os.path.join(case_dir, "oct.png")

    if not os.path.exists(cfp_path) or not os.path.exists(oct_path):
        raise ValueError(f"Images not found for case {case_id}")

    cfp_base64 = encode_image_to_base64(cfp_path)
    oct_base64 = encode_image_to_base64(oct_path)
    return {
        "cfp": (cfp_base64, _image_mime(cfp_path)),
        "oct": (oct_base64, _image_mime(oct_path)),
    }


def extract_from_response(response_text: str, tag: str) -> str:
    escaped_tag = re.escape(tag)

    if tag.startswith('<') and tag.endswith('>'):
        close_tag = tag[0] + '/' + tag[1:]
        escaped_close = re.escape(close_tag)
    elif tag.startswith('```'):
        escaped_close = re.escape('```')
    else:
        return ""

    pattern = re.compile(rf'{escaped_tag}(.*?){escaped_close}', re.DOTALL)
    matches = pattern.findall(response_text)

    return matches[-1].strip() if matches else ""


def get_model_config(model_key: str = "GPT"):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available keys: {list(MODELS.keys())}")

    config = MODELS[model_key]
    return config["base_url"], config["model_name"], config["key"]


def call_llm_api(case_id: str, content: List[Dict],
                 base_url: str, model_name: str, api_key: str,
                 max_tokens: int = MAX_TOKENS,
                 temperature: float = None) -> Tuple[str, Dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    try:
        response = requests.post(base_url, headers=headers, json=payload, timeout=1200)
        response.raise_for_status()
        result = response.json()
        full_response = result['choices'][0]['message']['content']

        if not full_response or not full_response.strip():
            return case_id, {"success": False, "error": "API returned empty response", "full_response": full_response}

        return case_id, {"success": True, "full_response": full_response, "raw_api_response": result}
    except Exception as e:
        return case_id, {"success": False, "error": str(e), "full_response": None}


def compute_confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}

    for true, pred in zip(y_true, y_pred):
        if true in cm and pred in cm[true]:
            cm[true][pred] += 1

    return cm


def compute_metrics_from_confusion(confusion_matrix):
    labels = list(confusion_matrix.keys())
    metrics = {}

    for label in labels:
        tp = confusion_matrix[label][label]
        fp = sum(confusion_matrix[other][label] for other in labels if other != label)
        fn = sum(confusion_matrix[label][other] for other in labels if other != label)

        support = sum(confusion_matrix[label].values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    return metrics


def compute_average_metrics(metrics, average='macro'):
    if average == 'macro':
        precision = np.mean([m['precision'] for m in metrics.values()])
        recall = np.mean([m['recall'] for m in metrics.values()])
        f1 = np.mean([m['f1'] for m in metrics.values()])
    elif average == 'weighted':
        total_support = sum(m['support'] for m in metrics.values())
        if total_support == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        precision = sum(m['precision'] * m['support'] for m in metrics.values()) / total_support
        recall = sum(m['recall'] * m['support'] for m in metrics.values()) / total_support
        f1 = sum(m['f1'] * m['support'] for m in metrics.values()) / total_support
    else:
        raise ValueError("average must be 'macro' or 'weighted'")

    return {'precision': precision, 'recall': recall, 'f1': f1}


def print_classification_report(metrics, average_metrics=None, class_names=None, accuracy=None):
    if class_names is None:
        class_names = {}

    if accuracy is not None:
        print(f"\n{'='*80}")
        print(f"Overall Metrics:")
        print(f"{'='*80}")
        print(f"Accuracy: {accuracy:.2f}%")
        if average_metrics:
            print(f"Macro Avg - Precision: {average_metrics.get('macro', {}).get('precision', 0):.4f}, "
                  f"Recall: {average_metrics.get('macro', {}).get('recall', 0):.4f}, "
                  f"F1: {average_metrics.get('macro', {}).get('f1', 0):.4f}")
            print(f"Weighted Avg - Precision: {average_metrics.get('weighted', {}).get('precision', 0):.4f}, "
                  f"Recall: {average_metrics.get('weighted', {}).get('recall', 0):.4f}, "
                  f"F1: {average_metrics.get('weighted', {}).get('f1', 0):.4f}")

    print(f"\n{'='*80}")
    print(f"Per-Class Metrics:")
    print(f"{'='*80}")

    for label, m in metrics.items():
        label_name = class_names.get(label, label)
        print(f"\n{label_name} (n={m['support']}):")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1-Score:  {m['f1']:.4f}")
        print(f"  TP={m['tp']}, FP={m['fp']}, FN={m['fn']}")

    if average_metrics and accuracy is None:
        print(f"\n{'='*80}")
        print(f"Average Metrics:")
        print(f"{'='*80}")
        print(f"Macro Avg - Precision: {average_metrics.get('macro', {}).get('precision', 0):.4f}, "
              f"Recall: {average_metrics.get('macro', {}).get('recall', 0):.4f}, "
              f"F1: {average_metrics.get('macro', {}).get('f1', 0):.4f}")
        print(f"Weighted Avg - Precision: {average_metrics.get('weighted', {}).get('precision', 0):.4f}, "
              f"Recall: {average_metrics.get('weighted', {}).get('recall', 0):.4f}, "
              f"F1: {average_metrics.get('weighted', {}).get('f1', 0):.4f}")


def print_confusion_matrix(confusion_matrix, class_names=None):
    if class_names is None:
        class_names = {}
    labels = list(confusion_matrix.keys())
    display_names = [class_names.get(l, l) for l in labels]
    col_w = max(len(n) for n in display_names) + 2

    print(f"\n{'='*80}")
    print("Confusion Matrix:")
    print('='*80)

    print(f"{'True \\ Pred':<{col_w + 2}}", end='')
    for n in display_names:
        print(f"{n:>{col_w}}", end='')
    print()

    print("-" * (col_w + 2 + col_w * len(labels)))

    for true_label, true_name in zip(labels, display_names):
        print(f"{true_name:<{col_w + 2}}", end='')
        for pred_label in labels:
            print(f"{confusion_matrix[true_label][pred_label]:>{col_w}}", end='')
        print()

    print('='*80)
