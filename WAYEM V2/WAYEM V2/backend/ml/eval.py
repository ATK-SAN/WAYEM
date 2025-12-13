# backend/ml/eval.py
from __future__ import annotations
import json
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import os

from .classify import MoodClassifier


# ---------------------------
# Metrics
# ---------------------------

def eval_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.35):
    """
    Compute micro/macro F1 given binary relevance at threshold.
    Expects y_score to be the full per-class probabilities (not truncated to top-k).
    """
    y_pred = (y_score >= thr).astype(int)
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def coverage_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3):
    """
    Fraction of samples for which at least one true label appears in top-k predictions.
    Uses the same y_score matrix; we only inspect the top-k indices per row.
    """
    idx = np.argsort(-y_score, axis=1)[:, :k]
    hits = [(y_true[i, topk] == 1).any() for i, topk in enumerate(idx)]
    return float(np.mean(hits))


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray):
    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(0.1, 0.9, 33):
        f1 = eval_at_threshold(y_true, y_score, thr)["macro_f1"]
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


# ---------------------------
# Dataset loading (with diagnostics)
# ---------------------------

def _load_dataset_jsonl(path: str, canonicals: List[str]) -> Tuple[List[str], np.ndarray, int]:
    """
    Load a JSONL dataset. Each line: {"text": "...", "labels": ["Mood1", "Mood2", ...]}
    Returns (texts, y_true_matrix, unknown_label_count).
    Raises ValueError with a clear message if no valid rows are found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    C = len(canonicals)
    label_to_idx = {c: i for i, c in enumerate(canonicals)}

    texts: List[str] = []
    y_true_rows: List[np.ndarray] = []
    unknown_label_count = 0

    # IMPORTANT: utf-8-sig to strip a BOM if present (Windows PowerShell often writes one)
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # skip malformed lines
                continue

            text = row.get("text", "")
            labels = row.get("labels") or []
            if not isinstance(labels, list):
                # skip if labels isn't a list
                continue

            vec = np.zeros(C, dtype=int)
            for l in labels:
                if l in label_to_idx:
                    vec[label_to_idx[l]] = 1
                else:
                    unknown_label_count += 1
            texts.append(text)
            y_true_rows.append(vec)

    if len(texts) == 0:
        raise ValueError(
            "No valid samples loaded from dataset. "
            "Check that the file is non-empty, each line is valid JSON, "
            "and includes a 'labels' list with ontology labels."
        )

    y_true = np.stack(y_true_rows, axis=0)
    return texts, y_true, unknown_label_count


# ---------------------------
# Evaluation Loop
# ---------------------------

def evaluate_dataset(
    path: str,
    classifier: MoodClassifier,
    k: int = 3,
    threshold: float | None = None,
) -> Dict[str, float]:
    """
    Evaluate the classifier on a JSONL dataset.
    Each line: {"text": "...", "labels": ["Mood1", "Mood2", ...]}
    - Threshold metrics computed on the FULL per-class probability vector (no top-k truncation).
    - coverage@k still computed with top-k over the same full vector.
    """
    # Ensure ontology exists
    classifier._maybe_reload()
    ont = classifier._ont
    assert ont is not None

    canonicals = ont.canonicals
    C = len(canonicals)

    # Load dataset with diagnostics
    texts, y_true, unknown_label_count = _load_dataset_jsonl(path, canonicals)

    if unknown_label_count > 0:
        print(f"[warn] {unknown_label_count} label(s) in the dataset were not in ontology and were ignored.")

    y_score = np.zeros((len(texts), C), dtype=float)

    # Full-distribution scoring per sample
    for i, text in enumerate(tqdm(texts, desc=f"Evaluating {path}")):
        labels, probs = classifier.predict_full_vector(text)  # labels aligned with canonicals
        if not labels:
            continue
        y_score[i, :] = probs

    # Threshold tuning on full probabilities
    thr = threshold or tune_threshold(y_true, y_score)

    f1s = eval_at_threshold(y_true, y_score, thr)
    covk = coverage_at_k(y_true, y_score, k=k)
    return {
        "thr": round(thr, 2),
        "micro_f1": round(f1s["micro_f1"], 4),
        "macro_f1": round(f1s["macro_f1"], 4),
        f"coverage@{k}": round(covk, 4),
        "n_samples": int(y_true.shape[0]),
        "n_classes": int(C),
    }


# ---------------------------
# CLI entry point
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate baseline mood classifier.")
    parser.add_argument("--data", type=str, default="data/test.jsonl")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    clf = MoodClassifier()
    results = evaluate_dataset(args.data, clf, k=args.k)
    print(f"\nResults on {args.data}:")
    for k, v in results.items():
        print(f"{k}: {v}")
