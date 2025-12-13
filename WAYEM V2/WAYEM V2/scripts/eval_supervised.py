# scripts/eval_supervised.py
from __future__ import annotations
import json
import numpy as np
from tqdm import tqdm
from backend.ml.supervised import SupervisedMoodClassifier

def evaluate_dataset_supervised(path: str, classifier, k: int = 3):
    """
    Evaluate SupervisedMoodClassifier on a JSONL dataset.
    Each line: {"text": "...", "labels": ["Mood1", "Mood2", ...]}
    """
    canonicals = classifier._loaded["canonicals"] if classifier._loaded else classifier.zero._ont.canonicals
    C = len(canonicals)
    label_to_idx = {c: i for i, c in enumerate(canonicals)}

    texts, y_true = [], []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            lbls = row.get("labels") or []
            vec = np.zeros(C, dtype=int)
            for l in lbls:
                if l in label_to_idx:
                    vec[label_to_idx[l]] = 1
            texts.append(row.get("text", ""))
            y_true.append(vec)

    y_true = np.stack(y_true, axis=0)
    y_score = np.zeros_like(y_true, dtype=float)

    for i, text in enumerate(tqdm(texts, desc=f"Evaluating {path}")):
        canon, probs = classifier.predict_full_vector(text)
        for j, lbl in enumerate(canon):
            y_score[i, j] = probs[j]

    # threshold-based evaluation
    thr = 0.35
    y_pred = (y_score >= thr).astype(int)
    from sklearn.metrics import f1_score
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # coverage@3
    idx = np.argsort(-y_score, axis=1)[:, :3]
    hits = [(y_true[i, topk] == 1).any() for i, topk in enumerate(idx)]
    cov3 = float(np.mean(hits))

    return {
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "coverage@3": round(cov3, 4),
        "n_samples": len(texts),
        "n_classes": C,
    }

if __name__ == "__main__":
    clf = SupervisedMoodClassifier()
    results = evaluate_dataset_supervised("data/test.jsonl", clf, k=3)
    print("\nResults on supervised model:")
    for k, v in results.items():
        print(f"{k}: {v}")
