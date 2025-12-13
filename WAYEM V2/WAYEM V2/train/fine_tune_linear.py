# train/fine_tune_linear.py
from __future__ import annotations
import json, os, numpy as np
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from backend.ml.embedder import embed
from backend.ml.classify import MoodClassifier

def load_jsonl(path: str) -> List[Dict]:
    rows=[]
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s=line.strip()
            if s: rows.append(json.loads(s))
    return rows

def make_y(rows: List[Dict], canon: List[str]) -> np.ndarray:
    idx = {c:i for i,c in enumerate(canon)}
    Y = np.zeros((len(rows), len(canon)), dtype=int)
    for i,r in enumerate(rows):
        for l in r.get("labels", []):
            if l in idx: Y[i, idx[l]] = 1
    return Y

def main(train_path="data/train.jsonl"):
    base = MoodClassifier()              # for ontology order
    canon = base._ont.canonicals

    rows = load_jsonl(train_path)
    if not rows:
        raise SystemExit(f"No rows in {train_path}. Generate data first.")

    X_text = [r["text"] for r in rows]
    X = embed(X_text)                    # (N, D)
    Y = make_y(rows, canon)              # (N, C)

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=500, class_weight="balanced")
        ))
    ])
    model.fit(X, Y)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"pipeline": model, "canonicals": canon}, "models/linear_head.joblib")
    print(f"Saved models/linear_head.joblib  (N={len(rows)}, C={len(canon)})")

if __name__ == "__main__":
    main()
