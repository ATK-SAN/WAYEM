from __future__ import annotations
import json, numpy as np
from sklearn.metrics import precision_recall_fscore_support
from backend.ml.classify import MoodClassifier
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_jsonl(path: str):
    rows=[]
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def main(path="data/test.jsonl", k=3):
    clf = MoodClassifier()
    canon = clf._ont.canonicals
    idx = {c:i for i,c in enumerate(canon)}
    y_true = []
    y_pred_top1 = []
    for r in load_jsonl(path):
        y = np.zeros(len(canon), dtype=int)
        for l in r.get("labels", []):
            if l in idx: y[idx[l]] = 1
        y_true.append(y)
        top = clf.predict(r["text"], k=k)
        yhat = np.zeros(len(canon), dtype=int)
        if top:
            yhat[idx[top[0]["label"]]] = 1
        y_pred_top1.append(yhat)
    Y = np.stack(y_true)
    P = np.stack(y_pred_top1)
    prec, rec, f1, support = precision_recall_fscore_support(
        Y, P, average=None, zero_division=0
    )
    print("label, support, prec_top1, rec_top1, f1_top1")
    for i,c in enumerate(canon):
        print(f"{c},{support[i]},{prec[i]:.3f},{rec[i]:.3f},{f1[i]:.3f}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv)>1 else "data/test.jsonl"
    main(path)

