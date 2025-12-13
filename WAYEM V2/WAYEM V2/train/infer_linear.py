# train/infer_linear.py
from __future__ import annotations
import sys, json
from backend.ml.supervised import SupervisedMoodClassifier

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m train.infer_linear \"your text here\" [k]")
        sys.exit(1)
    text = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    clf = SupervisedMoodClassifier()
    preds = clf.predict(text, k=k)
    print(json.dumps(preds, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
