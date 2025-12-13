# backend/ml/supervised.py
from __future__ import annotations
import os, time
from pathlib import Path
from typing import List, Dict, Tuple

import joblib
import numpy as np

from .embedder import embed
from .classify import MoodClassifier, _normalize_scores


class SupervisedMoodClassifier:
    """
    Wraps a linear head saved at models/linear_head.joblib.
    - Hot-reloads the joblib when the file changes.
    - Falls back to zero-shot MoodClassifier if the head isn't available.
    - Uses absolute ontology paths so current working directory doesn't matter.
    """

    def __init__(self, model_path: str = "models/linear_head.joblib", reload_interval: float = 5.0):
        self.model_path = model_path
        self.reload_interval = float(reload_interval)
        self._loaded = None
        self._mtime = -1.0
        self._last_check = 0.0

        # ----- make zero-shot fallback CWD-agnostic -----
        # repo root = backend/ml/ -> parents[2]
        root = Path(__file__).resolve().parents[2]
        moods_path = str(root / "ontology" / "moods.yml")
        aliases_path = str(root / "ontology" / "aliases.csv")
        self.zero = MoodClassifier(moods_path=moods_path, aliases_path=aliases_path)
        # -------------------------------------------------

        self._maybe_reload(force=True)

    def _maybe_reload(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_check) < self.reload_interval:
            return
        self._last_check = now
        try:
            m = os.path.getmtime(self.model_path)
        except OSError:
            self._loaded = None
            self._mtime = -1.0
            return
        if m != self._mtime:
            self._loaded = joblib.load(self.model_path)
            self._mtime = m

    def predict(self, text: str, k: int = 3):
        self._maybe_reload()
        text = (text or "").strip()
        if not text:
            return []
        if self._loaded is None:
            return self.zero.predict(text, k=k)

        canon = self._loaded["canonicals"]
        pipe = self._loaded["pipeline"]
        X = embed([text])
        logits = pipe.decision_function(X)
        if logits.ndim == 1:
            logits = logits[None, :]
        probs = _normalize_scores(logits[0].astype(np.float32))

        k = max(1, min(int(k), 8))
        idx = np.argsort(-probs)[:k]
        out = [{"label": canon[i], "score": float(probs[i])} for i in idx]
        s = sum(p["score"] for p in out) or 1.0
        for p in out:
            p["score"] /= s
        return out

    def predict_full_vector(self, text: str) -> Tuple[List[str], np.ndarray]:
        self._maybe_reload()
        text = (text or "").strip()
        if self._loaded is None:
            # fallback to zero-shot vector
            # (use MoodClassifier’s canonicals and a uniform vector if text is empty)
            canon = self.zero._ont.canonicals  # type: ignore[attr-defined]
            if not text:
                C = len(canon)
                return canon, np.ones((C,), dtype=np.float32) / float(C)
            X = embed([text])
            sims = (self.zero._ont.E_canon @ X[0].astype(np.float32))  # type: ignore[attr-defined]
            probs = _normalize_scores(sims.astype(np.float32))
            return canon, probs

        canon = self._loaded["canonicals"]
        if not text:
            C = len(canon)
            return canon, np.ones((C,), dtype=np.float32) / float(C)

        pipe = self._loaded["pipeline"]
        X = embed([text])
        logits = pipe.decision_function(X)
        if logits.ndim == 1:
            logits = logits[None, :]
        probs = _normalize_scores(logits[0].astype(np.float32))
        return canon, probs
