# backend/ml/classify.py
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from .embedder import embed

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # repo root
DEFAULT_MOODS = str(ROOT / "ontology" / "moods.yml")
DEFAULT_ALIASES = str(ROOT / "ontology" / "aliases.csv")


# ---------------------------
# Utilities
# ---------------------------

def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Softmax → (0,1), sums to 1, numerically stable."""
    if scores.size == 0:
        return scores
    x = scores.astype(np.float32, copy=False)
    x = x - float(x.max())  # stabilize
    e = np.exp(x)
    s = float(e.sum())
    if s <= 1e-12:
        # fallback to uniform if everything underflows
        return np.ones_like(e, dtype=np.float32) / float(e.size)
    return (e / s).astype(np.float32)


def _cosine_matrix_vector(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosine similarity for normalized embeddings = dot product."""
    if A.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (A @ v).astype(np.float32)


@dataclass
class _Ontology:
    canonicals: List[str]
    alias_to_canonical: Dict[str, str]
    E_canon: np.ndarray  # (C, d)
    E_alias_keys: np.ndarray  # (A, d)
    alias_keys: List[str]
    mtime_moods: float
    mtime_aliases: float
    mtime_prompts: float  # track mood_prompts.yml for hot-reload


# ---------------------------
# Classifier
# ---------------------------

class MoodClassifier:
    """
    Embedding-based, multi-label mood classifier with alias boosting.
    - predict(text, k): returns top-k [{"label": <canonical>, "score": <0..1>}], scores sum to 1 across returned items.
    - predict_full_vector(text): returns full-length probability vector aligned with canonicals (sum to 1 over all labels).
    """

    def __init__(
        self,
        moods_path: str = "ontology/moods.yml",
        aliases_path: str = "ontology/aliases.csv",
        alias_exact_boost: float = 0.9,
        alias_fuzzy_boost: float = 0.75,
        alias_fuzzy_threshold: float = 0.5,    # tightened to reduce spurious pulls
        alias_contains_min_len: int = 3,       # min alias length for substring matching
        alias_contains_boost: float = 0.75,    # boost for substring alias hits
        min_keep_score: float = 0.05,          # drop labels under this score (for top-k API)
    ):
        self.moods_path = moods_path
        self.aliases_path = aliases_path
        self.alias_exact_boost = float(np.clip(alias_exact_boost, 0.0, 1.0))
        self.alias_fuzzy_boost = float(np.clip(alias_fuzzy_boost, 0.0, 1.0))
        self.alias_fuzzy_threshold = float(np.clip(alias_fuzzy_threshold, 0.0, 1.0))
        self.alias_contains_min_len = int(alias_contains_min_len)
        self.alias_contains_boost = float(np.clip(alias_contains_boost, 0.0, 1.0))
        self.min_keep_score = float(np.clip(min_keep_score, 0.0, 1.0))

        self._ont: Optional[_Ontology] = None
        self._load_ontology()

    # ---------- public API ----------

    def predict(self, text: str, k: int = 3) -> List[Dict[str, float | str]]:
        """
        Multi-label prediction with alias (exact / substring / fuzzy) boosting.
        Returns top-k [{"label": <canonical>, "score": <0..1>}]. Scores sum to 1 over the returned set.
        """
        text = (text or "").strip().lower()
        k = max(1, min(int(k), 8))
        if not text:
            return []

        # Ensure ontology is fresh
        self._maybe_reload()
        ont = self._ont
        assert ont is not None

        labels_all, scores_all = self._score_all_labels(text, ont)

        # filter tiny scores, then take top-k and renormalize
        keep_idx = [i for i, s in enumerate(scores_all) if s >= self.min_keep_score]
        if not keep_idx:
            keep_idx = [int(np.argmax(scores_all))]

        labels = [labels_all[i] for i in keep_idx]
        scores = scores_all[keep_idx]

        if len(labels) > k:
            idx_top = np.argsort(-scores)[:k]
            labels = [labels[i] for i in idx_top]
            scores = scores[idx_top]

        scores = _normalize_scores(scores)
        return [{"label": lbl, "score": float(score)} for lbl, score in zip(labels, scores)]

    def predict_full_vector(self, text: str) -> Tuple[List[str], np.ndarray]:
        """
        Full probability distribution over all canonicals, aligned with self._ont.canonicals.
        Returns (labels, probs) with probs shape (C,), sum(probs) == 1.
        """
        text = (text or "").strip().lower()
        if not text:
            # return uniform over canonicals if empty text
            self._maybe_reload()
            ont = self._ont
            assert ont is not None
            C = len(ont.canonicals)
            if C == 0:
                return [], np.zeros((0,), dtype=np.float32)
            return ont.canonicals, np.ones((C,), dtype=np.float32) / float(C)

        self._maybe_reload()
        ont = self._ont
        assert ont is not None

        labels_all, scores_all = self._score_all_labels(text, ont)
        return labels_all, scores_all

    # ---------- core scoring logic (internal) ----------

    def _score_all_labels(self, text: str, ont: _Ontology) -> Tuple[List[str], np.ndarray]:
        """
        Compute *full* probability distribution for all canonicals after applying alias boosts.
        This is the single source of truth used by both predict() and predict_full_vector().
        """
        # Embed query (already L2-normalized by embedder)
        q_vec = embed([text])[0]

        # 1) Canonical similarity (raw logits = cosine sims)
        canon_sims = _cosine_matrix_vector(ont.E_canon, q_vec)  # (C,)
        labels_all = list(ont.canonicals)
        scores = canon_sims.astype(np.float32).copy()  # raw logits pre-softmax

        # 2) Alias logic
        # Primary alias gets a larger delta; secondary substring matches get smaller deltas.
        primary_label: Optional[str] = None
        primary_boost_kind: Optional[str] = None  # "exact" | "contains" | "fuzzy"

        # (a) exact alias (whole query)
        if text in ont.alias_to_canonical:
            primary_label = ont.alias_to_canonical[text]
            primary_boost_kind = "exact"

        # (b) substring alias — collect ALL matches; prefer longer aliases and stronger canon sim.
        substring_hits: List[Tuple[str, str, int, float]] = []
        if ont.alias_keys:
            for a in ont.alias_keys:
                if len(a) < self.alias_contains_min_len:
                    continue
                # robust token-ish boundaries; avoids \b pitfalls on unicode
                if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text):
                    lbl = ont.alias_to_canonical.get(a)
                    if lbl:
                        try:
                            j = ont.canonicals.index(lbl)
                            sim = float(canon_sims[j])
                        except ValueError:
                            sim = -1e9
                        substring_hits.append((a, lbl, len(a), sim))

        substring_hits.sort(key=lambda t: (t[2], t[3]), reverse=True)  # by length then sim

        if primary_label is None and substring_hits:
            primary_label = substring_hits[0][1]
            primary_boost_kind = "contains"

        # (c) fuzzy alias (embedding)
        if primary_label is None and ont.E_alias_keys.size:
            alias_sims = _cosine_matrix_vector(ont.E_alias_keys, q_vec)
            a_best = int(np.argmax(alias_sims))
            a_score = float(alias_sims[a_best])
            if a_score >= self.alias_fuzzy_threshold:
                a_key = ont.alias_keys[a_best]
                candidate = ont.alias_to_canonical.get(a_key)
                if candidate:
                    primary_label = candidate
                    primary_boost_kind = "fuzzy"

        # 3) Additive boosts BEFORE softmax (operate in logit space)
        # Scale deltas by the configured "boost" knobs so they actually mean something.
        EXACT_DELTA = 1.2 * self.alias_exact_boost        # default 0.9  -> 1.08
        CONTAINS_DELTA = 1.0 * self.alias_contains_boost  # default 0.75 -> 0.75
        FUZZY_DELTA = 0.9 * self.alias_fuzzy_boost        # default 0.75 -> 0.675

        def _add_delta(lbl: str, delta: float) -> None:
            nonlocal scores
            try:
                j = labels_all.index(lbl)
            except ValueError:
                return
            scores[j] = scores[j] + float(delta)

        # Primary boost
        if primary_label:
            if primary_boost_kind == "exact":
                _add_delta(primary_label, EXACT_DELTA)
            elif primary_boost_kind == "contains":
                _add_delta(primary_label, CONTAINS_DELTA)
            else:
                _add_delta(primary_label, FUZZY_DELTA)

        # Secondary boosts for additional substring hits (beyond primary)
        if substring_hits:
            primary_taken = primary_label
            for idx, (a, lbl, alen, _sim) in enumerate(substring_hits):
                if lbl == primary_taken:
                    continue
                # modest, length-aware delta; longer aliases get a hair more
                delta = 0.35 + min(0.25, alen * 0.02)
                _add_delta(lbl, delta)

        # 4) Softmax over *all* canonicals → distribution
        scores = _normalize_scores(scores)
        return labels_all, scores

    # ---------- optional helper ----------

    @staticmethod
    def blend_targets(
        mood_scores: List[Dict[str, float | str]],
        mood_to_features: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Weighted average of feature targets based on predicted mood scores."""
        agg: Dict[str, float] = {}
        for item in mood_scores:
            lbl = str(item.get("label", ""))
            w = float(item.get("score", 0.0))
            if w <= 0.0 or not lbl:
                continue
            feats = mood_to_features.get(lbl, {})
            for k, v in feats.items():
                agg[k] = agg.get(k, 0.0) + (w * float(v))
        return {k: round(v, 3) for k, v in agg.items()}

    # ---------- internal: ontology loading & caching ----------

    def _load_ontology(self) -> None:
        moods_mtime = _safe_mtime(self.moods_path)
        aliases_mtime = _safe_mtime(self.aliases_path)

        canonicals = _load_canonicals(self.moods_path)
        alias_to_canonical = _load_aliases(
            self.aliases_path, canonicals_set=set(canonicals)
        )

        # prompts (optional)
        prompts_path = os.path.join(os.path.dirname(self.moods_path), "mood_prompts.yml")
        prompts_mtime = _safe_mtime(prompts_path)
        prompts: Dict[str, str] = {}
        if os.path.exists(prompts_path):
            try:
                with open(prompts_path, "r", encoding="utf-8") as f:
                    prompts = yaml.safe_load(f) or {}
            except Exception:
                prompts = {}

        canon_texts = [(prompts.get(c) or c) for c in canonicals]
        E_canon = embed(canon_texts).astype(np.float32)

        alias_keys = list(alias_to_canonical.keys())
        if alias_keys:
            E_alias_keys = embed(alias_keys).astype(np.float32)
        else:
            E_alias_keys = np.zeros((0, E_canon.shape[1]), dtype=np.float32)

        self._ont = _Ontology(
            canonicals=canonicals,
            alias_to_canonical=alias_to_canonical,
            E_canon=E_canon,
            E_alias_keys=E_alias_keys,
            alias_keys=alias_keys,
            mtime_moods=moods_mtime,
            mtime_aliases=aliases_mtime,
            mtime_prompts=prompts_mtime,
        )

    def _maybe_reload(self) -> None:
        assert self._ont is not None
        m1 = _safe_mtime(self.moods_path)
        m2 = _safe_mtime(self.aliases_path)
        m3 = _safe_mtime(os.path.join(os.path.dirname(self.moods_path), "mood_prompts.yml"))
        if (
            abs(m1 - self._ont.mtime_moods) > 1e-9
            or abs(m2 - self._ont.mtime_aliases) > 1e-9
            or abs(m3 - self._ont.mtime_prompts) > 1e-9
        ):
            self._load_ontology()


# ---------------------------
# File loaders
# ---------------------------

def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


def _load_canonicals(moods_path: str) -> List[str]:
    if not os.path.exists(moods_path):
        raise FileNotFoundError(f"Missing moods file: {moods_path}")
    with open(moods_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    canonicals = data.get("canonicals") or []
    if not isinstance(canonicals, list) or not all(isinstance(x, str) for x in canonicals):
        raise ValueError("moods.yml must contain a 'canonicals' list of strings.")
    seen, cleaned = set(), []
    for c in canonicals:
        c = c.strip()
        if c and c not in seen:
            cleaned.append(c)
            seen.add(c)
    if not cleaned:
        raise ValueError("moods.yml canonicals list is empty after cleaning.")
    return cleaned


def _load_aliases(aliases_path: str, canonicals_set: set[str]) -> Dict[str, str]:
    if not os.path.exists(aliases_path):
        raise FileNotFoundError(f"Missing aliases file: {aliases_path}")
    alias_to_canonical: Dict[str, str] = {}
    with open(aliases_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and len(header) >= 2 and header[0].lower() == "alias":
            rows = list(reader)  # header present
        else:
            rows = ([header] if header else []) + list(reader)  # treat first row as data

        for row in rows:
            if not row or len(row) < 2:
                continue
            alias_raw, canonical_raw = row[0], row[1]
            alias = (alias_raw or "").strip().lower()
            canonical = (canonical_raw or "").strip()
            if not alias or not canonical:
                continue
            if canonical not in canonicals_set:
                continue
            alias_to_canonical[alias] = canonical
    return alias_to_canonical
