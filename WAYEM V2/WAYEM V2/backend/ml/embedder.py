# backend/ml/embedder.py
from __future__ import annotations

import os
import threading
from typing import Iterable, List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers is required for embedder.py. "
        "Install with: pip install sentence-transformers"
    ) from e


# ---------------------------
# Globals (lazy init + thread-safe)
# ---------------------------
_MODEL_LOCK = threading.RLock()
_MODEL: SentenceTransformer | None = None
_MODEL_NAME: str | None = None
_DEVICE = os.getenv("EMBEDDER_DEVICE")  # e.g., "cpu", "cuda", "mps"
_DEFAULT_MODEL = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
# Good alternatives:
# - "all-mpnet-base-v2" (higher quality, slower)
# - "multi-qa-MiniLM-L6-cos-v1" (QA-optimized)


# ---------------------------
# Public API
# ---------------------------

def get_model(name: str | None = None) -> SentenceTransformer:
    """
    Returns a singleton SentenceTransformer.
    Respects EMBEDDER_MODEL/EMBEDDER_DEVICE env vars unless `name` is provided.
    """
    global _MODEL, _MODEL_NAME
    with _MODEL_LOCK:
        target_name = name or _DEFAULT_MODEL
        if _MODEL is None or _MODEL_NAME != target_name:
            _MODEL = SentenceTransformer(target_name, device=_DEVICE or None)
            _MODEL_NAME = target_name
        return _MODEL


def warmup(sample_text: str = "warmup"):
    """
    Optional: call at startup to load weights into memory and compile kernels.
    """
    _ = embed([sample_text])


def is_ready() -> bool:
    """
    Quick readiness check (model loaded once).
    """
    return _MODEL is not None


def set_model(name: str):
    """
    Explicitly switch models at runtime (rarely needed).
    """
    get_model(name)  # re-inits if different


def embed(
    texts: Iterable[str],
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """
    Encode a list/iterable of strings into a 2D float32 array.
    - Returns shape (N, D), N==len(texts). For empty input, returns (0, D) with D inferred lazily.
    - Normalizes rows to unit length if `normalize_embeddings` is True (recommended for cosine).
    - Safe against None/empty strings; those become "".

    Raises RuntimeError if encoding fails.
    """
    model = get_model()
    # Coerce to list and sanitize
    arr: List[str] = [(_t or "").strip() for _t in texts]
    if not arr:
        # create a zero-row with correct dim by encoding a dummy once
        dummy = model.encode([""], normalize_embeddings=False, convert_to_numpy=True, batch_size=1)
        out = np.zeros((0, int(dummy.shape[1])), dtype=np.float32)
        return out

    try:
        vecs = model.encode(
            arr,
            normalize_embeddings=False,  # we normalize ourselves (float32 + guard rails)
            convert_to_numpy=True,
            batch_size=max(1, int(batch_size)),
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Sentence embedding failed: {e}") from e

    if normalize_embeddings:
        vecs = _normalize_rows(vecs)
    return vecs


# ---------------------------
# Internals
# ---------------------------

def _normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize each row; rows with ~zero norm become zeros (not NaN).
    """
    if X.size == 0:
        return X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scale = np.where(norms > eps, 1.0 / norms, 0.0).astype(np.float32)
    return (X * scale).astype(np.float32, copy=False)
