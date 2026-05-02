"""Train models on demand and cache the fit (in-memory + on-disk).

The first prediction for a given (phase, model, feature) triple trains the
model on a small subset; later predictions reuse the cached fit. Pickled
caches live under `demo/.cache/` so re-launches are instant.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from typing import Tuple

import numpy as np

from demo.data import build_dataset
from demo.registry import ModelSpec, get_spec


_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

_MEM_CACHE: dict = {}


def _cache_key(phase: str, model_key: str, feature_method: str, train_subset: int) -> str:
    raw = f"{phase}|{model_key}|{feature_method}|{train_subset}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def train_or_load(
    phase: str,
    model_key: str,
    feature_method: str,
    log,
) -> Tuple[object, callable, dict]:
    """Return (fitted_model, image_transform, info_dict)."""

    spec: ModelSpec = get_spec(phase, model_key)
    key = _cache_key(phase, model_key, feature_method, spec.train_subset)

    if key in _MEM_CACHE:
        log(f"Using in-memory cache for {spec.label} / {feature_method}.")
        return _MEM_CACHE[key]

    cache_file = os.path.join(_CACHE_DIR, f"{key}.pkl")
    if os.path.exists(cache_file):
        try:
            log(f"Loading cached model from {os.path.basename(cache_file)}…")
            with open(cache_file, "rb") as f:
                payload = pickle.load(f)
            ds = build_dataset(phase, feature_method, train_subset=spec.train_subset)
            bundle = (payload["model"], ds["transform_image"], payload["info"])
            _MEM_CACHE[key] = bundle
            return bundle
        except Exception as e:
            log(f"Cache miss ({e}); retraining.")

    log(f"Preparing {feature_method.upper()} features ({spec.train_subset} samples)…")
    ds = build_dataset(phase, feature_method, train_subset=spec.train_subset)
    X, y, weights = ds["X_train"], ds["y_train"], ds["weights"]

    log(f"Training {spec.label} on {X.shape[0]} samples × {X.shape[1]} features…")
    t0 = time.time()
    model = spec.builder()

    if spec.needs_class_weights and weights is not None:
        try:
            model.fit(X, y, class_weights=weights)
        except TypeError:
            model.fit(X, y)
    else:
        model.fit(X, y)

    elapsed = time.time() - t0
    log(f"Trained in {elapsed:.2f}s.")

    info = {
        "phase": phase,
        "model": spec.label,
        "feature": feature_method,
        "train_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "train_seconds": round(elapsed, 2),
    }

    try:
        with open(cache_file, "wb") as f:
            pickle.dump({"model": model, "info": info}, f)
    except Exception as e:
        log(f"(Could not persist cache: {e})")

    bundle = (model, ds["transform_image"], info)
    _MEM_CACHE[key] = bundle
    return bundle


def predict_image(model, transform_image, image_28x28: np.ndarray):
    """Return (predicted_label, score_array_or_none)."""
    feats = transform_image(image_28x28)
    pred = int(model.predict(feats)[0])
    score = None
    if hasattr(model, "predict_proba"):
        try:
            score = model.predict_proba(feats)[0]
        except Exception:
            score = None
    elif hasattr(model, "decision_function"):
        try:
            raw = model.decision_function(feats)[0]
            raw = np.atleast_1d(raw)
            ex = np.exp(raw - np.max(raw))
            score = ex / ex.sum()
        except Exception:
            score = None
    return pred, score
