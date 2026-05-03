"""Paths for FastAPI inference service."""

from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_FLASK_ROOT = _PKG.parent
_REPO_ROOT = _FLASK_ROOT.parent


def model_root() -> Path:
    raw = os.environ.get("MODEL_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return (_FLASK_ROOT / "models").resolve()


def repo_root() -> Path:
    """Repository root (contains ``models/``, ``scripts/``, ``data/``)."""
    return _REPO_ROOT.resolve()


def inference_manifest_path() -> Path:
    raw = os.environ.get("INFERENCE_MANIFEST")
    if raw:
        return Path(raw).expanduser().resolve()
    return (_FLASK_ROOT / "models" / "inference_manifest.json").resolve()
