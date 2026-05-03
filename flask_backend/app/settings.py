"""Paths and environment for the FastAPI service."""

from __future__ import annotations

import os
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
_FLASK_BACKEND_ROOT = _PKG_ROOT.parent
_REPO_ROOT = _FLASK_BACKEND_ROOT.parent


def model_root() -> Path:
    raw = os.environ.get("MODEL_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return (_REPO_ROOT / "models").resolve()


def inference_manifest_path() -> Path:
    raw = os.environ.get("INFERENCE_MANIFEST")
    if raw:
        return Path(raw).expanduser().resolve()
    return (_REPO_ROOT / "models" / "inference_manifest.json").resolve()
