"""Pytest root: ensure repo + scripts are importable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"

for p in (_ROOT, _SCRIPTS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _ROOT


@pytest.fixture(scope="session")
def inference_manifest(repo_root: Path) -> dict:
    import json

    path = repo_root / "models" / "inference_manifest.json"
    if not path.is_file():
        pytest.skip("models/inference_manifest.json missing")
    return json.loads(path.read_text(encoding="utf-8"))
