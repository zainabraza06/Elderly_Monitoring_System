"""Expected on-disk layout for the SisFall dataset (when you add it to ``data/``)."""

from __future__ import annotations

from pathlib import Path

from baseline_fall.config import repo_scripts_parents

REPO_ROOT = repo_scripts_parents()
SISFALL_DIR = REPO_ROOT / "data" / "SisFall_dataset"


def expect_sisfall_root() -> Path:
    """
    Return the conventional SisFall root. Does not require the folder to exist
    (training scripts may skip SisFall if missing).
    """
    return SISFALL_DIR
