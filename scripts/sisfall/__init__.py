"""
SisFall public dataset — layout hints and training integration.

Place extracted SisFall files under ``data/SisFall_dataset/`` (see `dataset_paths`).
Training scripts in this repo currently default to **MobiAct**; add a loader mirroring
``baseline_fall.mobiact_dataset`` for SisFall `.txt` / label files when you add the data.
"""

from .dataset_paths import SISFALL_DIR, expect_sisfall_root

__all__ = ["SISFALL_DIR", "expect_sisfall_root"]
