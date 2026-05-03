"""Fall-type (4-class) pipeline constants — matches Colab / inference_manifest.json."""

from __future__ import annotations

# MobiAct impact-centric windows (seconds * Hz).
SAMPLING_RATE_HZ = 50
WINDOW_DURATION_S = 6
WINDOW_SAMPLES = SAMPLING_RATE_HZ * WINDOW_DURATION_S  # 300

FALL_TYPE_RAW_DIM = 263
FALL_CODES = ("BSC", "FOL", "FKL", "SDL")

FALL_NAMES: dict[str, str] = {
    "BSC": "Back Fall",
    "FOL": "Forward Fall",
    "FKL": "Knees Fall",
    "SDL": "Side Fall",
}

# Colab mutual-information shortlist size before tree models.
TOP_MI_FEATURES_DEFAULT = 150
