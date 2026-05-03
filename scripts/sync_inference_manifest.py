"""
Update models/inference_manifest.json enhanced_feature_dim and fall_type_raw_dim
from joblib scalers (run after replacing models from Colab / Drive).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "inference_manifest.json"
MODELS = ROOT / "models"


def main() -> int:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sf = joblib.load(MODELS / "baseline_fall" / "scaler_fall.pkl")
    sa = joblib.load(MODELS / "baseline_adl" / "scaler_adl.pkl")
    st = joblib.load(MODELS / "baseline_falltype" / "scaler.pkl")
    ef = int(getattr(sf, "n_features_in_", m["enhanced_feature_dim"]))
    ea = int(getattr(sa, "n_features_in_", ef))
    if ef != ea:
        print("ERROR: fall vs ADL scaler width differs:", ef, ea, file=sys.stderr)
        return 1
    ft = int(getattr(st, "n_features_in_", m["fall_type_raw_dim"]))
    m["enhanced_feature_dim"] = ef
    m["fall_type_raw_dim"] = ft
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print(f"Updated manifest: enhanced_feature_dim={ef}, fall_type_raw_dim={ft}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
