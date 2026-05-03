#!/usr/bin/env python3
"""
Local smoke test for all three frozen model bundles (fall-binary, ADL, fall-type).

Run from repo root:
  set PYTHONPATH=scripts
  py scripts/verify_three_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_MODELS = _REPO / "models"
_MANIFEST = _MODELS / "inference_manifest.json"

if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))


def main() -> int:
    if not _MANIFEST.is_file():
        print("FAIL: models/inference_manifest.json missing")
        return 1
    m = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    art = m["artifacts"]
    d_enh = int(m["enhanced_feature_dim"])
    d_ft = int(m["fall_type_raw_dim"])

    # --- 1) Fall binary (116-D) ---
    try:
        fall_m = joblib.load(_MODELS / art["fall_binary"]["model_path"])
        fall_s = joblib.load(_MODELS / art["fall_binary"]["scaler_path"])
        assert getattr(fall_s, "n_features_in_", d_enh) == d_enh
        x = np.zeros((1, d_enh), dtype=np.float64)
        xs = fall_s.transform(x)
        proba = fall_m.predict_proba(xs)
        assert proba.shape[1] == 2
        print(f"OK  fall-binary  ({d_enh}-D)  proba_shape={proba.shape}")
    except Exception as e:
        print(f"FAIL fall-binary: {e}")
        return 1

    # --- 2) ADL multiclass (same 116-D) ---
    try:
        adl_m = joblib.load(_MODELS / art["adl"]["model_path"])
        adl_s = joblib.load(_MODELS / art["adl"]["scaler_path"])
        le = joblib.load(_MODELS / art["adl"]["label_encoder_path"])
        assert getattr(adl_s, "n_features_in_", d_enh) == d_enh
        x = np.zeros((1, d_enh), dtype=np.float64)
        xa = adl_s.transform(x)
        pred = adl_m.predict(xa)
        lab = le.inverse_transform(pred)
        print(f"OK  ADL          ({d_enh}-D)  n_classes={len(le.classes_)}  sample_label={lab[0]!r}")
    except Exception as e:
        print(f"FAIL ADL: {e}")
        return 1

    # --- 3) Fall type (263-D raw -> scaler -> MI columns -> classifier) ---
    try:
        ft_m = joblib.load(_MODELS / art["fall_type"]["model_path"])
        ft_s = joblib.load(_MODELS / art["fall_type"]["scaler_path"])
        idx = np.asarray(joblib.load(_MODELS / art["fall_type"]["feature_indices_path"]), dtype=int)
        ft_le = joblib.load(_MODELS / art["fall_type"]["label_encoder_path"])
        assert getattr(ft_s, "n_features_in_", d_ft) == d_ft
        raw = np.zeros((1, d_ft), dtype=np.float64)
        xs = ft_s.transform(raw)
        xsel = xs[:, idx]
        pred = ft_m.predict(xsel)
        code = str(ft_le.inverse_transform(pred)[0])
        print(f"OK  fall-type    ({d_ft}-D raw, {xsel.shape[1]} MI cols)  sample_class={code!r}")
    except Exception as e:
        print(f"FAIL fall-type: {e}")
        return 1

    print("\nAll three model bundles load and forward-pass locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
