"""Dimension contracts vs inference_manifest and frozen scalers."""

from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
import pytest

from baseline_fallandadl.enhanced_features import extract_enhanced_features
from baseline_falltype.config import WINDOW_SAMPLES
from baseline_falltype.feature_extractors import CompleteFallFeatureExtractor


def test_deployed_enhanced_dim_matches_manifest(repo_root: Path, inference_manifest: dict) -> None:
    """Fall + ADL inference scalers must agree with manifest (same vector for both tasks)."""
    d = int(inference_manifest["enhanced_feature_dim"])
    sf = joblib.load(repo_root / "models" / "baseline_fall" / "scaler_fall.pkl")
    sa = joblib.load(repo_root / "models" / "baseline_adl" / "scaler_adl.pkl")
    nf = getattr(sf, "n_features_in_", None)
    na = getattr(sa, "n_features_in_", None)
    assert nf == d, f"Fall scaler wants {nf}, manifest says {d}"
    assert na == d, f"ADL scaler wants {na}, manifest says {d}"


def test_baseline_fallandadl_extractor_is_116_dim() -> None:
    """Reference enhanced extractor (training notebook with full fusion) yields 116-D per window."""
    rng = np.random.default_rng(42)
    acc = rng.standard_normal((1, 300, 3))
    gyro = rng.standard_normal((1, 300, 3))
    ori = rng.standard_normal((1, 300, 3))
    X = extract_enhanced_features(acc, gyro, ori)
    assert X.shape == (1, 116)


def test_fall_type_manifest_matches_saved_scaler(repo_root: Path, inference_manifest: dict) -> None:
    scaler_path = repo_root / "models" / "baseline_falltype" / "scaler.pkl"
    if not scaler_path.is_file():
        pytest.skip("fall-type scaler not present")
    scaler = joblib.load(scaler_path)
    n = getattr(scaler, "n_features_in_", None)
    expected = int(inference_manifest["fall_type_raw_dim"])
    assert n == expected, f"Manifest fall_type_raw_dim {expected} != scaler n_features_in_ {n}"


def test_repo_extractor_fixed_width_is_350() -> None:
    """Repo multisensor extractor pads to 350-D; Colab fall-type training used 263-D — send 263-D per manifest."""
    rng = np.random.default_rng(43)
    acc = rng.standard_normal((1, WINDOW_SAMPLES, 3))
    gyro = rng.standard_normal((1, WINDOW_SAMPLES, 3))
    ori = rng.standard_normal((1, WINDOW_SAMPLES, 3))
    ext = CompleteFallFeatureExtractor(fs=50.0)
    X = ext.extract_batch(acc, gyro, ori, desc="test")
    assert X.shape == (1, 350)
