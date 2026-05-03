"""Load frozen models; fall → optional fall-type; else ADL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass(frozen=True)
class InferenceArtifacts:
    manifest: dict[str, Any]
    fall_model: Any
    fall_scaler: Any
    adl_model: Any
    adl_scaler: Any
    adl_encoder: Any
    fall_type_model: Any
    fall_type_scaler: Any
    fall_type_indices: np.ndarray
    fall_type_encoder: Any
    enhanced_dim: int
    fall_type_dim: int
    fall_threshold: float


def load_artifacts(manifest_path: Path, models_dir: Path) -> InferenceArtifacts:
    manifest_path = manifest_path.resolve()
    models_dir = models_dir.resolve()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    enhanced_dim = int(manifest["enhanced_feature_dim"])
    fall_type_dim = int(manifest["fall_type_raw_dim"])
    threshold = float(manifest.get("fall_probability_threshold", 0.5))
    art = manifest["artifacts"]

    def p(rel: str) -> Path:
        out = (models_dir / rel).resolve()
        if not out.is_file():
            raise FileNotFoundError(str(out))
        return out

    fall_model = joblib.load(p(art["fall_binary"]["model_path"]))
    fall_scaler = joblib.load(p(art["fall_binary"]["scaler_path"]))
    adl_model = joblib.load(p(art["adl"]["model_path"]))
    adl_scaler = joblib.load(p(art["adl"]["scaler_path"]))
    adl_encoder = joblib.load(p(art["adl"]["label_encoder_path"]))

    fall_type_model = joblib.load(p(art["fall_type"]["model_path"]))
    fall_type_scaler = joblib.load(p(art["fall_type"]["scaler_path"]))
    fall_type_indices = np.asarray(joblib.load(p(art["fall_type"]["feature_indices_path"])), dtype=int)
    fall_type_encoder = joblib.load(p(art["fall_type"]["label_encoder_path"]))

    nf = getattr(fall_scaler, "n_features_in_", None)
    if nf is not None and int(nf) != enhanced_dim:
        raise ValueError(f"Fall scaler wants {nf}, manifest {enhanced_dim}")
    na = getattr(adl_scaler, "n_features_in_", None)
    if na is not None and int(na) != enhanced_dim:
        raise ValueError(f"ADL scaler wants {na}, manifest {enhanced_dim}")
    ft_n = getattr(fall_type_scaler, "n_features_in_", None)
    if ft_n is not None and int(ft_n) != fall_type_dim:
        raise ValueError(f"Fall-type scaler wants {ft_n}, manifest {fall_type_dim}")

    return InferenceArtifacts(
        manifest=manifest,
        fall_model=fall_model,
        fall_scaler=fall_scaler,
        adl_model=adl_model,
        adl_scaler=adl_scaler,
        adl_encoder=adl_encoder,
        fall_type_model=fall_type_model,
        fall_type_scaler=fall_type_scaler,
        fall_type_indices=fall_type_indices,
        fall_type_encoder=fall_type_encoder,
        enhanced_dim=enhanced_dim,
        fall_type_dim=fall_type_dim,
        fall_threshold=threshold,
    )


def run_inference(
    art: InferenceArtifacts,
    enhanced_features: list[float],
    fall_type_features: list[float] | None,
    *,
    predict_fall_type: bool,
) -> dict[str, Any]:
    x = np.asarray(enhanced_features, dtype=np.float64).reshape(1, -1)
    if x.shape[1] != art.enhanced_dim:
        raise ValueError(f"enhanced_features length {x.shape[1]} != {art.enhanced_dim}")

    xf = art.fall_scaler.transform(x)
    p_fall = float(art.fall_model.predict_proba(xf)[0, 1])
    is_fall = p_fall >= art.fall_threshold

    out: dict[str, Any] = {
        "is_fall": is_fall,
        "fall_probability": p_fall,
        "fall_threshold": art.fall_threshold,
        "schema_version": str(art.manifest.get("schema_version", "1.0")),
    }

    if not is_fall:
        xa = art.adl_scaler.transform(x)
        cid = int(art.adl_model.predict(xa)[0])
        label = str(art.adl_encoder.inverse_transform(np.array([cid]))[0])
        out["branch"] = "adl"
        out["activity_class_index"] = cid
        out["activity_label"] = label
        out["fall_type_code"] = None
        out["fall_type_label"] = None
        out["fall_type_class_index"] = None
        out["fall_type_skipped_reason"] = None
        return out

    out["branch"] = "fall"
    out["activity_class_index"] = None
    out["activity_label"] = None

    if not predict_fall_type or fall_type_features is None:
        out["fall_type_code"] = None
        out["fall_type_label"] = None
        out["fall_type_class_index"] = None
        out["fall_type_skipped_reason"] = (
            "predict_fall_type_disabled" if not predict_fall_type else "fall_type_features_missing"
        )
        return out

    ft = np.asarray(fall_type_features, dtype=np.float64).reshape(1, -1)
    if ft.shape[1] != art.fall_type_dim:
        raise ValueError(f"fall_type_features length {ft.shape[1]} != {art.fall_type_dim}")

    xs = art.fall_type_scaler.transform(ft)
    xsel = xs[:, art.fall_type_indices]
    pred = art.fall_type_model.predict(xsel)
    fi = int(pred[0])
    code = str(art.fall_type_encoder.inverse_transform(np.array([fi]))[0])
    out["fall_type_class_index"] = fi
    out["fall_type_code"] = code
    out["fall_type_label"] = code
    out["fall_type_skipped_reason"] = None
    return out
