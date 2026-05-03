"""Request / response models for motion inference (strict validation)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MotionInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enhanced_features: list[float] = Field(
        ...,
        description=(
            "Float vector length = models/inference_manifest.json enhanced_feature_dim "
            "(frozen fall + ADL scalers; must match training extraction order)."
        ),
    )
    fall_type_features: list[float] | None = Field(
        default=None,
        description=(
            "263 floats: raw multi-sensor fall-type vector (same extraction as Colab training); "
            "must match models/inference_manifest.json fall_type_raw_dim after scaler.fit shape."
        ),
    )
    predict_fall_type: bool = Field(
        default=True,
        description="If true and fall_type_features provided, run fall-type model when fall is predicted.",
    )


class MotionInferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_fall: bool
    fall_probability: float
    fall_threshold: float
    branch: str
    activity_label: str | None = None
    activity_class_index: int | None = None
    fall_type_code: str | None = None
    fall_type_label: str | None = None
    fall_type_class_index: int | None = None
    fall_type_skipped_reason: str | None = None
    schema_version: str
