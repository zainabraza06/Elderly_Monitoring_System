"""Motion inference request/response."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MotionInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enhanced_features: list[float] = Field(
        ...,
        description="Length = inference_manifest enhanced_feature_dim (116 after train_mobiact_baselines).",
    )
    fall_type_features: list[float] | None = Field(
        default=None,
        description="Length = fall_type_raw_dim (263) when requesting 4-class fall type.",
    )
    predict_fall_type: bool = Field(default=True)


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
