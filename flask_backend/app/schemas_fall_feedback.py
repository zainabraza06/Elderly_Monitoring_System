"""Elder fall-confirmation events for logging / future ML QA."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


FallFeedbackKind = Literal[
    "okay",
    "need_help",
    "false_alarm",
    "wrong_fall_type",
    "correct_fall_type",
    "no_help_needed",
]


class FallFeedbackEvent(BaseModel):
    patient_id: str = Field(..., description="Stable patient / device identifier")
    caregiver_id: str | None = None
    response: FallFeedbackKind
    fall_detected: bool = True
    predicted_fall_type_code: str | None = None
    fall_probability: float | None = None
    client_event_id: str | None = None
    notes: str | None = None


class FallFeedbackAck(BaseModel):
    ok: bool = True
    logged_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
