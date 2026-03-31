from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AlertSeverity(str, Enum):
    low = "low"
    medium = "medium"
    high_risk = "high_risk"
    fall_detected = "fall_detected"


class AlertStatus(str, Enum):
    open = "open"
    acknowledged = "acknowledged"
    resolved = "resolved"


class SessionStatus(str, Enum):
    active = "active"
    stopped = "stopped"


class AccelerationUnit(str, Enum):
    g = "g"
    m_s2 = "m_s2"


class GyroscopeUnit(str, Enum):
    dps = "dps"
    rad_s = "rad_s"


class HealthResponse(BaseModel):
    status: str
    app_name: str
    timestamp: datetime


class APIErrorResponse(BaseModel):
    code: str
    message: str
    trace_id: str
    timestamp: datetime
    details: Any | None = None


class PatientCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str = Field(..., min_length=1, max_length=120)
    age: Optional[int] = Field(default=None, ge=0, le=130)
    room_label: Optional[str] = Field(default=None, max_length=80)
    emergency_contact: Optional[str] = Field(default=None, max_length=120)
    notes: Optional[str] = Field(default=None, max_length=500)

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("full_name cannot be blank.")
        return clean


class PatientRecord(BaseModel):
    id: str
    full_name: str
    age: Optional[int] = None
    room_label: Optional[str] = None
    emergency_contact: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime


class DeviceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., min_length=1, max_length=80)
    platform: str = Field(default="mobile_web", max_length=60)
    owner_name: Optional[str] = Field(default=None, max_length=80)

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("label cannot be blank.")
        return clean


class DeviceRecord(BaseModel):
    id: str
    label: str
    platform: str
    owner_name: Optional[str] = None
    created_at: datetime
    last_seen_at: Optional[datetime] = None


class SessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(..., min_length=1, max_length=64)
    device_id: str = Field(..., min_length=1, max_length=64)
    started_by: str = Field(default="system", max_length=80)
    sample_rate_hz: float = Field(default=50.0, gt=0, le=400)
    notes: Optional[str] = Field(default=None, max_length=200)

    @field_validator("patient_id", "device_id", "started_by")
    @classmethod
    def strip_non_empty_identifiers(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Field cannot be blank.")
        return clean


class SessionRecord(BaseModel):
    id: str
    patient_id: str
    device_id: str
    started_by: str
    sample_rate_hz: float
    notes: Optional[str] = None
    status: SessionStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    last_ingested_at: Optional[datetime] = None


class SessionStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stopped_by: str = Field(default="system", max_length=80)
    note: Optional[str] = Field(default=None, max_length=200)

    @field_validator("stopped_by")
    @classmethod
    def validate_stopped_by(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("stopped_by cannot be blank.")
        return clean


class SensorSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_ms: Optional[int] = Field(default=None, ge=0)
    acc_x: float = Field(..., ge=-500.0, le=500.0)
    acc_y: float = Field(..., ge=-500.0, le=500.0)
    acc_z: float = Field(..., ge=-500.0, le=500.0)
    gyro_x: float = Field(..., ge=-5000.0, le=5000.0)
    gyro_y: float = Field(..., ge=-5000.0, le=5000.0)
    gyro_z: float = Field(..., ge=-5000.0, le=5000.0)


class DetectorConfig(BaseModel):
    min_samples: int = Field(default=20, ge=5)
    impact_threshold_g: float = Field(default=2.3, gt=0)
    jerk_threshold_g_per_s: float = Field(default=12.0, gt=0)
    gyro_threshold_dps: float = Field(default=220.0, gt=0)
    stillness_acc_delta_g: float = Field(default=0.18, gt=0)
    stillness_gyro_threshold_dps: float = Field(default=35.0, gt=0)
    medium_risk_score: float = Field(default=0.35, ge=0, le=1)
    high_risk_score: float = Field(default=0.58, ge=0, le=1)
    fall_score: float = Field(default=0.8, ge=0, le=1)
    alert_cooldown_seconds: int = Field(default=25, ge=0)


class DetectorConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_samples: Optional[int] = Field(default=None, ge=5)
    impact_threshold_g: Optional[float] = Field(default=None, gt=0)
    jerk_threshold_g_per_s: Optional[float] = Field(default=None, gt=0)
    gyro_threshold_dps: Optional[float] = Field(default=None, gt=0)
    stillness_acc_delta_g: Optional[float] = Field(default=None, gt=0)
    stillness_gyro_threshold_dps: Optional[float] = Field(default=None, gt=0)
    medium_risk_score: Optional[float] = Field(default=None, ge=0, le=1)
    high_risk_score: Optional[float] = Field(default=None, ge=0, le=1)
    fall_score: Optional[float] = Field(default=None, ge=0, le=1)
    alert_cooldown_seconds: Optional[int] = Field(default=None, ge=0)


class DetectionResult(BaseModel):
    severity: AlertSeverity
    score: float = Field(..., ge=0, le=1)
    fall_probability: float = Field(..., ge=0, le=1)
    peak_acc_g: float = Field(..., ge=0)
    peak_gyro_dps: float = Field(..., ge=0)
    peak_jerk_g_per_s: float = Field(..., ge=0)
    stillness_ratio: float = Field(..., ge=0, le=1)
    samples_analyzed: int = Field(..., ge=0)
    message: str
    reasons: list[str]
    detected_at: datetime


class PatientLiveStatus(BaseModel):
    patient_id: str
    patient_name: str
    room_label: Optional[str] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    severity: AlertSeverity = AlertSeverity.low
    score: float = 0.0
    fall_probability: float = 0.0
    last_ingested_at: Optional[datetime] = None
    last_message: str = "No live data yet."
    sample_rate_hz: Optional[float] = None
    latest_metrics: dict[str, float] = Field(default_factory=dict)
    active_alert_ids: list[str] = Field(default_factory=list)


class AlertRecord(BaseModel):
    id: str
    patient_id: str
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    severity: AlertSeverity
    status: AlertStatus
    message: str
    score: float = Field(..., ge=0, le=1)
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    note: Optional[str] = None
    manually_triggered: bool = False


class AlertActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = Field(default="operator", max_length=80)
    note: Optional[str] = Field(default=None, max_length=200)

    @field_validator("actor")
    @classmethod
    def validate_actor(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("actor cannot be blank.")
        return clean


class ManualAlertCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(..., min_length=1, max_length=64)
    session_id: Optional[str] = Field(default=None, max_length=64)
    device_id: Optional[str] = Field(default=None, max_length=64)
    severity: AlertSeverity = AlertSeverity.high_risk
    message: str = Field(default="Manual alert triggered.", min_length=1, max_length=240)
    actor: str = Field(default="operator", max_length=80)
    note: Optional[str] = Field(default=None, max_length=200)

    @field_validator("patient_id", "message", "actor")
    @classmethod
    def validate_manual_alert_strings(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Field cannot be blank.")
        return clean


class SensorBatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(..., min_length=1, max_length=64)
    device_id: str = Field(..., min_length=1, max_length=64)
    session_id: str = Field(..., min_length=1, max_length=64)
    source: str = Field(default="mobile", max_length=40)
    sampling_rate_hz: float = Field(default=50.0, gt=0, le=400)
    acceleration_unit: AccelerationUnit = AccelerationUnit.m_s2
    gyroscope_unit: GyroscopeUnit = GyroscopeUnit.dps
    battery_level: Optional[float] = Field(default=None, ge=0, le=100)
    samples: list[SensorSample] = Field(..., min_length=1, max_length=4096)

    @field_validator("patient_id", "device_id", "session_id", "source")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Field cannot be blank.")
        return clean

    @model_validator(mode="after")
    def validate_timestamps_monotonic(self):
        timestamps = [sample.timestamp_ms for sample in self.samples if sample.timestamp_ms is not None]
        if timestamps and timestamps != sorted(timestamps):
            raise ValueError("samples timestamps must be monotonically non-decreasing when provided.")
        return self


class TelemetrySnapshot(BaseModel):
    patient_id: str
    patient_name: str
    room_label: Optional[str] = None
    session_id: str
    device_id: str
    source: str
    sampling_rate_hz: float = Field(..., gt=0)
    acceleration_unit: AccelerationUnit
    gyroscope_unit: GyroscopeUnit
    battery_level: Optional[float] = Field(default=None, ge=0, le=100)
    received_at: datetime
    samples_in_last_batch: int = Field(..., ge=0)
    latest_samples: list[SensorSample] = Field(default_factory=list)


class SensorBatchOut(BaseModel):
    ingested_samples: int
    detection: DetectionResult
    active_alert: Optional[AlertRecord] = None
    live_status: PatientLiveStatus
    telemetry: Optional[TelemetrySnapshot] = None


class MonitorEvent(BaseModel):
    id: str
    type: str
    created_at: datetime
    payload: dict[str, Any]


class SystemSummary(BaseModel):
    total_patients: int
    total_devices: int
    active_sessions: int
    open_alerts: int
    acknowledged_alerts: int
    last_event_at: Optional[datetime] = None
