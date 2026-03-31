from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from .schemas import (
    AlertActionRequest,
    AlertRecord,
    AlertSeverity,
    AlertStatus,
    DetectorConfig,
    DetectorConfigUpdate,
    DeviceCreate,
    DeviceRecord,
    ManualAlertCreate,
    MonitorEvent,
    PatientCreate,
    PatientLiveStatus,
    PatientRecord,
    SensorBatchIn,
    SensorBatchOut,
    SessionCreate,
    SessionRecord,
    SessionStatus,
    SessionStopRequest,
    SystemSummary,
    TelemetrySnapshot,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def model_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class BackendStore:
    def __init__(self, recent_event_limit: int = 500) -> None:
        self._lock = asyncio.Lock()
        self.detector_config = DetectorConfig()
        self.patients: dict[str, PatientRecord] = {}
        self.devices: dict[str, DeviceRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.alerts: dict[str, AlertRecord] = {}
        self.live_status: dict[str, PatientLiveStatus] = {}
        self.recent_events: deque[MonitorEvent] = deque(maxlen=recent_event_limit)
        self.telemetry_snapshots: dict[str, TelemetrySnapshot] = {}
        self.recent_telemetry: deque[TelemetrySnapshot] = deque(maxlen=120)

    def _new_event(self, event_type: str, payload: dict) -> MonitorEvent:
        event = MonitorEvent(
            id=f"evt_{uuid4().hex[:12]}",
            type=event_type,
            created_at=utcnow(),
            payload=payload,
        )
        self.recent_events.appendleft(event)
        return event

    def _active_alert_ids_for_patient(self, patient_id: str) -> list[str]:
        return [
            alert.id
            for alert in self.alerts.values()
            if alert.patient_id == patient_id and alert.status != AlertStatus.resolved
        ]

    def _last_open_alert_for_patient(self, patient_id: str) -> Optional[AlertRecord]:
        alerts = [
            alert
            for alert in self.alerts.values()
            if alert.patient_id == patient_id and alert.status != AlertStatus.resolved
        ]
        if not alerts:
            return None
        return max(alerts, key=lambda item: item.created_at)

    async def create_patient(self, data: PatientCreate) -> tuple[PatientRecord, MonitorEvent]:
        async with self._lock:
            patient = PatientRecord(
                id=f"pat_{uuid4().hex[:10]}",
                created_at=utcnow(),
                **model_to_dict(data),
            )
            self.patients[patient.id] = patient
            self.live_status[patient.id] = PatientLiveStatus(
                patient_id=patient.id,
                patient_name=patient.full_name,
                room_label=patient.room_label,
            )
            event = self._new_event("patient.created", model_to_dict(patient))
            return patient, event

    async def list_patients(self) -> list[PatientRecord]:
        async with self._lock:
            return list(self.patients.values())

    async def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        async with self._lock:
            return self.patients.get(patient_id)

    async def create_device(self, data: DeviceCreate) -> tuple[DeviceRecord, MonitorEvent]:
        async with self._lock:
            device = DeviceRecord(
                id=f"dev_{uuid4().hex[:10]}",
                created_at=utcnow(),
                **model_to_dict(data),
            )
            self.devices[device.id] = device
            event = self._new_event("device.created", model_to_dict(device))
            return device, event

    async def list_devices(self) -> list[DeviceRecord]:
        async with self._lock:
            return list(self.devices.values())

    async def get_device(self, device_id: str) -> Optional[DeviceRecord]:
        async with self._lock:
            return self.devices.get(device_id)

    async def start_session(self, data: SessionCreate) -> tuple[SessionRecord, list[MonitorEvent]]:
        async with self._lock:
            if data.patient_id not in self.patients:
                raise ValueError("Patient not found.")
            if data.device_id not in self.devices:
                raise ValueError("Device not found.")

            for session in self.sessions.values():
                if session.status == SessionStatus.active and (
                    session.patient_id == data.patient_id or session.device_id == data.device_id
                ):
                    raise ValueError("An active session already exists for this patient or device.")

            session = SessionRecord(
                id=f"ses_{uuid4().hex[:10]}",
                status=SessionStatus.active,
                started_at=utcnow(),
                **model_to_dict(data),
            )
            self.sessions[session.id] = session
            live = self.live_status[data.patient_id]
            live.session_id = session.id
            live.device_id = data.device_id
            live.sample_rate_hz = data.sample_rate_hz
            live.last_message = "Monitoring session started."

            events = [
                self._new_event("session.started", model_to_dict(session)),
                self._new_event("patient.live.updated", model_to_dict(live)),
            ]
            return session, events

    async def stop_session(self, session_id: str, data: SessionStopRequest) -> tuple[SessionRecord, list[MonitorEvent]]:
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError("Session not found.")
            if session.status == SessionStatus.stopped:
                return session, []

            session.status = SessionStatus.stopped
            session.ended_at = utcnow()
            if data.note:
                session.notes = data.note

            patient = self.patients.get(session.patient_id)
            live = self.live_status.get(session.patient_id)
            events: list[MonitorEvent] = [self._new_event("session.stopped", model_to_dict(session))]

            if live:
                live.session_id = None
                live.device_id = None
                live.last_message = "Monitoring session stopped."
                if patient:
                    live.patient_name = patient.full_name
                    live.room_label = patient.room_label
                events.append(self._new_event("patient.live.updated", model_to_dict(live)))

            return session, events

    async def list_sessions(self, active_only: bool = False) -> list[SessionRecord]:
        async with self._lock:
            sessions = list(self.sessions.values())
            if active_only:
                sessions = [session for session in sessions if session.status == SessionStatus.active]
            return sessions

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        async with self._lock:
            return self.sessions.get(session_id)

    async def get_live_status_for_patient(self, patient_id: str) -> Optional[PatientLiveStatus]:
        async with self._lock:
            return self.live_status.get(patient_id)

    async def list_live_statuses(self) -> list[PatientLiveStatus]:
        async with self._lock:
            return list(self.live_status.values())

    async def get_telemetry_for_patient(self, patient_id: str) -> Optional[TelemetrySnapshot]:
        async with self._lock:
            return self.telemetry_snapshots.get(patient_id)

    async def list_recent_telemetry(self, limit: int = 20) -> list[TelemetrySnapshot]:
        async with self._lock:
            return list(self.recent_telemetry)[:limit]

    async def ingest_detection(
        self,
        payload: SensorBatchIn,
        detection,
    ) -> tuple[SensorBatchOut, list[MonitorEvent]]:
        async with self._lock:
            patient = self.patients.get(payload.patient_id)
            if patient is None:
                raise ValueError("Patient not found.")
            device = self.devices.get(payload.device_id)
            if device is None:
                raise ValueError("Device not found.")
            session = self.sessions.get(payload.session_id)
            if session is None:
                raise ValueError("Session not found.")
            if session.status != SessionStatus.active:
                raise ValueError("Session is not active.")
            if session.patient_id != payload.patient_id or session.device_id != payload.device_id:
                raise ValueError("Session does not belong to the provided patient/device pair.")

            now = utcnow()
            session.last_ingested_at = now
            device.last_seen_at = now

            live = self.live_status.get(payload.patient_id)
            if live is None:
                live = PatientLiveStatus(
                    patient_id=payload.patient_id,
                    patient_name=patient.full_name,
                    room_label=patient.room_label,
                )
                self.live_status[payload.patient_id] = live

            live.patient_name = patient.full_name
            live.room_label = patient.room_label
            live.session_id = payload.session_id
            live.device_id = payload.device_id
            live.sample_rate_hz = payload.sampling_rate_hz
            live.severity = detection.severity
            live.score = detection.score
            live.fall_probability = detection.fall_probability
            live.last_ingested_at = now
            live.last_message = detection.message
            live.latest_metrics = {
                "peak_acc_g": detection.peak_acc_g,
                "peak_gyro_dps": detection.peak_gyro_dps,
                "peak_jerk_g_per_s": detection.peak_jerk_g_per_s,
                "stillness_ratio": detection.stillness_ratio,
            }
            if payload.battery_level is not None:
                live.latest_metrics["battery_level"] = payload.battery_level

            telemetry = TelemetrySnapshot(
                patient_id=payload.patient_id,
                patient_name=patient.full_name,
                room_label=patient.room_label,
                session_id=payload.session_id,
                device_id=payload.device_id,
                source=payload.source,
                sampling_rate_hz=payload.sampling_rate_hz,
                acceleration_unit=payload.acceleration_unit,
                gyroscope_unit=payload.gyroscope_unit,
                battery_level=payload.battery_level,
                received_at=now,
                samples_in_last_batch=len(payload.samples),
                latest_samples=payload.samples[-20:],
            )
            self.telemetry_snapshots[payload.patient_id] = telemetry
            self.recent_telemetry.appendleft(telemetry)

            events: list[MonitorEvent] = []
            active_alert: Optional[AlertRecord] = None

            if detection.severity in {AlertSeverity.high_risk, AlertSeverity.fall_detected}:
                last_alert = self._last_open_alert_for_patient(payload.patient_id)
                can_create_alert = True
                if last_alert is not None:
                    cooldown_deadline = last_alert.created_at + timedelta(seconds=self.detector_config.alert_cooldown_seconds)
                    if now <= cooldown_deadline and last_alert.severity == detection.severity:
                        can_create_alert = False
                        active_alert = last_alert

                if can_create_alert:
                    active_alert = AlertRecord(
                        id=f"alt_{uuid4().hex[:10]}",
                        patient_id=payload.patient_id,
                        session_id=payload.session_id,
                        device_id=payload.device_id,
                        severity=detection.severity,
                        status=AlertStatus.open,
                        message=detection.message,
                        score=detection.score,
                        created_at=now,
                        note="Auto-generated from live detection stream.",
                        manually_triggered=False,
                    )
                    self.alerts[active_alert.id] = active_alert
                    events.append(self._new_event("alert.created", model_to_dict(active_alert)))

            live.active_alert_ids = self._active_alert_ids_for_patient(payload.patient_id)
            events.append(self._new_event("telemetry.ingested", model_to_dict(telemetry)))
            events.append(
                self._new_event(
                    "detection.updated",
                    {
                        "patient_id": payload.patient_id,
                        "session_id": payload.session_id,
                        "device_id": payload.device_id,
                        "detection": model_to_dict(detection),
                    },
                )
            )
            events.append(self._new_event("patient.live.updated", model_to_dict(live)))

            response = SensorBatchOut(
                ingested_samples=detection.samples_analyzed,
                detection=detection,
                active_alert=active_alert,
                live_status=live,
                telemetry=telemetry,
            )
            return response, events

    async def list_alerts(
        self,
        status: Optional[AlertStatus] = None,
        patient_id: Optional[str] = None,
    ) -> list[AlertRecord]:
        async with self._lock:
            alerts = list(self.alerts.values())
            if status is not None:
                alerts = [alert for alert in alerts if alert.status == status]
            if patient_id is not None:
                alerts = [alert for alert in alerts if alert.patient_id == patient_id]
            alerts.sort(key=lambda item: item.created_at, reverse=True)
            return alerts

    async def create_manual_alert(self, data: ManualAlertCreate) -> tuple[AlertRecord, list[MonitorEvent]]:
        async with self._lock:
            if data.patient_id not in self.patients:
                raise ValueError("Patient not found.")

            alert = AlertRecord(
                id=f"alt_{uuid4().hex[:10]}",
                patient_id=data.patient_id,
                session_id=data.session_id,
                device_id=data.device_id,
                severity=data.severity,
                status=AlertStatus.open,
                message=data.message,
                score=1.0 if data.severity == AlertSeverity.fall_detected else 0.9,
                created_at=utcnow(),
                note=data.note,
                manually_triggered=True,
            )
            self.alerts[alert.id] = alert

            live = self.live_status.get(data.patient_id)
            events = [self._new_event("alert.created", model_to_dict(alert))]
            if live:
                live.active_alert_ids = self._active_alert_ids_for_patient(data.patient_id)
                live.severity = data.severity
                live.last_message = data.message
                events.append(self._new_event("patient.live.updated", model_to_dict(live)))
            return alert, events

    async def acknowledge_alert(self, alert_id: str, data: AlertActionRequest) -> tuple[AlertRecord, list[MonitorEvent]]:
        async with self._lock:
            alert = self.alerts.get(alert_id)
            if alert is None:
                raise ValueError("Alert not found.")
            if alert.status == AlertStatus.resolved:
                raise ValueError("Resolved alerts cannot be acknowledged.")

            alert.status = AlertStatus.acknowledged
            alert.acknowledged_at = utcnow()
            alert.acknowledged_by = data.actor
            if data.note:
                alert.note = data.note

            events = [self._new_event("alert.acknowledged", model_to_dict(alert))]
            live = self.live_status.get(alert.patient_id)
            if live:
                live.active_alert_ids = self._active_alert_ids_for_patient(alert.patient_id)
                events.append(self._new_event("patient.live.updated", model_to_dict(live)))
            return alert, events

    async def resolve_alert(self, alert_id: str, data: AlertActionRequest) -> tuple[AlertRecord, list[MonitorEvent]]:
        async with self._lock:
            alert = self.alerts.get(alert_id)
            if alert is None:
                raise ValueError("Alert not found.")

            alert.status = AlertStatus.resolved
            alert.resolved_at = utcnow()
            alert.resolved_by = data.actor
            if data.note:
                alert.note = data.note

            events = [self._new_event("alert.resolved", model_to_dict(alert))]
            live = self.live_status.get(alert.patient_id)
            if live:
                live.active_alert_ids = self._active_alert_ids_for_patient(alert.patient_id)
                events.append(self._new_event("patient.live.updated", model_to_dict(live)))
            return alert, events

    async def get_recent_events(self) -> list[MonitorEvent]:
        async with self._lock:
            return list(self.recent_events)

    async def get_summary(self) -> SystemSummary:
        async with self._lock:
            alerts = list(self.alerts.values())
            open_alerts = sum(1 for alert in alerts if alert.status == AlertStatus.open)
            acknowledged_alerts = sum(1 for alert in alerts if alert.status == AlertStatus.acknowledged)
            active_sessions = sum(1 for session in self.sessions.values() if session.status == SessionStatus.active)
            last_event_at = self.recent_events[0].created_at if self.recent_events else None
            return SystemSummary(
                total_patients=len(self.patients),
                total_devices=len(self.devices),
                active_sessions=active_sessions,
                open_alerts=open_alerts,
                acknowledged_alerts=acknowledged_alerts,
                last_event_at=last_event_at,
            )

    async def get_detector_config(self) -> DetectorConfig:
        async with self._lock:
            return self.detector_config

    async def update_detector_config(self, update: DetectorConfigUpdate) -> tuple[DetectorConfig, MonitorEvent]:
        async with self._lock:
            current = model_to_dict(self.detector_config)
            current.update({key: value for key, value in model_to_dict(update).items() if value is not None})
            self.detector_config = DetectorConfig(**current)
            event = self._new_event("detector.config.updated", model_to_dict(self.detector_config))
            return self.detector_config, event
