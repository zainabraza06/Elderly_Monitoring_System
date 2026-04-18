from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient

from .schemas import (
    AlertActionRequest,
    AlertRecord,
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
    SessionStopRequest,
    TelemetrySnapshot,
)
from .store import BackendStore, model_to_dict


logger = logging.getLogger(__name__)


def _model_to_json_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model_to_dict(model)


class MongoBackendStore(BackendStore):
    def __init__(
        self,
        recent_event_limit: int = 500,
        *,
        mongo_uri: str = "mongodb://localhost:27017",
        mongo_database: str = "elderly_monitoring",
    ) -> None:
        super().__init__(recent_event_limit=recent_event_limit)
        self._mongo_uri = mongo_uri
        self._mongo_database_name = mongo_database

        self._mongo_client: AsyncIOMotorClient | None = None
        self._mongo_db = None
        self._state_collection = None

        self._mongo_ready = False
        self._mongo_error: str | None = None

    def persistence_status(self) -> dict[str, Any]:
        return {
            "backend": "mongodb" if self._mongo_ready else "memory",
            "enabled": self._mongo_ready,
            "error": self._mongo_error,
            "uri": self._mongo_uri,
            "database": self._mongo_database_name,
        }

    async def startup(self) -> None:
        try:
            self._mongo_client = AsyncIOMotorClient(
                self._mongo_uri,
                serverSelectionTimeoutMS=3000,
            )
            await self._mongo_client.admin.command("ping")

            self._mongo_db = self._mongo_client[self._mongo_database_name]
            self._state_collection = self._mongo_db["app_state"]

            await self._hydrate_state()

            self._mongo_ready = True
            self._mongo_error = None
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._mongo_error = (
                "MongoDB persistence is unavailable. Falling back to in-memory mode. "
                f"Details: {exc}"
            )
            logger.warning(self._mongo_error)
            self._mongo_ready = False

            if self._mongo_client is not None:
                self._mongo_client.close()
            self._mongo_client = None
            self._mongo_db = None
            self._state_collection = None

    async def shutdown(self) -> None:
        if self._mongo_client is not None:
            self._mongo_client.close()

    async def _hydrate_state(self) -> None:
        if self._state_collection is None:
            return

        snapshot = await self._state_collection.find_one({"_id": "global"})
        if not snapshot:
            return

        detector_config_data = snapshot.get("detector_config")
        if isinstance(detector_config_data, dict) and detector_config_data:
            try:
                self.detector_config = DetectorConfig.model_validate(detector_config_data)
            except Exception:
                logger.exception("Failed to hydrate detector config from Mongo snapshot.")

        hydrated_patients: dict[str, PatientRecord] = {}
        for payload in snapshot.get("patients", []):
            try:
                patient = PatientRecord.model_validate(payload)
                hydrated_patients[patient.id] = patient
            except Exception:
                continue
        self.patients = hydrated_patients

        hydrated_devices: dict[str, DeviceRecord] = {}
        for payload in snapshot.get("devices", []):
            try:
                device = DeviceRecord.model_validate(payload)
                hydrated_devices[device.id] = device
            except Exception:
                continue
        self.devices = hydrated_devices

        hydrated_sessions: dict[str, SessionRecord] = {}
        for payload in snapshot.get("sessions", []):
            try:
                session = SessionRecord.model_validate(payload)
                hydrated_sessions[session.id] = session
            except Exception:
                continue
        self.sessions = hydrated_sessions

        hydrated_alerts: dict[str, AlertRecord] = {}
        for payload in snapshot.get("alerts", []):
            try:
                alert = AlertRecord.model_validate(payload)
                hydrated_alerts[alert.id] = alert
            except Exception:
                continue
        self.alerts = hydrated_alerts

        hydrated_live_status: dict[str, PatientLiveStatus] = {}
        for payload in snapshot.get("live_status", []):
            try:
                live = PatientLiveStatus.model_validate(payload)
                hydrated_live_status[live.patient_id] = live
            except Exception:
                continue
        self.live_status = hydrated_live_status

        hydrated_events: list[MonitorEvent] = []
        for payload in snapshot.get("recent_events", []):
            try:
                hydrated_events.append(MonitorEvent.model_validate(payload))
            except Exception:
                continue
        self.recent_events = deque(hydrated_events, maxlen=self.recent_events.maxlen)

        hydrated_telemetry_snapshots: dict[str, TelemetrySnapshot] = {}
        for payload in snapshot.get("telemetry_snapshots", []):
            try:
                telemetry = TelemetrySnapshot.model_validate(payload)
                hydrated_telemetry_snapshots[telemetry.patient_id] = telemetry
            except Exception:
                continue
        self.telemetry_snapshots = hydrated_telemetry_snapshots

        hydrated_recent_telemetry: list[TelemetrySnapshot] = []
        for payload in snapshot.get("recent_telemetry", []):
            try:
                hydrated_recent_telemetry.append(TelemetrySnapshot.model_validate(payload))
            except Exception:
                continue
        self.recent_telemetry = deque(hydrated_recent_telemetry, maxlen=self.recent_telemetry.maxlen)

    async def _persist_state(self) -> None:
        if not self._mongo_ready or self._state_collection is None:
            return

        async with self._lock:
            snapshot = {
                "detector_config": _model_to_json_dict(self.detector_config),
                "patients": [_model_to_json_dict(item) for item in self.patients.values()],
                "devices": [_model_to_json_dict(item) for item in self.devices.values()],
                "sessions": [_model_to_json_dict(item) for item in self.sessions.values()],
                "alerts": [_model_to_json_dict(item) for item in self.alerts.values()],
                "live_status": [_model_to_json_dict(item) for item in self.live_status.values()],
                "recent_events": [_model_to_json_dict(item) for item in self.recent_events],
                "telemetry_snapshots": [
                    _model_to_json_dict(item) for item in self.telemetry_snapshots.values()
                ],
                "recent_telemetry": [_model_to_json_dict(item) for item in self.recent_telemetry],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        try:
            await self._state_collection.update_one(
                {"_id": "global"},
                {"$set": snapshot},
                upsert=True,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._mongo_error = f"MongoDB persistence failed; switching to memory mode. Details: {exc}"
            logger.exception(self._mongo_error)
            self._mongo_ready = False

    async def create_patient(self, data: PatientCreate) -> tuple[PatientRecord, MonitorEvent]:
        result = await super().create_patient(data)
        await self._persist_state()
        return result

    async def create_device(self, data: DeviceCreate) -> tuple[DeviceRecord, MonitorEvent]:
        result = await super().create_device(data)
        await self._persist_state()
        return result

    async def start_session(self, data: SessionCreate) -> tuple[SessionRecord, list[MonitorEvent]]:
        result = await super().start_session(data)
        await self._persist_state()
        return result

    async def stop_session(self, session_id: str, data: SessionStopRequest) -> tuple[SessionRecord, list[MonitorEvent]]:
        result = await super().stop_session(session_id, data)
        await self._persist_state()
        return result

    async def ingest_detection(
        self,
        payload: SensorBatchIn,
        detection: Any,
    ) -> tuple[SensorBatchOut, list[MonitorEvent]]:
        result = await super().ingest_detection(payload, detection)
        await self._persist_state()
        return result

    async def create_manual_alert(self, data: ManualAlertCreate) -> tuple[AlertRecord, list[MonitorEvent]]:
        result = await super().create_manual_alert(data)
        await self._persist_state()
        return result

    async def acknowledge_alert(self, alert_id: str, data: AlertActionRequest) -> tuple[AlertRecord, list[MonitorEvent]]:
        result = await super().acknowledge_alert(alert_id, data)
        await self._persist_state()
        return result

    async def resolve_alert(self, alert_id: str, data: AlertActionRequest) -> tuple[AlertRecord, list[MonitorEvent]]:
        result = await super().resolve_alert(alert_id, data)
        await self._persist_state()
        return result

    async def update_detector_config(self, update: DetectorConfigUpdate) -> tuple[DetectorConfig, MonitorEvent]:
        result = await super().update_detector_config(update)
        await self._persist_state()
        return result
