from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4
import logging

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .detection import RealtimeDetector
from .realtime import MonitorConnectionManager
from .schemas import (
    APIErrorResponse,
    AlertActionRequest,
    AlertStatus,
    CaregiverLoginRequest,
    CaregiverSignupRequest,
    DetectorConfigUpdate,
    DeviceCreate,
    HealthResponse,
    PatientCredentialCreate,
    PatientLoginRequest,
    ManualAlertCreate,
    PatientCreate,
    SensorBatchIn,
    SessionCreate,
    SessionStopRequest,
)
from .store import BackendStore


settings = get_settings()
store = BackendStore(recent_event_limit=settings.recent_event_limit)
detector = RealtimeDetector(artifacts_dir=settings.offline_artifacts_dir)
monitor_manager = MonitorConnectionManager()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="FastAPI backend for live elderly monitoring, detection, and alert workflows.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_metadata(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or f"req_{uuid4().hex[:12]}"
        request.state.request_id = request_id
        started_at = perf_counter()

        response = await call_next(request)
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    @app.on_event("startup")
    async def report_detector_runtime() -> None:
        status = detector.status()
        if status.get("runtime_loaded"):
            logger.info(
                "Detector runtime loaded in %s mode from %s",
                status.get("mode"),
                status.get("artifact_path"),
            )
            return

        logger.warning(
            "Detector running in %s fallback. %s Expected bundle at %s",
            status.get("mode"),
            status.get("reason"),
            status.get("bundle_path"),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", f"req_{uuid4().hex[:12]}")
        error = APIErrorResponse(
            code="validation_error",
            message="Request validation failed.",
            trace_id=request_id,
            timestamp=datetime.now(timezone.utc),
            details=exc.errors(),
        )
        return JSONResponse(status_code=422, content=error.model_dump(mode="json"))

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", f"req_{uuid4().hex[:12]}")
        error = APIErrorResponse(
            code="http_error",
            message=str(exc.detail),
            trace_id=request_id,
            timestamp=datetime.now(timezone.utc),
            details={"status_code": exc.status_code},
        )
        return JSONResponse(status_code=exc.status_code, content=error.model_dump(mode="json"))

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError):
        request_id = getattr(request.state, "request_id", f"req_{uuid4().hex[:12]}")
        error = APIErrorResponse(
            code="domain_error",
            message=str(exc),
            trace_id=request_id,
            timestamp=datetime.now(timezone.utc),
        )
        return JSONResponse(status_code=400, content=error.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", f"req_{uuid4().hex[:12]}")
        logger.exception("Unhandled backend exception", extra={"request_id": request_id})
        error = APIErrorResponse(
            code="internal_error",
            message="Unexpected internal server error.",
            trace_id=request_id,
            timestamp=datetime.now(timezone.utc),
        )
        return JSONResponse(status_code=500, content=error.model_dump(mode="json"))

    @app.get("/", tags=["root"])
    async def root() -> dict[str, str]:
        return {
            "message": "Elderly Monitoring Backend is running.",
            "docs": "/docs",
            "api_prefix": settings.api_prefix,
            "websocket": "/ws/monitor",
        }

    @app.get(f"{settings.api_prefix}/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=settings.app_name,
            timestamp=datetime.now(timezone.utc),
        )

    @app.post(f"{settings.api_prefix}/auth/caregiver/signup", tags=["auth"])
    async def caregiver_signup(payload: CaregiverSignupRequest):
        try:
            auth = await store.signup_caregiver(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return auth

    @app.post(f"{settings.api_prefix}/auth/caregiver/login", tags=["auth"])
    async def caregiver_login(payload: CaregiverLoginRequest):
        try:
            auth = await store.login_caregiver(payload)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return auth

    @app.post(f"{settings.api_prefix}/auth/caregiver/patient-credentials", tags=["auth"])
    async def caregiver_generate_patient_credentials(payload: PatientCredentialCreate):
        try:
            generated = await store.generate_patient_credentials(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return generated

    @app.post(f"{settings.api_prefix}/auth/patient/login", tags=["auth"])
    async def patient_login(payload: PatientLoginRequest):
        try:
            auth = await store.login_patient(payload)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return auth

    @app.get(f"{settings.api_prefix}/summary", tags=["system"])
    async def get_summary():
        return await store.get_summary()

    @app.get(f"{settings.api_prefix}/events/recent", tags=["monitoring"])
    async def recent_events():
        return await store.get_recent_events()

    @app.get(f"{settings.api_prefix}/detector/config", tags=["detector"])
    async def get_detector_config():
        return await store.get_detector_config()

    @app.get(f"{settings.api_prefix}/detector/status", tags=["detector"])
    async def get_detector_status():
        return detector.status()

    @app.put(f"{settings.api_prefix}/detector/config", tags=["detector"])
    async def update_detector_config(payload: DetectorConfigUpdate):
        config, event = await store.update_detector_config(payload)
        await monitor_manager.broadcast(event)
        return config

    @app.post(f"{settings.api_prefix}/patients", tags=["patients"])
    async def create_patient(payload: PatientCreate):
        patient, event = await store.create_patient(payload)
        await monitor_manager.broadcast(event)
        return patient

    @app.get(f"{settings.api_prefix}/patients", tags=["patients"])
    async def list_patients():
        return await store.list_patients()

    @app.get(f"{settings.api_prefix}/monitor/patients/live", tags=["monitoring"])
    async def list_live_statuses():
        return await store.list_live_statuses()

    @app.get(f"{settings.api_prefix}/monitor/patients/{{patient_id}}/live", tags=["monitoring"])
    async def get_patient_live_status(patient_id: str):
        status = await store.get_live_status_for_patient(patient_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Live status not found for patient.")
        return status

    @app.get(f"{settings.api_prefix}/monitor/telemetry/recent", tags=["monitoring"])
    async def recent_telemetry(limit: int = Query(default=20, ge=1, le=100)):
        return await store.list_recent_telemetry(limit=limit)

    @app.get(f"{settings.api_prefix}/monitor/patients/{{patient_id}}/telemetry", tags=["monitoring"])
    async def patient_telemetry(patient_id: str):
        telemetry = await store.get_telemetry_for_patient(patient_id)
        if telemetry is None:
            raise HTTPException(status_code=404, detail="Telemetry not found for patient.")
        return telemetry

    @app.get(f"{settings.api_prefix}/patients/{{patient_id}}", tags=["patients"])
    async def get_patient(patient_id: str):
        patient = await store.get_patient(patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found.")
        return patient

    @app.post(f"{settings.api_prefix}/devices", tags=["devices"])
    async def create_device(payload: DeviceCreate):
        device, event = await store.create_device(payload)
        await monitor_manager.broadcast(event)
        return device

    @app.get(f"{settings.api_prefix}/devices", tags=["devices"])
    async def list_devices():
        return await store.list_devices()

    @app.get(f"{settings.api_prefix}/devices/{{device_id}}", tags=["devices"])
    async def get_device(device_id: str):
        device = await store.get_device(device_id)
        if device is None:
            raise HTTPException(status_code=404, detail="Device not found.")
        return device

    @app.post(f"{settings.api_prefix}/sessions", tags=["sessions"])
    async def start_session(payload: SessionCreate):
        try:
            session, events = await store.start_session(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for event in events:
            await monitor_manager.broadcast(event)
        return session

    @app.get(f"{settings.api_prefix}/sessions", tags=["sessions"])
    async def list_sessions(active_only: bool = Query(default=False)):
        return await store.list_sessions(active_only=active_only)

    @app.get(f"{settings.api_prefix}/sessions/{{session_id}}", tags=["sessions"])
    async def get_session(session_id: str):
        session = await store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session

    @app.post(f"{settings.api_prefix}/sessions/{{session_id}}/stop", tags=["sessions"])
    async def stop_session(session_id: str, payload: SessionStopRequest):
        try:
            session, events = await store.stop_session(session_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        for event in events:
            await monitor_manager.broadcast(event)
        return session

    @app.post(f"{settings.api_prefix}/ingest/live", tags=["ingestion"])
    async def ingest_live_sensor_batch(payload: SensorBatchIn):
        config = await store.get_detector_config()
        detection = detector.analyze(payload, config)

        try:
            response, events = await store.ingest_detection(
                payload=payload,
                detection=detection,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for event in events:
            await monitor_manager.broadcast(event)
        return response

    @app.get(f"{settings.api_prefix}/alerts", tags=["alerts"])
    async def list_alerts(
        status: AlertStatus | None = Query(default=None),
        patient_id: str | None = Query(default=None),
    ):
        return await store.list_alerts(status=status, patient_id=patient_id)

    @app.post(f"{settings.api_prefix}/alerts/manual", tags=["alerts"])
    async def create_manual_alert(payload: ManualAlertCreate):
        try:
            alert, events = await store.create_manual_alert(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for event in events:
            await monitor_manager.broadcast(event)
        return alert

    @app.post(f"{settings.api_prefix}/alerts/{{alert_id}}/acknowledge", tags=["alerts"])
    async def acknowledge_alert(alert_id: str, payload: AlertActionRequest):
        try:
            alert, events = await store.acknowledge_alert(alert_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for event in events:
            await monitor_manager.broadcast(event)
        return alert

    @app.post(f"{settings.api_prefix}/alerts/{{alert_id}}/resolve", tags=["alerts"])
    async def resolve_alert(alert_id: str, payload: AlertActionRequest):
        try:
            alert, events = await store.resolve_alert(alert_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for event in events:
            await monitor_manager.broadcast(event)
        return alert

    @app.websocket("/ws/monitor")
    async def websocket_monitor(websocket: WebSocket):
        await monitor_manager.connect(websocket)
        try:
            await websocket.send_json(
                {
                    "type": "connection.ready",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "payload": {"message": "Connected to live monitoring stream."},
                }
            )
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await monitor_manager.disconnect(websocket)
        except Exception:
            await monitor_manager.disconnect(websocket)

    return app


app = create_app()
