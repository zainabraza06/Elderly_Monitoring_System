from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4
import logging

from fastapi import Depends, FastAPI, HTTPException, Header, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth_service import AuthContext, AuthService
from .config import get_settings
from .detection import RealtimeDetector
from .mongo_store import MongoBackendStore
from .realtime import MonitorConnectionManager
from .schemas import (
    APIErrorResponse,
    AuthLoginRequest,
    AuthSessionResponse,
    AuthSignupPatientRequest,
    AuthSwitchRoleRequest,
    AuthUserProfile,
    AlertActionRequest,
    AlertStatus,
    DetectorConfigUpdate,
    DeviceCreate,
    HealthResponse,
    ManualAlertCreate,
    PatientCreate,
    SensorBatchIn,
    SessionCreate,
    SessionStopRequest,
    UserRole,
)
settings = get_settings()
store = MongoBackendStore(
    recent_event_limit=settings.recent_event_limit,
    mongo_uri=settings.mongo_uri,
    mongo_database=settings.mongo_database,
)
detector = RealtimeDetector(artifacts_dir=settings.offline_artifacts_dir)
auth_service = AuthService(
    token_ttl_seconds=settings.auth_token_ttl_seconds,
    secret=settings.auth_secret,
    mongo_uri=settings.mongo_uri,
    mongo_database=settings.mongo_database,
)
monitor_manager = MonitorConnectionManager()
logger = logging.getLogger(__name__)
startup_logger = logging.getLogger("uvicorn.error")


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
    async def startup_services() -> None:
        await store.startup()
        await auth_service.startup()

        persistence = store.persistence_status()
        if persistence.get("enabled"):
            logger.info("Persistence backend: %s", persistence.get("backend"))
        else:
            logger.warning(
                "Persistence backend fallback in use (%s). %s",
                persistence.get("backend"),
                persistence.get("error"),
            )

        auth_status = auth_service.status()
        if auth_status.get("enabled"):
            logger.info("Auth backend: %s", auth_status.get("backend"))
        else:
            logger.warning(
                "Auth backend fallback in use (%s). %s",
                auth_status.get("backend"),
                auth_status.get("error"),
            )

        if persistence.get("enabled") and auth_status.get("enabled"):
            startup_logger.info(
                "DB connected: backend=%s uri=%s database=%s",
                persistence.get("backend"),
                persistence.get("uri"),
                persistence.get("database"),
            )

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

    @app.on_event("shutdown")
    async def shutdown_services() -> None:
        await store.shutdown()
        await auth_service.shutdown()

    async def require_auth_context(
        authorization: str | None = Header(default=None),
    ) -> AuthContext:
        try:
            return await auth_service.authenticate_header(authorization)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def assert_allowed_roles(auth: AuthContext, allowed: set[UserRole]) -> None:
        if auth.selected_role not in allowed:
            allowed_labels = ", ".join(role.value for role in sorted(allowed, key=lambda item: item.value))
            raise HTTPException(
                status_code=403,
                detail=f"This action requires role: {allowed_labels}.",
            )

    def assert_patient_scope(auth: AuthContext, patient_id: str) -> None:
        if auth.patient_id is not None and auth.patient_id != patient_id:
            raise HTTPException(
                status_code=403,
                detail="You do not have access to this patient scope.",
            )

    async def assert_alert_scope(auth: AuthContext, alert_id: str) -> None:
        if auth.patient_id is None:
            return
        scoped_alerts = await store.list_alerts(patient_id=auth.patient_id)
        if not any(alert.id == alert_id for alert in scoped_alerts):
            raise HTTPException(status_code=403, detail="You do not have access to this alert.")

    @app.post(
        f"{settings.api_prefix}/auth/signup/patient",
        response_model=AuthSessionResponse,
        tags=["auth"],
    )
    async def signup_patient_account(payload: AuthSignupPatientRequest):
        await auth_service.ensure_email_available(payload.email)

        patient, event = await store.create_patient(
            PatientCreate(
                full_name=payload.full_name,
                age=payload.age,
                room_label=payload.room_label,
            )
        )
        await monitor_manager.broadcast(event)
        return await auth_service.register_patient_user(payload, patient_id=patient.id)

    @app.post(
        f"{settings.api_prefix}/auth/login",
        response_model=AuthSessionResponse,
        tags=["auth"],
    )
    async def login(payload: AuthLoginRequest):
        try:
            return await auth_service.login(payload)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.get(
        f"{settings.api_prefix}/auth/me",
        response_model=AuthUserProfile,
        tags=["auth"],
    )
    async def auth_me(auth: AuthContext = Depends(require_auth_context)):
        return await auth_service.get_user_profile(auth.user_id)

    @app.post(
        f"{settings.api_prefix}/auth/roles/caregiver",
        response_model=AuthUserProfile,
        tags=["auth"],
    )
    async def enable_caregiver_role(auth: AuthContext = Depends(require_auth_context)):
        return await auth_service.enable_caregiver_role(user_id=auth.user_id)

    @app.post(
        f"{settings.api_prefix}/auth/switch-role",
        response_model=AuthSessionResponse,
        tags=["auth"],
    )
    async def switch_role(
        payload: AuthSwitchRoleRequest,
        auth: AuthContext = Depends(require_auth_context),
    ):
        return await auth_service.switch_role(user_id=auth.user_id, role=payload.role)

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

    @app.get(f"{settings.api_prefix}/summary", tags=["system"])
    async def get_summary(auth: AuthContext = Depends(require_auth_context)):
        return await store.get_summary()

    @app.get(f"{settings.api_prefix}/events/recent", tags=["monitoring"])
    async def recent_events(auth: AuthContext = Depends(require_auth_context)):
        events = await store.get_recent_events()
        if auth.patient_id is None:
            return events

        scoped_events = []
        for event in events:
            payload = event.payload or {}
            event_patient_id = payload.get("patient_id")
            if event_patient_id is None or str(event_patient_id) == auth.patient_id:
                scoped_events.append(event)
        return scoped_events

    @app.get(f"{settings.api_prefix}/detector/config", tags=["detector"])
    async def get_detector_config(auth: AuthContext = Depends(require_auth_context)):
        return await store.get_detector_config()

    @app.get(f"{settings.api_prefix}/detector/status", tags=["detector"])
    async def get_detector_status(auth: AuthContext = Depends(require_auth_context)):
        return detector.status()

    @app.put(f"{settings.api_prefix}/detector/config", tags=["detector"])
    async def update_detector_config(
        payload: DetectorConfigUpdate,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_allowed_roles(auth, {UserRole.caregiver})
        config, event = await store.update_detector_config(payload)
        await monitor_manager.broadcast(event)
        return config

    @app.post(f"{settings.api_prefix}/patients", tags=["patients"])
    async def create_patient(
        payload: PatientCreate,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_allowed_roles(auth, {UserRole.patient})
        patient, event = await store.create_patient(payload)
        await monitor_manager.broadcast(event)
        return patient

    @app.get(f"{settings.api_prefix}/patients", tags=["patients"])
    async def list_patients(auth: AuthContext = Depends(require_auth_context)):
        if auth.patient_id is None:
            return await store.list_patients()

        patient = await store.get_patient(auth.patient_id)
        if patient is None:
            return []
        return [patient]

    @app.get(f"{settings.api_prefix}/monitor/patients/live", tags=["monitoring"])
    async def list_live_statuses(auth: AuthContext = Depends(require_auth_context)):
        if auth.patient_id is None:
            return await store.list_live_statuses()

        status = await store.get_live_status_for_patient(auth.patient_id)
        if status is None:
            return []
        return [status]

    @app.get(f"{settings.api_prefix}/monitor/patients/{{patient_id}}/live", tags=["monitoring"])
    async def get_patient_live_status(
        patient_id: str,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_patient_scope(auth, patient_id)
        status = await store.get_live_status_for_patient(patient_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Live status not found for patient.")
        return status

    @app.get(f"{settings.api_prefix}/monitor/telemetry/recent", tags=["monitoring"])
    async def recent_telemetry(
        limit: int = Query(default=20, ge=1, le=100),
        auth: AuthContext = Depends(require_auth_context),
    ):
        snapshots = await store.list_recent_telemetry(limit=limit)
        if auth.patient_id is None:
            return snapshots
        return [snapshot for snapshot in snapshots if snapshot.patient_id == auth.patient_id]

    @app.get(f"{settings.api_prefix}/monitor/patients/{{patient_id}}/telemetry", tags=["monitoring"])
    async def patient_telemetry(
        patient_id: str,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_patient_scope(auth, patient_id)
        telemetry = await store.get_telemetry_for_patient(patient_id)
        if telemetry is None:
            raise HTTPException(status_code=404, detail="Telemetry not found for patient.")
        return telemetry

    @app.get(f"{settings.api_prefix}/patients/{{patient_id}}", tags=["patients"])
    async def get_patient(
        patient_id: str,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_patient_scope(auth, patient_id)
        patient = await store.get_patient(patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found.")
        return patient

    @app.post(f"{settings.api_prefix}/devices", tags=["devices"])
    async def create_device(
        payload: DeviceCreate,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_allowed_roles(auth, {UserRole.patient})
        device, event = await store.create_device(payload)
        await monitor_manager.broadcast(event)
        return device

    @app.get(f"{settings.api_prefix}/devices", tags=["devices"])
    async def list_devices(auth: AuthContext = Depends(require_auth_context)):
        return await store.list_devices()

    @app.get(f"{settings.api_prefix}/devices/{{device_id}}", tags=["devices"])
    async def get_device(
        device_id: str,
        auth: AuthContext = Depends(require_auth_context),
    ):
        device = await store.get_device(device_id)
        if device is None:
            raise HTTPException(status_code=404, detail="Device not found.")
        return device

    @app.post(f"{settings.api_prefix}/sessions", tags=["sessions"])
    async def start_session(
        payload: SessionCreate,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_allowed_roles(auth, {UserRole.patient})
        assert_patient_scope(auth, payload.patient_id)
        try:
            session, events = await store.start_session(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for event in events:
            await monitor_manager.broadcast(event)
        return session

    @app.get(f"{settings.api_prefix}/sessions", tags=["sessions"])
    async def list_sessions(
        active_only: bool = Query(default=False),
        auth: AuthContext = Depends(require_auth_context),
    ):
        sessions = await store.list_sessions(active_only=active_only)
        if auth.patient_id is None:
            return sessions
        return [session for session in sessions if session.patient_id == auth.patient_id]

    @app.get(f"{settings.api_prefix}/sessions/{{session_id}}", tags=["sessions"])
    async def get_session(
        session_id: str,
        auth: AuthContext = Depends(require_auth_context),
    ):
        session = await store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        assert_patient_scope(auth, session.patient_id)
        return session

    @app.post(f"{settings.api_prefix}/sessions/{{session_id}}/stop", tags=["sessions"])
    async def stop_session(
        session_id: str,
        payload: SessionStopRequest,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_allowed_roles(auth, {UserRole.patient})

        current_session = await store.get_session(session_id)
        if current_session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        assert_patient_scope(auth, current_session.patient_id)

        try:
            session, events = await store.stop_session(session_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        for event in events:
            await monitor_manager.broadcast(event)
        return session

    @app.post(f"{settings.api_prefix}/ingest/live", tags=["ingestion"])
    async def ingest_live_sensor_batch(
        payload: SensorBatchIn,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_allowed_roles(auth, {UserRole.patient})
        assert_patient_scope(auth, payload.patient_id)
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
        auth: AuthContext = Depends(require_auth_context),
    ):
        if auth.patient_id is not None:
            if patient_id is not None and patient_id != auth.patient_id:
                raise HTTPException(status_code=403, detail="You do not have access to this patient scope.")
            patient_id = auth.patient_id
        return await store.list_alerts(status=status, patient_id=patient_id)

    @app.post(f"{settings.api_prefix}/alerts/manual", tags=["alerts"])
    async def create_manual_alert(
        payload: ManualAlertCreate,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_allowed_roles(auth, {UserRole.patient, UserRole.caregiver})
        assert_patient_scope(auth, payload.patient_id)
        try:
            alert, events = await store.create_manual_alert(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for event in events:
            await monitor_manager.broadcast(event)
        return alert

    @app.post(f"{settings.api_prefix}/alerts/{{alert_id}}/acknowledge", tags=["alerts"])
    async def acknowledge_alert(
        alert_id: str,
        payload: AlertActionRequest,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_allowed_roles(auth, {UserRole.patient, UserRole.caregiver})
        await assert_alert_scope(auth, alert_id)
        try:
            alert, events = await store.acknowledge_alert(alert_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        for event in events:
            await monitor_manager.broadcast(event)
        return alert

    @app.post(f"{settings.api_prefix}/alerts/{{alert_id}}/resolve", tags=["alerts"])
    async def resolve_alert(
        alert_id: str,
        payload: AlertActionRequest,
        auth: AuthContext = Depends(require_auth_context),
    ):
        assert_allowed_roles(auth, {UserRole.patient, UserRole.caregiver})
        await assert_alert_scope(auth, alert_id)
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
