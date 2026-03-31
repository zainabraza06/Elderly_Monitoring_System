# FastAPI Backend

This backend is the real-time API server for the Elderly Monitoring System.

It provides:

- patient management
- device registration
- monitoring session lifecycle
- live mobile sensor ingestion
- recent telemetry snapshots for dashboard monitoring
- rule-based fall / high-risk detection
- alert creation, acknowledgement, and resolution
- WebSocket monitoring feed for the web dashboard

## Run

```bash
python -m pip install -r flask_backend/requirements.txt
python -m uvicorn flask_backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Main API Groups

- `/api/v1/health`
- `/api/v1/patients`
- `/api/v1/devices`
- `/api/v1/sessions`
- `/api/v1/monitor/patients/live`
- `/api/v1/monitor/patients/{patient_id}/telemetry`
- `/api/v1/monitor/telemetry/recent`
- `/api/v1/ingest/live`
- `/api/v1/alerts`
- `/api/v1/events/recent`
- `/api/v1/detector/config`
- `/api/v1/summary`
- `/ws/monitor`

## Notes

- The current detector is rule-based so the mobile and web apps can be built immediately.
- The detection service is isolated in `flask_backend/app/detection.py`, so it can later be replaced with a trained ML model service without changing the API contract.
- State is currently in-memory for rapid development. Swap `store.py` to a database-backed implementation when persistence is needed.
- The standalone monitoring dashboard lives in the `web_frontend/` Next.js app and uses the REST endpoints for initial state plus `/ws/monitor` for live updates.

## Validation And Error Contract

- Request validation is enforced with Pydantic schemas and field constraints.
- Unknown fields are rejected for input payloads (`extra=forbid`) to avoid silent contract drift.
- Error responses use a consistent shape:

```json
{
	"code": "validation_error",
	"message": "Request validation failed.",
	"trace_id": "req_abc123...",
	"timestamp": "2026-03-31T18:11:29.123456+00:00",
	"details": []
}
```

- `X-Request-Id` and `X-Process-Time-Ms` headers are returned on API responses for diagnostics.

## Stress Testing

Use the built-in stress test tool to capture throughput and latency percentiles.

```bash
python flask_backend/scripts/stress_test.py --base-url http://127.0.0.1:8000 --requests 300 --concurrency 30
```

Outputs are written to `flask_backend/stress_reports/` as JSON and Markdown reports.
