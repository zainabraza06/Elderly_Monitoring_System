# Elderly Monitoring System API Documentation

## 1. Overview

This document is the complete API reference for the Elderly Monitoring System backend.

- Framework: FastAPI
- API version prefix: `/api/v1`
- Transport: HTTP/1.1 + WebSocket
- Data format: JSON
- Time format: ISO 8601 UTC timestamps
- Runtime docs: `/docs`, `/redoc`, `/openapi.json`

The API supports:

- patient and device onboarding
- monitoring session lifecycle
- live sensor ingestion and risk scoring
- alert management workflow
- dashboard summary and telemetry APIs
- real-time event streaming over WebSocket

## 2. Base URLs

Use your deployed backend host plus the API prefix.

Examples:

- Local development: `http://127.0.0.1:8000`
- Local API base: `http://127.0.0.1:8000/api/v1`
- WebSocket monitor: `ws://127.0.0.1:8000/ws/monitor`

## 3. Authentication and Authorization

Current version does not enforce API authentication.

- No bearer token required
- No API key required
- No role-based access control enforced at endpoint level

Important: Add auth before production use in untrusted environments.

## 4. Global API Conventions

### 4.1 Request headers

Recommended request headers:

- `Content-Type: application/json` for requests with body
- `Accept: application/json`
- Optional trace header: `X-Request-Id: <client_generated_id>`

If `X-Request-Id` is not supplied, the backend generates one.

### 4.2 Response headers

Most HTTP responses include:

- `X-Request-Id`: request correlation ID
- `X-Process-Time-Ms`: server processing time in milliseconds

### 4.3 Identifier patterns

The backend creates IDs with prefixes:

- patient: `pat_<random>`
- device: `dev_<random>`
- session: `ses_<random>`
- alert: `alt_<random>`
- event: `evt_<random>`
- request trace: `req_<random>`

### 4.4 Persistence model

Current store is in-memory.

- Data is reset when the backend process restarts.
- IDs and history are not durable across restarts.

## 5. Error Model

### 5.1 Standard error response

All handled API errors follow one envelope:

```json
{
  "code": "validation_error",
  "message": "Request validation failed.",
  "trace_id": "req_1a2b3c4d5e6f",
  "timestamp": "2026-03-31T18:11:29.123456+00:00",
  "details": []
}
```

### 5.2 Error fields

- `code` string:
  - `validation_error`
  - `http_error`
  - `domain_error`
  - `internal_error`
- `message` string: human-readable explanation
- `trace_id` string: correlates with `X-Request-Id`
- `timestamp` string: UTC timestamp
- `details` any/null: structured diagnostics when available

### 5.3 Typical status mappings

- `422`: payload/query/path validation failure
- `400`: domain or business-rule violation
- `404`: resource not found
- `500`: unexpected server error

## 6. Enums and Shared Types

### 6.1 Enum values

- `AlertSeverity`: `low`, `medium`, `high_risk`, `fall_detected`
- `AlertStatus`: `open`, `acknowledged`, `resolved`
- `SessionStatus`: `active`, `stopped`
- `AccelerationUnit`: `g`, `m_s2`
- `GyroscopeUnit`: `dps`, `rad_s`

### 6.2 Core response objects

#### HealthResponse

```json
{
  "status": "ok",
  "app_name": "Elderly Monitoring Backend",
  "timestamp": "2026-03-31T20:30:00.000000+00:00"
}
```

#### MonitorEvent

```json
{
  "id": "evt_a1b2c3d4e5f6",
  "type": "patient.live.updated",
  "created_at": "2026-03-31T20:30:00.000000+00:00",
  "payload": {}
}
```

#### SystemSummary

```json
{
  "total_patients": 4,
  "total_devices": 3,
  "active_sessions": 2,
  "open_alerts": 1,
  "acknowledged_alerts": 1,
  "last_event_at": "2026-03-31T20:30:00.000000+00:00"
}
```

## 7. Request Models and Validation

This section describes every request body accepted by the API.

### 7.1 PatientCreate

```json
{
  "full_name": "Amina Khan",
  "age": 78,
  "room_label": "Room-12A",
  "emergency_contact": "+92-300-1234567",
  "notes": "Uses walker"
}
```

Validation:

- `full_name`: required, 1..120, trimmed, cannot be blank
- `age`: optional, 0..130
- `room_label`: optional, max 80
- `emergency_contact`: optional, max 120
- `notes`: optional, max 500
- unknown fields are rejected

### 7.2 DeviceCreate

```json
{
  "label": "NurseStation-Phone-1",
  "platform": "android",
  "owner_name": "Ward B"
}
```

Validation:

- `label`: required, 1..80, trimmed, cannot be blank
- `platform`: optional/default `mobile_web`, max 60
- `owner_name`: optional, max 80
- unknown fields are rejected

### 7.3 SessionCreate

```json
{
  "patient_id": "pat_1234567890",
  "device_id": "dev_1234567890",
  "started_by": "operator_1",
  "sample_rate_hz": 50.0,
  "notes": "Night shift monitoring"
}
```

Validation:

- `patient_id`: required, 1..64, trimmed, cannot be blank
- `device_id`: required, 1..64, trimmed, cannot be blank
- `started_by`: optional/default `system`, max 80, trimmed, cannot be blank
- `sample_rate_hz`: optional/default 50.0, >0 and <=400
- `notes`: optional, max 200
- unknown fields are rejected

### 7.4 SessionStopRequest

```json
{
  "stopped_by": "operator_1",
  "note": "Vitals stable"
}
```

Validation:

- `stopped_by`: optional/default `system`, max 80, trimmed, cannot be blank
- `note`: optional, max 200
- unknown fields are rejected

### 7.5 DetectorConfigUpdate

```json
{
  "min_samples": 20,
  "impact_threshold_g": 2.3,
  "jerk_threshold_g_per_s": 12.0,
  "gyro_threshold_dps": 220.0,
  "stillness_acc_delta_g": 0.18,
  "stillness_gyro_threshold_dps": 35.0,
  "medium_risk_score": 0.35,
  "high_risk_score": 0.58,
  "fall_score": 0.8,
  "alert_cooldown_seconds": 25
}
```

Validation:

- all fields optional (partial updates allowed)
- numeric bounds enforced
- score fields must stay within 0..1
- unknown fields are rejected

### 7.6 SensorSample

```json
{
  "timestamp_ms": 1711910400123,
  "acc_x": 0.12,
  "acc_y": -0.04,
  "acc_z": 9.81,
  "gyro_x": 2.1,
  "gyro_y": 1.5,
  "gyro_z": 0.7
}
```

Validation:

- `timestamp_ms`: optional, >=0
- acceleration fields: each in [-500, 500]
- gyroscope fields: each in [-5000, 5000]
- unknown fields are rejected

### 7.7 SensorBatchIn

```json
{
  "patient_id": "pat_1234567890",
  "device_id": "dev_1234567890",
  "session_id": "ses_1234567890",
  "source": "mobile",
  "sampling_rate_hz": 50.0,
  "acceleration_unit": "m_s2",
  "gyroscope_unit": "dps",
  "battery_level": 86,
  "samples": [
    {
      "timestamp_ms": 1711910400123,
      "acc_x": 0.01,
      "acc_y": -0.02,
      "acc_z": 9.82,
      "gyro_x": 1.1,
      "gyro_y": 0.3,
      "gyro_z": 0.6
    },
    {
      "timestamp_ms": 1711910400143,
      "acc_x": 0.03,
      "acc_y": -0.01,
      "acc_z": 9.78,
      "gyro_x": 1.5,
      "gyro_y": 0.2,
      "gyro_z": 0.5
    }
  ]
}
```

Validation:

- `patient_id`, `device_id`, `session_id`, `source`: required, non-blank, trimmed
- `sampling_rate_hz`: >0 and <=400
- `acceleration_unit`: `g` or `m_s2`
- `gyroscope_unit`: `dps` or `rad_s`
- `battery_level`: optional, 0..100
- `samples`: required, 1..4096 entries
- if sample timestamps are present, they must be monotonic non-decreasing
- unknown fields are rejected

### 7.8 ManualAlertCreate

```json
{
  "patient_id": "pat_1234567890",
  "session_id": "ses_1234567890",
  "device_id": "dev_1234567890",
  "severity": "high_risk",
  "message": "Patient requested urgent assistance",
  "actor": "nurse_1",
  "note": "Bedside call button"
}
```

Validation:

- `patient_id`: required, 1..64, trimmed
- `session_id`: optional, max 64
- `device_id`: optional, max 64
- `severity`: enum, default `high_risk`
- `message`: required, 1..240, trimmed
- `actor`: required/default `operator`, max 80, trimmed
- `note`: optional, max 200
- unknown fields are rejected

### 7.9 AlertActionRequest

```json
{
  "actor": "nurse_1",
  "note": "Family informed"
}
```

Validation:

- `actor`: required/default `operator`, max 80, trimmed, cannot be blank
- `note`: optional, max 200
- unknown fields are rejected

## 8. Endpoint Reference

All endpoints below are relative to backend host unless absolute path is shown.

---

## 8.1 Root and Documentation Endpoints

### GET /

Service discovery endpoint.

- Purpose: quick backend reachability and link hints
- Request body: none
- Success response: `200 OK`

```json
{
  "message": "Elderly Monitoring Backend is running.",
  "docs": "/docs",
  "api_prefix": "/api/v1",
  "websocket": "/ws/monitor"
}
```

### GET /docs

Swagger UI endpoint provided by FastAPI.

- Purpose: interactive API testing
- Success response: `200 OK` (HTML)

### GET /redoc

ReDoc endpoint provided by FastAPI.

- Purpose: static API documentation view
- Success response: `200 OK` (HTML)

### GET /openapi.json

OpenAPI schema endpoint provided by FastAPI.

- Purpose: machine-readable API contract
- Success response: `200 OK` (JSON)

---

## 8.2 System Endpoints

### GET /api/v1/health

Health and identity check.

- Request body: none
- Success response: `200 OK` (`HealthResponse`)

```json
{
  "status": "ok",
  "app_name": "Elderly Monitoring Backend",
  "timestamp": "2026-03-31T20:30:00.000000+00:00"
}
```

### GET /api/v1/summary

Operational aggregate summary.

- Request body: none
- Success response: `200 OK` (`SystemSummary`)

```json
{
  "total_patients": 1,
  "total_devices": 1,
  "active_sessions": 1,
  "open_alerts": 0,
  "acknowledged_alerts": 0,
  "last_event_at": "2026-03-31T20:30:00.000000+00:00"
}
```

### GET /api/v1/events/recent

Recent backend events in reverse chronological order.

- Request body: none
- Success response: `200 OK`

```json
[
  {
    "id": "evt_1234567890ab",
    "type": "patient.live.updated",
    "created_at": "2026-03-31T20:30:10.000000+00:00",
    "payload": {
      "patient_id": "pat_1234567890"
    }
  }
]
```

Notes:

- Capacity is bounded by server setting `recent_event_limit` (default 500).

---

## 8.3 Detector Endpoints

### GET /api/v1/detector/config

Fetch current detector threshold configuration.

- Request body: none
- Success response: `200 OK`

```json
{
  "min_samples": 20,
  "impact_threshold_g": 2.3,
  "jerk_threshold_g_per_s": 12,
  "gyro_threshold_dps": 220,
  "stillness_acc_delta_g": 0.18,
  "stillness_gyro_threshold_dps": 35,
  "medium_risk_score": 0.35,
  "high_risk_score": 0.58,
  "fall_score": 0.8,
  "alert_cooldown_seconds": 25
}
```

### PUT /api/v1/detector/config

Partially update detector thresholds.

- Request body: `DetectorConfigUpdate`
- Success response: `200 OK` (updated full config)

Example request:

```json
{
  "impact_threshold_g": 2.1,
  "high_risk_score": 0.55
}
```

Example response:

```json
{
  "min_samples": 20,
  "impact_threshold_g": 2.1,
  "jerk_threshold_g_per_s": 12,
  "gyro_threshold_dps": 220,
  "stillness_acc_delta_g": 0.18,
  "stillness_gyro_threshold_dps": 35,
  "medium_risk_score": 0.35,
  "high_risk_score": 0.55,
  "fall_score": 0.8,
  "alert_cooldown_seconds": 25
}
```

Side effects:

- emits WebSocket event `detector.config.updated`

Common errors:

- `422 validation_error` for invalid field names or out-of-range values

### GET /api/v1/detector/status

Get detector runtime status and artifact loading diagnostics.

- Request body: none
- Success response: `200 OK`

```json
{
  "mode": "rule_based",
  "runtime_loaded": false,
  "artifacts_dir": "results/artifacts",
  "reason": "No offline detector bundle found in results/artifacts.",
  "import_error": null,
  "bundle_path": "results/artifacts/fall_detector_bundle.joblib",
  "legacy_model_path": "results/artifacts/fall_detector.joblib",
  "metadata_path": "results/artifacts/fall_detector_metadata.json",
  "window_size_samples": null,
  "step_size_samples": null,
  "target_fs": null,
  "feature_count": 0,
  "model_name": null,
  "artifact_path": null
}
```

---

## 8.4 Patient Endpoints

### POST /api/v1/patients

Create a patient record.

- Request body: `PatientCreate`
- Success response: `200 OK` (`PatientRecord`)

Example request:

```json
{
  "full_name": "Amina Khan",
  "age": 78,
  "room_label": "Room-12A",
  "emergency_contact": "+92-300-1234567",
  "notes": "Uses walker"
}
```

Example response:

```json
{
  "id": "pat_1234567890",
  "full_name": "Amina Khan",
  "age": 78,
  "room_label": "Room-12A",
  "emergency_contact": "+92-300-1234567",
  "notes": "Uses walker",
  "created_at": "2026-03-31T20:31:00.000000+00:00"
}
```

Side effects:

- initializes patient live status
- emits WebSocket event `patient.created`

### GET /api/v1/patients

List all patients.

- Request body: none
- Success response: `200 OK` array of `PatientRecord`

### GET /api/v1/patients/{patient_id}

Get a patient by ID.

- Path param: `patient_id`
- Request body: none
- Success response: `200 OK` (`PatientRecord`)
- Not found: `404 http_error`

---

## 8.5 Device Endpoints

### POST /api/v1/devices

Create a device record.

- Request body: `DeviceCreate`
- Success response: `200 OK` (`DeviceRecord`)

Example request:

```json
{
  "label": "NurseStation-Phone-1",
  "platform": "android",
  "owner_name": "Ward B"
}
```

Example response:

```json
{
  "id": "dev_1234567890",
  "label": "NurseStation-Phone-1",
  "platform": "android",
  "owner_name": "Ward B",
  "created_at": "2026-03-31T20:32:00.000000+00:00",
  "last_seen_at": null
}
```

Side effects:

- emits WebSocket event `device.created`

### GET /api/v1/devices

List all devices.

- Request body: none
- Success response: `200 OK` array of `DeviceRecord`

### GET /api/v1/devices/{device_id}

Get a device by ID.

- Path param: `device_id`
- Request body: none
- Success response: `200 OK` (`DeviceRecord`)
- Not found: `404 http_error`

---

## 8.6 Session Endpoints

### POST /api/v1/sessions

Start a monitoring session linking one patient and one device.

- Request body: `SessionCreate`
- Success response: `200 OK` (`SessionRecord`)

Example request:

```json
{
  "patient_id": "pat_1234567890",
  "device_id": "dev_1234567890",
  "started_by": "operator_1",
  "sample_rate_hz": 50,
  "notes": "Night shift"
}
```

Example response:

```json
{
  "id": "ses_1234567890",
  "patient_id": "pat_1234567890",
  "device_id": "dev_1234567890",
  "started_by": "operator_1",
  "sample_rate_hz": 50,
  "notes": "Night shift",
  "status": "active",
  "started_at": "2026-03-31T20:33:00.000000+00:00",
  "ended_at": null,
  "last_ingested_at": null
}
```

Business rules:

- patient must exist
- device must exist
- no other active session can exist for the same patient or device

Common errors:

- `400 http_error` with message for rule violation

Side effects:

- emits `session.started`
- emits `patient.live.updated`

### GET /api/v1/sessions

List sessions.

- Query params:
  - `active_only` boolean, default `false`
- Request body: none
- Success response: `200 OK` array of `SessionRecord`

### GET /api/v1/sessions/{session_id}

Get a session by ID.

- Path param: `session_id`
- Request body: none
- Success response: `200 OK` (`SessionRecord`)
- Not found: `404 http_error`

### POST /api/v1/sessions/{session_id}/stop

Stop a session.

- Path param: `session_id`
- Request body: `SessionStopRequest`
- Success response: `200 OK` (updated `SessionRecord`)

Example request:

```json
{
  "stopped_by": "operator_1",
  "note": "Monitoring complete"
}
```

Behavior:

- if session is active: marks `stopped`, sets `ended_at`, optionally updates notes
- if session already stopped: returns existing session (no new events)

Common errors:

- `404 http_error` when session not found

Side effects (active stop):

- emits `session.stopped`
- emits `patient.live.updated`

---

## 8.7 Monitoring Endpoints

### GET /api/v1/monitor/patients/live

List live status cards for all known patients.

- Request body: none
- Success response: `200 OK` array of `PatientLiveStatus`

### GET /api/v1/monitor/patients/{patient_id}/live

Get one patient live status.

- Path param: `patient_id`
- Request body: none
- Success response: `200 OK` (`PatientLiveStatus`)
- Not found: `404 http_error`

### GET /api/v1/monitor/telemetry/recent

List most recent telemetry snapshots.

- Query params:
  - `limit` integer, default 20, minimum 1, maximum 100
- Request body: none
- Success response: `200 OK` array of `TelemetrySnapshot`

Notes:

- backend retains an internal rolling queue (default max 120 snapshots)

### GET /api/v1/monitor/patients/{patient_id}/telemetry

Get latest telemetry snapshot for one patient.

- Path param: `patient_id`
- Request body: none
- Success response: `200 OK` (`TelemetrySnapshot`)
- Not found: `404 http_error`

---

## 8.8 Ingestion Endpoint

### POST /api/v1/ingest/live

Ingest one live sensor batch and run risk detection.

- Request body: `SensorBatchIn`
- Success response: `200 OK` (`SensorBatchOut`)

Processing flow:

1. validate payload schema and numeric ranges
2. load detector config
3. run detector analysis (offline model if loaded, otherwise rule-based)
4. validate session ownership and active status
5. update session/device timestamps and patient live status
6. write telemetry snapshot and recent telemetry queue
7. auto-create alert for `high_risk` or `fall_detected` (with cooldown logic)
8. return detection + live status + optional active alert + telemetry

Example request (shortened):

```json
{
  "patient_id": "pat_1234567890",
  "device_id": "dev_1234567890",
  "session_id": "ses_1234567890",
  "source": "mobile",
  "sampling_rate_hz": 50,
  "acceleration_unit": "m_s2",
  "gyroscope_unit": "dps",
  "battery_level": 84,
  "samples": [
    {
      "timestamp_ms": 1711910400123,
      "acc_x": 0.01,
      "acc_y": -0.02,
      "acc_z": 9.82,
      "gyro_x": 1.1,
      "gyro_y": 0.3,
      "gyro_z": 0.6
    }
  ]
}
```

Example response (shortened):

```json
{
  "ingested_samples": 64,
  "detection": {
    "severity": "low",
    "score": 0.12,
    "fall_probability": 0.12,
    "peak_acc_g": 1.09,
    "peak_gyro_dps": 32.4,
    "peak_jerk_g_per_s": 4.5,
    "stillness_ratio": 0.78,
    "samples_analyzed": 64,
    "message": "Movement window appears stable.",
    "reasons": [
      "Signal stayed within configured low-risk thresholds."
    ],
    "detected_at": "2026-03-31T20:34:00.000000+00:00"
  },
  "active_alert": null,
  "live_status": {
    "patient_id": "pat_1234567890",
    "patient_name": "Amina Khan",
    "room_label": "Room-12A",
    "session_id": "ses_1234567890",
    "device_id": "dev_1234567890",
    "severity": "low",
    "score": 0.12,
    "fall_probability": 0.12,
    "last_ingested_at": "2026-03-31T20:34:00.000000+00:00",
    "last_message": "Movement window appears stable.",
    "sample_rate_hz": 50,
    "latest_metrics": {
      "peak_acc_g": 1.09,
      "peak_gyro_dps": 32.4,
      "peak_jerk_g_per_s": 4.5,
      "stillness_ratio": 0.78,
      "battery_level": 84
    },
    "active_alert_ids": []
  },
  "telemetry": {
    "patient_id": "pat_1234567890",
    "patient_name": "Amina Khan",
    "room_label": "Room-12A",
    "session_id": "ses_1234567890",
    "device_id": "dev_1234567890",
    "source": "mobile",
    "sampling_rate_hz": 50,
    "acceleration_unit": "m_s2",
    "gyroscope_unit": "dps",
    "battery_level": 84,
    "received_at": "2026-03-31T20:34:00.000000+00:00",
    "samples_in_last_batch": 64,
    "latest_samples": []
  }
}
```

Common errors:

- `400 http_error` when patient/device/session references are invalid or inconsistent
- `422 validation_error` for malformed payload

Side effects:

- emits `telemetry.ingested`
- emits `detection.updated`
- emits `patient.live.updated`
- emits `alert.created` when a new alert is generated

---

## 8.9 Alert Endpoints

### GET /api/v1/alerts

List alerts with optional filtering.

- Query params:
  - `status`: optional enum `open|acknowledged|resolved`
  - `patient_id`: optional string
- Request body: none
- Success response: `200 OK` array of `AlertRecord` sorted newest first

### POST /api/v1/alerts/manual

Create a manual alert (operator-triggered).

- Request body: `ManualAlertCreate`
- Success response: `200 OK` (`AlertRecord`)

Behavior:

- requires patient to exist
- `score` is auto-assigned:
  - `1.0` for `fall_detected`
  - `0.9` for other severities
- alert starts as `open`

Common errors:

- `400 http_error` when patient does not exist
- `422 validation_error` for invalid payload

Side effects:

- emits `alert.created`
- may emit `patient.live.updated`

### POST /api/v1/alerts/{alert_id}/acknowledge

Acknowledge an alert.

- Path param: `alert_id`
- Request body: `AlertActionRequest`
- Success response: `200 OK` (updated `AlertRecord`)

Behavior:

- sets `status = acknowledged`
- sets `acknowledged_at` and `acknowledged_by`
- optional note overwrite when provided

Common errors:

- `400 http_error` when alert does not exist
- `400 http_error` when alert is already resolved
- `422 validation_error` for invalid payload

Side effects:

- emits `alert.acknowledged`
- emits `patient.live.updated`

### POST /api/v1/alerts/{alert_id}/resolve

Resolve an alert.

- Path param: `alert_id`
- Request body: `AlertActionRequest`
- Success response: `200 OK` (updated `AlertRecord`)

Behavior:

- sets `status = resolved`
- sets `resolved_at` and `resolved_by`
- optional note overwrite when provided

Common errors:

- `400 http_error` when alert does not exist
- `422 validation_error` for invalid payload

Side effects:

- emits `alert.resolved`
- emits `patient.live.updated`

---

## 8.10 WebSocket Endpoint

### WS /ws/monitor

Subscribe to live backend events.

- URL: `ws://<host>/ws/monitor`
- Request body: none (WebSocket handshake)
- Initial server message:

```json
{
  "type": "connection.ready",
  "created_at": "2026-03-31T20:35:00.000000+00:00",
  "payload": {
    "message": "Connected to live monitoring stream."
  }
}
```

After connection, server broadcasts event objects as JSON.

Event shape:

```json
{
  "id": "evt_1234567890ab",
  "type": "alert.created",
  "created_at": "2026-03-31T20:35:05.000000+00:00",
  "payload": {}
}
```

Common event types:

- `patient.created`
- `device.created`
- `session.started`
- `session.stopped`
- `patient.live.updated`
- `telemetry.ingested`
- `detection.updated`
- `alert.created`
- `alert.acknowledged`
- `alert.resolved`
- `detector.config.updated`

Client guidance:

- keep socket open continuously for dashboard/live views
- implement reconnect with backoff
- treat event payload as append-only stream snapshots

## 9. Detector Runtime Notes

`POST /api/v1/ingest/live` uses detector mode selected at runtime:

- `offline_model` when model artifacts load successfully
- `rule_based` fallback when artifacts or runtime dependencies are unavailable

Check mode using `GET /api/v1/detector/status`.

## 10. Integration Sequence (Recommended)

For first-time setup, call APIs in this order:

1. `POST /api/v1/patients`
2. `POST /api/v1/devices`
3. `POST /api/v1/sessions`
4. Stream data to `POST /api/v1/ingest/live`
5. Poll or subscribe:
   - `GET /api/v1/monitor/patients/live`
   - `GET /api/v1/alerts`
   - `WS /ws/monitor`
6. End workflow with `POST /api/v1/sessions/{session_id}/stop`

## 11. Quick Test Snippets

### 11.1 Health check

```bash
curl -s http://127.0.0.1:8000/api/v1/health
```

### 11.2 Create patient

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/patients \
  -H "Content-Type: application/json" \
  -d '{"full_name":"Amina Khan","age":78,"room_label":"Room-12A"}'
```

### 11.3 Read detector status

```bash
curl -s http://127.0.0.1:8000/api/v1/detector/status
```

## 12. Compatibility and Change Management

Current API is versioned under `/api/v1`.

Recommended rules for future changes:

- add fields in backward-compatible manner
- avoid changing enum values without version bump
- avoid changing ID prefixes in-place
- introduce `/api/v2` for breaking changes

---

For implementation-level details, review backend source files under `flask_backend/app`.
