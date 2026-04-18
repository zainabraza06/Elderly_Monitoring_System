# Assignment 3 Full Technical Analysis

## Model as a Service (Elderly Monitoring System)

- Submission target: Mar 31, 2026
- Course component: Assignment
- Scope covered in this document:
  - REST API functionality (FastAPI)
  - Input validation and schema enforcement
  - Exception handling and error contract
  - Stress testing and performance analysis
  - API documentation and deployability considerations
  - System-level reflection and long-term architectural consequences

This file is intentionally detailed so it can be distilled into a printed 4-5 page report.

---

## 1. Executive Summary

The backend service is implemented with FastAPI and exposes a structured real-time monitoring API for elderly fall-risk workflows. The system architecture separates offline model training from online inference APIs, which is the correct service boundary for production-like deployment. Request contracts are strongly validated with Pydantic (`extra="forbid"`, field bounds, custom validators), and centralized exception handling returns a standardized error envelope including trace ID and timestamp.

Performance testing was executed on `POST /api/v1/ingest/live` using an async stress harness. The current baseline run achieved:

- Throughput: 36.12 requests/sec
- Success rate: 100.00% (200/200)
- Latency: p50 2243.35 ms, p95 5046.06 ms, p99 5283.49 ms

System integration is functional, but current runtime detector mode is `rule_based`, not `offline_model`, because the trained model bundle is not present in `results/artifacts` at service runtime. This is the main notebook-to-service deployment gap.

---

## 2. Assignment Deliverables Coverage

### 2.1 Code deliverables

1. REST API using FastAPI: Completed.
2. Input validation and schema enforcement: Completed.
3. Exception handling: Completed with centralized handlers.
4. Stress testing (latency, throughput): Completed with reproducible scripts and saved reports.
5. API documentation (Swagger/OpenAPI): Completed (`/docs`, `/redoc`, `/openapi.json`).

### 2.2 Report deliverables

1. Architecture and API design: Covered in Sections 3 and 4.
2. Notebook-to-service gap: Covered in Section 5.
3. Robustness and reliability analysis: Covered in Section 6.
4. Performance analysis: Covered in Section 7.
5. Reflection on architectural consequence: Covered in Section 8.

---

## 3. Architecture and Service Boundary Design

## 3.1 Why this service boundary is correct

The architecture intentionally isolates responsibilities:

- Offline pipeline (`scripts/`): preprocessing, feature extraction, model training.
- Online service (`flask_backend/`): ingestion, risk scoring, alerting, monitoring state, and operator APIs.
- Client app (`app_frontend/`): sensor acquisition and API/WebSocket interaction.

This split is justified because:

- Training workloads are compute-heavy and dataset-bound; they should not run in the request path.
- Runtime inference/API should remain responsive and auditable.
- Deployment cadence differs: model retraining can be periodic while API uptime is continuous.

## 3.2 Runtime data flow

1. Client sends sensor window to `POST /api/v1/ingest/live`.
2. Service validates request schema and value ranges.
3. Detector resolves active mode:
   - `offline_model` if bundle is loadable.
   - `rule_based` fallback otherwise.
4. Store updates session, telemetry, live patient state, and alert state.
5. Monitor events are broadcast over `WS /ws/monitor`.
6. Dashboard/operator clients query REST endpoints and subscribe to live events.

## 3.3 Core backend modules

- `flask_backend/app/main.py`: FastAPI app, middleware, routes, exception handlers.
- `flask_backend/app/schemas.py`: all input/output contracts, constraints, validators.
- `flask_backend/app/detection.py`: detector runtime loading + rule-based/ML analysis logic.
- `flask_backend/app/store.py`: in-memory state machine for patients, devices, sessions, alerts, telemetry.
- `flask_backend/app/realtime.py`: WebSocket connection manager and event broadcasting.

---

## 4. API Design and Data Contracts

## 4.1 Endpoint groups

### System and detector

- `GET /api/v1/health`
- `GET /api/v1/summary`
- `GET /api/v1/detector/config`
- `PUT /api/v1/detector/config`
- `GET /api/v1/detector/status`

### Entity lifecycle

- `POST /api/v1/patients`, `GET /api/v1/patients`, `GET /api/v1/patients/{patient_id}`
- `POST /api/v1/devices`, `GET /api/v1/devices`, `GET /api/v1/devices/{device_id}`
- `POST /api/v1/sessions`, `GET /api/v1/sessions`, `GET /api/v1/sessions/{session_id}`
- `POST /api/v1/sessions/{session_id}/stop`

### Live monitoring and ingestion

- `POST /api/v1/ingest/live`
- `GET /api/v1/monitor/patients/live`
- `GET /api/v1/monitor/patients/{patient_id}/live`
- `GET /api/v1/monitor/telemetry/recent`
- `GET /api/v1/monitor/patients/{patient_id}/telemetry`
- `GET /api/v1/events/recent`
- `WS /ws/monitor`

### Alerts

- `GET /api/v1/alerts`
- `POST /api/v1/alerts/manual`
- `POST /api/v1/alerts/{alert_id}/acknowledge`
- `POST /api/v1/alerts/{alert_id}/resolve`

## 4.2 Endpoint-by-endpoint functionality

### Root and API docs

- `GET /`
   - Purpose: quick service discovery endpoint used to verify backend reachability.
   - Returns: welcome message, API prefix, and WebSocket path.

- `GET /docs`
   - Purpose: interactive Swagger UI for trying all REST endpoints.
   - Returns: OpenAPI-backed interactive documentation page.

- `GET /redoc`
   - Purpose: ReDoc API reference view.
   - Returns: static API documentation page.

- `GET /openapi.json`
   - Purpose: machine-readable contract for clients/tooling.
   - Returns: full OpenAPI schema for all HTTP routes.

### System and detector APIs

- `GET /api/v1/health`
   - Purpose: liveness and basic service identity check.
   - Returns: `status`, `app_name`, `timestamp`.

- `GET /api/v1/summary`
   - Purpose: compact operational snapshot for dashboards.
   - Returns: totals for patients/devices, active sessions, open/acknowledged alerts, and last event timestamp.

- `GET /api/v1/events/recent`
   - Purpose: fetch latest monitor events generated by the store.
   - Returns: reverse-chronological event list (`id`, `type`, `created_at`, `payload`).

- `GET /api/v1/detector/config`
   - Purpose: read currently active detector thresholds and cooldown policy.
   - Returns: full `DetectorConfig` object.

- `PUT /api/v1/detector/config`
   - Purpose: update detector tuning values at runtime.
   - Input: partial `DetectorConfigUpdate` payload.
   - Behavior: merges non-null fields into existing config, validates ranges, emits `detector.config.updated` event.
   - Returns: updated detector config.

- `GET /api/v1/detector/status`
   - Purpose: inspect runtime inference mode and artifact readiness.
   - Returns: mode (`offline_model` or `rule_based`), load flags, artifact paths, model metadata, and failure reason if fallback is active.

### Patient APIs

- `POST /api/v1/patients`
   - Purpose: register a patient.
   - Input: `PatientCreate` (name + optional demographics/contact/notes).
   - Behavior: creates patient record and initializes patient live status row.
   - Side effects: emits `patient.created` event over WebSocket stream.
   - Returns: created patient record.

- `GET /api/v1/patients`
   - Purpose: list all registered patients.
   - Returns: array of `PatientRecord`.

- `GET /api/v1/patients/{patient_id}`
   - Purpose: fetch one patient by ID.
   - Returns: `PatientRecord`.
   - Errors: `404` when patient does not exist.

### Device APIs

- `POST /api/v1/devices`
   - Purpose: register a data-source device (mobile/watch/etc.).
   - Input: `DeviceCreate`.
   - Behavior: creates device entry with created timestamp.
   - Side effects: emits `device.created`.
   - Returns: `DeviceRecord`.

- `GET /api/v1/devices`
   - Purpose: list known devices.
   - Returns: array of `DeviceRecord`.

- `GET /api/v1/devices/{device_id}`
   - Purpose: fetch one device by ID.
   - Returns: `DeviceRecord`.
   - Errors: `404` when device is not found.

### Session APIs

- `POST /api/v1/sessions`
   - Purpose: start a monitoring session linking one patient and one device.
   - Input: `SessionCreate`.
   - Behavior: validates patient/device existence and prevents parallel active sessions for same patient/device.
   - Side effects: updates patient live status and emits `session.started` + `patient.live.updated`.
   - Returns: active `SessionRecord`.

- `GET /api/v1/sessions`
   - Purpose: list sessions.
   - Query: `active_only` (default `false`).
   - Returns: array of `SessionRecord` (filtered when requested).

- `GET /api/v1/sessions/{session_id}`
   - Purpose: fetch one session by ID.
   - Returns: `SessionRecord`.
   - Errors: `404` if missing.

- `POST /api/v1/sessions/{session_id}/stop`
   - Purpose: stop an active session.
   - Input: `SessionStopRequest` (`stopped_by`, optional note).
   - Behavior: marks session stopped, sets `ended_at`, clears patient live session/device pointers.
   - Side effects: emits `session.stopped` + `patient.live.updated`.
   - Returns: updated `SessionRecord`.

### Live ingestion and monitoring APIs

- `POST /api/v1/ingest/live`
   - Purpose: ingest one sensor batch and compute fall-risk output in real time.
   - Input: `SensorBatchIn` (patient/device/session IDs, units, sampling rate, battery, sample array).
   - Behavior:
      - Validates entity/session ownership and active state.
      - Runs detector analysis (`offline_model` if available, otherwise `rule_based`).
      - Updates session/device timestamps, patient live status, telemetry snapshots.
      - Auto-creates alert for `high_risk`/`fall_detected` severity with cooldown control.
   - Side effects: emits `telemetry.ingested`, `detection.updated`, `patient.live.updated`, and optionally `alert.created`.
   - Returns: `SensorBatchOut` with detection, live status, optional active alert, and telemetry summary.

- `GET /api/v1/monitor/patients/live`
   - Purpose: list current live status for all patients.
   - Returns: array of `PatientLiveStatus`.

- `GET /api/v1/monitor/patients/{patient_id}/live`
   - Purpose: get live status for one patient.
   - Returns: `PatientLiveStatus`.
   - Errors: `404` when no live status exists.

- `GET /api/v1/monitor/telemetry/recent`
   - Purpose: recent telemetry timeline for UI widgets.
   - Query: `limit` (1..100, default `20`).
   - Returns: most recent `TelemetrySnapshot` entries.

- `GET /api/v1/monitor/patients/{patient_id}/telemetry`
   - Purpose: latest telemetry snapshot for one patient.
   - Returns: `TelemetrySnapshot`.
   - Errors: `404` when telemetry has not yet been ingested for that patient.

### Alert APIs

- `GET /api/v1/alerts`
   - Purpose: query alert history/current alert queue.
   - Query filters: `status` (`open|acknowledged|resolved`), `patient_id`.
   - Returns: reverse-chronological array of `AlertRecord`.

- `POST /api/v1/alerts/manual`
   - Purpose: operator-triggered manual escalation.
   - Input: `ManualAlertCreate`.
   - Behavior: creates open alert even without detector trigger.
   - Side effects: emits `alert.created`, updates patient live state.
   - Returns: created `AlertRecord`.

- `POST /api/v1/alerts/{alert_id}/acknowledge`
   - Purpose: mark an alert as seen/acknowledged by staff.
   - Input: `AlertActionRequest` (actor, optional note).
   - Behavior: sets alert status to `acknowledged` (unless already resolved).
   - Side effects: emits `alert.acknowledged`, updates patient live active alert IDs.
   - Returns: updated `AlertRecord`.

- `POST /api/v1/alerts/{alert_id}/resolve`
   - Purpose: close an alert lifecycle after intervention.
   - Input: `AlertActionRequest`.
   - Behavior: sets status to `resolved`, stamps resolver and time.
   - Side effects: emits `alert.resolved`, updates patient live state.
   - Returns: updated `AlertRecord`.

### WebSocket API

- `WS /ws/monitor`
   - Purpose: push real-time backend events to dashboards/operators.
   - On connect: sends `connection.ready` event.
   - Runtime behavior: server broadcasts store-generated events (`patient.live.updated`, `detection.updated`, `alert.*`, etc.).
   - Client pattern: keep connection alive while receiving live JSON event stream.

## 4.3 Contract quality and validation strategy

The service enforces strict contracts in `schemas.py`:

- Unknown fields rejected with `ConfigDict(extra="forbid")`.
- Numeric bounds on sensor values:
  - acceleration and gyro ranges constrained to reject impossible data.
  - sampling rates and scores constrained to valid operating intervals.
- String sanitization and non-empty checks via `field_validator`.
- Cross-field temporal rule in `SensorBatchIn`:
  - timestamps must be monotonically non-decreasing when present.

This protects the service from silent contract drift and malformed payloads.

## 4.4 Standardized error response contract

Error shape:

```json
{
  "code": "validation_error|http_error|domain_error|internal_error",
  "message": "...",
  "trace_id": "req_xxx",
  "timestamp": "UTC timestamp",
  "details": "optional structured diagnostics"
}
```

Operational headers attached to responses:

- `X-Request-Id`
- `X-Process-Time-Ms`

These improve observability and request-level debugging.

## 4.5 API documentation support

Auto-generated docs are available at runtime:

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI spec: `/openapi.json`

This satisfies API discoverability and contract transparency requirements.

---

## 5. Notebook to Service Gap Analysis

## 5.1 What worked offline

Offline scripts successfully generated early-stage artifacts in `results/artifacts`:

- `windows.parquet`
- `windows_signals.npz`

These indicate preprocessing/windowing worked.

## 5.2 What failed or diverged in deployment runtime

At backend runtime, detector status currently reports:

- `mode = rule_based`
- `runtime_loaded = false`
- reason: no offline detector bundle found in `results/artifacts`

The expected bundle path is:

- `results/artifacts/fall_detector_bundle.joblib`

This reveals a classic notebook-to-service gap: successful data prep does not guarantee deployable inference artifacts are present where the service expects them.

## 5.3 Root causes of the gap

1. Artifact completeness mismatch:
   - Runtime needs model bundle + metadata-compatible structure.
   - Current folder has only windows artifacts.

2. Environment split:
   - Root pipeline dependencies and backend dependencies are separate.
   - Optional or missing runtime packages can block model loading.

3. Working-directory/path coupling:
   - Detector reads relative `results/artifacts` path.
   - Container/runtime path differences can prevent artifact discovery.

4. Packaging boundary:
   - Backend Docker image copies `flask_backend/` code only.
   - Model artifacts are not automatically included unless copied or mounted.

## 5.4 How to close the gap

1. Ensure training produces and verifies:
   - `fall_detector_bundle.joblib`
   - `fall_detector_metadata.json` (if legacy path used)

2. Add startup readiness check that can fail deployment in strict mode if model bundle is required.

3. Explicitly configure and document artifact mount/copy strategy in deployment.

4. Add CI check for detector status endpoint to validate expected mode after deploy.

---

## 6. Robustness and Reliability Analysis

## 6.1 Built-in robustness mechanisms

1. Schema-level hard validation:
   - strict unknown-field rejection
   - typed enums for units/statuses
   - range and length constraints

2. Domain integrity checks in store layer:
   - patient/device/session existence validation
   - active-session ownership checks
   - alert lifecycle constraints

3. Centralized exception handling in app layer:
   - `RequestValidationError` -> 422
   - `HTTPException` -> mapped to standardized error envelope
   - `ValueError` -> domain-level 400
   - generic `Exception` -> 500 with internal error contract

4. Runtime fallback strategy:
   - detector automatically falls back to rule-based mode when model bundle cannot load.
   - improves availability but can hide ML-runtime regressions if not monitored.

## 6.2 Empirical malformed-input test matrix (live API)

The following cases were executed against running backend endpoints:

| Test case | Endpoint | Scenario | Observed status | Observed code | Observation |
|---|---|---|---:|---|---|
| TC1 | `POST /patients` | Missing `full_name` | 422 | `validation_error` | Correct schema rejection |
| TC4 | `POST /sessions` | Unknown patient ID | 400 | `http_error` | Correct domain integrity failure |
| TC7 | `POST /ingest/live` | Extra unknown field in payload | 422 | `validation_error` | Correct `extra=forbid` behavior |
| TC8 | `POST /ingest/live` | Valid shape but unknown patient ID | 400 | `http_error` | Correct semantic/entity validation |
| TC_VALID_INGEST | `POST /ingest/live` | Valid small sample window | 200 | N/A | Success path confirmed (`severity=low`) |

## 6.3 Important robustness finding

For non-monotonic timestamps in `SensorBatchIn` (a validator-defined invalid case), response observed was:

- status: 400
- code: `domain_error`
- message: `Unable to serialize unknown type: <class 'ValueError'>`

Expected behavior for this input class is typically a 422 validation response with structured details. This indicates an error serialization edge case in the validation-exception path for nested `ValueError` payloads.

### Suggested fix

In the validation exception handler, sanitize `exc.errors()` to ensure non-JSON-serializable values (for example `ctx.error`) are converted to strings before returning the response body.

## 6.4 Reliability strengths and current limits

Strengths:

- deterministic request contracts
- consistent error envelope and traceability headers
- explicit detector mode introspection endpoint
- separation between detector and API/store layers

Current limits:

- in-memory store only (no persistence or multi-instance consistency)
- single-process lock contention under load (`asyncio.Lock` around state mutation)
- model runtime readiness depends on external artifact management

---

## 7. Performance and Stress Testing Analysis

## 7.1 Methodology

Stress tool: `flask_backend/scripts/stress_test.py`

Key characteristics:

- async load generation with `httpx.AsyncClient`
- bootstrap step creates patient/device/session automatically
- target endpoint: `POST /api/v1/ingest/live`
- captures throughput, success rate, and latency percentiles
- writes both JSON and Markdown report artifacts

Run profile used for baseline report:

- requests: 200
- concurrency: 20
- samples/request: 64
- base URL: `http://127.0.0.1:8000`

## 7.2 Observed baseline metrics

From `flask_backend/stress_reports/stress_report_20260331_134622.json`:

- Duration: 5.536 s
- Throughput: 36.124 req/s
- Success count: 200
- Error count: 0
- Success rate: 100.00%
- Avg latency: 2546.95 ms
- Min latency: 290.83 ms
- P50 latency: 2243.35 ms
- P95 latency: 5046.06 ms
- P99 latency: 5283.49 ms
- Max latency: 5316.95 ms

## 7.3 Interpretation

1. Correctness under pressure is strong:
   - no failed requests at this workload.

2. Tail latency is high:
   - p95 and p99 are both around 5 seconds.
   - this indicates queueing/contention effects as concurrency rises.

3. Throughput is moderate for real-time ingestion:
   - suitable for prototype/single-node demo usage.
   - likely insufficient for larger multi-device production loads without optimization.

## 7.4 Likely bottlenecks (code-informed)

1. Shared mutable state lock in `store.py`:
   - write-heavy ingestion path serializes critical sections.

2. CPU work per request in detection path:
   - motion features and scoring computed per batch.

3. Event fan-out work after each ingest:
   - multiple event objects and WebSocket broadcasts per request.

4. Python process constraints:
   - single-process default deployment limits parallel CPU execution for compute-heavy paths.

## 7.5 Optimization roadmap

Priority 1 (quick wins):

- reduce work inside lock scope in `ingest_detection`
- batch/coalesce monitor events where possible
- profile detector path and avoid repeated allocations

Priority 2 (architecture):

- move state from in-memory dicts to external datastore (Redis/Postgres)
- introduce background queue for non-critical event processing
- scale backend workers and tune process model

Priority 3 (ML runtime):

- ensure offline model bundle is loaded in production mode
- benchmark `offline_model` vs `rule_based` latency and accuracy trade-offs

---

## 8. Reflection: One Architectural Decision With Long-Term Consequences

Chosen decision:

- Keep backend state fully in-memory (`BackendStore`) during initial implementation.

Why this was attractive initially:

- very fast development loop
- simple logic debugging
- no external infrastructure needed

Long-term consequences:

1. No durable state:
   - restart causes data loss for sessions/alerts/telemetry.

2. Limited horizontal scaling:
   - each instance has isolated state, making multi-instance consistency difficult.

3. Performance ceiling under concurrency:
   - lock-based serialization constrains throughput and tail latency.

4. Operational risk in healthcare-like contexts:
   - alert state durability and auditability are critical for trust and incident review.

How this would be revised in next iteration:

- externalize state to a shared transactional store
- keep an append-only event log for auditability
- preserve the same API contracts to avoid client rewrites

---

## 9. Deployment Reflection (Assignment Rubric)

Backend deployment automation is configured through GitHub Actions + CapRover (`flask_backend/Dockerfile`, `flask_backend/captain-definition`, `.github/workflows/deploy-backend.yml`).

Important deployment insight:

- Service containerization and API availability do not automatically imply ML readiness.
- The detector can start in fallback mode when artifacts are absent, so deployment validation must include functional mode checks (`/api/v1/detector/status`) not only health checks.

Recommended deployment gate:

1. Run `/api/v1/health`.
2. Run `/api/v1/detector/status`.
3. Assert expected mode (`offline_model` for production).
4. Fail pipeline if mode mismatch persists.

---

## 10. Rubric Mapping Table

| Rubric criterion | Weight | Evidence in this project |
|---|---:|---|
| API functionality and correctness | 25% | FastAPI endpoint suite, successful valid ingest, entity/session/alert lifecycle |
| Robustness and validation | 20% | strict schemas, forbidden extras, validators, standardized error envelope, malformed-input matrix |
| Performance testing and analysis | 15% | async stress tool + stored JSON/MD report + percentile analysis |
| System-level reasoning (report) | 25% | architecture boundary rationale, notebook-to-service gap, bottleneck analysis, mitigation roadmap |
| Deployment reflection | 15% | CapRover workflow + runtime mode gap and deployment gating strategy |

---

## 11. Reproducibility Commands

## 11.1 Start backend

```bash
python -m pip install -r flask_backend/requirements.txt
python -m uvicorn flask_backend.app.main:app --host 0.0.0.0 --port 8000
```

## 11.2 Check docs and runtime mode

```bash
# docs
http://127.0.0.1:8000/docs

# detector mode
curl http://127.0.0.1:8000/api/v1/detector/status
```

## 11.3 Run stress test and save reports

```bash
python flask_backend/scripts/stress_test.py \
  --base-url http://127.0.0.1:8000 \
  --requests 200 \
  --concurrency 20 \
  --samples-per-request 64 \
  --output-dir flask_backend/stress_reports
```

---

## 12. Final Submission Notes

If you need a strict 4-5 page printed version, condense this file by keeping:

1. Executive summary
2. Architecture and endpoint design (short form)
3. Notebook-to-service gap (artifact/runtime mismatch)
4. Robustness matrix (table only)
5. Performance table + 3 key insights
6. Reflection section

That compressed subset will still retain strong rubric coverage while staying concise enough for A4 page limits.
