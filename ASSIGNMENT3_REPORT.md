# Assignment 3 Report: Model as a Service

## 1. Architecture And API Design

### 1.1 Service Boundaries
The service is implemented as a FastAPI backend that exposes REST endpoints and a WebSocket stream for live monitoring. The architecture separates responsibilities into four core modules:

- `backend/app/main.py`: API routing, middleware, and exception handling.
- `backend/app/schemas.py`: request/response data contracts and validation rules.
- `backend/app/store.py`: state management and domain workflows (patients, devices, sessions, alerts, telemetry).
- `backend/app/detection.py`: real-time detection logic (rule-based fallback + optional offline-trained model bundle).

This decomposition enforces clear boundaries:
- API layer focuses on transport concerns and protocol behavior.
- Store layer enforces domain rules and workflow correctness.
- Detection layer is replaceable without changing endpoint contracts.

### 1.2 Endpoint Design Rationale
Endpoints were grouped by domain capability instead of client type:

- System: `/api/v1/health`, `/api/v1/summary`
- Core entities: `/api/v1/patients`, `/api/v1/devices`, `/api/v1/sessions`
- Monitoring: `/api/v1/monitor/*`, `/api/v1/events/recent`
- Detection and ingestion: `/api/v1/ingest/live`, `/api/v1/detector/*`
- Incident management: `/api/v1/alerts/*`
- Real-time stream: `/ws/monitor`

This structure reduces coupling between clients (mobile app, web dashboard, stress-test utility) and backend internals while maintaining an intuitive URL hierarchy.

### 1.3 Data Contract Strategy
Pydantic schemas define explicit contracts with constraint checks:

- Numeric range constraints for sensor values and rates.
- String non-empty and max-length constraints for identifiers and actor names.
- `extra=forbid` to reject unknown fields and prevent contract drift.
- Timestamp monotonic validation for sample batches when timestamps are present.

The API now returns consistent error payloads with fields:
- `code`
- `message`
- `trace_id`
- `timestamp`
- `details`

This standardization improves debuggability and enables client-side error handling logic.

## 2. Notebook-to-Service Gap

### 2.1 What Worked Offline
The notebook/pipeline workflow handled:
- Sensor preprocessing and windowing
- Feature extraction
- Model training
- Artifact export to files

Offline workflows were tolerant of local assumptions, ad hoc file paths, and non-uniform data availability.

### 2.2 What Failed During Service Integration
While deploying as a backend service, several gaps appeared:

1. Dependency mismatch at runtime:
- Missing runtime packages (`numpy`, parquet engine dependencies) caused import/runtime errors.

2. Environment/entrypoint mismatch:
- Shell-level `python` alias conflicts and import-path assumptions created startup issues.

3. Data pipeline assumptions:
- Pipeline execution assumed a local dataset folder existed. Without explicit checks, this surfaced later as secondary failures.

4. Interface hardening needs:
- Offline scripts accepted implicit assumptions that are unsafe in network APIs (unknown fields, malformed payloads, out-of-range values).

### 2.3 Gap Mitigations Applied
- Added strict request validation and forbidden unknown fields.
- Added centralized exception handlers and request trace IDs.
- Added fail-fast behavior for missing data directories in pipeline scripts.
- Added load-testing tooling to evaluate service behavior under concurrency.

## 3. Robustness And Reliability Analysis

### 3.1 Validation Behavior Under Malformed Inputs
Observed behavior from direct malformed-input tests:

- Blank patient name payload:
  - Result: HTTP 400, error code `domain_error`
- Extra unexpected field in patient payload:
  - Result: HTTP 422, error code `validation_error`
- Ingestion for unknown patient/device/session:
  - Result: HTTP 400, error code `http_error`

These responses confirm that validation and domain-level checks are now externally visible and machine-readable.

### 3.2 Exception Handling Strategy
The backend now includes centralized handlers for:
- `RequestValidationError` -> 422
- `HTTPException` -> passthrough status with normalized payload
- `ValueError` -> 400 (domain errors)
- uncaught `Exception` -> 500 with generic internal error

Each response includes `trace_id`, and each request exposes:
- `X-Request-Id`
- `X-Process-Time-Ms`

This provides a basis for tracing and production troubleshooting.

### 3.3 Reliability Risks Still Present
- In-memory store only: all state is lost on restart.
- No authentication/authorization yet.
- No persistence-backed queue/event log.
- No distributed load balancing or horizontal scaling strategy.

These are acceptable for a coursework prototype but should be addressed in production.

## 4. Performance Analysis

### 4.1 Test Setup
A dedicated stress test utility was implemented at:
- `backend/scripts/stress_test.py`

Test configuration used:
- Total requests: 200
- Concurrency: 20
- Payload: 64 sensor samples per request
- Endpoint under load: `POST /api/v1/ingest/live`

Generated report:
- `backend/stress_reports/stress_report_20260331_134622.json`
- `backend/stress_reports/stress_report_20260331_134622.md`

### 4.2 Measured Results
From the generated report:

- Throughput: 36.12 requests/second
- Success rate: 100.00%
- Error count: 0
- Average latency: 2546.95 ms
- P50 latency: 2243.35 ms
- P95 latency: 5046.06 ms
- P99 latency: 5283.49 ms
- Max latency: 5316.95 ms

### 4.3 Bottleneck Interpretation
Primary likely bottlenecks:
- CPU-bound per-request feature extraction and detection computations.
- Python GIL and single-process execution characteristics in this run.
- Additional state update and event creation overhead in the in-memory store.

Latency tail (P95/P99) indicates performance variance under concurrent bursts. The system remained functionally stable, but response times are high for real-time alerting expectations.

### 4.4 Improvement Opportunities
- Introduce asynchronous ingestion queue and decouple heavy processing.
- Pre-aggregate features or lower-frequency inference windows.
- Add multi-worker deployment and benchmark per-worker scaling.
- Cache immutable model artifacts and pre-allocate computation buffers.

## 5. Reflection: Long-Term Architectural Decision

The most consequential decision was keeping detection logic modular and separate from the API/store layer (`detection.py` boundary). This had two opposite effects:

Positive impact:
- Allowed rapid replacement path from rule-based logic to offline-trained model integration without changing endpoint contracts.
- Preserved API stability for mobile/web clients.

Long-term consequence:
- Created operational complexity around model dependencies and artifact lifecycle in deployment (import errors, artifact format assumptions, runtime fallback behavior).

In short, the abstraction was architecturally correct, but it shifted complexity from coding-time to deployment-time. That tradeoff is acceptable for maintainability, but requires stronger packaging, artifact versioning, and environment reproducibility.

## 6. Submission Checklist Mapping

- REST API (FastAPI): Completed.
- Input validation and schema enforcement: Completed with strict schema constraints and unknown-field rejection.
- Exception handling: Completed with centralized handlers and normalized error payloads.
- Stress testing (latency/throughput): Completed with automated script and generated reports.
- API documentation (Swagger/OpenAPI): Available at `/docs`, `/redoc`, `/openapi.json`.

This implementation satisfies the assignment's technical deliverables and provides evidence for report-based analysis sections.
