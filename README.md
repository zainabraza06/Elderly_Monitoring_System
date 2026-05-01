# Elderly Monitoring System

An end-to-end AI-assisted monitoring platform for elderly care, with a strong backend-first design and a practical ML pipeline for fall-risk detection.

This repository combines:

- an offline AI training pipeline for motion-risk models
- a FastAPI backend for real-time ingestion, detection, alerting, and live monitoring APIs
- a Flutter mobile frontend that streams accelerometer and gyroscope windows to the backend

## Why this project matters

The system is built to support near real-time patient monitoring with auditable server-side decision logic.

- AI pipeline creates reproducible model artifacts from public motion datasets.
- Backend owns inference, state transitions, and alert lifecycle.
- Frontend acts as a data collection and operator interface layer.

This architecture keeps critical safety logic centralized in the backend while allowing frontend clients to stay lightweight.

## Current repository layout

```text
Elderly_Monitoring_System/
	app_frontend/                  # Flutter mobile app (sensor streaming client)
	flask_backend/                 # FastAPI backend (ingestion, detection, alerts, websocket)
		stress_reports/              # Generated stress test reports (JSON + Markdown)
	scripts/                       # AI pipeline scripts (preprocess -> features -> models)
	results/
		artifacts/                   # Generated model and dataset artifacts
	.github/workflows/             # CapRover deployment workflows
	requirements.txt               # Pipeline/training dependencies
	README_pipeline.md             # Legacy pipeline quick guide
```

## System architecture

```text
Phone Sensors (acc + gyro)
					|
					v
Flutter App (app_frontend)
	- 50 Hz sampling
	- 128-sample windows, 64-step overlap
					|
					v
FastAPI Backend (flask_backend)
	- /api/v1/ingest/live
	- RealtimeDetector
	- Alert + Session + Telemetry state
	- WebSocket broadcast (/ws/monitor)
					|
					+--> REST APIs for monitoring clients
					+--> Event stream for dashboards

Offline AI Pipeline (scripts)
	data -> preprocessing -> feature extraction -> model training
										 -> fall_detector_bundle.joblib
										 -> consumed by backend at runtime
```

## AI pipeline (core)

The AI pipeline is in the `scripts/` folder and trains models used by backend inference.

### 1. Data preprocessing

- File: `scripts/preprocessing.py`
- Inputs expected under `data/`:
	- `SisFall_dataset`
	- `MobiAct_Dataset_v2.0`
	- `UCI HAR Dataset`
- Output:
	- `results/artifacts/windows.parquet`
	- `results/artifacts/windows_signals.npz`

The pipeline normalizes sensor streams to 50 Hz and creates fixed windows (default 2.56 seconds, 50% overlap).

### 2. Feature engineering

- File: `scripts/feature_extraction.py`
- Produces rich motion features:
	- time-domain, frequency-domain, entropy, gait, tremor, asymmetry, proxy clinical indicators
- Output:
	- `results/artifacts/features.parquet` (fallback CSV if parquet engine unavailable)

### 3. Model training

- File: `scripts/modeling.py`
- Main trained models:
	- fall detector (Balanced Random Forest)
	- MET class classifier
	- proxy multi-output regressor
- Key outputs:
	- `results/artifacts/fall_detector.joblib`
	- `results/artifacts/fall_detector_bundle.joblib`
	- `results/artifacts/fall_detector_metadata.json`
	- `results/artifacts/metrics_summary.csv`
	- confusion matrix and ROC plots

### 4. Backend runtime integration

- File: `flask_backend/app/detection.py`
- `RealtimeDetector` loads offline artifacts from `results/artifacts`.
- Runtime mode:
	- `offline_model` when model bundle is available and loadable
	- `rule_based` fallback when bundle/dependencies are missing

### Run full pipeline

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --data-root data --sisfall-only
```

For larger experiments, remove `--sisfall-only` and provide complete datasets.

## Backend (FastAPI, backend-first design)

The backend is the system of record for monitoring state, inference, and alerting.

### Backend capabilities

- patient/device/session lifecycle management
- live sensor batch ingestion
- realtime detection and severity scoring
- auto/manual alert management (open, acknowledge, resolve)
- telemetry snapshots and monitoring summaries
- websocket stream for live updates

### Backend robustness features

- strict request schema validation (`extra = forbid`)
- centralized exception handling with consistent error payload
- request tracing and timing headers:
	- `X-Request-Id`
	- `X-Process-Time-Ms`
- detector status introspection endpoint

### Main API groups

- `GET /api/v1/health`
- `GET /api/v1/summary`
- `POST /api/v1/patients`, `GET /api/v1/patients`
- `POST /api/v1/devices`, `GET /api/v1/devices`
- `POST /api/v1/sessions`, `POST /api/v1/sessions/{session_id}/stop`
- `POST /api/v1/ingest/live`
- `GET /api/v1/alerts`, `POST /api/v1/alerts/manual`
- `POST /api/v1/alerts/{alert_id}/acknowledge`
- `POST /api/v1/alerts/{alert_id}/resolve`
- `GET /api/v1/detector/config`, `PUT /api/v1/detector/config`
- `GET /api/v1/detector/status`
- `GET /api/v1/events/recent`
- `WS /ws/monitor`

### Run backend locally

```bash
python -m pip install -r flask_backend/requirements.txt
python -m uvicorn flask_backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Docs after startup:

- Swagger: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### Complete API reference

- Full standalone API documentation: `API_DOCUMENTATION.md`

### Verify detector mode

```bash
curl http://127.0.0.1:8000/api/v1/detector/status
```

Check `mode` and `runtime_loaded`:

- `offline_model` + `true` means trained artifact is active
- `rule_based` + `false` means artifact is missing or failed to load

## Stress testing and reports

Stress-test reporting is part of the backend quality evidence for this project.

- Script: `flask_backend/scripts/stress_test.py`
- Report index: `STRESS_REPORTS.md` (root)
- Report output directory: `flask_backend/stress_reports/`

Latest generated report in this repository:

- `flask_backend/stress_reports/stress_report_20260331_134622.md`
- `flask_backend/stress_reports/stress_report_20260331_134622.json`

Baseline metrics (from the report above):

- Throughput: 36.12 req/s
- Success rate: 100.00% (200/200)
- Avg latency: 2546.95 ms
- P50 latency: 2243.35 ms
- P95 latency: 5046.06 ms
- P99 latency: 5283.49 ms

Run a new stress test and write reports to the backend report folder:

```bash
python flask_backend/scripts/stress_test.py --base-url http://127.0.0.1:8000 --requests 200 --concurrency 20 --samples-per-request 64 --output-dir flask_backend/stress_reports
```

## Frontend documentation

### Mobile frontend (Flutter)

Path: `app_frontend/`

The Flutter app:

- saves backend URL and patient/device setup locally
- probes accelerometer and gyroscope availability
- creates patient/device/session through backend APIs
- streams live sensor windows to `/api/v1/ingest/live`
- shows live detection severity and supports manual emergency alerts

Run locally:

```bash
cd app_frontend
flutter pub get
flutter run
```

Backend URL notes:

- Android emulator: `http://10.0.2.2:8000`
- iOS simulator: `http://127.0.0.1:8000`
- Physical device: `http://<your-lan-ip>:8000`

### Web dashboard note

Some legacy docs/workflows reference `web_frontend/` or `webapp/`, but that folder is not present in the current workspace snapshot.

If you add a separate dashboard frontend folder, update:

- `.github/workflows/deploy-frontend.yml`
- backend/mobile docs that reference dashboard path names

## End-to-end local run (recommended)

1. Train artifacts (AI pipeline)

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --data-root data --sisfall-only
```

2. Start backend

```bash
python -m pip install -r flask_backend/requirements.txt
python -m uvicorn flask_backend.app.main:app --host 0.0.0.0 --port 8000
```

3. Launch Flutter app

```bash
cd app_frontend
flutter pub get
flutter run
```

4. Validate runtime

- open backend docs at `/docs`
- check `/api/v1/health`
- check `/api/v1/detector/status`
- stream from app and confirm alerts/events update

## Deployment (CapRover + GitHub Actions)

### Backend deployment

- Workflow: `.github/workflows/deploy-backend.yml`
- Trigger paths:
	- `flask_backend/**`
	- workflow file itself
- Required secrets:
	- `CAPROVER_SERVER`
	- `CAPROVER_BACKEND_APP`
	- `CAPROVER_BACKEND_APP_TOKEN`

### Frontend deployment workflow status

- Workflow exists: `.github/workflows/deploy-frontend.yml`
- Current path filter and archive target: `web_frontend/**`

Because `web_frontend/` is not present in this workspace snapshot, adjust this workflow when your actual frontend folder is confirmed.

## Troubleshooting

### Detector stays in `rule_based`

- Ensure `results/artifacts/fall_detector_bundle.joblib` exists.
- Confirm backend can read `results/artifacts` from its working directory.
- Confirm dependencies in `flask_backend/requirements.txt` are installed.

### Pipeline says data root does not exist

- Create `data/` and place datasets with expected folder names.
- Re-run `python scripts/run_pipeline.py --data-root data`.

### Parquet errors during feature extraction

- Install a parquet engine:

```bash
python -m pip install pyarrow
```

### Flutter cannot build Android

- Run `flutter doctor`.
- Install Android SDK and accept licenses.

## Tech stack

- Backend: FastAPI, Pydantic v2, Uvicorn
- AI/ML: NumPy, SciPy, scikit-learn, imbalanced-learn, PyWavelets, pandas
- Frontend mobile: Flutter, sensors_plus, http, shared_preferences
- Deployment: GitHub Actions, CapRover, Docker

## Notes

- Backend persistence uses MongoDB through `flask_backend/app/mongo_store.py` when configured.
- If MongoDB is unavailable, backend automatically falls back to in-memory state.
- AI inference is backend-hosted; frontend clients should not duplicate model logic.
