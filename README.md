# SafeStep AI — elderly monitoring (MobiAct-style IMU baselines)

IMU-based **fall detection**, **ADL classification**, and **fall-type** models (MobiAct-style training), a **FastAPI** backend (monitoring REST + ML), and a **Flutter** mobile app (sensors + caregiver/elder flows).

---

## Table of contents

1. [Overview](#overview)
2. [How the pieces fit together](#how-the-pieces-fit-together)
3. [Repository layout](#repository-layout)
4. [Prerequisites](#prerequisites)
5. [Backend (FastAPI)](#backend-fastapi)
6. [Flutter mobile app](#flutter-mobile-app)
7. [Machine learning & inference](#machine-learning--inference)
8. [Models and manifest](#models-and-manifest)
9. [Testing](#testing)
10. [Deployment (CapRover & GitHub Actions)](#deployment-caprover--github-actions)
11. [Documentation index](#documentation-index)
12. [Disclaimer](#disclaimer)

---

## Overview

| Layer | What it does |
|-------|----------------|
| **Training** (`scripts/`) | Builds fall-binary, ADL, and fall-type models from annotated MobiAct-style data; writes joblibs under `models/` and reports under `results/`. |
| **Inference** (`scripts/inference/`) | Single pipeline: load artifacts from manifest → fall probability → ADL or fall-type branch. Shared by tests and backend. |
| **Backend** (`flask_backend/`) | FastAPI: SQLite persistence, patient/device/session lifecycle, **live sensor ingest** with server-side ML, caregiver/admin JWT flows, alerts, optional `POST /api/v1/inference/motion`. |
| **Mobile** (`app_frontend/`) | Flutter: accelerometer/gyro/orientation batching, `POST /api/v1/ingest/live`, UI for setup, monitoring, emergencies. |

Canonical system diagram and dimensions: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## How the pieces fit together

**Live monitoring (default product path)**

1. App saves backend URL + patient/device info (`shared_preferences`).
2. App calls `POST /api/v1/patients`, `POST /api/v1/devices`, `POST /api/v1/sessions`.
3. **Sensor streaming:** `SensorStreamingService` batches windows (~50 Hz target, 128-sample windows, 64-sample step — see [`app_frontend/lib/src/sensor_streaming_service.dart`](app_frontend/lib/src/sensor_streaming_service.dart)).
4. Each batch is **`POST /api/v1/ingest/live`** with `samples` (acc/gyro/orientation fields per reading).
5. Backend converts samples → 116-D features + resampled windows → **`run_inference`** → updates **`patient_live`**, may open **alerts** / **fall_incidents**, returns `detection`, `live_status`, `telemetry`.

**Optional direct motion API**

- `POST /api/v1/inference/motion` accepts precomputed **116-D** `enhanced_features` and optional windows for server-side fall-type features (see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §2). Used when you call inference without going through full ingest payload shaping from raw samples.

**On-device 116-D (parity path)**

- Flutter can compute 116-D features aligned with Python (`motion_feature_extractor.dart`) for experiments or hybrid flows; primary integration in this repo still centers on **ingest** + server-side feature extraction from batched samples.

---

## Repository layout

| Path | Role |
|------|------|
| [`app_frontend/`](app_frontend/) | Flutter mobile client (Material 3, `sensors_plus`, `http`). |
| [`flask_backend/`](flask_backend/) | FastAPI app (Python package name `flask_backend`). Default model dir: `flask_backend/models/`. |
| [`flask_backend/captain-definition`](flask_backend/captain-definition) | CapRover / Dockerfile lines for backend image. |
| [`scripts/baseline_fall/`](scripts/baseline_fall/) | 116-D enhanced features, fall-binary training, viz. |
| [`scripts/baseline_adl/`](scripts/baseline_adl/) | ADL multiclass training + viz. |
| [`scripts/baseline_falltype/`](scripts/baseline_falltype/) | 263-D fall-type vector, MI selection, 4-class training + viz (runtime import on server for fall-type from windows). |
| [`scripts/inference/`](scripts/inference/) | **`motion_pipeline.py`**: `load_artifacts`, `run_inference` (canonical). |
| [`scripts/run_training.py`](scripts/run_training.py) | CLI: `fall-detection`, `adl`, `fall-type`, `all`, `sync-manifest`. |
| [`scripts/sync_inference_manifest.py`](scripts/sync_inference_manifest.py) | Regenerates root `models/inference_manifest.json` from artifact shapes. |
| [`models/`](models/) | Repo-root exported weights + `inference_manifest.json` (training output paths; backend may use copies under `flask_backend/models/` for deploy). |
| [`results/`](results/) | Training plots/CSVs (regenerable). |
| [`tests/`](tests/) | `pytest`: HTTP inference, feature dimensions, motion artifacts, baseline imports. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | End-to-end architecture, roles roadmap, feedback stub. |
| [`.github/workflows/`](.github/workflows/) | `deploy-backend.yml` (CapRover). `deploy-frontend.yml` targets **`web_frontend/`**, which is **not** present; mobile app lives under **`app_frontend/`** (see [Deployment](#deployment-caprover--github-actions)). |

SQLite DB file (created on first use): `data/elder_monitor.db` under the **repository root** resolved from `flask_backend.app.settings.repo_root()` (i.e. parent of the `flask_backend` package directory). Ensure `data/` is writable in production or mount a volume there.

---

## Prerequisites

- **Python 3.11+** (matches Docker base image in `captain-definition`).
- **Flutter SDK** (for `app_frontend/`) — run `flutter doctor`.
- **MobiAct v2** (or compatible layout) only if you retrain — path to `Annotated Data` for training scripts.

---

## Backend (FastAPI)

### Install and run (local)

```bash
pip install -r flask_backend/requirements.txt
cd flask_backend
uvicorn flask_backend.app.main:app --host 0.0.0.0 --port 8000
```

- Health: `GET /api/v1/health` (includes `inference_ready`, library versions).
- Inference load status: `GET /api/v1/inference/status` (if exposed — check [`flask_backend/app/main.py`](flask_backend/app/main.py) and routes file).

### Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `MODEL_ROOT` | Directory containing model joblibs (default: `flask_backend/models`). |
| `INFERENCE_MANIFEST` | Path to `inference_manifest.json` (default: `MODEL_ROOT`/… or packaged manifest next to models). |

Exact resolution: [`flask_backend/app/settings.py`](flask_backend/app/settings.py).

### REST API surface (monitoring app)

Implemented in [`flask_backend/app/monitoring_routes.py`](flask_backend/app/monitoring_routes.py) (prefix as mounted on app; all below are route paths on the FastAPI app):

**Health / detector**

- `GET /api/v1/health` (in `main.py`)
- `GET /api/v1/inference/status` (in `main.py`)
- `PUT /api/v1/detector/config` (detector tuning for ingest path)

**Auth & roles**

- `POST /api/v1/auth/caregiver/signup`, `POST /api/v1/auth/caregiver/login`
- `POST /api/v1/auth/admin/login`, `POST /api/v1/auth/elder/login`
- `POST /api/v1/auth/caregiver/patient-credentials`
- `GET /api/v1/caregiver/my-patients`, `DELETE /api/v1/caregiver/my-patients/{patient_id}`

**Patients, devices, sessions**

- `POST /api/v1/patients`, `GET /api/v1/patients/{patient_id}`
- `POST /api/v1/devices`, `GET /api/v1/devices/{device_id}`
- `POST /api/v1/sessions`, `POST /api/v1/sessions/{session_id}/stop`

**Live data & alerts**

- **`POST /api/v1/ingest/live`** — primary path: batched sensor samples → ML → `patient_live` + optional alerts.
- `POST /api/v1/alerts/manual`
- `GET /api/v1/alerts`, `POST /api/v1/alerts/{alert_id}/acknowledge`, `POST /api/v1/alerts/{alert_id}/resolve`
- `GET /api/v1/summary`
- `GET /api/v1/monitor/patients/live` — caregiver live board
- `POST /api/v1/patients/me/location` — elder GPS for map

**ML**

- **`POST /api/v1/inference/motion`** — JSON with `enhanced_features` (+ optional windows); returns structured inference response.

**Admin**

- `GET /api/v1/admin/dashboard`, `GET/POST/DELETE .../caregivers`, `GET/POST/DELETE .../patients`

**Feedback (stub)**

- `POST /api/v1/events/fall-feedback` — append-only JSONL under `data/feedback/` (see architecture doc).

CORS is currently permissive for development (`allow_origins=["*"]` in `main.py`); tighten for production.

---

## Flutter mobile app

Location: [`app_frontend/`](app_frontend/).

### Commands

```bash
cd app_frontend
flutter pub get
flutter run
flutter test   # optional
```

### Backend URL by environment

| Environment | Example base URL |
|-------------|------------------|
| Android emulator | `http://10.0.2.2:8000` |
| iOS simulator | `http://127.0.0.1:8000` |
| Physical device | `http://<PC-LAN-IP>:8000` |

### Typical in-app flow

1. Enter backend URL, patient name, age, room/device labels → **Save Setup**.
2. **Check Sensors** (accelerometer + gyroscope).
3. **Start Monitoring** → creates/uses patient, device, session → streams batches to **`/api/v1/ingest/live`**.
4. UI shows risk/detection/telemetry from responses; **Emergency Trigger** → `POST /api/v1/alerts/manual`.

### Important `lib/src` files

| File | Role |
|------|------|
| `monitoring_controller.dart` | Orchestrates API + sensor service + motion helper. |
| `sensor_streaming_service.dart` | `sensors_plus` streams, windowing, batch callback. |
| `api_client.dart` | REST client (`ingestLiveBatch`, patients, sessions, alerts, …). |
| `models.dart` | Request/response DTOs. |
| `motion_feature_extractor.dart` | 116-D parity with Python (experiments / hybrid). |
| `motion_inference_helper.dart` | Optional calls to inference API. |
| `roles/app_roles.dart` | Caregiver vs elder roles. |

More detail, timeouts, and troubleshooting: [`app_frontend/README.md`](app_frontend/README.md).

---

## Machine learning & inference

### Training entrypoint

Install training dependencies:

```bash
pip install -r scripts/requirements-training.txt
```

**Windows (PowerShell):**

```powershell
$env:PYTHONPATH = "scripts"
py scripts/run_training.py all --data-root "D:\path\to\MobiAct_Dataset_v2.0\Annotated Data"
py scripts/sync_inference_manifest.py
```

**Linux / macOS:**

```bash
export PYTHONPATH=scripts
python scripts/run_training.py all --data-root "/path/to/MobiAct_Dataset_v2.0/Annotated Data"
python scripts/sync_inference_manifest.py
```

Individual stages:

```text
py scripts/run_training.py fall-detection --data-root "...\Annotated Data"
py scripts/run_training.py adl            --data-root "...\Annotated Data"
py scripts/run_training.py fall-type      --data-root "...\Annotated Data"
py scripts/run_training.py sync-manifest
```

Pipeline index: [`scripts/README_PIPELINE.md`](scripts/README_PIPELINE.md).

### Inference in Python

With `PYTHONPATH=scripts`:

```python
from pathlib import Path
from inference.motion_pipeline import load_artifacts, run_inference

art = load_artifacts(Path("models/inference_manifest.json"), Path("models"))
# art = load_artifacts(..., Path("flask_backend/models"))  # if using packaged models
```

---

## Models and manifest

- **`models/inference_manifest.json`** (repo root): schema version, artifact relative paths, `enhanced_feature_dim`, `fall_type_raw_dim`, threshold. Updated after training via `sync_inference_manifest.py`.
- **Backend** loads from `flask_backend/models/` by default when deployed (see `settings.py`); keep manifest and joblibs in sync.
- **Docker / CapRover:** image copies `flask_backend`, `scripts/inference`, and `scripts/baseline_falltype` so `ingest` can run fall-type feature extraction on the server.

---

## Testing

From repository root:

```bash
pip install -r requirements.txt
pytest tests -q
```

Notable suites:

| Test module | Focus |
|-------------|--------|
| `tests/test_inference_http.py` | HTTP API / inference wiring |
| `tests/test_feature_dimensions.py` | Feature vector lengths vs manifest |
| `tests/test_motion_artifacts.py` | Motion / ingest edge cases |
| `tests/test_fall_detection_fall_type_models.py` | Model stack smoke tests |
| `tests/test_baseline_*` | Import / unit checks for training packages |

---

## Deployment (CapRover & GitHub Actions)

### Backend

Workflow: [`.github/workflows/deploy-backend.yml`](.github/workflows/deploy-backend.yml).

- Copies [`flask_backend/captain-definition`](flask_backend/captain-definition) to repo root for the build.
- **`deploy.tar`** includes: `captain-definition`, `flask_backend`, `scripts/inference`, `scripts/baseline_falltype` (required for `import inference` and fall-type windows in container).

**GitHub Actions secrets**

| Secret | Meaning |
|--------|---------|
| `CAPROVER_SERVER` | Captain URL, e.g. `https://captain.your-domain.com` |
| `CAPROVER_BACKEND_APP` | CapRover app name (exact match) |
| `CAPROVER_BACKEND_APP_TOKEN` | App **Deployment** tab token (not GitHub PAT) |

Container command: `uvicorn flask_backend.app.main:app --host 0.0.0.0 --port 8000` (see captain-definition).

### Frontend workflow note

[`.github/workflows/deploy-frontend.yml`](.github/workflows/deploy-frontend.yml) expects **`web_frontend/captain-definition`** and a **`web_frontend/`** tree (Next.js-style). **This repository uses `app_frontend/` (Flutter), not `web_frontend/`.** The frontend CapRover workflow will fail until you either add a `web_frontend` project with its own `captain-definition` or change the workflow to build/deploy your Flutter/web strategy (e.g. static build + nginx, or store-only mobile).

---

## Documentation index

| Document | Content |
|----------|---------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Sensor → features → models; mermaid flow; caregiver/elder/admin roadmap; feedback stub. |
| [`scripts/README_PIPELINE.md`](scripts/README_PIPELINE.md) | Training commands and module map. |
| [`app_frontend/README.md`](app_frontend/README.md) | Flutter setup, API list, troubleshooting. |

---

## Disclaimer

This project is intended for **research, coursework, and prototyping**. It is **not** a certified medical device or emergency service. Clinical or production eldercare use requires regulatory review, validated hardware, security hardening, and operational procedures beyond this repository.
