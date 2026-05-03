# Elderly monitoring — SisFall / MobiAct baselines

## Suggested GitHub “About” description

Copy this into **Repository → ⚙ Settings → General → Description** (GitHub does not store the short description in git):

> IMU-based fall detection, ADL classification, and fall-type models trained on MobiAct-style data, with FastAPI inference and a Flutter mobile client for elder monitoring.

**Suggested topics:** `fall-detection`, `elderly-monitoring`, `imu-sensors`, `mobiact`, `fastapi`, `flutter`, `xgboost`, `machine-learning`, `python`

---

## Overview

This repository contains **baseline ML pipelines** (fall vs non-fall, ADL multiclass, four-class fall type), **frozen model artifacts**, a **FastAPI** backend for health checks, patient/device/session APIs, and **motion inference** aligned with on-device **116-D** feature extraction in the Flutter app. Training and evaluation scripts target **MobiAct**-style annotated accelerometer data; architecture and data flow are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Highlights

- **Three-model cascade:** binary fall → ADL when not fall → fall type when fall (see architecture doc).
- **Single inference contract:** [`scripts/inference/motion_pipeline.py`](scripts/inference/motion_pipeline.py) and the backend service load paths from [`models/inference_manifest.json`](models/inference_manifest.json).
- **Flutter client** ([`app_frontend/`](app_frontend/)): sensor windows, REST integration, caregiver vs patient flows (see [`app_frontend/README.md`](app_frontend/README.md)).
- **CI-friendly tests** under [`tests/`](tests/) for inference, features, and motion artifacts.

## Repository layout

| Path | Role |
|------|------|
| `scripts/baseline_fall/` | Enhanced features, fall-binary training |
| `scripts/baseline_adl/` | ADL multiclass training |
| `scripts/baseline_falltype/` | Fall-type features, selection, 4-class training |
| `scripts/inference/` | Canonical `load_artifacts` + `run_inference` |
| `scripts/run_training.py` | Entrypoint: `fall-detection`, `adl`, `fall-type`, `all`, `sync-manifest` |
| `flask_backend/` | FastAPI app (package name retained as `flask_backend`) |
| `app_frontend/` | Flutter mobile app |
| `models/` | Exported joblibs + `inference_manifest.json` |
| `results/` | Training outputs (figures, CSV; safe to regenerate) |

## Quick start

### Backend (local)

```bash
pip install -r flask_backend/requirements.txt
cd flask_backend
uvicorn flask_backend.app.main:app --host 0.0.0.0 --port 8000
```

Health check: `GET http://localhost:8000/api/v1/health`

### Training (MobiAct data)

Install training deps, set `PYTHONPATH` to include `scripts`, then run the stack (adjust `--data-root` to your **Annotated Data** folder):

```bash
pip install -r scripts/requirements-training.txt
set PYTHONPATH=scripts
py scripts/run_training.py all --data-root "path\to\MobiAct_Dataset_v2.0\Annotated Data"
py scripts/sync_inference_manifest.py
```

More detail: [`scripts/README_PIPELINE.md`](scripts/README_PIPELINE.md).

### Tests

```bash
pip install -r requirements.txt
pytest tests -q
```

### Frontend

See [`app_frontend/README.md`](app_frontend/README.md) (`flutter doctor`, run on device/emulator, point the app at your backend URL).

## Deployment (CapRover)

Backend deploy is driven by [`.github/workflows/deploy-backend.yml`](.github/workflows/deploy-backend.yml) using [`flask_backend/captain-definition`](flask_backend/captain-definition). Configure GitHub Actions secrets:

- `CAPROVER_SERVER` — CapRover captain URL (e.g. `https://captain.your-domain.com`)
- `CAPROVER_BACKEND_APP` — app name exactly as in CapRover
- `CAPROVER_BACKEND_APP_TOKEN` — **Deployment → App token** for that app (not your GitHub token)

## Documentation

- [System architecture](docs/ARCHITECTURE.md) — sensor → API → models, roles, feedback stub
- [Training / inference index](scripts/README_PIPELINE.md)

## Disclaimer

This project is for **research and coursework** contexts. Clinical or emergency use requires separate validation, compliance, and operational design.
