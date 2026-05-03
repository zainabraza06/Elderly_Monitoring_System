# Elderly Monitoring Frontend (Flutter Mobile)

This folder contains the Flutter mobile client for the Elderly Monitoring System.

The app is responsible for sensor capture and operator interaction, while AI inference and alert logic run on the backend.

## Frontend role in the full system

The mobile app performs these steps:

1. Save setup details (backend URL, patient info, device label).
2. Check local sensor availability (accelerometer and gyroscope).
3. Create patient/device/session through backend APIs.
4. Stream motion windows to backend ingestion endpoint.
5. Show returned detection state, telemetry summary, and active alerts.
6. Allow manual emergency alert creation.

## Tech stack

- Flutter (Material 3 UI)
- sensors_plus (accelerometer and gyroscope streams)
- http (REST communication)
- shared_preferences (local setup persistence)

## Sensor streaming behavior

The app streams batched sensor windows tuned to backend inference settings:

- target sample rate: 50 Hz
- window size: 128 samples (2.56 seconds)
- overlap step: 64 samples

This matches the backend detector and training pipeline assumptions.

## Backend API contract used by app

Main endpoints called by the app:

- `GET /api/v1/health`
- `POST /api/v1/patients`
- `GET /api/v1/patients/{id}`
- `POST /api/v1/devices`
- `GET /api/v1/devices/{id}`
- `POST /api/v1/sessions`
- `POST /api/v1/sessions/{id}/stop`
- `POST /api/v1/ingest/live`
- `POST /api/v1/alerts/manual`

## Prerequisites

- Flutter SDK installed
- backend running at a reachable URL (default backend port: 8000)

Check Flutter toolchain:

```bash
flutter doctor
```

## Run locally

```bash
flutter pub get
flutter run
```

Optional checks:

```bash
flutter test
```

## Backend URL setup by target device

- Android emulator: `http://10.0.2.2:8000`
- iOS simulator: `http://127.0.0.1:8000`
- Physical phone: `http://<your-computer-lan-ip>:8000`

## In-app setup guide

1. Enter backend URL, patient name, age, room label, and device label.
2. Press **Save Setup**.
3. Press **Check Sensors**.
4. Press **Start Monitoring**.
5. Keep device near monitored person for live telemetry.
6. Use **Emergency Trigger** when immediate manual escalation is needed.

## UI modules

- Patient Setup
- Sensor Access
- Live Session
- Risk Detection
- Emergency Trigger

Each section maps directly to backend actions and status.

## Error handling behavior

- request timeout handling (default 8 seconds)
- backend reachability checks before key operations
- sensor availability checks before streaming
- UI status banners for actionable failures

## Project structure

```text
lib/
	main.dart
	src/
		app.dart                      # UI shell and screens
		monitoring_controller.dart    # state and orchestration
		sensor_streaming_service.dart # sensor stream and batch windowing
		api_client.dart               # backend REST client
		models.dart                   # DTOs for requests/responses
```

## Troubleshooting

### Backend not reachable

- confirm backend is running on port 8000
- verify URL scheme (`http://`)
- on Android emulator, use `10.0.2.2` (not `localhost`)

### Sensor check fails

- grant motion/sensor permissions if prompted
- test on physical device if emulator lacks sensors

### Session starts but no detections

- verify phone remains moving enough to generate batches
- confirm backend `/api/v1/ingest/live` responses are successful
- check backend detector status at `/api/v1/detector/status`

## Notes

- This frontend does not run ML models locally.
- AI inference remains backend-hosted for consistency, observability, and easier model updates.
