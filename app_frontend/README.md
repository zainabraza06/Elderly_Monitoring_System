# Elderly Monitor Mobile App

Flutter mobile client for the Elderly Monitoring System. The app is designed to:

- register a patient and phone device with the FastAPI backend
- start a live monitoring session
- stream accelerometer and gyroscope data from the phone in near real time
- show backend risk classification and latest motion metrics
- trigger a manual emergency alert from the phone

## Backend URL notes

- Android emulator: `http://10.0.2.2:8000`
- iOS simulator: `http://127.0.0.1:8000`
- Physical device: `http://<your-computer-lan-ip>:8000`

## Main flow

1. Open the app and fill in the backend URL, patient details, room, and device label.
2. Tap `Save Setup` to persist the configuration on the phone.
3. Tap `Check Sensors` to verify the phone can read the accelerometer and gyroscope.
4. Tap `Start Monitoring` to create or reuse backend patient/device records and open a session.
5. Keep the phone with the monitored user so accelerometer and gyroscope data can be streamed.
6. Watch the `Risk Detection` and `Emergency Trigger` sections for live alerts.

## Monitoring dashboard

Use the standalone Next.js dashboard in the `webapp/` folder:

```text
http://localhost:3000
```

It shows:

- live patient monitoring state
- recent telemetry batches from the mobile app
- alert feed and recent backend events

## Flutter commands

```bash
flutter pub get
flutter run
flutter test
```
