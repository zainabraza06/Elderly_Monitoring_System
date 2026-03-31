# Elderly Monitor Dashboard

Standalone Next.js web dashboard for the Elderly Monitoring System.

It connects to the FastAPI backend and shows:

- live patient monitoring state
- recent telemetry batches from the mobile app
- alert activity and backend event feed

## Getting Started

Make sure the FastAPI backend is running on `http://127.0.0.1:8000` or set a custom backend URL:

```bash
set NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

You can also change the backend URL directly from the dashboard UI after it loads.

Then run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to view the dashboard.

## Main files

- `app/page.tsx`: realtime dashboard UI
- `app/layout.tsx`: app metadata and root layout
- `app/globals.css`: shared theme and global styling

## Backend endpoints used

- `/api/v1/summary`
- `/api/v1/monitor/patients/live`
- `/api/v1/monitor/telemetry/recent`
- `/api/v1/alerts`
- `/api/v1/events/recent`
- `/ws/monitor`
