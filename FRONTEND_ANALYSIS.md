# Frontend Analysis (`web_frontend`)

## 1. Frontend Scope and Purpose

This frontend is a **Next.js dashboard** that visualizes live monitoring data from the backend.  
It is focused on operational visibility for the Elderly Monitoring System and does not perform ML inference locally.

Primary goals:
- Show live monitoring state (patients, alerts, events, latest telemetry).
- Keep data close to real-time using WebSocket updates plus API refreshes.
- Allow backend endpoint switching at runtime without code changes.

This analysis covers `web_frontend` only. The Flutter app in `app_frontend` is intentionally excluded here and will be documented separately.

---

## 2. Tech Stack and Runtime

- Framework: `Next.js 16` (App Router)
- Language: `TypeScript` + `React 19`
- Styling: `Tailwind CSS v4` with custom CSS variables and utility classes
- Linting: `ESLint 9` + `eslint-config-next` presets
- Build/Run: Node.js (`npm run dev`, `npm run build`, `npm run start`)
- Containerization: Dockerfile included for production image

Key package details are declared in `web_frontend/package.json`.

---

## 3. Frontend Directory Structure

High-signal structure (excluding platform/cache/output folders):

- `web_frontend/app/layout.tsx`  
  Root layout, global metadata, global stylesheet registration, font setup.

- `web_frontend/app/page.tsx`  
  Main and only application page. Contains all dashboard logic and UI sections.

- `web_frontend/app/globals.css`  
  Tailwind import and global theme tokens (`--background`, `--ink`, etc.).

- `web_frontend/public/*`  
  Static SVG assets (mostly template/static assets, not central to logic).

- `web_frontend/next.config.ts`  
  Next.js config placeholder (currently default/no custom behavior).

- `web_frontend/tsconfig.json`  
  TypeScript strictness and build-time compiler settings.

- `web_frontend/eslint.config.mjs`  
  Lint policy using Next.js core web vitals + TypeScript rules.

- `web_frontend/postcss.config.mjs`  
  PostCSS plugin wiring for Tailwind v4.

- `web_frontend/Dockerfile`  
  Production build and runtime container image definition.

- `web_frontend/captain-definition`  
  Deployment descriptor (points to frontend Dockerfile).

---

## 4. Routes and Navigation

Current route surface is intentionally minimal:

- `/` -> `app/page.tsx` (single dashboard screen)

No nested routes, auth pages, settings pages, or multi-view navigation exist yet.  
All dashboard sections are rendered in one long, responsive page.

---

## 5. File-by-File Functional Analysis

## `web_frontend/app/layout.tsx`

Responsibilities:
- Defines app-level metadata:
  - Title: `Elderly Monitor Dashboard`
  - Description: realtime telemetry/alert dashboard
- Loads Google fonts (`Geist`, `Geist_Mono`) and maps them to CSS variables.
- Wraps all pages in root HTML/body with base classes (`h-full`, `antialiased`, `min-h-full`, flex column).
- Imports `globals.css`.

Impact:
- Centralizes branding + typography baseline.
- Ensures consistent rendering and metadata for SEO/browser tab identity.

---

## `web_frontend/app/globals.css`

Responsibilities:
- Imports Tailwind: `@import "tailwindcss";`
- Defines global color/theme variables:
  - `--background`, `--foreground`, `--ink`, `--muted`, `--line`
- Bridges CSS variables to Tailwind theme tokens via `@theme inline`.
- Sets base `html` and `body` colors and default font-family.

Impact:
- Establishes visual language for the dashboard (warm neutral background + teal/earth accents).
- Enables utility-driven styling with semantic custom variables.

---

## `web_frontend/app/page.tsx`

This file contains **all business logic and all UI sections** for the web dashboard.

### A) State and Domain Types

Strongly typed models are declared in-file:
- `SystemSummary`
- `LivePatient`
- `AlertRecord`
- `SensorSample`
- `TelemetrySnapshot`
- `MonitorEvent`

Runtime state includes:
- Backend connection settings (`backendUrl`, input field value, connection label)
- Data buckets (`summary`, `patients`, `alerts`, `telemetry`, `events`)
- UX state (`loadError`, `isRefreshing`)

### B) Backend URL and Environment Logic

- Default backend is derived from `NEXT_PUBLIC_API_BASE_URL` or fallback `http://127.0.0.1:8000`.
- URL normalization adds scheme if missing and removes trailing slash.
- Value persists in `localStorage` (`ems_dashboard_backend_url`).
- User can change backend at runtime from UI and reconnect without rebuild/redeploy.

Why this matters:
- Useful for demos, staging/prod switching, and local testing.
- Reduces operator friction and config coupling.

### C) Data Fetching Strategy

`loadDashboard()` fetches five datasets concurrently:
- `/api/v1/summary`
- `/api/v1/monitor/patients/live`
- `/api/v1/alerts`
- `/api/v1/monitor/telemetry/recent?limit=1`
- `/api/v1/events/recent`

Implementation notes:
- Uses `Promise.all` to minimize latency across independent API calls.
- Uses `cache: "no-store"` to avoid stale dashboard data.
- Uses `startTransition` to keep UI responsive while committing heavy state updates.
- Trims data volume for rendering efficiency:
  - alerts limited to latest 8
  - events limited to latest 10
  - telemetry shows only most recent snapshot

### D) Realtime Stream Handling

- WebSocket target: `${backendUrl}/ws/monitor` with protocol conversion `http -> ws`.
- Connection lifecycle:
  - show connecting status
  - on open: mark as connected
  - on close/error: reconnect after 2 seconds
- Incoming events are prepended to local event feed (capped at 10).
- If event type is in the refresh set:
  - `patient.live.updated`
  - `alert.created`
  - `alert.acknowledged`
  - `alert.resolved`
  - `session.started`
  - `session.stopped`
  then the app triggers full REST refresh for consistency.

Why this hybrid approach is used:
- WebSocket provides low-latency activity updates.
- Periodic/triggered REST refresh ensures full-state correctness and prevents drift.

### E) Error and Loading Behavior

- Any fetch failure is surfaced in a dedicated error card.
- Refresh button displays `Refreshing...` while active.
- Empty states are explicitly handled for patients, alerts, telemetry, and events.

### F) Utility Helpers

- `formatTime()` converts timestamps to local `toLocaleTimeString()` or `N/A`.
- `severityClasses()` maps alert/patient severity to color styles.
- `severityLabel()` maps raw identifiers (e.g., `fall_detected`) to user-friendly labels.

---

## 6. UI Layout and Section-by-Section Details

Page composition in visual order:

1. **Hero Header**
   - Gradient background block with product title/subtitle.
   - Context copy describing stream + risk monitoring purpose.
   - Current backend URL and live connection state badge.

2. **Dashboard Source / Backend Selector**
   - Input for backend base URL.
   - `Apply URL` button updates storage + reconnects websocket.
   - `Reload Data` button manually refreshes all API datasets.

3. **Error Banner (conditional)**
   - Displays data load errors in prominent alert-style container.

4. **Summary Metrics Grid**
   - Patients, devices, active sessions, open alerts, acknowledged alerts, last event time.
   - Responsive card layout (`md` and `xl` breakpoints).

5. **Live Patients Panel**
   - Patient cards with:
     - name
     - room / recency
     - severity badge
     - latest message
     - score, fall probability, active alert count

6. **Open Alerts Panel**
   - Alert cards showing severity, status, message, patient ID, and created time.
   - Emphasizes operational actionability.

7. **Latest Telemetry Panel**
   - Meta tiles: patient/device/time/source/batch size/battery.
   - Tabular sample preview for acceleration and gyroscope vectors.
   - Includes timestamp + sensor tuple values (formatted to 2 decimals).

8. **Recent Events Panel**
   - Realtime event stream with event type + event time.
   - Uses deferred value to smooth rendering pressure during updates.

Design language:
- Rounded cards and soft background tones for readability.
- Strong color coding for severity and action states.
- Dense but structured information hierarchy for monitoring workflows.

---

## 7. Frontend-to-Backend Contract

## HTTP endpoints consumed

- `GET /api/v1/summary`
- `GET /api/v1/monitor/patients/live`
- `GET /api/v1/alerts`
- `GET /api/v1/monitor/telemetry/recent?limit=1`
- `GET /api/v1/events/recent`

## WebSocket endpoint consumed

- `GET /ws/monitor` (WebSocket upgrade)

Contract assumptions in frontend:
- Alert and patient severities are backend-provided strings with known categories.
- Event payload shape is generic map and may vary by type.
- Latest telemetry list endpoint may return an empty array.

---

## 8. AI/ML Usage in Frontend

Direct AI/ML usage in `web_frontend`: **none**.

What frontend does regarding AI:
- Displays inference outcomes (`severity`, `score`, `fall_probability`) already computed upstream.
- Consumes and visualizes these model-derived signals without running local model code.

Inference, feature engineering, preprocessing, and model evaluation are backend/pipeline concerns and are expected to be documented in the backend analysis file.

---

## 9. Configuration and Deployment Notes

## Runtime configuration

- Primary config var: `NEXT_PUBLIC_API_BASE_URL`
- Runtime override in UI persisted to browser local storage

## Build/deploy

- Docker build installs dependencies and runs production build.
- Container starts via `npm run start` on configured port (default `3000`).
- `captain-definition` integrates deployment with CapRover-style workflows.

---

## 10. Strengths, Gaps, and Improvement Opportunities

## Strengths

- Clean single-page monitoring experience.
- Practical realtime + REST consistency model.
- Strong TypeScript typing for API payloads.
- Runtime backend switching without redeploy.

## Gaps / Risks

- No authentication/authorization at frontend layer.
- No pagination/filtering/search for alerts/events.
- All logic concentrated in one large page file (maintainability risk as product grows).
- Limited resilience around malformed websocket payloads (JSON parse is assumed valid).

## Recommended next frontend refactors

1. Split `app/page.tsx` into section components (`SummaryCards`, `LivePatientsPanel`, etc.).
2. Introduce a typed API client layer in `lib/` to isolate fetch/error handling.
3. Add route-level expansion (e.g., alerts page, patient detail page, audit/event history).
4. Add charting for telemetry trends (not just latest batch table).
5. Add UI tests for key states (empty/loading/error/reconnect).

---

## 11. Quick Reference: All Frontend Functionalities

- Realtime dashboard landing page
- Backend URL runtime selection + persistence
- Manual data reload
- Live connection status indicator
- Summary KPIs
- Live patient risk cards
- Open alert feed
- Latest telemetry snapshot + sensor sample table
- Recent backend event timeline
- Empty/error state handling

