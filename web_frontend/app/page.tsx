"use client";

import {
  startTransition,
  useDeferredValue,
  useEffect,
  useEffectEvent,
  useState,
} from "react";

const defaultBackendBaseUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");
const backendStorageKey = "ems_dashboard_backend_url";

type SystemSummary = {
  total_patients: number;
  total_devices: number;
  active_sessions: number;
  open_alerts: number;
  acknowledged_alerts: number;
  last_event_at: string | null;
};

type LivePatient = {
  patient_id: string;
  patient_name: string;
  severity: string;
  score: number;
  fall_probability: number;
  last_message: string;
  last_ingested_at: string | null;
  active_alert_ids: string[];
};

type AlertRecord = {
  id: string;
  patient_id: string;
  severity: string;
  status: string;
  message: string;
  score: number;
  created_at: string;
};

type SensorSample = {
  timestamp_ms: number | null;
  acc_x: number;
  acc_y: number;
  acc_z: number;
  gyro_x: number;
  gyro_y: number;
  gyro_z: number;
};

type TelemetrySnapshot = {
  patient_id: string;
  patient_name: string;
  session_id: string;
  device_id: string;
  source: string;
  sampling_rate_hz: number;
  acceleration_unit: string;
  gyroscope_unit: string;
  battery_level: number | null;
  received_at: string;
  samples_in_last_batch: number;
  latest_samples: SensorSample[];
};

type MonitorEvent = {
  id?: string;
  type: string;
  created_at: string;
  payload: Record<string, unknown>;
};

const refreshEventTypes = new Set([
  "patient.live.updated",
  "alert.created",
  "alert.acknowledged",
  "alert.resolved",
  "session.started",
  "session.stopped",
]);

function normalizeBackendUrl(rawValue: string) {
  const trimmed = rawValue.trim();
  if (!trimmed) {
    return defaultBackendBaseUrl;
  }

  const withScheme =
    trimmed.startsWith("http://") || trimmed.startsWith("https://")
      ? trimmed
      : `http://${trimmed}`;

  return withScheme.replace(/\/$/, "");
}

async function fetchJson<T>(apiBaseUrl: string, path: string): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function formatTime(value: string | null | undefined) {
  if (!value) return "N/A";
  return new Date(value).toLocaleTimeString();
}

function severityClasses(severity: string) {
  switch (severity) {
    case "medium":
      return "bg-[#f0a542] text-white";
    case "high_risk":
      return "bg-[#de6b48] text-white";
    case "fall_detected":
      return "bg-[#b53b34] text-white";
    default:
      return "bg-[#1b9b8b] text-white";
  }
}

function severityLabel(severity: string) {
  switch (severity) {
    case "high_risk":
      return "High Risk";
    case "fall_detected":
      return "Fall Detected";
    default:
      return severity.replaceAll("_", " ");
  }
}

export default function Home() {
  const [backendUrl, setBackendUrl] = useState(defaultBackendBaseUrl);
  const [backendUrlInput, setBackendUrlInput] = useState(defaultBackendBaseUrl);
  const [summary, setSummary] = useState<SystemSummary | null>(null);
  const [patients, setPatients] = useState<LivePatient[]>([]);
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot | null>(null);
  const [events, setEvents] = useState<MonitorEvent[]>([]);
  const [connectionState, setConnectionState] = useState("Connecting...");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const deferredEvents = useDeferredValue(events);
  const apiBase = `${backendUrl}/api/v1`;
  const wsUrl = `${backendUrl.replace(/^http/, "ws")}/ws/monitor`;

  useEffect(() => {
    const savedBackendUrl = window.localStorage.getItem(backendStorageKey);
    if (!savedBackendUrl) {
      return;
    }

    const normalized = normalizeBackendUrl(savedBackendUrl);
    setBackendUrl(normalized);
    setBackendUrlInput(normalized);
  }, []);

  const loadDashboard = useEffectEvent(async () => {
    setIsRefreshing(true);

    try {
      const [summaryData, patientData, alertData, telemetryData, eventData] =
        await Promise.all([
          fetchJson<SystemSummary>(apiBase, "/summary"),
          fetchJson<LivePatient[]>(apiBase, "/monitor/patients/live"),
          fetchJson<AlertRecord[]>(apiBase, "/alerts"),
          fetchJson<TelemetrySnapshot[]>(apiBase, "/monitor/telemetry/recent?limit=1"),
          fetchJson<MonitorEvent[]>(apiBase, "/events/recent"),
        ]);

      startTransition(() => {
        setSummary(summaryData);
        setPatients(patientData);
        setAlerts(alertData.slice(0, 8));
        setTelemetry(telemetryData[0] ?? null);
        setEvents(eventData.slice(0, 10));
        setLoadError(null);
      });
    } catch (error) {
      startTransition(() => {
        setLoadError(error instanceof Error ? error.message : "Unable to load dashboard.");
      });
    } finally {
      setIsRefreshing(false);
    }
  });

  const handleRealtimeEvent = useEffectEvent((event: MonitorEvent) => {
    startTransition(() => {
      setEvents((current) => [event, ...current].slice(0, 10));
      if (event.type === "telemetry.ingested") {
        setTelemetry(event.payload as unknown as TelemetrySnapshot);
      }
    });

    if (refreshEventTypes.has(event.type)) {
      void loadDashboard();
    }
  });

  const applyBackendUrl = useEffectEvent(() => {
    const normalized = normalizeBackendUrl(backendUrlInput);
    window.localStorage.setItem(backendStorageKey, normalized);
    setBackendUrl(normalized);
    setBackendUrlInput(normalized);
    setConnectionState("Reconnecting...");
  });

  useEffect(() => {
    let isActive = true;
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (!isActive) return;

      setConnectionState(`Connecting to ${backendUrl}...`);
      socket = new WebSocket(wsUrl);
      socket.onopen = () => {
        if (isActive) {
          setConnectionState("Live stream connected");
        }
      };
      socket.onmessage = (message) => {
        const parsed = JSON.parse(message.data) as MonitorEvent;
        handleRealtimeEvent(parsed);
      };
      socket.onclose = () => {
        if (!isActive) return;
        setConnectionState("Reconnecting...");
        retryTimer = setTimeout(connect, 2000);
      };
      socket.onerror = () => {
        socket?.close();
      };
    };

    void loadDashboard();
    connect();

    return () => {
      isActive = false;
      if (retryTimer) clearTimeout(retryTimer);
      socket?.close();
    };
  }, [backendUrl]);

  const summaryCards = summary
    ? [
        ["Patients", summary.total_patients],
        ["Devices", summary.total_devices],
        ["Active Sessions", summary.active_sessions],
        ["Open Alerts", summary.open_alerts],
        ["Acknowledged", summary.acknowledged_alerts],
        ["Last Event", formatTime(summary.last_event_at)],
      ]
    : [];

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_#e2f2ea_0%,_#f4eee4_48%,_#faf7f2_100%)] px-5 py-6 text-[var(--ink)] sm:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <section className="rounded-[2rem] bg-[linear-gradient(135deg,_#0f8579,_#174b6d)] p-6 text-white shadow-[0_20px_60px_rgba(23,75,109,0.18)]">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
            <div className="max-w-3xl">
              <p className="text-sm uppercase tracking-[0.25em] text-[#d7efe8]">
                Live Monitoring
              </p>
              <h1 className="mt-3 text-4xl font-semibold tracking-tight">
                Elderly Monitor Dashboard
              </h1>
              <p className="mt-4 max-w-2xl text-sm leading-7 text-[#edf7f4] sm:text-base">
                Watch the mobile app stream sensor batches into FastAPI, follow
                patient risk levels, and react to fall or emergency alerts in
                real time.
              </p>
              <p className="mt-4 text-sm text-[#d7efe8]">
                Backend: <span className="font-semibold">{backendUrl}</span>
              </p>
            </div>
            <div className="rounded-full bg-white/15 px-4 py-3 text-sm font-semibold">
              {connectionState}
            </div>
          </div>
        </section>

        <section className="rounded-[1.5rem] border border-[var(--line)] bg-white p-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <h2 className="text-2xl font-semibold tracking-tight">Dashboard Source</h2>
              <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
                Point the dashboard at any running FastAPI backend without editing
                source files or environment variables.
              </p>
            </div>
            <div className="flex w-full flex-col gap-3 lg:max-w-2xl">
              <label className="text-sm font-semibold text-[var(--muted)]">
                Backend URL
              </label>
              <div className="flex flex-col gap-3 sm:flex-row">
                <input
                  value={backendUrlInput}
                  onChange={(event) => setBackendUrlInput(event.target.value)}
                  className="min-h-12 flex-1 rounded-full border border-[#d9cbb9] bg-[#f8f4ee] px-4 text-sm outline-none transition focus:border-[#0f8579]"
                  placeholder="http://127.0.0.1:8000"
                />
                <button
                  type="button"
                  onClick={applyBackendUrl}
                  className="min-h-12 rounded-full bg-[#0f8579] px-5 text-sm font-semibold text-white transition hover:bg-[#0d7469]"
                >
                  Apply URL
                </button>
                <button
                  type="button"
                  onClick={() => void loadDashboard()}
                  className="min-h-12 rounded-full border border-[#d9cbb9] px-5 text-sm font-semibold text-[var(--ink)] transition hover:bg-[#f8f4ee]"
                >
                  {isRefreshing ? "Refreshing..." : "Reload Data"}
                </button>
              </div>
            </div>
          </div>
        </section>

        {loadError ? (
          <section className="rounded-[1.5rem] border border-[#f0c5bf] bg-[#fce2df] px-5 py-4 text-[#6f2a25]">
            {loadError}
          </section>
        ) : null}

        <section className="grid gap-4 md:grid-cols-3 xl:grid-cols-6">
          {summaryCards.map(([label, value]) => (
            <article
              key={String(label)}
              className="rounded-[1.5rem] border border-[var(--line)] bg-white p-5"
            >
              <p className="text-sm text-[var(--muted)]">{label}</p>
              <p className="mt-3 text-3xl font-semibold tracking-tight">
                {String(value)}
              </p>
            </article>
          ))}
        </section>

        <section className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
          <article className="rounded-[1.5rem] border border-[var(--line)] bg-white p-5">
            <h2 className="text-2xl font-semibold tracking-tight">Live Patients</h2>
            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
              Current risk state for each monitored patient.
            </p>
            <div className="mt-5 grid gap-4">
              {patients.length === 0 ? (
                <p className="text-sm text-[var(--muted)]">No live patient data yet.</p>
              ) : (
                patients.map((patient) => (
                  <div
                    key={patient.patient_id}
                    className="rounded-[1.25rem] bg-[#f8f4ee] p-4"
                  >
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div>
                        <div className="text-lg font-semibold">{patient.patient_name}</div>
                        <div className="mt-1 text-sm text-[var(--muted)]">
                          Last data {formatTime(patient.last_ingested_at)}
                        </div>
                      </div>
                      <span
                        className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] ${severityClasses(patient.severity)}`}
                      >
                        {severityLabel(patient.severity)}
                      </span>
                    </div>
                    <p className="mt-3 text-sm leading-6 text-[#30453b]">
                      {patient.last_message}
                    </p>
                    <div className="mt-4 flex flex-wrap gap-3 text-sm text-[var(--muted)]">
                      <span>Score {(patient.score * 100).toFixed(0)}%</span>
                      <span>Fall {(patient.fall_probability * 100).toFixed(0)}%</span>
                      <span>{patient.active_alert_ids.length} active alerts</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </article>

          <article className="rounded-[1.5rem] border border-[var(--line)] bg-white p-5">
            <h2 className="text-2xl font-semibold tracking-tight">Open Alerts</h2>
            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
              Fall detections and manually triggered emergencies.
            </p>
            <div className="mt-5 grid gap-4">
              {alerts.length === 0 ? (
                <p className="text-sm text-[var(--muted)]">No alerts yet.</p>
              ) : (
                alerts.map((alert) => (
                  <div key={alert.id} className="rounded-[1.25rem] bg-[#fff4ec] p-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <span
                        className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] ${severityClasses(alert.severity)}`}
                      >
                        {severityLabel(alert.severity)}
                      </span>
                      <span className="text-sm font-semibold uppercase tracking-[0.18em] text-[#6f473b]">
                        {alert.status}
                      </span>
                    </div>
                    <p className="mt-3 text-base font-semibold">{alert.message}</p>
                    <div className="mt-3 text-sm text-[#7b5a4f]">
                      Patient {alert.patient_id} - {formatTime(alert.created_at)}
                    </div>
                  </div>
                ))
              )}
            </div>
          </article>
        </section>

        <section className="grid gap-5 xl:grid-cols-[1.15fr_0.85fr]">
          <article className="rounded-[1.5rem] border border-[var(--line)] bg-white p-5">
            <h2 className="text-2xl font-semibold tracking-tight">Latest Telemetry</h2>
            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
              Most recent sensor batch captured from the mobile app.
            </p>
            {telemetry ? (
              <>
                <div className="mt-5 grid gap-3 md:grid-cols-3">
                  <div className="rounded-[1.25rem] bg-[#f8f4ee] p-4">
                    <p className="text-sm text-[var(--muted)]">Patient</p>
                    <p className="mt-2 font-semibold">{telemetry.patient_name}</p>
                  </div>
                  <div className="rounded-[1.25rem] bg-[#f8f4ee] p-4">
                    <p className="text-sm text-[var(--muted)]">Device</p>
                    <p className="mt-2 font-semibold">{telemetry.device_id}</p>
                  </div>
                  <div className="rounded-[1.25rem] bg-[#f8f4ee] p-4">
                    <p className="text-sm text-[var(--muted)]">Received</p>
                    <p className="mt-2 font-semibold">{formatTime(telemetry.received_at)}</p>
                  </div>
                  <div className="rounded-[1.25rem] bg-[#f8f4ee] p-4">
                    <p className="text-sm text-[var(--muted)]">Source</p>
                    <p className="mt-2 font-semibold">{telemetry.source}</p>
                  </div>
                  <div className="rounded-[1.25rem] bg-[#f8f4ee] p-4">
                    <p className="text-sm text-[var(--muted)]">Batch Size</p>
                    <p className="mt-2 font-semibold">{telemetry.samples_in_last_batch}</p>
                  </div>
                  <div className="rounded-[1.25rem] bg-[#f8f4ee] p-4">
                    <p className="text-sm text-[var(--muted)]">Battery</p>
                    <p className="mt-2 font-semibold">
                      {telemetry.battery_level ?? "N/A"}
                    </p>
                  </div>
                </div>
                <div className="mt-5 overflow-hidden rounded-[1.25rem] border border-[#eee4d7]">
                  <div className="grid grid-cols-[160px_1fr_1fr] bg-[#f8f4ee] px-4 py-3 text-xs font-semibold uppercase tracking-[0.2em] text-[var(--muted)]">
                    <span>Timestamp</span>
                    <span>Acceleration</span>
                    <span>Gyroscope</span>
                  </div>
                  <div className="divide-y divide-[#eee4d7]">
                    {telemetry.latest_samples.map((sample, index) => (
                      <div
                        key={`${sample.timestamp_ms ?? index}`}
                        className="grid grid-cols-[160px_1fr_1fr] gap-3 px-4 py-3 text-sm"
                      >
                        <span>{sample.timestamp_ms ?? "N/A"}</span>
                        <span>
                          {sample.acc_x.toFixed(2)}, {sample.acc_y.toFixed(2)},{" "}
                          {sample.acc_z.toFixed(2)}
                        </span>
                        <span>
                          {sample.gyro_x.toFixed(2)}, {sample.gyro_y.toFixed(2)},{" "}
                          {sample.gyro_z.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <p className="mt-5 text-sm text-[var(--muted)]">
                No telemetry batches have arrived yet.
              </p>
            )}
          </article>

          <article className="rounded-[1.5rem] border border-[var(--line)] bg-white p-5">
            <h2 className="text-2xl font-semibold tracking-tight">Recent Events</h2>
            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">
              Realtime backend activity from the WebSocket feed.
            </p>
            <div className="mt-5 grid gap-3">
              {deferredEvents.length === 0 ? (
                <p className="text-sm text-[var(--muted)]">No events yet.</p>
              ) : (
                deferredEvents.map((event, index) => (
                  <div
                    key={event.id ?? `${event.type}-${index}`}
                    className="rounded-[1.25rem] bg-[#f8f4ee] p-4"
                  >
                    <div className="text-sm uppercase tracking-[0.2em] text-[var(--muted)]">
                      {event.type}
                    </div>
                    <div className="mt-2 font-semibold">{formatTime(event.created_at)}</div>
                  </div>
                ))
              )}
            </div>
          </article>
        </section>
      </div>
    </main>
  );
}
