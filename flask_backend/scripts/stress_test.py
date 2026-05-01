from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx


@dataclass
class RequestResult:
    latency_ms: float
    status_code: int
    ok: bool
    error: str | None = None


@dataclass
class StressSummary:
    timestamp_utc: str
    base_url: str
    api_prefix: str
    total_requests: int
    concurrency: int
    samples_per_request: int
    duration_seconds: float
    throughput_rps: float
    success_count: int
    error_count: int
    success_rate_percent: float
    latency_ms_avg: float
    latency_ms_min: float
    latency_ms_max: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    status_code_counts: dict[int, int]
    sample_errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test the Elderly Monitoring FastAPI backend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-prefix", default="/api/v1")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--samples-per-request", type=int, default=64)
    parser.add_argument("--sampling-rate-hz", type=float, default=50.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=Path("backend/stress_reports"))
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = int(math.ceil((p / 100.0) * len(ordered))) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


def build_sample_batch(samples_per_request: int, sampling_rate_hz: float, patient_id: str, device_id: str, session_id: str) -> dict:
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    samples = []
    dt_ms = int(1000.0 / max(sampling_rate_hz, 1.0))

    for i in range(samples_per_request):
        # Synthetic motion with small random variation around normal walking.
        t = i / max(sampling_rate_hz, 1.0)
        samples.append(
            {
                "timestamp_ms": now_ms + i * dt_ms,
                "acc_x": 0.15 * math.sin(2 * math.pi * 1.2 * t) + random.uniform(-0.03, 0.03),
                "acc_y": 0.10 * math.cos(2 * math.pi * 1.1 * t) + random.uniform(-0.03, 0.03),
                "acc_z": 9.75 + random.uniform(-0.15, 0.15),
                "gyro_x": random.uniform(-25.0, 25.0),
                "gyro_y": random.uniform(-25.0, 25.0),
                "gyro_z": random.uniform(-25.0, 25.0),
            }
        )

    return {
        "patient_id": patient_id,
        "device_id": device_id,
        "session_id": session_id,
        "source": "stress_test",
        "sampling_rate_hz": sampling_rate_hz,
        "acceleration_unit": "m_s2",
        "gyroscope_unit": "dps",
        "battery_level": random.uniform(35, 95),
        "samples": samples,
    }


async def post_checked(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    resp = await client.post(url, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"{url} failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


async def bootstrap_entities(client: httpx.AsyncClient, api_prefix: str) -> tuple[str, str, str]:
    token = uuid4().hex[:8]

    patient = await post_checked(
        client,
        f"{api_prefix}/patients",
        {"full_name": f"Stress Test Patient {token}", "age": 72},
    )
    device = await post_checked(
        client,
        f"{api_prefix}/devices",
        {"label": f"Stress Device {token}", "platform": "stress_tool"},
    )
    session = await post_checked(
        client,
        f"{api_prefix}/sessions",
        {
            "patient_id": patient["id"],
            "device_id": device["id"],
            "started_by": "stress_tester",
            "sample_rate_hz": 50,
            "notes": "automated load test",
        },
    )

    return patient["id"], device["id"], session["id"]


async def run_load(
    client: httpx.AsyncClient,
    api_prefix: str,
    total_requests: int,
    concurrency: int,
    samples_per_request: int,
    sampling_rate_hz: float,
    patient_id: str,
    device_id: str,
    session_id: str,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one_request() -> RequestResult:
        ingest_payload = build_sample_batch(
            samples_per_request=samples_per_request,
            sampling_rate_hz=sampling_rate_hz,
            patient_id=patient_id,
            device_id=device_id,
            session_id=session_id,
        )

        t0 = perf_counter()
        try:
            async with semaphore:
                resp = await client.post(f"{api_prefix}/ingest/live", json=ingest_payload)
            elapsed_ms = (perf_counter() - t0) * 1000.0
            if resp.status_code >= 400:
                return RequestResult(
                    latency_ms=elapsed_ms,
                    status_code=resp.status_code,
                    ok=False,
                    error=resp.text[:240],
                )
            return RequestResult(latency_ms=elapsed_ms, status_code=resp.status_code, ok=True)
        except Exception as exc:
            elapsed_ms = (perf_counter() - t0) * 1000.0
            return RequestResult(latency_ms=elapsed_ms, status_code=0, ok=False, error=str(exc))

    tasks = [asyncio.create_task(one_request()) for _ in range(total_requests)]
    return await asyncio.gather(*tasks)


def summarize(
    results: list[RequestResult],
    started_at: float,
    ended_at: float,
    base_url: str,
    api_prefix: str,
    total_requests: int,
    concurrency: int,
    samples_per_request: int,
) -> StressSummary:
    latencies = [item.latency_ms for item in results]
    success_count = sum(1 for item in results if item.ok)
    error_count = len(results) - success_count

    status_code_counts: dict[int, int] = {}
    for item in results:
        status_code_counts[item.status_code] = status_code_counts.get(item.status_code, 0) + 1

    sample_errors = [item.error for item in results if item.error][:5]
    duration_seconds = max(ended_at - started_at, 1e-9)

    return StressSummary(
        timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
        base_url=base_url,
        api_prefix=api_prefix,
        total_requests=total_requests,
        concurrency=concurrency,
        samples_per_request=samples_per_request,
        duration_seconds=duration_seconds,
        throughput_rps=total_requests / duration_seconds,
        success_count=success_count,
        error_count=error_count,
        success_rate_percent=(success_count / max(len(results), 1)) * 100.0,
        latency_ms_avg=(sum(latencies) / max(len(latencies), 1)),
        latency_ms_min=min(latencies) if latencies else 0.0,
        latency_ms_max=max(latencies) if latencies else 0.0,
        latency_ms_p50=percentile(latencies, 50),
        latency_ms_p95=percentile(latencies, 95),
        latency_ms_p99=percentile(latencies, 99),
        status_code_counts=status_code_counts,
        sample_errors=sample_errors,
    )


def write_report(summary: StressSummary, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"stress_report_{stamp}.json"
    md_path = output_dir / f"stress_report_{stamp}.md"

    json_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    md = [
        "# Stress Test Report",
        "",
        f"- Timestamp (UTC): {summary.timestamp_utc}",
        f"- Base URL: {summary.base_url}",
        f"- API Prefix: {summary.api_prefix}",
        f"- Total Requests: {summary.total_requests}",
        f"- Concurrency: {summary.concurrency}",
        f"- Samples Per Request: {summary.samples_per_request}",
        "",
        "## Results",
        "",
        f"- Duration: {summary.duration_seconds:.3f} s",
        f"- Throughput: {summary.throughput_rps:.2f} req/s",
        f"- Success: {summary.success_count}",
        f"- Errors: {summary.error_count}",
        f"- Success Rate: {summary.success_rate_percent:.2f}%",
        "",
        "## Latency (ms)",
        "",
        f"- Avg: {summary.latency_ms_avg:.2f}",
        f"- Min: {summary.latency_ms_min:.2f}",
        f"- P50: {summary.latency_ms_p50:.2f}",
        f"- P95: {summary.latency_ms_p95:.2f}",
        f"- P99: {summary.latency_ms_p99:.2f}",
        f"- Max: {summary.latency_ms_max:.2f}",
        "",
        "## Status Code Counts",
        "",
    ]

    for code, count in sorted(summary.status_code_counts.items(), key=lambda x: x[0]):
        md.append(f"- {code}: {count}")

    if summary.sample_errors:
        md.extend(["", "## Sample Errors", ""])
        for err in summary.sample_errors:
            md.append(f"- {err}")

    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path


async def async_main(args: argparse.Namespace) -> int:
    timeout = httpx.Timeout(args.timeout_seconds)

    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=timeout) as client:
        health = await client.get(f"{args.api_prefix}/health")
        if health.status_code >= 400:
            print(f"Health check failed ({health.status_code}): {health.text[:300]}")
            return 2

        patient_id, device_id, session_id = await bootstrap_entities(client, args.api_prefix)

        started = perf_counter()
        results = await run_load(
            client=client,
            api_prefix=args.api_prefix,
            total_requests=args.requests,
            concurrency=args.concurrency,
            samples_per_request=args.samples_per_request,
            sampling_rate_hz=args.sampling_rate_hz,
            patient_id=patient_id,
            device_id=device_id,
            session_id=session_id,
        )
        ended = perf_counter()

        # Best-effort session cleanup.
        await client.post(
            f"{args.api_prefix}/sessions/{session_id}/stop",
            json={"stopped_by": "stress_tester", "note": "stress test complete"},
        )

    summary = summarize(
        results=results,
        started_at=started,
        ended_at=ended,
        base_url=args.base_url,
        api_prefix=args.api_prefix,
        total_requests=args.requests,
        concurrency=args.concurrency,
        samples_per_request=args.samples_per_request,
    )

    json_path, md_path = write_report(summary, args.output_dir)

    print("Stress test completed.")
    print(f"Throughput: {summary.throughput_rps:.2f} req/s")
    print(
        "Latency (ms): "
        f"avg={summary.latency_ms_avg:.2f}, p50={summary.latency_ms_p50:.2f}, "
        f"p95={summary.latency_ms_p95:.2f}, p99={summary.latency_ms_p99:.2f}"
    )
    print(f"Success rate: {summary.success_rate_percent:.2f}%")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")

    return 0 if summary.error_count == 0 else 1


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
