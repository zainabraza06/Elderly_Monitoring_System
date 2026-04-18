# Stress Reports

This file tracks backend stress-test outputs and baseline performance evidence for the Elderly Monitoring System.

## Report generation

- Generator script: `flask_backend/scripts/stress_test.py`
- Primary output folder: `flask_backend/stress_reports/`
- Output format: one JSON + one Markdown file per run
- Filename pattern: `stress_report_YYYYMMDD_HHMMSS.json` and `stress_report_YYYYMMDD_HHMMSS.md`

Note: the script default output path in code is `backend/stress_reports`. In this repository, use `--output-dir flask_backend/stress_reports` so reports are written to the existing backend folder.

## Latest baseline report in this repository

- Timestamp (UTC): `2026-03-31T13:46:22.114038+00:00`
- Markdown report: `flask_backend/stress_reports/stress_report_20260331_134622.md`
- JSON report: `flask_backend/stress_reports/stress_report_20260331_134622.json`
- Target endpoint under load: `POST /api/v1/ingest/live`

## Baseline metrics summary

- Total requests: 200
- Concurrency: 20
- Samples per request: 64
- Duration: 5.536 s
- Throughput: 36.12 req/s
- Success count: 200
- Error count: 0
- Success rate: 100.00%
- Avg latency: 2546.95 ms
- P50 latency: 2243.35 ms
- P95 latency: 5046.06 ms
- P99 latency: 5283.49 ms
- Max latency: 5316.95 ms

## Run a new stress test

```bash
python flask_backend/scripts/stress_test.py --base-url http://127.0.0.1:8000 --requests 200 --concurrency 20 --samples-per-request 64 --output-dir flask_backend/stress_reports
```

## Suggested process for future runs

1. Keep the generated JSON and Markdown pair for each run.
2. Add a short entry here with timestamp, test parameters, and key metrics.
3. Compare p95/p99 and throughput across runs to detect regressions early.
