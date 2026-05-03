"""FastAPI smoke tests (loads models on lifespan)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from flask_backend.app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    if not body.get("inference_ready"):
        pytest.skip(body.get("load_error") or "inference artifacts not loaded")


def test_inference_status(client: TestClient, inference_manifest: dict) -> None:
    r = client.get("/api/v1/inference/status")
    if r.status_code == 503:
        pytest.skip(r.json().get("detail", "artifacts unavailable"))
    assert r.status_code == 200
    j = r.json()
    assert j["loaded"] is True
    assert j["enhanced_feature_dim"] == int(inference_manifest["enhanced_feature_dim"])
    assert j["fall_type_raw_dim"] == int(inference_manifest["fall_type_raw_dim"])


def test_motion_validation_error(client: TestClient) -> None:
    """Too few floats -> 422 once models are loaded; 503 if artifacts unavailable."""
    r = client.post(
        "/api/v1/inference/motion",
        json={"enhanced_features": [0.0, 0.0], "predict_fall_type": False},
    )
    assert r.status_code in (422, 503)
    assert "detail" in r.json()


def test_motion_zeros_ok(client: TestClient, inference_manifest: dict) -> None:
    ft_dim = int(inference_manifest["fall_type_raw_dim"])
    en_dim = int(inference_manifest["enhanced_feature_dim"])
    r = client.post(
        "/api/v1/inference/motion",
        json={
            "enhanced_features": [0.0] * en_dim,
            "predict_fall_type": True,
            "fall_type_features": [0.0] * ft_dim,
        },
    )
    if r.status_code == 503:
        pytest.skip("inference service unavailable")
    assert r.status_code == 200
    body = r.json()
    assert "is_fall" in body
    assert "branch" in body
    assert body["schema_version"] == inference_manifest["schema_version"]
