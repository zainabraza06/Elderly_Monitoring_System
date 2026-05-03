"""FastAPI: fall / ADL / fall-type inference."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager

import sklearn
import xgboost
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from flask_backend.app.schemas_fall_feedback import FallFeedbackAck, FallFeedbackEvent
from flask_backend.app.schemas_motion import MotionInferenceRequest, MotionInferenceResponse
from flask_backend.app.services.motion_xgb_service import InferenceArtifacts, load_artifacts, run_inference
from flask_backend.app.settings import inference_manifest_path, model_root, repo_root


def _versions() -> dict[str, str]:
    return {
        "numpy": __import__("numpy").__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }


_state: dict[str, InferenceArtifacts | str | None] = {"art": None, "load_error": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        _state["art"] = load_artifacts(inference_manifest_path(), model_root())
        _state["load_error"] = None
    except Exception as exc:
        _state["art"] = None
        _state["load_error"] = str(exc)
    yield


app = FastAPI(title="Motion inference", version="1.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    res = await call_next(request)
    res.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
    return res


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "inference_ready": _state.get("art") is not None,
        "load_error": _state.get("load_error"),
        "versions": _versions(),
    }


@app.get("/api/v1/inference/status")
def inference_status():
    art = _state.get("art")
    if art is None:
        raise HTTPException(503, detail=_state.get("load_error", "not loaded"))
    return {
        "loaded": True,
        "schema_version": art.manifest.get("schema_version"),
        "enhanced_feature_dim": art.enhanced_dim,
        "fall_type_raw_dim": art.fall_type_dim,
        "fall_threshold": art.fall_threshold,
        "model_root": str(model_root()),
    }


@app.post("/api/v1/inference/motion", response_model=MotionInferenceResponse)
def inference_motion(body: MotionInferenceRequest):
    art = _state.get("art")
    if art is None:
        raise HTTPException(503, detail=f"Inference not loaded: {_state.get('load_error')}")
    try:
        raw = run_inference(
            art,
            body.enhanced_features,
            body.fall_type_features,
            predict_fall_type=body.predict_fall_type,
            acc_window=body.acc_window,
            gyro_window=body.gyro_window,
            ori_window=body.ori_window,
        )
        return MotionInferenceResponse(**raw)
    except ValueError as e:
        raise HTTPException(422, detail=str(e)) from e


@app.post("/api/v1/events/fall-feedback", response_model=FallFeedbackAck)
def fall_feedback(body: FallFeedbackEvent):
    """Append elder/caretaker fall confirmation to JSONL for QA and future model tuning."""
    log_dir = repo_root() / "data" / "feedback"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "fall_events.jsonl"
    row = body.model_dump()
    row["_server_logged_at"] = FallFeedbackAck().logged_at
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return FallFeedbackAck()
