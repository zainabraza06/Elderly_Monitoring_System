"""FastAPI: fall / ADL / fall-type inference."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import sklearn
import xgboost
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from flask_backend.app.schemas_motion import MotionInferenceRequest, MotionInferenceResponse
from flask_backend.app.services.motion_xgb_service import InferenceArtifacts, load_artifacts, run_inference
from flask_backend.app.settings import inference_manifest_path, model_root


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
        )
        return MotionInferenceResponse(**raw)
    except ValueError as e:
        raise HTTPException(422, detail=str(e)) from e
