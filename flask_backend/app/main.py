"""FastAPI entrypoint: XGBoost motion inference (fall / ADL / fall-type)."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

import sklearn
import xgboost
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from flask_backend.app.schemas_motion import MotionInferenceRequest, MotionInferenceResponse
from flask_backend.app.services.motion_xgb_service import InferenceArtifacts, load_artifacts, run_inference
from flask_backend.app.settings import inference_manifest_path, model_root


def _pkg_versions() -> dict[str, str]:
    return {
        "numpy": __import__("numpy").__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": __import__("joblib").__version__,
    }


_state: dict[str, InferenceArtifacts | str | None] = {"art": None, "load_error": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Single-thread BLAS reduces cross-platform numerical drift for sklearn transforms.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        _state["art"] = load_artifacts(inference_manifest_path(), model_root())
        _state["load_error"] = None
    except Exception as exc:
        _state["art"] = None
        _state["load_error"] = str(exc)
    yield


app = FastAPI(
    title="Motion inference (XGBoost)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


def get_artifacts() -> InferenceArtifacts:
    art = _state.get("art")
    if art is None:
        err = _state.get("load_error", "unknown error")
        raise HTTPException(
            status_code=503,
            detail=f"Inference artifacts not loaded: {err}",
        )
    return art


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "inference_ready": _state.get("art") is not None,
        "load_error": _state.get("load_error"),
        "versions": _pkg_versions(),
    }


@app.get("/api/v1/inference/status")
def inference_status(art: Annotated[InferenceArtifacts, Depends(get_artifacts)]):
    return {
        "loaded": True,
        "manifest": art.manifest.get("schema_version"),
        "enhanced_feature_dim": art.enhanced_dim,
        "fall_type_raw_dim": art.fall_type_dim,
        "fall_threshold": art.fall_threshold,
        "library_versions": _pkg_versions(),
        "model_root": str(model_root()),
    }


@app.post("/api/v1/inference/motion", response_model=MotionInferenceResponse)
def inference_motion(body: MotionInferenceRequest):
    """Validate JSON body first; return 503 only after body is structurally valid."""
    art = _state.get("art")
    if art is None:
        err = _state.get("load_error", "unknown error")
        raise HTTPException(
            status_code=503,
            detail=f"Inference artifacts not loaded: {err}",
        )
    try:
        raw = run_inference(
            art,
            body.enhanced_features,
            body.fall_type_features,
            predict_fall_type=body.predict_fall_type,
        )
        return MotionInferenceResponse(**raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


