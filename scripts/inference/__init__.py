"""Shared inference pipeline (fall binary → ADL or fall-type)."""

from .motion_pipeline import InferenceArtifacts, load_artifacts, run_inference

__all__ = ["InferenceArtifacts", "load_artifacts", "run_inference"]
