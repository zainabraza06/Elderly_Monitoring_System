from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any

from .schemas import (
    AccelerationUnit,
    AlertSeverity,
    DetectionResult,
    DetectorConfig,
    GyroscopeUnit,
    SensorBatchIn,
)


MPS2_TO_G = 1.0 / 9.80665
RAD_S_TO_DPS = 57.29577951308232
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import joblib
    import numpy as np
    from scipy import signal as scipy_signal
    from scripts.feature_extraction import build_single_feature_row

    OFFLINE_RUNTIME_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - depends on optional runtime deps
    joblib = None
    np = None
    scipy_signal = None
    build_single_feature_row = None
    OFFLINE_RUNTIME_IMPORT_ERROR = str(exc)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MotionWindow:
    rows: list[list[float]]
    peak_acc_g: float
    peak_gyro_dps: float
    peak_jerk_g_per_s: float
    stillness_ratio: float


@dataclass(slots=True)
class OfflineModelBundle:
    model: object
    feature_columns: list[str]
    target_fs: int
    window_sec: float
    overlap: float
    window_size_samples: int
    step_size_samples: int
    model_name: str
    artifact_path: str
    met_model: object | None = None
    met_label_encoder: object | None = None
    proxy_regressor: object | None = None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class RealtimeDetector:
    """Realtime detector that prefers the offline-trained model when available."""

    def __init__(self, artifacts_dir: str | Path = "results/artifacts") -> None:
        self._artifacts_dir = Path(artifacts_dir)
        self._runtime: OfflineModelBundle | None = None
        self._runtime_signature: tuple[str, float, float] | None = None
        self._status_reason = "Offline detector bundle has not been loaded yet."
        self._refresh_runtime_if_needed(force=True)

    def status(self) -> dict[str, Any]:
        self._refresh_runtime_if_needed()
        runtime = self._runtime
        return {
            "mode": "offline_model" if runtime is not None else "rule_based",
            "runtime_loaded": runtime is not None,
            "artifacts_dir": str(self._artifacts_dir),
            "reason": self._status_reason,
            "import_error": OFFLINE_RUNTIME_IMPORT_ERROR,
            "bundle_path": str(self._artifacts_dir / "fall_detector_bundle.joblib"),
            "legacy_model_path": str(self._artifacts_dir / "fall_detector.joblib"),
            "metadata_path": str(self._artifacts_dir / "fall_detector_metadata.json"),
            "window_size_samples": runtime.window_size_samples if runtime is not None else None,
            "step_size_samples": runtime.step_size_samples if runtime is not None else None,
            "target_fs": runtime.target_fs if runtime is not None else None,
            "feature_count": len(runtime.feature_columns) if runtime is not None else 0,
            "model_name": runtime.model_name if runtime is not None else None,
            "artifact_path": runtime.artifact_path if runtime is not None else None,
            "has_activity_classifier": runtime.met_model is not None if runtime is not None else False,
            "has_proxy_regressor": runtime.proxy_regressor is not None if runtime is not None else False,
        }

    def analyze(self, payload: SensorBatchIn, config: DetectorConfig) -> DetectionResult:
        self._refresh_runtime_if_needed()

        samples = payload.samples
        if len(samples) < config.min_samples:
            return self._not_enough_samples_result(len(samples))

        motion = self._prepare_motion_window(payload, config)

        if self._runtime is not None:
            model_result = self._analyze_with_offline_model(payload, motion, config)
            if model_result is not None:
                return model_result

        return self._analyze_rule_based(payload, motion, config)

    def _not_enough_samples_result(self, sample_count: int) -> DetectionResult:
        now = datetime.now(timezone.utc)
        return DetectionResult(
            severity=AlertSeverity.low,
            score=0.0,
            fall_probability=0.0,
            peak_acc_g=0.0,
            peak_gyro_dps=0.0,
            peak_jerk_g_per_s=0.0,
            stillness_ratio=0.0,
            samples_analyzed=sample_count,
            message="Not enough samples to evaluate live risk yet.",
            reasons=["Collect a slightly larger sample window before evaluating."],
            detected_at=now,
        )

    def _prepare_motion_window(self, payload: SensorBatchIn, config: DetectorConfig) -> MotionWindow:
        acc_scale = 1.0 if payload.acceleration_unit == AccelerationUnit.g else MPS2_TO_G
        gyro_scale = 1.0 if payload.gyroscope_unit == GyroscopeUnit.dps else RAD_S_TO_DPS
        dt = 1.0 / payload.sampling_rate_hz

        rows: list[list[float]] = []
        acc_mag: list[float] = []
        gyro_mag: list[float] = []

        for sample in payload.samples:
            ax = sample.acc_x * acc_scale
            ay = sample.acc_y * acc_scale
            az = sample.acc_z * acc_scale
            gx = sample.gyro_x * gyro_scale
            gy = sample.gyro_y * gyro_scale
            gz = sample.gyro_z * gyro_scale

            acc_magnitude = sqrt(ax * ax + ay * ay + az * az)
            gyro_magnitude = sqrt(gx * gx + gy * gy + gz * gz)

            rows.append([ax, ay, az, acc_magnitude, gx, gy, gz, gyro_magnitude])
            acc_mag.append(acc_magnitude)
            gyro_mag.append(gyro_magnitude)

        jerk_values = [
            abs(acc_mag[idx] - acc_mag[idx - 1]) / max(dt, 1e-6)
            for idx in range(1, len(acc_mag))
        ]

        peak_acc_g = max(acc_mag) if acc_mag else 0.0
        peak_gyro_dps = max(gyro_mag) if gyro_mag else 0.0
        peak_jerk_g_per_s = max(jerk_values) if jerk_values else 0.0

        tail_start = max(0, int(len(acc_mag) * 0.7))
        tail_acc = acc_mag[tail_start:] or acc_mag
        tail_gyro = gyro_mag[tail_start:] or gyro_mag

        stillness_hits = 0
        for acc_value, gyro_value in zip(tail_acc, tail_gyro):
            if (
                abs(acc_value - 1.0) <= config.stillness_acc_delta_g
                and gyro_value <= config.stillness_gyro_threshold_dps
            ):
                stillness_hits += 1
        stillness_ratio = stillness_hits / max(len(tail_acc), 1)

        return MotionWindow(
            rows=rows,
            peak_acc_g=peak_acc_g,
            peak_gyro_dps=peak_gyro_dps,
            peak_jerk_g_per_s=peak_jerk_g_per_s,
            stillness_ratio=stillness_ratio,
        )

    def _analyze_rule_based(
        self,
        payload: SensorBatchIn,
        motion: MotionWindow,
        config: DetectorConfig,
    ) -> DetectionResult:
        impact_score = min(motion.peak_acc_g / config.impact_threshold_g, 1.0)
        jerk_score = min(motion.peak_jerk_g_per_s / config.jerk_threshold_g_per_s, 1.0)
        gyro_score = min(motion.peak_gyro_dps / config.gyro_threshold_dps, 1.0)
        score = min(1.0, 0.42 * impact_score + 0.23 * jerk_score + 0.15 * gyro_score + 0.20 * motion.stillness_ratio)

        reasons: list[str] = []
        if motion.peak_acc_g >= config.impact_threshold_g:
            reasons.append(f"Impact peak {motion.peak_acc_g:.2f}g crossed the configured threshold.")
        if motion.peak_jerk_g_per_s >= config.jerk_threshold_g_per_s:
            reasons.append(f"Jerk peak {motion.peak_jerk_g_per_s:.2f} g/s suggests abrupt motion.")
        if motion.peak_gyro_dps >= config.gyro_threshold_dps:
            reasons.append(f"Gyroscope peak {motion.peak_gyro_dps:.1f} dps indicates strong rotation.")
        if motion.stillness_ratio >= 0.5:
            reasons.append(f"Post-event stillness ratio is {motion.stillness_ratio:.2f}, which can follow a fall.")

        if (
            score >= config.fall_score
            and motion.peak_acc_g >= config.impact_threshold_g * 1.1
            and motion.peak_jerk_g_per_s >= config.jerk_threshold_g_per_s * 0.85
        ):
            severity = AlertSeverity.fall_detected
            message = "Fall-like pattern detected from live sensor window."
        elif score >= config.high_risk_score:
            severity = AlertSeverity.high_risk
            message = "High-risk movement detected. Immediate monitoring is recommended."
        elif score >= config.medium_risk_score:
            severity = AlertSeverity.medium
            message = "Elevated-risk movement detected. Keep observing the patient."
        else:
            severity = AlertSeverity.low
            message = "Movement window appears stable."

        if self._runtime is not None and len(payload.samples) < self._runtime.window_size_samples:
            reasons.append(
                "Offline detector bundle is available, but this batch was too short for the trained model so rule-based fallback was used."
            )

        if not reasons:
            reasons.append("Signal stayed within configured low-risk thresholds.")

        return DetectionResult(
            severity=severity,
            score=round(score, 4),
            fall_probability=round(score, 4),
            peak_acc_g=round(motion.peak_acc_g, 4),
            peak_gyro_dps=round(motion.peak_gyro_dps, 4),
            peak_jerk_g_per_s=round(motion.peak_jerk_g_per_s, 4),
            stillness_ratio=round(motion.stillness_ratio, 4),
            samples_analyzed=len(payload.samples),
            message=message,
            reasons=reasons,
            detected_at=datetime.now(timezone.utc),
        )

    def _analyze_with_offline_model(
        self,
        payload: SensorBatchIn,
        motion: MotionWindow,
        config: DetectorConfig,
    ) -> DetectionResult | None:
        runtime = self._runtime
        if runtime is None or np is None or scipy_signal is None or build_single_feature_row is None:
            return None

        expected_input_samples = max(1, int(round(runtime.window_sec * payload.sampling_rate_hz)))
        minimum_ml_samples = max(config.min_samples, int(round(expected_input_samples * 0.75)))
        if len(motion.rows) < minimum_ml_samples:
            return None

        raw_rows = motion.rows[-expected_input_samples:] if len(motion.rows) >= expected_input_samples else motion.rows

        try:
            raw_signal = np.asarray(raw_rows, dtype=np.float64)
            if raw_signal.shape[0] != runtime.window_size_samples:
                raw_signal = scipy_signal.resample(raw_signal, runtime.window_size_samples, axis=0)

            feature_row = build_single_feature_row(raw_signal, runtime.target_fs)
            feature_vector = np.asarray(
                [[float(feature_row.get(column, 0.0)) for column in runtime.feature_columns]],
                dtype=np.float32,
            )

            if hasattr(runtime.model, "predict_proba"):
                probabilities = runtime.model.predict_proba(feature_vector)[0]
                classes = list(getattr(runtime.model, "classes_", []))
                positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
                fall_probability = clamp01(float(probabilities[positive_index]))
            else:
                prediction = runtime.model.predict(feature_vector)[0]
                fall_probability = 1.0 if int(prediction) == 1 else 0.0
        except Exception as exc:  # pragma: no cover - depends on artifact/runtime state
            logger.exception("Offline detector inference failed; falling back to rule-based detection.")
            self._status_reason = f"Offline detector bundle loaded but inference failed: {exc}"
            return None

        predicted_activity_class: str | None = None
        if runtime.met_model is not None:
            try:
                activity_prediction = runtime.met_model.predict(feature_vector)[0]
                if runtime.met_label_encoder is not None and hasattr(runtime.met_label_encoder, "inverse_transform"):
                    decoded = runtime.met_label_encoder.inverse_transform([int(activity_prediction)])
                    predicted_activity_class = str(decoded[0])
                else:
                    predicted_activity_class = str(activity_prediction)
            except Exception:  # pragma: no cover - depends on artifact/runtime state
                logger.exception("Offline activity classifier inference failed.")

        frailty_proxy_score: float | None = None
        gait_stability_score: float | None = None
        movement_disorder_score: float | None = None
        if runtime.proxy_regressor is not None:
            try:
                proxy_prediction = runtime.proxy_regressor.predict(feature_vector)
                if len(proxy_prediction) > 0:
                    proxy_vector = proxy_prediction[0]
                    if len(proxy_vector) >= 3:
                        frailty_proxy_score = clamp01(float(proxy_vector[0]))
                        gait_stability_score = clamp01(float(proxy_vector[1]))
                        movement_disorder_score = clamp01(float(proxy_vector[2]))
            except Exception:  # pragma: no cover - depends on artifact/runtime state
                logger.exception("Offline proxy regressor inference failed.")

        severity = self._severity_from_probability(fall_probability, config)
        reasons = [
            f"Offline model fall probability is {fall_probability:.2f} for the current live window.",
        ]
        if predicted_activity_class is not None:
            reasons.append(f"Predicted activity intensity class is {predicted_activity_class}.")
        if motion.peak_acc_g >= config.impact_threshold_g:
            reasons.append(f"Impact peak {motion.peak_acc_g:.2f}g supports the model result.")
        if motion.peak_gyro_dps >= config.gyro_threshold_dps:
            reasons.append(f"Rotation peak {motion.peak_gyro_dps:.1f} dps supports the model result.")
        if motion.stillness_ratio >= 0.5:
            reasons.append(f"Post-event stillness ratio reached {motion.stillness_ratio:.2f}.")

        if severity == AlertSeverity.fall_detected:
            message = "Offline model detected a fall-like event from the live sensor window."
        elif severity == AlertSeverity.high_risk:
            message = "Offline model flagged this live window as high risk."
        elif severity == AlertSeverity.medium:
            message = "Offline model flagged this live window as elevated risk."
        else:
            message = "Offline model indicates the live window is low risk."

        return DetectionResult(
            severity=severity,
            score=round(fall_probability, 4),
            fall_probability=round(fall_probability, 4),
            predicted_activity_class=predicted_activity_class,
            frailty_proxy_score=round(frailty_proxy_score, 4) if frailty_proxy_score is not None else None,
            gait_stability_score=round(gait_stability_score, 4) if gait_stability_score is not None else None,
            movement_disorder_score=round(movement_disorder_score, 4)
            if movement_disorder_score is not None
            else None,
            peak_acc_g=round(motion.peak_acc_g, 4),
            peak_gyro_dps=round(motion.peak_gyro_dps, 4),
            peak_jerk_g_per_s=round(motion.peak_jerk_g_per_s, 4),
            stillness_ratio=round(motion.stillness_ratio, 4),
            samples_analyzed=len(payload.samples),
            message=message,
            reasons=reasons,
            detected_at=datetime.now(timezone.utc),
        )

    def _severity_from_probability(self, probability: float, config: DetectorConfig) -> AlertSeverity:
        if probability >= config.fall_score:
            return AlertSeverity.fall_detected
        if probability >= config.high_risk_score:
            return AlertSeverity.high_risk
        if probability >= config.medium_risk_score:
            return AlertSeverity.medium
        return AlertSeverity.low

    def _artifact_signature(self) -> tuple[str, float, float] | None:
        optional_artifacts = (
            "met_classifier.joblib",
            "met_label_encoder.joblib",
            "proxy_regressor.joblib",
        )

        bundle_path = self._artifacts_dir / "fall_detector_bundle.joblib"
        if bundle_path.exists():
            stat = bundle_path.stat()
            latest_mtime = stat.st_mtime
            total_size = float(stat.st_size)
            for name in optional_artifacts:
                artifact = self._artifacts_dir / name
                if artifact.exists():
                    aux_stat = artifact.stat()
                    latest_mtime = max(latest_mtime, aux_stat.st_mtime)
                    total_size += float(aux_stat.st_size)
            return (str(bundle_path), latest_mtime, total_size)

        legacy_model_path = self._artifacts_dir / "fall_detector.joblib"
        metadata_path = self._artifacts_dir / "fall_detector_metadata.json"
        if legacy_model_path.exists() and metadata_path.exists():
            model_stat = legacy_model_path.stat()
            metadata_stat = metadata_path.stat()
            latest_mtime = max(model_stat.st_mtime, metadata_stat.st_mtime)
            total_size = float(model_stat.st_size + metadata_stat.st_size)
            for name in optional_artifacts:
                artifact = self._artifacts_dir / name
                if artifact.exists():
                    aux_stat = artifact.stat()
                    latest_mtime = max(latest_mtime, aux_stat.st_mtime)
                    total_size += float(aux_stat.st_size)
            return (
                str(legacy_model_path),
                latest_mtime,
                total_size,
            )

        return None

    def _refresh_runtime_if_needed(self, force: bool = False) -> None:
        signature = self._artifact_signature()
        if signature is None:
            self._runtime = None
            self._runtime_signature = None
            self._status_reason = f"No offline detector bundle found in {self._artifacts_dir}."
            return

        if not force and signature == self._runtime_signature and self._runtime is not None:
            return

        runtime, reason = self._load_runtime_bundle()
        self._runtime = runtime
        self._runtime_signature = signature
        self._status_reason = reason

    def _load_runtime_bundle(self) -> tuple[OfflineModelBundle | None, str]:
        if OFFLINE_RUNTIME_IMPORT_ERROR is not None or joblib is None:
            return None, f"Offline detector runtime dependencies are unavailable: {OFFLINE_RUNTIME_IMPORT_ERROR}"

        bundle_path = self._artifacts_dir / "fall_detector_bundle.joblib"
        if bundle_path.exists():
            try:
                bundle = joblib.load(bundle_path)
                runtime = self._coerce_runtime_bundle(bundle, artifact_path=bundle_path)
                return runtime, self._runtime_reason_message(
                    prefix=f"Loaded offline detector bundle from {bundle_path}.",
                    runtime=runtime,
                )
            except Exception as exc:  # pragma: no cover - depends on artifact/runtime state
                logger.exception("Failed to load offline detector bundle from %s", bundle_path)
                return None, f"Failed to load offline detector bundle: {exc}"

        legacy_model_path = self._artifacts_dir / "fall_detector.joblib"
        metadata_path = self._artifacts_dir / "fall_detector_metadata.json"
        if legacy_model_path.exists() and metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                metadata["model"] = joblib.load(legacy_model_path)
                runtime = self._coerce_runtime_bundle(metadata, artifact_path=legacy_model_path)
                return runtime, self._runtime_reason_message(
                    prefix=f"Loaded legacy offline detector artifacts from {legacy_model_path}.",
                    runtime=runtime,
                )
            except Exception as exc:  # pragma: no cover - depends on artifact/runtime state
                logger.exception("Failed to load legacy offline detector artifacts from %s", legacy_model_path)
                return None, f"Failed to load legacy offline detector artifacts: {exc}"

        return None, f"No offline detector artifacts were found in {self._artifacts_dir}."

    def _coerce_runtime_bundle(self, bundle: dict[str, Any], artifact_path: Path) -> OfflineModelBundle:
        if not isinstance(bundle, dict):
            raise TypeError("Offline detector artifact is not a dictionary bundle.")
        if "model" not in bundle:
            raise KeyError("Offline detector bundle is missing the trained model.")

        feature_columns = [str(column) for column in bundle.get("feature_columns", [])]
        if not feature_columns:
            raise ValueError("Offline detector bundle does not define feature columns.")

        target_fs = int(bundle.get("target_fs", 50))
        window_sec = float(bundle.get("window_sec", 2.56))
        overlap = float(bundle.get("overlap", 0.5))
        window_size_samples = int(bundle.get("window_size_samples", round(target_fs * window_sec)))
        step_size_samples = int(
            bundle.get(
                "step_size_samples",
                max(1, int(round(window_size_samples * (1.0 - overlap)))),
            )
        )

        met_model = bundle.get("met_model")
        if met_model is None:
            met_model = self._load_optional_joblib_artifact("met_classifier.joblib")

        met_label_encoder = bundle.get("met_label_encoder")
        if met_label_encoder is None:
            met_label_encoder = self._load_optional_joblib_artifact("met_label_encoder.joblib")

        proxy_regressor = bundle.get("proxy_regressor")
        if proxy_regressor is None:
            proxy_regressor = self._load_optional_joblib_artifact("proxy_regressor.joblib")

        return OfflineModelBundle(
            model=bundle["model"],
            feature_columns=feature_columns,
            target_fs=target_fs,
            window_sec=window_sec,
            overlap=overlap,
            window_size_samples=window_size_samples,
            step_size_samples=step_size_samples,
            model_name=str(bundle.get("model_name", "fall_detector")),
            artifact_path=str(artifact_path),
            met_model=met_model,
            met_label_encoder=met_label_encoder,
            proxy_regressor=proxy_regressor,
        )

    def _load_optional_joblib_artifact(self, filename: str) -> object | None:
        if joblib is None:
            return None
        artifact_path = self._artifacts_dir / filename
        if not artifact_path.exists():
            return None
        try:
            return joblib.load(artifact_path)
        except Exception:  # pragma: no cover - depends on artifact/runtime state
            logger.exception("Failed to load optional detector artifact from %s", artifact_path)
            return None

    def _runtime_reason_message(self, prefix: str, runtime: OfflineModelBundle) -> str:
        has_activity = "yes" if runtime.met_model is not None else "no"
        has_proxy = "yes" if runtime.proxy_regressor is not None else "no"
        return f"{prefix} Activity classifier loaded: {has_activity}. Proxy regressor loaded: {has_proxy}."
