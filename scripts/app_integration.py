from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def export_to_onnx_placeholder(model_path: Path, output_path: Path) -> None:
    model = joblib.load(model_path)
    _ = model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ONNX export placeholder. Use skl2onnx/onnxmltools in a dedicated export script.\n")


def export_tflite_placeholder(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("TFLite export placeholder. Train equivalent TF model and convert with TFLiteConverter.\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare model export artifacts for mobile integration.")
    p.add_argument("--artifacts", type=Path, default=Path("results/artifacts"))
    p.add_argument("--output", type=Path, default=Path("results/mobile_export"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    fall_model = args.artifacts / "fall_detector.joblib"
    met_model = args.artifacts / "met_classifier.joblib"

    if fall_model.exists():
        export_to_onnx_placeholder(fall_model, args.output / "fall_detector.onnx.txt")
    if met_model.exists():
        export_to_onnx_placeholder(met_model, args.output / "met_classifier.onnx.txt")

    export_tflite_placeholder(args.output / "proxy_model.tflite.txt")

    with (args.output / "integration_notes.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Android Integration Notes\n\n"
            "- Keep sensor sampling at 50 Hz.\n"
            "- Use 2.56s windows with 50% overlap (128 samples, step 64).\n"
            "- Reuse exact feature extraction logic from Python.\n"
            "- Decision logic: if fall_prob > threshold -> FALL else activity + scores.\n"
        )

    print(f"Export notes and placeholders saved to {args.output}")


if __name__ == "__main__":
    main()
