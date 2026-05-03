# Training + inference (quick index)

- **Run everything** (after installing `scripts/requirements-training.txt`):  
  `py scripts/run_training.py all --data-root "…\\Annotated Data"`  
  then `py scripts/sync_inference_manifest.py`.
- **Inference (Python)** lives in `scripts/inference/motion_pipeline.py` (same logic the API uses).
- **Full system diagram** (Flutter → API → models, three dashboards): see `docs/ARCHITECTURE.md`.

Per-category modules:

| Task | Package | Main runner |
|------|---------|-------------|
| Fall vs ADL | `baseline_fall/` | `train_fall_detection_mobiact.py` |
| ADL classes | `baseline_adl/` | `train_mobiact_adl.py` |
| Fall type (4-class) | `baseline_falltype/` | `train_fall_type_mobiact.py` |

Frozen weights go under `models/`; CSV/plots under `results/` (regenerable).
