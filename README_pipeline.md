# Multi-Task Mobility Health Pipeline

This repository now includes a practical pipeline under `scripts/`:

- `preprocessing.py`: Parse SisFall, MobiAct, UCI and build standardized 50Hz windows.
- `feature_extraction.py`: Extract time/frequency/nonlinear features + proxy targets.
- `modeling.py`: Train and evaluate fall detection, MET classification, and proxy regressors.
- `app_integration.py`: Generate mobile-export placeholders and integration notes.
- `run_pipeline.py`: Execute the full pipeline end-to-end.

## Quick Start

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --sisfall-only --max-files-per-dataset 120
```

For full training (can be long):

```bash
python scripts/run_pipeline.py --sisfall-only
```

For multi-dataset training:

```bash
python scripts/run_pipeline.py
```

## Outputs

- `results/artifacts/windows.pkl`
- `results/artifacts/features.pkl`
- `results/artifacts/*.joblib`
- `results/artifacts/metrics_summary.csv`
- `results/artifacts/*.png` (confusion matrix and ROC)
- `results/mobile_export/*`
