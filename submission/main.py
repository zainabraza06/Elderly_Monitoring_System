# main.py
"""
SisFall Fall Detection System - Main Pipeline

This script implements a comprehensive fall detection system with:
1. Fall detection (binary classification)
2. Gait stability assessment
3. Frailty prediction

Evaluation methods:
- Leave-One-Subject-Out Cross-Validation (LOSO)
- Young → Elderly transfer learning evaluation
- SHAP-based explainability

Author: SisFall Analysis Team
"""
import os
import sys
import numpy as np
import warnings
from pathlib import Path

# Force UTF-8 output so Unicode characters (→, ±, ², etc.) print correctly
# on Windows terminals that default to cp1252.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from src.loader import load_sisfall, get_dataset_info
from src.eda import basic_eda
from src.dataset_builder import (
    build_dataset, 
    build_walking_dataset,
    build_elderly_dataset,
    split_young_elderly,
    get_scale_pos_weight,
    print_dataset_summary
)
from src.model import (
    create_fall_detector,
    FallDetectorSVM, FallDetectorLDA,
    GaitStabilityRegressor,
    FrailtyPredictor,
    UnifiedFallDetectionSystem,
    compute_frailty_proxy,
    compute_stability_score,
    HAS_XGBOOST
)
from src.evaluation import (
    LOSOCrossValidator,
    ElderlyTestValidator,
    evaluate,
    compute_classification_metrics,
    compute_regression_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    generate_evaluation_report
)
from src.preprocessing import ZScoreNormalizer


# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = r"C:\Users\User\Documents\4rth semester\AI\SisFall_dataset\data\SisFall_dataset"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Fixed held-out test subjects — NEVER used during training or validation
TEST_SUBJECTS_YOUNG   = {'SA20', 'SA21', 'SA22', 'SA23'}   # 4 young
TEST_SUBJECTS_ELDERLY = {'SE13', 'SE14', 'SE15'}            # 3 elderly
TEST_SUBJECTS = TEST_SUBJECTS_YOUNG | TEST_SUBJECTS_ELDERLY

FALL_THRESHOLD = 0.5


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def subject_split(X, y, subjects_arr):
    """Split feature matrix into (train, held-out test) by subject."""
    test_mask  = np.array([s in TEST_SUBJECTS  for s in subjects_arr])
    train_mask = ~test_mask
    return (X[train_mask], y[train_mask], subjects_arr[train_mask],
            X[test_mask],  y[test_mask],  subjects_arr[test_mask])


def coral_transform(X_src, X_tgt, reg=1e-4):
    """CORAL domain adaptation: align 2nd-order stats of source to target."""
    d = X_src.shape[1]
    Cs = np.cov(X_src, rowvar=False) + reg * np.eye(d)
    Ct = np.cov(X_tgt, rowvar=False) + reg * np.eye(d)
    vals_s, vecs_s = np.linalg.eigh(Cs)
    vals_t, vecs_t = np.linalg.eigh(Ct)
    vals_s = np.clip(vals_s, 1e-8, None)
    vals_t = np.clip(vals_t, 1e-8, None)
    A = (vecs_s @ np.diag(vals_s ** -0.5) @ vecs_s.T @
         vecs_t @ np.diag(vals_t **  0.5) @ vecs_t.T)
    return X_src @ A


def calibrate_threshold(model, X_cal, y_cal, metric='balanced'):
    """Find decision threshold via Youden-J (balanced) or target sensitivity."""
    from sklearn.metrics import roc_curve
    proba = model.predict_proba(X_cal)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_cal, proba)
    if metric == 'balanced':
        j = tpr - fpr
        return float(np.clip(thresholds[np.argmax(j)], 0.05, 0.95))
    return 0.5


def print_model_table(results_dict):
    """Aligned metric comparison table for multiple models."""
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1', 'mcc', 'roc_auc']
    header = f"{'Model':<18}" + "".join(f"{m:>13}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, m in results_dict.items():
        row = f"{name:<18}" + "".join(
            f"{m.get(k, float('nan')):>13.4f}" for k in metrics)
        print(row)


def run_loso(model_name, factory_fn, X_tr, y_tr, subjects_tr):
    """Run LOSO CV for a single model."""
    import time
    t0 = time.time()
    n_sub = len(np.unique(subjects_tr))
    print(f"  [{model_name}] LOSO on {n_sub} subjects ...", end='', flush=True)
    validator = LOSOCrossValidator(verbose=False)
    result = validator.validate(X_tr, y_tr, subjects_tr, factory_fn)
    print(f"  done in {time.time()-t0:.0f}s")
    return result


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 70)
    print("SISFALL FALL DETECTION SYSTEM  --  COMPREHENSIVE PIPELINE")
    print("=" * 70)

    # =========================================================================
    # PHASE 1: Load Dataset
    # =========================================================================
    print("\nPHASE 1: Loading Dataset")
    print("-" * 40)
    from src.loader import load_sisfall, get_dataset_info

    dataset_info = get_dataset_info(DATA_PATH)
    print(f"Dataset: {dataset_info['total_trials']} trials, "
          f"{dataset_info['n_subjects']} subjects "
          f"({dataset_info['n_young']} young / {dataset_info['n_elderly']} elderly)")

    X_trials, y_trials, subjects, activity_codes = load_sisfall(DATA_PATH)
    print(f"Loaded {len(X_trials)} trials")
    print(f"Held-out TEST subjects ({len(TEST_SUBJECTS)}): {sorted(TEST_SUBJECTS)}")

    # =========================================================================
    # PHASE 2: Exploratory Data Analysis
    # =========================================================================
    print("\nPHASE 2: Exploratory Data Analysis")
    print("-" * 40)
    from src.eda import basic_eda
    basic_eda(X_trials, y_trials)

    # =========================================================================
    # PHASE 3: Build Feature Dataset  (from disk cache if available)
    # =========================================================================
    print("\nPHASE 3: Building Feature Dataset")
    print("-" * 40)

    X, y, metadata = build_dataset(
        X_trials, y_trials, subjects, activity_codes,
        include_gait_features=False, verbose=True, data_root=DATA_PATH
    )
    print_dataset_summary(X, y, metadata)
    window_subjects = np.array(metadata['subjects'])
    feature_names   = metadata.get('feature_names', [f'f{i}' for i in range(X.shape[1])])
    y = y.astype(int)

    # =========================================================================
    # PHASE 4: Subject-Aware 3-Way Split
    # =========================================================================
    print("\nPHASE 4: Subject-Aware Train / Val / Test Split")
    print("-" * 40)

    (X_tr, y_tr, subj_tr,
     X_te, y_te, subj_te) = subject_split(X, y, window_subjects)

    # 20% of training subjects held out as validation
    unique_tr_subj = np.unique(subj_tr)
    np.random.seed(42)
    val_idx   = np.random.choice(len(unique_tr_subj),
                                  max(1, len(unique_tr_subj) // 5), replace=False)
    val_subj  = set(unique_tr_subj[val_idx])
    loso_subj = set(unique_tr_subj) - val_subj

    loso_mask = np.array([s in loso_subj for s in subj_tr])
    val_mask  = ~loso_mask

    X_loso, y_loso, subj_loso = X_tr[loso_mask], y_tr[loso_mask], subj_tr[loso_mask]
    X_val,  y_val,  subj_val  = X_tr[val_mask],  y_tr[val_mask],  subj_tr[val_mask]

    normalizer = ZScoreNormalizer()
    X_loso_n = normalizer.fit_transform(X_loso)
    X_val_n  = normalizer.transform(X_val)
    X_te_n   = normalizer.transform(X_te)

    print(f"LOSO-train : {len(np.unique(subj_loso))} subjects, {len(X_loso):,} samples")
    print(f"Validation : {len(np.unique(subj_val))} subjects, {len(X_val):,} samples")
    print(f"Test (sealed): {len(np.unique(subj_te))} subjects, {len(X_te):,} samples")

    # =========================================================================
    # PHASE 5: Multi-Model Comparison on Validation Set
    # =========================================================================
    print("\nPHASE 5: Multi-Model Comparison  (Validation set)")
    print("-" * 40)

    scale_pos_weight = get_scale_pos_weight(y_loso)

    model_factories = {
        'RandomForest': lambda: create_fall_detector('rf', n_estimators=200),
        'XGBoost'     : lambda: create_fall_detector(
                            'xgb', n_estimators=200,
                            scale_pos_weight=scale_pos_weight),
        'MLP'         : lambda: create_fall_detector(
                            'mlp', hidden_layers=(256, 128, 64), max_iter=500),
        'SVM'         : lambda: create_fall_detector('svm', C=1.0),
    }

    val_results    = {}
    trained_models = {}

    for name, factory in model_factories.items():
        print(f"  Training {name} ...", end='', flush=True)
        m = factory()
        m.fit(X_loso_n, y_loso)
        trained_models[name] = m
        yp   = m.predict(X_val_n)
        ypr  = m.predict_proba(X_val_n)[:, 1] if hasattr(m, 'predict_proba') else None
        val_results[name] = compute_classification_metrics(y_val, yp, ypr)
        r = val_results[name]
        print(f"  Acc={r['accuracy']:.3f}  Sens={r['sensitivity']:.3f}  "
              f"Spec={r['specificity']:.3f}  F1={r['f1']:.3f}  "
              f"AUC={r.get('roc_auc', float('nan')):.3f}")

    print("\nValidation Set Comparison:")
    print_model_table(val_results)

    with open(OUTPUT_DIR / "val_comparison.txt", 'w', encoding='utf-8') as f:
        f.write("VALIDATION SET MODEL COMPARISON\n" + "="*70 + "\n")
        mkeys = ['accuracy', 'sensitivity', 'specificity', 'f1', 'mcc', 'roc_auc']
        hdr = f"{'Model':<18}" + "".join(f"{m:>13}" for m in mkeys)
        f.write(hdr + "\n" + "-"*len(hdr) + "\n")
        for nm, res in val_results.items():
            f.write(f"{nm:<18}" + "".join(f"{res.get(k, float('nan')):>13.4f}" for k in mkeys) + "\n")

    # =========================================================================
    # PHASE 6: LOSO Cross-Validation (all 4 models)
    # =========================================================================
    print("\nPHASE 6: LOSO Cross-Validation  (all 4 models)")
    print("-" * 40)
    print("Running Leave-One-Subject-Out for each model -- takes several minutes...")

    loso_results = {}
    for name, factory in model_factories.items():
        loso_results[name] = run_loso(name, factory, X_loso_n, y_loso, subj_loso)

    print("\nLOSO Overall Metrics:")
    loso_overall = {n: r['overall'] for n, r in loso_results.items()}
    print_model_table(loso_overall)

    print("\nLOSO Mean +/- Std across subjects:")
    for name, res in loso_results.items():
        m, s = res['mean_metrics'], res['std_metrics']
        print(f"  {name:<14}  "
              f"Acc={m['accuracy']:.3f}+/-{s['accuracy']:.3f}  "
              f"Sens={m['sensitivity']:.3f}+/-{s['sensitivity']:.3f}  "
              f"Spec={m['specificity']:.3f}+/-{s['specificity']:.3f}  "
              f"F1={m['f1']:.3f}+/-{s['f1']:.3f}")

    for name, res in loso_results.items():
        generate_evaluation_report(res, output_path=OUTPUT_DIR / f"loso_{name.lower()}.txt")
    print(f"\nLOSO reports saved to: {OUTPUT_DIR}")

    best_model_name = max(loso_overall, key=lambda n: loso_overall[n].get('f1', 0))
    print(f"\nBest model by LOSO F1: {best_model_name}")

    # =========================================================================
    # PHASE 7: Feature Pattern Analysis
    # =========================================================================
    print("\nPHASE 7: Feature Pattern Analysis")
    print("-" * 40)

    rf_model    = trained_models['RandomForest']
    importances = rf_model.get_feature_importance()
    ranked_idx  = np.argsort(importances)[::-1]

    print("\nTop 20 Features (Random Forest Gini importance):")
    print(f"  {'Rank':<5} {'Feature':<38} {'Importance':>12}")
    print("  " + "-" * 57)
    for rank, idx in enumerate(ranked_idx[:20], 1):
        fname = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
        print(f"  {rank:<5} {fname:<38} {importances[idx]:>12.4f}")

    # Sensor group importance
    sensor_groups = {
        'ADXL345 acc  (ch 0-2)' : list(range(0, 3)),
        'ITG3200 gyro (ch 3-5)' : list(range(3, 6)),
        'MMA8451Q acc (ch 6-8)' : list(range(6, 9)),
        'Magnitude    (ch 9-11)': list(range(9, 12)),
    }
    n_feat_per_ch = max(1, X.shape[1] // 12)

    print("\nSensor Group Importance (sum over features):")
    for grp, channels in sensor_groups.items():
        idxs = [blk * 12 + ch for blk in range(n_feat_per_ch) for ch in channels
                if blk * 12 + ch < X.shape[1]]
        if idxs:
            gi  = importances[idxs].sum()
            bar = '#' * max(1, int(gi * 40))
            print(f"  {grp:<26}  {gi:.4f}  {bar}")

    # Fall vs ADL stats (top 5 features)
    print("\nFall vs ADL -- top-5 feature means:")
    print(f"  {'Feature':<38} {'ADL mean':>10} {'Fall mean':>10} {'Diff%':>8}")
    print("  " + "-" * 70)
    adl_mask_tr  = y_loso == 0
    fall_mask_tr = y_loso == 1
    for idx in ranked_idx[:5]:
        fname   = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
        adl_mu  = X_loso[adl_mask_tr,  idx].mean()
        fall_mu = X_loso[fall_mask_tr, idx].mean()
        pct     = (fall_mu - adl_mu) / (abs(adl_mu) + 1e-8) * 100
        print(f"  {fname:<38} {adl_mu:>10.3f} {fall_mu:>10.3f} {pct:>7.1f}%")

    # Per-subject LOSO F1 breakdown by age group
    per_subj   = sorted(loso_results[best_model_name]['per_subject'],
                        key=lambda r: r['subject'])
    young_f1   = [r['f1'] for r in per_subj if r['subject'].startswith('SA')]
    elderly_f1 = [r['f1'] for r in per_subj if r['subject'].startswith('SE')]
    print(f"\nPer-subject LOSO F1 ({best_model_name}):")
    if young_f1:
        print(f"  Young   n={len(young_f1):2d}  avg={np.mean(young_f1):.4f}  "
              f"min={np.min(young_f1):.4f}  max={np.max(young_f1):.4f}")
    if elderly_f1:
        print(f"  Elderly n={len(elderly_f1):2d}  avg={np.mean(elderly_f1):.4f}  "
              f"min={np.min(elderly_f1):.4f}  max={np.max(elderly_f1):.4f}")

    with open(OUTPUT_DIR / "feature_analysis.txt", 'w', encoding='utf-8') as f:
        f.write("FEATURE IMPORTANCE REPORT\n" + "="*60 + "\n\n")
        f.write("Top 30 Features (RF Gini importance):\n")
        for rank, idx in enumerate(ranked_idx[:30], 1):
            fname = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
            f.write(f"  {rank:2d}. {fname:<42} {importances[idx]:.6f}\n")
        if young_f1 and elderly_f1:
            f.write("\nPer-subject LOSO F1:\n")
            f.write(f"  Young   avg={np.mean(young_f1):.4f}  "
                    f"min={np.min(young_f1):.4f}  max={np.max(young_f1):.4f}\n")
            f.write(f"  Elderly avg={np.mean(elderly_f1):.4f}  "
                    f"min={np.min(elderly_f1):.4f}  max={np.max(elderly_f1):.4f}\n")
    print(f"\nFeature analysis -> {OUTPUT_DIR / 'feature_analysis.txt'}")

    # =========================================================================
    # PHASE 8: Young -> Elderly Transfer  (4 methods)
    # =========================================================================
    print("\nPHASE 8: Young -> Elderly Transfer Learning")
    print("-" * 40)

    young_mask_all   = np.array([s.startswith('SA') for s in window_subjects])
    elderly_mask_all = np.array([s.startswith('SE') for s in window_subjects])
    elderly_te_mask  = np.array([s in TEST_SUBJECTS_ELDERLY for s in window_subjects])
    elderly_cal_mask = elderly_mask_all & ~elderly_te_mask

    X_young_n   = normalizer.transform(X[young_mask_all])
    y_young     = y[young_mask_all]
    X_eld_cal_n = normalizer.transform(X[elderly_cal_mask])
    y_eld_cal   = y[elderly_cal_mask]
    X_eld_te_n  = normalizer.transform(X[elderly_te_mask])
    y_eld_te    = y[elderly_te_mask]

    transfer_results = {}
    best_factory     = model_factories[best_model_name]

    # Method 1: Baseline (train on young, test on elderly)
    m_base = best_factory()
    m_base.fit(X_young_n, y_young)
    yp  = m_base.predict(X_eld_te_n)
    ypr = m_base.predict_proba(X_eld_te_n)[:, 1] if hasattr(m_base, 'predict_proba') else None
    transfer_results['Baseline'] = compute_classification_metrics(y_eld_te, yp, ypr)

    # Method 2: Youden threshold calibration on elderly cal set
    if len(X_eld_cal_n) > 0 and hasattr(m_base, 'predict_proba'):
        t_opt = calibrate_threshold(m_base, X_eld_cal_n, y_eld_cal, 'balanced')
        yp_t  = (m_base.predict_proba(X_eld_te_n)[:, 1] >= t_opt).astype(int)
        transfer_results[f'ThreshCal(t={t_opt:.2f})'] = \
            compute_classification_metrics(y_eld_te, yp_t, ypr)
        print(f"  Optimal threshold (Youden-J on elderly cal): {t_opt:.3f}")

    # Method 3: CORAL domain adaptation
    try:
        X_src_raw   = X[young_mask_all]
        X_tgt_raw   = X[elderly_cal_mask] if elderly_cal_mask.sum() > 0 else X[elderly_te_mask]
        X_src_c     = coral_transform(X_src_raw, X_tgt_raw)
        from sklearn.preprocessing import StandardScaler as _SS
        sc_c = _SS()
        X_src_cn    = sc_c.fit_transform(X_src_c)
        X_eld_te_cn = sc_c.transform(X[elderly_te_mask])
        m_coral = best_factory()
        m_coral.fit(X_src_cn, y_young)
        yp_c  = m_coral.predict(X_eld_te_cn)
        ypr_c = m_coral.predict_proba(X_eld_te_cn)[:, 1] if hasattr(m_coral, 'predict_proba') else None
        transfer_results['CORAL'] = compute_classification_metrics(y_eld_te, yp_c, ypr_c)
        print("  CORAL domain adaptation applied")
    except Exception as e:
        print(f"  CORAL skipped: {e}")

    # Method 4: Cost-sensitive (upweighted minority)
    try:
        from sklearn.utils.class_weight import compute_sample_weight as _csw
        sw   = _csw('balanced', y_young) * 1.5
        m_cs = best_factory()
        try:
            m_cs.model.fit(X_young_n, y_young, sample_weight=sw)
        except Exception:
            m_cs.fit(X_young_n, y_young)
        yp_cs  = m_cs.predict(X_eld_te_n)
        ypr_cs = m_cs.predict_proba(X_eld_te_n)[:, 1] if hasattr(m_cs, 'predict_proba') else None
        transfer_results['CostSensitive'] = compute_classification_metrics(y_eld_te, yp_cs, ypr_cs)
    except Exception as e:
        print(f"  CostSensitive skipped: {e}")

    print("\nYoung -> Elderly Transfer Results:")
    print_model_table(transfer_results)
    print(f"  Elderly test: {len(y_eld_te)} samples  "
          f"Falls={y_eld_te.sum()}  ADL={(y_eld_te == 0).sum()}")

    with open(OUTPUT_DIR / "transfer_results.txt", 'w', encoding='utf-8') as f:
        f.write("YOUNG -> ELDERLY TRANSFER RESULTS\n" + "="*70 + "\n\n")
        mkeys = ['accuracy', 'sensitivity', 'specificity', 'f1', 'mcc', 'roc_auc']
        hdr   = f"{'Method':<26}" + "".join(f"{m:>13}" for m in mkeys)
        f.write(hdr + "\n" + "-"*len(hdr) + "\n")
        for nm, res in transfer_results.items():
            f.write(f"{nm:<26}" + "".join(f"{res.get(k, float('nan')):>13.4f}" for k in mkeys) + "\n")
    print(f"Transfer results -> {OUTPUT_DIR / 'transfer_results.txt'}")

    # =========================================================================
    # PHASE 9: Gait Stability & Frailty Prediction
    # =========================================================================
    print("\nPHASE 9: Gait Stability & Frailty Prediction")
    print("-" * 40)

    from sklearn.model_selection import train_test_split as _tts

    stab_model = sc_stab = Xws_tr = yws_tr = None
    X_walk, y_walk, walk_meta = build_walking_dataset(
        X_trials, y_trials, subjects, activity_codes, verbose=True,
        data_root=DATA_PATH)

    if len(X_walk) > 0:
        stab_sc     = compute_stability_score(X_walk, is_walking=True)
        Xws_tr, Xws_te, yws_tr, yws_te = _tts(X_walk, stab_sc, test_size=0.2, random_state=42)
        sc_stab     = ZScoreNormalizer()
        stab_model  = GaitStabilityRegressor(model_type='gbr')
        stab_model.fit(sc_stab.fit_transform(Xws_tr), yws_tr)
        sm = compute_regression_metrics(yws_te, stab_model.predict(sc_stab.transform(Xws_te)))
        print(f"Gait Stability  RMSE={sm['rmse']:.4f}  MAE={sm['mae']:.4f}  R2={sm['r2']:.4f}")

    frail_model = sc_frail = Xfr_tr = yfr_tr = None
    X_eld_feat, _, _ = build_elderly_dataset(
        X_trials, y_trials, subjects, activity_codes, verbose=True,
        data_root=DATA_PATH)

    if len(X_eld_feat) > 0:
        frail_sc     = compute_frailty_proxy(X_eld_feat)
        Xfr_tr, Xfr_te, yfr_tr, yfr_te = _tts(X_eld_feat, frail_sc, test_size=0.2, random_state=42)
        sc_frail     = ZScoreNormalizer()
        frail_model  = FrailtyPredictor(model_type='mlp')
        frail_model.fit(sc_frail.fit_transform(Xfr_tr), yfr_tr)
        fm = compute_regression_metrics(yfr_te, frail_model.predict(sc_frail.transform(Xfr_te)))
        print(f"Frailty         RMSE={fm['rmse']:.4f}  MAE={fm['mae']:.4f}  R2={fm['r2']:.4f}")

    # =========================================================================
    # PHASE 10: SHAP Explainability
    # =========================================================================
    print("\nPHASE 10: SHAP Explainability")
    print("-" * 40)
    try:
        from src.explainability import FallDetectionExplainer, generate_shap_report
        best_trained = trained_models[best_model_name]
        explainer    = FallDetectionExplainer(best_trained,
                                              X_background=X_loso_n[:100],
                                              feature_names=feature_names)
        shap_values  = explainer.compute_shap_values(X_val_n[:200])
        imp_df       = explainer.get_feature_importance()
        print(f"\nSHAP Top 10 ({best_model_name}):")
        print(f"  {'Rank':<5} {'Feature':<38} {'SHAP importance':>16}")
        print("  " + "-" * 61)
        for i, row in imp_df.head(10).iterrows():
            print(f"  {i+1:<5} {str(row['feature'])[:38]:<38} {row['importance']:>16.6f}")
        generate_shap_report(explainer, X_val_n[:200],
                             output_path=OUTPUT_DIR / "shap_report.txt")
        print(f"SHAP report -> {OUTPUT_DIR / 'shap_report.txt'}")
    except Exception as e:
        print(f"SHAP skipped: {e}")

    # =========================================================================
    # PHASE 11: Unified System -- Final Test on Sealed Subjects
    # =========================================================================
    print("\nPHASE 11: Unified System -- Final Test (Sealed Subjects)")
    print("-" * 40)
    print(f"Test subjects : {sorted(TEST_SUBJECTS)}")
    print(f"Test samples  : {len(X_te)}  "
          f"(Falls={y_te.sum()}  ADL={(y_te == 0).sum()})")

    # Train final fall detector on ALL non-test data (loso + val)
    X_full_n = normalizer.transform(X_tr)
    y_full   = y_tr

    fall_type = best_model_name.lower()
    if fall_type not in ('rf', 'xgb', 'mlp', 'svm', 'lda'):
        fall_type = 'rf'

    unified = UnifiedFallDetectionSystem(fall_model=fall_type,
                                         stability_model='gbr',
                                         frailty_model='mlp',
                                         fall_threshold=FALL_THRESHOLD)
    unified.fit_fall_detector(X_full_n, y_full)
    print(f"Fall detector ({best_model_name}) trained on {len(X_full_n):,} samples")

    if stab_model is not None and Xws_tr is not None:
        unified.stability_regressor    = stab_model
        unified._n_features_stability  = Xws_tr.shape[1]
        unified.is_fitted['stability'] = True
    if frail_model is not None and Xfr_tr is not None:
        unified.frailty_predictor      = frail_model
        unified._n_features_frailty    = Xfr_tr.shape[1]
        unified.is_fitted['frailty']   = True

    preds       = unified.predict(X_te_n)
    risk_levels = unified.get_risk_level(X_te_n)

    y_pred_final  = preds['is_fall'].astype(int)
    y_proba_final = preds['fall_prob']
    final_metrics = compute_classification_metrics(y_te, y_pred_final, y_proba_final)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS  (sealed held-out subjects)")
    print("=" * 60)
    for k in ('accuracy', 'sensitivity', 'specificity', 'f1', 'mcc', 'roc_auc', 'pr_auc'):
        v = final_metrics.get(k, float('nan'))
        print(f"  {k:<20}: {v:.4f}")
    print("=" * 60)

    from collections import Counter
    risk_counts = Counter(risk_levels)
    print("\nRisk distribution on test set:")
    for lvl in ('fall_detected', 'high', 'medium', 'low'):
        print(f"  {lvl:<16}: {risk_counts.get(lvl, 0):>6,}")

    # Young vs Elderly breakdown on sealed test set
    young_te_m   = np.array([s.startswith('SA') for s in subj_te])
    elderly_te_m = ~young_te_m
    if young_te_m.sum() > 0:
        ym = compute_classification_metrics(
            y_te[young_te_m], y_pred_final[young_te_m], y_proba_final[young_te_m])
        print(f"\nTest Young  : Acc={ym['accuracy']:.4f}  "
              f"Sens={ym['sensitivity']:.4f}  Spec={ym['specificity']:.4f}  F1={ym['f1']:.4f}")
    if elderly_te_m.sum() > 0:
        em = compute_classification_metrics(
            y_te[elderly_te_m], y_pred_final[elderly_te_m], y_proba_final[elderly_te_m])
        print(f"Test Elderly: Acc={em['accuracy']:.4f}  "
              f"Sens={em['sensitivity']:.4f}  Spec={em['specificity']:.4f}  F1={em['f1']:.4f}")

    unified.save(str(OUTPUT_DIR / f"unified_{best_model_name.lower()}.pkl"))

    with open(OUTPUT_DIR / "final_test_results.txt", 'w', encoding='utf-8') as f:
        f.write("FINAL TEST RESULTS (sealed held-out subjects)\n" + "="*60 + "\n\n")
        for k, v in final_metrics.items():
            f.write(f"{k:<25}: {v}\n")
        f.write("\nTest subjects: " + str(sorted(TEST_SUBJECTS)) + "\n")
    print(f"\nFinal results -> {OUTPUT_DIR / 'final_test_results.txt'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nBest model by LOSO F1 : {best_model_name}")
    print(f"Results directory     : {OUTPUT_DIR}")
    print("\nOutputs:")
    print("  val_comparison.txt       Phase 5 multi-model comparison")
    print("  loso_<model>.txt         Phase 6 per-model LOSO reports")
    print("  feature_analysis.txt     Phase 7 feature importance")
    print("  transfer_results.txt     Phase 8 Young->Elderly methods")
    print("  shap_report.txt          Phase 10 explainability")
    print("  final_test_results.txt   Phase 11 sealed-test evaluation")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
