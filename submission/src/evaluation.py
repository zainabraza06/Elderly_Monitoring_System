# src/evaluation.py
"""
Comprehensive evaluation module for fall detection system.
Includes LOSO cross-validation, multiple metrics, and visualization.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings

# Scikit-learn imports
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

warnings.filterwarnings('ignore')


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================
def compute_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    metrics['sensitivity'] = metrics['recall']  # Alias
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Specificity (TNR)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Additional metrics from confusion matrix
    metrics['true_positives'] = tp
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    
    # AUC-ROC (if probabilities available)
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
    
    return metrics


def compute_regression_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Additional metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['max_error'] = np.max(np.abs(y_true - y_pred))
    
    return metrics


# =============================================================================
# LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION
# =============================================================================
class LOSOCrossValidator:
    """
    Leave-One-Subject-Out Cross-Validator.
    Ensures model generalizes across different individuals.
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = []
        self.fold_predictions = []
    
    def validate(self, X, y, subjects, model_factory, normalizer_class=StandardScaler):
        """
        Perform LOSO cross-validation.
        
        Args:
            X: Features array
            y: Labels array
            subjects: Subject IDs for each sample
            model_factory: Function that returns a new model instance
            normalizer_class: Normalizer class to use (default StandardScaler)
        
        Returns:
            Dictionary with aggregated results
        """
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        
        self.results = []
        self.fold_predictions = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        logo = LeaveOneGroupOut()
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, subjects)):
            test_subject = subjects[test_idx[0]]
            
            if self.verbose:
                print(f"Fold {fold + 1}/{n_subjects}: Testing on subject {test_subject}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Normalize (fit only on training data!)
            normalizer = normalizer_class()
            X_train_norm = normalizer.fit_transform(X_train)
            X_test_norm = normalizer.transform(X_test)
            
            # Train model
            model = model_factory()
            model.fit(X_train_norm, y_train)
            
            # Predict
            y_pred = model.predict(X_test_norm)
            
            # Get probabilities if available
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test_norm)[:, 1]
                except:
                    pass
            
            # Compute metrics for this fold
            fold_metrics = compute_classification_metrics(y_test, y_pred, y_proba)
            fold_metrics['subject'] = test_subject
            fold_metrics['n_test_samples'] = len(test_idx)
            self.results.append(fold_metrics)
            
            # Store predictions
            self.fold_predictions.append({
                'subject': test_subject,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba
            })
            
            # Aggregate predictions
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            if y_proba is not None:
                all_y_proba.extend(y_proba)
        
        # Compute overall metrics
        all_y_proba = np.array(all_y_proba) if all_y_proba else None
        overall_metrics = compute_classification_metrics(
            np.array(all_y_true), np.array(all_y_pred), all_y_proba
        )
        
        # Add summary statistics
        results_df = pd.DataFrame(self.results)
        summary = {
            'overall': overall_metrics,
            'per_subject': self.results,
            'mean_metrics': {
                col: results_df[col].mean() 
                for col in results_df.columns if col not in ['subject', 'n_test_samples']
            },
            'std_metrics': {
                col: results_df[col].std() 
                for col in results_df.columns if col not in ['subject', 'n_test_samples']
            }
        }
        
        if self.verbose:
            self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary):
        """Print summary of LOSO results."""
        print("\n" + "="*60)
        print("LOSO Cross-Validation Results")
        print("="*60)
        
        overall = summary['overall']
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:    {overall['accuracy']:.4f}")
        print(f"  Sensitivity: {overall['sensitivity']:.4f}")
        print(f"  Specificity: {overall['specificity']:.4f}")
        print(f"  F1 Score:    {overall['f1']:.4f}")
        if 'roc_auc' in overall and not np.isnan(overall['roc_auc']):
            print(f"  ROC-AUC:     {overall['roc_auc']:.4f}")
        
        mean = summary['mean_metrics']
        std = summary['std_metrics']
        print(f"\nMean ± Std (across subjects):")
        print(f"  Accuracy:    {mean['accuracy']:.4f} ± {std['accuracy']:.4f}")
        print(f"  Sensitivity: {mean['sensitivity']:.4f} ± {std['sensitivity']:.4f}")
        print(f"  Specificity: {mean['specificity']:.4f} ± {std['specificity']:.4f}")
        print(f"  F1 Score:    {mean['f1']:.4f} ± {std['f1']:.4f}")


class ElderlyTestValidator:
    """
    Train on young subjects, test on elderly subjects.
    Clinically interesting evaluation approach.
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def validate(self, X, y, subjects, model_factory, normalizer_class=StandardScaler):
        """
        Train on young (SA*), test on elderly (SE*).
        
        Args:
            X: Features array
            y: Labels array
            subjects: Subject IDs
            model_factory: Function that returns a new model instance
            normalizer_class: Normalizer class
        
        Returns:
            Dictionary with results
        """
        # Separate young and elderly subjects
        subjects = np.array(subjects)
        
        young_mask = np.array([s.startswith('SA') for s in subjects])
        elderly_mask = np.array([s.startswith('SE') for s in subjects])
        
        X_train, y_train = X[young_mask], y[young_mask]
        X_test, y_test = X[elderly_mask], y[elderly_mask]
        
        if len(X_test) == 0:
            print("Warning: No elderly subjects found for testing")
            return None
        
        if self.verbose:
            print(f"Training on {len(X_train)} samples from young subjects")
            print(f"Testing on {len(X_test)} samples from elderly subjects")
        
        # Normalize
        normalizer = normalizer_class()
        X_train_norm = normalizer.fit_transform(X_train)
        X_test_norm = normalizer.transform(X_test)
        
        # Train and evaluate
        model = model_factory()
        model.fit(X_train_norm, y_train)
        
        y_pred = model.predict(X_test_norm)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test_norm)[:, 1]
            except:
                pass
        
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        
        if self.verbose:
            print("\nYoung → Elderly Transfer Results:")
            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            print(f"  F1 Score:    {metrics['f1']:.4f}")
        
        return {
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'model': model,
            'normalizer': normalizer
        }


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, labels=['ADL', 'Fall'], 
                          title='Confusion Matrix', save_path=None):
    """Plot confusion matrix heatmap."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_proba, title='ROC Curve', save_path=None):
    """Plot ROC curve."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(y_true, y_proba, title='Precision-Recall Curve', 
                                 save_path=None):
    """Plot precision-recall curve."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AP = {ap:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_comparison(results_list, model_names, metric_names=['accuracy', 'sensitivity', 'specificity', 'f1'],
                            title='Model Comparison', save_path=None):
    """Plot bar chart comparing multiple models."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(results_list)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (results, name) in enumerate(zip(results_list, model_names)):
        values = [results.get(m, 0) for m in metric_names]
        offset = width * i - width * len(results_list) / 2
        ax.bar(x + offset, values, width, label=name)
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_subject_performance(loso_results, metric='f1', save_path=None):
    """Plot per-subject performance from LOSO results."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return
    
    subjects = [r['subject'] for r in loso_results]
    values = [r[metric] for r in loso_results]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(subjects)), values, color='steelblue')
    
    # Color elderly subjects differently
    for i, (subj, bar) in enumerate(zip(subjects, bars)):
        if subj.startswith('SE'):
            bar.set_color('coral')
    
    plt.axhline(y=np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
    plt.xlabel('Subject')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} by Subject (LOSO CV)')
    plt.xticks(range(len(subjects)), subjects, rotation=90)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# REPORT GENERATION
# =============================================================================
def generate_evaluation_report(results, output_path=None):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Optional path to save the report
    
    Returns:
        Report string
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FALL DETECTION SYSTEM - EVALUATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Overall metrics
    if 'overall' in results:
        report_lines.append("OVERALL CLASSIFICATION METRICS")
        report_lines.append("-" * 40)
        overall = results['overall']
        report_lines.append(f"Accuracy:        {overall['accuracy']:.4f}")
        report_lines.append(f"Precision:       {overall['precision']:.4f}")
        report_lines.append(f"Recall/Sens.:    {overall['sensitivity']:.4f}")
        report_lines.append(f"Specificity:     {overall['specificity']:.4f}")
        report_lines.append(f"F1 Score:        {overall['f1']:.4f}")
        report_lines.append(f"MCC:             {overall['mcc']:.4f}")
        if 'roc_auc' in overall and not np.isnan(overall['roc_auc']):
            report_lines.append(f"ROC-AUC:         {overall['roc_auc']:.4f}")
            report_lines.append(f"PR-AUC:          {overall['pr_auc']:.4f}")
        report_lines.append("")
        
        report_lines.append("CONFUSION MATRIX")
        report_lines.append("-" * 40)
        report_lines.append(f"True Positives:  {overall['true_positives']}")
        report_lines.append(f"True Negatives:  {overall['true_negatives']}")
        report_lines.append(f"False Positives: {overall['false_positives']}")
        report_lines.append(f"False Negatives: {overall['false_negatives']}")
        report_lines.append("")
    
    # Cross-validation statistics
    if 'mean_metrics' in results and 'std_metrics' in results:
        report_lines.append("CROSS-VALIDATION STATISTICS (Mean ± Std)")
        report_lines.append("-" * 40)
        mean = results['mean_metrics']
        std = results['std_metrics']
        for metric in ['accuracy', 'sensitivity', 'specificity', 'f1']:
            if metric in mean:
                report_lines.append(f"{metric.capitalize():15}: {mean[metric]:.4f} ± {std[metric]:.4f}")
        report_lines.append("")
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


# =============================================================================
# LEGACY FUNCTION (for backward compatibility)
# =============================================================================
def evaluate(model, X_test, y_test, verbose=True):
    """
    Legacy evaluation function.
    Maintained for backward compatibility.
    """
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except:
            pass
    
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    
    if verbose:
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['ADL', 'Fall']))
        print(f"\nSensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
            print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    
    return metrics
