# src/explainability.py
"""
SHAP-based explainability for fall detection models.
Provides insights into feature importance and prediction drivers.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Try importing SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Install with: pip install shap")

# Try importing matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# =============================================================================
# SHAP EXPLAINER
# =============================================================================
class FallDetectionExplainer:
    """
    SHAP-based explainer for fall detection models.
    Highlights important features for predictions.
    """
    
    def __init__(self, model, X_background=None, feature_names=None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model (should have predict or predict_proba method)
            X_background: Background data for SHAP (sample of training data)
            feature_names: List of feature names
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None
        
        # Create appropriate explainer based on model type
        self._create_explainer(X_background)
    
    def _create_explainer(self, X_background):
        """Create SHAP explainer based on model type."""
        model_type = type(self.model).__name__
        
        # Get the underlying sklearn model if wrapped
        underlying_model = getattr(self.model, 'model', self.model)
        
        if X_background is None:
            print("Warning: No background data provided. Using KernelExplainer which may be slow.")
            
        # Choose explainer based on model type
        if 'XGB' in model_type or 'xgb' in str(type(underlying_model)).lower():
            self.explainer = shap.TreeExplainer(underlying_model)
            self.explainer_type = 'tree'
        elif 'RandomForest' in model_type or 'GradientBoosting' in model_type:
            self.explainer = shap.TreeExplainer(underlying_model)
            self.explainer_type = 'tree'
        elif X_background is not None:
            # Use KernelExplainer for other models
            if hasattr(underlying_model, 'predict_proba'):
                self.explainer = shap.KernelExplainer(
                    underlying_model.predict_proba, 
                    shap.sample(X_background, min(100, len(X_background)))
                )
            else:
                self.explainer = shap.KernelExplainer(
                    underlying_model.predict,
                    shap.sample(X_background, min(100, len(X_background)))
                )
            self.explainer_type = 'kernel'
        else:
            raise ValueError("X_background is required for non-tree models")
    
    def compute_shap_values(self, X, check_additivity=False):
        """
        Compute SHAP values for given samples.
        
        Args:
            X: Input features to explain
            check_additivity: Whether to check SHAP additivity
        
        Returns:
            SHAP values array
        """
        if self.explainer_type == 'tree':
            self.shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
        else:
            self.shap_values = self.explainer.shap_values(X)
        
        # For binary classification, get values for positive class (fall)
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            self.shap_values = self.shap_values[1]
        
        return self.shap_values
    
    def get_feature_importance(self, X=None):
        """
        Get global feature importance based on mean absolute SHAP values.
        
        Args:
            X: Optional data to compute SHAP values if not already computed
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.shap_values is None and X is not None:
            self.compute_shap_values(X)
        
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        # Mean absolute SHAP value per feature
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        if self.feature_names is not None:
            df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
        else:
            df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            })
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def plot_summary(self, X=None, max_display=20, save_path=None):
        """
        Create SHAP summary plot.
        
        Args:
            X: Feature matrix (if SHAP values not computed)
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if not HAS_PLOTTING:
            print("Matplotlib not available for plotting")
            return
        
        if self.shap_values is None and X is not None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            self.shap_values, 
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_bar(self, X=None, max_display=20, save_path=None):
        """
        Create SHAP bar plot (mean absolute values).
        
        Args:
            X: Feature matrix (if SHAP values not computed)
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if not HAS_PLOTTING:
            print("Matplotlib not available for plotting")
            return
        
        if self.shap_values is None and X is not None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_waterfall(self, X, sample_idx=0, save_path=None):
        """
        Create waterfall plot for a single prediction.
        
        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            save_path: Path to save the plot
        """
        if not HAS_PLOTTING:
            print("Matplotlib not available for plotting")
            return
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(
            shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                data=X[sample_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_force(self, X, sample_idx=0, save_path=None):
        """
        Create force plot for a single prediction.
        
        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            save_path: Path to save the plot
        """
        if not HAS_PLOTTING:
            print("Matplotlib not available for plotting")
            return
        
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]  # For binary classification
        
        shap.force_plot(
            expected_value,
            self.shap_values[sample_idx],
            X[sample_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def explain_prediction(self, X, sample_idx, top_k=10):
        """
        Get human-readable explanation for a single prediction.
        
        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            top_k: Number of top features to include
        
        Returns:
            Dictionary with explanation
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        sample_shap = self.shap_values[sample_idx]
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:top_k]
        
        explanations = []
        for idx in sorted_idx:
            feature = self.feature_names[idx] if self.feature_names else f'feature_{idx}'
            value = X[sample_idx, idx]
            shap_value = sample_shap[idx]
            direction = 'increases' if shap_value > 0 else 'decreases'
            
            explanations.append({
                'feature': feature,
                'value': value,
                'shap_value': shap_value,
                'direction': direction,
                'explanation': f"{feature}={value:.3f} {direction} fall probability"
            })
        
        return {
            'sample_idx': sample_idx,
            'top_features': explanations,
            'total_shap_effect': np.sum(sample_shap)
        }


# =============================================================================
# PHYSIOLOGICAL INTERPRETATION
# =============================================================================
def interpret_fall_features(feature_importance_df, top_k=10):
    """
    Provide physiological interpretation of important features.
    
    Args:
        feature_importance_df: DataFrame with feature importance
        top_k: Number of top features to interpret
    
    Returns:
        List of interpretations
    """
    # Mapping of feature patterns to physiological meaning
    interpretations_map = {
        'svm': 'Signal Vector Magnitude - overall acceleration intensity',
        'peak': 'Peak acceleration - indicates sudden impact events',
        'jerk': 'Rate of acceleration change - higher in falls due to abrupt movements',
        'impact': 'Impact characteristics - duration and intensity of collision',
        'fall_index': 'Ratio of peak to mean acceleration - sharp peaks indicate falls',
        'spec_entropy': 'Spectral complexity - falls have distinct frequency patterns',
        'energy_low': 'Low frequency energy (0-5 Hz) - body movement component',
        'energy_high': 'High frequency energy (5-20 Hz) - impact/vibration component',
        'rms': 'Root mean square - overall signal power',
        'std': 'Standard deviation - signal variability',
        'range': 'Signal range - amplitude of movement',
        'sample_entropy': 'Signal regularity - falls show irregular patterns',
        'zcr': 'Zero crossing rate - frequency indicator',
        'autocorr': 'Autocorrelation - signal self-similarity',
        'harmonic': 'Harmonic ratio - gait smoothness indicator',
        'stride': 'Stride characteristics - gait regularity',
        'step': 'Step characteristics - walking pattern'
    }
    
    interpretations = []
    
    for i, row in feature_importance_df.head(top_k).iterrows():
        feature = row['feature']
        importance = row['importance']
        
        # Find matching interpretation
        interpretation = "General motion characteristic"
        for pattern, meaning in interpretations_map.items():
            if pattern.lower() in feature.lower():
                interpretation = meaning
                break
        
        interpretations.append({
            'rank': i + 1,
            'feature': feature,
            'importance': importance,
            'physiological_meaning': interpretation
        })
    
    return interpretations


def generate_shap_report(explainer, X, feature_names=None, output_path=None):
    """
    Generate comprehensive SHAP-based explanation report.
    
    Args:
        explainer: FallDetectionExplainer instance
        X: Feature matrix
        feature_names: Optional feature names
        output_path: Path to save report
    
    Returns:
        Report string
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FALL DETECTION MODEL - EXPLAINABILITY REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Compute SHAP values if not done
    if explainer.shap_values is None:
        explainer.compute_shap_values(X)
    
    # Feature importance
    importance_df = explainer.get_feature_importance()
    
    report_lines.append("TOP 20 MOST IMPORTANT FEATURES")
    report_lines.append("-" * 50)
    report_lines.append(f"{'Rank':<6}{'Feature':<40}{'Importance':<15}")
    report_lines.append("-" * 50)
    
    for i, row in importance_df.head(20).iterrows():
        report_lines.append(f"{i+1:<6}{row['feature']:<40}{row['importance']:<15.6f}")
    
    report_lines.append("")
    
    # Physiological interpretation
    interpretations = interpret_fall_features(importance_df)
    
    report_lines.append("PHYSIOLOGICAL INTERPRETATION OF KEY FEATURES")
    report_lines.append("-" * 70)
    
    for interp in interpretations:
        report_lines.append(f"\n{interp['rank']}. {interp['feature']}")
        report_lines.append(f"   Importance: {interp['importance']:.6f}")
        report_lines.append(f"   Meaning: {interp['physiological_meaning']}")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("KEY INSIGHTS FOR FALL DETECTION")
    report_lines.append("-" * 70)
    report_lines.append("""
1. Peak Acceleration: Falls produce significantly higher peak accelerations
   compared to normal activities due to sudden impact with ground/objects.

2. Jerk Features: The rate of change of acceleration (jerk) is typically
   much higher during falls, indicating abrupt changes in motion.

3. Signal Vector Magnitude: The combined magnitude across all axes shows
   distinct patterns during falls vs activities of daily living.

4. Spectral Entropy: Falls have characteristic frequency compositions that
   differ from regular activities, captured by spectral entropy.

5. Impact Duration: True falls show specific temporal patterns of impact
   that distinguish them from activities like sitting down quickly.
""")
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def analyze_misclassifications(explainer, X, y_true, y_pred, feature_names=None, n_samples=5):
    """
    Analyze misclassified samples using SHAP.
    
    Args:
        explainer: FallDetectionExplainer instance
        X: Feature matrix
        y_true: True labels
        y_pred: Predicted labels
        feature_names: Feature names
        n_samples: Number of samples to analyze
    
    Returns:
        Analysis results
    """
    # Find misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    # Separate false positives and false negatives
    false_positives = misclassified_idx[y_pred[misclassified_idx] == 1]
    false_negatives = misclassified_idx[y_pred[misclassified_idx] == 0]
    
    analysis = {
        'n_misclassified': len(misclassified_idx),
        'n_false_positives': len(false_positives),
        'n_false_negatives': len(false_negatives),
        'false_positive_explanations': [],
        'false_negative_explanations': []
    }
    
    # Analyze false positives
    for idx in false_positives[:n_samples]:
        exp = explainer.explain_prediction(X, idx)
        analysis['false_positive_explanations'].append(exp)
    
    # Analyze false negatives
    for idx in false_negatives[:n_samples]:
        exp = explainer.explain_prediction(X, idx)
        analysis['false_negative_explanations'].append(exp)
    
    return analysis
