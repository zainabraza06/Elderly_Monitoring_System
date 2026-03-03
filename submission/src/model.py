# src/model.py
"""
Model implementations for fall detection, gait stability, and frailty prediction.
Includes Random Forest, XGBoost, MLP, and 1D CNN models.
"""
import numpy as np
import pickle
from pathlib import Path

# Scikit-learn imports
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Try importing XGBoost (optional dependency)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

# Try importing TensorFlow/Keras for deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, Flatten, Dense, Dropout, 
        BatchNormalization, GlobalAveragePooling1D
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")


# =============================================================================
# FALL DETECTION MODELS (Binary Classification)
# =============================================================================
class FallDetectorRF:
    """Random Forest classifier for fall detection."""
    
    def __init__(self, n_estimators=200, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, class_weight='balanced', random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )
        self.name = "RandomForest"
    
    def fit(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        return self.model.feature_importances_


class FallDetectorXGB:
    """XGBoost classifier for fall detection."""
    
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1,
                 scale_pos_weight=None, random_state=42):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,  # Handle class imbalance
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_state,
            use_label_encoder=False,
            n_jobs=-1
        )
        self.name = "XGBoost"
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        """Train the model."""
        if eval_set is not None and early_stopping_rounds is not None:
            self.model.fit(X, y, eval_set=eval_set, 
                          early_stopping_rounds=early_stopping_rounds, verbose=False)
        else:
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        return self.model.feature_importances_


class FallDetectorMLP:
    """Multi-layer Perceptron classifier for fall detection."""
    
    def __init__(self, hidden_layers=(256, 128, 64), activation='relu',
                 alpha=0.001, max_iter=500, random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            alpha=alpha,  # L2 regularization
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state
        )
        self.name = "MLP"
    
    def fit(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)


class FallDetectorSVM:
    """
    Linear SVM with Platt calibration for fall detection.
    Fast on large datasets; good generalisation with proper normalisation.
    """

    def __init__(self, C=1.0, class_weight='balanced', random_state=42, cv=3):
        base = LinearSVC(C=C, class_weight=class_weight,
                         max_iter=2000, random_state=random_state)
        self.model = CalibratedClassifierCV(base, cv=cv, method='sigmoid')
        self.name = "SVM"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class FallDetectorLDA:
    """
    Linear Discriminant Analysis – fast linear baseline.
    Also acts as a feature-space projector.
    """

    def __init__(self):
        self.model = LinearDiscriminantAnalysis()
        self.name = "LDA"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class FallDetectorCNN:
    """1D CNN classifier for fall detection on raw windows."""
    
    def __init__(self, input_shape=(125, 12), n_filters=64, 
                 kernel_size=3, dropout_rate=0.3):
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.model = None
        self.name = "1D-CNN"
        self._build_model()
    
    def _build_model(self):
        """Build the CNN architecture."""
        self.model = Sequential([
            # First conv block
            Conv1D(self.n_filters, self.kernel_size, activation='relu', 
                   input_shape=self.input_shape, padding='same'),
            BatchNormalization(),
            Conv1D(self.n_filters, self.kernel_size, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(self.dropout_rate),
            
            # Second conv block
            Conv1D(self.n_filters * 2, self.kernel_size, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(self.n_filters * 2, self.kernel_size, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Dropout(self.dropout_rate),
            
            # Third conv block
            Conv1D(self.n_filters * 4, self.kernel_size, activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            
            # Dense layers
            Dense(128, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, X, y, validation_data=None, epochs=50, batch_size=32):
        """Train the model."""
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return self
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.model.predict(X, verbose=0)
        return (proba.flatten() > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        proba = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])


# =============================================================================
# GAIT STABILITY REGRESSOR
# =============================================================================
class GaitStabilityRegressor:
    """
    Gradient Boosting Regressor for gait stability estimation.
    Trained only on walking activities (D01, D02 in SisFall).
    Outputs stability score between 0 (unstable) and 1 (stable).
    """
    
    def __init__(self, model_type='gbr', n_estimators=100, max_depth=5,
                 learning_rate=0.1, random_state=42):
        self.model_type = model_type
        
        if model_type == 'gbr':
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
        elif model_type == 'xgb' and HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
        
        self.name = f"GaitStabilityRegressor_{model_type.upper()}"
    
    def fit(self, X, y):
        """
        Train the model.
        y should be stability scores in [0, 1] range.
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict stability scores.
        Returns values clipped to [0, 1] range.
        """
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, 1)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


# =============================================================================
# FRAILTY PREDICTOR
# =============================================================================
class FrailtyPredictor:
    """
    Frailty prediction model for elderly subjects.
    Outputs frailty index between 0 (robust) and 1 (frail).
    
    Frailty proxy is computed from:
    - Gait variability
    - Reduced acceleration amplitude
    - Increased jerk
    - Increased step time variability
    """
    
    def __init__(self, model_type='mlp', random_state=42):
        self.model_type = model_type
        
        if model_type == 'mlp':
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                alpha=0.01,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=random_state
            )
        elif model_type == 'xgb' and HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'gbr':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            self.model = Ridge(alpha=1.0, random_state=random_state)
        
        self.name = f"FrailtyPredictor_{model_type.upper()}"
    
    def fit(self, X, y):
        """
        Train the model.
        y should be frailty scores in [0, 1] range.
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict frailty scores.
        Returns values clipped to [0, 1] range.
        """
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, 1)


# =============================================================================
# FRAILTY SCORE COMPUTATION (Synthetic Labels)
# =============================================================================
def compute_frailty_proxy(features, feature_names=None):
    """
    Compute synthetic frailty index from extracted features.
    This creates pseudo-labels for elderly subjects based on physiological indicators.
    
    Frailty indicators:
    1. High gait variability (stride_variability)
    2. Low acceleration amplitude (reduced RMS)
    3. High jerk values (movement irregularity)
    4. Low step/stride regularity
    
    Args:
        features: Feature array (n_samples, n_features) or single sample
        feature_names: List of feature names for identifying relevant features
    
    Returns:
        Frailty scores in [0, 1] range
    """
    features = np.atleast_2d(features)
    n_samples = features.shape[0]
    
    # Initialize scores
    frailty_scores = np.zeros(n_samples)
    
    # Component weights
    weights = {
        'variability': 0.25,
        'amplitude': 0.25,
        'jerk': 0.25,
        'regularity': 0.25
    }
    
    # Compute components (using statistical proxies)
    # Higher variability in features → higher frailty
    feature_std = np.std(features, axis=1)
    variability_component = (feature_std - np.min(feature_std)) / (np.max(feature_std) - np.min(feature_std) + 1e-8)
    
    # Lower RMS amplitude → higher frailty (inverse relationship)
    # Assuming RMS features are in certain columns
    rms_features = features[:, 3*12:4*12].mean(axis=1) if features.shape[1] > 48 else np.mean(features[:, :12], axis=1)
    amplitude_component = 1 - (rms_features - np.min(rms_features)) / (np.max(rms_features) - np.min(rms_features) + 1e-8)
    
    # Higher jerk → higher frailty
    # Assuming jerk features are available
    jerk_idx_start = 10 * 12  # After skewness and kurtosis
    if features.shape[1] > jerk_idx_start:
        jerk_features = features[:, jerk_idx_start:jerk_idx_start+12].mean(axis=1)
    else:
        jerk_features = np.std(np.diff(features, axis=1), axis=1)
    jerk_component = (jerk_features - np.min(jerk_features)) / (np.max(jerk_features) - np.min(jerk_features) + 1e-8)
    
    # Lower regularity → higher frailty
    regularity_component = 1 - variability_component
    
    # Combine components
    frailty_scores = (
        weights['variability'] * variability_component +
        weights['amplitude'] * amplitude_component +
        weights['jerk'] * jerk_component +
        weights['regularity'] * regularity_component
    )
    
    # Normalize to [0, 1]
    frailty_scores = (frailty_scores - np.min(frailty_scores)) / (np.max(frailty_scores) - np.min(frailty_scores) + 1e-8)
    
    return frailty_scores


def compute_stability_score(features, is_walking=True):
    """
    Compute gait stability score from features.
    
    Stability indicators:
    1. Low stride variability
    2. High step/stride regularity
    3. High harmonic ratio
    4. Consistent acceleration patterns
    
    Args:
        features: Feature array
        is_walking: Whether the activity is walking
    
    Returns:
        Stability scores in [0, 1] range
    """
    features = np.atleast_2d(features)
    
    if not is_walking:
        return np.ones(features.shape[0]) * 0.5  # Neutral score for non-walking
    
    # Compute stability from feature statistics
    # Lower variance = more stable
    feature_var = np.var(features, axis=1)
    stability = 1 - (feature_var - np.min(feature_var)) / (np.max(feature_var) - np.min(feature_var) + 1e-8)
    
    return np.clip(stability, 0, 1)


# =============================================================================
# UNIFIED FALL DETECTION SYSTEM
# =============================================================================
class UnifiedFallDetectionSystem:
    """
    Unified system combining:
    1. Fall detection (binary classification)
    2. Gait stability assessment (regression)
    3. Frailty estimation (regression)
    
    Decision Logic:
    if Fall_Prob > threshold:
        Output = FALL
    else:
        Output = "ADL"
        Also output:
            - Stability Score
            - Frailty Score
    """
    
    def __init__(self, fall_model='xgb', stability_model='gbr', 
                 frailty_model='mlp', fall_threshold=0.5):
        self.fall_threshold = fall_threshold
        
        # Initialize models
        if fall_model == 'xgb' and HAS_XGBOOST:
            self.fall_detector = FallDetectorXGB()
        elif fall_model == 'rf':
            self.fall_detector = FallDetectorRF()
        else:
            self.fall_detector = FallDetectorMLP()
        
        self.stability_regressor = GaitStabilityRegressor(model_type=stability_model)
        self.frailty_predictor = FrailtyPredictor(model_type=frailty_model)
        
        self.is_fitted = {
            'fall': False,
            'stability': False,
            'frailty': False
        }
        self._n_features_stability = None
        self._n_features_frailty = None
    
    def fit_fall_detector(self, X, y):
        """Train fall detection model."""
        self.fall_detector.fit(X, y)
        self.is_fitted['fall'] = True
        return self
    
    def fit_stability_regressor(self, X, y):
        """Train stability regressor on walking data."""
        self.stability_regressor.fit(X, y)
        self._n_features_stability = X.shape[1]
        self.is_fitted['stability'] = True
        return self

    def fit_frailty_predictor(self, X, y):
        """Train frailty predictor on elderly data."""
        self.frailty_predictor.fit(X, y)
        self._n_features_frailty = X.shape[1]
        self.is_fitted['frailty'] = True
        return self
    
    def predict(self, X, return_all=True):
        """
        Make unified prediction.
        
        Args:
            X: Input features
            return_all: Whether to return all scores or just fall prediction
        
        Returns:
            Dictionary with:
            - 'is_fall': Boolean array
            - 'fall_prob': Fall probabilities
            - 'stability_score': Stability scores (if fitted)
            - 'frailty_score': Frailty scores (if fitted)
        """
        results = {}
        
        # Fall detection
        if self.is_fitted['fall']:
            fall_proba = self.fall_detector.predict_proba(X)[:, 1]
            results['fall_prob'] = fall_proba
            results['is_fall'] = fall_proba > self.fall_threshold
        
        if return_all:
            # Stability assessment — only run if feature count matches training data
            if self.is_fitted['stability']:
                if self._n_features_stability is None or X.shape[1] == self._n_features_stability:
                    results['stability_score'] = self.stability_regressor.predict(X)
                # else: input has different feature set (e.g. no gait features); skip silently

            # Frailty estimation — only run if feature count matches training data
            if self.is_fitted['frailty']:
                if self._n_features_frailty is None or X.shape[1] == self._n_features_frailty:
                    results['frailty_score'] = self.frailty_predictor.predict(X)
                # else: input has different feature set; skip silently
        
        return results
    
    def get_risk_level(self, X):
        """
        Get overall risk level for each sample.
        
        Returns:
            Risk levels: 'low', 'medium', 'high', 'fall_detected'
        """
        predictions = self.predict(X)
        
        risk_levels = []
        for i in range(len(X)):
            if predictions.get('is_fall', np.zeros(len(X), dtype=bool))[i]:
                risk_levels.append('fall_detected')
            else:
                stab_arr = predictions.get('stability_score', None)
                frail_arr = predictions.get('frailty_score', None)
                stability = float(stab_arr[i]) if stab_arr is not None else 1.0
                frailty   = float(frail_arr[i]) if frail_arr is not None else 0.0
                
                # Combined risk score
                risk_score = (1 - stability + frailty) / 2
                
                if risk_score < 0.3:
                    risk_levels.append('low')
                elif risk_score < 0.6:
                    risk_levels.append('medium')
                else:
                    risk_levels.append('high')
        
        return risk_levels
    
    def save(self, filepath):
        """Save the unified system to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load a unified system from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# MODEL FACTORY FUNCTION
# =============================================================================
def create_fall_detector(model_type='xgb', **kwargs):
    """
    Factory function to create fall detection model.
    
    Args:
        model_type: 'rf', 'xgb', 'mlp', or 'cnn'
        **kwargs: Model-specific parameters
    
    Returns:
        Model instance
    """
    if model_type == 'rf':
        return FallDetectorRF(**kwargs)
    elif model_type == 'xgb':
        if not HAS_XGBOOST:
            print("XGBoost not available, falling back to Random Forest")
            return FallDetectorRF(**kwargs)
        return FallDetectorXGB(**kwargs)
    elif model_type == 'mlp':
        return FallDetectorMLP(**kwargs)
    elif model_type == 'svm':
        svm_kwargs = {k: v for k, v in kwargs.items() if k in ('C', 'class_weight', 'random_state', 'cv')}
        return FallDetectorSVM(**svm_kwargs)
    elif model_type == 'lda':
        return FallDetectorLDA()
    elif model_type == 'cnn':
        if not HAS_TENSORFLOW:
            print("TensorFlow not available, falling back to MLP")
            return FallDetectorMLP(**kwargs)
        return FallDetectorCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# LEGACY FUNCTION (for backward compatibility)
# =============================================================================
def train_model(X_train, y_train, model_type='rf'):
    """
    Legacy function to train a fall detection model.
    Maintained for backward compatibility with existing code.
    """
    model = create_fall_detector(model_type)
    model.fit(X_train, y_train)
    return model
