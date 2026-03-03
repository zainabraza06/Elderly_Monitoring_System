# src/dataset_builder.py
"""
Dataset building utilities for SisFall fall detection system.
Handles preprocessing, windowing, and feature extraction.
"""
import numpy as np
import os
from pathlib import Path
from src.cache import (
    DEFAULT_CACHE_DIR, make_cache_key,
    save_cache, load_cache, cache_exists, cache_info
)
from src.preprocessing import (
    preprocess_single_trial, 
    preprocess_trials,
    sliding_window,
    WINDOW_SIZE,
    OVERLAP,
    TARGET_HZ
)
from src.features import extract_features, get_feature_names, extract_features_batch


# =============================================================================
# ACTIVITY CODE PARSING
# =============================================================================
# Walking activities for gait analysis
WALKING_ACTIVITIES = ['D01', 'D02', 'D03', 'D04', 'D05', 'D06']

# Fall activities
FALL_ACTIVITIES = [f'F{i:02d}' for i in range(1, 16)]

# ADL activities (non-fall)
ADL_ACTIVITIES = [f'D{i:02d}' for i in range(1, 20)]


def get_activity_code(filename):
    """Extract activity code from filename."""
    return filename.split('_')[0]


def is_walking_activity(filename):
    """Check if the file is from a walking activity."""
    activity = get_activity_code(filename)
    return activity in WALKING_ACTIVITIES


def is_fall_activity(filename):
    """Check if the file is from a fall activity."""
    activity = get_activity_code(filename)
    return activity.startswith('F')


def is_elderly_subject(subject_id):
    """Check if subject is elderly (SE*)."""
    return subject_id.startswith('SE')


# =============================================================================
# DATASET BUILDING
# =============================================================================
def build_dataset(X_trials, y_trials, subjects=None, activity_codes=None,
                  include_gait_features=False, verbose=True,
                  data_root=None, cache_dir=None, force_rebuild=False):
    """
    Build feature dataset from raw trials.

    Results are cached to disk on the first run.  Subsequent calls with the
    same parameters and dataset return instantly from the cache.

    Args:
        X_trials: List of raw trial arrays
        y_trials: List of labels (0=ADL, 1=Fall)
        subjects: List of subject IDs (optional)
        activity_codes: List of activity codes (optional)
        include_gait_features: Whether to include gait-specific features
        verbose: Print progress
        data_root: Dataset root path used for cache key (auto-disabled if None)
        cache_dir: Directory to store cache (default: .feature_cache/)
        force_rebuild: Ignore existing cache and recompute

    Returns:
        X: Feature matrix (n_windows, n_features)
        y: Labels array (n_windows,)
        metadata: Dictionary with subjects and activities per window
    """
    # ---- Cache lookup -------------------------------------------------------
    _cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    _cache_key = None

    if data_root and not force_rebuild:
        _params = dict(
            window_size=WINDOW_SIZE, overlap=OVERLAP, target_hz=TARGET_HZ,
            include_gait=include_gait_features, dataset_type="full"
        )
        _cache_key = make_cache_key(data_root, _params)

        if cache_exists(_cache_dir, _cache_key):
            if verbose:
                print(f"[Cache] Loading features from cache (key: {_cache_key})")
                info = cache_info(_cache_dir)
                print(f"        Cache size: {info['total_size_mb']} MB  |  "
                      f"path: {info['path']}")
            return load_cache(_cache_dir, _cache_key)

    # ---- Build from scratch -------------------------------------------------
    X_all = []
    y_all = []
    window_subjects = []
    window_activities = []
    
    total_windows = 0
    
    for i, (trial, label) in enumerate(zip(X_trials, y_trials)):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processing trial {i + 1}/{len(X_trials)}")
        
        # Preprocess trial
        trial_processed = preprocess_single_trial(trial)
        
        # Create windows
        windows = sliding_window(trial_processed, WINDOW_SIZE, OVERLAP)
        
        # Extract features for each window
        for window in windows:
            # Determine if this is a walking activity for gait feature extraction
            is_walking = False
            if activity_codes is not None:
                is_walking = activity_codes[i] in WALKING_ACTIVITIES
            
            feats = extract_features(
                window, 
                include_gait=include_gait_features and is_walking,
                fs=TARGET_HZ
            )
            X_all.append(feats)
            y_all.append(label)
            
            if subjects is not None:
                window_subjects.append(subjects[i])
            if activity_codes is not None:
                window_activities.append(activity_codes[i])
            
            total_windows += 1
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if verbose:
        print(f"\nDataset built:")
        print(f"  Total windows: {len(X)}")
        print(f"  Feature dimensions: {X.shape[1]}")
        print(f"  Fall windows: {np.sum(y)}")
        print(f"  ADL windows: {len(y) - np.sum(y)}")
    
    metadata = {
        'subjects': window_subjects if subjects else None,
        'activities': window_activities if activity_codes else None,
        'n_features': X.shape[1],
        'feature_names': get_feature_names(n_channels=12, include_gait=include_gait_features)
    }

    # ---- Persist to cache ---------------------------------------------------
    if _cache_key:
        save_cache(_cache_dir, _cache_key, X, y, metadata)
        if verbose:
            info = cache_info(_cache_dir)
            print(f"[Cache] Features saved  (key: {_cache_key},  "
                  f"size: {info['total_size_mb']} MB)")

    return X, y, metadata


def build_walking_dataset(X_trials, y_trials, subjects, activity_codes, verbose=True,
                          data_root=None, cache_dir=None, force_rebuild=False):
    """
    Build dataset containing only walking activities.
    Used for gait stability analysis.

    Args:
        X_trials: List of raw trial arrays
        y_trials: List of labels
        subjects: List of subject IDs
        activity_codes: List of activity codes
        verbose: Print progress
        data_root: Dataset root used for cache keying
        cache_dir: Cache directory
        force_rebuild: Ignore existing cache

    Returns:
        X, y, metadata for walking activities only
    """
    _cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    _cache_key = None

    if data_root and not force_rebuild:
        _params = dict(
            window_size=WINDOW_SIZE, overlap=OVERLAP, target_hz=TARGET_HZ,
            include_gait=True, dataset_type="walking"
        )
        _cache_key = make_cache_key(data_root, _params)
        if cache_exists(_cache_dir, _cache_key):
            if verbose:
                print(f"[Cache] Loading walking features from cache (key: {_cache_key})")
            return load_cache(_cache_dir, _cache_key)

    # Filter for walking activities only
    walking_mask = [code in WALKING_ACTIVITIES for code in activity_codes]

    X_walking = [X_trials[i] for i in range(len(X_trials)) if walking_mask[i]]
    y_walking = [y_trials[i] for i in range(len(y_trials)) if walking_mask[i]]
    subjects_walking = [subjects[i] for i in range(len(subjects)) if walking_mask[i]]
    activities_walking = [activity_codes[i] for i in range(len(activity_codes)) if walking_mask[i]]

    if verbose:
        print(f"Walking trials: {len(X_walking)} out of {len(X_trials)}")

    X, y, metadata = build_dataset(
        X_walking, y_walking, subjects_walking, activities_walking,
        include_gait_features=True, verbose=verbose
        # No data_root here — prevents double-caching via the inner call
    )

    if _cache_key:
        save_cache(_cache_dir, _cache_key, X, y, metadata)
        if verbose:
            info = cache_info(_cache_dir)
            print(f"[Cache] Walking features saved  (key: {_cache_key},  "
                  f"size: {info['total_size_mb']} MB)")

    return X, y, metadata


def build_elderly_dataset(X_trials, y_trials, subjects, activity_codes=None, verbose=True,
                          data_root=None, cache_dir=None, force_rebuild=False):
    """
    Build dataset containing only elderly subjects.
    Used for frailty prediction.

    Args:
        X_trials: List of raw trial arrays
        y_trials: List of labels
        subjects: List of subject IDs
        activity_codes: List of activity codes (optional)
        verbose: Print progress
        data_root: Dataset root used for cache keying
        cache_dir: Cache directory
        force_rebuild: Ignore existing cache

    Returns:
        X, y, metadata for elderly subjects only
    """
    _cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    _cache_key = None

    if data_root and not force_rebuild:
        _params = dict(
            window_size=WINDOW_SIZE, overlap=OVERLAP, target_hz=TARGET_HZ,
            include_gait=False, dataset_type="elderly"
        )
        _cache_key = make_cache_key(data_root, _params)
        if cache_exists(_cache_dir, _cache_key):
            if verbose:
                print(f"[Cache] Loading elderly features from cache (key: {_cache_key})")
            return load_cache(_cache_dir, _cache_key)

    # Filter for elderly subjects only
    elderly_mask = [is_elderly_subject(s) for s in subjects]

    X_elderly = [X_trials[i] for i in range(len(X_trials)) if elderly_mask[i]]
    y_elderly = [y_trials[i] for i in range(len(y_trials)) if elderly_mask[i]]
    subjects_elderly = [subjects[i] for i in range(len(subjects)) if elderly_mask[i]]

    activities_elderly = None
    if activity_codes is not None:
        activities_elderly = [activity_codes[i] for i in range(len(activity_codes)) if elderly_mask[i]]

    if verbose:
        print(f"Elderly subjects trials: {len(X_elderly)} out of {len(X_trials)}")

    result = build_dataset(
        X_elderly, y_elderly, subjects_elderly, activities_elderly,
        include_gait_features=False, verbose=verbose
    )
    X, y, metadata = result

    if _cache_key:
        save_cache(_cache_dir, _cache_key, X, y, metadata)
        if verbose:
            info = cache_info(_cache_dir)
            print(f"[Cache] Elderly features saved  (key: {_cache_key},  "
                  f"size: {info['total_size_mb']} MB)")

    return X, y, metadata


def build_raw_windows_dataset(X_trials, y_trials, subjects=None, verbose=True):
    """
    Build dataset of raw (preprocessed) windows for deep learning models.
    Does not extract features - keeps raw signal windows.
    
    Args:
        X_trials: List of raw trial arrays
        y_trials: List of labels
        subjects: List of subject IDs (optional)
        verbose: Print progress
    
    Returns:
        X: Raw windows (n_windows, window_size, n_channels)
        y: Labels (n_windows,)
        window_subjects: Subject IDs per window
    """
    X_all = []
    y_all = []
    window_subjects = []
    
    for i, (trial, label) in enumerate(zip(X_trials, y_trials)):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processing trial {i + 1}/{len(X_trials)}")
        
        # Preprocess trial
        trial_processed = preprocess_single_trial(trial)
        
        # Create windows
        windows = sliding_window(trial_processed, WINDOW_SIZE, OVERLAP)
        
        for window in windows:
            X_all.append(window)
            y_all.append(label)
            if subjects is not None:
                window_subjects.append(subjects[i])
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    if verbose:
        print(f"\nRaw windows dataset built:")
        print(f"  Shape: {X.shape}")
        print(f"  Fall windows: {np.sum(y)}")
        print(f"  ADL windows: {len(y) - np.sum(y)}")
    
    return X, y, window_subjects if subjects else None


# =============================================================================
# DATA SPLITTING
# =============================================================================
def split_by_subjects(X, y, subjects, test_subjects, verbose=True):
    """
    Split data by subject IDs.
    
    Args:
        X: Feature matrix
        y: Labels
        subjects: Subject ID per sample
        test_subjects: List of subjects for test set
        verbose: Print info
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    subjects = np.array(subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    X_train = X[~test_mask]
    X_test = X[test_mask]
    y_train = y[~test_mask]
    y_test = y[test_mask]
    
    if verbose:
        print(f"Train: {len(X_train)} samples from {len(np.unique(subjects[~test_mask]))} subjects")
        print(f"Test: {len(X_test)} samples from {len(test_subjects)} subjects")
    
    return X_train, X_test, y_train, y_test


def split_young_elderly(X, y, subjects, verbose=True):
    """
    Split data into young (train) and elderly (test) subjects.
    
    Args:
        X: Feature matrix
        y: Labels
        subjects: Subject ID per sample
        verbose: Print info
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    subjects = np.array(subjects)
    elderly_mask = np.array([is_elderly_subject(s) for s in subjects])
    
    X_train = X[~elderly_mask]
    X_test = X[elderly_mask]
    y_train = y[~elderly_mask]
    y_test = y[elderly_mask]
    
    if verbose:
        print(f"Young (train): {len(X_train)} samples")
        print(f"Elderly (test): {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_class_weights(y):
    """
    Compute class weights for imbalanced data.
    
    Args:
        y: Labels array
    
    Returns:
        Dictionary of class weights
    """
    n_samples = len(y)
    n_classes = len(np.unique(y))
    
    weights = {}
    for cls in np.unique(y):
        weights[cls] = n_samples / (n_classes * np.sum(y == cls))
    
    return weights


def get_scale_pos_weight(y):
    """
    Compute scale_pos_weight for XGBoost.
    Used to handle class imbalance.
    
    Args:
        y: Labels array
    
    Returns:
        scale_pos_weight value
    """
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    
    return n_neg / n_pos if n_pos > 0 else 1.0


def print_dataset_summary(X, y, metadata=None):
    """Print summary of dataset."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"ADL samples (0): {np.sum(y == 0)}")
    print(f"Fall samples (1): {np.sum(y == 1)}")
    print(f"Class ratio (ADL:Fall): {np.sum(y == 0) / np.sum(y == 1):.2f}:1")
    
    if metadata and metadata.get('subjects'):
        subjects = np.array(metadata['subjects'])
        unique_subjects = np.unique(subjects)
        young = [s for s in unique_subjects if s.startswith('SA')]
        elderly = [s for s in unique_subjects if s.startswith('SE')]
        print(f"Unique subjects: {len(unique_subjects)}")
        print(f"  Young (SA): {len(young)}")
        print(f"  Elderly (SE): {len(elderly)}")
    
    print("=" * 50)

