# src/loader.py
"""
Data loading utilities for SisFall dataset.
"""
import os
import numpy as np


def load_single_sisfall_file(file_path):
    """
    Load a single SisFall text file into a NumPy array.
    Handles trailing semicolons and extra spaces.
    Returns shape: (num_samples, 9)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        line = line.strip().rstrip(';')  # Remove trailing semicolon
        if line:  # skip empty lines
            cleaned_lines.append(line)

    if not cleaned_lines:
        return None  # empty file

    try:
        data = np.genfromtxt(cleaned_lines, delimiter=',', dtype=np.float64)
        return data
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None


def load_sisfall(data_root, verbose=True):
    """
    Load all SisFall trials from a dataset directory.
    
    Args:
        data_root: Path to SisFall dataset root
        verbose: Print loading progress
    
    Returns:
        X_trials: list of np.array of shape (num_samples, 9)
        y_trials: list of labels (0 = ADL, 1 = Fall)
        subjects: list of subject IDs (SA01, SE01, etc.)
        activity_codes: list of activity codes (D01, F01, etc.)
    """
    X_trials = []
    y_trials = []
    subjects = []
    activity_codes = []

    subject_folders = sorted([f for f in os.listdir(data_root) 
                              if os.path.isdir(os.path.join(data_root, f))])
    
    for subject_folder in subject_folders:
        subject_path = os.path.join(data_root, subject_folder)

        files = sorted([f for f in os.listdir(subject_path) if f.endswith(".txt")])
        
        for file in files:
            file_path = os.path.join(subject_path, file)
            data = load_single_sisfall_file(file_path)
            
            if data is None:
                continue  # skip corrupted/empty files

            # Determine label
            label = 1 if file.startswith("F") else 0
            
            # Extract activity code (e.g., D01, F05)
            activity_code = file.split('_')[0]

            X_trials.append(data)
            y_trials.append(label)
            subjects.append(subject_folder)
            activity_codes.append(activity_code)

    if verbose:
        print(f"Loaded {len(X_trials)} trials")
        print(f"  Fall trials: {sum(y_trials)}")
        print(f"  ADL trials: {len(y_trials) - sum(y_trials)}")
        print(f"  Unique subjects: {len(set(subjects))}")
        print(f"  Young subjects (SA): {len([s for s in set(subjects) if s.startswith('SA')])}")
        print(f"  Elderly subjects (SE): {len([s for s in set(subjects) if s.startswith('SE')])}")
    
    return X_trials, y_trials, subjects, activity_codes


def load_sisfall_by_subject(data_root, subject_ids, verbose=True):
    """
    Load SisFall trials for specific subjects only.
    
    Args:
        data_root: Path to SisFall dataset root
        subject_ids: List of subject IDs to load
        verbose: Print loading progress
    
    Returns:
        X_trials, y_trials, subjects, activity_codes
    """
    X_trials = []
    y_trials = []
    subjects = []
    activity_codes = []
    
    for subject_id in subject_ids:
        subject_path = os.path.join(data_root, subject_id)
        
        if not os.path.isdir(subject_path):
            if verbose:
                print(f"Warning: Subject folder not found: {subject_id}")
            continue
        
        files = sorted([f for f in os.listdir(subject_path) if f.endswith(".txt")])
        
        for file in files:
            file_path = os.path.join(subject_path, file)
            data = load_single_sisfall_file(file_path)
            
            if data is None:
                continue
            
            label = 1 if file.startswith("F") else 0
            activity_code = file.split('_')[0]
            
            X_trials.append(data)
            y_trials.append(label)
            subjects.append(subject_id)
            activity_codes.append(activity_code)
    
    if verbose:
        print(f"Loaded {len(X_trials)} trials from {len(subject_ids)} subjects")
    
    return X_trials, y_trials, subjects, activity_codes


def get_subject_list(data_root, subject_type='all'):
    """
    Get list of available subjects in dataset.
    
    Args:
        data_root: Path to SisFall dataset root
        subject_type: 'all', 'young' (SA), or 'elderly' (SE)
    
    Returns:
        List of subject IDs
    """
    subjects = sorted([f for f in os.listdir(data_root) 
                       if os.path.isdir(os.path.join(data_root, f))])
    
    if subject_type == 'young':
        return [s for s in subjects if s.startswith('SA')]
    elif subject_type == 'elderly':
        return [s for s in subjects if s.startswith('SE')]
    else:
        return subjects


def get_dataset_info(data_root):
    """
    Get information about the dataset.
    
    Args:
        data_root: Path to SisFall dataset root
    
    Returns:
        Dictionary with dataset information
    """
    subjects = get_subject_list(data_root)
    young = [s for s in subjects if s.startswith('SA')]
    elderly = [s for s in subjects if s.startswith('SE')]
    
    # Count activities
    adl_count = 0
    fall_count = 0
    
    for subject in subjects:
        subject_path = os.path.join(data_root, subject)
        files = [f for f in os.listdir(subject_path) if f.endswith(".txt")]
        
        for f in files:
            if f.startswith('F'):
                fall_count += 1
            else:
                adl_count += 1
    
    return {
        'n_subjects': len(subjects),
        'n_young': len(young),
        'n_elderly': len(elderly),
        'young_subjects': young,
        'elderly_subjects': elderly,
        'n_adl_trials': adl_count,
        'n_fall_trials': fall_count,
        'total_trials': adl_count + fall_count
    }

