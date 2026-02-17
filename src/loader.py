# src/loader.py
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


def load_sisfall(data_root):
    """
    Load all SisFall trials from a dataset directory.
    Returns:
        X_trials: list of np.array of shape (num_samples, 9)
        y_trials: list of labels (0 = ADL, 1 = Fall)
        subjects: list of subject IDs
    """
    X_trials = []
    y_trials = []
    subjects = []

    for subject_folder in os.listdir(data_root):
        subject_path = os.path.join(data_root, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for file in os.listdir(subject_path):
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(subject_path, file)
            data = load_single_sisfall_file(file_path)
            if data is None:
                continue  # skip corrupted/empty files

            # Determine label
            label = 1 if file.startswith("F") else 0

            X_trials.append(data)
            y_trials.append(label)
            subjects.append(subject_folder)

    print(f"Total trials: {len(X_trials)}")
    print(f"Fall trials: {sum(y_trials)}")
    print(f"ADL trials: {len(y_trials) - sum(y_trials)}")
    return X_trials, y_trials, subjects
