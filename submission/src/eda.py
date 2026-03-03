# src/eda.py
"""
Exploratory Data Analysis for SisFall dataset.
"""
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def basic_eda(X_trials, y_trials, subjects=None, activity_codes=None):
    """
    Perform basic exploratory data analysis on the dataset.
    
    Args:
        X_trials: List of trial data arrays
        y_trials: List of labels (0=ADL, 1=Fall)
        subjects: Optional list of subject IDs
        activity_codes: Optional list of activity codes
    """
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Basic counts
    n_trials = len(X_trials)
    n_falls = sum(y_trials)
    n_adl = n_trials - n_falls
    
    print(f"\nDataset Overview:")
    print(f"  Total trials: {n_trials}")
    print(f"  Fall trials: {n_falls} ({100*n_falls/n_trials:.1f}%)")
    print(f"  ADL trials: {n_adl} ({100*n_adl/n_trials:.1f}%)")
    print(f"  Class ratio (ADL:Fall): {n_adl/n_falls:.2f}:1")
    
    # Trial length statistics
    lengths = [x.shape[0] for x in X_trials]
    durations = [l / 200 for l in lengths]  # 200 Hz sampling
    
    print(f"\nTrial Length Statistics:")
    print(f"  Min samples: {np.min(lengths)} ({np.min(durations):.1f}s)")
    print(f"  Max samples: {np.max(lengths)} ({np.max(durations):.1f}s)")
    print(f"  Mean samples: {np.mean(lengths):.0f} ({np.mean(durations):.1f}s)")
    print(f"  Std samples: {np.std(lengths):.0f}")
    
    # Channel information
    n_channels = X_trials[0].shape[1]
    print(f"\nSensor Information:")
    print(f"  Number of channels: {n_channels}")
    print(f"  Channels 0-2: ADXL345 Accelerometer (X, Y, Z)")
    print(f"  Channels 3-5: ITG3200 Gyroscope (X, Y, Z)")
    print(f"  Channels 6-8: MMA8451Q Accelerometer (X, Y, Z)")
    
    # Subject analysis
    if subjects is not None:
        unique_subjects = set(subjects)
        young = [s for s in unique_subjects if s.startswith('SA')]
        elderly = [s for s in unique_subjects if s.startswith('SE')]
        
        print(f"\nSubject Analysis:")
        print(f"  Total subjects: {len(unique_subjects)}")
        print(f"  Young (SA): {len(young)}")
        print(f"  Elderly (SE): {len(elderly)}")
    
    # Activity analysis
    if activity_codes is not None:
        unique_activities = set(activity_codes)
        adl_activities = [a for a in unique_activities if a.startswith('D')]
        fall_activities = [a for a in unique_activities if a.startswith('F')]
        
        print(f"\nActivity Analysis:")
        print(f"  Total activity types: {len(unique_activities)}")
        print(f"  ADL types: {len(adl_activities)}")
        print(f"  Fall types: {len(fall_activities)}")
    
    # Signal statistics (sample)
    print(f"\nSignal Statistics (first 100 trials):")
    sample_data = np.vstack([x for x in X_trials[:100]])
    for i, name in enumerate(['ADXL_X', 'ADXL_Y', 'ADXL_Z', 
                              'GYRO_X', 'GYRO_Y', 'GYRO_Z',
                              'MMA_X', 'MMA_Y', 'MMA_Z']):
        print(f"  {name}: mean={np.mean(sample_data[:,i]):.2f}, "
              f"std={np.std(sample_data[:,i]):.2f}, "
              f"range=[{np.min(sample_data[:,i]):.0f}, {np.max(sample_data[:,i]):.0f}]")
    
    print("=" * 50)
    return {
        'n_trials': n_trials,
        'n_falls': n_falls,
        'n_adl': n_adl,
        'lengths': lengths,
        'n_channels': n_channels
    }


def plot_trial_distribution(X_trials, y_trials, save_path=None):
    """Plot distribution of trial lengths by class."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return
    
    lengths = [x.shape[0] / 200 for x in X_trials]  # Convert to seconds
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of lengths
    ax1 = axes[0]
    fall_lengths = [l for l, y in zip(lengths, y_trials) if y == 1]
    adl_lengths = [l for l, y in zip(lengths, y_trials) if y == 0]
    
    ax1.hist(adl_lengths, bins=30, alpha=0.7, label='ADL', color='blue')
    ax1.hist(fall_lengths, bins=30, alpha=0.7, label='Fall', color='red')
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Trial Duration Distribution')
    ax1.legend()
    
    # Class balance
    ax2 = axes[1]
    ax2.bar(['ADL', 'Fall'], [len(adl_lengths), len(fall_lengths)], 
            color=['blue', 'red'])
    ax2.set_ylabel('Count')
    ax2.set_title('Class Distribution')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_trial(trial, label, fs=200, save_path=None):
    """Plot a sample trial showing all sensor channels."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return
    
    time = np.arange(len(trial)) / fs
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # ADXL345 Accelerometer
    ax1 = axes[0]
    ax1.plot(time, trial[:, 0], label='X', alpha=0.8)
    ax1.plot(time, trial[:, 1], label='Y', alpha=0.8)
    ax1.plot(time, trial[:, 2], label='Z', alpha=0.8)
    ax1.set_ylabel('ADXL345 Acc')
    ax1.legend(loc='upper right')
    ax1.set_title(f"Sample Trial ({'Fall' if label == 1 else 'ADL'})")
    ax1.grid(True, alpha=0.3)
    
    # ITG3200 Gyroscope
    ax2 = axes[1]
    ax2.plot(time, trial[:, 3], label='X', alpha=0.8)
    ax2.plot(time, trial[:, 4], label='Y', alpha=0.8)
    ax2.plot(time, trial[:, 5], label='Z', alpha=0.8)
    ax2.set_ylabel('ITG3200 Gyro')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # MMA8451Q Accelerometer
    ax3 = axes[2]
    ax3.plot(time, trial[:, 6], label='X', alpha=0.8)
    ax3.plot(time, trial[:, 7], label='Y', alpha=0.8)
    ax3.plot(time, trial[:, 8], label='Z', alpha=0.8)
    ax3.set_ylabel('MMA8451Q Acc')
    ax3.set_xlabel('Time (s)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compare_fall_adl(X_trials, y_trials, n_samples=5, save_path=None):
    """Compare fall vs ADL signal characteristics."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return
    
    # Get sample of each class
    fall_trials = [x for x, y in zip(X_trials, y_trials) if y == 1][:n_samples]
    adl_trials = [x for x, y in zip(X_trials, y_trials) if y == 0][:n_samples]
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    for i, trial in enumerate(fall_trials):
        ax = axes[0, i]
        mag = np.sqrt(np.sum(trial[:, 0:3]**2, axis=1))
        ax.plot(mag, color='red', alpha=0.8)
        ax.set_title(f'Fall {i+1}')
        ax.set_ylabel('Acc Magnitude')
        
    for i, trial in enumerate(adl_trials):
        ax = axes[1, i]
        mag = np.sqrt(np.sum(trial[:, 0:3]**2, axis=1))
        ax.plot(mag, color='blue', alpha=0.8)
        ax.set_title(f'ADL {i+1}')
        ax.set_ylabel('Acc Magnitude')
        ax.set_xlabel('Sample')
    
    plt.suptitle('Fall vs ADL - Acceleration Magnitude Comparison')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

