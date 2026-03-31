"""Utility functions for loading preprocessed data."""

from pathlib import Path
import pandas as pd
import numpy as np


def load_features(features_path: Path) -> pd.DataFrame:
    """
    Load features from various formats, trying different extensions.
    """
    base_path = features_path.with_suffix('')
    
    # Try Parquet first (preferred)
    parquet_path = base_path.with_suffix('.parquet')
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"  Warning: Could not read Parquet: {e}")

    # Try CSV as a fallback
    csv_path = base_path.with_suffix('.csv')
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Warning: Could not read CSV: {e}")
            
    # Try pickle format as a last resort
    pickle_path = base_path.with_suffix('.pkl')
    if pickle_path.exists():
        try:
            return pd.read_pickle(pickle_path)
        except Exception as e:
            print(f"  Warning: Could not read pickle: {e}")
    
    raise FileNotFoundError(f"Could not find or read features file: {features_path} or alternatives (.parquet, .csv, .pkl)")


def load_windows(windows_path: Path) -> pd.DataFrame:
    """
    Load windows from various formats.
    Handles split format (parquet metadata + signals.npz).
    """
    windows_path = Path(windows_path)
    
    # Try pickle format first
    if windows_path.suffix.lower() == ".pkl" and windows_path.exists():
        try:
            return pd.read_pickle(windows_path)
        except Exception as e:
            print(f"  Warning: Could not read pickle: {e}")
    
    # Try split format (parquet + signals.npz)
    parquet_path = windows_path.with_suffix(".parquet")
    signals_path = windows_path.with_stem(windows_path.stem + "_signals").with_suffix(".npz")
    
    if parquet_path.exists() and signals_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            signals_data = np.load(signals_path, allow_pickle=True)
            signals = signals_data['signals']
            df['signal'] = list(signals)
            return df
        except Exception as e:
            print(f"  Warning: Could not read split format: {e}")
    elif parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"  Warning: Could not read parquet: {e}")
    
    raise FileNotFoundError(f"Could not find or read windows file: {windows_path} or alternatives")
