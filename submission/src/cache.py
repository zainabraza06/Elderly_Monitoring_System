# src/cache.py
"""
Disk-based caching for expensive feature extraction.
Cache key is derived from preprocessing parameters + dataset fingerprint,
so the cache is automatically invalidated when anything changes.
"""
import os
import hashlib
import json
import numpy as np
from pathlib import Path


# Default cache directory (relative to submission root)
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".feature_cache"


# =============================================================================
# CACHE KEY GENERATION
# =============================================================================
def _hash_dataset(data_root: str) -> str:
    """
    Create a fingerprint of the dataset by hashing the sorted list of
    (filename, file-size) pairs.  Fast — no file content is read.
    """
    entries = []
    data_root = Path(data_root)
    for filepath in sorted(data_root.rglob("*.txt")):
        entries.append(f"{filepath.relative_to(data_root)}:{filepath.stat().st_size}")
    raw = "\n".join(entries).encode()
    return hashlib.md5(raw).hexdigest()[:12]


def make_cache_key(data_root: str, params: dict) -> str:
    """
    Build a unique cache key string from dataset fingerprint + parameters.

    Args:
        data_root: Path to SisFall dataset root directory.
        params:    Dict of preprocessing / feature parameters, e.g.:
                   {window_size, overlap, target_hz, include_gait, dataset_type}

    Returns:
        A short hex string usable as a filename stem.
    """
    dataset_hash = _hash_dataset(data_root)
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    return f"{dataset_hash}_{params_hash}"


# =============================================================================
# SAVE / LOAD HELPERS
# =============================================================================
def save_cache(cache_dir: Path, key: str, X, y, metadata: dict) -> Path:
    """
    Persist features to disk using numpy's compressed format.

    Arrays are stored in a single `.npz` file; metadata (subjects list,
    activity list, feature names, etc.) lives alongside it as JSON.

    Returns the path to the saved `.npz` file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.meta.json"

    # Save arrays
    np.savez_compressed(npz_path, X=X, y=y)

    # Serialise metadata — convert numpy types to plain Python first
    serialisable_meta = {}
    for k, v in metadata.items():
        if isinstance(v, np.ndarray):
            serialisable_meta[k] = v.tolist()
        elif isinstance(v, list) and v and isinstance(v[0], np.integer):
            serialisable_meta[k] = [int(i) for i in v]
        else:
            serialisable_meta[k] = v

    with open(meta_path, "w") as f:
        json.dump(serialisable_meta, f)

    return npz_path


def load_cache(cache_dir: Path, key: str):
    """
    Load features from disk.

    Returns:
        (X, y, metadata) if the cache entry exists, else None.
    """
    cache_dir = Path(cache_dir)
    npz_path  = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.meta.json"

    if not npz_path.exists() or not meta_path.exists():
        return None

    data = np.load(npz_path, allow_pickle=False)
    X = data["X"]
    y = data["y"]

    with open(meta_path) as f:
        metadata = json.load(f)

    return X, y, metadata


def cache_exists(cache_dir: Path, key: str) -> bool:
    """Return True if both cache files exist for the given key."""
    cache_dir = Path(cache_dir)
    return (cache_dir / f"{key}.npz").exists() and \
           (cache_dir / f"{key}.meta.json").exists()


def list_cache(cache_dir: Path) -> list:
    """List all cache entries (keys) in the cache directory."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    return sorted({p.stem.replace(".meta", "") for p in cache_dir.glob("*.npz")})


def clear_cache(cache_dir: Path, key: str = None):
    """
    Remove one or all cache entries.

    Args:
        cache_dir: Cache directory.
        key:       If given, only that entry is removed; otherwise all are removed.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return

    if key:
        for suffix in (".npz", ".meta.json"):
            p = cache_dir / f"{key}{suffix}"
            if p.exists():
                p.unlink()
        print(f"Removed cache entry: {key}")
    else:
        removed = 0
        for p in cache_dir.iterdir():
            if p.suffix in (".npz", ".json"):
                p.unlink()
                removed += 1
        print(f"Cleared {removed} cache files from {cache_dir}")


def cache_info(cache_dir: Path) -> dict:
    """Return size/count information about the cache directory."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {"entries": 0, "total_size_mb": 0.0, "path": str(cache_dir)}

    npz_files = list(cache_dir.glob("*.npz"))
    total_bytes = sum(p.stat().st_size for p in cache_dir.iterdir())

    return {
        "entries": len(npz_files),
        "total_size_mb": round(total_bytes / 1_048_576, 2),
        "path": str(cache_dir),
        "keys": [p.stem for p in npz_files],
    }
