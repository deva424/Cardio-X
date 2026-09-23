from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .constants import SIGNAL_LENGTH


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a headerless MIT-BIH CSV: 187 signal values and a final class column."""
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != SIGNAL_LENGTH + 1:
        raise ValueError(f"Expected {SIGNAL_LENGTH} signal columns plus one label column; got {frame.shape[1]} columns.")
    return frame.iloc[:, :SIGNAL_LENGTH].to_numpy(dtype=np.float32), frame.iloc[:, -1].to_numpy(dtype=int)


def fit_transform_training_data(features: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(features).astype(np.float32), scaler


def transform_features(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(features).astype(np.float32).reshape(-1, SIGNAL_LENGTH, 1)


def prepare_uploaded_features(features: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    """Prepare uploaded signals, using the training scaler when it is available.

    Older exported models may not have a matching scaler artifact. In that case
    this follows the original notebook and scales the uploaded batch itself.
    """
    if scaler is None:
        scaler = StandardScaler().fit(features)
    return transform_features(features, scaler)


def save_scaler(scaler: StandardScaler, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def extract_uploaded_signals(frame: pd.DataFrame) -> np.ndarray:
    """Extract the first 187 numeric values per row from an uploaded CSV."""
    numeric = frame.drop(columns=["class_label"], errors="ignore").select_dtypes(include=[np.number])
    if numeric.shape[1] < SIGNAL_LENGTH:
        raise ValueError(f"The CSV needs at least {SIGNAL_LENGTH} numeric signal columns.")
    return numeric.iloc[:, :SIGNAL_LENGTH].to_numpy(dtype=np.float32)
