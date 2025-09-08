# utils_preprocess.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, List

SENSOR_COLUMNS = ["ph", "temperature", "methane", "co2", "pressure"]

def read_csv_time_series(path: str, timestamp_col="timestamp") -> pd.DataFrame:
    """Read CSV with timestamp and sensor columns. Timestamp should be parseable by pandas."""
    df = pd.read_csv(path)
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)
    return df

def make_windows(df: pd.DataFrame, window_size: int, step: int = 1,
                 sensor_cols: List[str] = SENSOR_COLUMNS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert dataframe to sliding windows.
    Returns: X_windows (num_windows, window_size, n_sensors),
             X_stats (num_windows, n_stat_features),
             y (num_windows,)  where y is the label aligned to the window end.
    Assumes df has a 'label' column for supervised training (categorical or numeric).
    """
    data = df[sensor_cols].values
    labels = df['label'].values if 'label' in df.columns else None
    n = len(df)
    windows = []
    stats = []
    ys = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        w = data[start:end]    # shape (window_size, n_sensors)
        windows.append(w)
        # statistical features per sensor: mean, std, min, max, median
        stat_feats = []
        stat_feats.extend(np.nanmean(w, axis=0).tolist())
        stat_feats.extend(np.nanstd(w, axis=0).tolist())
        stat_feats.extend(np.nanmin(w, axis=0).tolist())
        stat_feats.extend(np.nanmax(w, axis=0).tolist())
        stat_feats.extend(np.nanmedian(w, axis=0).tolist())
        stats.append(stat_feats)
        if labels is not None:
            ys.append(labels[end - 1])  # label assigned to window end
    Xw = np.stack(windows).astype(np.float32)
    Xs = np.stack(stats).astype(np.float32)
    y_arr = np.array(ys) if labels is not None else None
    return Xw, Xs, y_arr

def fit_scalers(X_windows: np.ndarray, X_stats: np.ndarray, scaler_dir: str):
    """
    Fit scalers for window-level normalization (per sensor) and stat features.
    Saves scalers to disk with joblib in scaler_dir.
    """
    # Flatten windows across time to fit sensor-wise scaler
    n_windows, wsize, n_sensors = X_windows.shape
    flat = X_windows.reshape(-1, n_sensors)  # (n_windows * wsize, n_sensors)
    sensor_scaler = StandardScaler().fit(flat)
    stat_scaler = StandardScaler().fit(X_stats)
    joblib.dump(sensor_scaler, f"{scaler_dir}/sensor_scaler.pkl")
    joblib.dump(stat_scaler, f"{scaler_dir}/stat_scaler.pkl")
    return sensor_scaler, stat_scaler

def transform_windows(X_windows: np.ndarray, X_stats: np.ndarray, scaler_dir: str):
    sensor_scaler = joblib.load(f"{scaler_dir}/sensor_scaler.pkl")
    stat_scaler = joblib.load(f"{scaler_dir}/stat_scaler.pkl")
    n_windows, wsize, n_sensors = X_windows.shape
    flat = X_windows.reshape(-1, n_sensors)
    flat_scaled = sensor_scaler.transform(flat)
    Xw_scaled = flat_scaled.reshape(n_windows, wsize, n_sensors)
    Xs_scaled = stat_scaler.transform(X_stats)
    return Xw_scaled.astype(np.float32), Xs_scaled.astype(np.float32)

def encode_labels(y_raw):
    """If y_raw is categorical strings, map to ints and return mapping."""
    if y_raw is None:
        return None, None
    if y_raw.dtype.kind in ('U', 'S', 'O'):
        labels_unique = np.unique(y_raw)
        mapping = {lab: i for i, lab in enumerate(labels_unique)}
        y = np.vectorize(lambda v: mapping[v])(y_raw)
        return y.astype(int), mapping
    else:
        return y_raw.astype(int), None
