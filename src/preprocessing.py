"""
preprocessing.py
----------------
Handles all data cleaning, feature engineering, and encoding steps.
Keeps training and inference pipelines identical.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# ─── Columns ────────────────────────────────────────────────────────────────

# Columns to drop before modelling (metadata / target)
META_COLS = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "label"]

# Categorical columns that need encoding
CAT_COLS = ["protocol_type"]

# Numeric feature columns (derived after encoding)
NUMERIC_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "count",
    "srv_count", "serror_rate", "dst_host_count", "dst_host_srv_count",
    "protocol_tcp", "protocol_udp", "protocol_icmp",   # one-hot encoded
]

# Label mapping: normal=0, attacks=1 for binary model
LABEL_BINARY = {
    "normal":      0,
    "ddos":        1,
    "brute_force": 1,
    "port_scan":   1,
    "data_exfil":  1,
}

# Multi-class label mapping
LABEL_MULTI = {
    "normal":      0,
    "ddos":        1,
    "brute_force": 2,
    "port_scan":   3,
    "data_exfil":  4,
}

LABEL_NAMES = {v: k for k, v in LABEL_MULTI.items()}


# ─── Core preprocessing ──────────────────────────────────────────────────────

def load_raw(filepath: str) -> pd.DataFrame:
    """Load CSV log file; handle common format issues."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
    - One-hot encode protocol_type
    - Log-scale heavy-tailed byte columns
    - Clamp outliers at 99th percentile
    """
    df = df.copy()

    # One-hot encode protocol (drops original col)
    ohe = pd.get_dummies(df["protocol_type"], prefix="protocol")
    for col in ["protocol_tcp", "protocol_udp", "protocol_icmp"]:
        if col not in ohe.columns:
            ohe[col] = 0
    df = pd.concat([df.drop("protocol_type", axis=1), ohe[["protocol_tcp", "protocol_udp", "protocol_icmp"]]], axis=1)

    # Log-scale byte columns (avoid log(0))
    for col in ["src_bytes", "dst_bytes"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Clip extreme values at 99th percentile (suppress noise)
    for col in ["count", "srv_count", "dst_host_count", "dst_host_srv_count"]:
        if col in df.columns:
            cap = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=cap)

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the model input columns (drops metadata/label)."""
    drop = [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=drop)

    # Ensure all expected features exist
    for col in NUMERIC_FEATURES:
        if col not in X.columns:
            X[col] = 0

    return X[NUMERIC_FEATURES]


def get_labels(df: pd.DataFrame, multiclass: bool = True) -> np.ndarray:
    """Map string labels to integers."""
    mapping = LABEL_MULTI if multiclass else LABEL_BINARY
    return df["label"].map(mapping).fillna(0).astype(int).values


def preprocess_pipeline(
    filepath: str,
    test_size: float = 0.20,
    scaler_path: str = "models/scaler.pkl",
    multiclass: bool = True,
) -> dict:
    """
    Full preprocessing pipeline for training.
    Returns dict with X_train, X_test, y_train, y_test, feature_names.
    Also saves the fitted StandardScaler to disk.
    """
    df = load_raw(filepath)
    df = engineer_features(df)

    X = get_feature_matrix(df)
    y = get_labels(df, multiclass=multiclass)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Persist scaler for inference
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved → {scaler_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_names": list(X.columns),
        "scaler": scaler,
        "df_raw": df,
    }


def preprocess_for_inference(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler.pkl",
) -> np.ndarray:
    """
    Transform a new dataframe for inference using the saved scaler.
    Mirrors the same steps as training preprocessing.
    """
    df = engineer_features(df)
    X = get_feature_matrix(df)
    scaler = joblib.load(scaler_path)
    return scaler.transform(X)