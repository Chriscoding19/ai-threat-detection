"""
preprocessor.py
---------------
Feature engineering and data preprocessing pipeline for network traffic data.

Steps:
  1. Clean and validate raw data
  2. Engineer derived features (ratios, flags, statistical)
  3. Encode categorical variables
  4. Scale numerical features
  5. Prepare train/test splits
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

NUMERIC_FEATURES = [
    "duration", "packet_size", "bytes_sent", "bytes_recv",
    "packet_count", "conn_rate", "failed_logins", "unique_ports",
    "ttl", "tcp_flags", "hour", "is_business_hours",
    # Engineered features (added below)
    "bytes_ratio", "bytes_per_packet", "conn_per_packet",
    "is_night", "is_high_port_scan", "is_high_login_fail",
    "is_large_upload", "is_flood",
]

CATEGORICAL_FEATURES = ["protocol"]


class TrafficPreprocessor:
    """
    Fit-transform pipeline for network traffic feature engineering.
    Can be saved and loaded for use at inference time.
    """

    def __init__(self):
        self.scaler        = StandardScaler()
        self.proto_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self._fitted       = False

    def fit_transform(self, df: pd.DataFrame):
        """Engineer features, fit scalers, return (X, y, feature_names)."""
        df = self._engineer_features(df.copy())
        df = self._encode_categoricals(df, fit=True)

        X = df[self._get_feature_cols(df)].values.astype(np.float32)
        y_raw = df["label"].values

        self.label_encoder.fit(y_raw)
        y = self.label_encoder.transform(y_raw)

        X = self.scaler.fit_transform(X)
        self.feature_names = self._get_feature_cols(df)
        self._fitted = True

        print(f"  Features engineered: {len(self.feature_names)}")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
        return X, y

    def transform(self, df: pd.DataFrame):
        """Transform new data using fitted scalers."""
        assert self._fitted, "Call fit_transform first."
        df = self._engineer_features(df.copy())
        df = self._encode_categoricals(df, fit=False)
        X  = df[self.feature_names].values.astype(np.float32)
        return self.scaler.transform(X)

    def inverse_label(self, y: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y)

    def get_label_names(self) -> list:
        return list(self.label_encoder.classes_)

    def save(self, path: str = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Preprocessor saved: {path}")

    @staticmethod
    def load(path: str = "models/preprocessor.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive additional features from raw columns."""
        eps = 1e-9  # Prevent division by zero

        # Upload/download ratio — exfiltration sends more than it receives
        df["bytes_ratio"] = df["bytes_sent"] / (df["bytes_recv"] + eps)

        # Average bytes per packet — large = bulk transfer, tiny = flood/scan
        df["bytes_per_packet"] = (df["bytes_sent"] + df["bytes_recv"]) / (df["packet_count"] + eps)

        # Connection rate per packet
        df["conn_per_packet"] = df["conn_rate"] / (df["packet_count"] + eps)

        # Binary threat indicators (domain knowledge rules)
        df["is_night"]           = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        df["is_high_port_scan"]  = (df["unique_ports"] > 100).astype(int)
        df["is_high_login_fail"] = (df["failed_logins"] > 10).astype(int)
        df["is_large_upload"]    = (df["bytes_sent"] > 100000).astype(int)
        df["is_flood"]           = (df["conn_rate"] > 100).astype(int)

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Label-encode the protocol column."""
        if fit:
            df["protocol_enc"] = self.proto_encoder.fit_transform(df["protocol"])
        else:
            # Handle unseen protocols gracefully
            known = list(self.proto_encoder.classes_)
            df["protocol"] = df["protocol"].apply(
                lambda x: x if x in known else known[0]
            )
            df["protocol_enc"] = self.proto_encoder.transform(df["protocol"])
        return df

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """Return the list of numeric feature column names to use."""
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        if "protocol_enc" in df.columns:
            cols.append("protocol_enc")
        return cols


def prepare_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Full pipeline: preprocess → split → return train/test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor
    """
    preprocessor = TrafficPreprocessor()
    X, y = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test, preprocessor
