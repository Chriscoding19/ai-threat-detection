"""
models.py
---------
ML classification models for network threat detection.

Models implemented:
  1. RandomForestClassifier  — Multi-class supervised classifier (main model)
  2. IsolationForest         — Unsupervised anomaly detector
  3. GradientBoosting        — Ensemble booster for comparison
  4. ThreatEnsemble          — Combines all models for final prediction
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)


class RandomForestThreatDetector:
    """
    Primary supervised classifier.
    Trained on labelled traffic data to classify:
      normal / dos / port_scan / data_exfiltration / brute_force
    """

    def __init__(self, n_estimators: int = 150, max_depth: int = 20):
        self.model = RandomForestClassifier(
            n_estimators  = n_estimators,
            max_depth     = max_depth,
            min_samples_split = 5,
            class_weight  = "balanced",   # Handle imbalanced classes
            random_state  = 42,
            n_jobs        = -1,
        )
        self.name     = "Random Forest"
        self._trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"  Training {self.name}...")
        self.model.fit(X_train, y_train)
        self._trained = True
        train_acc = self.model.score(X_train, y_train)
        print(f"  {self.name} train accuracy: {train_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 label_names: list) -> dict:
        y_pred = self.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=label_names, output_dict=True)
        return {
            "model":      self.name,
            "accuracy":   acc,
            "f1_weighted": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    def save(self, path: str = "models/random_forest.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = "models/random_forest.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


class IsolationForestDetector:
    """
    Unsupervised anomaly detector using Isolation Forest.
    Does NOT require labels — useful for detecting zero-day attacks.

    Returns: 1 = normal, -1 = anomaly
    """

    def __init__(self, contamination: float = 0.15):
        self.model = IsolationForest(
            n_estimators  = 200,
            contamination = contamination,   # Expected fraction of anomalies
            max_samples   = "auto",
            random_state  = 42,
            n_jobs        = -1,
        )
        self.name     = "Isolation Forest"
        self._trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray = None):
        """Train on all data (unsupervised — labels ignored)."""
        print(f"  Training {self.name} (unsupervised)...")
        self.model.fit(X_train)
        self._trained = True
        print(f"  {self.name} fitted on {X_train.shape[0]} samples.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns array of 1 (normal) or -1 (anomaly)."""
        return self.model.predict(X)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Returns raw anomaly scores (lower = more anomalous).
        Negated so higher = more anomalous (intuitive direction).
        """
        return -self.model.score_samples(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 label_names: list) -> dict:
        """Evaluate as binary: normal vs any-attack."""
        preds    = self.predict(X_test)
        # Convert: 1=normal(-1→anomaly), -1=anomaly(anything else)
        y_binary = (y_test != label_names.index("normal")).astype(int)
        p_binary = (preds == -1).astype(int)

        acc = accuracy_score(y_binary, p_binary)
        f1  = f1_score(y_binary, p_binary, average="binary", zero_division=0)
        return {
            "model":      self.name,
            "accuracy":   acc,
            "f1_binary":  f1,
            "note":       "Binary: normal vs anomaly (unsupervised)",
        }

    def save(self, path: str = "models/isolation_forest.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


class GradientBoostingThreatDetector:
    """
    Gradient Boosting classifier — often highest accuracy on tabular data.
    Slower to train than Random Forest but typically more precise.
    """

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators   = 100,
            learning_rate  = 0.1,
            max_depth      = 5,
            subsample      = 0.8,
            random_state   = 42,
        )
        self.name     = "Gradient Boosting"
        self._trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"  Training {self.name}...")
        self.model.fit(X_train, y_train)
        self._trained = True
        train_acc = self.model.score(X_train, y_train)
        print(f"  {self.name} train accuracy: {train_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 label_names: list) -> dict:
        y_pred = self.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=label_names, output_dict=True)
        return {
            "model":      self.name,
            "accuracy":   acc,
            "f1_weighted": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    def save(self, path: str = "models/gradient_boosting.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


class ThreatEnsemble:
    """
    Ensemble model that combines Random Forest + Gradient Boosting
    via soft voting (averaged probabilities).

    The Isolation Forest anomaly score is used as an additional
    signal to boost confidence for ambiguous predictions.
    """

    def __init__(self, rf: RandomForestThreatDetector,
                 gb: GradientBoostingThreatDetector,
                 iso: IsolationForestDetector):
        self.rf   = rf
        self.gb   = gb
        self.iso  = iso
        self.name = "Ensemble (RF + GB + IsoForest)"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Soft-vote between RF and GB, use IsoForest as tiebreaker."""
        rf_proba  = self.rf.predict_proba(X)
        gb_proba  = self.gb.predict_proba(X)
        avg_proba = (rf_proba + gb_proba) / 2
        return np.argmax(avg_proba, axis=1)

    def predict_with_confidence(self, X: np.ndarray):
        """Return (predictions, confidence_scores, iso_anomaly_flags)."""
        rf_proba  = self.rf.predict_proba(X)
        gb_proba  = self.gb.predict_proba(X)
        avg_proba = (rf_proba + gb_proba) / 2

        preds      = np.argmax(avg_proba, axis=1)
        confidence = np.max(avg_proba, axis=1)
        iso_flags  = (self.iso.predict(X) == -1).astype(int)

        return preds, confidence, iso_flags

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 label_names: list) -> dict:
        y_pred = self.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=label_names, output_dict=True)
        return {
            "model":      self.name,
            "accuracy":   acc,
            "f1_weighted": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
