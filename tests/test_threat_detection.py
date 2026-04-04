"""
tests/test_threat_detection.py
-------------------------------
Unit tests for the AI Threat Detection system.

Run with:
  python tests/test_threat_detection.py
  or: python -m pytest tests/ -v
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.data_generator import generate_dataset
from core.preprocessor import TrafficPreprocessor, prepare_data
from models.models import (
    RandomForestThreatDetector,
    IsolationForestDetector,
    GradientBoostingThreatDetector,
    ThreatEnsemble,
)


class TestDataGenerator(unittest.TestCase):

    def setUp(self):
        self.df = generate_dataset(n_samples=300, anomaly_ratio=0.2)

    def test_dataset_shape(self):
        self.assertEqual(len(self.df), 300)

    def test_label_column_exists(self):
        self.assertIn("label", self.df.columns)

    def test_label_values(self):
        valid = {"normal", "dos", "port_scan", "data_exfiltration", "brute_force"}
        self.assertTrue(set(self.df["label"].unique()).issubset(valid))

    def test_normal_ratio(self):
        normal_ratio = (self.df["label"] == "normal").mean()
        self.assertAlmostEqual(normal_ratio, 0.80, delta=0.05)

    def test_required_columns(self):
        required = ["duration", "packet_size", "bytes_sent", "bytes_recv",
                    "conn_rate", "failed_logins", "unique_ports", "protocol"]
        for col in required:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_no_null_values(self):
        self.assertEqual(self.df.isnull().sum().sum(), 0)

    def test_conn_rate_positive(self):
        self.assertTrue((self.df["conn_rate"] >= 0).all())

    def test_dos_has_high_conn_rate(self):
        dos = self.df[self.df["label"] == "dos"]
        if len(dos) > 0:
            self.assertGreater(dos["conn_rate"].mean(), 50)

    def test_port_scan_has_many_ports(self):
        ps = self.df[self.df["label"] == "port_scan"]
        if len(ps) > 0:
            self.assertGreater(ps["unique_ports"].mean(), 50)

    def test_brute_force_has_failed_logins(self):
        bf = self.df[self.df["label"] == "brute_force"]
        if len(bf) > 0:
            self.assertGreater(bf["failed_logins"].mean(), 5)


class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.df = generate_dataset(n_samples=300, anomaly_ratio=0.2)
        self.preprocessor = TrafficPreprocessor()

    def test_fit_transform_shapes(self):
        X, y = self.preprocessor.fit_transform(self.df)
        self.assertEqual(X.shape[0], len(self.df))
        self.assertEqual(y.shape[0], len(self.df))
        self.assertGreater(X.shape[1], 10)

    def test_labels_encoded(self):
        X, y = self.preprocessor.fit_transform(self.df)
        self.assertTrue(np.issubdtype(y.dtype, np.integer))

    def test_features_scaled(self):
        X, y = self.preprocessor.fit_transform(self.df)
        # After StandardScaler, mean should be near 0
        self.assertAlmostEqual(X.mean(), 0.0, delta=0.5)

    def test_inverse_label(self):
        X, y = self.preprocessor.fit_transform(self.df)
        labels = self.preprocessor.inverse_label(y[:5])
        for l in labels:
            self.assertIn(l, ["normal","dos","port_scan","data_exfiltration","brute_force"])

    def test_transform_after_fit(self):
        X, y = self.preprocessor.fit_transform(self.df)
        X2 = self.preprocessor.transform(self.df)
        np.testing.assert_array_almost_equal(X, X2)

    def test_prepare_data_split(self):
        X_train, X_test, y_train, y_test, prep = prepare_data(self.df, test_size=0.2)
        total = len(X_train) + len(X_test)
        self.assertEqual(total, len(self.df))
        self.assertAlmostEqual(len(X_test) / total, 0.2, delta=0.02)


class TestModels(unittest.TestCase):

    def setUp(self):
        df = generate_dataset(n_samples=500, anomaly_ratio=0.2)
        self.X_train, self.X_test, self.y_train, self.y_test, self.prep = prepare_data(df)
        self.label_names = self.prep.get_label_names()

    def test_random_forest_trains(self):
        rf = RandomForestThreatDetector(n_estimators=10)
        rf.train(self.X_train, self.y_train)
        preds = rf.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_random_forest_accuracy(self):
        rf = RandomForestThreatDetector(n_estimators=20)
        rf.train(self.X_train, self.y_train)
        results = rf.evaluate(self.X_test, self.y_test, self.label_names)
        self.assertGreater(results["accuracy"], 0.7)

    def test_isolation_forest_trains(self):
        iso = IsolationForestDetector()
        iso.train(self.X_train)
        preds = iso.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(set(preds).issubset({1, -1}))

    def test_isolation_forest_anomaly_scores(self):
        iso = IsolationForestDetector()
        iso.train(self.X_train)
        scores = iso.anomaly_scores(self.X_test)
        self.assertEqual(len(scores), len(self.X_test))
        self.assertTrue((scores >= 0).all())

    def test_gradient_boosting_trains(self):
        gb = GradientBoostingThreatDetector()
        gb.train(self.X_train, self.y_train)
        preds = gb.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_ensemble_predict(self):
        rf  = RandomForestThreatDetector(n_estimators=10)
        gb  = GradientBoostingThreatDetector()
        iso = IsolationForestDetector()
        rf.train(self.X_train, self.y_train)
        gb.train(self.X_train, self.y_train)
        iso.train(self.X_train)

        ensemble = ThreatEnsemble(rf, gb, iso)
        preds, conf, flags = ensemble.predict_with_confidence(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertEqual(len(conf),  len(self.X_test))
        self.assertTrue((conf >= 0).all() and (conf <= 1).all())

    def test_feature_importance_length(self):
        rf = RandomForestThreatDetector(n_estimators=10)
        rf.train(self.X_train, self.y_train)
        imp = rf.feature_importances()
        self.assertEqual(len(imp), self.X_train.shape[1])


if __name__ == "__main__":
    print("Running AI Threat Detection tests...\n")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDataGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestModels))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
