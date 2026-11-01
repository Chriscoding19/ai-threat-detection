#!/usr/bin/env python3
"""
main.py — AI-Based Threat Detection System
==========================================

Trains multiple ML classification models on simulated network traffic data
and evaluates their ability to detect cyber attacks.

Usage:
  python main.py                         # Full pipeline (default 5000 samples)
  python main.py --samples 10000         # Larger dataset
  python main.py --no-plots              # Skip chart generation
  python main.py --predict sample.csv   # Predict on new data
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data.data_generator import generate_dataset
from core.preprocessor import prepare_data
from models.models import (
    RandomForestThreatDetector,
    IsolationForestDetector,
    GradientBoostingThreatDetector,
    ThreatEnsemble,
)
from reports.visualizer import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_anomaly_scores,
)
from reports.report_generator import generate_html_report


def parse_args():
    p = argparse.ArgumentParser(description="AI-Based Threat Detection System")
    p.add_argument("--samples",   type=int,   default=5000,  help="Number of traffic records (default 5000)")
    p.add_argument("--anomaly",   type=float, default=0.15,  help="Fraction of anomalies (default 0.15)")
    p.add_argument("--no-plots",  action="store_true",        help="Skip plot generation")
    p.add_argument("--no-report", action="store_true",        help="Skip HTML report")
    p.add_argument("--predict",   type=str,   default=None,  help="CSV file to run predictions on")
    return p.parse_args()


def print_header():
    print("\n" + "═"*65)
    print("   AI-BASED THREAT DETECTION SYSTEM")
    print("   Machine Learning for Network Security Analytics")
    print("═"*65)


def section(title):
    print(f"\n┌─────────────────────────────────────────────────────────────┐")
    print(f"│  {title}")
    print(f"└─────────────────────────────────────────────────────────────┘")


def main():
    args = parse_args()
    print_header()
    start = time.time()

    # ── Step 1: Generate Dataset ───────────────────────────────────
    section("STEP 1/6 — Generate Simulated Network Traffic Dataset")
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(n_samples=args.samples, anomaly_ratio=args.anomaly)
    df.to_csv("data/network_traffic.csv", index=False)
    print(f"  Saved: data/network_traffic.csv")

    dataset_stats = {
        "total":   len(df),
        "normal":  int((df["label"] == "normal").sum()),
        "attacks": int((df["label"] != "normal").sum()),
    }

    # ── Step 2: Preprocessing ──────────────────────────────────────
    section("STEP 2/6 — Feature Engineering & Preprocessing")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    label_names = preprocessor.get_label_names()
    preprocessor.save("models/preprocessor.pkl")

    # ── Step 3: Train Models ───────────────────────────────────────
    section("STEP 3/6 — Training ML Models")

    rf = RandomForestThreatDetector(n_estimators=150)
    rf.train(X_train, y_train)
    rf.save()

    iso = IsolationForestDetector(contamination=args.anomaly)
    iso.train(X_train)

    gb = GradientBoostingThreatDetector()
    gb.train(X_train, y_train)
    gb.save()

    ensemble = ThreatEnsemble(rf, gb, iso)

    # ── Step 4: Evaluate ───────────────────────────────────────────
    section("STEP 4/6 — Evaluating Models on Test Set")
    results = []

    rf_res  = rf.evaluate(X_test, y_test, label_names)
    gb_res  = gb.evaluate(X_test, y_test, label_names)
    iso_res = iso.evaluate(X_test, y_test, label_names)
    ens_res = ensemble.evaluate(X_test, y_test, label_names)

    results = [rf_res, gb_res, iso_res, ens_res]

    print(f"\n  {'MODEL':<35} {'ACCURACY':>10}  {'F1':>8}")
    print(f"  {'─'*35} {'─'*10}  {'─'*8}")
    for r in results:
        if "accuracy" in r:
            f1 = r.get("f1_weighted", r.get("f1_binary", 0))
            print(f"  {r['model']:<35} {r['accuracy']:>10.4f}  {f1:>8.4f}")

    best = max((r for r in results if "accuracy" in r), key=lambda r: r["accuracy"])
    print(f"\n  ✅  Best model: {best['model']} — Accuracy: {best['accuracy']*100:.2f}%")

    # Per-class breakdown
    if "classification_report" in best:
        print(f"\n  Per-class breakdown ({best['model']}):")
        print(f"  {'CLASS':<25} {'PRECISION':>10}  {'RECALL':>8}  {'F1':>8}")
        print(f"  {'─'*25} {'─'*10}  {'─'*8}  {'─'*8}")
        for cls in label_names:
            if cls in best["classification_report"]:
                m = best["classification_report"][cls]
                print(f"  {cls:<25} {m['precision']:>10.3f}  {m['recall']:>8.3f}  {m['f1-score']:>8.3f}")

    # ── Step 5: Visualisations ─────────────────────────────────────
    section("STEP 5/6 — Generating Visualisations")
    if not args.no_plots:
        iso_scores = iso.anomaly_scores(X_test)
        plot_class_distribution(df["label"])
        plot_confusion_matrix(rf_res["confusion_matrix"],  label_names, "Random Forest")
        plot_confusion_matrix(ens_res["confusion_matrix"], label_names, "Ensemble (RF + GB + IsoForest)")
        plot_feature_importance(rf.feature_importances(), preprocessor.feature_names)
        plot_model_comparison(results)
        plot_anomaly_scores(iso_scores, y_test, label_names)
        print("  All plots saved to reports/figures/")
    else:
        print("  Skipped (--no-plots)")

    # ── Step 6: Report ─────────────────────────────────────────────
    section("STEP 6/6 — Generating HTML Report")
    if not args.no_report:
        html_path = generate_html_report(results, label_names, dataset_stats)
        print(f"  Open in browser: {html_path}")
    else:
        print("  Skipped (--no-report)")

    # ── Live Prediction Demo ───────────────────────────────────────
    section("BONUS — Live Prediction Demo")
    _demo_predictions(ensemble, preprocessor, label_names)

    elapsed = time.time() - start
    print(f"\n{'═'*65}")
    print(f"  Complete! Total time: {elapsed:.1f}s")
    print(f"  Report : reports/threat_detection_report_*.html")
    print(f"  Models : models/")
    print(f"{'═'*65}\n")


def _demo_predictions(ensemble: ThreatEnsemble, preprocessor, label_names: list):
    """Demonstrate real-time prediction on crafted traffic samples."""
    test_cases = [
        {
            "name": "Normal web browsing",
            "protocol": "HTTPS", "duration": 2.1, "packet_size": 520.0,
            "bytes_sent": 4800.0, "bytes_recv": 14500.0, "packet_count": 18,
            "conn_rate": 4.5, "failed_logins": 0, "unique_ports": 2,
            "ttl": 64, "tcp_flags": 1, "hour": 14, "is_business_hours": 1,
        },
        {
            "name": "DoS flood attack",
            "protocol": "UDP", "duration": 0.05, "packet_size": 58.0,
            "bytes_sent": 68000.0, "bytes_recv": 800.0, "packet_count": 620,
            "conn_rate": 245.0, "failed_logins": 0, "unique_ports": 1,
            "ttl": 64, "tcp_flags": 2, "hour": 3, "is_business_hours": 0,
        },
        {
            "name": "Port scan reconnaissance",
            "protocol": "TCP", "duration": 0.03, "packet_size": 58.0,
            "bytes_sent": 280.0, "bytes_recv": 80.0, "packet_count": 2,
            "conn_rate": 65.0, "failed_logins": 0, "unique_ports": 650,
            "ttl": 128, "tcp_flags": 2, "hour": 2, "is_business_hours": 0,
        },
        {
            "name": "Data exfiltration (DNS tunneling)",
            "protocol": "DNS", "duration": 180.0, "packet_size": 1398.0,
            "bytes_sent": 720000.0, "bytes_recv": 400.0, "packet_count": 520,
            "conn_rate": 1.8, "failed_logins": 0, "unique_ports": 1,
            "ttl": 64, "tcp_flags": 1, "hour": 3, "is_business_hours": 0,
        },
        {
            "name": "SSH brute force",
            "protocol": "SSH", "duration": 0.4, "packet_size": 195.0,
            "bytes_sent": 980.0, "bytes_recv": 750.0, "packet_count": 12,
            "conn_rate": 32.0, "failed_logins": 22, "unique_ports": 1,
            "ttl": 128, "tcp_flags": 2, "hour": 4, "is_business_hours": 0,
        },
    ]

    print(f"\n  {'TRAFFIC SAMPLE':<35} {'PREDICTION':<22} {'CONFIDENCE':>10}  {'ISO FLAG':>8}")
    print(f"  {'─'*35} {'─'*22} {'─'*10}  {'─'*8}")

    for tc in test_cases:
        name = tc.pop("name")
        df   = pd.DataFrame([tc])
        X    = preprocessor.transform(df)
        preds, conf, iso_flags = ensemble.predict_with_confidence(X)

        pred_label = preprocessor.inverse_label(preds)[0]
        confidence = conf[0]
        iso_flag   = "⚠ ANOMALY" if iso_flags[0] else "OK"

        print(f"  {name:<35} {pred_label:<22} {confidence:>10.1%}  {iso_flag:>8}")


if __name__ == "__main__":
    main()
