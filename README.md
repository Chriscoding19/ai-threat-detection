# AI-Based Threat Detection System

A machine learning system that classifies network traffic and detects cyber attacks using multiple ML models including Random Forest, Gradient Boosting, and Isolation Forest.

---

## Architecture

```
ai-threat-detection/
├── main.py                              ← Full pipeline entry point
├── requirements.txt
├── data/
│   └── data_generator.py               ← Simulated traffic dataset generator
├── core/
│   └── preprocessor.py                 ← Feature engineering + scaling pipeline
├── models/
│   └── models.py                       ← RF, GradientBoosting, IsoForest, Ensemble
├── reports/
│   ├── visualizer.py                   ← Confusion matrix, feature importance, charts
│   ├── report_generator.py             ← HTML report output
│   └── figures/                        ← Generated PNG charts
└── tests/
    └── test_threat_detection.py        ← 23 unit tests (all passing)
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generates data, trains models, evaluates, reports)
python main.py

# Larger dataset for better accuracy
python main.py --samples 10000

# Skip plot generation (faster)
python main.py --no-plots

# Run unit tests
python tests/test_threat_detection.py
```

---

## What It Detects

| Attack Class | Key Indicators | Severity |
|---|---|---|
| **Normal** | Business hours, typical packet sizes | - |
| **DoS** | conn_rate > 100, tiny packets, packet flood | Critical |
| **Port Scan** | unique_ports > 100, SYN-only, short sessions | High |
| **Data Exfiltration** | bytes_sent > 100KB, off-hours, slow & steady | Critical |
| **Brute Force** | failed_logins > 10, same port, rapid connections | High |

---

## ML Models

### 1. Random Forest Classifier (Primary)
- 150 decision trees, balanced class weights
- Trained on 21 engineered features
- **Accuracy: 100%** on test set

### 2. Gradient Boosting Classifier
- 100 estimators, learning_rate=0.1
- Sequential boosting for error correction
- **Accuracy: 100%** on test set

### 3. Isolation Forest (Unsupervised)
- Does NOT require labels - detects zero-day attacks
- Flags statistically unusual traffic patterns
- **Accuracy: 98.7%** binary (normal vs anomaly)

### 4. Ensemble (RF + GB + IsoForest)
- Soft voting between RF and GB probabilities
- IsoForest provides additional anomaly signal
- **Accuracy: 100%** on test set

---

## Feature Engineering (21 features)

Raw features + engineered:

| Feature | Description |
|---|---|
| `conn_rate` | Connections per second |
| `unique_ports` | Number of distinct destination ports |
| `failed_logins` | Authentication failures |
| `bytes_sent/recv` | Volume of data transferred |
| `bytes_ratio` | Upload/download ratio (exfil indicator) |
| `bytes_per_packet` | Avg packet payload size |
| `is_night` | Off-hours binary flag |
| `is_high_port_scan` | unique_ports > 100 |
| `is_high_login_fail` | failed_logins > 10 |
| `is_large_upload` | bytes_sent > 100KB |
| `is_flood` | conn_rate > 100 |

---

## Results

```
MODEL                               ACCURACY       F1
─────────────────────────────────────────────────────
Random Forest                         1.0000   1.0000
Gradient Boosting                     1.0000   1.0000
Isolation Forest                      0.9870   0.9562
Ensemble (RF + GB + IsoForest)        1.0000   1.0000
```

### Live Prediction Demo
```
TRAFFIC SAMPLE                     PREDICTION          CONFIDENCE  ISO FLAG
Normal web browsing                normal                  100.0%       OK
DoS flood attack                   dos                     100.0%  ⚠ ANOMALY
Port scan reconnaissance           port_scan               100.0%  ⚠ ANOMALY
Data exfiltration (DNS tunneling)  data_exfiltration       100.0%  ⚠ ANOMALY
SSH brute force                    brute_force             100.0%  ⚠ ANOMALY
```

---

## Technologies

- **Language**: Python 3.10+
- **ML**: scikit-learn (RandomForest, GradientBoosting, IsolationForest)
- **Data**: pandas, numpy
- **Visualisation**: matplotlib, seaborn
- **Output**: HTML report + 6 PNG charts
