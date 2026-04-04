"""
data_generator.py
-----------------
Generates realistic simulated network traffic data for training and testing
anomaly detection models.

Features generated:
  - Packet size, duration, bytes transferred
  - Protocol, port numbers, flags
  - Connection rate, failed attempts
  - Time-based patterns (business hours vs off-hours)
  - Labelled anomalies: DoS, Port Scan, Data Exfiltration, Brute Force
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

PROTOCOLS = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS", "FTP", "SSH"]
ATTACK_TYPES = ["normal", "dos", "port_scan", "data_exfiltration", "brute_force"]


def generate_dataset(n_samples: int = 5000, anomaly_ratio: float = 0.15) -> pd.DataFrame:
    """
    Generate a labelled network traffic dataset.

    Parameters
    ----------
    n_samples     : Total number of traffic records
    anomaly_ratio : Fraction of records that are anomalous (attacks)

    Returns
    -------
    pd.DataFrame with features and 'label' column
    """
    n_normal  = int(n_samples * (1 - anomaly_ratio))
    n_anomaly = n_samples - n_normal

    normal_df = _generate_normal_traffic(n_normal)
    attack_df = _generate_attack_traffic(n_anomaly)

    df = pd.concat([normal_df, attack_df], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  Dataset generated: {len(df)} records")
    print(f"  Normal  : {len(df[df['label']=='normal'])} ({len(df[df['label']=='normal'])/len(df)*100:.1f}%)")
    for attack in ["dos", "port_scan", "data_exfiltration", "brute_force"]:
        count = len(df[df['label']==attack])
        if count > 0:
            print(f"  {attack:<22}: {count} ({count/len(df)*100:.1f}%)")

    return df


def _generate_normal_traffic(n: int) -> pd.DataFrame:
    """Generate realistic benign network traffic."""
    records = []
    base_time = datetime(2024, 1, 15, 8, 0, 0)

    for i in range(n):
        # Business hours bias
        hour = np.random.choice(range(24), p=_hour_probabilities())
        timestamp = base_time + timedelta(hours=i * 0.05, minutes=random.randint(0, 59))

        protocol = np.random.choice(
            ["TCP", "UDP", "HTTP", "HTTPS", "DNS"],
            p=[0.30, 0.15, 0.25, 0.20, 0.10]
        )

        # Normal traffic characteristics
        duration     = abs(np.random.normal(2.5, 1.5))           # seconds
        packet_size  = abs(np.random.normal(512, 200))            # bytes
        bytes_sent   = abs(np.random.normal(5000, 3000))
        bytes_recv   = abs(np.random.normal(15000, 8000))
        packet_count = max(1, int(np.random.normal(20, 10)))
        conn_rate    = abs(np.random.normal(5, 2))                # connections/sec
        failed_logins = max(0, int(np.random.normal(0.1, 0.3)))
        unique_ports  = max(1, int(np.random.normal(3, 2)))
        ttl           = np.random.choice([64, 128, 255], p=[0.4, 0.4, 0.2])
        tcp_flags     = np.random.choice([0, 1, 2, 4], p=[0.1, 0.6, 0.2, 0.1])
        is_business_hours = 1 if 8 <= hour <= 18 else 0

        records.append({
            "timestamp":        timestamp.isoformat(),
            "hour":             hour,
            "protocol":         protocol,
            "duration":         round(duration, 3),
            "packet_size":      round(packet_size, 1),
            "bytes_sent":       round(bytes_sent, 1),
            "bytes_recv":       round(bytes_recv, 1),
            "packet_count":     packet_count,
            "conn_rate":        round(conn_rate, 3),
            "failed_logins":    failed_logins,
            "unique_ports":     unique_ports,
            "ttl":              ttl,
            "tcp_flags":        tcp_flags,
            "is_business_hours": is_business_hours,
            "label":            "normal",
        })

    return pd.DataFrame(records)


def _generate_attack_traffic(n: int) -> pd.DataFrame:
    """Generate labelled attack traffic for 4 attack types."""
    attack_types = ["dos", "port_scan", "data_exfiltration", "brute_force"]
    weights      = [0.35, 0.25, 0.20, 0.20]

    counts = np.random.multinomial(n, weights)
    frames = []
    base_time = datetime(2024, 1, 15, 2, 0, 0)  # Attacks often at night

    for attack, count in zip(attack_types, counts):
        for i in range(count):
            hour = np.random.choice(
                range(24),
                p=_attack_hour_probabilities(attack)
            )
            timestamp = base_time + timedelta(minutes=i * 0.5)

            rec = _make_attack_record(attack, hour, timestamp)
            frames.append(rec)

    return pd.DataFrame(frames)


def _make_attack_record(attack: str, hour: int, timestamp: datetime) -> dict:
    base = {
        "timestamp":        timestamp.isoformat(),
        "hour":             hour,
        "is_business_hours": 1 if 8 <= hour <= 18 else 0,
        "label":            attack,
    }

    if attack == "dos":
        # DoS: massive packet rate, tiny packets, many connections
        base.update({
            "protocol":     np.random.choice(["TCP", "UDP", "ICMP"], p=[0.4, 0.4, 0.2]),
            "duration":     abs(np.random.normal(0.1, 0.05)),
            "packet_size":  abs(np.random.normal(60, 10)),          # tiny packets
            "bytes_sent":   abs(np.random.normal(50000, 20000)),    # huge volume
            "bytes_recv":   abs(np.random.normal(1000, 500)),
            "packet_count": max(100, int(np.random.normal(500, 200))),  # flood
            "conn_rate":    abs(np.random.normal(200, 50)),         # very high rate
            "failed_logins": 0,
            "unique_ports": max(1, int(np.random.normal(2, 1))),
            "ttl":          np.random.choice([1, 64, 128]),
            "tcp_flags":    np.random.choice([2, 4, 8], p=[0.5, 0.3, 0.2]),  # SYN flood
        })

    elif attack == "port_scan":
        # Port scan: many unique ports, short durations, lots of failures
        base.update({
            "protocol":     np.random.choice(["TCP", "UDP"], p=[0.7, 0.3]),
            "duration":     abs(np.random.normal(0.05, 0.02)),
            "packet_size":  abs(np.random.normal(60, 20)),
            "bytes_sent":   abs(np.random.normal(300, 100)),
            "bytes_recv":   abs(np.random.normal(100, 50)),
            "packet_count": max(1, int(np.random.normal(3, 1))),
            "conn_rate":    abs(np.random.normal(50, 20)),
            "failed_logins": 0,
            "unique_ports": max(50, int(np.random.normal(500, 200))),  # key indicator
            "ttl":          np.random.choice([64, 128]),
            "tcp_flags":    2,  # SYN only
        })

    elif attack == "data_exfiltration":
        # Exfiltration: large outbound data, off-hours, unusual protocols
        base.update({
            "protocol":     np.random.choice(["FTP", "HTTPS", "DNS"], p=[0.3, 0.4, 0.3]),
            "duration":     abs(np.random.normal(120, 60)),         # long sessions
            "packet_size":  abs(np.random.normal(1400, 100)),       # max packet size
            "bytes_sent":   abs(np.random.normal(500000, 200000)),  # massive outbound
            "bytes_recv":   abs(np.random.normal(500, 200)),
            "packet_count": max(50, int(np.random.normal(400, 100))),
            "conn_rate":    abs(np.random.normal(2, 1)),            # slow and steady
            "failed_logins": 0,
            "unique_ports": max(1, int(np.random.normal(2, 1))),
            "ttl":          np.random.choice([64, 128]),
            "tcp_flags":    1,
        })

    elif attack == "brute_force":
        # Brute force: many failed logins, rapid connections to same port
        base.update({
            "protocol":     np.random.choice(["SSH", "FTP", "HTTP"], p=[0.5, 0.3, 0.2]),
            "duration":     abs(np.random.normal(0.5, 0.2)),
            "packet_size":  abs(np.random.normal(200, 50)),
            "bytes_sent":   abs(np.random.normal(1000, 300)),
            "bytes_recv":   abs(np.random.normal(800, 200)),
            "packet_count": max(5, int(np.random.normal(15, 5))),
            "conn_rate":    abs(np.random.normal(30, 10)),
            "failed_logins": max(5, int(np.random.normal(20, 8))),  # key indicator
            "unique_ports": max(1, int(np.random.normal(1, 0.5))),  # same port
            "ttl":          np.random.choice([64, 128]),
            "tcp_flags":    np.random.choice([1, 2], p=[0.5, 0.5]),
        })

    # Round numeric fields
    for k in ["duration", "packet_size", "bytes_sent", "bytes_recv", "conn_rate"]:
        if k in base:
            base[k] = round(float(base[k]), 3)

    return base


def _hour_probabilities():
    """Higher probability during business hours for normal traffic."""
    probs = np.ones(24)
    probs[8:19] *= 4   # Business hours 4x more traffic
    probs[0:6]  *= 0.3 # Night very low
    return probs / probs.sum()


def _attack_hour_probabilities(attack: str):
    """Attacks tend to happen at night."""
    probs = np.ones(24)
    if attack in ["data_exfiltration", "brute_force"]:
        probs[0:6]  *= 5   # Heavy night activity
        probs[8:18] *= 0.5
    else:
        probs[0:8]  *= 3
    return probs / probs.sum()


if __name__ == "__main__":
    df = generate_dataset(5000)
    df.to_csv("data/network_traffic.csv", index=False)
    print(f"\n  Saved to data/network_traffic.csv")
    print(df.head())
