"""
generate_dataset.py
-------------------
Generates a synthetic network intrusion dataset that mimics real-world
traffic patterns (similar to KDD Cup 1999 / NSL-KDD structure).

Run: python src/generate_dataset.py
Output: data/sample/network_logs.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility
np.random.seed(42)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
N_SAMPLES = 15_000
ATTACK_RATIO = 0.30          # 30 % attack traffic
OUTPUT_PATH = Path("data/sample/network_logs.csv")

ATTACK_TYPES = {
    "normal":      0.70,
    "ddos":        0.10,
    "brute_force": 0.08,
    "port_scan":   0.07,
    "data_exfil":  0.05,
}


# ─────────────────────────────────────────────
#  Feature generators per traffic class
# ─────────────────────────────────────────────

def gen_normal(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "duration":          np.random.exponential(5,  n).clip(0, 120),
        "src_bytes":         np.random.lognormal(8,  1.5, n).clip(0, 1e6),
        "dst_bytes":         np.random.lognormal(9,  1.5, n).clip(0, 1e6),
        "land":              np.zeros(n, dtype=int),
        "wrong_fragment":    np.random.choice([0, 1], n, p=[0.98, 0.02]),
        "urgent":            np.zeros(n, dtype=int),
        "hot":               np.random.randint(0,  5,  n),
        "num_failed_logins": np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "logged_in":         np.random.choice([0, 1], n, p=[0.30, 0.70]),
        "num_compromised":   np.zeros(n, dtype=int),
        "count":             np.random.randint(1,  50, n),
        "srv_count":         np.random.randint(1,  30, n),
        "serror_rate":       np.random.beta(1,  20, n),
        "dst_host_count":    np.random.randint(1, 100, n),
        "dst_host_srv_count":np.random.randint(1,  60, n),
        "protocol_type":     np.random.choice(["tcp", "udp", "icmp"], n, p=[0.6, 0.3, 0.1]),
        "label":             np.full(n, "normal"),
    })


def gen_ddos(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "duration":          np.random.exponential(0.5, n).clip(0, 10),
        "src_bytes":         np.random.lognormal(4,  0.5, n),
        "dst_bytes":         np.random.lognormal(3,  0.5, n),
        "land":              np.zeros(n, dtype=int),
        "wrong_fragment":    np.random.choice([0, 1], n, p=[0.60, 0.40]),
        "urgent":            np.zeros(n, dtype=int),
        "hot":               np.zeros(n, dtype=int),
        "num_failed_logins": np.zeros(n, dtype=int),
        "logged_in":         np.zeros(n, dtype=int),
        "num_compromised":   np.zeros(n, dtype=int),
        "count":             np.random.randint(400, 512, n),   # very high
        "srv_count":         np.random.randint(400, 512, n),
        "serror_rate":       np.random.beta(15, 2,  n).clip(0, 1),  # high error
        "dst_host_count":    np.random.randint(200, 256, n),
        "dst_host_srv_count":np.random.randint(200, 256, n),
        "protocol_type":     np.random.choice(["tcp", "udp", "icmp"], n, p=[0.3, 0.3, 0.4]),
        "label":             np.full(n, "ddos"),
    })


def gen_brute_force(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "duration":          np.random.uniform(0.1, 2, n),
        "src_bytes":         np.random.lognormal(5,  0.5, n),
        "dst_bytes":         np.random.lognormal(4,  0.5, n),
        "land":              np.zeros(n, dtype=int),
        "wrong_fragment":    np.zeros(n, dtype=int),
        "urgent":            np.zeros(n, dtype=int),
        "hot":               np.random.randint(0,  3, n),
        "num_failed_logins": np.random.randint(3, 10, n),   # many failed logins
        "logged_in":         np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "num_compromised":   np.random.choice([0, 1], n, p=[0.80, 0.20]),
        "count":             np.random.randint(100, 300, n),
        "srv_count":         np.random.randint(1,   10, n),
        "serror_rate":       np.random.beta(2, 5, n),
        "dst_host_count":    np.random.randint(1,  20, n),
        "dst_host_srv_count":np.random.randint(1,  10, n),
        "protocol_type":     np.random.choice(["tcp", "udp"], n, p=[0.95, 0.05]),
        "label":             np.full(n, "brute_force"),
    })


def gen_port_scan(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "duration":          np.zeros(n),                     # instant probes
        "src_bytes":         np.random.randint(0, 50, n).astype(float),
        "dst_bytes":         np.zeros(n),
        "land":              np.zeros(n, dtype=int),
        "wrong_fragment":    np.zeros(n, dtype=int),
        "urgent":            np.zeros(n, dtype=int),
        "hot":               np.zeros(n, dtype=int),
        "num_failed_logins": np.zeros(n, dtype=int),
        "logged_in":         np.zeros(n, dtype=int),
        "num_compromised":   np.zeros(n, dtype=int),
        "count":             np.random.randint(200, 512, n),  # scanning many ports
        "srv_count":         np.random.randint(1,   10, n),
        "serror_rate":       np.random.beta(10, 2, n).clip(0, 1),
        "dst_host_count":    np.random.randint(200, 256, n),
        "dst_host_srv_count":np.random.randint(1,   20, n),
        "protocol_type":     np.random.choice(["tcp", "udp"], n, p=[0.90, 0.10]),
        "label":             np.full(n, "port_scan"),
    })


def gen_data_exfil(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "duration":          np.random.uniform(60, 600, n),   # long sessions
        "src_bytes":         np.random.lognormal(6, 0.5, n),
        "dst_bytes":         np.random.lognormal(13, 0.8, n), # huge outbound
        "land":              np.zeros(n, dtype=int),
        "wrong_fragment":    np.zeros(n, dtype=int),
        "urgent":            np.zeros(n, dtype=int),
        "hot":               np.random.randint(5, 30, n),
        "num_failed_logins": np.zeros(n, dtype=int),
        "logged_in":         np.ones(n, dtype=int),           # logged in
        "num_compromised":   np.random.randint(1, 5, n),
        "count":             np.random.randint(1, 30, n),
        "srv_count":         np.random.randint(1, 10, n),
        "serror_rate":       np.random.beta(1, 20, n),
        "dst_host_count":    np.random.randint(1, 20, n),
        "dst_host_srv_count":np.random.randint(1, 10, n),
        "protocol_type":     np.random.choice(["tcp", "udp"], n, p=[0.85, 0.15]),
        "label":             np.full(n, "data_exfil"),
    })


# ─────────────────────────────────────────────
#  Build dataset
# ─────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    counts = {k: int(v * N_SAMPLES) for k, v in ATTACK_TYPES.items()}
    # fix rounding so totals match exactly
    counts["normal"] += N_SAMPLES - sum(counts.values())

    generators = {
        "normal":      gen_normal,
        "ddos":        gen_ddos,
        "brute_force": gen_brute_force,
        "port_scan":   gen_port_scan,
        "data_exfil":  gen_data_exfil,
    }

    frames = [generators[k](v) for k, v in counts.items()]
    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)

    # Add timestamp column (simulate last 24 hours of logs)
    df.insert(0, "timestamp", pd.date_range(
        end=pd.Timestamp.now(), periods=len(df), freq="6s"
    ).strftime("%Y-%m-%d %H:%M:%S"))

    # Add synthetic source/destination IPs
    def rand_ip(n):
        return [
            f"{np.random.randint(1,255)}.{np.random.randint(0,255)}"
            f".{np.random.randint(0,255)}.{np.random.randint(1,255)}"
            for _ in range(n)
        ]

    df.insert(1, "src_ip", rand_ip(len(df)))
    df.insert(2, "dst_ip", rand_ip(len(df)))
    df.insert(3, "src_port", np.random.randint(1024, 65535, len(df)))
    df.insert(4, "dst_port", np.random.choice(
        [22, 80, 443, 3306, 3389, 8080, 21, 25],
        len(df), p=[0.15, 0.25, 0.25, 0.1, 0.1, 0.05, 0.05, 0.05]
    ))

    return df


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic network intrusion dataset...")
    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df):,} records → {OUTPUT_PATH}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")