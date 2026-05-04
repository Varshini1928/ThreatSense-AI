"""
predict.py
----------
Inference engine used by the Streamlit dashboard.
Handles:
  - Loading saved models
  - Running predictions
  - Risk scoring
  - Explainability via feature importance
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

from preprocessing import (
    engineer_features, get_feature_matrix,
    LABEL_NAMES, NUMERIC_FEATURES,
)

# ─── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR   = Path("models")
RF_PATH     = MODEL_DIR / "random_forest.pkl"
ISO_PATH    = MODEL_DIR / "isolation_forest.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# ─── Risk score config ──────────────────────────────────────────────────────
# Each attack type carries a base severity weight
SEVERITY = {
    "normal":      0.0,
    "ddos":        0.90,
    "brute_force": 0.85,
    "port_scan":   0.60,
    "data_exfil":  0.95,
}

# Risk bucket thresholds
def risk_level(score: float) -> str:
    if score >= 0.80: return "CRITICAL"
    if score >= 0.60: return "HIGH"
    if score >= 0.35: return "MEDIUM"
    if score > 0.0:   return "LOW"
    return "SAFE"

RISK_COLOR = {
    "CRITICAL": "#FF2D55",
    "HIGH":     "#FF9500",
    "MEDIUM":   "#FFD60A",
    "LOW":      "#30D158",
    "SAFE":     "#0A84FF",
}


# ─── Model loader (cached via @st.cache_resource in app) ────────────────────

class ThreatDetector:
    """Wraps all models and provides a unified predict() interface."""

    def __init__(self):
        self.rf     = joblib.load(RF_PATH)
        self.iso    = joblib.load(ISO_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.feature_names = NUMERIC_FEATURES

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        """Full preprocessing pipeline for inference."""
        df_eng = engineer_features(df.copy())
        X      = get_feature_matrix(df_eng)
        return self.scaler.transform(X)

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all models; return enriched DataFrame with:
          - predicted_label  (attack type string)
          - predicted_class  (int)
          - anomaly_score    (Isolation Forest – lower = more anomalous)
          - attack_prob      (RF probability of attack)
          - risk_score       (0-1 combined score)
          - risk_level       (SAFE / LOW / MEDIUM / HIGH / CRITICAL)
          - is_threat        (bool)
          - top_features     (list – explanation)
        """
        X = self._transform(df)

        # 1. Random Forest – class label + probabilities
        rf_classes = self.rf.predict(X)
        rf_proba   = self.rf.predict_proba(X)           # shape (n, n_classes)

        # Probability that traffic is ANY attack type
        normal_idx  = list(self.rf.classes_).index(0) if 0 in self.rf.classes_ else 0
        attack_prob = 1.0 - rf_proba[:, normal_idx]

        # 2. Isolation Forest – anomaly score (normalised 0-1)
        raw_iso  = self.iso.decision_function(X)        # negative = more anomalous
        iso_norm = 1.0 - (raw_iso - raw_iso.min()) / (np.ptp(raw_iso) + 1e-9)

        # 3. Combine into risk score
        label_names   = [LABEL_NAMES.get(c, "normal") for c in rf_classes]
        base_severity = np.array([SEVERITY.get(l, 0.0) for l in label_names])
        risk_scores   = 0.50 * attack_prob + 0.30 * iso_norm + 0.20 * base_severity
        risk_scores   = np.clip(risk_scores, 0, 1)

        # 4. Explanation: top 3 features that pushed this prediction
        importances = self.rf.feature_importances_
        top_idx     = np.argsort(importances)[::-1][:5]
        top_feats   = [self.feature_names[i] for i in top_idx]

        # ── Build result DataFrame ──────────────────────────────────────────
        result = df.copy()
        result["predicted_label"] = label_names
        result["predicted_class"] = rf_classes
        result["attack_prob"]     = attack_prob.round(4)
        result["anomaly_score"]   = iso_norm.round(4)
        result["risk_score"]      = risk_scores.round(4)
        result["risk_level"]      = [risk_level(s) for s in risk_scores]
        result["is_threat"]       = result["predicted_label"] != "normal"

        # Add per-row explanation (top features + their scaled values)
        X_df = pd.DataFrame(X, columns=self.feature_names)
        explanations = []
        for i, row in X_df.iterrows():
            top = sorted(
                zip(self.feature_names, importances, row.values),
                key=lambda x: -x[1]
            )[:3]
            parts = [f"{f} ({v:.2f})" for f, _, v in top]
            explanations.append(" | ".join(parts))
        result["explanation"] = explanations

        return result

    def predict_single(self, row_dict: dict) -> dict:
        """Convenience wrapper for a single log entry (dict)."""
        df = pd.DataFrame([row_dict])
        return self.predict(df).iloc[0].to_dict()


# ─── Risk summary helpers (used by dashboard) ───────────────────────────────

def compute_summary(result_df: pd.DataFrame) -> dict:
    """Aggregate prediction results into dashboard KPIs."""
    total       = len(result_df)
    threats     = result_df["is_threat"].sum()
    safe        = total - threats
    avg_risk    = result_df["risk_score"].mean()
    max_risk    = result_df["risk_score"].max()

    attack_dist = (
        result_df[result_df["is_threat"]]["predicted_label"]
        .value_counts()
        .to_dict()
    )

    risk_dist = result_df["risk_level"].value_counts().to_dict()

    return {
        "total":       total,
        "threats":     int(threats),
        "safe":        int(safe),
        "threat_pct":  round(threats / total * 100, 1) if total else 0,
        "avg_risk":    round(float(avg_risk), 3),
        "max_risk":    round(float(max_risk), 3),
        "attack_dist": attack_dist,
        "risk_dist":   risk_dist,
    }