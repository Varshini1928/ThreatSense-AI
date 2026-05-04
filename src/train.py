"""
train.py
--------
Trains three complementary models:
  1. Random Forest  – multi-class attack classifier (primary)
  2. Isolation Forest – unsupervised anomaly detector
  3. One-Class SVM  – secondary anomaly detector

Run:
    python src/train.py
    python src/train.py --data data/sample/network_logs.csv
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
)

from preprocessing import (
    preprocess_pipeline, LABEL_NAMES, NUMERIC_FEATURES
)

# ─── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR    = Path("models")
RF_PATH      = MODEL_DIR / "random_forest.pkl"
ISO_PATH     = MODEL_DIR / "isolation_forest.pkl"
SVM_PATH     = MODEL_DIR / "ocsvm.pkl"
SCALER_PATH  = MODEL_DIR / "scaler.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


# ─── Training functions ──────────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    """
    Random Forest – supervised multi-class classifier.
    Detects AND classifies attack type.
    """
    print("\n[1/3] Training Random Forest …")
    t0 = time.time()

    clf = RandomForestClassifier(
        n_estimators=200,       # more trees = more stable predictions
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",  # handles class imbalance automatically
        random_state=42,
        n_jobs=-1,              # use all CPU cores
    )
    clf.fit(X_train, y_train)
    print(f"  ✅ Done in {time.time()-t0:.1f}s")
    return clf


def train_isolation_forest(X_train):
    """
    Isolation Forest – unsupervised anomaly detector.
    Does NOT need labels; learns what 'normal' looks like.
    Useful for detecting zero-day / unknown attacks.
    """
    print("\n[2/3] Training Isolation Forest …")
    t0 = time.time()

    # Train ONLY on normal traffic rows (contamination=0)
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # expect ~5% outliers in production
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)
    print(f"  ✅ Done in {time.time()-t0:.1f}s")
    return iso


def train_ocsvm(X_train):
    """
    One-Class SVM – secondary anomaly baseline.
    Faster inference than Isolation Forest on small datasets.
    """
    print("\n[3/3] Training One-Class SVM …")
    t0 = time.time()

    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    # SVM is slow on large data; train on a 5k subsample
    idx = np.random.choice(len(X_train), min(5000, len(X_train)), replace=False)
    svm.fit(X_train[idx])
    print(f"  ✅ Done in {time.time()-t0:.1f}s")
    return svm


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(clf, X_test, y_test) -> dict:
    """Compute and print evaluation metrics; return as dict."""
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    # AUC (one-vs-rest for multi-class)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)
        classes = sorted(set(y_test))
        try:
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes)
        except Exception:
            auc = None
    else:
        auc = None

    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n  Accuracy : {acc:.4f}")
    if auc:
        print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    return {
        "accuracy":   round(acc, 4),
        "roc_auc":    round(auc, 4) if auc else None,
        "report":     report,
        "confusion_matrix": cm,
    }


def feature_importance_report(clf, feature_names: list) -> list:
    """Return top-15 feature importances as list of dicts."""
    importances = clf.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    top15 = [{"feature": f, "importance": round(i, 4)} for f, i in ranked[:15]]
    print("\n  Top features:")
    for item in top15[:10]:
        bar = "█" * int(item["importance"] * 100)
        print(f"    {item['feature']:<25} {item['importance']:.4f} {bar}")
    return top15


# ─── Main ────────────────────────────────────────────────────────────────────

def main(data_path: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AI Threat Detection – Model Training")
    print("=" * 60)
    print(f"\nDataset : {data_path}")

    # ── Preprocess ──
    print("\n[Preprocessing]")
    data = preprocess_pipeline(
        filepath=data_path,
        scaler_path=str(SCALER_PATH),
        multiclass=True,
    )
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    feat_names      = data["feature_names"]

    print(f"  Train : {X_train.shape[0]:,} samples")
    print(f"  Test  : {X_test.shape[0]:,} samples")

    # ── Train ──
    rf  = train_random_forest(X_train, y_train)
    iso = train_isolation_forest(X_train)
    svm = train_ocsvm(X_train)

    # ── Evaluate RF ──
    print("\n[Evaluation – Random Forest]")
    metrics = evaluate(rf, X_test, y_test)
    metrics["feature_importance"] = feature_importance_report(rf, feat_names)

    # ── Save models ──
    print("\n[Saving models]")
    joblib.dump(rf,  RF_PATH);  print(f"  ✅ {RF_PATH}")
    joblib.dump(iso, ISO_PATH); print(f"  ✅ {ISO_PATH}")
    joblib.dump(svm, SVM_PATH); print(f"  ✅ {SVM_PATH}")

    # Persist metrics for dashboard
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ Metrics → {METRICS_PATH}")

    print("\n✅ Training complete. Run: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train threat detection models")
    parser.add_argument(
        "--data",
        default="data/sample/network_logs.csv",
        help="Path to the network log CSV file",
    )
    args = parser.parse_args()
    main(args.data)
