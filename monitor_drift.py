"""Compare recent PostgreSQL inference traffic with the model's training data."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import f1_score, precision_score, recall_score
from sqlalchemy import create_engine, desc, select

from ml_pipeline import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    REFERENCE_PATH,
)
from models import Transaction

logger = logging.getLogger("frauddetect.drift")


def _numeric_drift(
    reference: pd.Series, current: pd.Series, alpha: float, ks_threshold: float
) -> dict[str, Any]:
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if not len(reference_values) or not len(current_values):
        return {
            "test": "kolmogorov_smirnov",
            "status": "insufficient_values",
            "reference_missing_rate": float(reference.isna().mean()),
            "current_missing_rate": float(current.isna().mean()),
            "drift_detected": False,
        }

    statistic, p_value = ks_2samp(reference_values, current_values, method="auto")
    reference_missing_rate = float(reference.isna().mean())
    current_missing_rate = float(current.isna().mean())
    missing_rate_shift = abs(reference_missing_rate - current_missing_rate)
    detected = (p_value < alpha and statistic >= ks_threshold) or missing_rate_shift >= 0.05
    return {
        "test": "kolmogorov_smirnov",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "reference_missing_rate": reference_missing_rate,
        "current_missing_rate": current_missing_rate,
        "missing_rate_shift": missing_rate_shift,
        "drift_detected": bool(detected),
    }


def _categorical_drift(
    reference: pd.Series, current: pd.Series, alpha: float, effect_threshold: float
) -> dict[str, Any]:
    reference_values = reference.fillna("__MISSING__").astype(str)
    current_values = current.fillna("__MISSING__").astype(str)
    categories = sorted(set(reference_values.unique()) | set(current_values.unique()))
    if not categories or not len(reference_values) or not len(current_values):
        return {
            "test": "chi_square",
            "status": "insufficient_values",
            "drift_detected": False,
        }

    reference_counts = reference_values.value_counts().reindex(categories, fill_value=0)
    current_counts = current_values.value_counts().reindex(categories, fill_value=0)
    statistic, p_value, _, _ = chi2_contingency(
        np.vstack((reference_counts.to_numpy(), current_counts.to_numpy())),
        correction=False,
    )
    denominator = len(reference_values) + len(current_values)
    cramers_v = (
        float(np.sqrt(statistic / denominator))
        if len(categories) > 1 and denominator
        else 0.0
    )
    detected = p_value < alpha and cramers_v >= effect_threshold
    return {
        "test": "chi_square",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
        "category_count": len(categories),
        "drift_detected": bool(detected),
    }


def _concept_drift(
    current: pd.DataFrame,
    threshold: float,
    baseline_metrics: dict[str, Any],
    minimum_labeled_samples: int,
    performance_drop_threshold: float,
) -> dict[str, Any]:
    if "actual_is_fraud" not in current:
        return {
            "status": "unavailable",
            "reason": "No adjudicated production fraud labels are stored.",
            "drift_detected": False,
        }

    labeled = current.loc[current["actual_is_fraud"].notna()]
    if len(labeled) < minimum_labeled_samples:
        return {
            "status": "awaiting_labels",
            "labeled_samples": int(len(labeled)),
            "minimum_labeled_samples": minimum_labeled_samples,
            "reason": "Concept drift requires adjudicated production outcomes.",
            "drift_detected": False,
        }

    truth = labeled["actual_is_fraud"].astype(int)
    prediction = (labeled["fraud_score"].astype(float) >= threshold).astype(int)
    metrics = {
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1_score": float(f1_score(truth, prediction, zero_division=0)),
    }
    baseline_f1 = baseline_metrics.get("f1_score")
    baseline_recall = baseline_metrics.get("recall")
    deltas = {}
    if baseline_f1 is not None:
        deltas["f1_score_drop"] = float(baseline_f1) - metrics["f1_score"]
    if baseline_recall is not None:
        deltas["recall_drop"] = float(baseline_recall) - metrics["recall"]
    detected = any(
        value >= performance_drop_threshold
        for value in deltas.values()
    )
    return {
        "status": "evaluated",
        "labeled_samples": int(len(labeled)),
        "production_metrics": metrics,
        "baseline_metrics": {
            key: baseline_metrics.get(key) for key in ("precision", "recall", "f1_score")
        },
        "metric_deltas": deltas,
        "performance_drop_threshold": performance_drop_threshold,
        "drift_detected": detected,
    }


def run_monitor(
    database_url: str,
    *,
    reference_path: Path = REFERENCE_PATH,
    model_path: Path = MODEL_PATH,
    report_path: Path = BASE_DIR / "drift_report.json",
    window_hours: int = 24,
    minimum_samples: int = 200,
    minimum_labeled_samples: int = 100,
    max_transactions: int = 10_000,
    alpha: float = 0.01,
    ks_threshold: float = 0.10,
    categorical_effect_threshold: float = 0.10,
    performance_drop_threshold: float = 0.10,
) -> dict[str, Any]:
    """Run feature-distribution tests and labeled-outcome performance checks."""
    if not database_url:
        raise ValueError("A PostgreSQL DATABASE_URL is required")
    if window_hours < 1 or minimum_samples < 1 or max_transactions < 1:
        raise ValueError("window_hours, minimum_samples, and max_transactions must be positive")
    if not reference_path.exists():
        raise FileNotFoundError(f"Training reference data not found: {reference_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Fraud model artifact not found: {model_path}")

    reference = pd.read_csv(reference_path)
    required_features = set(FEATURE_COLUMNS)
    missing_features = required_features - set(reference.columns)
    if missing_features:
        raise ValueError(f"Reference data is missing features: {sorted(missing_features)}")
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or not {"threshold", "metrics"} <= artifact.keys():
        raise ValueError("Model artifact must include its threshold and evaluation metrics")

    since = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    statement = (
        select(
            Transaction.timestamp,
            *[getattr(Transaction, name) for name in FEATURE_COLUMNS],
            Transaction.fraud_score,
            Transaction.actual_is_fraud,
        )
        .where(Transaction.timestamp >= since)
        .order_by(desc(Transaction.timestamp))
        .limit(max_transactions)
    )
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            current = pd.read_sql(statement, connection)
    finally:
        engine.dispose()

    generated_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "generated_at": generated_at,
        "window_start": since.isoformat(),
        "window_hours": window_hours,
        "reference_dataset": str(reference_path),
        "model_artifact": str(model_path),
        "model_threshold": float(artifact["threshold"]),
        "reference_samples": int(len(reference)),
        "production_samples": int(len(current)),
        "feature_tests": {},
        "data_drift_detected": False,
        "concept_drift": {},
        "retrain_recommended": False,
    }

    if len(current) < minimum_samples:
        report["status"] = "insufficient_production_samples"
        report["minimum_samples"] = minimum_samples
        report["concept_drift"] = {
            "status": "not_evaluated",
            "reason": "Not enough recent inference records.",
            "drift_detected": False,
        }
    else:
        tests: dict[str, dict[str, Any]] = {}
        for feature in NUMERIC_FEATURES:
            tests[feature] = _numeric_drift(
                reference[feature], current[feature], alpha, ks_threshold
            )
        for feature in CATEGORICAL_FEATURES:
            tests[feature] = _categorical_drift(
                reference[feature],
                current[feature],
                alpha,
                categorical_effect_threshold,
            )
        data_drift = any(test["drift_detected"] for test in tests.values())
        concept_drift = _concept_drift(
            current,
            float(artifact["threshold"]),
            artifact["metrics"],
            minimum_labeled_samples,
            performance_drop_threshold,
        )
        report.update(
            {
                "status": "complete",
                "feature_tests": tests,
                "data_drift_detected": data_drift,
                "concept_drift": concept_drift,
                "retrain_recommended": data_drift or concept_drift["drift_detected"],
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["retrain_recommended"]:
        logger.error("Drift detected; model retraining review is recommended")
    elif report["status"] == "insufficient_production_samples":
        logger.warning("Drift report has insufficient production data")
    else:
        logger.info("No actionable drift detected")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--minimum-samples", type=int, default=200)
    parser.add_argument("--minimum-labeled-samples", type=int, default=100)
    parser.add_argument("--max-transactions", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--ks-threshold", type=float, default=0.10)
    parser.add_argument("--categorical-effect-threshold", type=float, default=0.10)
    parser.add_argument("--performance-drop-threshold", type=float, default=0.10)
    parser.add_argument("--report", type=Path, default=BASE_DIR / "drift_report.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        parser.error("Set DATABASE_URL to a PostgreSQL connection string")
    report = run_monitor(
        database_url,
        report_path=args.report,
        window_hours=args.window_hours,
        minimum_samples=args.minimum_samples,
        minimum_labeled_samples=args.minimum_labeled_samples,
        max_transactions=args.max_transactions,
        alpha=args.alpha,
        ks_threshold=args.ks_threshold,
        categorical_effect_threshold=args.categorical_effect_threshold,
        performance_drop_threshold=args.performance_drop_threshold,
    )
    print(json.dumps(report, indent=2))
    return 2 if report["retrain_recommended"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
