"""Generate synthetic payment data and train a calibrated fraud-risk pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.pkl"
METRICS_PATH = BASE_DIR / "fraud_model_metrics.json"
REFERENCE_PATH = BASE_DIR / "fraud_training_reference.csv"

LOCATIONS = (
    "New York", "London", "Berlin", "Paris", "San Francisco", "Chicago",
    "Madrid", "Rome", "Toronto", "Singapore", "Tokyo", "Lagos", "Moscow",
    "Tehran", "Pyongyang",
)
HIGH_RISK_COUNTRIES = frozenset({"Nigeria", "Russia", "Iran", "North Korea"})
LOCATION_COUNTRIES = {
    "New York": "United States", "London": "United Kingdom", "Berlin": "Germany",
    "Paris": "France", "San Francisco": "United States", "Chicago": "United States",
    "Madrid": "Spain", "Rome": "Italy", "Toronto": "Canada", "Singapore": "Singapore",
    "Tokyo": "Japan", "Lagos": "Nigeria", "Moscow": "Russia", "Tehran": "Iran",
    "Pyongyang": "North Korea",
}
MERCHANT_CATEGORIES = (
    "Retail", "Travel", "Food & Drink", "Electronics",
    "Entertainment", "Digital Goods", "Grocery",
)
NUMERIC_FEATURES = (
    "amount",
    "amount_to_user_avg_ratio",
    "is_high_risk_country_and_vpn",
    "velocity_amount_product",
    "transaction_frequency_1h",
    "is_vpn",
    "hour",
    "day_of_week",
)
CATEGORICAL_FEATURES = ("location", "merchant_category")
FEATURE_COLUMNS = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
TARGET_PRECISION = 0.85
TARGET_RECALL = 0.85


def generate_synthetic_dataset(
    n_samples: int = 16_000, random_state: int = 42
) -> pd.DataFrame:
    """Create reproducible payment data with sparse, behavior-driven fraud."""
    if n_samples < 200:
        raise ValueError("n_samples must be at least 200")

    rng = np.random.default_rng(random_state)
    user_ids = np.array([f"USR-{value:05d}" for value in rng.integers(1, 4_001, n_samples)])
    locations = rng.choice(LOCATIONS, size=n_samples)
    categories = rng.choice(MERCHANT_CATEGORIES, size=n_samples)
    amounts = np.round(rng.lognormal(mean=3.45, sigma=1.05, size=n_samples), 2)
    vpn = rng.binomial(1, 0.09, size=n_samples)
    frequency = rng.poisson(lam=1.2, size=n_samples)
    timestamps = pd.to_datetime(
        rng.integers(
            pd.Timestamp("2025-01-01", tz="UTC").value // 10**9,
            pd.Timestamp("2025-04-01", tz="UTC").value // 10**9,
            size=n_samples,
        ),
        unit="s",
        utc=True,
    )
    forced_fraud = rng.random(n_samples) < 0.012
    vpn[forced_fraud] = 1
    frequency[forced_fraud] = rng.integers(5, 13, size=forced_fraud.sum())
    amounts[forced_fraud] = np.round(
        rng.lognormal(mean=7.0, sigma=0.45, size=forced_fraud.sum()), 2
    )

    user_amounts = pd.Series(amounts).groupby(user_ids)
    prior_amount_sum = user_amounts.transform("sum") - amounts
    prior_amount_count = user_amounts.transform("count") - 1
    user_average = prior_amount_sum.div(
        prior_amount_count.where(prior_amount_count > 0)
    ).fillna(float(np.mean(amounts)))
    amount_ratio = amounts / np.maximum(user_average, 1)
    risky_country_vpn = np.array(
        [LOCATION_COUNTRIES[location] in HIGH_RISK_COUNTRIES for location in locations]
    ) & vpn.astype(bool)
    high_risk_pattern = (
        risky_country_vpn
        | ((vpn == 1) & (frequency >= 5))
        | ((amounts >= 1_200) & (frequency >= 4))
        | ((amount_ratio >= 6) & (frequency >= 3))
        | forced_fraud
    )
    fraud_probability = np.where(high_risk_pattern, 0.97, 0.0005)
    is_fraud = rng.binomial(1, fraud_probability)

    return pd.DataFrame(
        {
            "transaction_id": [f"TX-{i:08d}" for i in range(n_samples)],
            "user_id": user_ids,
            "amount": amounts,
            "timestamp": timestamps,
            "location": locations,
            "merchant_category": categories,
            "is_vpn": vpn.astype(int),
            "transaction_frequency_1h": frequency.astype(int),
            "is_fraud": is_fraud.astype(int),
        }
    )


def prepare_features(
    transactions: pd.DataFrame,
    user_average_amount: dict[str, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """Build matching model features using prior user spend when provided."""
    timestamps = pd.to_datetime(transactions["timestamp"], utc=True, errors="raise")
    features = transactions.loc[
        :, ["amount", "transaction_frequency_1h", "is_vpn", "location", "merchant_category"]
    ].copy()

    if user_average_amount is None:
        if "user_id" in transactions:
            grouped = transactions.groupby("user_id")["amount"]
            user_sums = grouped.transform("sum") - transactions["amount"]
            prior_counts = grouped.transform("count") - 1
            fallback = float(transactions["amount"].mean())
            average = user_sums.div(prior_counts.where(prior_counts > 0)).fillna(fallback)
        else:
            average = pd.Series(
                float(transactions["amount"].mean()), index=transactions.index
            )
    elif isinstance(user_average_amount, pd.Series):
        average = transactions["user_id"].map(user_average_amount)
    else:
        average = transactions["user_id"].map(user_average_amount)
    average = average.fillna(float(transactions["amount"].mean())).clip(lower=1.0)

    countries = features["location"].map(LOCATION_COUNTRIES).fillna("Unknown")
    features["amount_to_user_avg_ratio"] = features["amount"] / average
    features["is_high_risk_country_and_vpn"] = (
        countries.isin(HIGH_RISK_COUNTRIES) & features["is_vpn"].astype(bool)
    ).astype(int)
    features["velocity_amount_product"] = (
        features["amount"] * features["transaction_frequency_1h"]
    )
    features["hour"] = timestamps.dt.hour
    features["day_of_week"] = timestamps.dt.dayofweek
    return features.loc[:, FEATURE_COLUMNS]


def _choose_threshold(
    y_true: pd.Series, probabilities: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Choose the highest-recall PR threshold satisfying configured targets."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "targets_met": 0.0}

    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]
    meets_targets = (threshold_precision >= TARGET_PRECISION) & (
        threshold_recall >= TARGET_RECALL
    )
    if meets_targets.any():
        candidates = np.flatnonzero(meets_targets)
        selected = candidates[np.argmax(threshold_recall[candidates])]
    else:
        precision_candidates = np.flatnonzero(threshold_precision >= TARGET_PRECISION)
        if precision_candidates.size:
            selected = precision_candidates[np.argmax(threshold_recall[precision_candidates])]
        else:
            selected = int(np.argmax(
                2 * threshold_precision * threshold_recall
                / np.maximum(threshold_precision + threshold_recall, 1e-12)
            ))
    threshold = float(np.clip(thresholds[selected], 0.0, 1.0))
    return threshold, {
        "precision": float(threshold_precision[selected]),
        "recall": float(threshold_recall[selected]),
        "targets_met": float(bool(meets_targets.any())),
    }


def train_model(
    n_samples: int = 16_000, random_state: int = 42, save: bool = True
) -> tuple[Pipeline, dict[str, Any]]:
    """Train with SMOTENC and tune the alert threshold on a validation split."""
    data = generate_synthetic_dataset(n_samples=n_samples, random_state=random_state)
    train_data, holdout_data = train_test_split(
        data, test_size=0.4, stratify=data["is_fraud"], random_state=random_state
    )
    validation_data, test_data = train_test_split(
        holdout_data,
        test_size=0.5,
        stratify=holdout_data["is_fraud"],
        random_state=random_state,
    )

    training_averages = train_data.groupby("user_id")["amount"].mean()
    x_train = prepare_features(train_data)
    x_validation = prepare_features(validation_data, training_averages)
    x_test = prepare_features(test_data, training_averages)
    y_train = train_data["is_fraud"]
    y_validation = validation_data["is_fraud"]
    y_test = test_data["is_fraud"]

    numeric_indices = list(range(len(NUMERIC_FEATURES)))
    categorical_indices = list(
        range(len(NUMERIC_FEATURES), len(FEATURE_COLUMNS))
    )
    imputation = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), list(NUMERIC_FEATURES)),
            (
                "categorical",
                SimpleImputer(strategy="constant", fill_value="Unknown"),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        verbose_feature_names_out=False,
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_indices),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_indices,
            ),
        ]
    )
    selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=80,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        threshold="median",
    )
    pipeline = Pipeline(
        steps=[
            ("imputation", imputation),
            (
                "sampling",
                SMOTENC(
                    categorical_features=categorical_indices,
                    sampling_strategy=0.5,
                    random_state=random_state,
                    k_neighbors=3,
                ),
            ),
            ("preprocessing", preprocessing),
            ("feature_selection", selector),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=220,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    validation_probabilities = pipeline.predict_proba(x_validation)[:, 1]
    threshold, validation_metrics = _choose_threshold(y_validation, validation_probabilities)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "evaluation_dataset": "synthetic_demo",
        "threshold": threshold,
        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "threshold_targets_met": validation_metrics["targets_met"],
    }

    if save:
        anomaly_preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    list(NUMERIC_FEATURES),
                ),
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy="constant", fill_value="Unknown"),
                            ),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    list(CATEGORICAL_FEATURES),
                ),
            ]
        )
        anomaly_training_features = anomaly_preprocessor.fit_transform(x_train)
        anomaly_detector = IsolationForest(
            n_estimators=150,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        anomaly_detector.fit(anomaly_training_features)
        reference = x_train.copy()
        reference["is_fraud"] = y_train.to_numpy()
        reference.to_csv(REFERENCE_PATH, index=False)
        artifact: dict[str, Any] = {
            "model": pipeline,
            "threshold": threshold,
            "metrics": metrics,
            "user_average_amount": training_averages.to_dict(),
            "feature_columns": list(FEATURE_COLUMNS),
            "anomaly_preprocessor": anomaly_preprocessor,
            "anomaly_detector": anomaly_detector,
        }
        joblib.dump(artifact, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return pipeline, metrics


if __name__ == "__main__":
    _, evaluation = train_model()
    print("Holdout classification metrics (threshold selected on validation data)")
    for metric, value in evaluation.items():
        if isinstance(value, float):
            formatted = f"{value:.4f}"
        else:
            formatted = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        print(f"{metric.replace('_', ' ').title()}: {formatted}")
