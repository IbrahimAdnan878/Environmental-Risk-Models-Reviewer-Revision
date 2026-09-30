#!/usr/bin/env python3
"""
Reviewer-revision Random Forest TRIAL 1 (LOCAL execution).

Purpose
-------
This script is a reviewer-safe local version of the Random Forest workflow.
It is designed to be placed in the SAME directory as LATEST.parquet and run
from that directory.

Main changes for the major revision
-----------------------------------
1. Uses chronological 70% train / 15% validation / 15% final test splits.
2. Removes direct drought-label construction variables from drought features.
3. Removes direct/current precipitation proxies from precipitation regression.
4. Uses class balancing for Random Forest classification.
5. Uses one fixed, unified Random Forest configuration for every target/city.
6. Reports BOTH validation and final-test metrics in the metrics CSV.
7. Adds TP, TN, FP and FN for binary classifiers.
8. Stores the full confusion matrix and classification report.
9. Stores the exact feature list and selected hyperparameters for each target.
10. Prints dust class counts by city to diagnose Erbil/Kirkuk imbalance.
11. Reads LATEST.parquet from the script directory by default.
12. Tunes a binary probability threshold on validation data to avoid default-0.50 collapse on imbalanced classes.\n13. Re-fits selected hyperparameters on train+validation before final untouched-test evaluation.\n14. Writes prediction and metrics CSVs to the script directory by default.

Important interpretation
------------------------
The 70/15/15 held-out results are chronological backtesting results.  The
30-day output generated afterwards is a prospective prediction horizon built
from historical analogue rows; it is NOT independently validated against
future observations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# -----------------------------------------------------------------------------
# Paths / defaults
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "LATEST.parquet"
DEFAULT_PREDICTIONS = SCRIPT_DIR / "rf_trial1_predictions.csv"
DEFAULT_METRICS = SCRIPT_DIR / "rf_trial1_metrics.csv"
DEFAULT_FEATURE_AUDIT = SCRIPT_DIR / "rf_trial1_feature_audit.csv"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# -----------------------------------------------------------------------------
# Leakage-control definitions
# -----------------------------------------------------------------------------
# Non-predictive identifiers / source markers / labels that should not be used
# as general model inputs.
COMMON_EXCLUDE = {
    "city",
    "timestamp",
    "date",
    "latitude",
    "longitude",
    "is_simulated_dust",
    "is_simulated_drought",
    "sim_source_dust_code",
    "sim_source_drought_code",
    "data_source_dust_met_code",
    "data_source_drought_code",
}

# The collection pipeline derives drought_flag using these three engineered
# variables and derives drought_severity primarily from precip_30d_sum.
# We additionally exclude their same-day source variables to make the revised
# experiment conservative and easier to defend as leakage-safe.
DROUGHT_DIRECT_LEAKAGE = {
    "drought_flag",
    "drought_severity_code",
    "precip_30d_sum",
    "soil_moisture_rootzone_30d_mean",
    "vpd_30d_mean",
    # Direct same-day/source components feeding those engineered indicators.
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "precip_7d_sum",
    "soil_moisture_rootzone_daily_mean",
    "soil_moisture_0_to_7cm_daily_mean",
    "soil_moisture_7_to_28cm_daily_mean",
    "soil_moisture_28_to_100cm_daily_mean",
    "soil_moisture_100_to_255cm_daily_mean",
    "vapour_pressure_deficit_daily_mean",
}

# For precipitation_sum regression, remove variables that directly or nearly
# directly describe the current-day rainfall target, and remove drought labels
# that were themselves constructed from precipitation-related quantities.
PRECIP_DIRECT_LEAKAGE = {
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "precip_7d_sum",
    "precip_30d_sum",
    "drought_flag",
    "drought_severity_code",
    "soil_moisture_rootzone_30d_mean",
    "vpd_30d_mean",
}

# Dust label construction uses dust-event/intensity/duration information and,
# for proxy-derived events, same-day PM/AOD/wind thresholds.  To avoid giving
# the classifier the rule that created the proxy label, exclude current-day
# proxy variables and rolling windows that include the current row.  Lag1/lag2
# values remain available because they represent previous observations.
DUST_DIRECT_LEAKAGE = {
    "dust_event",
    "dust_intensity_code",
    "dust_intensity_level",
    "duration_hours",
    "pm25",
    "pm10",
    "aod",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_speed_mean",
    "dust_event_x",
}

DUST_CURRENT_ROLLING_PREFIXES = (
    "pm25_roll",
    "pm10_roll",
    "aod_roll",
    "wind_speed_10m_roll",
    "wind_gusts_10m_roll",
    "wind_speed_mean_roll",
)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


def safe_float(value) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    if "timestamp" not in df.columns or "city" not in df.columns:
        raise ValueError("Dataset must contain 'timestamp' and 'city'.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["city", "timestamp"]).copy()
    df = df.sort_values(["timestamp", "city"]).reset_index(drop=True)
    return df


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def sanitize_frame(
    df: pd.DataFrame,
    feature_cols: List[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert to numeric and impute using medians from TRAINING data only."""
    work = df.loc[:, feature_cols].copy()
    ref = reference_df.loc[:, feature_cols].copy()

    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        ref[col] = pd.to_numeric(ref[col], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan)
    ref = ref.replace([np.inf, -np.inf], np.nan)
    medians = ref.median(numeric_only=True).reindex(feature_cols)

    for col in feature_cols:
        fill = medians.get(col, np.nan)
        if pd.isna(fill):
            fill = 0.0
        work[col] = work[col].fillna(fill)

    return work.astype(float)


def chronological_70_15_15(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by UNIQUE calendar timestamps so the same date cannot appear in two
    different subsets.  This is chronological, not shuffled.
    """
    work = df.sort_values(["timestamp", "city"]).reset_index(drop=True).copy()
    unique_dates = np.array(sorted(pd.Series(work["timestamp"].dt.normalize().unique()).tolist()))

    if len(unique_dates) < 10:
        raise ValueError("Too few unique dates for a reliable 70/15/15 split.")

    train_end = max(1, int(np.floor(len(unique_dates) * TRAIN_RATIO)))
    val_end = max(train_end + 1, int(np.floor(len(unique_dates) * (TRAIN_RATIO + VALIDATION_RATIO))))
    val_end = min(val_end, len(unique_dates) - 1)

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    norm = work["timestamp"].dt.normalize()
    train = work[norm.isin(train_dates)].copy()
    val = work[norm.isin(val_dates)].copy()
    test = work[norm.isin(test_dates)].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Chronological split produced an empty subset: "
            f"train={len(train)}, validation={len(val)}, test={len(test)}"
        )
    return train, val, test


def auto_detect_dust_target(df: pd.DataFrame) -> str:
    if "dust_event" in df.columns:
        return "dust_event"
    if "dust_event_x" in df.columns:
        return "dust_event_x"
    raise ValueError("Dust target not found (expected dust_event or dust_event_x).")


# -----------------------------------------------------------------------------
# Feature selectors
# -----------------------------------------------------------------------------
def select_drought_features(df: pd.DataFrame) -> List[str]:
    exclude = COMMON_EXCLUDE | DROUGHT_DIRECT_LEAKAGE | {
        "dust_event",
        "dust_event_x",
        "dust_intensity_code",
        "dust_intensity_level",
        "duration_hours",
    }
    return [c for c in numeric_columns(df) if c not in exclude]


def select_precip_features(df: pd.DataFrame) -> List[str]:
    exclude = COMMON_EXCLUDE | PRECIP_DIRECT_LEAKAGE | {
        "dust_event",
        "dust_event_x",
        "dust_intensity_code",
        "dust_intensity_level",
        "duration_hours",
    }
    return [c for c in numeric_columns(df) if c not in exclude]


def select_dust_features(df: pd.DataFrame, target_col: str) -> List[str]:
    exclude = COMMON_EXCLUDE | DUST_DIRECT_LEAKAGE | {
        target_col,
        "drought_flag",
        "drought_severity_code",
    }
    features = []
    for c in numeric_columns(df):
        if c in exclude:
            continue
        if any(c.startswith(prefix) for prefix in DUST_CURRENT_ROLLING_PREFIXES):
            continue
        features.append(c)
    return features


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def classifier_metric_row(
    *,
    model_name: str,
    target_name: str,
    split: str,
    city: str,
    y_true,
    y_pred,
    y_proba,
    labels: List[int],
    average: str,
    feature_cols: List[str],
    params: Dict[str, object],
    train_n: int,
    validation_n: int,
    test_n: int,
    class_counts: Optional[Dict[str, object]] = None,
    decision_threshold: Optional[float] = None,
) -> Dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tn = fp = fn = tp = None
    if len(labels) == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    row = {
        "model_name": model_name,
        "model_type": "classifier",
        "target_name": target_name,
        "city": city,
        "split": split,
        "n_samples": int(len(y_true)),
        "train_n": int(train_n),
        "validation_n": int(validation_n),
        "test_n": int(test_n),
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "precision": safe_float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": safe_float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": safe_float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "loss": None,
        "auc_roc": None,
        "r2_score": None,
        "mse": None,
        "rmse": None,
        "mae": None,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": safe_json(cm.tolist()),
        "classification_report": safe_json(
            classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
        ),
        "n_features": len(feature_cols),
        "feature_list": safe_json(feature_cols),
        "selected_hyperparameters": safe_json(params),
        "class_counts": safe_json(class_counts or {}),
        "decision_threshold": decision_threshold,
    }

    if y_proba is not None:
        try:
            row["loss"] = safe_float(log_loss(y_true, y_proba, labels=labels))
        except Exception:
            pass
        try:
            proba = np.asarray(y_proba)
            if len(np.unique(y_true)) == 2 and proba.ndim == 2 and proba.shape[1] >= 2:
                row["auc_roc"] = safe_float(roc_auc_score(y_true, proba[:, 1]))
            elif len(np.unique(y_true)) > 2 and proba.ndim == 2:
                row["auc_roc"] = safe_float(
                    roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
                )
        except Exception:
            pass
    return row


def regressor_metric_row(
    *,
    model_name: str,
    target_name: str,
    split: str,
    city: str,
    y_true,
    y_pred,
    feature_cols: List[str],
    params: Dict[str, object],
    train_n: int,
    validation_n: int,
    test_n: int,
) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model_name": model_name,
        "model_type": "regressor",
        "target_name": target_name,
        "city": city,
        "split": split,
        "n_samples": int(len(y_true)),
        "train_n": int(train_n),
        "validation_n": int(validation_n),
        "test_n": int(test_n),
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "loss": safe_float(mse),
        "auc_roc": None,
        "r2_score": safe_float(r2_score(y_true, y_pred)),
        "mse": safe_float(mse),
        "rmse": safe_float(np.sqrt(mse)),
        "mae": safe_float(mean_absolute_error(y_true, y_pred)),
        "tn": None,
        "fp": None,
        "fn": None,
        "tp": None,
        "confusion_matrix": None,
        "classification_report": None,
        "n_features": len(feature_cols),
        "feature_list": safe_json(feature_cols),
        "selected_hyperparameters": safe_json(params),
        "class_counts": None,
    }


# -----------------------------------------------------------------------------
# Trial 1: FIXED, UNIFIED Random Forest hyperparameters
# -----------------------------------------------------------------------------
# The user requested a controlled first trial in which every Random Forest
# model uses the same tree architecture.  There is NO hyperparameter search in
# this trial.  For binary classifiers only, the probability threshold is still
# selected on the validation set because F1 is the principal classification
# metric and the positive class is imbalanced.
FIXED_RF_PARAMS = {
    "n_estimators": 700,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

FIXED_RF_CLASSIFIER_PARAMS = {
    **FIXED_RF_PARAMS,
    "class_weight": "balanced",
}


def fixed_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[RandomForestClassifier, Dict[str, object]]:
    model = RandomForestClassifier(**FIXED_RF_CLASSIFIER_PARAMS)
    model.fit(X_train, y_train)
    return model, dict(FIXED_RF_CLASSIFIER_PARAMS)


def fixed_binary_classifier_with_threshold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[RandomForestClassifier, Dict[str, object], float]:
    """
    Train the fixed RF configuration and select only the binary probability
    threshold on the validation set. The test set is never used here.
    """
    model = RandomForestClassifier(**FIXED_RF_CLASSIFIER_PARAMS)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise RuntimeError("Binary classifier did not return two-class probabilities.")

    p1 = proba[:, 1]
    thresholds = np.round(np.arange(0.05, 0.501, 0.01), 2)
    best_threshold = 0.50
    best_f1 = -np.inf
    best_recall = -np.inf

    for threshold in thresholds:
        pred = (p1 >= threshold).astype(int)
        score = f1_score(y_val, pred, average="binary", zero_division=0)
        rec = recall_score(y_val, pred, average="binary", zero_division=0)
        if (score > best_f1) or (np.isclose(score, best_f1) and rec > best_recall):
            best_f1 = float(score)
            best_recall = float(rec)
            best_threshold = float(threshold)

    return model, dict(FIXED_RF_CLASSIFIER_PARAMS), best_threshold


def fixed_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[RandomForestRegressor, Dict[str, object]]:
    model = RandomForestRegressor(**FIXED_RF_PARAMS)
    model.fit(X_train, y_train)
    return model, dict(FIXED_RF_PARAMS)


# -----------------------------------------------------------------------------
# Historical-analogue prospective horizon helper
# -----------------------------------------------------------------------------
def _update_calendar_fields(row: pd.Series, future_date: pd.Timestamp) -> pd.Series:
    row = row.copy()
    row["timestamp"] = future_date
    if "date" in row.index:
        row["date"] = future_date.normalize().date().isoformat()
    replacements = {
        "year": future_date.year,
        "month": future_date.month,
        "day_of_year": future_date.dayofyear,
        "day_of_week": future_date.weekday(),
        "week_of_year": int(future_date.isocalendar().week),
        "month_drought": future_date.month,
        "day_of_week_drought": future_date.weekday(),
        "month_dust": future_date.month,
        "day_of_week_dust": future_date.weekday(),
    }
    for col, value in replacements.items():
        if col in row.index:
            row[col] = value
    return row


def build_future_by_historical_sampling(
    df: pd.DataFrame,
    days_ahead: int,
    target_cols_to_clear: List[str],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    work = df.sort_values(["city", "timestamp"]).copy()

    for city, group in work.groupby("city"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        last_date = pd.to_datetime(group["timestamp"].max())
        pool = group[group["timestamp"] < last_date].copy()
        if pool.empty:
            pool = group.copy()

        for step in range(1, days_ahead + 1):
            future_date = last_date + timedelta(days=step)
            candidates = pool[
                (pool["timestamp"].dt.month == future_date.month)
                & (pool["timestamp"].dt.day == future_date.day)
            ]
            if candidates.empty:
                candidates = pool[pool["timestamp"].dt.month == future_date.month]
            if candidates.empty:
                candidates = pool

            sampled = candidates.loc[rng.choice(candidates.index.to_numpy())].copy()
            sampled = _update_calendar_fields(sampled, future_date)
            for c in target_cols_to_clear:
                if c in sampled.index:
                    sampled[c] = np.nan
            rows.append(sampled)

    return pd.DataFrame(rows).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Drought + precipitation training
# -----------------------------------------------------------------------------
@dataclass
class DroughtBundle:
    flag_model: RandomForestClassifier
    severity_model: RandomForestClassifier
    precip_model: RandomForestRegressor
    flag_params: Dict[str, object]
    severity_params: Dict[str, object]
    precip_params: Dict[str, object]
    drought_features: List[str]
    precip_features: List[str]
    severity_encoding: Dict[str, int]
    severity_decoding: Dict[int, str]
    climatology: Dict[Tuple[str, int], float]
    metrics_rows: List[Dict[str, object]]


def compute_climatology(df: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    work = df.copy()
    work["doy"] = work["timestamp"].dt.dayofyear
    clim = work.groupby(["city", "doy"])["precipitation_sum"].mean().reset_index()
    return {(r["city"], int(r["doy"])): float(r["precipitation_sum"]) for _, r in clim.iterrows()}


def train_drought_and_precip(df: pd.DataFrame) -> DroughtBundle:
    drought_features = select_drought_features(df)
    precip_features = select_precip_features(df)

    if not drought_features or not precip_features:
        raise ValueError("No usable drought/precipitation features after leakage exclusions.")

    metrics_rows: List[Dict[str, object]] = []

    # ---------------- Drought classification ----------------
    target_df = df.dropna(subset=["drought_flag", "drought_severity"]).copy()
    target_df = target_df.sort_values(["timestamp", "city"]).reset_index(drop=True)
    tr, va, te = chronological_70_15_15(target_df)

    labels_order = [x for x in ["none", "moderate", "severe", "extreme"] if x in target_df["drought_severity"].astype(str).unique()]
    severity_encoding = {lab: i for i, lab in enumerate(labels_order)}
    severity_decoding = {i: lab for lab, i in severity_encoding.items()}

    # Training-only imputation reference.
    Xtr = sanitize_frame(tr, drought_features, tr)
    Xva = sanitize_frame(va, drought_features, tr)
    Xte = sanitize_frame(te, drought_features, tr)

    y_flag_tr = tr["drought_flag"].astype(int)
    y_flag_va = va["drought_flag"].astype(int)
    y_flag_te = te["drought_flag"].astype(int)

    # Binary drought_flag: select both RF hyperparameters and probability
    # threshold on validation data only.
    _, flag_params, flag_threshold = fixed_binary_classifier_with_threshold(
        Xtr, y_flag_tr, Xva, y_flag_va
    )

    # Refit the selected configuration on TRAIN + VALIDATION, then evaluate
    # exactly once on the untouched TEST set.
    train_val = pd.concat([tr, va], ignore_index=True).sort_values(["timestamp", "city"])
    X_train_val = sanitize_frame(train_val, drought_features, train_val)
    y_flag_train_val = train_val["drought_flag"].astype(int)

    flag_model = RandomForestClassifier(**FIXED_RF_CLASSIFIER_PARAMS)
    flag_model.fit(X_train_val, y_flag_train_val)

    counts = {
        "train": y_flag_tr.value_counts().sort_index().to_dict(),
        "validation": y_flag_va.value_counts().sort_index().to_dict(),
        "test": y_flag_te.value_counts().sort_index().to_dict(),
    }

    # Validation metrics are produced with the TRAIN-only tuned model, because
    # validation is the selection set.
    flag_selection_model, _, _ = fixed_binary_classifier_with_threshold(
        Xtr, y_flag_tr, Xva, y_flag_va
    )
    val_proba = flag_selection_model.predict_proba(Xva)
    val_pred = (val_proba[:, 1] >= flag_threshold).astype(int)
    metrics_rows.append(
        classifier_metric_row(
            model_name="drought_flag_random_forest_classifier",
            target_name="drought_flag",
            split="validation",
            city="ALL",
            y_true=y_flag_va,
            y_pred=val_pred,
            y_proba=val_proba,
            labels=[0, 1],
            average="binary",
            feature_cols=drought_features,
            params=flag_params,
            train_n=len(tr), validation_n=len(va), test_n=len(te),
            class_counts=counts,
            decision_threshold=flag_threshold,
        )
    )

    test_proba = flag_model.predict_proba(Xte)
    test_pred = (test_proba[:, 1] >= flag_threshold).astype(int)
    metrics_rows.append(
        classifier_metric_row(
            model_name="drought_flag_random_forest_classifier",
            target_name="drought_flag",
            split="test",
            city="ALL",
            y_true=y_flag_te,
            y_pred=test_pred,
            y_proba=test_proba,
            labels=[0, 1],
            average="binary",
            feature_cols=drought_features,
            params=flag_params,
            train_n=len(tr), validation_n=len(va), test_n=len(te),
            class_counts=counts,
            decision_threshold=flag_threshold,
        )
    )

    y_sev_tr = tr["drought_severity"].astype(str).map(severity_encoding).astype(int)
    y_sev_va = va["drought_severity"].astype(str).map(severity_encoding).astype(int)
    y_sev_te = te["drought_severity"].astype(str).map(severity_encoding).astype(int)

    # Multiclass drought severity: select hyperparameters on validation,
    # then refit on train + validation and evaluate on untouched test.
    _, sev_params = fixed_classifier(Xtr, y_sev_tr)
    sev_labels = sorted(severity_decoding.keys())

    val_model = RandomForestClassifier(**FIXED_RF_CLASSIFIER_PARAMS)
    val_model.fit(Xtr, y_sev_tr)
    val_pred = val_model.predict(Xva)
    val_proba = val_model.predict_proba(Xva)

    sev_counts = {
        "train": y_sev_tr.value_counts().sort_index().to_dict(),
        "validation": y_sev_va.value_counts().sort_index().to_dict(),
        "test": y_sev_te.value_counts().sort_index().to_dict(),
    }
    metrics_rows.append(
        classifier_metric_row(
            model_name="drought_severity_random_forest_classifier",
            target_name="drought_severity",
            split="validation",
            city="ALL",
            y_true=y_sev_va,
            y_pred=val_pred,
            y_proba=val_proba,
            labels=sev_labels,
            average="weighted",
            feature_cols=drought_features,
            params=sev_params,
            train_n=len(tr), validation_n=len(va), test_n=len(te),
            class_counts=sev_counts,
        )
    )

    y_sev_train_val = train_val["drought_severity"].astype(str).map(severity_encoding).astype(int)
    sev_model = RandomForestClassifier(**FIXED_RF_CLASSIFIER_PARAMS)
    sev_model.fit(X_train_val, y_sev_train_val)
    test_pred = sev_model.predict(Xte)
    test_proba = sev_model.predict_proba(Xte)
    metrics_rows.append(
        classifier_metric_row(
            model_name="drought_severity_random_forest_classifier",
            target_name="drought_severity",
            split="test",
            city="ALL",
            y_true=y_sev_te,
            y_pred=test_pred,
            y_proba=test_proba,
            labels=sev_labels,
            average="weighted",
            feature_cols=drought_features,
            params=sev_params,
            train_n=len(tr), validation_n=len(va), test_n=len(te),
            class_counts=sev_counts,
        )
    )

    # ---------------- Precipitation regression ----------------
    p_df = df.dropna(subset=["precipitation_sum"]).copy().sort_values(["timestamp", "city"]).reset_index(drop=True)
    ptr, pva, pte = chronological_70_15_15(p_df)
    PXtr = sanitize_frame(ptr, precip_features, ptr)
    PXva = sanitize_frame(pva, precip_features, ptr)
    PXte = sanitize_frame(pte, precip_features, ptr)
    pytr = pd.to_numeric(ptr["precipitation_sum"], errors="coerce").astype(float)
    pyva = pd.to_numeric(pva["precipitation_sum"], errors="coerce").astype(float)
    pyte = pd.to_numeric(pte["precipitation_sum"], errors="coerce").astype(float)

    # Select regression hyperparameters on validation only.
    val_precip_model, precip_params = fixed_regressor(PXtr, pytr)

    val_pred = val_precip_model.predict(PXva)
    metrics_rows.append(
        regressor_metric_row(
            model_name="precipitation_sum_random_forest_regressor",
            target_name="precipitation_sum",
            split="validation",
            city="ALL",
            y_true=pyva,
            y_pred=val_pred,
            feature_cols=precip_features,
            params=precip_params,
            train_n=len(ptr), validation_n=len(pva), test_n=len(pte),
        )
    )

    # Refit selected configuration on train + validation, evaluate untouched test.
    p_train_val = pd.concat([ptr, pva], ignore_index=True).sort_values(["timestamp", "city"])
    PX_train_val = sanitize_frame(p_train_val, precip_features, p_train_val)
    py_train_val = pd.to_numeric(p_train_val["precipitation_sum"], errors="coerce").astype(float)

    precip_model = RandomForestRegressor(**FIXED_RF_PARAMS)
    precip_model.fit(PX_train_val, py_train_val)
    test_pred = precip_model.predict(PXte)
    metrics_rows.append(
        regressor_metric_row(
            model_name="precipitation_sum_random_forest_regressor",
            target_name="precipitation_sum",
            split="test",
            city="ALL",
            y_true=pyte,
            y_pred=test_pred,
            feature_cols=precip_features,
            params=precip_params,
            train_n=len(ptr), validation_n=len(pva), test_n=len(pte),
        )
    )

    return DroughtBundle(
        flag_model=flag_model,
        severity_model=sev_model,
        precip_model=precip_model,
        flag_params=flag_params,
        severity_params=sev_params,
        precip_params=precip_params,
        drought_features=drought_features,
        precip_features=precip_features,
        severity_encoding=severity_encoding,
        severity_decoding=severity_decoding,
        climatology=compute_climatology(df),
        metrics_rows=metrics_rows,
    )


def predict_drought_future(df: pd.DataFrame, bundle: DroughtBundle, days: int) -> pd.DataFrame:
    horizon = build_future_by_historical_sampling(
        df,
        days_ahead=days,
        target_cols_to_clear=["drought_flag", "drought_severity", "drought_severity_code"],
    )

    # For future imputation, use historical data as the reference (not future rows).
    Xd = sanitize_frame(horizon, bundle.drought_features, df)
    Xp = sanitize_frame(horizon, bundle.precip_features, df)

    flag = bundle.flag_model.predict(Xd).astype(int)
    sev_code = bundle.severity_model.predict(Xd).astype(int)
    sev = [bundle.severity_decoding.get(int(x), "unknown") for x in sev_code]
    precip = bundle.precip_model.predict(Xp)

    out = pd.DataFrame({
        "city": horizon["city"].astype(str).values,
        "timestamp": pd.to_datetime(horizon["timestamp"]).values,
        "drought_flag_pred": flag,
        "drought_severity_pred": sev,
        "precipitation_sum_pred": precip,
    })

    normal, deficit = [], []
    for city, ts, pred in zip(out["city"], out["timestamp"], out["precipitation_sum_pred"]):
        key = (city, int(pd.to_datetime(ts).dayofyear))
        n = bundle.climatology.get(key, np.nan)
        normal.append(n)
        deficit.append(np.nan if pd.isna(n) else n - pred)
    out["precipitation_sum_normal"] = normal
    out["precip_deficit_pred"] = deficit

    out = out.sort_values(["city", "timestamp"]).reset_index(drop=True)
    out["drought_duration_pred"] = 0
    for city, idx in out.groupby("city").groups.items():
        ids = list(idx)
        flags = out.loc[ids, "drought_flag_pred"].to_numpy(dtype=int)
        durations = np.zeros(len(flags), dtype=int)
        for i in range(len(flags)):
            if flags[i] == 1:
                j = i
                while j < len(flags) and flags[j] == 1:
                    durations[i] += 1
                    j += 1
        out.loc[ids, "drought_duration_pred"] = durations
    return out


# -----------------------------------------------------------------------------
# Dust training / prediction per city
# -----------------------------------------------------------------------------
def train_and_predict_dust(df: pd.DataFrame, days: int) -> Tuple[pd.DataFrame, List[Dict[str, object]], List[str]]:
    target = auto_detect_dust_target(df)
    features = select_dust_features(df, target)
    if not features:
        raise ValueError("No usable dust features after leakage exclusions.")

    all_future = []
    metrics_rows = []

    for city in sorted(df["city"].dropna().astype(str).unique()):
        cdf = df[df["city"].astype(str) == city].dropna(subset=[target]).copy()
        cdf = cdf.sort_values("timestamp").reset_index(drop=True)
        if len(cdf) < 30:
            print(f"[WARN] {city}: too few rows ({len(cdf)}), skipping dust model.")
            continue

        tr, va, te = chronological_70_15_15(cdf)
        Xtr = sanitize_frame(tr, features, tr)
        Xva = sanitize_frame(va, features, tr)
        Xte = sanitize_frame(te, features, tr)
        ytr = tr[target].astype(int)
        yva = va[target].astype(int)
        yte = te[target].astype(int)

        counts = {
            "train": ytr.value_counts().sort_index().to_dict(),
            "validation": yva.value_counts().sort_index().to_dict(),
            "test": yte.value_counts().sort_index().to_dict(),
        }
        print(f"\n[INFO] Dust class counts for {city}: {counts}")

        if ytr.nunique() < 2:
            print(f"[WARN] {city}: training subset contains only one dust class; skipping.")
            continue

        # Jointly tune RF hyperparameters and the classification threshold
        # using validation data only. This avoids the pathological behaviour
        # of a fixed 0.50 threshold predicting no dust events.
        selection_model, params, threshold = fixed_binary_classifier_with_threshold(
            Xtr, ytr, Xva, yva
        )
        labels = [0, 1]

        val_proba = selection_model.predict_proba(Xva)
        val_pred = (val_proba[:, 1] >= threshold).astype(int)
        val_row = classifier_metric_row(
            model_name=f"dust_random_forest_classifier_{city.lower().replace(' ', '_')}",
            target_name=target,
            split="validation",
            city=city,
            y_true=yva,
            y_pred=val_pred,
            y_proba=val_proba,
            labels=labels,
            average="binary",
            feature_cols=features,
            params=params,
            train_n=len(tr), validation_n=len(va), test_n=len(te),
            class_counts=counts,
            decision_threshold=threshold,
        )
        metrics_rows.append(val_row)

        # After model/threshold selection, refit the chosen RF hyperparameters
        # on train + validation. The test set remains untouched.
        train_val = pd.concat([tr, va], ignore_index=True).sort_values("timestamp")
        X_train_val = sanitize_frame(train_val, features, train_val)
        y_train_val = train_val[target].astype(int)
        final_test_model = RandomForestClassifier(**FIXED_RF_CLASSIFIER_PARAMS)
        final_test_model.fit(X_train_val, y_train_val)

        test_proba = final_test_model.predict_proba(Xte)
        test_pred = (test_proba[:, 1] >= threshold).astype(int)
        test_row = classifier_metric_row(
            model_name=f"dust_random_forest_classifier_{city.lower().replace(' ', '_')}",
            target_name=target,
            split="test",
            city=city,
            y_true=yte,
            y_pred=test_pred,
            y_proba=test_proba,
            labels=labels,
            average="binary",
            feature_cols=features,
            params=params,
            train_n=len(tr), validation_n=len(va), test_n=len(te),
            class_counts=counts,
            decision_threshold=threshold,
        )
        metrics_rows.append(test_row)

        for row in (val_row, test_row):
            print(
                f"[INFO] {city} {row['split']}: threshold={threshold:.2f}, "
                f"F1={row['f1_score']:.4f}, Precision={row['precision']:.4f}, "
                f"Recall={row['recall']:.4f}, TP={row['tp']}, TN={row['tn']}, "
                f"FP={row['fp']}, FN={row['fn']}"
            )

        # Refit with the chosen hyperparameters on train + validation for the
        # prospective horizon.  The final test metrics above remain untouched.
        train_val = pd.concat([tr, va], ignore_index=True).sort_values("timestamp")
        X_train_val = sanitize_frame(train_val, features, train_val)
        y_train_val = train_val[target].astype(int)
        final_model = RandomForestClassifier(**FIXED_RF_CLASSIFIER_PARAMS)
        final_model.fit(X_train_val, y_train_val)

        horizon = build_future_by_historical_sampling(cdf, days, [target])
        Xf = sanitize_frame(horizon, features, train_val)
        proba = final_model.predict_proba(Xf)
        p1 = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.full(len(Xf), np.nan)
        pred = (p1 >= threshold).astype(int)

        for i, hrow in horizon.iterrows():
            rec = {
                "city": city,
                "timestamp": hrow["timestamp"],
                "dust_event_pred": int(pred[i]),
                "dust_event_prob": float(p1[i]) if not pd.isna(p1[i]) else np.nan,
            }
            for c in ["dust_intensity_level", "dust_intensity_code", "pm10", "pm25", "aod", "temp_mean", "wind_speed_mean"]:
                if c in hrow.index:
                    rec[c] = hrow[c]
            all_future.append(rec)

    if not all_future:
        raise RuntimeError("No dust future records were generated.")
    return pd.DataFrame(all_future).sort_values(["city", "timestamp"]).reset_index(drop=True), metrics_rows, features


# -----------------------------------------------------------------------------
# Feature audit export
# -----------------------------------------------------------------------------
def build_feature_audit(df: pd.DataFrame, drought_features: List[str], precip_features: List[str], dust_features: List[str]) -> pd.DataFrame:
    rows = []
    numeric = set(numeric_columns(df))
    for target, used, direct_exclusions in [
        ("drought_flag_and_severity", set(drought_features), COMMON_EXCLUDE | DROUGHT_DIRECT_LEAKAGE),
        ("precipitation_sum", set(precip_features), COMMON_EXCLUDE | PRECIP_DIRECT_LEAKAGE),
        ("dust_event", set(dust_features), COMMON_EXCLUDE | DUST_DIRECT_LEAKAGE),
    ]:
        for col in df.columns:
            if col in used:
                status = "USED"
                reason = "Selected numeric predictor"
            elif col not in numeric:
                status = "EXCLUDED"
                reason = "Non-numeric / identifier / categorical text"
            else:
                status = "EXCLUDED"
                if col in direct_exclusions:
                    reason = "Leakage/target/source exclusion"
                elif target == "dust_event" and any(col.startswith(p) for p in DUST_CURRENT_ROLLING_PREFIXES):
                    reason = "Current-row rolling proxy exclusion"
                else:
                    reason = "Target/cross-target/other controlled exclusion"
            rows.append({"target": target, "column": col, "status": status, "reason": reason})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def resolve_same_directory(path_text: str, default_path: Path) -> Path:
    if not path_text:
        return default_path
    p = Path(path_text).expanduser()
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest Trial 1: fixed 700 trees / depth 20 local experiment")
    parser.add_argument("--input", default="LATEST.parquet", help="Dataset file in this script directory (default: LATEST.parquet)")
    parser.add_argument("--output", default=DEFAULT_PREDICTIONS.name, help="Predictions CSV written in script directory")
    parser.add_argument("--metrics_output", default=DEFAULT_METRICS.name, help="Metrics CSV written in script directory")
    parser.add_argument("--feature_audit_output", default=DEFAULT_FEATURE_AUDIT.name, help="Feature audit CSV written in script directory")
    parser.add_argument("--horizon_days", type=int, default=30)
    args = parser.parse_args()

    input_path = resolve_same_directory(args.input, DEFAULT_INPUT)
    pred_path = resolve_same_directory(args.output, DEFAULT_PREDICTIONS)
    metrics_path = resolve_same_directory(args.metrics_output, DEFAULT_METRICS)
    audit_path = resolve_same_directory(args.feature_audit_output, DEFAULT_FEATURE_AUDIT)

    print(f"[INFO] Script directory : {SCRIPT_DIR}")
    print(f"[INFO] Input dataset    : {input_path}")
    print(f"[INFO] Predictions out  : {pred_path}")
    print(f"[INFO] Metrics out      : {metrics_path}")
    print(f"[INFO] Feature audit out: {audit_path}")
    print("[INFO] Split            : chronological 70% train / 15% validation / 15% test")
    print(f"[INFO] Trial 1 RF params : {FIXED_RF_PARAMS}")
    print("[INFO] Class balancing  : balanced for classification")
    print("[INFO] Binary threshold : selected on validation F1 only")

    df = load_dataset(input_path)
    print(f"[INFO] Dataset shape    : {df.shape}")

    print("\n[INFO] Training drought + precipitation Random Forest models...")
    drought_bundle = train_drought_and_precip(df)

    print("\n[INFO] Building prospective drought/precipitation horizon...")
    drought_future = predict_drought_future(df, drought_bundle, args.horizon_days)

    print("\n[INFO] Training/evaluating per-city dust Random Forest models...")
    dust_future, dust_metrics, dust_features = train_and_predict_dust(df, args.horizon_days)

    merged = drought_future.merge(dust_future, on=["city", "timestamp"], how="outer", sort=True)
    merged = merged.sort_values(["city", "timestamp"]).reset_index(drop=True)

    metrics_rows = drought_bundle.metrics_rows + dust_metrics
    metrics = pd.DataFrame(metrics_rows)

    preferred_cols = [
        "model_name", "model_type", "target_name", "city", "split",
        "n_samples", "train_n", "validation_n", "test_n",
        "accuracy", "precision", "recall", "f1_score", "auc_roc", "loss",
        "r2_score", "mse", "rmse", "mae",
        "tn", "fp", "fn", "tp",
        "confusion_matrix", "classification_report",
        "n_features", "feature_list", "selected_hyperparameters", "class_counts", "decision_threshold",
    ]
    for c in preferred_cols:
        if c not in metrics.columns:
            metrics[c] = None
    metrics = metrics[preferred_cols]

    audit = build_feature_audit(
        df,
        drought_features=drought_bundle.drought_features,
        precip_features=drought_bundle.precip_features,
        dust_features=dust_features,
    )

    merged.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    audit.to_csv(audit_path, index=False)

    print("\n[INFO] Finished successfully.")
    print(f"[INFO] Saved predictions : {pred_path}")
    print(f"[INFO] Saved metrics     : {metrics_path}")
    print(f"[INFO] Saved feature audit: {audit_path}")

    print("\n[INFO] FINAL TEST rows to report in the paper:")
    display_cols = ["target_name", "city", "split", "accuracy", "precision", "recall", "f1_score", "rmse", "r2_score", "tn", "fp", "fn", "tp"]
    print(metrics.loc[metrics["split"] == "test", display_cols].to_string(index=False))

    print("\n[INFO] VALIDATION rows (used for model/hyperparameter selection):")
    print(metrics.loc[metrics["split"] == "validation", display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
