#!/usr/bin/env python3
"""
XGBoost Trial 4 - reviewer revision, LOCAL execution.

Place this script in the SAME directory as LATEST.parquet.

Shared XGBoost hyperparameters for all targets:
    n_estimators=600
    max_depth=10
    learning_rate=0.05
    subsample=0.8
    colsample_bytree=0.8
    reg_lambda=2.0
    reg_alpha=0.0
    min_child_weight=3.0
    gamma=0.0
    random_state=42
    n_jobs=-1
    tree_method="hist"

Task-specific settings that cannot be identical:
    Binary classification: objective="binary:logistic", eval_metric="logloss"
    Multiclass classification: objective="multi:softprob", eval_metric="mlogloss"
    Regression: objective="reg:squarederror", eval_metric="rmse"

Reviewer-revision methodology:
    - chronological 70% train / 15% validation / 15% final test
    - no shuffling
    - leakage-sensitive features excluded
    - class balancing for classifiers using training-set sample weights
    - binary decision threshold selected ONLY on validation F1
    - final test kept separate
    - validation + test metrics saved
    - TP/TN/FP/FN and confusion matrices saved
    - feature audit saved
    - 30-day output is a prospective horizon, not future-validated ground truth
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
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError as e:
    raise ImportError(
        "xgboost is required. Install it with: python -m pip install xgboost"
    ) from e


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "LATEST.parquet"
DEFAULT_PREDICTIONS = SCRIPT_DIR / "xgb_trial4_predictions.csv"
DEFAULT_METRICS = SCRIPT_DIR / "xgb_trial4_metrics.csv"
DEFAULT_FEATURE_AUDIT = SCRIPT_DIR / "xgb_trial4_feature_audit.csv"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# -------------------------------------------------------------------------
# Shared XGBoost hyperparameters: SAME across all XGBoost target models.
# -------------------------------------------------------------------------
XGB_SHARED_PARAMS = {
    "n_estimators": 600,
    "max_depth": 10,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.0,
    "min_child_weight": 3.0,
    "gamma": 0.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}

# -------------------------------------------------------------------------
# Leakage control
# -------------------------------------------------------------------------
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

DROUGHT_DIRECT_LEAKAGE = {
    "drought_flag",
    "drought_severity_code",
    "precip_30d_sum",
    "soil_moisture_rootzone_30d_mean",
    "vpd_30d_mean",
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

DUST_DIRECT_LEAKAGE = {
    "dust_event",
    "dust_event_x",
    "dust_intensity_code",
    "dust_intensity_level",
    "duration_hours",
    "pm25",
    "pm10",
    "aod",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_speed_mean",
}

DUST_CURRENT_ROLLING_PREFIXES = (
    "pm25_roll",
    "pm10_roll",
    "aod_roll",
    "wind_speed_10m_roll",
    "wind_gusts_10m_roll",
    "wind_speed_mean_roll",
)

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
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
        raise ValueError("Dataset must contain 'timestamp' and 'city' columns.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["city", "timestamp"]).copy()
    return df.sort_values(["timestamp", "city"]).reset_index(drop=True)


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def sanitize_frame(
    df: pd.DataFrame,
    feature_cols: List[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Numeric conversion + imputation using the reference (training) subset only."""
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


def chronological_70_15_15(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.sort_values(["timestamp", "city"]).reset_index(drop=True).copy()
    unique_dates = np.array(
        sorted(pd.Series(work["timestamp"].dt.normalize().unique()).tolist())
    )

    if len(unique_dates) < 10:
        raise ValueError("Too few unique dates for 70/15/15 chronological split.")

    train_end = max(1, int(np.floor(len(unique_dates) * TRAIN_RATIO)))
    val_end = max(
        train_end + 1,
        int(np.floor(len(unique_dates) * (TRAIN_RATIO + VALIDATION_RATIO))),
    )
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
            f"Empty split: train={len(train)}, validation={len(val)}, test={len(test)}"
        )
    return train, val, test


def auto_detect_dust_target(df: pd.DataFrame) -> str:
    if "dust_event" in df.columns:
        return "dust_event"
    if "dust_event_x" in df.columns:
        return "dust_event_x"
    raise ValueError("Dust target not found.")


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
    for col in numeric_columns(df):
        if col in exclude:
            continue
        if any(col.startswith(prefix) for prefix in DUST_CURRENT_ROLLING_PREFIXES):
            continue
        features.append(col)
    return features


# -------------------------------------------------------------------------
# XGBoost model constructors
# -------------------------------------------------------------------------
def make_binary_classifier() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        **XGB_SHARED_PARAMS,
    )


def make_multiclass_classifier(num_classes: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=num_classes,
        **XGB_SHARED_PARAMS,
    )


def make_regressor() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        **XGB_SHARED_PARAMS,
    )


# -------------------------------------------------------------------------
# Threshold selection: VALIDATION SET ONLY
# -------------------------------------------------------------------------
def select_best_binary_threshold(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
) -> Tuple[float, float]:
    best_threshold = 0.50
    best_f1 = -1.0
    best_recall = -1.0

    # Fine but controlled grid.
    thresholds = np.arange(0.05, 0.501, 0.01)
    for threshold in thresholds:
        pred = (positive_probability >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        if (score > best_f1) or (
            np.isclose(score, best_f1) and rec > best_recall
        ):
            best_f1 = float(score)
            best_recall = float(rec)
            best_threshold = float(round(threshold, 2))

    return best_threshold, best_f1


# -------------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------------
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
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

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
        "precision": safe_float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": safe_float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1_score": safe_float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        ),
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
            classification_report(
                y_true,
                y_pred,
                labels=labels,
                zero_division=0,
                output_dict=True,
            )
        ),
        "n_features": len(feature_cols),
        "feature_list": safe_json(feature_cols),
        "selected_hyperparameters": safe_json(XGB_SHARED_PARAMS),
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
            if len(labels) == 2 and proba.ndim == 2 and proba.shape[1] >= 2:
                if len(np.unique(y_true)) == 2:
                    row["auc_roc"] = safe_float(roc_auc_score(y_true, proba[:, 1]))
            elif len(labels) > 2 and proba.ndim == 2:
                if len(np.unique(y_true)) > 1:
                    row["auc_roc"] = safe_float(
                        roc_auc_score(
                            y_true, proba, multi_class="ovr", average="weighted"
                        )
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
        "selected_hyperparameters": safe_json(XGB_SHARED_PARAMS),
        "class_counts": None,
        "decision_threshold": None,
    }


# -------------------------------------------------------------------------
# Future-horizon helpers
# -------------------------------------------------------------------------
def _update_calendar_fields(
    row: pd.Series,
    future_date: pd.Timestamp,
) -> pd.Series:
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

            sampled = candidates.loc[
                rng.choice(candidates.index.to_numpy())
            ].copy()
            sampled = _update_calendar_fields(sampled, future_date)

            for col in target_cols_to_clear:
                if col in sampled.index:
                    sampled[col] = np.nan

            rows.append(sampled)

    if not rows:
        raise RuntimeError("No future records were generated.")

    return pd.DataFrame(rows).reset_index(drop=True)


# -------------------------------------------------------------------------
# Drought + precipitation
# -------------------------------------------------------------------------
@dataclass
class DroughtBundle:
    flag_model: XGBClassifier
    severity_model: XGBClassifier
    precip_model: XGBRegressor
    flag_threshold: float
    drought_features: List[str]
    precip_features: List[str]
    severity_encoding: Dict[str, int]
    severity_decoding: Dict[int, str]
    climatology: Dict[Tuple[str, int], float]
    metrics_rows: List[Dict[str, object]]


def compute_climatology(df: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    work = df.copy()
    work["doy"] = work["timestamp"].dt.dayofyear
    clim = (
        work.groupby(["city", "doy"])["precipitation_sum"]
        .mean()
        .reset_index()
    )
    return {
        (row["city"], int(row["doy"])): float(row["precipitation_sum"])
        for _, row in clim.iterrows()
    }


def train_drought_and_precip(df: pd.DataFrame) -> DroughtBundle:
    drought_features = select_drought_features(df)
    precip_features = select_precip_features(df)

    target_df = (
        df.dropna(subset=["drought_flag", "drought_severity"])
        .sort_values(["timestamp", "city"])
        .reset_index(drop=True)
    )
    tr, va, te = chronological_70_15_15(target_df)

    Xtr = sanitize_frame(tr, drought_features, tr)
    Xva = sanitize_frame(va, drought_features, tr)
    Xte = sanitize_frame(te, drought_features, tr)

    metrics_rows: List[Dict[str, object]] = []

    # ----- drought flag -----
    ytr = tr["drought_flag"].astype(int)
    yva = va["drought_flag"].astype(int)
    yte = te["drought_flag"].astype(int)

    sample_weight = compute_sample_weight(class_weight="balanced", y=ytr)
    model_flag = make_binary_classifier()
    model_flag.fit(Xtr, ytr, sample_weight=sample_weight)

    val_proba = model_flag.predict_proba(Xva)
    flag_threshold, _ = select_best_binary_threshold(
        yva.to_numpy(), val_proba[:, 1]
    )
    val_pred = (val_proba[:, 1] >= flag_threshold).astype(int)

    counts = {
        "train": ytr.value_counts().sort_index().to_dict(),
        "validation": yva.value_counts().sort_index().to_dict(),
        "test": yte.value_counts().sort_index().to_dict(),
    }
    metrics_rows.append(
        classifier_metric_row(
            model_name="drought_flag_xgboost_classifier",
            target_name="drought_flag",
            split="validation",
            city="ALL",
            y_true=yva,
            y_pred=val_pred,
            y_proba=val_proba,
            labels=[0, 1],
            average="binary",
            feature_cols=drought_features,
            train_n=len(tr),
            validation_n=len(va),
            test_n=len(te),
            class_counts=counts,
            decision_threshold=flag_threshold,
        )
    )

    # Refit on train+validation using same fixed hyperparameters.
    train_val = pd.concat([tr, va], ignore_index=True).sort_values(["timestamp", "city"])
    Xtv = sanitize_frame(train_val, drought_features, train_val)
    ytv = train_val["drought_flag"].astype(int)
    tv_weight = compute_sample_weight(class_weight="balanced", y=ytv)

    final_flag_model = make_binary_classifier()
    final_flag_model.fit(Xtv, ytv, sample_weight=tv_weight)
    Xte_final = sanitize_frame(te, drought_features, train_val)
    test_proba = final_flag_model.predict_proba(Xte_final)
    test_pred = (test_proba[:, 1] >= flag_threshold).astype(int)

    metrics_rows.append(
        classifier_metric_row(
            model_name="drought_flag_xgboost_classifier",
            target_name="drought_flag",
            split="test",
            city="ALL",
            y_true=yte,
            y_pred=test_pred,
            y_proba=test_proba,
            labels=[0, 1],
            average="binary",
            feature_cols=drought_features,
            train_n=len(tr),
            validation_n=len(va),
            test_n=len(te),
            class_counts=counts,
            decision_threshold=flag_threshold,
        )
    )

    # ----- drought severity -----
    labels_order = [
        x
        for x in ["none", "moderate", "severe", "extreme"]
        if x in target_df["drought_severity"].astype(str).unique()
    ]
    severity_encoding = {lab: i for i, lab in enumerate(labels_order)}
    severity_decoding = {i: lab for lab, i in severity_encoding.items()}

    sytr = tr["drought_severity"].astype(str).map(severity_encoding).astype(int)
    syva = va["drought_severity"].astype(str).map(severity_encoding).astype(int)
    syte = te["drought_severity"].astype(str).map(severity_encoding).astype(int)
    sev_labels = sorted(severity_decoding.keys())

    sev_weight = compute_sample_weight(class_weight="balanced", y=sytr)
    sev_model = make_multiclass_classifier(len(sev_labels))
    sev_model.fit(Xtr, sytr, sample_weight=sev_weight)

    sev_val_proba = sev_model.predict_proba(Xva)
    sev_val_pred = np.argmax(sev_val_proba, axis=1)

    sev_counts = {
        "train": sytr.value_counts().sort_index().to_dict(),
        "validation": syva.value_counts().sort_index().to_dict(),
        "test": syte.value_counts().sort_index().to_dict(),
    }
    metrics_rows.append(
        classifier_metric_row(
            model_name="drought_severity_xgboost_classifier",
            target_name="drought_severity",
            split="validation",
            city="ALL",
            y_true=syva,
            y_pred=sev_val_pred,
            y_proba=sev_val_proba,
            labels=sev_labels,
            average="weighted",
            feature_cols=drought_features,
            train_n=len(tr),
            validation_n=len(va),
            test_n=len(te),
            class_counts=sev_counts,
        )
    )

    sy_tv = train_val["drought_severity"].astype(str).map(severity_encoding).astype(int)
    sev_tv_weight = compute_sample_weight(class_weight="balanced", y=sy_tv)
    final_sev_model = make_multiclass_classifier(len(sev_labels))
    final_sev_model.fit(Xtv, sy_tv, sample_weight=sev_tv_weight)
    sev_test_proba = final_sev_model.predict_proba(Xte_final)
    sev_test_pred = np.argmax(sev_test_proba, axis=1)

    metrics_rows.append(
        classifier_metric_row(
            model_name="drought_severity_xgboost_classifier",
            target_name="drought_severity",
            split="test",
            city="ALL",
            y_true=syte,
            y_pred=sev_test_pred,
            y_proba=sev_test_proba,
            labels=sev_labels,
            average="weighted",
            feature_cols=drought_features,
            train_n=len(tr),
            validation_n=len(va),
            test_n=len(te),
            class_counts=sev_counts,
        )
    )

    # ----- precipitation regression -----
    p_df = (
        df.dropna(subset=["precipitation_sum"])
        .sort_values(["timestamp", "city"])
        .reset_index(drop=True)
    )
    ptr, pva, pte = chronological_70_15_15(p_df)

    PXtr = sanitize_frame(ptr, precip_features, ptr)
    PXva = sanitize_frame(pva, precip_features, ptr)
    PXte = sanitize_frame(pte, precip_features, ptr)

    pytr = pd.to_numeric(ptr["precipitation_sum"], errors="coerce").astype(float)
    pyva = pd.to_numeric(pva["precipitation_sum"], errors="coerce").astype(float)
    pyte = pd.to_numeric(pte["precipitation_sum"], errors="coerce").astype(float)

    preg = make_regressor()
    preg.fit(PXtr, pytr)
    pval_pred = preg.predict(PXva)

    metrics_rows.append(
        regressor_metric_row(
            model_name="precipitation_sum_xgboost_regressor",
            target_name="precipitation_sum",
            split="validation",
            city="ALL",
            y_true=pyva,
            y_pred=pval_pred,
            feature_cols=precip_features,
            train_n=len(ptr),
            validation_n=len(pva),
            test_n=len(pte),
        )
    )

    p_train_val = pd.concat([ptr, pva], ignore_index=True).sort_values(["timestamp", "city"])
    Ptv = sanitize_frame(p_train_val, precip_features, p_train_val)
    pytv = pd.to_numeric(p_train_val["precipitation_sum"], errors="coerce").astype(float)
    final_preg = make_regressor()
    final_preg.fit(Ptv, pytv)
    PXte_final = sanitize_frame(pte, precip_features, p_train_val)
    ptest_pred = final_preg.predict(PXte_final)

    metrics_rows.append(
        regressor_metric_row(
            model_name="precipitation_sum_xgboost_regressor",
            target_name="precipitation_sum",
            split="test",
            city="ALL",
            y_true=pyte,
            y_pred=ptest_pred,
            feature_cols=precip_features,
            train_n=len(ptr),
            validation_n=len(pva),
            test_n=len(pte),
        )
    )

    return DroughtBundle(
        flag_model=final_flag_model,
        severity_model=final_sev_model,
        precip_model=final_preg,
        flag_threshold=flag_threshold,
        drought_features=drought_features,
        precip_features=precip_features,
        severity_encoding=severity_encoding,
        severity_decoding=severity_decoding,
        climatology=compute_climatology(df),
        metrics_rows=metrics_rows,
    )


def predict_drought_future(
    df: pd.DataFrame,
    bundle: DroughtBundle,
    days: int,
) -> pd.DataFrame:
    horizon = build_future_by_historical_sampling(
        df,
        days_ahead=days,
        target_cols_to_clear=[
            "drought_flag",
            "drought_severity",
            "drought_severity_code",
        ],
    )

    Xd = sanitize_frame(horizon, bundle.drought_features, df)
    Xp = sanitize_frame(horizon, bundle.precip_features, df)

    fproba = bundle.flag_model.predict_proba(Xd)
    flag = (fproba[:, 1] >= bundle.flag_threshold).astype(int)

    sev_code = bundle.severity_model.predict(Xd).astype(int)
    sev = [
        bundle.severity_decoding.get(int(x), "unknown")
        for x in sev_code
    ]
    precip = bundle.precip_model.predict(Xp)

    out = pd.DataFrame(
        {
            "city": horizon["city"].astype(str).values,
            "timestamp": pd.to_datetime(horizon["timestamp"]).values,
            "drought_flag_pred": flag,
            "drought_severity_pred": sev,
            "precipitation_sum_pred": precip,
        }
    )

    normal, deficit = [], []
    for city, ts, pred in zip(
        out["city"],
        out["timestamp"],
        out["precipitation_sum_pred"],
    ):
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


# -------------------------------------------------------------------------
# Dust per city
# -------------------------------------------------------------------------
def train_and_predict_dust(
    df: pd.DataFrame,
    days: int,
) -> Tuple[pd.DataFrame, List[Dict[str, object]], List[str]]:
    target = auto_detect_dust_target(df)
    features = select_dust_features(df, target)

    all_future = []
    metrics_rows = []

    for city in sorted(df["city"].dropna().astype(str).unique()):
        cdf = (
            df[df["city"].astype(str) == city]
            .dropna(subset=[target])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(cdf) < 30:
            print(f"[WARN] {city}: too few rows; skipping.")
            continue

        tr, va, te = chronological_70_15_15(cdf)

        Xtr = sanitize_frame(tr, features, tr)
        Xva = sanitize_frame(va, features, tr)

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
            print(f"[WARN] {city}: only one training class; skipping.")
            continue

        sw = compute_sample_weight(class_weight="balanced", y=ytr)
        model = make_binary_classifier()
        model.fit(Xtr, ytr, sample_weight=sw)

        val_proba = model.predict_proba(Xva)
        threshold, _ = select_best_binary_threshold(
            yva.to_numpy(), val_proba[:, 1]
        )
        val_pred = (val_proba[:, 1] >= threshold).astype(int)

        val_row = classifier_metric_row(
            model_name=f"dust_xgboost_classifier_{city.lower().replace(' ', '_')}",
            target_name=target,
            split="validation",
            city=city,
            y_true=yva,
            y_pred=val_pred,
            y_proba=val_proba,
            labels=[0, 1],
            average="binary",
            feature_cols=features,
            train_n=len(tr),
            validation_n=len(va),
            test_n=len(te),
            class_counts=counts,
            decision_threshold=threshold,
        )
        metrics_rows.append(val_row)

        # Refit on train + validation with the same shared hyperparameters.
        train_val = (
            pd.concat([tr, va], ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        Xtv = sanitize_frame(train_val, features, train_val)
        ytv = train_val[target].astype(int)
        tv_sw = compute_sample_weight(class_weight="balanced", y=ytv)

        final_model = make_binary_classifier()
        final_model.fit(Xtv, ytv, sample_weight=tv_sw)

        Xte = sanitize_frame(te, features, train_val)
        test_proba = final_model.predict_proba(Xte)
        test_pred = (test_proba[:, 1] >= threshold).astype(int)

        test_row = classifier_metric_row(
            model_name=f"dust_xgboost_classifier_{city.lower().replace(' ', '_')}",
            target_name=target,
            split="test",
            city=city,
            y_true=yte,
            y_pred=test_pred,
            y_proba=test_proba,
            labels=[0, 1],
            average="binary",
            feature_cols=features,
            train_n=len(tr),
            validation_n=len(va),
            test_n=len(te),
            class_counts=counts,
            decision_threshold=threshold,
        )
        metrics_rows.append(test_row)

        print(
            f"[INFO] {city} validation: threshold={threshold:.2f}, "
            f"F1={val_row['f1_score']:.4f}, Precision={val_row['precision']:.4f}, "
            f"Recall={val_row['recall']:.4f}"
        )
        print(
            f"[INFO] {city} test: threshold={threshold:.2f}, "
            f"F1={test_row['f1_score']:.4f}, Precision={test_row['precision']:.4f}, "
            f"Recall={test_row['recall']:.4f}, "
            f"TP={test_row['tp']}, TN={test_row['tn']}, "
            f"FP={test_row['fp']}, FN={test_row['fn']}"
        )

        horizon = build_future_by_historical_sampling(
            cdf,
            days,
            [target],
        )
        Xf = sanitize_frame(horizon, features, train_val)
        fproba = final_model.predict_proba(Xf)
        fpred = (fproba[:, 1] >= threshold).astype(int)

        for i, hrow in horizon.iterrows():
            rec = {
                "city": city,
                "timestamp": hrow["timestamp"],
                "dust_event_pred": int(fpred[i]),
                "dust_event_prob": float(fproba[i, 1]),
            }
            for col in [
                "dust_intensity_level",
                "dust_intensity_code",
                "pm10",
                "pm25",
                "aod",
                "temp_mean",
                "wind_speed_mean",
            ]:
                if col in hrow.index:
                    rec[col] = hrow[col]
            all_future.append(rec)

    if not all_future:
        raise RuntimeError("No dust future records generated.")

    return (
        pd.DataFrame(all_future)
        .sort_values(["city", "timestamp"])
        .reset_index(drop=True),
        metrics_rows,
        features,
    )


# -------------------------------------------------------------------------
# Feature audit
# -------------------------------------------------------------------------
def build_feature_audit(
    df: pd.DataFrame,
    drought_features: List[str],
    precip_features: List[str],
    dust_features: List[str],
) -> pd.DataFrame:
    rows = []
    numeric = set(numeric_columns(df))

    configs = [
        (
            "drought_flag_and_severity",
            set(drought_features),
            COMMON_EXCLUDE | DROUGHT_DIRECT_LEAKAGE,
        ),
        (
            "precipitation_sum",
            set(precip_features),
            COMMON_EXCLUDE | PRECIP_DIRECT_LEAKAGE,
        ),
        (
            "dust_event",
            set(dust_features),
            COMMON_EXCLUDE | DUST_DIRECT_LEAKAGE,
        ),
    ]

    for target, used, direct_exclusions in configs:
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
                elif target == "dust_event" and any(
                    col.startswith(prefix)
                    for prefix in DUST_CURRENT_ROLLING_PREFIXES
                ):
                    reason = "Current-row rolling proxy exclusion"
                else:
                    reason = "Target/cross-target/other controlled exclusion"

            rows.append(
                {
                    "target": target,
                    "column": col,
                    "status": status,
                    "reason": reason,
                }
            )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def resolve_same_directory(path_text: str, default_path: Path) -> Path:
    if not path_text:
        return default_path
    p = Path(path_text).expanduser()
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XGBoost Trial 4 - fixed unified hyperparameters, 600 trees"
    )
    parser.add_argument("--input", default="LATEST.parquet")
    parser.add_argument("--output", default=DEFAULT_PREDICTIONS.name)
    parser.add_argument("--metrics_output", default=DEFAULT_METRICS.name)
    parser.add_argument(
        "--feature_audit_output",
        default=DEFAULT_FEATURE_AUDIT.name,
    )
    parser.add_argument("--horizon_days", type=int, default=30)
    args = parser.parse_args()

    input_path = resolve_same_directory(args.input, DEFAULT_INPUT)
    pred_path = resolve_same_directory(args.output, DEFAULT_PREDICTIONS)
    metrics_path = resolve_same_directory(args.metrics_output, DEFAULT_METRICS)
    audit_path = resolve_same_directory(
        args.feature_audit_output,
        DEFAULT_FEATURE_AUDIT,
    )

    print(f"[INFO] Script directory : {SCRIPT_DIR}")
    print(f"[INFO] Input dataset    : {input_path}")
    print(f"[INFO] Predictions out  : {pred_path}")
    print(f"[INFO] Metrics out      : {metrics_path}")
    print(f"[INFO] Feature audit out: {audit_path}")
    print("[INFO] Split            : chronological 70% train / 15% validation / 15% test")
    print("[INFO] Unified XGB params:", XGB_SHARED_PARAMS)

    df = load_dataset(input_path)
    print(f"[INFO] Dataset shape    : {df.shape}")

    print("\n[INFO] Training drought + precipitation XGBoost models...")
    drought_bundle = train_drought_and_precip(df)

    print("\n[INFO] Building prospective drought/precipitation horizon...")
    drought_future = predict_drought_future(
        df,
        drought_bundle,
        args.horizon_days,
    )

    print("\n[INFO] Training/evaluating per-city dust XGBoost models...")
    dust_future, dust_metrics, dust_features = train_and_predict_dust(
        df,
        args.horizon_days,
    )

    merged = drought_future.merge(
        dust_future,
        on=["city", "timestamp"],
        how="outer",
        sort=True,
    )
    merged = merged.sort_values(["city", "timestamp"]).reset_index(drop=True)

    metrics = pd.DataFrame(
        drought_bundle.metrics_rows + dust_metrics
    )

    preferred_cols = [
        "model_name",
        "model_type",
        "target_name",
        "city",
        "split",
        "n_samples",
        "train_n",
        "validation_n",
        "test_n",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc_roc",
        "loss",
        "r2_score",
        "mse",
        "rmse",
        "mae",
        "tn",
        "fp",
        "fn",
        "tp",
        "confusion_matrix",
        "classification_report",
        "n_features",
        "feature_list",
        "selected_hyperparameters",
        "class_counts",
        "decision_threshold",
    ]
    for col in preferred_cols:
        if col not in metrics.columns:
            metrics[col] = None
    metrics = metrics[preferred_cols]

    audit = build_feature_audit(
        df,
        drought_bundle.drought_features,
        drought_bundle.precip_features,
        dust_features,
    )

    merged.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    audit.to_csv(audit_path, index=False)

    print("\n[INFO] Finished successfully.")
    print(f"[INFO] Saved predictions : {pred_path}")
    print(f"[INFO] Saved metrics     : {metrics_path}")
    print(f"[INFO] Saved feature audit: {audit_path}")

    display_cols = [
        "target_name",
        "city",
        "split",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "rmse",
        "r2_score",
        "tn",
        "fp",
        "fn",
        "tp",
        "decision_threshold",
    ]

    print("\n[INFO] FINAL TEST rows:")
    print(
        metrics.loc[
            metrics["split"] == "test",
            display_cols,
        ].to_string(index=False)
    )

    print("\n[INFO] VALIDATION rows:")
    print(
        metrics.loc[
            metrics["split"] == "validation",
            display_cols,
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
