#!/usr/bin/env python3
"""
LSTM Trial 6 - reviewer revision, LOCAL execution.

Place this script in the SAME directory as LATEST.parquet.

Fixed LSTM hyperparameters for all targets:
    sequence_length = 14
    lstm_units      = 128
    dense_units     = 32
    dropout_rate    = 0.30
    learning_rate   = 0.0005
    epochs          = 25 maximum
    batch_size      = 32
    early_stopping_patience = 4
    optimiser       = Adam
    random_state    = 42

Task-specific output/loss:
    Binary classification:
        sigmoid + binary_crossentropy
    Multiclass classification:
        softmax + sparse_categorical_crossentropy
    Regression:
        linear output + mse

Reviewer-revision methodology:
    - chronological 70% train / 15% validation / 15% final test
    - no shuffling
    - same leakage-sensitive exclusions as revised RF/XGBoost
    - StandardScaler fit on TRAIN only for validation phase
    - class weights computed from TRAIN only
    - binary decision threshold selected on VALIDATION F1 only
    - early stopping uses VALIDATION only
    - final model is rebuilt and trained on TRAIN+VALIDATION for the
      number of epochs selected during validation
    - final TEST remains separate
    - TP/TN/FP/FN and confusion matrices saved
    - feature audit saved
    - 30-day output remains a prospective horizon, not independently
      future-validated ground truth
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models, optimizers
except ImportError as e:
    raise ImportError(
        "TensorFlow is required. Install it with: python -m pip install tensorflow"
    ) from e


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "LATEST.parquet"
DEFAULT_PREDICTIONS = SCRIPT_DIR / "lstm_trial6_predictions.csv"
DEFAULT_METRICS = SCRIPT_DIR / "lstm_trial6_metrics.csv"
DEFAULT_FEATURE_AUDIT = SCRIPT_DIR / "lstm_trial6_feature_audit.csv"

RANDOM_STATE = 42

SEQUENCE_LENGTH = 14
LSTM_UNITS = 128
DENSE_UNITS = 32
DROPOUT_RATE = 0.30
LEARNING_RATE = 0.0005
MAX_EPOCHS = 25
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 4

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

LSTM_SHARED_PARAMS = {
    "sequence_length": SEQUENCE_LENGTH,
    "lstm_units": LSTM_UNITS,
    "dense_units": DENSE_UNITS,
    "dropout_rate": DROPOUT_RATE,
    "learning_rate": LEARNING_RATE,
    "max_epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "optimizer": "Adam",
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------------
# Leakage-control policy: aligned with revised RF/XGBoost
# ---------------------------------------------------------------------
COMMON_EXCLUDE = {
    "city",
    "timestamp",
    "date",
    "latitude",
    "longitude",
    "is_simulated_dust",
    "is_simulated_drought",
    "sim_source_dust",
    "sim_source_drought",
    "sim_source_dust_code",
    "sim_source_drought_code",
    "data_source_dust",
    "data_source_drought",
    "data_source_dust_met_code",
    "data_source_drought_code",
}

DROUGHT_DIRECT_LEAKAGE = {
    "drought_flag",
    "drought_severity",
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
    "drought_severity",
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


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------
def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


def safe_float(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    if "timestamp" not in df.columns or "city" not in df.columns:
        raise ValueError("Dataset must contain 'timestamp' and 'city'.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "city"]).copy()
    return df.sort_values(["timestamp", "city"]).reset_index(drop=True)


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def auto_detect_dust_target(df: pd.DataFrame) -> str:
    if "dust_event" in df.columns:
        return "dust_event"
    if "dust_event_x" in df.columns:
        return "dust_event_x"
    raise ValueError("No dust_event target was found.")


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


def select_dust_features(df: pd.DataFrame, target: str) -> List[str]:
    exclude = COMMON_EXCLUDE | DUST_DIRECT_LEAKAGE | {
        target,
        "drought_flag",
        "drought_severity",
        "drought_severity_code",
    }

    cols = []
    for c in numeric_columns(df):
        if c in exclude:
            continue
        if any(c.startswith(prefix) for prefix in DUST_CURRENT_ROLLING_PREFIXES):
            continue
        cols.append(c)
    return cols


def impute_using_reference(
    data: pd.DataFrame,
    features: List[str],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    x = data[features].copy()
    ref = reference[features].copy()

    for c in features:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        ref[c] = pd.to_numeric(ref[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)
    ref = ref.replace([np.inf, -np.inf], np.nan)

    med = ref.median(numeric_only=True).reindex(features)
    for c in features:
        fill = med.get(c, np.nan)
        if pd.isna(fill):
            fill = 0.0
        x[c] = x[c].fillna(fill)

    return x.astype(float)


# ---------------------------------------------------------------------
# Sequence construction and chronological 70/15/15
# ---------------------------------------------------------------------
def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seqs, ys, timestamps, cities = [], [], [], []

    source = (
        df.dropna(subset=["city", "timestamp", target_col])
        .sort_values(["city", "timestamp"])
        .reset_index(drop=True)
    )

    # At sequence-building stage only impute to make arrays. The scaler is
    # still fitted strictly on the training split later.
    x_all = impute_using_reference(source, feature_cols, source)
    source_x = source[["city", "timestamp", target_col]].copy()
    source_x = pd.concat([source_x.reset_index(drop=True), x_all.reset_index(drop=True)], axis=1)

    for city, cdf in source_x.groupby("city", sort=False):
        cdf = cdf.sort_values("timestamp").reset_index(drop=True)
        if len(cdf) < sequence_length:
            continue

        mat = cdf[feature_cols].to_numpy(dtype=float)
        yv = cdf[target_col].to_numpy()

        for end in range(sequence_length - 1, len(cdf)):
            seqs.append(mat[end - sequence_length + 1:end + 1])
            ys.append(yv[end])
            timestamps.append(pd.Timestamp(cdf.iloc[end]["timestamp"]))
            cities.append(str(city))

    if not seqs:
        raise RuntimeError(f"No sequences generated for {target_col}.")

    X = np.asarray(seqs, dtype=np.float32)
    y = np.asarray(ys)
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    city_arr = np.asarray(cities, dtype=object)

    order = np.argsort(ts)
    return X[order], y[order], ts[order], city_arr[order]


def chronological_70_15_15_indices(
    timestamps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(timestamps).normalize()
    unique_dates = np.array(sorted(pd.Series(dates).unique()))

    if len(unique_dates) < 10:
        raise ValueError("Too few unique dates for chronological 70/15/15 split.")

    train_end = max(1, int(np.floor(len(unique_dates) * TRAIN_RATIO)))
    val_end = max(
        train_end + 1,
        int(np.floor(len(unique_dates) * (TRAIN_RATIO + VALIDATION_RATIO))),
    )
    val_end = min(val_end, len(unique_dates) - 1)

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    idx = np.arange(len(timestamps))
    train_idx = idx[pd.Series(dates).isin(train_dates).to_numpy()]
    val_idx = idx[pd.Series(dates).isin(val_dates).to_numpy()]
    test_idx = idx[pd.Series(dates).isin(test_dates).to_numpy()]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("An empty train/validation/test split was produced.")

    return train_idx, val_idx, test_idx


def fit_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X.reshape(-1, X.shape[-1]))
    return scaler


def transform_sequences(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, t, f = X.shape
    transformed = scaler.transform(X.reshape(-1, f))
    return transformed.reshape(n, t, f).astype(np.float32)


def class_weight_dict(y: np.ndarray) -> Optional[Dict[int, float]]:
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    if len(classes) < 2:
        return None
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ---------------------------------------------------------------------
# LSTM architectures
# ---------------------------------------------------------------------
def build_classifier(input_shape: Tuple[int, int], n_classes: int):
    output_units = 1 if n_classes == 2 else n_classes
    output_activation = "sigmoid" if n_classes == 2 else "softmax"
    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(LSTM_UNITS),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(DENSE_UNITS, activation="relu"),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(output_units, activation=output_activation),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def build_regressor(input_shape: Tuple[int, int]):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(LSTM_UNITS),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(DENSE_UNITS, activation="relu"),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def select_binary_threshold(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    best_t, best_f1, best_recall = 0.50, -1.0, -1.0
    for t in np.arange(0.05, 0.501, 0.01):
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and rec > best_recall):
            best_t = round(float(t), 2)
            best_f1 = float(f1)
            best_recall = float(rec)
    return best_t, best_f1


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def classifier_metrics_row(
    *,
    model_name: str,
    target_name: str,
    city: str,
    split: str,
    y_true,
    y_pred,
    y_proba,
    labels: List[int],
    average: str,
    feature_cols: List[str],
    train_n: int,
    validation_n: int,
    test_n: int,
    class_counts: Dict[str, object],
    threshold: Optional[float],
    selected_epochs: int,
) -> Dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tn = fp = fn = tp = None
    if len(labels) == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = map(int, cm.ravel())

    row = {
        "model_name": model_name,
        "model_type": "classifier",
        "target_name": target_name,
        "city": city,
        "split": split,
        "n_samples": len(y_true),
        "train_n": train_n,
        "validation_n": validation_n,
        "test_n": test_n,
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "precision": safe_float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": safe_float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": safe_float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "auc_roc": None,
        "loss": None,
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
                y_true, y_pred, labels=labels, output_dict=True, zero_division=0
            )
        ),
        "n_features": len(feature_cols),
        "feature_list": safe_json(feature_cols),
        "selected_hyperparameters": safe_json(LSTM_SHARED_PARAMS),
        "class_counts": safe_json(class_counts),
        "decision_threshold": threshold,
        "selected_epochs": int(selected_epochs),
    }

    try:
        row["loss"] = safe_float(log_loss(y_true, y_proba, labels=labels))
    except Exception:
        pass

    try:
        yp = np.asarray(y_proba)
        if len(labels) == 2 and yp.ndim == 2 and yp.shape[1] >= 2 and len(np.unique(y_true)) == 2:
            row["auc_roc"] = safe_float(roc_auc_score(y_true, yp[:, 1]))
        elif len(labels) > 2 and yp.ndim == 2 and len(np.unique(y_true)) > 1:
            row["auc_roc"] = safe_float(
                roc_auc_score(y_true, yp, multi_class="ovr", average="weighted")
            )
    except Exception:
        pass

    return row


def regressor_metrics_row(
    *,
    model_name: str,
    target_name: str,
    city: str,
    split: str,
    y_true,
    y_pred,
    feature_cols: List[str],
    train_n: int,
    validation_n: int,
    test_n: int,
    selected_epochs: int,
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
        "n_samples": len(y_true),
        "train_n": train_n,
        "validation_n": validation_n,
        "test_n": test_n,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "auc_roc": None,
        "loss": safe_float(mse),
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
        "selected_hyperparameters": safe_json(LSTM_SHARED_PARAMS),
        "class_counts": None,
        "decision_threshold": None,
        "selected_epochs": int(selected_epochs),
    }


# ---------------------------------------------------------------------
# Training bundles
# ---------------------------------------------------------------------
@dataclass
class ClassificationBundle:
    model: object
    scaler: StandardScaler
    features: List[str]
    threshold: Optional[float]
    label_encoding: Dict[str, int]
    label_decoding: Dict[int, str]
    selected_epochs: int


@dataclass
class RegressionBundle:
    model: object
    scaler: StandardScaler
    features: List[str]
    selected_epochs: int


def train_classifier_reviewer_safe(
    *,
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model_name: str,
    city: str,
) -> Tuple[ClassificationBundle, List[Dict[str, object]]]:
    set_seed()

    X_raw, y_raw, ts, _ = build_sequences(
        df, features, target, SEQUENCE_LENGTH
    )

    if np.issubdtype(y_raw.dtype, np.number):
        y = pd.Series(y_raw).astype(int).to_numpy()
        classes_original = sorted(np.unique(y).tolist())
        encoding = {str(c): int(c) for c in classes_original}
        decoding = {int(c): str(c) for c in classes_original}
    else:
        labels_text = sorted(pd.Series(y_raw).astype(str).unique().tolist())
        encoding = {lab: i for i, lab in enumerate(labels_text)}
        decoding = {i: lab for lab, i in encoding.items()}
        y = pd.Series(y_raw).astype(str).map(encoding).astype(int).to_numpy()

    tr_idx, va_idx, te_idx = chronological_70_15_15_indices(ts)

    Xtr_raw, Xva_raw, Xte_raw = X_raw[tr_idx], X_raw[va_idx], X_raw[te_idx]
    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

    n_classes = len(np.unique(y))
    labels = sorted(np.unique(y).tolist())
    average = "binary" if n_classes == 2 else "weighted"

    scaler = fit_scaler(Xtr_raw)
    Xtr = transform_sequences(Xtr_raw, scaler)
    Xva = transform_sequences(Xva_raw, scaler)

    model = build_classifier(
        input_shape=(Xtr.shape[1], Xtr.shape[2]),
        n_classes=n_classes,
    )

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict(ytr),
        shuffle=False,
        verbose=0,
        callbacks=[es],
    )

    selected_epochs = int(np.argmin(history.history["val_loss"]) + 1)

    if n_classes == 2:
        p_val_pos = model.predict(Xva, verbose=0).reshape(-1)
        threshold, _ = select_binary_threshold(yva, p_val_pos)
        pred_val = (p_val_pos >= threshold).astype(int)
        proba_val = np.column_stack([1.0 - p_val_pos, p_val_pos])
    else:
        threshold = None
        proba_val = model.predict(Xva, verbose=0)
        pred_val = np.argmax(proba_val, axis=1)

    counts = {
        "train": pd.Series(ytr).value_counts().sort_index().to_dict(),
        "validation": pd.Series(yva).value_counts().sort_index().to_dict(),
        "test": pd.Series(yte).value_counts().sort_index().to_dict(),
    }

    val_row = classifier_metrics_row(
        model_name=model_name,
        target_name=target,
        city=city,
        split="validation",
        y_true=yva,
        y_pred=pred_val,
        y_proba=proba_val,
        labels=labels,
        average=average,
        feature_cols=features,
        train_n=len(tr_idx),
        validation_n=len(va_idx),
        test_n=len(te_idx),
        class_counts=counts,
        threshold=threshold,
        selected_epochs=selected_epochs,
    )

    # Final model: train + validation, using epoch count chosen on validation.
    train_val_idx = np.concatenate([tr_idx, va_idx])
    Xtv_raw = X_raw[train_val_idx]
    ytv = y[train_val_idx]

    final_scaler = fit_scaler(Xtv_raw)
    Xtv = transform_sequences(Xtv_raw, final_scaler)
    Xte = transform_sequences(Xte_raw, final_scaler)

    set_seed()
    final_model = build_classifier(
        input_shape=(Xtv.shape[1], Xtv.shape[2]),
        n_classes=n_classes,
    )
    final_model.fit(
        Xtv,
        ytv,
        epochs=max(1, selected_epochs),
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict(ytv),
        shuffle=False,
        verbose=0,
    )

    if n_classes == 2:
        p_test_pos = final_model.predict(Xte, verbose=0).reshape(-1)
        pred_test = (p_test_pos >= threshold).astype(int)
        proba_test = np.column_stack([1.0 - p_test_pos, p_test_pos])
    else:
        proba_test = final_model.predict(Xte, verbose=0)
        pred_test = np.argmax(proba_test, axis=1)

    test_row = classifier_metrics_row(
        model_name=model_name,
        target_name=target,
        city=city,
        split="test",
        y_true=yte,
        y_pred=pred_test,
        y_proba=proba_test,
        labels=labels,
        average=average,
        feature_cols=features,
        train_n=len(tr_idx),
        validation_n=len(va_idx),
        test_n=len(te_idx),
        class_counts=counts,
        threshold=threshold,
        selected_epochs=selected_epochs,
    )

    bundle = ClassificationBundle(
        model=final_model,
        scaler=final_scaler,
        features=features,
        threshold=threshold,
        label_encoding=encoding,
        label_decoding=decoding,
        selected_epochs=selected_epochs,
    )
    return bundle, [val_row, test_row]


def train_regressor_reviewer_safe(
    *,
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model_name: str,
    city: str,
) -> Tuple[RegressionBundle, List[Dict[str, object]]]:
    set_seed()

    X_raw, y_raw, ts, _ = build_sequences(
        df, features, target, SEQUENCE_LENGTH
    )
    y = pd.to_numeric(pd.Series(y_raw), errors="coerce").astype(float).to_numpy()

    tr_idx, va_idx, te_idx = chronological_70_15_15_indices(ts)
    Xtr_raw, Xva_raw, Xte_raw = X_raw[tr_idx], X_raw[va_idx], X_raw[te_idx]
    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

    scaler = fit_scaler(Xtr_raw)
    Xtr = transform_sequences(Xtr_raw, scaler)
    Xva = transform_sequences(Xva_raw, scaler)

    model = build_regressor((Xtr.shape[1], Xtr.shape[2]))
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
        callbacks=[es],
    )

    selected_epochs = int(np.argmin(history.history["val_loss"]) + 1)
    val_pred = model.predict(Xva, verbose=0).reshape(-1)

    val_row = regressor_metrics_row(
        model_name=model_name,
        target_name=target,
        city=city,
        split="validation",
        y_true=yva,
        y_pred=val_pred,
        feature_cols=features,
        train_n=len(tr_idx),
        validation_n=len(va_idx),
        test_n=len(te_idx),
        selected_epochs=selected_epochs,
    )

    train_val_idx = np.concatenate([tr_idx, va_idx])
    Xtv_raw = X_raw[train_val_idx]
    ytv = y[train_val_idx]

    final_scaler = fit_scaler(Xtv_raw)
    Xtv = transform_sequences(Xtv_raw, final_scaler)
    Xte = transform_sequences(Xte_raw, final_scaler)

    set_seed()
    final_model = build_regressor((Xtv.shape[1], Xtv.shape[2]))
    final_model.fit(
        Xtv,
        ytv,
        epochs=max(1, selected_epochs),
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )

    test_pred = final_model.predict(Xte, verbose=0).reshape(-1)

    test_row = regressor_metrics_row(
        model_name=model_name,
        target_name=target,
        city=city,
        split="test",
        y_true=yte,
        y_pred=test_pred,
        feature_cols=features,
        train_n=len(tr_idx),
        validation_n=len(va_idx),
        test_n=len(te_idx),
        selected_epochs=selected_epochs,
    )

    bundle = RegressionBundle(
        model=final_model,
        scaler=final_scaler,
        features=features,
        selected_epochs=selected_epochs,
    )
    return bundle, [val_row, test_row]


# ---------------------------------------------------------------------
# Simple future analogue generation + sequence inference
# ---------------------------------------------------------------------
def update_calendar(row: pd.Series, future_date: pd.Timestamp) -> pd.Series:
    row = row.copy()
    row["timestamp"] = future_date
    if "date" in row.index:
        row["date"] = future_date.normalize().date().isoformat()

    mapping = {
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
    for c, v in mapping.items():
        if c in row.index:
            row[c] = v
    return row


def build_future_rows(
    df: pd.DataFrame,
    days: int,
    clear_targets: List[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for city, cdf in df.groupby("city"):
        cdf = cdf.sort_values("timestamp").reset_index(drop=True)
        last_date = pd.Timestamp(cdf["timestamp"].max())
        pool = cdf[cdf["timestamp"] < last_date].copy()
        if pool.empty:
            pool = cdf.copy()

        for step in range(1, days + 1):
            fdate = last_date + timedelta(days=step)
            candidates = pool[
                (pool["timestamp"].dt.month == fdate.month)
                & (pool["timestamp"].dt.day == fdate.day)
            ]
            if candidates.empty:
                candidates = pool[pool["timestamp"].dt.month == fdate.month]
            if candidates.empty:
                candidates = pool

            row = candidates.loc[rng.choice(candidates.index.to_numpy())].copy()
            row = update_calendar(row, fdate)

            for c in clear_targets:
                if c in row.index:
                    row[c] = np.nan
            rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def build_inference_sequences(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    city: str,
    features: List[str],
    scaler: StandardScaler,
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    hist = history_df[history_df["city"].astype(str) == str(city)].copy()
    fut = future_df[future_df["city"].astype(str) == str(city)].copy()

    hist = hist.sort_values("timestamp")
    fut = fut.sort_values("timestamp")

    combined = pd.concat([hist, fut], ignore_index=True, sort=False)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    xdf = impute_using_reference(combined, features, hist if not hist.empty else combined)
    xscaled = scaler.transform(xdf[features].to_numpy(dtype=float))

    future_times = set(pd.to_datetime(fut["timestamp"]).tolist())
    seqs, ts_out = [], []

    for i, ts in enumerate(pd.to_datetime(combined["timestamp"])):
        if ts not in future_times:
            continue
        start = max(0, i - SEQUENCE_LENGTH + 1)
        window = xscaled[start:i + 1]
        if len(window) < SEQUENCE_LENGTH:
            pad = np.repeat(window[[0]], SEQUENCE_LENGTH - len(window), axis=0)
            window = np.vstack([pad, window])
        seqs.append(window[-SEQUENCE_LENGTH:])
        ts_out.append(ts)

    return np.asarray(seqs, dtype=np.float32), ts_out


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------
def train_all(df: pd.DataFrame, horizon_days: int):
    drought_features = select_drought_features(df)
    precip_features = select_precip_features(df)
    dust_target = auto_detect_dust_target(df)
    dust_features = select_dust_features(df, dust_target)

    metrics_rows = []

    print("\n[INFO] Training drought_flag LSTM...")
    flag_bundle, rows = train_classifier_reviewer_safe(
        df=df.dropna(subset=["drought_flag"]),
        features=drought_features,
        target="drought_flag",
        model_name="drought_flag_lstm_classifier",
        city="ALL",
    )
    metrics_rows.extend(rows)

    print("[INFO] Training drought_severity LSTM...")
    severity_bundle, rows = train_classifier_reviewer_safe(
        df=df.dropna(subset=["drought_severity"]),
        features=drought_features,
        target="drought_severity",
        model_name="drought_severity_lstm_classifier",
        city="ALL",
    )
    metrics_rows.extend(rows)

    print("[INFO] Training precipitation_sum LSTM...")
    precip_bundle, rows = train_regressor_reviewer_safe(
        df=df.dropna(subset=["precipitation_sum"]),
        features=precip_features,
        target="precipitation_sum",
        model_name="precipitation_sum_lstm_regressor",
        city="ALL",
    )
    metrics_rows.extend(rows)

    dust_bundles = {}
    for city in sorted(df["city"].dropna().astype(str).unique()):
        print(f"[INFO] Training dust LSTM for {city}...")
        cdf = df[df["city"].astype(str) == city].copy()
        try:
            bundle, rows = train_classifier_reviewer_safe(
                df=cdf,
                features=dust_features,
                target=dust_target,
                model_name=f"dust_lstm_classifier_{city.lower().replace(' ', '_')}",
                city=city,
            )
            dust_bundles[city] = bundle
            metrics_rows.extend(rows)
        except Exception as exc:
            print(f"[WARN] Skipping {city}: {exc}")

    # Future analogue rows
    future = build_future_rows(
        df,
        horizon_days,
        clear_targets=[
            "drought_flag",
            "drought_severity",
            "drought_severity_code",
            "dust_event",
            "dust_event_x",
            "precipitation_sum",
        ],
    )

    outputs = []

    for city in sorted(future["city"].astype(str).unique()):
        city_future = future[future["city"].astype(str) == city].copy()

        # Drought flag
        Xf, tsf = build_inference_sequences(
            df, future, city, flag_bundle.features, flag_bundle.scaler
        )
        p_flag = flag_bundle.model.predict(Xf, verbose=0).reshape(-1)
        flag_pred = (p_flag >= float(flag_bundle.threshold)).astype(int)

        # Drought severity
        Xs, tss = build_inference_sequences(
            df, future, city, severity_bundle.features, severity_bundle.scaler
        )
        sev_proba = severity_bundle.model.predict(Xs, verbose=0)
        if sev_proba.ndim == 1 or sev_proba.shape[-1] == 1:
            sev_codes = (sev_proba.reshape(-1) >= 0.5).astype(int)
        else:
            sev_codes = np.argmax(sev_proba, axis=1)
        sev_labels = [
            severity_bundle.label_decoding.get(int(c), str(c)) for c in sev_codes
        ]

        # Precipitation
        Xp, tsp = build_inference_sequences(
            df, future, city, precip_bundle.features, precip_bundle.scaler
        )
        precip_pred = precip_bundle.model.predict(Xp, verbose=0).reshape(-1)

        # Dust
        dust_pred = np.full(len(tsf), np.nan)
        dust_prob = np.full(len(tsf), np.nan)
        if city in dust_bundles:
            db = dust_bundles[city]
            Xd, tsd = build_inference_sequences(
                df, future, city, db.features, db.scaler
            )
            prob = db.model.predict(Xd, verbose=0).reshape(-1)
            pred = (prob >= float(db.threshold)).astype(int)
            dust_pred[:len(pred)] = pred
            dust_prob[:len(prob)] = prob

        city_out = pd.DataFrame(
            {
                "city": city,
                "timestamp": tsf,
                "drought_flag_pred": flag_pred,
                "drought_severity_pred": sev_labels,
                "precipitation_sum_pred": precip_pred,
                "dust_event_pred": dust_pred,
                "dust_event_prob": dust_prob,
            }
        )
        outputs.append(city_out)

    predictions = pd.concat(outputs, ignore_index=True)
    return predictions, pd.DataFrame(metrics_rows), drought_features, precip_features, dust_features


def build_feature_audit(
    df: pd.DataFrame,
    drought_features: List[str],
    precip_features: List[str],
    dust_features: List[str],
) -> pd.DataFrame:
    numeric = set(numeric_columns(df))
    rows = []

    configs = [
        ("drought_flag_and_severity", set(drought_features), COMMON_EXCLUDE | DROUGHT_DIRECT_LEAKAGE),
        ("precipitation_sum", set(precip_features), COMMON_EXCLUDE | PRECIP_DIRECT_LEAKAGE),
        ("dust_event", set(dust_features), COMMON_EXCLUDE | DUST_DIRECT_LEAKAGE),
    ]

    for target, used, excluded in configs:
        for col in df.columns:
            if col in used:
                status, reason = "USED", "Selected numeric predictor"
            elif col not in numeric:
                status, reason = "EXCLUDED", "Non-numeric / identifier / text"
            elif col in excluded:
                status, reason = "EXCLUDED", "Leakage/target/source exclusion"
            elif target == "dust_event" and any(col.startswith(p) for p in DUST_CURRENT_ROLLING_PREFIXES):
                status, reason = "EXCLUDED", "Current-row rolling proxy exclusion"
            else:
                status, reason = "EXCLUDED", "Controlled cross-target/other exclusion"

            rows.append({
                "target": target,
                "column": col,
                "status": status,
                "reason": reason,
            })

    return pd.DataFrame(rows)


def resolve_path(text: str, default: Path) -> Path:
    if not text:
        return default
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="LSTM Trial 6 - reviewer-safe chronological 70/15/15"
    )
    parser.add_argument("--input", default="LATEST.parquet")
    parser.add_argument("--output", default=DEFAULT_PREDICTIONS.name)
    parser.add_argument("--metrics_output", default=DEFAULT_METRICS.name)
    parser.add_argument("--feature_audit_output", default=DEFAULT_FEATURE_AUDIT.name)
    parser.add_argument("--horizon_days", type=int, default=30)
    args = parser.parse_args()

    input_path = resolve_path(args.input, DEFAULT_INPUT)
    output_path = resolve_path(args.output, DEFAULT_PREDICTIONS)
    metrics_path = resolve_path(args.metrics_output, DEFAULT_METRICS)
    audit_path = resolve_path(args.feature_audit_output, DEFAULT_FEATURE_AUDIT)

    print(f"[INFO] Input      : {input_path}")
    print(f"[INFO] Predictions: {output_path}")
    print(f"[INFO] Metrics    : {metrics_path}")
    print(f"[INFO] Audit      : {audit_path}")
    print("[INFO] Split      : chronological 70% train / 15% validation / 15% test")
    print("[INFO] LSTM params:", LSTM_SHARED_PARAMS)

    df = load_dataset(input_path)
    print(f"[INFO] Dataset shape: {df.shape}")

    predictions, metrics, drought_features, precip_features, dust_features = train_all(
        df, args.horizon_days
    )

    audit = build_feature_audit(
        df, drought_features, precip_features, dust_features
    )

    predictions.to_csv(output_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    audit.to_csv(audit_path, index=False)

    print("\n[INFO] Finished successfully.")
    print(f"[INFO] Saved predictions : {output_path}")
    print(f"[INFO] Saved metrics     : {metrics_path}")
    print(f"[INFO] Saved feature audit: {audit_path}")

    cols = [
        "target_name", "city", "split",
        "accuracy", "precision", "recall", "f1_score",
        "rmse", "r2_score",
        "tn", "fp", "fn", "tp",
        "decision_threshold", "selected_epochs",
    ]
    for c in cols:
        if c not in metrics.columns:
            metrics[c] = None

    print("\n[INFO] VALIDATION RESULTS")
    print(metrics.loc[metrics["split"] == "validation", cols].to_string(index=False))

    print("\n[INFO] FINAL TEST RESULTS")
    print(metrics.loc[metrics["split"] == "test", cols].to_string(index=False))


if __name__ == "__main__":
    main()
