from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_FILES = {
    "rf_metrics": SCRIPT_DIR / "rf_trial6_metrics.csv",
    "rf_predictions": SCRIPT_DIR / "rf_trial6_predictions.csv",
    "xgb_metrics": SCRIPT_DIR / "xgb_trial4_metrics.csv",
    "xgb_predictions": SCRIPT_DIR / "xgb_trial4_predictions.csv",
    "lstm_metrics": SCRIPT_DIR / "lstm_trial6_metrics.csv",
    "lstm_predictions": SCRIPT_DIR / "lstm_trial6_predictions.csv",
}


# ---------------------------------------------------------------------
# Canonical target names
# ---------------------------------------------------------------------
def canonical_target_name(value: object) -> str:
    s = str(value).strip().lower()
    aliases = {
        "drought_severity_code": "drought_severity",
        "drought_severity": "drought_severity",
        "drought_flag": "drought_flag",
        "precipitation_sum": "precipitation_sum",
        "dust_event": "dust_event",
        "dust_event_x": "dust_event",
    }
    return aliases.get(s, s)


TARGET_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "drought_flag": ["drought_flag_pred"],
    "drought_severity": [
        "drought_severity_pred",
        "drought_severity_code_pred",
    ],
    "precipitation_sum": ["precipitation_sum_pred"],
    "dust_event": ["dust_event_pred", "dust_event_x_pred"],
}

COMPANION_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "drought_flag": [],
    "drought_severity": [],
    "precipitation_sum": [
        "precipitation_sum_normal",
        "precip_deficit_pred",
    ],
    "dust_event": [
        "dust_event_prob",
        "dust_event_x_prob",
    ],
}

FINAL_STANDARD_COLUMNS: Dict[str, str] = {
    "drought_flag": "drought_flag_pred",
    "drought_severity": "drought_severity_pred",
    "precipitation_sum": "precipitation_sum_pred",
    "dust_event": "dust_event_pred",
}


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def resolve_local_path(text: str) -> Path:
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p.resolve()


def load_metrics(path: Path, source_alias: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    df = pd.read_csv(path)
    if "target_name" not in df.columns:
        raise ValueError(f"{path.name} does not contain target_name.")
    df["source_model"] = source_alias
    df["source_metrics_file"] = path.name
    df["target_name"] = df["target_name"].map(canonical_target_name)
    if "city" not in df.columns:
        df["city"] = "ALL"
    df["city"] = df["city"].fillna("ALL").astype(str)
    return df


def load_predictions(path: Path, source_alias: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "city" not in df.columns:
        raise ValueError(
            f"{path.name} must contain both city and timestamp columns."
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "city"]).copy()
    df["city"] = df["city"].astype(str)
    df["source_predictions_file"] = path.name
    return df


# ---------------------------------------------------------------------
# Metric selection
# ---------------------------------------------------------------------
def normalize_model_type(val: object) -> str:
    s = str(val).strip().lower()
    if s in {"classifier", "classification"}:
        return "classification"
    if s in {"regressor", "regression"}:
        return "regression"
    return s


def metric_sort_values(row: pd.Series) -> Tuple:
    target = canonical_target_name(row["target_name"])
    model_type = normalize_model_type(row.get("model_type"))

    if target == "precipitation_sum" or model_type == "regression":
        rmse = row.get("rmse")
        r2 = row.get("r2_score")
        mae = row.get("mae")
        return (
            np.inf if pd.isna(rmse) else float(rmse),
            -(float(r2)) if pd.notna(r2) else np.inf,
            np.inf if pd.isna(mae) else float(mae),
        )

    f1 = row.get("f1_score")
    recall = row.get("recall")
    precision = row.get("precision")
    accuracy = row.get("accuracy")
    return (
        -(float(f1)) if pd.notna(f1) else np.inf,
        -(float(recall)) if pd.notna(recall) else np.inf,
        -(float(precision)) if pd.notna(precision) else np.inf,
        -(float(accuracy)) if pd.notna(accuracy) else np.inf,
    )


def choose_best_metrics_rows(
    metrics_df: pd.DataFrame,
    selection_split: str = "validation",
) -> pd.DataFrame:
    work = metrics_df.copy()
    work["target_name"] = work["target_name"].map(canonical_target_name)

    # CRITICAL: revised metric files contain both validation and test rows.
    # Model selection must use validation, not the final test set.
    if "split" in work.columns:
        available = set(work["split"].dropna().astype(str).str.lower())
        wanted = selection_split.lower()
        if wanted not in available:
            raise ValueError(
                f"Requested split '{selection_split}' not found. "
                f"Available splits: {sorted(available)}"
            )
        work = work[
            work["split"].astype(str).str.lower() == wanted
        ].copy()

    numeric_cols = [
        "accuracy", "loss", "precision", "recall", "f1_score",
        "auc_roc", "r2_score", "mse", "rmse", "mae",
    ]
    for c in numeric_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    winners = []
    global_targets = {
        "drought_flag",
        "drought_severity",
        "precipitation_sum",
    }

    for target, group in work.groupby("target_name"):
        if target in global_targets:
            grouped_rows = [("ALL", group)]
        elif target == "dust_event":
            grouped_rows = [
                (city, city_group)
                for city, city_group in group.groupby("city")
            ]
        else:
            print(f"[WARN] Unknown target '{target}' ignored.")
            continue

        for city, g in grouped_rows:
            valid = g.copy()
            valid["sort_key"] = valid.apply(metric_sort_values, axis=1)
            valid = valid.sort_values("sort_key")
            winner = valid.iloc[0].drop(labels=["sort_key"]).copy()
            winner["selection_city"] = city
            winners.append(winner)

    if not winners:
        raise RuntimeError("No winner rows could be selected.")

    return pd.DataFrame(winners).reset_index(drop=True)


# ---------------------------------------------------------------------
# Prediction preparation
# ---------------------------------------------------------------------
def choose_existing_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_prediction_source(
    df: pd.DataFrame,
    source_alias: str,
) -> pd.DataFrame:
    work = df.copy()
    rename_map = {}

    for target, candidates in TARGET_COLUMN_CANDIDATES.items():
        existing = choose_existing_column(work, candidates)
        if existing:
            rename_map[existing] = (
                f"{FINAL_STANDARD_COLUMNS[target]}__{source_alias}"
            )

    for target, companions in COMPANION_COLUMN_CANDIDATES.items():
        for col in companions:
            if col in work.columns:
                rename_map[col] = f"{col}__{source_alias}"

    keep = ["city", "timestamp"] + list(rename_map.keys())
    return work[keep].rename(columns=rename_map).copy()


def merge_prediction_sources(
    prepared_dfs: List[pd.DataFrame],
) -> pd.DataFrame:
    merged = prepared_dfs[0].copy()
    for df in prepared_dfs[1:]:
        merged = merged.merge(
            df,
            on=["city", "timestamp"],
            how="outer",
        )
    return (
        merged.sort_values(["city", "timestamp"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Apply selected models
# ---------------------------------------------------------------------
def apply_global_selection(
    result: pd.DataFrame,
    merged: pd.DataFrame,
    winners: pd.DataFrame,
    target: str,
) -> None:
    rows = winners[winners["target_name"] == target]
    if rows.empty:
        raise KeyError(f"No selected model for target: {target}")

    row = rows.iloc[0]
    source = str(row["source_model"])
    final_col = FINAL_STANDARD_COLUMNS[target]
    source_col = f"{final_col}__{source}"

    if source_col not in merged.columns:
        raise KeyError(
            f"Selected prediction column missing: {source_col}"
        )

    result[final_col] = merged[source_col]
    result[f"{target}_source_model"] = source

    for companion in COMPANION_COLUMN_CANDIDATES.get(target, []):
        source_comp = f"{companion}__{source}"
        if source_comp in merged.columns:
            result[companion] = merged[source_comp]


def apply_city_selection(
    result: pd.DataFrame,
    merged: pd.DataFrame,
    winners: pd.DataFrame,
    target: str,
) -> None:
    final_col = FINAL_STANDARD_COLUMNS[target]
    result[final_col] = np.nan
    result[f"{target}_source_model"] = None

    target_rows = winners[winners["target_name"] == target]

    for _, row in target_rows.iterrows():
        city = str(row["selection_city"])
        source = str(row["source_model"])
        source_col = f"{final_col}__{source}"

        if source_col not in merged.columns:
            print(
                f"[WARN] Missing {source_col}; skipping "
                f"{target} for {city}."
            )
            continue

        mask = result["city"].astype(str) == city
        result.loc[mask, final_col] = merged.loc[mask, source_col]
        result.loc[
            mask, f"{target}_source_model"
        ] = source

        for companion in COMPANION_COLUMN_CANDIDATES.get(target, []):
            source_comp = f"{companion}__{source}"
            if source_comp in merged.columns:
                if companion not in result.columns:
                    result[companion] = np.nan
                result.loc[mask, companion] = merged.loc[
                    mask, source_comp
                ]


def compute_drought_duration(df: pd.DataFrame) -> pd.DataFrame:
    if "drought_flag_pred" not in df.columns:
        return df

    out = df.sort_values(["city", "timestamp"]).copy()
    out["drought_duration_pred"] = 0

    for _, idx in out.groupby("city").groups.items():
        ids = list(idx)
        flags = (
            pd.to_numeric(
                out.loc[ids, "drought_flag_pred"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        durations = np.zeros(len(flags), dtype=int)

        for i in range(len(flags)):
            if flags[i] == 1:
                j = i
                while j < len(flags) and flags[j] == 1:
                    durations[i] += 1
                    j += 1

        out.loc[ids, "drought_duration_pred"] = durations

    return out


def build_combined_predictions(
    merged: pd.DataFrame,
    winners: pd.DataFrame,
) -> pd.DataFrame:
    result = merged[["city", "timestamp"]].copy()

    apply_global_selection(
        result, merged, winners, "drought_flag"
    )
    apply_global_selection(
        result, merged, winners, "drought_severity"
    )
    apply_global_selection(
        result, merged, winners, "precipitation_sum"
    )
    apply_city_selection(
        result, merged, winners, "dust_event"
    )

    if (
        "precipitation_sum_pred" in result.columns
        and "precipitation_sum_normal" in result.columns
    ):
        pred = pd.to_numeric(
            result["precipitation_sum_pred"],
            errors="coerce",
        )
        normal = pd.to_numeric(
            result["precipitation_sum_normal"],
            errors="coerce",
        )
        result["precip_deficit_pred"] = normal - pred

    return compute_drought_duration(result)


def build_winner_summary(
    winners: pd.DataFrame,
    all_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, win in winners.iterrows():
        target = canonical_target_name(win["target_name"])
        city = str(win.get("selection_city", "ALL"))
        source = str(win["source_model"])

        test_match = all_metrics[
            (all_metrics["target_name"].map(canonical_target_name) == target)
            & (all_metrics["source_model"].astype(str) == source)
        ].copy()

        if target == "dust_event":
            test_match = test_match[
                test_match["city"].astype(str) == city
            ]

        if "split" in test_match.columns:
            test_rows = test_match[
                test_match["split"].astype(str).str.lower() == "test"
            ]
        else:
            test_rows = pd.DataFrame()

        test_row = test_rows.iloc[0] if not test_rows.empty else None

        rows.append({
            "target_name": target,
            "city": city,
            "selected_model": source,
            "selection_split": str(win.get("split", "validation")),
            "validation_f1": win.get("f1_score"),
            "validation_rmse": win.get("rmse"),
            "validation_r2": win.get("r2_score"),
            "test_f1": None if test_row is None else test_row.get("f1_score"),
            "test_rmse": None if test_row is None else test_row.get("rmse"),
            "test_r2": None if test_row is None else test_row.get("r2_score"),
            "test_precision": None if test_row is None else test_row.get("precision"),
            "test_recall": None if test_row is None else test_row.get("recall"),
            "test_tn": None if test_row is None else test_row.get("tn"),
            "test_fp": None if test_row is None else test_row.get("fp"),
            "test_fn": None if test_row is None else test_row.get("fn"),
            "test_tp": None if test_row is None else test_row.get("tp"),
        })

    return pd.DataFrame(rows).sort_values(
        ["target_name", "city"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "LOCAL Windows combiner for revised RF, XGBoost and LSTM "
            "metrics/predictions."
        )
    )

    parser.add_argument(
        "--rf_metrics",
        default=DEFAULT_FILES["rf_metrics"].name,
    )
    parser.add_argument(
        "--rf_predictions",
        default=DEFAULT_FILES["rf_predictions"].name,
    )
    parser.add_argument(
        "--xgb_metrics",
        default=DEFAULT_FILES["xgb_metrics"].name,
    )
    parser.add_argument(
        "--xgb_predictions",
        default=DEFAULT_FILES["xgb_predictions"].name,
    )
    parser.add_argument(
        "--lstm_metrics",
        default=DEFAULT_FILES["lstm_metrics"].name,
    )
    parser.add_argument(
        "--lstm_predictions",
        default=DEFAULT_FILES["lstm_predictions"].name,
    )

    parser.add_argument(
        "--selection_split",
        choices=["validation", "test"],
        default="validation",
        help=(
            "Use validation for reviewer-safe model selection. "
            "Do not use test unless intentionally doing exploratory analysis."
        ),
    )
    parser.add_argument(
        "--output",
        default="combined_best_predictions_LOCAL.csv",
    )
    parser.add_argument(
        "--winners_output",
        default="combined_best_model_selection_LOCAL.csv",
    )
    args = parser.parse_args()

    sources = [
        (
            "random_forest",
            resolve_local_path(args.rf_metrics),
            resolve_local_path(args.rf_predictions),
        ),
        (
            "xgboost",
            resolve_local_path(args.xgb_metrics),
            resolve_local_path(args.xgb_predictions),
        ),
        (
            "lstm",
            resolve_local_path(args.lstm_metrics),
            resolve_local_path(args.lstm_predictions),
        ),
    ]

    metrics_frames = []
    prepared_predictions = []

    print("[INFO] Loading LOCAL revised metrics and predictions...")
    for alias, metrics_path, pred_path in sources:
        print(f"[INFO] {alias}")
        print(f"       metrics     : {metrics_path}")
        print(f"       predictions : {pred_path}")

        metrics_frames.append(
            load_metrics(metrics_path, alias)
        )
        pred_df = load_predictions(pred_path, alias)
        prepared_predictions.append(
            prepare_prediction_source(pred_df, alias)
        )

    all_metrics = pd.concat(metrics_frames, ignore_index=True)
    winners = choose_best_metrics_rows(
        all_metrics,
        selection_split=args.selection_split,
    )

    merged_predictions = merge_prediction_sources(
        prepared_predictions
    )
    combined = build_combined_predictions(
        merged_predictions,
        winners,
    )
    winner_summary = build_winner_summary(
        winners,
        all_metrics,
    )

    output_path = resolve_local_path(args.output)
    winners_path = resolve_local_path(args.winners_output)

    combined.to_csv(output_path, index=False)
    winner_summary.to_csv(winners_path, index=False)

    print("\n[INFO] SELECTED MODELS")
    print(winner_summary.to_string(index=False))

    print("\n[INFO] Done.")
    print(f"[INFO] Combined predictions: {output_path}")
    print(f"[INFO] Winner summary      : {winners_path}")
    print("\n[INFO] First combined rows:")
    print(combined.head().to_string(index=False))


if __name__ == "__main__":
    main()
