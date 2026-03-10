from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = ROOT / "data" / "raw" / "site_telemetry_raw.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "data" / "models"
PROCESSED_PARQUET_PATH = PROCESSED_DIR / "site_telemetry_clean.parquet"
DUCKDB_PATH = PROCESSED_DIR / "energy.duckdb"
MODEL_PATH = MODELS_DIR / "forecast_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = out["timestamp"].dt.hour
    dow = out["timestamp"].dt.dayofweek
    month = out["timestamp"].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["is_weekend"] = (dow >= 5).astype(int)
    return out

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "site_id"]).sort_values(["site_id", "timestamp"]).reset_index(drop=True)

    numeric_cols = [
        "temperature_c",
        "demand_mw",
        "electricity_price_usd_mwh",
        "carbon_intensity_kgco2_mwh",
        "outage_risk_score",
    ]

    cleaned_groups = []
    for _, g in df.groupby("site_id"):
        g = g.sort_values("timestamp").copy()
        g[numeric_cols] = g[numeric_cols].interpolate(method="linear", limit_direction="both")
        g[numeric_cols] = g[numeric_cols].ffill().bfill()
        q_low = g["demand_mw"].quantile(0.01)
        q_high = g["demand_mw"].quantile(0.99)
        g["demand_mw"] = g["demand_mw"].clip(lower=q_low, upper=q_high)
        cleaned_groups.append(g)
    return pd.concat(cleaned_groups, ignore_index=True)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("site_id")["demand_mw"]
    out["lag_1"] = g.shift(1)
    out["lag_24"] = g.shift(24)
    out["lag_168"] = g.shift(168)
    out["roll_mean_24"] = g.shift(1).rolling(24).mean().reset_index(level=0, drop=True)
    out["roll_mean_168"] = g.shift(1).rolling(168).mean().reset_index(level=0, drop=True)
    return out

def main() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            "Raw CSV not found. Run scripts/generate_synthetic_data.py first."
        )

    print("Loading raw data...")
    df = pd.read_csv(RAW_CSV_PATH)
    df = clean_data(df)
    print(f"After cleaning: {df.shape}")
    df = add_time_features(df)
    df = add_lag_features(df)
    df = df.dropna().reset_index(drop=True)
    print(f"After feature engineering: {df.shape}")

    feature_df = pd.get_dummies(df, columns=["site_id"], drop_first=False)

    target_col = "demand_mw"
    feature_cols = [
        c for c in feature_df.columns
        if c not in {"timestamp", target_col}
    ]

    split_ts = feature_df["timestamp"].quantile(0.8)
    train_mask = feature_df["timestamp"] <= split_ts
    valid_mask = feature_df["timestamp"] > split_ts

    X_train = feature_df.loc[train_mask, feature_cols]
    y_train = feature_df.loc[train_mask, target_col]
    X_valid = feature_df.loc[valid_mask, feature_cols]
    y_valid = feature_df.loc[valid_mask, target_col]

    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("Training forecast model...")
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)

    rmse = float(np.sqrt(mean_squared_error(y_valid, preds)))
    mae = float(mean_absolute_error(y_valid, preds))
    mape = float(np.mean(np.abs((y_valid - preds) / np.maximum(y_valid, 1e-6))) * 100)

    df.to_parquet(PROCESSED_PARQUET_PATH, index=False)
    conn = duckdb.connect(str(DUCKDB_PATH))
    conn.register("df_view", df)
    conn.execute("CREATE OR REPLACE TABLE telemetry AS SELECT * FROM df_view")
    conn.close()

    joblib.dump(model, MODEL_PATH)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_cols, indent=2))
    METRICS_PATH.write_text(json.dumps({"mae": mae, "rmse": rmse, "mape_pct": mape}, indent=2))

    print(f"Saved processed parquet to {PROCESSED_PARQUET_PATH}")
    print(f"Saved DuckDB warehouse to {DUCKDB_PATH}")
    print(f"Saved model to {MODEL_PATH}")
    print(json.dumps({"mae": mae, "rmse": rmse, "mape_pct": mape}, indent=2))


if __name__ == "__main__":
    main()