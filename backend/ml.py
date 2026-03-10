from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "models" / "forecast_model.joblib"
FEATURE_COLUMNS_PATH = ROOT / "data" / "models" / "feature_columns.json"


def _load_model_and_features():
    if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run scripts/train_forecast_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    return model, feature_columns


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
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


def _build_hourly_profile(history: pd.DataFrame, col: str) -> dict[int, float]:
    tmp = history.copy()
    tmp["hour"] = pd.to_datetime(tmp["timestamp"]).dt.hour
    return tmp.groupby("hour")[col].mean().to_dict()


def _feature_row_from_history(
    history: pd.DataFrame,
    ts: pd.Timestamp,
    site_id: str,
    temperature_c: float,
    electricity_price_usd_mwh: float,
    carbon_intensity_kgco2_mwh: float,
    outage_risk_score: float,
) -> pd.DataFrame:
    h = history.sort_values("timestamp").reset_index(drop=True).copy()
    demand_series = h["demand_mw"].astype(float)

    lag_1 = float(demand_series.iloc[-1])
    lag_24 = float(demand_series.iloc[-24]) if len(h) >= 24 else float(demand_series.mean())
    lag_168 = float(demand_series.iloc[-168]) if len(h) >= 168 else float(demand_series.mean())
    roll_mean_24 = float(demand_series.iloc[-24:].mean()) if len(h) >= 24 else float(demand_series.mean())
    roll_mean_168 = float(demand_series.iloc[-168:].mean()) if len(h) >= 168 else float(demand_series.mean())

    row = pd.DataFrame(
        {
            "timestamp": [pd.to_datetime(ts)],
            "temperature_c": [temperature_c],
            "electricity_price_usd_mwh": [electricity_price_usd_mwh],
            "carbon_intensity_kgco2_mwh": [carbon_intensity_kgco2_mwh],
            "outage_risk_score": [outage_risk_score],
            "demand_mw": [lag_1],
            "lag_1": [lag_1],
            "lag_24": [lag_24],
            "lag_168": [lag_168],
            "roll_mean_24": [roll_mean_24],
            "roll_mean_168": [roll_mean_168],
            "site_id": [site_id],
        }
    )
    row = _add_time_features(row)
    row = pd.get_dummies(row, columns=["site_id"], drop_first=False)
    return row


def forecast_demand(history: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    model, feature_columns = _load_model_and_features()
    if history.empty:
        raise ValueError("History is empty")

    hist = history.sort_values("timestamp").reset_index(drop=True).copy()
    site_id = str(hist["site_id"].iloc[0])
    last_ts = pd.to_datetime(hist["timestamp"].iloc[-1])

    temp_profile = _build_hourly_profile(hist, "temperature_c")
    price_profile = _build_hourly_profile(hist, "electricity_price_usd_mwh")
    carbon_profile = _build_hourly_profile(hist, "carbon_intensity_kgco2_mwh")
    risk_profile = _build_hourly_profile(hist, "outage_risk_score")

    fallback_temp = float(hist["temperature_c"].iloc[-24:].mean())
    fallback_price = float(hist["electricity_price_usd_mwh"].iloc[-24:].mean())
    fallback_carbon = float(hist["carbon_intensity_kgco2_mwh"].iloc[-24:].mean())
    fallback_risk = float(hist["outage_risk_score"].iloc[-24:].mean())

    future_rows: list[dict] = []

    for _ in range(horizon_hours):
        next_ts = last_ts + pd.Timedelta(hours=1)
        hour = next_ts.hour

        future_temp = float(temp_profile.get(hour, fallback_temp))
        future_price = float(price_profile.get(hour, fallback_price))
        future_carbon = float(carbon_profile.get(hour, fallback_carbon))
        future_risk = float(risk_profile.get(hour, fallback_risk))

        row = _feature_row_from_history(
            hist,
            next_ts,
            site_id,
            future_temp,
            future_price,
            future_carbon,
            future_risk,
        )
        X = row.reindex(columns=feature_columns, fill_value=0)
        pred = float(model.predict(X)[0])
        pred = max(pred, 0.1)

        template = hist.iloc[-1].copy()
        template["timestamp"] = next_ts
        template["demand_mw"] = pred
        template["temperature_c"] = future_temp
        template["electricity_price_usd_mwh"] = future_price
        template["carbon_intensity_kgco2_mwh"] = future_carbon
        template["outage_risk_score"] = future_risk

        hist = pd.concat([hist, pd.DataFrame([template])], ignore_index=True)

        future_rows.append(
            {
                "timestamp": next_ts,
                "site_id": site_id,
                "demand_mw": pred,
                "temperature_c": future_temp,
                "electricity_price_usd_mwh": future_price,
                "carbon_intensity_kgco2_mwh": future_carbon,
                "outage_risk_score": future_risk,
            }
        )
        last_ts = next_ts

    return pd.DataFrame(future_rows)