from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV_PATH = RAW_DIR / "site_telemetry_raw.csv"


def main() -> None:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2025-01-01", periods=24 * 210, freq="h")

    site_configs = {
        "alpha": {"base_load": 18, "temp_sensitivity": 0.65, "price_zone": 1.00},
        "bravo": {"base_load": 24, "temp_sensitivity": 0.55, "price_zone": 1.06},
        "charlie": {"base_load": 14, "temp_sensitivity": 0.80, "price_zone": 0.96},
        "delta": {"base_load": 28, "temp_sensitivity": 0.48, "price_zone": 1.10},
    }

    rows = []
    for site_id, cfg in site_configs.items():
        for ts in timestamps:
            hour = ts.hour
            dow = ts.dayofweek
            doy = ts.day_of_year
            weekend = dow >= 5

            seasonal_temp = 17 + 9 * np.sin(2 * np.pi * doy / 365)
            daily_temp = 6 * np.sin(2 * np.pi * (hour - 8) / 24)
            temperature_c = seasonal_temp + daily_temp + rng.normal(0, 1.4)

            business_load = 4.5 if 8 <= hour <= 18 and not weekend else -1.5
            temp_effect = max(temperature_c - 22, 0) * cfg["temp_sensitivity"] + max(10 - temperature_c, 0) * 0.2
            evening_peak = 5.5 if 17 <= hour <= 21 else 0.0
            noise = rng.normal(0, 1.3)

            demand_mw = cfg["base_load"] + business_load + temp_effect + evening_peak + noise
            if rng.random() < 0.008:
                demand_mw += rng.uniform(8, 16)

            electricity_price_usd_mwh = (
                28
                + 1.25 * demand_mw
                + (12 if 17 <= hour <= 20 else 0)
                + rng.normal(0, 3)
            ) * cfg["price_zone"]
            electricity_price_usd_mwh = max(electricity_price_usd_mwh, 5)

            carbon_intensity_kgco2_mwh = max(
                180 + 0.9 * demand_mw - (15 if 10 <= hour <= 15 else 0) + rng.normal(0, 10),
                80,
            )
            outage_risk_score = np.clip(
                0.18 + 0.015 * max(temperature_c - 30, 0) + 0.012 * max(demand_mw - 32, 0) + rng.normal(0, 0.015),
                0.02,
                0.95,
            )

            rows.append(
                {
                    "timestamp": ts,
                    "site_id": site_id,
                    "temperature_c": round(float(temperature_c), 3),
                    "demand_mw": round(float(max(demand_mw, 3)), 3),
                    "electricity_price_usd_mwh": round(float(electricity_price_usd_mwh), 3),
                    "carbon_intensity_kgco2_mwh": round(float(carbon_intensity_kgco2_mwh), 3),
                    "outage_risk_score": round(float(outage_risk_score), 4),
                }
            )

            df = pd.DataFrame(rows)

    missing_mask = rng.random(len(df)) < 0.01
    df.loc[missing_mask, "temperature_c"] = np.nan
    missing_mask = rng.random(len(df)) < 0.01
    df.loc[missing_mask, "electricity_price_usd_mwh"] = np.nan
    missing_mask = rng.random(len(df)) < 0.004
    df.loc[missing_mask, "demand_mw"] = np.nan

    duplicate_rows = df.sample(frac=0.002, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    outlier_idx = df.sample(frac=0.003, random_state=7).index
    df.loc[outlier_idx, "demand_mw"] = df.loc[outlier_idx, "demand_mw"].fillna(0) * rng.uniform(1.8, 2.7)

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"Wrote synthetic raw data to {RAW_CSV_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()