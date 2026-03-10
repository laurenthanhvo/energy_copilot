from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PARQUET_PATH = ROOT / "data" / "processed" / "site_telemetry_clean.parquet"

def _load_data() -> pd.DataFrame:
    if not PROCESSED_PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_PARQUET_PATH}. "
            "Run scripts/generate_synthetic_data.py and scripts/train_forecast_model.py first."
        )
    df = pd.read_parquet(PROCESSED_PARQUET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)


def available_sites() -> list[str]:
    df = _load_data()
    return sorted(df["site_id"].dropna().unique().tolist())


def latest_timestamp() -> pd.Timestamp:
    df = _load_data()
    return pd.to_datetime(df["timestamp"]).max()


def read_site_history(site_id: str) -> pd.DataFrame:
    df = _load_data()
    out = df[df["site_id"] == site_id].copy()
    if out.empty:
        raise ValueError(f"Unknown site_id: {site_id}")
    return out.sort_values("timestamp").reset_index(drop=True)


def read_recent_window(site_id: str, hours: int) -> pd.DataFrame:
    df = read_site_history(site_id)
    if hours >= len(df):
        return df.copy()
    return df.iloc[-hours:].reset_index(drop=True)