from __future__ import annotations

from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from backend.data_store import available_sites, latest_timestamp, read_recent_window, read_site_history
from backend.ml import forecast_demand
from backend.nlp import parse_question
from backend.optimizer import optimize_dispatch
from backend.schemas import AskRequest, RecommendationRequest

app = FastAPI(title="Energy Reliability & Dispatch Copilot", version="0.1.0")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Energy Reliability & Dispatch Copilot API",
        "sites": available_sites(),
        "latest_timestamp": str(latest_timestamp()),
    }

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/sites")
def sites() -> Dict[str, Any]:
    return {"sites": available_sites()}


@app.get("/timeseries")
def timeseries(site_id: str = Query(...), hours: int = Query(168, ge=24, le=24 * 90)) -> Dict[str, Any]:
    try:
        df = read_recent_window(site_id=site_id, hours=hours)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"rows": df.to_dict(orient="records")}

@app.get("/kpis")
def kpis(site_id: str = Query(...), lookback_hours: int = Query(168, ge=24, le=24 * 90)) -> Dict[str, Any]:
    try:
        df = read_recent_window(site_id=site_id, hours=lookback_hours)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    total_cost = float((df["demand_mw"] * df["electricity_price_usd_mwh"]).sum())
    total_emissions = float((df["demand_mw"] * df["carbon_intensity_kgco2_mwh"]).sum())
    avg_demand = float(df["demand_mw"].mean())
    peak_demand = float(df["demand_mw"].max())
    load_factor = avg_demand / peak_demand if peak_demand > 0 else 0.0
    volatility = float(df["demand_mw"].std())
    anomaly_threshold = df["demand_mw"].mean() + 2.5 * df["demand_mw"].std()
    spike_hours = int((df["demand_mw"] > anomaly_threshold).sum())

    total_energy_mwh = float(df["demand_mw"].sum())
    avg_hourly_cost = total_cost / max(lookback_hours, 1)
    cost_per_mwh = total_cost / max(total_energy_mwh, 1e-6)

    return {
        "site_id": site_id,
        "lookback_hours": lookback_hours,
        "avg_demand_mw": avg_demand,
        "peak_demand_mw": peak_demand,
        "load_factor": load_factor,
        "volatility_mw": volatility,
        "total_cost_selected_window_usd": total_cost,
        "avg_hourly_cost_usd": avg_hourly_cost,
        "cost_per_mwh_usd": cost_per_mwh,
        "estimated_emissions_kg": total_emissions,
        "spike_hours": spike_hours,
    }

@app.get("/forecast")
def forecast(site_id: str = Query(...), horizon_hours: int = Query(24, ge=6, le=168)) -> Dict[str, Any]:
    try:
        history = read_site_history(site_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    forecast_df = forecast_demand(history, horizon_hours=horizon_hours)
    return {
        "site_id": site_id,
        "horizon_hours": horizon_hours,
        "rows": forecast_df.to_dict(orient="records"),
    }

@app.get("/anomalies")
def anomalies(site_id: str = Query(...), lookback_hours: int = Query(168, ge=24, le=24 * 90)) -> Dict[str, Any]:
    try:
        df = read_recent_window(site_id=site_id, hours=lookback_hours)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    rolling_mean = df["demand_mw"].rolling(window=24, min_periods=12).mean()
    rolling_std = df["demand_mw"].rolling(window=24, min_periods=12).std().replace(0, np.nan)
    zscore = (df["demand_mw"] - rolling_mean) / rolling_std
    out = df.copy()
    out["zscore"] = zscore.fillna(0.0)
    anomalies_df = out[np.abs(out["zscore"]) >= 2.5].copy()
    anomalies_df["severity"] = np.where(np.abs(anomalies_df["zscore"]) >= 3.5, "high", "medium")

    return {
        "site_id": site_id,
        "lookback_hours": lookback_hours,
        "count": int(len(anomalies_df)),
        "rows": anomalies_df[["timestamp", "demand_mw", "zscore", "severity"]].to_dict(orient="records"),
    }

@app.post("/recommendation")
def recommendation(req: RecommendationRequest) -> Dict[str, Any]:
    try:
        history = read_site_history(req.site_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    forecast_df = forecast_demand(history, horizon_hours=req.horizon_hours)
    schedule, summary = optimize_dispatch(
        forecast_df=forecast_df,
        battery_mwh=req.battery_mwh,
        max_charge_mw=req.max_charge_mw,
        max_discharge_mw=req.max_discharge_mw,
        soc0_mwh=req.soc0_mwh,
        emissions_weight=req.emissions_weight,
        peak_demand_weight=req.peak_demand_weight,
        mode=req.mode,
    )

    return {
        "site_id": req.site_id,
        "summary": summary,
        "rows": schedule.to_dict(orient="records"),
    }

@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    parsed = parse_question(req.question)
    intent = parsed["intent"]
    site_id = str(parsed["site_id"])

    if intent == "forecast":
        return {
            "parsed": parsed,
            "response": forecast(site_id=site_id, horizon_hours=int(parsed["horizon_hours"])),
        }
    if intent == "anomalies":
        return {
            "parsed": parsed,
            "response": anomalies(site_id=site_id, lookback_hours=int(parsed["lookback_hours"])),
        }
    if intent == "recommendation":
        rec = RecommendationRequest(site_id=site_id, horizon_hours=int(parsed["horizon_hours"]))
        return {
            "parsed": parsed,
            "response": recommendation(rec),
        }
    return {
        "parsed": parsed,
        "response": kpis(site_id=site_id, lookback_hours=int(parsed["lookback_hours"])),
    }