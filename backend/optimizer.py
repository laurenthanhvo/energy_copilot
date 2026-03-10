from __future__ import annotations

import pandas as pd


def optimize_dispatch(
    forecast_df: pd.DataFrame,
    battery_mwh: float,
    max_charge_mw: float,
    max_discharge_mw: float,
    soc0_mwh: float,
    emissions_weight: float = 0.02,
    peak_demand_weight: float = 15.0,
):
    df = forecast_df.copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("forecast_df is empty")

    demand = df["demand_mw"].astype(float)
    price = df["electricity_price_usd_mwh"].astype(float)
    carbon = df["carbon_intensity_kgco2_mwh"].astype(float)

    charge_eff = 0.95
    discharge_eff = 0.95

    # Looser thresholds so the battery actually moves
    low_price = float(price.quantile(0.45))
    high_price = float(price.quantile(0.70))
    low_carbon = float(carbon.quantile(0.45))
    target_peak = float(demand.quantile(0.70))

    soc = min(max(soc0_mwh, 0.0), battery_mwh)
    schedule_rows = []

    baseline_cost = float((demand * price).sum())
    baseline_emissions = float((demand * carbon).sum())
    baseline_peak = float(demand.max())

    for i, row in df.iterrows():
        d = float(row["demand_mw"])
        p = float(row["electricity_price_usd_mwh"])
        c = float(row["carbon_intensity_kgco2_mwh"])

        charge_mw = 0.0
        discharge_mw = 0.0

        future_demand = demand.iloc[i + 1 :]
        future_price = price.iloc[i + 1 :]

        future_peak_exists = bool((future_demand > target_peak).any()) if len(future_demand) else False
        future_high_price_exists = bool((future_price >= high_price).any()) if len(future_price) else False

        # Keep some energy reserved if higher future peaks still exist
        reserve_soc_mwh = 0.30 * battery_mwh if future_peak_exists else 0.0

        # 1) Peak shaving has top priority
        if soc > 1e-9 and d > target_peak:
            desired_discharge = d - target_peak
            max_deliverable = soc * discharge_eff
            discharge_mw = min(max_discharge_mw, d, max_deliverable, desired_discharge)

        # 2) Charge during cheap OR clean hours if future valuable hours exist
        elif (
            soc < battery_mwh - 1e-9
            and (p <= low_price or c <= low_carbon)
            and (future_peak_exists or future_high_price_exists)
        ):
            remaining_capacity = (battery_mwh - soc) / charge_eff
            charge_mw = min(max_charge_mw, remaining_capacity)

        # 3) Discharge during expensive hours, but do not consume reserved energy
        elif soc > reserve_soc_mwh and p >= high_price:
            usable_soc = soc - reserve_soc_mwh
            max_deliverable = usable_soc * discharge_eff
            discharge_mw = min(max_discharge_mw, d, max_deliverable)

        soc = soc + charge_mw * charge_eff - discharge_mw / discharge_eff
        soc = min(max(soc, 0.0), battery_mwh)

        grid_mw = max(d + charge_mw - discharge_mw, 0.0)

        schedule_rows.append(
            {
                "timestamp": row["timestamp"],
                "site_id": row.get("site_id", "unknown"),
                "demand_mw": d,
                "electricity_price_usd_mwh": p,
                "carbon_intensity_kgco2_mwh": c,
                "charge_mw": charge_mw,
                "discharge_mw": discharge_mw,
                "grid_mw": grid_mw,
                "soc_mwh": soc,
            }
        )

    schedule = pd.DataFrame(schedule_rows)

    optimized_cost = float(
        (schedule["grid_mw"] * schedule["electricity_price_usd_mwh"]).sum()
    )
    optimized_emissions = float(
        (schedule["grid_mw"] * schedule["carbon_intensity_kgco2_mwh"]).sum()
    )
    optimized_peak = float(schedule["grid_mw"].max())

    summary = {
        "baseline_cost_usd": baseline_cost,
        "optimized_cost_usd": optimized_cost,
        "cost_savings_usd": baseline_cost - optimized_cost,
        "baseline_emissions_kg": baseline_emissions,
        "optimized_emissions_kg": optimized_emissions,
        "emissions_reduction_kg": baseline_emissions - optimized_emissions,
        "baseline_peak_mw": baseline_peak,
        "peak_grid_mw": optimized_peak,
        "peak_reduction_mw": baseline_peak - optimized_peak,
        "hours_charged": int((schedule["charge_mw"] > 0).sum()),
        "hours_discharged": int((schedule["discharge_mw"] > 0).sum()),
    }

    return schedule, summary