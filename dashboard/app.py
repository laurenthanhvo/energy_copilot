from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Energy Copilot Dashboard", layout="wide")
st.title("Energy Reliability & Dispatch Copilot")
st.caption("Forecasting, anomaly detection, KPI analytics, and battery dispatch optimization")

with st.expander("About this project"):
    st.markdown("""
    **Motivation**
    - Industrial energy operators need fast ways to understand demand, cost, emissions, and storage decisions.

    **What this demo includes**
    - Synthetic site telemetry with missing data, duplicates, and outliers
    - Forecasting, anomaly detection, KPI computation, and battery dispatch recommendation
    - FastAPI backend + Streamlit dashboard + lightweight natural-language interface

    **Current assumptions**
    - Uses synthetic but realistic-looking data
    - Battery dispatch is heuristic, not a full mathematical optimizer
    - Results depend on the chosen dispatch mode and forecast horizon

    **Next steps**
    - Swap in real public energy/load/weather/carbon datasets
    - Add baseline model comparisons
    - Replace heuristic dispatch with formal constrained optimization
    """)

def api_get(path: str, params: dict | None = None) -> dict:
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


sites = api_get("/sites")["sites"]
site_id = st.sidebar.selectbox("Site", sites)
lookback_hours = st.sidebar.slider("Lookback hours", min_value=24, max_value=24 * 30, value=24 * 7, step=24)
horizon_hours = st.sidebar.slider("Forecast horizon", min_value=6, max_value=168, value=24, step=6)
initial_soc_mwh = st.sidebar.slider(
    "Initial battery SOC (MWh)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.5,
)

dispatch_mode = st.sidebar.selectbox(
    "Dispatch mode",
    ["balanced", "cost_saving", "carbon_aware", "peak_shaving"],
    index=0,
)

kpi_data = api_get("/kpis", {"site_id": site_id, "lookback_hours": lookback_hours})
series = pd.DataFrame(api_get("/timeseries", {"site_id": site_id, "hours": lookback_hours})["rows"])
forecast = pd.DataFrame(api_get("/forecast", {"site_id": site_id, "horizon_hours": horizon_hours})["rows"])
anomaly_data = api_get("/anomalies", {"site_id": site_id, "lookback_hours": lookback_hours})
anomalies = pd.DataFrame(anomaly_data["rows"])

series["timestamp"] = pd.to_datetime(series["timestamp"])
forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
if not anomalies.empty:
    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg demand (MW)", f"{kpi_data['avg_demand_mw']:.2f}")
c2.metric("Peak demand (MW)", f"{kpi_data['peak_demand_mw']:.2f}")
c3.metric(
    f"Total cost over {lookback_hours}h ($)",
    f"{kpi_data.get('total_cost_selected_window_usd', kpi_data.get('estimated_cost_usd', 0)):.0f}",
)
c4.metric("Spike hours", f"{kpi_data['spike_hours']}")

hist_fig = px.line(series, x="timestamp", y="demand_mw", title="Historical demand")
st.plotly_chart(hist_fig, use_container_width=True)

forecast_fig = px.line(forecast, x="timestamp", y="demand_mw", title="Demand forecast")
st.plotly_chart(forecast_fig, use_container_width=True)

col_a, col_b = st.columns([1.2, 1])
with col_a:
    st.subheader("Detected anomalies")
    if anomalies.empty:
        st.info("No anomalies detected in the selected window.")
    else:
        st.dataframe(anomalies.sort_values("timestamp", ascending=False), use_container_width=True)

with col_b:
    st.subheader("Natural-language query")
    question = st.text_input(
        "Ask a question",
        value=f"forecast next {horizon_hours} hours for {site_id}",
    )

    if st.button("Run query"):
        answer = api_post("/ask", {"question": question})

        parsed = answer.get("parsed", {})
        response = answer.get("response", {})
        intent = parsed.get("intent", "unknown")

        st.caption(f"Intent: {intent} | Site: {parsed.get('site_id', 'unknown')}")

        if intent == "forecast":
            rows = response.get("rows", [])
            forecast_df = pd.DataFrame(rows)

            if forecast_df.empty:
                st.info("No forecast rows returned.")
            else:
                forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
                st.success(
                    f"Forecast generated for {response.get('horizon_hours', len(forecast_df))} hours."
                )

                preview_cols = [
                    c for c in ["timestamp", "site_id", "demand_mw"] if c in forecast_df.columns
                ]
                st.dataframe(
                    forecast_df[preview_cols].head(10),
                    use_container_width=True,
                    height=260,
                )

                with st.expander("Show forecast chart", expanded=True):
                    query_fig = px.line(
                        forecast_df,
                        x="timestamp",
                        y="demand_mw",
                        title="Query result: forecasted demand",
                    )
                    st.plotly_chart(query_fig, use_container_width=True)

        elif intent == "anomalies":
            rows = response.get("rows", [])
            anomalies_df = pd.DataFrame(rows)

            st.info(f"Detected {response.get('count', len(anomalies_df))} anomalies.")
            if not anomalies_df.empty:
                if "timestamp" in anomalies_df.columns:
                    anomalies_df["timestamp"] = pd.to_datetime(anomalies_df["timestamp"])
                st.dataframe(anomalies_df, use_container_width=True, height=260)

        elif intent == "recommendation":
            summary = response.get("summary", {})
            st.success("Dispatch recommendation generated.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Cost savings ($)", f"{summary.get('cost_savings_usd', 0):.2f}")
            m2.metric(
                "Emissions reduction (kg)",
                f"{summary.get('emissions_reduction_kg', 0):.2f}",
            )
            m3.metric("Peak reduction (MW)", f"{summary.get('peak_reduction_mw', 0):.2f}")

        elif intent == "kpis":
            st.success("KPI summary generated.")

            m1, m2 = st.columns(2)
            m1.metric("Avg demand (MW)", f"{response.get('avg_demand_mw', 0):.2f}")
            m2.metric("Peak demand (MW)", f"{response.get('peak_demand_mw', 0):.2f}")

            m3, m4 = st.columns(2)
            m3.metric(
                "Total cost in selected window ($)",
                f"{response.get('total_cost_selected_window_usd', response.get('estimated_cost_usd', 0)):.0f}",
            )
            m4.metric("Spike hours", f"{response.get('spike_hours', 0)}")

        else:
            st.warning("Could not interpret the query cleanly.")
            with st.expander("Raw response"):
                st.json(answer)

st.subheader("Battery dispatch recommendation")
rec = api_post(
    "/recommendation",
    {
        "site_id": site_id,
        "horizon_hours": horizon_hours,
        "battery_mwh": 20.0,
        "max_charge_mw": 5.0,
        "max_discharge_mw": 5.0,
        "soc0_mwh": initial_soc_mwh,
        "emissions_weight": 0.02,
        "peak_demand_weight": 15.0,
        "mode": dispatch_mode,
    },
)

summary = rec["summary"]
rec_df = pd.DataFrame(rec["rows"])
rec_df["timestamp"] = pd.to_datetime(rec_df["timestamp"])

active_dispatch = rec_df[
    (rec_df["charge_mw"] > 0) | (rec_df["discharge_mw"] > 0)
].copy()

with st.expander("Show active battery events"):
    if active_dispatch.empty:
        st.info("No charging or discharging events in this schedule.")
    else:
        st.dataframe(
            active_dispatch[
                [
                    "timestamp",
                    "demand_mw",
                    "electricity_price_usd_mwh",
                    "charge_mw",
                    "discharge_mw",
                    "grid_mw",
                    "soc_mwh",
                ]
            ],
            use_container_width=True,
        )

r1, r2, r3, r4 = st.columns(4)
r1.metric("Cost savings ($)", f"{summary['cost_savings_usd']:.2f}")
r2.metric("Emissions reduction (kg)", f"{summary['emissions_reduction_kg']:.2f}")
r3.metric("Peak reduction (MW)", f"{summary['peak_reduction_mw']:.2f}")
r4.metric("Optimized peak (MW)", f"{summary['peak_grid_mw']:.2f}")

r5, r6, r7, r8 = st.columns(4)
r5.metric("Baseline peak (MW)", f"{summary['baseline_peak_mw']:.2f}")
r6.metric("Hours charged", f"{summary.get('hours_charged', 0)}")
r7.metric("Hours discharged", f"{summary.get('hours_discharged', 0)}")
r8.metric("Optimized cost ($)", f"{summary['optimized_cost_usd']:.2f}")

dispatch_fig = px.line(
    rec_df,
    x="timestamp",
    y=["demand_mw", "grid_mw", "charge_mw", "discharge_mw", "soc_mwh"],
    title="Optimized dispatch schedule",
)
st.plotly_chart(dispatch_fig, use_container_width=True)