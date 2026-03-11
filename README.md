# Energy Reliability & Dispatch Copilot

An end-to-end energy analytics and decision-support project built with **Python, FastAPI, Streamlit, DuckDB, scikit-learn, pandas, and Plotly**.

This project simulates multi-site hourly energy telemetry, cleans imperfect time-series data, forecasts site demand, detects anomalies, computes operational KPIs, and recommends battery dispatch schedules under different optimization modes.

## Motivation

Industrial and distributed energy systems generate large amounts of operational telemetry, but raw time-series data alone is not enough to support fast decision-making.

This project was built to explore how a data product can turn messy telemetry into operational insights by combining:

- data cleaning and feature engineering
- machine learning forecasting
- anomaly detection
- KPI generation
- battery dispatch decision support
- REST APIs and interactive dashboards

The goal is to demonstrate a full analytics workflow that resembles the kind of decision-support tooling used in energy, operations, and computational systems teams.

## What the project does

The application supports several workflows for a selected site:

- **Historical telemetry analysis** for hourly demand trends
- **KPI generation** for average demand, peak demand, estimated cost, and spike hours
- **Demand forecasting** across user-selected forecast horizons
- **Anomaly detection** on recent demand behavior
- **Battery dispatch recommendation** to simulate charge/discharge schedules under different modes:
  - `balanced`
  - `cost_saving`
  - `carbon_aware`
  - `peak_shaving`
- **Natural-language query support** for simple analytics questions

## Important note on the data

This MVP currently uses **synthetic but structured energy telemetry**, not real utility or plant data.

The dataset is simulated to look realistic and includes:

- multiple sites (`alpha`, `bravo`, `charlie`, `delta`)
- hourly timestamps
- temperature
- site demand
- electricity price
- carbon intensity
- outage risk score
- missing values, duplicates, and outliers

So while the data is simulated, the pipeline itself: the backend, dashboard, feature engineering, model training, anomaly logic, and dispatch engine all run end to end on a consistent dataset.

## Tech stack

- **Backend:** FastAPI
- **Dashboard:** Streamlit
- **Storage / analytics:** DuckDB, Parquet
- **ML / forecasting:** scikit-learn
- **Data processing:** pandas, NumPy
- **Visualization:** Plotly
- **Containerization:** Docker / Docker Compose

## Repository structure

```text
energy_copilot/
├── backend/
│   ├── main.py
│   ├── data_store.py
│   ├── ml.py
│   ├── nlp.py
│   ├── optimizer.py
│   └── schemas.py
├── dashboard/
│   └── app.py
├── scripts/
│   ├── generate_synthetic_data.py
│   └── train_forecast_model.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── requirements.txt
├── docker-compose.yml
└── README.md

## How it works

### 1. Synthetic telemetry generation
A script generates hourly multi-site telemetry with realistic structure and injected data-quality issues.

### 2. Data cleaning and feature engineering
The training pipeline:

- removes duplicates
- interpolates missing values
- clips extreme outliers
- creates lag features
- creates rolling statistics
- creates calendar/time-based features

### 3. Demand forecasting
A time-series forecasting model is trained to predict future site demand across configurable forecast horizons.

### 4. Anomaly detection
Recent demand is scanned for abnormal spikes using statistical detection logic.

### 5. Battery dispatch recommendation
A dispatch engine simulates charge/discharge schedules using forecasted demand and operating signals such as:

- price
- carbon intensity
- peak demand

### 6. API + dashboard
FastAPI serves analytics endpoints, and Streamlit provides an interactive dashboard for exploring the outputs.

---

## Running locally

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python scripts/generate_synthetic_data.py
```

### 4. Train the model and build processed artifacts

```bash
python scripts/train_forecast_model.py
```

### 5. Start the FastAPI backend

```bash
python -m uvicorn backend.main:app --reload
```

Open API docs at:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the Streamlit dashboard

In a second terminal:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

Open the dashboard at:

```text
http://localhost:8501
```

---

## Running with Docker

```bash
docker compose up --build
```

Then open:

* API docs: `http://localhost:8000/docs`
* Dashboard: `http://localhost:8501`

---

## Main API endpoints

### `GET /sites`

Returns available sites.

### `GET /timeseries`

Returns recent telemetry for a site.

Example:

```text
/sites
/timeseries?site_id=alpha&hours=168
```

### `GET /kpis`

Returns site-level KPI summaries.

### `GET /forecast`

Returns forecasted demand rows for a site and horizon.

### `GET /anomalies`

Returns detected anomalies in recent demand.

### `POST /recommendation`

Generates a battery dispatch recommendation.

Example request body:

```json
{
  "site_id": "alpha",
  "horizon_hours": 24,
  "battery_mwh": 20,
  "max_charge_mw": 5,
  "max_discharge_mw": 5,
  "soc0_mwh": 2,
  "emissions_weight": 0.02,
  "peak_demand_weight": 15,
  "mode": "balanced"
}
```

### `POST /ask`

Supports simple natural-language analytics questions.

Example request body:

```json
{
  "question": "forecast next 24 hours for alpha"
}
```

---

## Current dashboard features

* site selector
* lookback window control
* forecast horizon control
* initial battery state-of-charge control
* dispatch mode selector
* historical demand chart
* demand forecast chart
* anomaly table
* recommendation metrics
* dispatch schedule visualization
* natural-language query panel

---

## Current limitations

* The project currently uses synthetic data, not real operational energy data
* The battery dispatch engine is heuristic, not a full constrained mathematical optimizer
* Forecast and recommendation quality depend on the selected dispatch mode and horizon
* Tradeoffs between cost, emissions, and peak shaving may vary across horizons

---

## Future improvements

* Replace synthetic telemetry with real public energy, weather, price, and carbon datasets
* Benchmark multiple forecasting models
* Add formal optimization for dispatch scheduling
* Add automated tests and CI
* Improve the natural-language analytics layer
* Deploy the app publicly
