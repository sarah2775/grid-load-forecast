# ⚡ Grid Load Forecast — Delhi Electricity Demand Forecasting

> Forecasting short-term electricity demand for Delhi (NCT) using classical, ML, and deep learning methods — with uncertainty quantification and a live interactive dashboard.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Status](https://img.shields.io/badge/status-in--progress-yellow.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

**🔗 Live demo:** _(coming soon)_
**📄 Report:** [`reports/report.pdf`](reports/report.pdf) _(coming soon)_

---

## Why this project

Delhi's power grid faces a hard forecasting problem: demand has multiple overlapping seasonalities (daily, weekly, festival, and annual), is extremely weather-sensitive (summer AC load routinely pushes peak demand well above winter levels), and saw a real regime shift during COVID-19 lockdowns. Accurate short-term load forecasting is critical for grid stability and renewable integration — discoms that under- or over-forecast either risk local blackouts/load-shedding or waste procured generation capacity.

This project builds and rigorously compares forecasting approaches — from classical statistical models to modern deep learning — on real Delhi grid data, with proper backtesting and uncertainty estimates (not just point forecasts), and ships the result as a usable dashboard.

## Problem statement

Given historical 5-min/hourly electricity load for Delhi, plus weather and calendar features, forecast demand 24–168 hours ahead with calibrated prediction intervals.

## Data sources

| Source | What | Link |
|---|---|---|
| Kaggle: Delhi SLDC Load Data (5-min resolution) | Historical Delhi electricity load, scraped from delhisldc.org | [kaggle.com/datasets/prash4nt/delhi-sldc-load-data-5-min-resolution](https://www.kaggle.com/datasets/prash4nt/delhi-sldc-load-data-5-min-resolution) |
| Open-Meteo | Historical weather for Delhi (temperature, humidity) — free, no API key | open-meteo.com |
| `holidays` (Python) + manual additions | Indian public holidays & major festivals (Diwali, Holi, etc.) | — |

> **Note on data ethics:** delhisldc.org's `robots.txt` disallows automated scraping, so this project deliberately uses the existing, publicly available Kaggle dataset rather than scraping the site directly. Worth a line in the report — it's a real data-ethics decision, not just a technical one.

_(Exact download/merge steps in `src/data_pipeline.py`. Raw data is not committed — see `data/raw/README.md` for how to fetch it.)_

## Methodology

1. **EDA** — seasonal decomposition, autocorrelation, stationarity tests, COVID-period anomaly analysis
2. **Baselines** — naive, seasonal naive, SARIMA/ETS
3. **Machine learning** — LightGBM/XGBoost on engineered lag, rolling-window, calendar, and weather features
4. **Deep learning** — LSTM/GRU and N-BEATS or Temporal Fusion Transformer (via `neuralforecast`/`darts`)
5. **Evaluation** — walk-forward (expanding-window) backtesting; MAPE, RMSE, and pinball loss for prediction intervals
6. **Explainability** — SHAP on the ML model to explain demand drivers
7. **Deployment** — Streamlit app for interactive region-level forecasts

## Results

_(Table filled in as models are trained — this is the section admissions/hiring reviewers read first, so it stays at the top once populated.)_

| Model | MAPE | RMSE | 90% Interval Coverage |
|---|---|---|---|
| Seasonal Naive | — | — | — |
| SARIMA | — | — | — |
| LightGBM | — | — | — |
| LSTM | — | — | — |
| TFT / N-BEATS | — | — | — |

## Repo structure

```
grid-load-forecast/
├── data/
│   ├── raw/          # untouched downloads (gitignored, not committed)
│   └── processed/    # cleaned, feature-engineered datasets
├── notebooks/         # EDA and experimentation (numbered, e.g. 01_eda.ipynb)
├── src/                # reusable pipeline code
│   ├── data_pipeline.py
│   ├── features.py
│   ├── models/
│   └── evaluation.py
├── app/                # Streamlit dashboard
│   └── streamlit_app.py
├── reports/            # final write-up (PDF) + figures
├── tests/              # unit tests for src/
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/sarah2775/grid-load-forecast.git
cd grid-load-forecast
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Roadmap

See [`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md) for the week-by-week build plan.

## Author

Sarah — final-year B.Tech CSE (AI & ML) — [GitHub](https://github.com/sarah2775)
