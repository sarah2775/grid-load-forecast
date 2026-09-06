# ⚡ Delhi Electricity Demand Forecasting

Comparing classical statistical and machine learning approaches for short-term
electricity load forecasting in Delhi, with weather and seasonal demand
pattern analysis.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

## Problem Statement

Delhi's power grid faces a genuinely hard forecasting problem: demand has
overlapping daily, weekly, and annual seasonalities, is extremely
weather-sensitive (summer AC load pushes peak demand well above winter
levels), and is subject to real regime shifts (e.g. monsoon onset causing a
sharp multi-day temperature and demand drop). Accurate short-term load
forecasting matters operationally — discoms that under- or over-forecast
either risk local blackouts/load-shedding or waste procured generation
capacity, and better forecasting directly supports renewable energy
integration.

**Task:** given historical hourly electricity load for Delhi, plus weather
and calendar features, forecast demand 24 hours ahead.

This project builds and rigorously backtests multiple forecasting
approaches — from naive baselines through classical statistics to gradient
boosting — on real, messy, publicly sourced grid data, with proper
walk-forward validation rather than a single train/test split, and
compares them honestly, including where the "smarter" model didn't win.

## Dataset & Preprocessing

| Source | What | Notes |
|---|---|---|
| [Kaggle: Delhi SLDC Load Data (5-min resolution)](https://www.kaggle.com/datasets/prash4nt/delhi-sldc-load-data-5-min-resolution) | Historical Delhi electricity load | Scraped from delhisldc.org, which itself disallows automated scraping (`robots.txt`) — this project deliberately uses the existing public dataset rather than scraping the site directly |
| [Open-Meteo](https://open-meteo.com) archive API | Historical weather for Delhi (28.6139°N, 77.2090°E): temperature, humidity | Free, no API key |
| `holidays` (Python package) | Indian public holidays | Coverage cross-checked manually for major festivals |

**Preprocessing (`src/data_pipeline.py`):**
- Raw 5-min load readings: ~2.7% were invalid/missing (likely SLDC comms
  link failures). Short gaps (≤30 min) were linearly interpolated; longer
  gaps were left as `NaN` and excluded rather than fabricated.
- Resampled to hourly resolution (5-min was finer than needed for a
  24h-ahead forecasting task and inflated training time considerably for
  limited benefit).
- Merged with weather and an Indian holiday calendar into one time-indexed
  table (`data/processed/load_weather_hourly.csv`).

## Methodology

1. **EDA** (`01_eda.ipynb`) — seasonal decomposition, ACF/PACF, stationarity
   testing, weather/holiday relationship checks.
2. **Feature engineering** (`src/features.py`) — lag features (24h, 48h,
   168h, chosen from the ACF/PACF findings), rolling mean/std with proper
   leakage guards, cyclical calendar encodings.
3. **Models** (see below), all evaluated with the **same walk-forward
   backtest harness** (`src/evaluation.py`) for a fair comparison.
4. **Error analysis** (`04_error_analysis.ipynb`) — where each model wins
   or loses, broken down by season, time of day, and weather conditions.

### Walk-forward validation

Rather than a single train/test split (which only tells you how a model
does on one arbitrary stretch of time), every model here is evaluated with
**expanding-window walk-forward backtesting**: train on all data up to a
point, forecast the next 24 hours, then slide forward and repeat across the
dataset. This gives a far more honest estimate of real-world performance
than a lucky or unlucky single split, and is the standard approach for time
series evaluation.

*(SARIMA and LightGBM were retrained fresh at every fold. A note on scope:
this is a compute-cost trade-off — refitting a model 60+ times is only
practical for models that are cheap to fit; see Limitations.)*

## Models Compared

| Model | Type | Key idea |
|---|---|---|
| Naive | Baseline | "Next hour = current load" |
| Seasonal Naive (168h) | Baseline | "This hour = same hour, one week ago" |
| SARIMA | Classical statistical | Explicit trend + seasonal (daily) structure |
| LightGBM | Gradient-boosted trees | Lag/rolling/calendar features + weather + holidays |

## Results

Walk-forward backtest, mean MAPE across all valid folds:

| Model | Mean MAPE |
|---|---|
| **SARIMA** | **4.50%** |
| LightGBM | 4.96% |
| Seasonal Naive (168h) | 9.50% |
| Naive | 20.00% |

Full per-fold results and figures are in `02_baselines.ipynb` and
`03_ml_models.ipynb`.

## Key Findings

- **Each added bit of model complexity clearly earned its keep**: SARIMA
  roughly halves seasonal naive's error, and seasonal naive in turn roughly
  halves plain naive's error.
- **LightGBM did not beat SARIMA**, despite having access to temperature,
  humidity, holiday flags, and multiple lag features that SARIMA can't use
  directly. It came close (within ~0.5 points) and comfortably beat both
  naive baselines. This suggests Delhi's daily/weekly seasonal structure is
  regular enough that SARIMA's explicit seasonal modeling captures it about
  as well as a feature-based tree model does, at least without further
  hyperparameter tuning.
- **SHAP analysis on LightGBM cross-checks cleanly against the EDA**:
  `load_lag_24h` dominates (matches the ACF finding that daily seasonality
  is the strongest signal), temperature has a meaningful, visible impact
  (matches the weather-correlation EDA finding), and `is_holiday` sits at
  near-zero SHAP impact (matches the EDA's "holiday effect was
  noisy/inconclusive" finding). This consistency across independently-run
  analyses is a good sanity check that the findings aren't contradicting
  each other.
- **Seasonal naive fails predictably at regime changes**: a sample fold
  showed actual load running ~1,200 MW below its own week-ago prediction
  during a likely monsoon onset — a real-world illustration of why
  "look like last week" breaks exactly when it matters most.
- See `04_error_analysis.ipynb` for a deeper breakdown of *when* each model
  performs relatively better or worse (season, time of day, weather
  extremes).

## Limitations

- **Evaluation scope differs by model cost.** SARIMA and LightGBM were
  refit at every walk-forward fold; a full grid of hyperparameter search
  was not run for either, so both are likely leaving some accuracy on the
  table rather than representing a fully-tuned ceiling.
- **Weather data availability.** Open-Meteo's archive endpoint had not yet
  finalized roughly 14 months of recent data at initial fetch time
  (confirmed via direct API query) — resolved by re-fetching once the
  archive caught up, but this is a reminder that pipelines depending on
  live external APIs can have non-obvious reproducibility gaps.
- **Single-city, single-variable target.** This models total Delhi demand
  only, not per-discom breakdown, and doesn't account for supply-side
  factors (generation mix, transmission constraints).
- **~1% of raw load readings remain unfilled** after gap interpolation
  (long comms-failure gaps were deliberately left as missing rather than
  fabricated), which required defensive handling in the feature/backtest
  pipeline (see `progress_log.md` for the specific bugs this surfaced and
  how they were fixed).
- **SARIMA order was not extensively tuned** — a fixed `(2,0,2)x(1,1,1,24)`
  order was used throughout rather than a full grid search, for runtime
  reasons.

## Reproducibility

```bash
git clone https://github.com/sarah2775/grid-load-forecast.git
cd grid-load-forecast
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

1. Download the [Kaggle load dataset](https://www.kaggle.com/datasets/prash4nt/delhi-sldc-load-data-5-min-resolution) into `data/raw/`
2. Build the merged dataset:
   ```bash
   python src/data_pipeline.py --start 2023-04-01 --end 2026-01-12
   ```
3. Run the notebooks in order: `01_eda.ipynb` → `02_baselines.ipynb` →
   `03_ml_models.ipynb` → `04_error_analysis.ipynb`
4. (Optional) Launch the dashboard: `streamlit run app/streamlit_app.py`

## Repo Structure

```
grid-load-forecast/
├── data/
│   ├── raw/            # untouched downloads (gitignored)
│   └── processed/      # cleaned, merged dataset
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_ml_models.ipynb
│   └── 04_error_analysis.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── features.py
│   └── evaluation.py    # shared walk-forward backtest harness
├── app/
│   └── streamlit_app.py
├── reports/
│   └── progress_log.md  # week-by-week build log, including bugs found & fixed
├── requirements.txt
└── README.md
```

## Is This Novel?

Honestly: **no, not methodologically.** This is a comparative
implementation study, not a new forecasting method — SARIMA, LightGBM, and
walk-forward backtesting are all well-established techniques. What this
project offers is a rigorous, honestly-reported *application* of those
techniques to a specific, real, messy dataset (Delhi grid load), with
proper validation and transparent reporting of what worked, what didn't,
and the real engineering bugs hit along the way. That's a legitimate and
worthwhile thing to build and document well — it's just not a research
contribution in the "novel method" sense, and this project doesn't claim
to be one.

## Author

Sarah — final-year B.Tech CSE (AI & ML) — [GitHub](https://github.com/sarah2775)
