"""
Data pipeline: fetch and merge electricity load, weather, and calendar data
for Delhi (NCT).

Load data source: Kaggle dataset "Delhi Electricity Load Data (5-min resolution)"
https://www.kaggle.com/datasets/prash4nt/delhi-sldc-load-data-5-min-resolution
(scraped from delhisldc.org, which itself disallows automated scraping —
use this maintained dataset rather than scraping the site directly.)

Usage:
    # 1. Download the Kaggle dataset manually (or via kaggle CLI, see below)
    #    into data/raw/
    # 2. python src/data_pipeline.py --start 2019-01-01 --end 2024-12-31
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DELHI_LAT, DELHI_LON = 28.6139, 77.2090

# Option A (recommended): manually download from Kaggle and place the CSV in data/raw/
# Option B: use the Kaggle CLI (requires a free Kaggle API token, ~/.kaggle/kaggle.json):
#   pip install kaggle
#   kaggle datasets download -d prash4nt/delhi-sldc-load-data-5-min-resolution -p data/raw --unzip


def load_raw_load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the raw Delhi SLDC CSV (5-min resolution) and do basic cleaning.

    Confirmed schema: two columns, 'timestamp' (e.g. "4/1/2023 0:00") and
    'load_MW' (total Delhi load only — no per-discom breakdown).
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", dayfirst=False)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["load_MW"] = pd.to_numeric(df["load_MW"], errors="coerce")

    # Flag suspicious readings before dropping/interpolating: zeros or huge
    # jumps usually mean a comms link failure at the SLDC end, not a real
    # demand collapse. Worth a paragraph in the report on how you handled this.
    df.loc[df["load_MW"] <= 0, "load_MW"] = pd.NA

    n_missing = df["load_MW"].isna().sum()
    print(f"Missing/invalid readings: {n_missing} of {len(df)} ({100*n_missing/len(df):.2f}%)")

    # Short gaps: linear interpolation is reasonable at 5-min resolution.
    # Long gaps (multiple hours+) should probably be left as NaN and
    # excluded from training rather than interpolated across, since
    # interpolating a multi-hour gap invents fake demand curve shape.
    df["load_MW"] = df["load_MW"].interpolate(method="linear", limit=6)  # limit=6 -> max 30 min gap filled

    df = df.set_index("timestamp")
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 5-min load data to hourly (mean). Recommended for v1: 5-min
    resolution is finer than needed for a 24-168hr-ahead forecasting task
    and inflates training time considerably with little forecasting benefit
    at that horizon. State this as a deliberate scoping decision in the report.
    """
    hourly = df[["load_MW"]].resample("1h").mean()
    return hourly


def fetch_weather_data(lat: float = DELHI_LAT, lon: float = DELHI_LON, start: str = "2019-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Fetch historical weather via Open-Meteo's free archive API (no key required).
    Docs: https://open-meteo.com/en/docs/historical-weather-api
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_holiday_calendar(start: str, end: str) -> pd.DataFrame:
    """
    Build a calendar of Indian public holidays/festivals over the date range.
    """
    import holidays as holidays_pkg

    india_holidays = holidays_pkg.India(years=range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": dates})
    df["is_holiday"] = df["date"].dt.date.astype(str).isin([str(d) for d in india_holidays.keys()]).astype(int)
    df["holiday_name"] = df["date"].dt.date.map(lambda d: india_holidays.get(d, ""))
    # NOTE: the `holidays` package's India coverage is incomplete for
    # movable festivals (esp. regional ones). Cross-check Diwali/Holi/Eid
    # dates for your exact years manually and patch any that are missing —
    # this is worth doing carefully since festival demand spikes are a
    # genuinely interesting feature.
    return df.set_index("date")


def merge_all(load_df: pd.DataFrame, weather_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly load, weather, and calendar data into one clean, time-indexed dataframe."""
    weather_hourly = weather_df.set_index("time")
    merged = load_df.join(weather_hourly, how="left")

    # IMPORTANT: pandas' .merge() on a column key always returns a fresh
    # default RangeIndex and silently drops whatever index you had before
    # (here, the datetime timestamp index). So we explicitly pull the
    # timestamp out into a column before merging, then restore it as the
    # index afterward, instead of losing it.
    merged = merged.reset_index()  # 'timestamp' becomes a normal column
    merged["date"] = merged["timestamp"].dt.date.astype(str)

    calendar_df = calendar_df.reset_index()
    calendar_df["date"] = calendar_df["date"].dt.date.astype(str)

    merged = merged.merge(calendar_df[["date", "is_holiday", "holiday_name"]], on="date", how="left")
    merged = merged.drop(columns=["date"])
    merged["is_holiday"] = merged["is_holiday"].fillna(0).astype(int)

    merged = merged.set_index("timestamp").sort_index()
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_csv", type=str, default="data/raw/delhi_load.csv",
                         help="Path to the downloaded Kaggle CSV")
    parser.add_argument("--start", type=str, default="2023-04-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning load data...")
    load_df = load_raw_load_data(Path(args.load_csv))
    hourly_load = resample_hourly(load_df)

    print("Fetching weather data...")
    weather_df = fetch_weather_data(start=args.start, end=args.end)

    print("Building holiday calendar...")
    calendar_df = build_holiday_calendar(args.start, args.end)

    print("Merging...")
    final_df = merge_all(hourly_load, weather_df, calendar_df)
    assert isinstance(final_df.index, pd.DatetimeIndex), "merge_all() must return a DatetimeIndex!"

    out_path = PROCESSED_DIR / "load_weather_hourly.csv"
    final_df.to_csv(out_path)
    print(f"Saved {len(final_df)} rows to {out_path}")
