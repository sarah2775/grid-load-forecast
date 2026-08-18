"""
Data pipeline: fetch and merge electricity load, weather, and calendar data.

Usage:
    python src/data_pipeline.py --region "northern" --start 2019-01-01 --end 2024-12-31
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def fetch_load_data(region: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch electricity load data for a given region and date range.

    TODO: Grid-India / POSOCO publishes daily/hourly regional load reports.
    Start here: https://posoco.in/ and https://www.grid-india.in/
    Some data is in PDF reports (may need pdf-table extraction) and some
    in downloadable CSVs depending on the source page — check both.
    """
    raise NotImplementedError("Fill in once you've located the exact data source/format.")


def fetch_weather_data(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
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
    TODO: use the `holidays` package (holidays.India()) as a starting point,
    then manually add major festivals if the package misses regional ones.
    """
    raise NotImplementedError


def merge_all(load_df: pd.DataFrame, weather_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Merge load, weather, and calendar data into one clean, time-indexed dataframe."""
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: wire up the functions above once data sources are confirmed
    print(f"Pipeline stub ready for region={args.region}, {args.start} to {args.end}")
