"""
Data loading and cleaning for the OWID COVID-19 dataset.

The dataset mixes country-level rows with aggregate rows (continents,
income groups, "World"). Aggregates have no `continent` value, which is
the cleanest signal for splitting them out.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/"
    "master/public/data/owid-covid-data.csv"
)
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "owid-covid-data.csv"


def download_data(cache_path: Path = DEFAULT_CACHE, force: bool = False) -> Path:
    """Download the OWID dataset, caching it locally."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        print(f"Using cached data: {cache_path} "
              f"({cache_path.stat().st_size / 1e6:.1f} MB)")
        return cache_path

    print(f"Downloading dataset from {DATA_URL} ...")
    # Stream via pandas — works without extra deps.
    df = pd.read_csv(DATA_URL, low_memory=False)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df):,} rows to {cache_path}")
    return cache_path


def load_data(cache_path: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached CSV with proper dtypes."""
    if not cache_path.exists():
        download_data(cache_path)
    df = pd.read_csv(cache_path, low_memory=False, parse_dates=["date"])
    return df


def split_countries_and_aggregates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    OWID uses missing `continent` to flag aggregate rows
    (World, continents, income groups, EU). Split them so we never
    accidentally double-count by mixing aggregates with countries.
    """
    countries = df[df["continent"].notna()].copy()
    aggregates = df[df["continent"].isna()].copy()
    return countries, aggregates


def get_country_latest(countries: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    For a given metric, return the most recent non-null value per country.
    Useful for "total cases per million" style rankings where you want
    each country's final value rather than a daily reading.
    """
    valid = countries.dropna(subset=[metric])
    idx = valid.groupby("location")["date"].idxmax()
    return (
        valid.loc[idx, ["location", "iso_code", "continent", "date", metric, "population"]]
        .sort_values(metric, ascending=False)
        .reset_index(drop=True)
    )


def filter_countries(
    countries: pd.DataFrame,
    names: list[str],
) -> pd.DataFrame:
    """Subset to a list of countries, preserving the input order for plotting."""
    sub = countries[countries["location"].isin(names)].copy()
    sub["location"] = pd.Categorical(sub["location"], categories=names, ordered=True)
    return sub.sort_values(["location", "date"])


if __name__ == "__main__":
    # Quick smoke test
    path = download_data()
    df = load_data(path)
    countries, aggregates = split_countries_and_aggregates(df)
    print(f"Loaded {len(df):,} rows")
    print(f"  Countries: {countries['location'].nunique()}")
    print(f"  Aggregates: {sorted(aggregates['location'].unique())}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
