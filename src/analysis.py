"""
Analysis helpers: derived metrics that aren't in the raw dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_case_fatality_rate(countries: pd.DataFrame) -> pd.DataFrame:
    """
    CFR = cumulative deaths / cumulative cases.

    Note: CFR is an *imperfect* mortality measure because reported cases
    massively undercount true infections (especially before widespread
    testing). Use alongside excess mortality for a fuller picture.
    """
    df = countries.copy()
    df["case_fatality_rate"] = np.where(
        df["total_cases"] > 0,
        df["total_deaths"] / df["total_cases"] * 100,
        np.nan,
    )
    return df


def rolling_average(
    countries: pd.DataFrame,
    metric: str,
    window: int = 7,
) -> pd.DataFrame:
    """Add a rolling-average column for a per-day metric, computed per country."""
    df = countries.sort_values(["location", "date"]).copy()
    df[f"{metric}_roll{window}"] = (
        df.groupby("location", observed=True)[metric]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return df


def top_n_by_metric(
    countries: pd.DataFrame,
    metric: str,
    n: int = 10,
    min_population: float = 1_000_000,
) -> list[str]:
    """
    Return the top-n country names by the latest value of a metric.
    Filter out tiny populations to avoid microstates dominating per-capita
    rankings (where one outbreak can spike the rate).
    """
    valid = (
        countries.dropna(subset=[metric, "population"])
        .query("population >= @min_population")
    )
    latest_idx = valid.groupby("location", observed=True)["date"].idxmax()
    latest = valid.loc[latest_idx].sort_values(metric, ascending=False)
    return latest.head(n)["location"].tolist()


def detect_waves(
    series: pd.Series,
    prominence_frac: float = 0.15,
    min_distance_days: int = 60,
) -> list[pd.Timestamp]:
    """
    Lightweight wave detector — finds peaks in a smoothed series.
    `prominence_frac` is the minimum height relative to the series max.

    We use a simple local-maximum approach so the project stays free of
    SciPy as a hard dependency. Good enough for annotating global waves.
    """
    s = series.dropna()
    if s.empty:
        return []
    threshold = s.max() * prominence_frac
    peaks: list[pd.Timestamp] = []
    values = s.values
    dates = s.index
    last_peak_idx = -10**9
    for i in range(1, len(values) - 1):
        if (
            values[i] > values[i - 1]
            and values[i] >= values[i + 1]
            and values[i] >= threshold
            and (i - last_peak_idx) >= min_distance_days
        ):
            peaks.append(dates[i])
            last_peak_idx = i
    return peaks


def _last_valid(series: pd.Series):
    """Return the last non-null value in a series, or None."""
    valid = series.dropna()
    return valid.iloc[-1] if not valid.empty else None


def summarize_country(countries: pd.DataFrame, name: str) -> dict:
    """Compact summary stats for a single country — handy for reports.

    Uses the last *non-null* value for each metric, because the very last
    date in the dataset often has missing values for some columns (different
    reporting cadences across metrics).
    """
    sub = countries[countries["location"] == name].sort_values("date")
    if sub.empty:
        return {}
    peak_cases_row = sub.loc[sub["new_cases_smoothed"].idxmax()] \
        if sub["new_cases_smoothed"].notna().any() else None
    return {
        "country": name,
        "population": _last_valid(sub["population"]),
        "total_cases": _last_valid(sub["total_cases"]),
        "total_deaths": _last_valid(sub["total_deaths"]),
        "cases_per_million": _last_valid(sub["total_cases_per_million"]),
        "deaths_per_million": _last_valid(sub["total_deaths_per_million"]),
        "pct_fully_vaccinated": _last_valid(sub["people_fully_vaccinated_per_hundred"]),
        "peak_daily_cases_date": peak_cases_row["date"].date()
            if peak_cases_row is not None else None,
        "peak_daily_cases": peak_cases_row["new_cases_smoothed"]
            if peak_cases_row is not None else None,
    }
