"""
Choropleth world maps using Plotly's built-in country geometries.

No GeoPandas / GDAL — Plotly ships with a `country names` and ISO-3 location
mode, and OWID already provides `iso_code` (ISO-3) for every country row, so
we get clean world maps with zero geo-dependency pain.

Two entry points:
  * build_choropleth(...)   -> a plotly.graph_objects.Figure (reused by Dash)
  * save_choropleth_html(...) -> writes a standalone interactive .html

The Dash app (interactive/app.py) imports build_choropleth directly so the
map logic lives in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import plotly.graph_objects as go

# Metrics that make sense on a world map, with display metadata.
# key -> (label, colorscale, is_per_capita, log_friendly)
MAP_METRICS = {
    "total_cases_per_million": ("Total cases per million", "YlOrRd", True, True),
    "total_deaths_per_million": ("Total deaths per million", "Reds", True, True),
    "people_fully_vaccinated_per_hundred": (
        "% fully vaccinated", "Greens", False, False),
    "excess_mortality_cumulative_per_million": (
        "Excess mortality per million", "Purples", True, False),
    "stringency_index": ("Govt. stringency index", "Blues", False, False),
    "total_cases": ("Total cases (absolute)", "YlOrRd", False, True),
    "total_deaths": ("Total deaths (absolute)", "Reds", False, True),
}


def _latest_valid_per_country(countries: pd.DataFrame, metric: str,
                              as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Most recent non-null value of `metric` per country, optionally as of a
    given date (for the Dash time-slider). Carries iso_code + location.
    """
    df = countries.dropna(subset=[metric, "iso_code"])
    if as_of is not None:
        df = df[df["date"] <= as_of]
    if df.empty:
        return pd.DataFrame(columns=["location", "iso_code", metric, "date"])
    idx = df.groupby("location", observed=True)["date"].idxmax()
    return df.loc[idx, ["location", "iso_code", "date", metric]].reset_index(drop=True)


def build_choropleth(countries: pd.DataFrame, metric: str,
                     as_of: pd.Timestamp | None = None,
                     log_scale: bool = False) -> go.Figure:
    """
    Build a world choropleth for `metric`.

    Parameters
    ----------
    metric : one of MAP_METRICS keys (other columns work too, just without
        the nice label / colorscale defaults)
    as_of : if given, use each country's latest value on or before this date
    log_scale : plot log10(value); useful for the heavily right-skewed
        absolute-count metrics. Per-capita rates are usually fine linear.
    """
    label, colorscale, _is_pc, _log_ok = MAP_METRICS.get(
        metric, (metric.replace("_", " ").title(), "Viridis", False, False)
    )

    snap = _latest_valid_per_country(countries, metric, as_of=as_of)
    if snap.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for {label}"
                  + (f" as of {as_of.date()}" if as_of is not None else ""),
            geo=dict(showframe=False),
        )
        return fig

    z = snap[metric].astype(float)
    if log_scale:
        # log10, guarding non-positive values.
        z_plot = np.log10(z.clip(lower=1e-9))
        colorbar_title = f"log₁₀({label})"
        # Custom hovertext keeps the *real* value visible.
        hover = snap.apply(
            lambda r: f"<b>{r['location']}</b><br>{label}: {r[metric]:,.1f}"
                      f"<br>as of {pd.Timestamp(r['date']).date()}",
            axis=1,
        )
    else:
        z_plot = z
        colorbar_title = label
        hover = snap.apply(
            lambda r: f"<b>{r['location']}</b><br>{label}: {r[metric]:,.1f}"
                      f"<br>as of {pd.Timestamp(r['date']).date()}",
            axis=1,
        )

    fig = go.Figure(
        go.Choropleth(
            locations=snap["iso_code"],
            locationmode="ISO-3",
            z=z_plot,
            text=hover,
            hoverinfo="text",
            colorscale=colorscale,
            colorbar_title=colorbar_title,
            marker_line_color="white",
            marker_line_width=0.4,
        )
    )
    title = label
    if as_of is not None:
        title += f" — as of {pd.Timestamp(as_of).date()}"
    fig.update_layout(
        title=title,
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="natural earth",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def save_choropleth_html(countries: pd.DataFrame, outdir: Path,
                         metric: str = "total_deaths_per_million",
                         log_scale: bool = False) -> Path:
    """Write a standalone interactive choropleth to outputs/ as .html."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig = build_choropleth(countries, metric, log_scale=log_scale)
    path = outdir / f"14_choropleth_{metric}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def save_choropleth_png(countries: pd.DataFrame, outdir: Path,
                        metric: str = "total_deaths_per_million",
                        log_scale: bool = False) -> Path | None:
    """
    Optional static PNG export. Requires the `kaleido` package; if it's not
    installed we skip rather than fail the dashboard run.
    """
    try:
        import kaleido  # noqa: F401
    except ImportError:
        print("    (kaleido not installed — skipping choropleth PNG export; "
              "the interactive .html was still written)")
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    fig = build_choropleth(countries, metric, log_scale=log_scale)
    path = outdir / f"14_choropleth_{metric}.png"
    fig.write_image(str(path), width=1400, height=800, scale=2)
    return path
