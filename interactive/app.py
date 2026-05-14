"""
Interactive COVID-19 dashboard — Plotly Dash app.

Run from the project root:
    python interactive/app.py

then open http://127.0.0.1:8050 in a browser.

What's interactive here
-----------------------
* Country multi-select + date-range slider drive every time-series chart.
* A metric dropdown + date slider drive the world choropleth map.
* A "forecast" tab fits SIR/SEIR — and Prophet, when installed — live to
  whatever date window you've selected for a single focus country.
* Variant-dominance bands are overlaid on the time-series charts via the
  shared src/variants.py helper, so they stay in sync with the static
  dashboard.

Design notes
------------
* Data is loaded ONCE at startup (module scope) — the OWID CSV is ~94 MB and
  re-reading it per callback would make the UI crawl. Callbacks only filter.
* Everything is plain Dash + Plotly; no dash-bootstrap-components dependency,
  just a little inline CSS, so `pip install dash plotly` is the whole story.
* Callback functions are deliberately small and each owns one output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- make src/ and models/ importable -----------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models"))

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update

from data_loader import load_data, split_countries_and_aggregates
from variants import plotly_variant_shapes, plotly_variant_annotations
from choropleth import build_choropleth, MAP_METRICS

try:
    from epidemic_models import fit_sir, fit_seir, slice_wave
    _HAVE_MODELS = True
except ImportError:
    _HAVE_MODELS = False

# Prophet is optional (heavy dependency). The forecast tab adds a Prophet
# line when it's installed and silently omits it otherwise.
try:
    from prophet_model import fit_prophet, HAVE_PROPHET
except ImportError:
    HAVE_PROPHET = False


# =============================================================================
# Data — loaded once at import time
# =============================================================================

print("Loading dataset (one-time, ~94 MB) ...")
_df = load_data()
COUNTRIES, AGGREGATES = split_countries_and_aggregates(_df)

# Pre-compute some things the callbacks need repeatedly.
ALL_COUNTRIES = sorted(COUNTRIES["location"].unique())
DATE_MIN = COUNTRIES["date"].min()
DATE_MAX = COUNTRIES["date"].max()

# Month-resolution marks for the range sliders (full slider of ~1500 days is
# unreadable; we snap to month starts).
_months = pd.date_range(DATE_MIN, DATE_MAX, freq="MS")
_MONTH_LIST = list(_months)
# Slider works in integer index space over _MONTH_LIST.
_SLIDER_MARKS = {
    i: m.strftime("%b\n%Y")
    for i, m in enumerate(_MONTH_LIST)
    if m.month in (1, 7)  # label Jan + Jul only, else it's a mess
}

DEFAULT_COUNTRIES = [c for c in
                     ["United States", "United Kingdom", "Germany",
                      "Brazil", "India", "South Africa"]
                     if c in ALL_COUNTRIES]

# Time-series metrics offered in the dropdown.
TS_METRICS = {
    "new_cases_smoothed_per_million": "New cases per million (7d avg)",
    "new_deaths_smoothed_per_million": "New deaths per million (7d avg)",
    "people_fully_vaccinated_per_hundred": "% fully vaccinated",
    "hosp_patients_per_million": "Hospital patients per million",
    "icu_patients_per_million": "ICU patients per million",
    "stringency_index": "Government stringency index",
}


# =============================================================================
# Helpers
# =============================================================================

def _idx_to_date(i: int) -> pd.Timestamp:
    """Clamp a slider index into _MONTH_LIST and return the timestamp."""
    i = max(0, min(int(i), len(_MONTH_LIST) - 1))
    return _MONTH_LIST[i]


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _base_layout(fig: go.Figure, title: str, ylabel: str) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=ylabel,
        margin=dict(l=50, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


# =============================================================================
# App + layout
# =============================================================================

app = Dash(__name__, title="COVID-19 Interactive Dashboard")
server = app.server  # exposed for gunicorn / deployment

_CONTROL_STYLE = {"marginBottom": "18px"}

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto",
           "fontFamily": "DejaVu Sans, Arial, sans-serif", "padding": "16px"},
    children=[
        html.H1("COVID-19 Interactive Dashboard",
                style={"marginBottom": "2px"}),
        html.P("Live filtering by country and date range. Data: Our World in "
               "Data (final snapshot, Aug 2024).",
               style={"color": "#666", "marginTop": "0"}),

        # ---- shared controls -------------------------------------------------
        html.Div(
            style={"background": "#f7f7f9", "padding": "16px",
                   "borderRadius": "8px", "marginBottom": "20px"},
            children=[
                html.Div(style=_CONTROL_STYLE, children=[
                    html.Label("Countries", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="country-select",
                        options=[{"label": c, "value": c}
                                 for c in ALL_COUNTRIES],
                        value=DEFAULT_COUNTRIES,
                        multi=True,
                    ),
                ]),
                html.Div(style=_CONTROL_STYLE, children=[
                    html.Label("Date range", style={"fontWeight": "bold"}),
                    dcc.RangeSlider(
                        id="date-range",
                        min=0,
                        max=len(_MONTH_LIST) - 1,
                        value=[0, len(_MONTH_LIST) - 1],
                        marks=_SLIDER_MARKS,
                        step=1,
                        allowCross=False,
                        tooltip={"placement": "bottom"},
                    ),
                ]),
            ],
        ),

        dcc.Tabs(id="tabs", value="tab-timeseries", children=[
            # ---- TAB 1: time series -----------------------------------------
            dcc.Tab(label="Time series", value="tab-timeseries", children=[
                html.Div(style={"paddingTop": "16px"}, children=[
                    html.Label("Metric", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="ts-metric",
                        options=[{"label": v, "value": k}
                                 for k, v in TS_METRICS.items()],
                        value="new_cases_smoothed_per_million",
                        clearable=False,
                        style={"maxWidth": "420px", "marginBottom": "8px"},
                    ),
                    dcc.Checklist(
                        id="ts-options",
                        options=[
                            {"label": " Overlay variant bands",
                             "value": "variants"},
                            {"label": " Log y-axis", "value": "log"},
                        ],
                        value=["variants"],
                        inline=True,
                        style={"marginBottom": "8px"},
                    ),
                    dcc.Graph(id="ts-graph"),
                ]),
            ]),

            # ---- TAB 2: choropleth ------------------------------------------
            dcc.Tab(label="World map", value="tab-map", children=[
                html.Div(style={"paddingTop": "16px"}, children=[
                    html.Label("Map metric", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="map-metric",
                        options=[{"label": lbl, "value": k}
                                 for k, (lbl, *_), in MAP_METRICS.items()],
                        value="total_deaths_per_million",
                        clearable=False,
                        style={"maxWidth": "420px", "marginBottom": "8px"},
                    ),
                    dcc.Checklist(
                        id="map-options",
                        options=[{"label": " Log colour scale",
                                  "value": "log"}],
                        value=[],
                        inline=True,
                        style={"marginBottom": "8px"},
                    ),
                    html.Label("Snapshot date (uses each country's latest "
                               "value on or before this date)",
                               style={"fontSize": "13px", "color": "#666"}),
                    dcc.Slider(
                        id="map-date",
                        min=0,
                        max=len(_MONTH_LIST) - 1,
                        value=len(_MONTH_LIST) - 1,
                        marks=_SLIDER_MARKS,
                        step=1,
                        tooltip={"placement": "bottom"},
                    ),
                    dcc.Graph(id="map-graph", style={"height": "600px"}),
                ]),
            ]),

            # ---- TAB 3: forecast --------------------------------------------
            dcc.Tab(label="Forecast", value="tab-forecast",
                    children=[
                html.Div(style={"paddingTop": "16px"}, children=[
                    html.P("Fits SIR and SEIR compartmental models — plus "
                           "Prophet, if installed — to the selected date "
                           "window for one focus country, then projects "
                           "forward. The mechanistic models typically "
                           "overshoot once real-world behaviour adapts; "
                           "Prophet instead over-trusts the local trend. "
                           "Comparing how they each miss is the point.",
                           style={"color": "#666", "fontSize": "13px"}),
                    html.Label("Focus country", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="forecast-country",
                        options=[{"label": c, "value": c}
                                 for c in ALL_COUNTRIES],
                        value=(DEFAULT_COUNTRIES[0]
                               if DEFAULT_COUNTRIES else ALL_COUNTRIES[0]),
                        clearable=False,
                        style={"maxWidth": "420px", "marginBottom": "8px"},
                    ),
                    html.Label("Forecast horizon (days past the selected "
                               "window)", style={"fontSize": "13px",
                                                  "color": "#666"}),
                    dcc.Slider(
                        id="forecast-horizon", min=15, max=120, step=15,
                        value=45,
                        marks={d: str(d) for d in range(15, 121, 15)},
                    ),
                    dcc.Loading(dcc.Graph(id="forecast-graph"), type="dot"),
                    html.Pre(id="forecast-summary",
                             style={"background": "#f7f7f9",
                                    "padding": "12px", "borderRadius": "6px",
                                    "fontSize": "12px", "overflowX": "auto"}),
                ]),
            ]),
        ]),

        html.Hr(),
        html.P("Built with Dash + Plotly. See README for the static "
               "dashboard and notebook.",
               style={"color": "#999", "fontSize": "12px",
                      "textAlign": "center"}),
    ],
)


# =============================================================================
# Callbacks
# =============================================================================

@app.callback(
    Output("ts-graph", "figure"),
    Input("country-select", "value"),
    Input("date-range", "value"),
    Input("ts-metric", "value"),
    Input("ts-options", "value"),
)
def update_timeseries(selected_countries, date_range, metric, options):
    """One line per selected country for the chosen metric + date window."""
    if not selected_countries:
        return _empty_fig("Select at least one country above.")

    lo = _idx_to_date(date_range[0])
    hi = _idx_to_date(date_range[1])

    sub = COUNTRIES[
        COUNTRIES["location"].isin(selected_countries)
        & (COUNTRIES["date"] >= lo)
        & (COUNTRIES["date"] <= hi)
    ]
    if sub.empty or sub[metric].notna().sum() == 0:
        return _empty_fig("No data for this metric / country / date "
                          "combination.")

    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i, country in enumerate(selected_countries):
        cs = sub[sub["location"] == country].sort_values("date")
        if cs[metric].notna().sum() == 0:
            continue  # silently skip countries with no data for this metric
        fig.add_trace(go.Scatter(
            x=cs["date"], y=cs[metric], mode="lines", name=country,
            line=dict(color=palette[i % len(palette)], width=1.8),
        ))

    ylabel = TS_METRICS.get(metric, metric)
    _base_layout(fig, ylabel, ylabel)

    if "log" in options:
        # symlog isn't a plotly option; guard against non-positive values.
        fig.update_yaxes(type="log")

    if "variants" in options:
        fig.update_layout(
            shapes=plotly_variant_shapes(),
            annotations=plotly_variant_annotations(),
        )

    return fig


@app.callback(
    Output("map-graph", "figure"),
    Input("map-metric", "value"),
    Input("map-date", "value"),
    Input("map-options", "value"),
)
def update_map(metric, date_idx, options):
    """World choropleth for the chosen metric, as of the slider date."""
    as_of = _idx_to_date(date_idx)
    log_scale = "log" in options
    return build_choropleth(COUNTRIES, metric, as_of=as_of,
                            log_scale=log_scale)


@app.callback(
    Output("forecast-graph", "figure"),
    Output("forecast-summary", "children"),
    Input("forecast-country", "value"),
    Input("date-range", "value"),
    Input("forecast-horizon", "value"),
)
def update_forecast(country, date_range, horizon):
    """Fit SIR + SEIR (+ Prophet if installed) to the selected window for
    one country, project forward."""
    if not _HAVE_MODELS:
        return (_empty_fig("scipy not installed — forecasting unavailable. "
                           "`pip install scipy`."), "")
    if not country:
        return _empty_fig("Pick a focus country."), ""

    lo = _idx_to_date(date_range[0])
    hi = _idx_to_date(date_range[1])

    try:
        dates, values, pop = slice_wave(
            COUNTRIES, country, str(lo.date()), str(hi.date()))
    except ValueError as e:
        return _empty_fig(str(e)), ""

    # Need a reasonable number of points for a stable fit.
    if len(dates) < 21:
        return (_empty_fig("Selected window is too short to fit — widen the "
                           "date range to at least ~3 weeks."), "")

    try:
        sir = fit_sir(dates, values, population=pop)
        seir = fit_seir(dates, values, population=pop)
    except Exception as e:  # fitting can fail on pathological windows
        return _empty_fig(f"Model fitting failed: {e}"), ""

    sir_fc = sir.forecast(horizon_days=horizon)
    seir_fc = seir.forecast(horizon_days=horizon)

    # Prophet is optional — add it only if the dependency is installed and
    # the fit succeeds. A failure here must not break the SIR/SEIR chart.
    prophet_fc = None
    if HAVE_PROPHET:
        try:
            prophet_fc = fit_prophet(dates, values).forecast(
                horizon_days=horizon)
        except Exception as e:
            print(f"[update_forecast] Prophet skipped for {country}: {e}")
            prophet_fc = None

    # Observed line: fit window + horizon, so the forecast can be eyeballed.
    full = COUNTRIES[COUNTRIES["location"] == country].sort_values("date")
    obs_end = hi + pd.Timedelta(days=horizon)
    obs = full[(full["date"] >= lo) & (full["date"] <= obs_end)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obs["date"], y=obs["new_cases_smoothed"], mode="lines",
        name="Actual (7d avg)", line=dict(color="#333", width=2.4)))

    model_traces = [(sir_fc, "SIR", "#1f77b4"),
                    (seir_fc, "SEIR", "#ff7f0e")]
    if prophet_fc is not None:
        model_traces.append((prophet_fc, "Prophet", "#2ca02c"))

    for fc, name, color in model_traces:
        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["new_cases"], mode="lines", name=name,
            line=dict(color=color, width=1.8)))
        fc_only = fc[fc["kind"] == "forecast"]
        # CI band as a filled area.
        fig.add_trace(go.Scatter(
            x=list(fc_only["date"]) + list(fc_only["date"][::-1]),
            y=list(fc_only["upper"]) + list(fc_only["lower"][::-1]),
            fill="toself", fillcolor=color, opacity=0.12,
            line=dict(width=0), hoverinfo="skip", showlegend=False))

    # Vertical line where forecast begins.
    fig.add_vline(x=hi, line=dict(color="grey", dash="dot"))
    model_list = "SIR/SEIR" + ("/Prophet" if prophet_fc is not None else "")
    _base_layout(fig, f"{country} — {model_list} fit + {horizon}-day forecast",
                 "New cases per day")

    summary = (f"Fit window: {lo.date()} -> {hi.date()}  "
               f"({len(dates)} days)\n\n"
               + sir.summary() + "\n\n" + seir.summary())
    if prophet_fc is not None:
        summary += ("\n\nProphet: statistical additive model (no R0). "
                    "Fit on the same window.")
    elif HAVE_PROPHET is False:
        summary += ("\n\nProphet: not installed (optional dependency) — "
                    "showing SIR/SEIR only.")
    summary += ("\n\nNote: constant-parameter compartmental models cannot "
                "capture intervention/behaviour change; treat the forecast "
                "as 'if conditions held constant'.")
    return fig, summary


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print(f"Loaded {len(_df):,} rows | {len(ALL_COUNTRIES)} countries | "
          f"{DATE_MIN.date()} -> {DATE_MAX.date()}")
    print("Starting Dash server on http://127.0.0.1:8050 ...")
    app.run(debug=True)
