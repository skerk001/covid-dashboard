"""
Additional static charts that extend the original dashboard:

  11. Hospitalization & ICU occupancy  (plot_hospitalization)
  12. Global waves with variant bands  (plot_waves_with_variants)
  13. SIR/SEIR/Prophet forecast vs actuals  (plot_forecast)

These live in a separate module so the original `visualizations.py` stays
a clean, untouched file. `dashboard.py` imports from both.

Style helpers (apply_style, save, PALETTE) are reused from visualizations.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualizations import PALETTE, save
from variants import overlay_variants, VARIANTS
from analysis import detect_waves

# models/ is a sibling of src/; dashboard.py adds it to sys.path. When this
# module is imported from there the plain import works. Guard it so importing
# visualizations_extra alone (e.g. in tests) doesn't hard-fail.
try:
    from epidemic_models import fit_sir, fit_seir, slice_wave
    _HAVE_MODELS = True
except ImportError:  # pragma: no cover
    _HAVE_MODELS = False

# Prophet is an optional, heavier dependency. If it's missing, plot_forecast
# still draws SIR + SEIR and just omits the Prophet line.
try:
    from prophet_model import fit_prophet, HAVE_PROPHET
except ImportError:  # pragma: no cover
    HAVE_PROPHET = False


# -----------------------------------------------------------------------------
# 11. Hospitalization & ICU occupancy
# -----------------------------------------------------------------------------

def plot_hospitalization(countries: pd.DataFrame, outdir: Path,
                         selected: list[str] | None = None) -> Path:
    """
    Hospital + ICU patients per million for a handful of countries.

    OWID's hospitalisation columns are *sparsely* populated — only ~30-40
    mostly-high-income countries ever reported them, and some only for part
    of the pandemic. We pick a default set known to have decent coverage and
    drop any requested country that turns out to be all-NaN.
    """
    selected = selected or ["United States", "United Kingdom", "France",
                            "Italy", "Israel", "Canada"]

    hosp_col = "hosp_patients_per_million"
    icu_col = "icu_patients_per_million"

    # Keep only countries that actually have hospitalisation data.
    have_data = []
    for name in selected:
        sub = countries[countries["location"] == name]
        if sub[hosp_col].notna().any() or sub[icu_col].notna().any():
            have_data.append(name)
    if not have_data:
        # Degrade gracefully rather than crash the whole dashboard run.
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No hospitalisation data available for the "
                          "selected countries.",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return save(fig, outdir, "11_hospitalization")

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    for i, name in enumerate(have_data):
        cs = countries[countries["location"] == name].sort_values("date")
        color = PALETTE[i % len(PALETTE)]
        axes[0].plot(cs["date"], cs[hosp_col], label=name,
                     color=color, linewidth=1.6)
        axes[1].plot(cs["date"], cs[icu_col], label=name,
                     color=color, linewidth=1.6)

    axes[0].set_ylabel("Hospital patients per million")
    axes[0].set_title("Hospital occupancy")
    axes[1].set_ylabel("ICU patients per million")
    axes[1].set_title("ICU occupancy")

    # Variant context helps explain the wave structure of occupancy.
    for ax in axes:
        overlay_variants(ax, label=(ax is axes[0]), text_y=0.95)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", ncol=3, frameon=True, framealpha=0.95)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Hospital & ICU occupancy per million — variant bands shaded",
                 fontsize=15, fontweight="bold", y=0.995)
    return save(fig, outdir, "11_hospitalization")


# -----------------------------------------------------------------------------
# 12. Global waves, with variant emergence bands
# -----------------------------------------------------------------------------

def plot_waves_with_variants(aggregates: pd.DataFrame, outdir: Path) -> Path:
    """
    The original global-waves chart, re-cut with variant-dominance bands and
    WHO designation lines overlaid. Makes the "which variant drove which
    wave" story legible at a glance.
    """
    world = aggregates[aggregates["location"] == "World"].sort_values("date")
    ts = world.set_index("date")["new_cases_smoothed"]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(ts.index, ts.values, color=PALETTE[0], alpha=0.20)
    ax.plot(ts.index, ts.values, color=PALETTE[0], linewidth=1.8)

    # Keep the peak annotations from the original chart for continuity.
    waves = detect_waves(ts, prominence_frac=0.20, min_distance_days=90)
    for i, peak in enumerate(waves, start=1):
        y = ts.loc[peak]
        ax.annotate(
            f"{int(y):,}/day",
            xy=(peak, y), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=8, color="#444",
        )

    overlay_variants(ax, alpha=0.13, label=True, designation_lines=True,
                     text_y=0.88)

    ax.set_title("Global daily new cases — variant-dominance bands overlaid")
    ax.set_xlabel("")
    ax.set_ylabel("New cases (7-day rolling avg)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.text(
        0.01, 0.97,
        "Bands = period each variant drove the global wave.\n"
        "Dashed lines = WHO variant-designation dates.\n"
        "Timing is approximate and differed by country.",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc",
                  edgecolor="#cccccc", alpha=0.9),
    )
    return save(fig, outdir, "12_waves_with_variants")


# -----------------------------------------------------------------------------
# 13. SIR / SEIR forecast vs actuals
# -----------------------------------------------------------------------------

# Default wave windows to fit. Chosen as single-variant-dominated rises so a
# constant-parameter compartmental model is at least defensible.
DEFAULT_FORECAST_WAVES = {
    "United States": ("2021-06-15", "2021-10-15"),   # Delta wave
    "United Kingdom": ("2021-05-15", "2021-08-31"),  # Delta wave
}


def plot_forecast(countries: pd.DataFrame, outdir: Path,
                  waves: dict[str, tuple[str, str]] | None = None,
                  horizon_days: int = 45) -> Path:
    """
    Fit SIR, SEIR and (if installed) Prophet to a chosen wave window per
    country, then plot the in-sample fit plus an out-of-sample forecast
    against what actually happened next.

    SIR/SEIR are mechanistic (constant-parameter compartmental models);
    Prophet is a statistical additive model. They tend to fail differently,
    which is the point of showing them together. The constant-parameter
    compartmental models will typically *overshoot*, because in reality
    behaviour change and interventions bent the curve down. Prophet, fit on
    a single wave, instead tends to over-trust the local trend. The "actuals
    beyond the fit window" line is the honesty check for all three.

    Prophet is optional: if it isn't installed, the chart simply shows
    SIR + SEIR and no Prophet line.
    """
    if not _HAVE_MODELS:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "epidemic_models not importable — is scipy "
                          "installed and models/ on sys.path?",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return save(fig, outdir, "13_forecast")

    waves = waves or DEFAULT_FORECAST_WAVES
    n = len(waves)
    fig, axes = plt.subplots(n, 1, figsize=(13, 5.2 * n), squeeze=False)
    axes = axes[:, 0]

    for ax, (country, (start, end)) in zip(axes, waves.items()):
        try:
            dates, values, pop = slice_wave(countries, country, start, end)
        except ValueError as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center")
            ax.axis("off")
            continue

        sir = fit_sir(dates, values, population=pop)
        seir = fit_seir(dates, values, population=pop)
        sir_fc = sir.forecast(horizon_days=horizon_days)
        seir_fc = seir.forecast(horizon_days=horizon_days)

        # Prophet is optional — fit it only if the dependency is installed.
        prophet_fc = None
        if HAVE_PROPHET:
            try:
                prophet_fit = fit_prophet(dates, values)
                prophet_fc = prophet_fit.forecast(horizon_days=horizon_days)
            except Exception as e:  # prophet/cmdstan can fail at runtime
                print(f"  [plot_forecast] Prophet skipped for {country}: {e}")
                prophet_fc = None

        # Observed: the full window we fit on PLUS what came after, so the
        # forecast can be visually checked against reality.
        full = countries[countries["location"] == country].sort_values("date")
        obs_end = (pd.Timestamp(end) + pd.Timedelta(days=horizon_days))
        obs = full[(full["date"] >= start) & (full["date"] <= obs_end)]

        ax.plot(obs["date"], obs["new_cases_smoothed"],
                color="#333", linewidth=2.0, label="Actual (7d avg)", zorder=5)

        # SIR
        ax.plot(sir_fc["date"], sir_fc["new_cases"],
                color=PALETTE[0], linewidth=1.6, linestyle="-",
                label=f"SIR fit+forecast (R0={sir.r0:.2f})")
        fc_only = sir_fc[sir_fc["kind"] == "forecast"]
        ax.fill_between(fc_only["date"], fc_only["lower"], fc_only["upper"],
                        color=PALETTE[0], alpha=0.15)

        # SEIR
        ax.plot(seir_fc["date"], seir_fc["new_cases"],
                color=PALETTE[1], linewidth=1.6, linestyle="-",
                label=f"SEIR fit+forecast (R0={seir.r0:.2f})")
        fc_only = seir_fc[seir_fc["kind"] == "forecast"]
        ax.fill_between(fc_only["date"], fc_only["lower"], fc_only["upper"],
                        color=PALETTE[1], alpha=0.15)

        # Prophet (statistical, not mechanistic — no R0 to report)
        if prophet_fc is not None:
            ax.plot(prophet_fc["date"], prophet_fc["new_cases"],
                    color=PALETTE[2], linewidth=1.6, linestyle="--",
                    label="Prophet fit+forecast (additive model)")
            fc_only = prophet_fc[prophet_fc["kind"] == "forecast"]
            ax.fill_between(fc_only["date"], fc_only["lower"], fc_only["upper"],
                            color=PALETTE[2], alpha=0.15)

        # Mark where the fit window ends / forecast begins.
        ax.axvline(pd.Timestamp(end), color="grey", linestyle=":",
                   linewidth=1.4)
        ax.text(pd.Timestamp(end), ax.get_ylim()[1] * 0.95, " forecast →",
                fontsize=8, color="grey", va="top")

        ax.set_title(f"{country} — SIR/SEIR fit on {start}..{end}, "
                     f"{horizon_days}-day forecast")
        ax.set_ylabel("New cases per day")
        ax.legend(loc="upper right", fontsize=8.5, frameon=True,
                  framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    fig.suptitle("Wave forecasts vs actuals — mechanistic (SIR/SEIR) and "
                 "statistical (Prophet) models, fit on one wave",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    return save(fig, outdir, "13_forecast")
