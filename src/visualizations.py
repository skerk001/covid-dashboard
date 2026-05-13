"""
All plotting functions for the dashboard.

Style choices:
- seaborn whitegrid for clean, GitHub-friendly visuals
- A consistent qualitative palette across all charts
- 300 DPI PNG export so charts look sharp when embedded in READMEs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis import detect_waves, rolling_average, top_n_by_metric

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })


def save(fig: plt.Figure, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# 1. Global timeline of new cases, with waves annotated
# -----------------------------------------------------------------------------

def plot_global_waves(aggregates: pd.DataFrame, outdir: Path) -> Path:
    world = aggregates[aggregates["location"] == "World"].sort_values("date")
    ts = world.set_index("date")["new_cases_smoothed"]

    waves = detect_waves(ts, prominence_frac=0.20, min_distance_days=90)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(ts.index, ts.values, color=PALETTE[0], alpha=0.25)
    ax.plot(ts.index, ts.values, color=PALETTE[0], linewidth=1.8)

    for i, peak in enumerate(waves, start=1):
        y = ts.loc[peak]
        ax.axvline(peak, color="grey", linestyle="--", alpha=0.4)
        ax.annotate(
            f"Wave {i}\n{peak.strftime('%b %Y')}\n{int(y):,}/day",
            xy=(peak, y),
            xytext=(0, 30),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="grey", alpha=0.5),
        )

    ax.set_title("Global COVID-19 daily new cases — major waves")
    ax.set_xlabel("")
    ax.set_ylabel("New cases (7-day rolling avg)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save(fig, outdir, "01_global_waves")


# -----------------------------------------------------------------------------
# 2. Top countries by cases per million (rankings)
# -----------------------------------------------------------------------------

def plot_top_countries_per_million(countries: pd.DataFrame, outdir: Path) -> Path:
    top = top_n_by_metric(countries, "total_cases_per_million", n=15)
    sub = (
        countries[countries["location"].isin(top)]
        .dropna(subset=["total_cases_per_million"])
        .sort_values("date")
        .groupby("location", observed=True)
        .tail(1)
        .sort_values("total_cases_per_million", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(sub["location"], sub["total_cases_per_million"] / 1000,
                   color=PALETTE[0], edgecolor="white")
    for bar, val in zip(bars, sub["total_cases_per_million"]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                f"{val/1000:.0f}k", va="center", fontsize=9)

    ax.set_title("Top 15 countries by reported cases per million people")
    ax.set_xlabel("Cases per 1,000 people (cumulative)")
    ax.set_ylabel("")
    return save(fig, outdir, "02_top_countries_cases_per_million")


# -----------------------------------------------------------------------------
# 3. Continental new-cases comparison over time
# -----------------------------------------------------------------------------

def plot_continental_timeline(aggregates: pd.DataFrame, outdir: Path) -> Path:
    continents = ["Africa", "Asia", "Europe", "North America",
                  "Oceania", "South America"]
    sub = aggregates[aggregates["location"].isin(continents)]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, c in enumerate(continents):
        cs = sub[sub["location"] == c].sort_values("date")
        ax.plot(cs["date"], cs["new_cases_smoothed_per_million"],
                label=c, color=PALETTE[i], linewidth=1.8)

    ax.set_title("New daily cases per million — by continent")
    ax.set_xlabel("")
    ax.set_ylabel("New cases per million (7-day avg)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save(fig, outdir, "03_continental_timeline")


# -----------------------------------------------------------------------------
# 4. Vaccination coverage: top + bottom countries
# -----------------------------------------------------------------------------

def plot_vaccination_coverage(countries: pd.DataFrame, outdir: Path) -> Path:
    valid = (
        countries.dropna(subset=["people_fully_vaccinated_per_hundred"])
        .query("population >= 5_000_000")
    )
    idx = valid.groupby("location", observed=True)["date"].idxmax()
    latest = valid.loc[idx].sort_values("people_fully_vaccinated_per_hundred")

    top10 = latest.tail(10)
    bot10 = latest.head(10)

    # Two separate subplots with their own y-axes (no sharing) so labels
    # stay on their own side and don't collide with the other panel.
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].barh(top10["location"], top10["people_fully_vaccinated_per_hundred"],
                 color=PALETTE[2], edgecolor="white")
    axes[0].set_title("Highest full-vaccination coverage")
    axes[0].set_xlabel("% of population fully vaccinated")
    axes[0].set_xlim(0, 115)

    axes[1].barh(bot10["location"], bot10["people_fully_vaccinated_per_hundred"],
                 color=PALETTE[3], edgecolor="white")
    axes[1].set_title("Lowest full-vaccination coverage")
    axes[1].set_xlabel("% of population fully vaccinated")
    axes[1].set_xlim(0, 115)
    # Put the right panel's country labels on the right side
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")

    for ax in axes:
        for p in ax.patches:
            ax.text(p.get_width() + 1.5, p.get_y() + p.get_height()/2,
                    f"{p.get_width():.1f}%", va="center", fontsize=9)

    fig.suptitle("Vaccination coverage — countries with population ≥ 5M",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.25)
    return save(fig, outdir, "04_vaccination_coverage")


# -----------------------------------------------------------------------------
# 5. Wealth vs vaccination — equity scatter
# -----------------------------------------------------------------------------

def plot_wealth_vs_vaccination(countries: pd.DataFrame, outdir: Path) -> Path:
    valid = countries.dropna(
        subset=["people_fully_vaccinated_per_hundred", "gdp_per_capita",
                "population", "continent"]
    )
    idx = valid.groupby("location", observed=True)["date"].idxmax()
    latest = valid.loc[idx]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=latest,
        x="gdp_per_capita",
        y="people_fully_vaccinated_per_hundred",
        hue="continent",
        size="population",
        sizes=(30, 800),
        alpha=0.75,
        palette=PALETTE[:6],
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_title("Vaccination coverage vs GDP per capita")
    ax.set_xlabel("GDP per capita (US$, log scale)")
    ax.set_ylabel("% fully vaccinated")

    # Annotate a few notable countries
    notable = ["United States", "United Kingdom", "India", "Nigeria",
               "Brazil", "China", "Japan", "South Africa"]
    for name in notable:
        row = latest[latest["location"] == name]
        if not row.empty:
            r = row.iloc[0]
            ax.annotate(name,
                        xy=(r["gdp_per_capita"], r["people_fully_vaccinated_per_hundred"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.95)
    return save(fig, outdir, "05_wealth_vs_vaccination")


# -----------------------------------------------------------------------------
# 6. Case-fatality-rate timeline for selected countries
# -----------------------------------------------------------------------------

def plot_cfr_timeline(countries: pd.DataFrame, outdir: Path,
                      selected: list[str] | None = None) -> Path:
    selected = selected or ["United States", "United Kingdom", "Germany",
                            "Brazil", "India", "Japan", "South Africa"]

    df = countries.copy()
    df["cfr"] = np.where(df["total_cases"] > 100,  # avoid divide-by-tiny noise
                         df["total_deaths"] / df["total_cases"] * 100, np.nan)
    # The first few weeks have huge CFR spikes (reporting lag between deaths
    # and confirmed cases). Skip them so the chart shows the meaningful trend.
    df = df[df["date"] >= "2020-04-01"]
    sub = df[df["location"].isin(selected)]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, name in enumerate(selected):
        cs = sub[sub["location"] == name].sort_values("date")
        ax.plot(cs["date"], cs["cfr"], label=name,
                color=PALETTE[i % len(PALETTE)], linewidth=1.7)

    ax.set_title("Case fatality rate over time (cumulative deaths ÷ cumulative cases)")
    ax.set_xlabel("")
    ax.set_ylabel("CFR (%)")
    ax.set_ylim(0, 12)  # cap — anything above this is reporting lag, not signal
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Caveat box — important context for a mortality chart
    ax.text(
        0.01, 0.97,
        "Note: CFR ≠ true mortality.\nUndercounting of cases inflates this metric,\n"
        "especially early in the pandemic.\nUse alongside excess mortality.",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc",
                  edgecolor="#cccccc", alpha=0.9),
    )
    return save(fig, outdir, "06_cfr_timeline")


# -----------------------------------------------------------------------------
# 7. Deaths-per-million heatmap (country × month)
# -----------------------------------------------------------------------------

def plot_deaths_heatmap(countries: pd.DataFrame, outdir: Path) -> Path:
    selected = top_n_by_metric(countries, "total_deaths_per_million", n=20)
    sub = countries[countries["location"].isin(selected)].copy()
    sub["month"] = sub["date"].dt.to_period("M").dt.to_timestamp()

    pivot = (
        sub.groupby(["location", "month"], observed=True)["new_deaths_smoothed_per_million"]
        .mean()
        .unstack("month")
    )
    # Order rows by total deaths
    order = sub.groupby("location", observed=True)["total_deaths_per_million"].max() \
                .sort_values(ascending=False).index
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(15, 9))
    sns.heatmap(
        pivot,
        cmap="rocket_r",
        cbar_kws={"label": "Daily deaths per million (monthly avg)"},
        linewidths=0,
        ax=ax,
    )
    ax.set_title("Monthly death intensity — 20 hardest-hit countries")
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Sparser x-tick labels
    xticks = pivot.columns
    keep_idx = list(range(0, len(xticks), 6))
    ax.set_xticks([i + 0.5 for i in keep_idx])
    ax.set_xticklabels([xticks[i].strftime("%b %Y") for i in keep_idx],
                       rotation=45, ha="right")
    return save(fig, outdir, "07_deaths_heatmap")


# -----------------------------------------------------------------------------
# 8. Country comparison: cases + deaths + vaccination
# -----------------------------------------------------------------------------

def plot_country_comparison(countries: pd.DataFrame, outdir: Path,
                            selected: list[str] | None = None) -> Path:
    selected = selected or ["United States", "Germany", "Brazil",
                            "India", "Japan", "South Africa"]
    df = rolling_average(countries, "new_cases_per_million", window=7)
    df = rolling_average(df, "new_deaths_per_million", window=7)
    sub = df[df["location"].isin(selected)]

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    metrics = [
        ("new_cases_per_million_roll7", "Daily cases per million (7d avg)",
         "log", axes[0]),
        ("new_deaths_per_million_roll7", "Daily deaths per million (7d avg)",
         "log", axes[1]),
        ("people_fully_vaccinated_per_hundred", "% fully vaccinated",
         "linear", axes[2]),
    ]

    for metric, label, scale, ax in metrics:
        for i, name in enumerate(selected):
            cs = sub[sub["location"] == name].sort_values("date")
            ax.plot(cs["date"], cs[metric], label=name,
                    color=PALETTE[i], linewidth=1.6)
        ax.set_ylabel(label)
        if scale == "log":
            ax.set_yscale("symlog", linthresh=1)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", ncol=3, frameon=True, framealpha=0.95)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Country comparison — cases, deaths & vaccination",
                 fontsize=15, fontweight="bold", y=0.995)
    return save(fig, outdir, "08_country_comparison")


# -----------------------------------------------------------------------------
# 9. Excess mortality — a more honest mortality measure
# -----------------------------------------------------------------------------

def plot_excess_mortality(countries: pd.DataFrame, outdir: Path) -> Path:
    """
    Excess mortality (deaths above expected baseline) captures the pandemic's
    *true* toll — including uncounted COVID deaths and indirect deaths from
    overwhelmed health systems.
    """
    # For each country, take the latest non-null value of *each* metric
    # independently (they're reported on different cadences).
    def latest_valid(group: pd.DataFrame, col: str):
        v = group[col].dropna()
        return v.iloc[-1] if not v.empty else np.nan

    grouped = countries.groupby("location", observed=True)
    summary = pd.DataFrame({
        "excess_mortality_cumulative_per_million": grouped.apply(
            lambda g: latest_valid(g, "excess_mortality_cumulative_per_million")
        ),
        "total_deaths_per_million": grouped.apply(
            lambda g: latest_valid(g, "total_deaths_per_million")
        ),
        "population": grouped["population"].max(),
    }).reset_index()
    summary = summary.dropna(subset=["excess_mortality_cumulative_per_million"])
    summary = summary.query("population >= 3_000_000")
    summary = summary.sort_values(
        "excess_mortality_cumulative_per_million", ascending=False
    ).head(20).sort_values("excess_mortality_cumulative_per_million")

    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(11, 9))
    bar_h = 0.4
    ax.barh(y - bar_h/2, summary["excess_mortality_cumulative_per_million"],
            bar_h, label="Excess mortality (all causes)",
            color=PALETTE[3], edgecolor="white")
    ax.barh(y + bar_h/2, summary["total_deaths_per_million"],
            bar_h, label="Reported COVID deaths",
            color=PALETTE[0], edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(summary["location"])
    ax.set_xlabel("Cumulative deaths per million people")
    ax.set_title("Excess mortality vs reported COVID deaths — top 20 by excess mortality")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    # Move caveat to the top-left where there's white space
    ax.text(
        0.99, 0.40,
        "Excess mortality > reported deaths\nsuggests COVID undercounting.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc",
                  edgecolor="#cccccc", alpha=0.9),
    )
    return save(fig, outdir, "09_excess_mortality")


# -----------------------------------------------------------------------------
# 10. Stringency index vs cases — did lockdowns track outbreaks?
# -----------------------------------------------------------------------------

def plot_stringency_vs_cases(countries: pd.DataFrame, outdir: Path,
                             country: str = "United States") -> Path:
    sub = countries[countries["location"] == country].sort_values("date")
    sub = sub.dropna(subset=["stringency_index"], how="any")

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(sub["date"], sub["new_cases_smoothed_per_million"],
             color=PALETTE[0], label="New cases per million (7d avg)",
             linewidth=1.8)
    ax1.set_ylabel("New cases per million", color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])

    ax2 = ax1.twinx()
    ax2.plot(sub["date"], sub["stringency_index"],
             color=PALETTE[3], label="Stringency index",
             linewidth=1.5, alpha=0.85)
    ax2.set_ylabel("Stringency index (0–100)", color=PALETTE[3])
    ax2.tick_params(axis="y", labelcolor=PALETTE[3])
    ax2.set_ylim(0, 100)
    ax2.grid(False)

    ax1.set_title(f"Government response vs case load — {country}")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return save(fig, outdir, "10_stringency_vs_cases")
