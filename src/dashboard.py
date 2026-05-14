"""
COVID-19 Trends Dashboard — main entry point.

Run from the project root:
    python src/dashboard.py

This will (1) ensure the dataset is cached locally, (2) generate all
charts into outputs/, and (3) print a textual summary report.

Charts 01–10 are the original static set (visualizations.py).
Charts 11–14 are the extended set:
    11  hospitalization & ICU occupancy
    12  global waves with variant bands
    13  SIR/SEIR/Prophet forecast vs actuals
    14  interactive choropleth (.html; .png too if `kaleido` is installed)

For the fully interactive experience (live country/date filtering), run
the Dash app instead:
    python interactive/app.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# models/ is a sibling of src/ — add it so the forecast charts can import
# epidemic_models. (src/ itself is already the script's own directory.)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models"))

from data_loader import (
    download_data,
    load_data,
    split_countries_and_aggregates,
)
from analysis import summarize_country
from visualizations import (
    apply_style,
    plot_global_waves,
    plot_top_countries_per_million,
    plot_continental_timeline,
    plot_vaccination_coverage,
    plot_wealth_vs_vaccination,
    plot_cfr_timeline,
    plot_deaths_heatmap,
    plot_country_comparison,
    plot_excess_mortality,
    plot_stringency_vs_cases,
)
from visualizations_extra import (
    plot_hospitalization,
    plot_waves_with_variants,
    plot_forecast,
)
from choropleth import save_choropleth_html, save_choropleth_png


OUTDIR = ROOT / "outputs"


def print_summary(countries: pd.DataFrame, aggregates: pd.DataFrame) -> None:
    """Print a textual summary of headline numbers."""
    world = aggregates[aggregates["location"] == "World"].sort_values("date")
    last = world.iloc[-1]["date"].date()

    def lv(col: str) -> float:
        v = world[col].dropna()
        return float(v.iloc[-1]) if not v.empty else float("nan")

    print("\n" + "=" * 70)
    print("GLOBAL COVID-19 — HEADLINE FIGURES")
    print("=" * 70)
    print(f"As of {last}:")
    print(f"  Total cases:          {lv('total_cases'):>20,.0f}")
    print(f"  Total deaths:         {lv('total_deaths'):>20,.0f}")
    print(f"  Cases/million:        {lv('total_cases_per_million'):>20,.0f}")
    print(f"  Deaths/million:       {lv('total_deaths_per_million'):>20,.0f}")
    print(f"  Fully vaccinated:     {lv('people_fully_vaccinated_per_hundred'):>19.1f}%")

    print("\nCountry spotlights:")
    spotlight = ["United States", "United Kingdom", "India",
                 "Brazil", "Japan", "South Africa"]
    for name in spotlight:
        s = summarize_country(countries, name)
        if not s or s.get("total_cases") is None:
            continue
        print(f"\n  {s['country']}")
        print(f"    Cases:          {s['total_cases']:,.0f}")
        print(f"    Deaths:         {s['total_deaths']:,.0f}")
        print(f"    Deaths/M:       {s['deaths_per_million']:,.0f}")
        print(f"    Fully vax:      {s.get('pct_fully_vaccinated') or 0:.1f}%")
        if s.get("peak_daily_cases_date"):
            print(f"    Peak day:       {s['peak_daily_cases_date']} "
                  f"({s['peak_daily_cases']:,.0f} cases/day)")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="COVID-19 Trends Dashboard")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download of the dataset")
    parser.add_argument("--outdir", type=Path, default=OUTDIR,
                        help="Output directory for charts (default: ./outputs)")
    parser.add_argument("--skip-extras", action="store_true",
                        help="Only render the original charts 01-10")
    args = parser.parse_args()

    apply_style()

    # Load
    if args.refresh:
        download_data(force=True)
    df = load_data()
    countries, aggregates = split_countries_and_aggregates(df)
    print(f"Loaded {len(df):,} rows | "
          f"{countries['location'].nunique()} countries | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    # Render
    args.outdir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("Global waves", plot_global_waves, (aggregates, args.outdir)),
        ("Top countries by cases/million", plot_top_countries_per_million,
         (countries, args.outdir)),
        ("Continental timeline", plot_continental_timeline,
         (aggregates, args.outdir)),
        ("Vaccination coverage", plot_vaccination_coverage,
         (countries, args.outdir)),
        ("Wealth vs vaccination", plot_wealth_vs_vaccination,
         (countries, args.outdir)),
        ("CFR timeline", plot_cfr_timeline, (countries, args.outdir)),
        ("Deaths heatmap", plot_deaths_heatmap, (countries, args.outdir)),
        ("Country comparison", plot_country_comparison,
         (countries, args.outdir)),
        ("Excess mortality", plot_excess_mortality, (countries, args.outdir)),
        ("Stringency vs cases", plot_stringency_vs_cases,
         (countries, args.outdir)),
    ]
    if not args.skip_extras:
        charts += [
            ("Hospitalization & ICU", plot_hospitalization,
             (countries, args.outdir)),
            ("Waves with variant bands", plot_waves_with_variants,
             (aggregates, args.outdir)),
            ("SIR/SEIR/Prophet forecast", plot_forecast, (countries, args.outdir)),
        ]

    for label, fn, fargs in charts:
        print(f"  Rendering: {label} ...", end=" ", flush=True)
        path = fn(*fargs)
        print(f"→ {path.relative_to(ROOT)}")

    # Choropleth is handled separately — it writes HTML (always) and PNG
    # (only if kaleido is available), so it doesn't fit the uniform loop.
    if not args.skip_extras:
        print("  Rendering: Choropleth map ...", end=" ", flush=True)
        html_path = save_choropleth_html(countries, args.outdir,
                                         metric="total_deaths_per_million")
        print(f"→ {html_path.relative_to(ROOT)}")
        save_choropleth_png(countries, args.outdir,
                            metric="total_deaths_per_million")

    print_summary(countries, aggregates)
    print(f"All charts saved to: {args.outdir}")
    if not args.skip_extras:
        print("\nFor live country/date filtering, run the interactive app:")
        print("    python interactive/app.py\n")


if __name__ == "__main__":
    main()
