"""
Variant emergence timeline + a small helper to overlay variant bands
onto any matplotlib date axis.

Dates are approximate "became globally notable / dominant" windows based on
WHO designation dates and the period each variant drove a wave in most
regions. They're meant for *annotation*, not epidemiological precision —
variant timing differed by country, and OWID's main dataset doesn't carry
per-country variant sequencing (that lives in a separate OWID file).

If you want true per-country variant share, see:
    https://github.com/owid/covid-19-data/tree/master/public/data/variants
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Variant:
    name: str
    # Window during which this variant was the dominant global story.
    start: date
    end: date
    color: str
    # WHO designation date — plotted as a thin marker line if within range.
    designated: date | None = None


# Ordered earliest -> latest. End dates are deliberately the *start* of the
# next variant's dominance so the bands tile without gaps.
VARIANTS: list[Variant] = [
    Variant(
        name="Ancestral",
        start=date(2020, 1, 1),
        end=date(2020, 12, 1),
        color="#9e9e9e",
    ),
    Variant(
        name="Alpha (B.1.1.7)",
        start=date(2020, 12, 1),
        end=date(2021, 5, 1),
        color="#5b8def",
        designated=date(2020, 12, 18),
    ),
    Variant(
        name="Delta (B.1.617.2)",
        start=date(2021, 5, 1),
        end=date(2021, 12, 1),
        color="#e6a23c",
        designated=date(2021, 5, 11),
    ),
    Variant(
        name="Omicron (B.1.1.529)",
        start=date(2021, 12, 1),
        end=date(2023, 6, 1),
        color="#c0504d",
        designated=date(2021, 11, 26),
    ),
]


def variant_table() -> pd.DataFrame:
    """Variant windows as a tidy DataFrame (handy for notebooks / Dash)."""
    return pd.DataFrame(
        [
            {
                "variant": v.name,
                "start": pd.Timestamp(v.start),
                "end": pd.Timestamp(v.end),
                "designated": pd.Timestamp(v.designated) if v.designated else pd.NaT,
                "color": v.color,
            }
            for v in VARIANTS
        ]
    )


def overlay_variants(
    ax,
    *,
    alpha: float = 0.10,
    label: bool = True,
    designation_lines: bool = True,
    text_y: float = 0.92,
) -> None:
    """
    Shade variant-dominance windows onto a matplotlib Axes with a date x-axis.

    Call this *after* plotting your series so the axvspan picks up sensible
    x-limits. It clips bands to the current x-range, so it's safe to call on
    a chart that only covers part of the pandemic.

    Parameters
    ----------
    ax : matplotlib Axes
    alpha : band opacity
    label : draw the variant name at the top of each band
    designation_lines : draw a thin dashed line on the WHO designation date
    text_y : vertical position of the label, in axes fraction (0-1)
    """
    xmin, xmax = ax.get_xlim()
    import matplotlib.dates as mdates

    lo = mdates.num2date(xmin).replace(tzinfo=None)
    hi = mdates.num2date(xmax).replace(tzinfo=None)

    for v in VARIANTS:
        vs = pd.Timestamp(v.start).to_pydatetime()
        ve = pd.Timestamp(v.end).to_pydatetime()
        # Skip bands entirely outside the visible range.
        if ve < lo or vs > hi:
            continue
        band_lo = max(vs, lo)
        band_hi = min(ve, hi)
        ax.axvspan(band_lo, band_hi, color=v.color, alpha=alpha, zorder=0)

        if designation_lines and v.designated is not None:
            d = pd.Timestamp(v.designated).to_pydatetime()
            if lo <= d <= hi:
                ax.axvline(d, color=v.color, linestyle="--",
                           linewidth=1.0, alpha=0.55, zorder=1)

        if label:
            mid = band_lo + (band_hi - band_lo) / 2
            ax.text(
                mid, text_y, v.name.split(" (")[0],
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color=v.color,
                fontweight="bold", alpha=0.9, zorder=2,
            )


def plotly_variant_shapes(y_domain: tuple[float, float] = (0, 1)) -> list[dict]:
    """
    Return Plotly layout `shapes` (vrects) for variant bands, plus we expose
    annotations separately via `plotly_variant_annotations`.

    Use with: fig.update_layout(shapes=plotly_variant_shapes())
    """
    shapes = []
    for v in VARIANTS:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=pd.Timestamp(v.start),
                x1=pd.Timestamp(v.end),
                y0=y_domain[0],
                y1=y_domain[1],
                fillcolor=v.color,
                opacity=0.10,
                layer="below",
                line_width=0,
            )
        )
        if v.designated is not None:
            shapes.append(
                dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=pd.Timestamp(v.designated),
                    x1=pd.Timestamp(v.designated),
                    y0=0,
                    y1=1,
                    line=dict(color=v.color, width=1, dash="dash"),
                    opacity=0.55,
                    layer="below",
                )
            )
    return shapes


def plotly_variant_annotations(text_y: float = 0.97) -> list[dict]:
    """Plotly annotations labelling each variant band."""
    anns = []
    for v in VARIANTS:
        mid = pd.Timestamp(v.start) + (pd.Timestamp(v.end) - pd.Timestamp(v.start)) / 2
        anns.append(
            dict(
                x=mid,
                y=text_y,
                xref="x",
                yref="paper",
                text=v.name.split(" (")[0],
                showarrow=False,
                font=dict(size=10, color=v.color),
                opacity=0.9,
            )
        )
    return anns
