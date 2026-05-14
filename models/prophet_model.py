"""
Prophet-based forecasting for COVID daily case series.

Why a separate module from epidemic_models.py?
-----------------------------------------------
SIR/SEIR are *mechanistic* — they encode an epidemic process and the fitted
parameters mean something (R0, infectious period). Prophet is *statistical* —
it's a generalised additive model (trend + seasonality + holidays) that knows
nothing about epidemics. They answer different questions, so they live in
different files but share one forecast contract (see below).

Prophet is an optional dependency. It pulls in cmdstanpy and a compiler
toolchain, which is why it is NOT in the core requirements and why every
entry point here guards on `_HAVE_PROPHET`. If Prophet is not installed the
rest of the project (SIR/SEIR included) still runs.

    pip install prophet

Shared forecast contract
------------------------
`ProphetFit.forecast(horizon_days)` returns the *same* tidy DataFrame shape
as SIRFit/SEIRFit.forecast:

    date | kind ('fitted'|'forecast') | new_cases | lower | upper

so visualizations_extra.plot_forecast and the Dash app can treat all three
models interchangeably.

Caveats — read before trusting any forecast
--------------------------------------------
* Prophet will happily extrapolate a trend off a cliff. On an epidemic curve
  that is past its peak, the linear-trend default can forecast negative
  cases; we clip at zero and default to a flat ('flat') trend for the
  forecast tail, which is the less-wrong choice for a wave that has turned.
* The weekly seasonality it picks up is mostly a *reporting* artefact
  (weekend test-centre closures), not real transmission dynamics.
* Prophet's uncertainty interval is a simulation of trend + noise, not a
  calibrated predictive interval. Treat it as "uncertainty exists".
* Like the compartmental models, this is fit to a single wave window. Handing
  it the whole multi-wave pandemic produces a confident, meaningless line.

Public API
----------
    fit_prophet(dates, new_cases, ...)        -> ProphetFit
    ProphetFit.forecast(horizon_days)         -> DataFrame
    ProphetFit.summary()                      -> str
    HAVE_PROPHET                              -> bool   (cheap capability check)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from prophet import Prophet
    _HAVE_PROPHET = True
except ImportError:  # pragma: no cover - exercised only when prophet absent
    _HAVE_PROPHET = False

# Public, cheap-to-import capability flag so callers don't have to catch
# ImportError themselves.
HAVE_PROPHET = _HAVE_PROPHET


# -----------------------------------------------------------------------------
# Fit result container
# -----------------------------------------------------------------------------

@dataclass
class ProphetFit:
    """
    Result of fitting Prophet to one wave of reported daily cases.

    Mirrors the SIRFit/SEIRFit surface: a `.forecast()` that returns the shared
    tidy DataFrame, and a `.summary()` text block. There is deliberately no
    `.r0` — Prophet has no mechanistic parameters to derive one from.
    """
    model: str
    fit_dates: pd.DatetimeIndex
    observed_new: np.ndarray
    fitted_new: np.ndarray
    residual_std: float
    _prophet: object          # the fitted Prophet object (kept for forecasting)
    _horizon_trend: str       # 'flat' or 'linear' — how the tail extrapolates

    def _rmse(self) -> float:
        return float(np.sqrt(np.nanmean((self.observed_new - self.fitted_new) ** 2)))

    def summary(self) -> str:
        return "\n".join([
            f"{self.model} fit",
            f"  window      : {self.fit_dates[0].date()} -> "
            f"{self.fit_dates[-1].date()} ({len(self.fit_dates)} days)",
            f"  trend (tail): {self._horizon_trend}",
            f"  RMSE        : {self._rmse():,.0f} cases/day",
        ])

    def forecast(self, horizon_days: int = 45) -> pd.DataFrame:
        """
        Roll the fitted model forward `horizon_days` past the fit window.

        Returns the shared tidy DataFrame:
            date, kind ('fitted'|'forecast'), new_cases, lower, upper
        (lower/upper are NaN on the fitted rows, matching SIRFit/SEIRFit.)
        """
        if not _HAVE_PROPHET:  # pragma: no cover
            raise ImportError("prophet is required for ProphetFit.forecast "
                              "(`pip install prophet`).")

        future = self._prophet.make_future_dataframe(periods=horizon_days,
                                                     freq="D")
        fc = self._prophet.predict(future)

        # Clip negatives — Prophet has no floor and a turned-over wave can
        # push the additive trend below zero.
        yhat = np.clip(fc["yhat"].to_numpy(), 0, None)
        lower = np.clip(fc["yhat_lower"].to_numpy(), 0, None)
        upper = np.clip(fc["yhat_upper"].to_numpy(), 0, None)

        n_fit = len(self.fit_dates)
        kind = np.array(["fitted"] * n_fit + ["forecast"] * horizon_days)

        # Match SIRFit/SEIRFit: no interval on the in-sample portion.
        lo = np.concatenate([np.full(n_fit, np.nan), lower[n_fit:]])
        hi = np.concatenate([np.full(n_fit, np.nan), upper[n_fit:]])

        return pd.DataFrame({
            "date": pd.to_datetime(fc["ds"].to_numpy()),
            "kind": kind,
            "new_cases": yhat,
            "lower": lo,
            "upper": hi,
        })


# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------

def _prep_series(dates, new_cases) -> pd.Series:
    """7-day-centred smoothing, identical preprocessing to epidemic_models."""
    s = pd.Series(np.asarray(new_cases, dtype=float),
                  index=pd.DatetimeIndex(dates))
    s = s.sort_index()
    s = s.rolling(7, min_periods=1, center=True).mean()
    return s.dropna()


def fit_prophet(dates, new_cases,
                horizon_trend: str = "flat",
                weekly_seasonality: bool = True,
                changepoint_prior_scale: float = 0.05) -> ProphetFit:
    """
    Fit Prophet to one wave of reported daily cases.

    Parameters
    ----------
    dates, new_cases : observed daily *new case* series for ONE wave
        (slice it yourself — see epidemic_models.slice_wave, which returns
        dates/values/pop; Prophet ignores the population).
    horizon_trend : 'flat' or 'linear'.
        'flat' (default) holds the trend constant past the fit window — the
        less-wrong choice for a wave that has already peaked. 'linear' lets
        Prophet keep extrapolating the last trend slope.
    weekly_seasonality : model the weekly (reporting-artefact) cycle.
    changepoint_prior_scale : Prophet's trend flexibility knob. Higher =>
        the trend bends more readily to the data.

    Notes
    -----
    `population` is intentionally not a parameter — Prophet is curve-fitting,
    not compartmental. `fit_sir`/`fit_seir` take population; this does not.
    That asymmetry is expected, not an oversight.
    """
    if not _HAVE_PROPHET:
        raise ImportError("prophet is not installed. It is an optional "
                          "dependency (heavy: cmdstanpy + a compiler "
                          "toolchain). Install with `pip install prophet`, "
                          "or use fit_sir / fit_seir from epidemic_models "
                          "instead.")
    if horizon_trend not in ("flat", "linear"):
        raise ValueError("horizon_trend must be 'flat' or 'linear', "
                         f"got {horizon_trend!r}")

    s = _prep_series(dates, new_cases)
    df = pd.DataFrame({"ds": s.index, "y": s.values})

    model = Prophet(
        growth=horizon_trend if horizon_trend == "linear" else "flat",
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.90,   # match the 90% residual-bootstrap band of SIR/SEIR
    )
    model.fit(df)

    # In-sample fit, for RMSE and the 'fitted' rows of forecast().
    in_sample = model.predict(df)
    fitted_new = np.clip(in_sample["yhat"].to_numpy(), 0, None)
    obs = s.to_numpy()
    resid_std = float(np.nanstd(obs - fitted_new))

    return ProphetFit(
        model="Prophet",
        fit_dates=s.index,
        observed_new=obs,
        fitted_new=fitted_new,
        residual_std=resid_std,
        _prophet=model,
        _horizon_trend=horizon_trend,
    )


if __name__ == "__main__":
    # Smoke test on synthetic data so this runs without the dataset.
    if not _HAVE_PROPHET:
        raise SystemExit("prophet not installed; skipping smoke test. "
                         "This is expected — prophet is optional.")
    rng = np.random.default_rng(0)
    days = pd.date_range("2021-06-01", periods=120, freq="D")
    # A rough bell-shaped wave + weekly wobble + noise.
    t = np.arange(120)
    wave = 40_000 * np.exp(-((t - 55) ** 2) / (2 * 18 ** 2))
    weekly = 1 + 0.15 * np.sin(2 * np.pi * t / 7)
    noisy = np.clip(wave * weekly + rng.normal(0, 1500, size=120), 0, None)

    fit = fit_prophet(days, noisy)
    print(fit.summary())
    fc = fit.forecast(horizon_days=30)
    print(fc.tail())
