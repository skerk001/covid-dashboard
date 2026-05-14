"""
Compartmental epidemic models (SIR / SEIR) with least-squares parameter
fitting against observed case data.

Why this and not Prophet?
-------------------------
Prophet is a heavy dependency (pystan / cmdstanpy, a compiler toolchain).
SIR/SEIR need only numpy + scipy, which we already pull in transitively,
and they're *mechanistic* — the fitted parameters (R0, infectious period)
are interpretable, unlike Prophet's additive components.

Caveats — read before trusting any forecast
--------------------------------------------
* A single SIR/SEIR with constant parameters cannot reproduce multi-wave
  dynamics. Interventions, behaviour change, seasonality and variants all
  shift beta over time. We therefore fit to a *single wave window*, not the
  whole pandemic, and treat the forecast as "if conditions held constant".
* "Cases" are not "infections". Reported cases undercount true infections
  by a time-varying factor. We fit to the *shape* of the reported curve and
  scale the susceptible pool via an `ascertainment` knob; absolute compartment
  sizes are therefore only loosely identifiable.
* Forecast intervals here are a crude residual bootstrap, not a real
  posterior. Treat them as "uncertainty exists", not as calibrated bands.

Public API
----------
    fit_sir(dates, new_cases, population, ...)   -> SIRFit
    fit_seir(dates, new_cases, population, ...)  -> SEIRFit
    SIRFit.forecast(horizon_days)                -> DataFrame
    SEIRFit.forecast(horizon_days)               -> DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False


# -----------------------------------------------------------------------------
# ODE right-hand sides
# -----------------------------------------------------------------------------

def _sir_rhs(t, y, beta, gamma, N):
    S, I, R = y
    new_inf = beta * S * I / N
    return [-new_inf, new_inf - gamma * I, gamma * I]


def _seir_rhs(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    new_exp = beta * S * I / N
    return [
        -new_exp,
        new_exp - sigma * E,
        sigma * E - gamma * I,
        gamma * I,
    ]


# -----------------------------------------------------------------------------
# Fit result containers
# -----------------------------------------------------------------------------

@dataclass
class _BaseFit:
    model: str
    params: dict
    population: float
    t0: pd.Timestamp
    fit_dates: pd.DatetimeIndex
    observed_new: np.ndarray
    fitted_new: np.ndarray
    residual_std: float
    y0: tuple

    @property
    def r0(self) -> float:
        """Basic reproduction number implied by the fitted parameters."""
        p = self.params
        return p["beta"] / p["gamma"]

    def _rmse(self) -> float:
        return float(np.sqrt(np.nanmean((self.observed_new - self.fitted_new) ** 2)))

    def summary(self) -> str:
        lines = [
            f"{self.model} fit",
            f"  window      : {self.fit_dates[0].date()} -> {self.fit_dates[-1].date()} "
            f"({len(self.fit_dates)} days)",
            f"  R0          : {self.r0:.2f}",
        ]
        for k, v in self.params.items():
            lines.append(f"  {k:<12}: {v:.4f}")
        lines.append(f"  RMSE        : {self._rmse():,.0f} cases/day")
        return "\n".join(lines)


@dataclass
class SIRFit(_BaseFit):
    def forecast(self, horizon_days: int = 60, n_boot: int = 200,
                 seed: int = 0) -> pd.DataFrame:
        return _forecast(self, horizon_days, n_boot, seed, model="SIR")


@dataclass
class SEIRFit(_BaseFit):
    def forecast(self, horizon_days: int = 60, n_boot: int = 200,
                 seed: int = 0) -> pd.DataFrame:
        return _forecast(self, horizon_days, n_boot, seed, model="SEIR")


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def _simulate(model: str, params: dict, y0: tuple, n_days: int, N: float):
    """Integrate the ODE for n_days and return daily *new* infections."""
    t_eval = np.arange(n_days)
    if model == "SIR":
        sol = solve_ivp(
            _sir_rhs, (0, n_days - 1), y0, t_eval=t_eval,
            args=(params["beta"], params["gamma"], N),
            method="RK45", rtol=1e-6, atol=1e-6,
        )
        S = sol.y[0]
        # New infections per day = -dS (people leaving susceptible).
        new = -np.gradient(S)
    else:  # SEIR
        sol = solve_ivp(
            _seir_rhs, (0, n_days - 1), y0, t_eval=t_eval,
            args=(params["beta"], params["sigma"], params["gamma"], N),
            method="RK45", rtol=1e-6, atol=1e-6,
        )
        S = sol.y[0]
        new = -np.gradient(S)
    new = np.clip(new, 0, None)
    return new, sol


# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------

def _prep_series(dates, new_cases):
    s = pd.Series(np.asarray(new_cases, dtype=float),
                  index=pd.DatetimeIndex(dates))
    s = s.sort_index()
    # Light smoothing — daily case reports are extremely noisy (weekend dips).
    s = s.rolling(7, min_periods=1, center=True).mean()
    s = s.dropna()
    return s


def fit_sir(dates, new_cases, population: float,
            ascertainment: float = 0.2,
            i0_frac: float | None = None) -> SIRFit:
    """
    Fit an SIR model to one wave of reported daily cases.

    Parameters
    ----------
    dates, new_cases : the observed daily *new case* series for ONE wave
        (slice it yourself — see models/wave_windows or analysis.detect_waves)
    population : country population
    ascertainment : fraction of true infections that get reported as cases.
        Lower => larger effective susceptible pool. Rough knob, default 0.2.
    i0_frac : initial infectious fraction; if None, estimated from day-0 cases.
    """
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for model fitting "
                          "(`pip install scipy`).")

    s = _prep_series(dates, new_cases)
    obs = s.values
    n = len(s)
    N = float(population)

    # Effective susceptible pool, deflated by ascertainment.
    N_eff = N * ascertainment
    if i0_frac is None:
        i0 = max(obs[0], 1.0) / max(ascertainment, 1e-6)
    else:
        i0 = i0_frac * N
    y0 = (N_eff - i0, i0, 0.0)

    def resid(theta):
        beta, gamma = theta
        params = {"beta": beta, "gamma": gamma}
        new, _ = _simulate("SIR", params, y0, n, N_eff)
        return (new * ascertainment) - obs

    # beta, gamma. gamma ~ 1/infectious_period (~5-10 days).
    x0 = [0.35, 0.14]
    bounds = ([0.01, 1 / 21], [2.0, 1 / 3])
    res = least_squares(resid, x0, bounds=bounds, method="trf", max_nfev=4000)

    beta, gamma = res.x
    params = {"beta": float(beta), "gamma": float(gamma)}
    fitted_new, _ = _simulate("SIR", params, y0, n, N_eff)
    fitted_new = fitted_new * ascertainment
    resid_std = float(np.nanstd(obs - fitted_new))

    return SIRFit(
        model="SIR", params=params, population=N, t0=s.index[0],
        fit_dates=s.index, observed_new=obs, fitted_new=fitted_new,
        residual_std=resid_std, y0=y0,
    )


def fit_seir(dates, new_cases, population: float,
             ascertainment: float = 0.2,
             incubation_days: float = 4.0,
             i0_frac: float | None = None) -> SEIRFit:
    """
    Fit an SEIR model (adds an Exposed/latent compartment) to one wave.

    `incubation_days` seeds sigma = 1/incubation_days; sigma is then refined
    within a narrow band by the optimiser.
    """
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for model fitting "
                          "(`pip install scipy`).")

    s = _prep_series(dates, new_cases)
    obs = s.values
    n = len(s)
    N = float(population)
    N_eff = N * ascertainment

    if i0_frac is None:
        i0 = max(obs[0], 1.0) / max(ascertainment, 1e-6)
    else:
        i0 = i0_frac * N
    e0 = i0  # seed exposed ~ infectious at the start
    y0 = (N_eff - i0 - e0, e0, i0, 0.0)

    def resid(theta):
        beta, sigma, gamma = theta
        params = {"beta": beta, "sigma": sigma, "gamma": gamma}
        new, _ = _simulate("SEIR", params, y0, n, N_eff)
        return (new * ascertainment) - obs

    sigma0 = 1.0 / incubation_days
    x0 = [0.45, sigma0, 0.14]
    bounds = (
        [0.01, 1 / 14, 1 / 21],
        [2.5, 1 / 1.5, 1 / 3],
    )
    res = least_squares(resid, x0, bounds=bounds, method="trf", max_nfev=6000)

    beta, sigma, gamma = res.x
    params = {"beta": float(beta), "sigma": float(sigma), "gamma": float(gamma)}
    fitted_new, _ = _simulate("SEIR", params, y0, n, N_eff)
    fitted_new = fitted_new * ascertainment
    resid_std = float(np.nanstd(obs - fitted_new))

    return SEIRFit(
        model="SEIR", params=params, population=N, t0=s.index[0],
        fit_dates=s.index, observed_new=obs, fitted_new=fitted_new,
        residual_std=resid_std, y0=y0,
    )


# -----------------------------------------------------------------------------
# Forecast (in-sample fit + out-of-sample projection + crude bootstrap CI)
# -----------------------------------------------------------------------------

def _forecast(fit: _BaseFit, horizon_days: int, n_boot: int,
              seed: int, model: str) -> pd.DataFrame:
    """
    Roll the fitted model forward `horizon_days` past the fit window.

    Returns a tidy DataFrame with columns:
        date, kind ('fitted'|'forecast'), new_cases,
        lower, upper (residual-bootstrap 90% band; NaN on fitted rows)
    """
    asc = 0.2  # keep in sync with the fit default; see caveats in module docstring
    N_eff = fit.population * asc
    total_days = len(fit.fit_dates) + horizon_days

    central, _ = _simulate(model, fit.params, fit.y0, total_days, N_eff)
    central = central * asc

    all_dates = pd.date_range(fit.fit_dates[0], periods=total_days, freq="D")
    n_fit = len(fit.fit_dates)

    # Residual bootstrap: resample fit residuals, add to the forecast tail.
    rng = np.random.default_rng(seed)
    resid = fit.observed_new - fit.fitted_new
    resid = resid[np.isfinite(resid)]
    boot = np.empty((n_boot, horizon_days))
    for b in range(n_boot):
        draws = rng.choice(resid, size=horizon_days, replace=True)
        boot[b] = np.clip(central[n_fit:] + draws, 0, None)
    lower = np.nanpercentile(boot, 5, axis=0)
    upper = np.nanpercentile(boot, 95, axis=0)

    kind = np.array(["fitted"] * n_fit + ["forecast"] * horizon_days)
    lo = np.concatenate([np.full(n_fit, np.nan), lower])
    hi = np.concatenate([np.full(n_fit, np.nan), upper])

    return pd.DataFrame({
        "date": all_dates,
        "kind": kind,
        "new_cases": central,
        "lower": lo,
        "upper": hi,
    })


# -----------------------------------------------------------------------------
# Convenience: pull one wave window out of a country's series
# -----------------------------------------------------------------------------

def slice_wave(countries: pd.DataFrame, country: str,
               start: str, end: str,
               metric: str = "new_cases_smoothed"):
    """
    Helper to extract (dates, values, population) for a single wave window,
    ready to hand to fit_sir / fit_seir.
    """
    sub = countries[countries["location"] == country].sort_values("date")
    sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]
    sub = sub.dropna(subset=[metric])
    if sub.empty:
        raise ValueError(f"No data for {country} in {start}..{end}")
    pop = float(sub["population"].dropna().iloc[-1])
    return sub["date"].values, sub[metric].values, pop


if __name__ == "__main__":
    # Smoke test on synthetic data so this runs without the dataset.
    if not _HAVE_SCIPY:
        raise SystemExit("scipy not installed; skipping smoke test.")
    rng = np.random.default_rng(0)
    days = pd.date_range("2021-06-01", periods=120, freq="D")
    true = {"beta": 0.42, "gamma": 0.16}
    N_eff = 1_000_000 * 0.2
    y0 = (N_eff - 500, 500, 0.0)
    new, _ = _simulate("SIR", true, y0, 120, N_eff)
    noisy = np.clip(new * 0.2 + rng.normal(0, new.max() * 0.02, size=120), 0, None)
    fit = fit_sir(days, noisy, population=1_000_000)
    print(fit.summary())
    fc = fit.forecast(horizon_days=30)
    print(fc.tail())
