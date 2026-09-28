"""
config.py  —  Configuration for P2 MORPHOGENESIS
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-morphogenesis-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

TOP_N = 3

# ---------------------------------------------------------------------------
# The 2D "state surface" the market is mapped onto. Every trading day each
# ETF is a point on the surface; the cloud of points is the morphogen source
# that drives the reaction-diffusion field (see morpho_model.py).
#
#   return_vol      x = 21d return              y = 21d realized vol (log)
#   momentum_value  x = 63d return              y = "cheapness" = minus the
#                                                   z-score of log price vs
#                                                   its own 252d mean. This
#                                                   is a PRICE-ONLY valuation
#                                                   proxy (distance from own
#                                                   history), NOT fundamental
#                                                   valuation: the master
#                                                   dataset has no fundamentals.
#   macro_risk      x = 63d beta to daily     y = 21d realized vol (log)
#                       change in log VIX
#                       (needs the VIX column; skipped if absent)
#
# Each axis is converted to a position in (0,1) as the normal CDF of its
# z-score against the trailing pooled (all ETFs x last NORM_WINDOW days)
# distribution. Unlike a per-day rank, this keeps absolute level: on a
# crisis day the whole cloud shifts toward high vol instead of being
# re-spread uniformly.
# ---------------------------------------------------------------------------
SURFACES = ["return_vol", "momentum_value", "macro_risk"]

SURFACE_PARAMS = {
    "return_lookback": 21,
    "vol_lookback": 21,
    "momentum_lookback": 63,
    "valuation_lookback": 252,
    "beta_lookback": 63,
    "norm_window": 756,
    "norm_min_days": 126,
}

AXIS_LABELS = {
    "return_vol": ("21d return (pooled percentile)", "21d volatility (pooled percentile)"),
    "momentum_value": ("63d momentum (pooled percentile)", "Cheapness vs own 252d history (pooled percentile)"),
    "macro_risk": ("63d beta to VIX changes (pooled percentile)", "21d volatility (pooled percentile)"),
}

# ---------------------------------------------------------------------------
# Reaction-diffusion field: Gierer-Meinhardt activator-inhibitor kinetics
# with saturation, driven by the ETF density (basal production term).
#
#   du/dt = Du*lap(u) + rho*u^2 / ((v+eps)(1+kappa*u^2)) - mu*u + sigma*s
#   dv/dt = Dv*lap(v) + rho*u^2 / (1+kappa*u^2)          - nu*v
#
# u = activator (self-reinforcing, short range), v = inhibitor (long range,
# Dv >> Du), s = kernel density of today's ETF cloud (mean 1). Explicit Euler
# on a grid x grid lattice with no-flux boundaries; Du*dt and Dv*dt must stay
# below 0.25 for stability. The field is a persistent state carried from day
# to day (it has memory), advanced `substeps` steps per trading day.
# ---------------------------------------------------------------------------
RD_PARAMS = {
    "grid": 24,
    "kde_bandwidth": 0.12,     # at 36 ETFs; scaled by (n_etf/36)^(-1/6)
    "substeps": 6,
    "dt": 1.0,
    "Du": 0.02,
    "Dv": 0.20,
    "rho": 0.06,
    "kappa": 0.10,
    "mu": 0.06,
    "nu": 0.05,
    "sigma": 0.03,
    "eps": 0.10,
    "u_max": 30.0,
    "v_max": 30.0,
    "delta_lookback": 5,       # days for the du/dt feature
    "peak_sigma": 0.5,         # peak = local max above mean + this many std
}

# ---------------------------------------------------------------------------
# Prediction: pooled ridge regression of the cross-sectionally z-scored
# forward H-day return on cross-sectionally z-scored features, refit at
# every evaluation date on the trailing WINDOW days.
#
# Two models share the same machinery:
#   baseline : own position on the surface (x, y, x^2, y^2, xy) plus the
#              STATIC crowding density at that spot. No dynamics.
#   morpho   : baseline + the reaction-diffusion field read at the ETF's
#              location (u, v, du/dx, du/dy, lap u, u change over 5 days).
# The morpho model is only credited to the extent it beats the baseline.
# ---------------------------------------------------------------------------
HORIZONS = [1, 5, 20]
WINDOWS = [252, 504, 756, 1008]
RIDGE_LAMBDA = 0.01            # relative to trace(X'X)/p
MIN_TRAIN_DAYS = 60

# Evaluation: test points are spaced `horizon` days apart (no overlapping
# forward-return periods). The first SELECTION_FRACTION of test periods pick
# the configuration; the remainder is a holdout used for the honest numbers
# and the confidence label.
SELECTION_FRACTION = 0.70
MIN_HOLDOUT_PERIODS = 30
TRADING_COST_BPS = 15          # round trip, charged on the fraction of the top-N replaced
IC_T_HIGH = 1.65               # one-sided 5% on a single pre-selected config

# Optional: pin the live configuration instead of re-selecting daily, e.g.
# {"surface": "return_vol", "horizon": 5, "window": 756}. None = select.
PINNED_CONFIG = None

# Regime-lead diagnostic: does the risk-concentration index of the pattern
# spike BEFORE a conventional VIX-based stress trigger? Both trigger
# definitions are fixed here in advance, not tuned.
LEAD_PARAMS = {
    "z_window": 252,
    "morph_z": 1.5,
    "vix_pctl": 75,
    "vix_smooth": 5,
    "dedup_days": 21,
    "lead_window": 20,
}

MIGRATION_DAYS = 20
SERIES_STEP = 3                # decimation of stored time series for the dashboard
