"""
morpho_model.py  —  Reaction-diffusion "morphogenesis" signal engine

Idea
----
Turing/Meinhardt reaction-diffusion systems turn a nearly uniform medium into
spatial patterns: a short-range self-reinforcing activator u and a long-range
inhibitor v amplify small inhomogeneities into distinct peaks (nuclei) with
suppressed surroundings. Here the "medium" is a 2D STATE SURFACE of the market
(e.g. return x volatility). Each day every ETF is a point on it, and the cloud
of points is the morphogen source that drives the field:

    du/dt = Du*lap(u) + rho*u^2/((v+eps)(1+kappa*u^2)) - mu*u + sigma*s(x,y,t)
    dv/dt = Dv*lap(v) + rho*u^2/(1+kappa*u^2)          - nu*v

with Dv >> Du. The field is a persistent state advanced a few steps per trading
day, so it has memory: it amplifies clusters of ETFs, inhibits their surround,
and lags/leads as the cloud moves.

Honest scope
------------
* This is not literally "spontaneous" Turing patterning. The kinetics are run
  near threshold and FORCED by the market's own occupancy of the surface, so
  the field is a nonlinear, memory-carrying spatial filter of the cloud
  (center-surround amplification). What is read out is how that filter
  responds, not an autonomous pattern.
* The forecast is a small linear model on features read from the field at each
  ETF's location. It is always compared with a BASELINE that sees the ETF's
  own position and the static crowding density but none of the dynamics, so
  the field is credited only for what it adds beyond that.
* "Detects regime nucleation / risk concentration before conventional regime
  models" is a testable claim, not an assumption: see regime_lead_diagnostic.

Time convention (same as the other engines)
-------------------------------------------
Row k of every array is the trading day dates[k]; returns[k] is the return
realized ON that day. Features at row k use information up to and including
that day's close. forward_returns[k] is the cumulative log return over rows
k+1..k+H.
"""

import numpy as np
import pandas as pd
from scipy.special import ndtr
from typing import Dict, List, Optional, Tuple

BASE_FEATURES = ["x", "y", "x2", "y2", "xy", "density"]
FIELD_FEATURES = ["u", "v", "ux", "uy", "lap_u", "du_dt"]
FEATURE_NAMES = BASE_FEATURES + FIELD_FEATURES
N_BASE = len(BASE_FEATURES)
N_FEAT = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------
def rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing sum over up to `w` rows ending at each row (partial windows early on)."""
    a = np.asarray(a, dtype=float)
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)], axis=0)
    lo = np.maximum(np.arange(a.shape[0]) - w + 1, 0)
    return c[1:] - c[lo]


def trailing_sum(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing sum over exactly `w` rows; NaN until a full window exists."""
    out = rolling_sum(a, w)
    out[: w - 1] = np.nan
    return out


def compute_forward_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
    """forward[t] = sum(returns[t+1 : t+1+horizon]) (log return from close t to close t+H); NaN at the tail."""
    n, T = returns.shape
    cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
    fwd = np.full((n, T), np.nan)
    end = n - horizon
    if end > 0:
        fwd[:end] = cum[horizon + 1: n + 1] - cum[1: n - horizon + 1]
    return fwd


# ---------------------------------------------------------------------------
# State surface
# ---------------------------------------------------------------------------
def compute_raw_axes(surface: str, returns: np.ndarray, log_prices: np.ndarray,
                     dvix: Optional[np.ndarray], sp: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Raw (pre-normalization) x and y axis values, shape (n, T), NaN during warm-up."""
    L = sp["vol_lookback"]
    s1 = trailing_sum(returns, L)
    s2 = trailing_sum(returns ** 2, L)
    var = (s2 - s1 ** 2 / L) / (L - 1)
    log_vol = 0.5 * np.log(np.maximum(var, 1e-10))

    if surface == "return_vol":
        return trailing_sum(returns, sp["return_lookback"]), log_vol

    if surface == "momentum_value":
        mom = trailing_sum(returns, sp["momentum_lookback"])
        W = sp["valuation_lookback"]
        m = trailing_sum(log_prices, W) / W
        m2 = trailing_sum(log_prices ** 2, W) / W
        sd = np.sqrt(np.maximum(m2 - m ** 2, 1e-8))
        cheap = -(log_prices - m) / sd
        return mom, cheap

    if surface == "macro_risk":
        if dvix is None:
            raise ValueError("macro_risk surface needs the VIX series")
        B = sp["beta_lookback"]
        d = dvix[:, None]
        sr = trailing_sum(returns, B)
        sd_ = trailing_sum(d, B)
        srd = trailing_sum(returns * d, B)
        sdd = trailing_sum(d ** 2, B)
        cov = srd / B - (sr / B) * (sd_ / B)
        var_d = np.maximum(sdd / B - (sd_ / B) ** 2, 1e-12)
        return cov / var_d, log_vol

    raise ValueError(f"unknown surface: {surface}")


def to_positions(raw: np.ndarray, window: int, min_days: int) -> np.ndarray:
    """
    Map raw axis values to positions in (0,1): normal CDF of the z-score against
    the trailing pooled (all ETFs x last `window` days) mean/std, using only data
    up to and including each row. Keeps absolute level (a crisis pushes the whole
    cloud up instead of being re-spread uniformly by a per-day rank).
    """
    n, T = raw.shape
    fin = np.isfinite(raw)
    a = np.where(fin, raw, 0.0)
    cnt = rolling_sum(fin.sum(axis=1).astype(float), window)
    s1 = rolling_sum(a.sum(axis=1), window)
    s2 = rolling_sum((a ** 2).sum(axis=1), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / cnt
        sd = np.sqrt(np.maximum(s2 / cnt - mu ** 2, 1e-12))
        pos = ndtr((raw - mu[:, None]) / sd[:, None])
    pos[(cnt < min_days * T)] = np.nan
    pos[~fin] = np.nan
    return pos


def build_surface_positions(surface: str, returns: np.ndarray, log_prices: np.ndarray,
                            dvix: Optional[np.ndarray], sp: Dict) -> Tuple[np.ndarray, np.ndarray]:
    rx, ry = compute_raw_axes(surface, returns, log_prices, dvix, sp)
    px = to_positions(rx, sp["norm_window"], sp["norm_min_days"])
    py = to_positions(ry, sp["norm_window"], sp["norm_min_days"])
    return px, py


# ---------------------------------------------------------------------------
# Reaction-diffusion field
# ---------------------------------------------------------------------------
def _lap(f: np.ndarray) -> np.ndarray:
    p = np.pad(f, 1, mode="edge")
    return p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * f


def kde_source(px: np.ndarray, py: np.ndarray, xs: np.ndarray, h: float) -> np.ndarray:
    """Gaussian kernel density of the ETF cloud on the grid, rows=y, cols=x, normalized to mean 1."""
    gx = np.exp(-((xs[None, :] - px[:, None]) ** 2) / (2 * h * h))
    gy = np.exp(-((xs[None, :] - py[:, None]) ** 2) / (2 * h * h))
    s = gy.T @ gx
    m = s.mean()
    return s / m if m > 0 else np.ones_like(s)


def bilinear(F: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    G = F.shape[0]
    gx = np.clip(px, 0.0, 1.0) * (G - 1)
    gy = np.clip(py, 0.0, 1.0) * (G - 1)
    x0 = np.minimum(gx.astype(int), G - 2)
    y0 = np.minimum(gy.astype(int), G - 2)
    fx, fy = gx - x0, gy - y0
    return (F[y0, x0] * (1 - fx) * (1 - fy) + F[y0, x0 + 1] * fx * (1 - fy)
            + F[y0 + 1, x0] * (1 - fx) * fy + F[y0 + 1, x0 + 1] * fx * fy)


def rd_step(u: np.ndarray, v: np.ndarray, s: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray]:
    u2 = u * u
    sat = 1.0 + p["kappa"] * u2
    du = p["Du"] * _lap(u) + p["rho"] * u2 / ((v + p["eps"]) * sat) - p["mu"] * u + p["sigma"] * s
    dv = p["Dv"] * _lap(v) + p["rho"] * u2 / sat - p["nu"] * v
    u = np.clip(u + p["dt"] * du, 0.0, p["u_max"])
    v = np.clip(v + p["dt"] * dv, 1e-6, p["v_max"])
    return u, v


def init_state(p: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Homogeneous steady state of the reaction terms under a uniform source (s=1)."""
    G = p["grid"]
    u, v = np.ones((G, G)), np.ones((G, G))
    s = np.ones((G, G))
    q = dict(p, Du=0.0, Dv=0.0)
    for _ in range(2000):
        u, v = rd_step(u, v, s, q)
    return u, v


def find_peaks(u: np.ndarray, sigma: float) -> np.ndarray:
    """(row, col) of local maxima above mean + sigma*std."""
    c = np.pad(u, 1, mode="constant", constant_values=-np.inf)
    core = c[1:-1, 1:-1]
    m = np.ones_like(core, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            m &= core >= c[1 + dr: c.shape[0] - 1 + dr, 1 + dc: c.shape[1] - 1 + dc]
    thr = u.mean() + sigma * u.std()
    return np.argwhere(m & (u > thr))


def simulate_morphogenesis(px: np.ndarray, py: np.ndarray, rd: Dict) -> Dict:
    """
    Run one continuous reaction-diffusion simulation through history.

    Returns per-day features for every ETF (n, T, N_FEAT, NaN where invalid),
    per-day pattern diagnostics, and a snapshot of the final day.
    """
    n, T = px.shape
    G = rd["grid"]
    xs = np.linspace(0.0, 1.0, G)
    valid = np.isfinite(px).all(axis=1) & np.isfinite(py).all(axis=1)

    feats = np.full((n, T, N_FEAT), np.nan)
    amp = np.full(n, np.nan)
    conc = np.full(n, np.nan)
    npeaks = np.full(n, np.nan)
    peak_share = np.full(n, np.nan)
    centroid = np.full((n, 2), np.nan)
    U = np.full((n, G, G), np.nan, dtype=np.float32)
    snapshot = None

    if not valid.any():
        return dict(feats=feats, amp=amp, conc=conc, npeaks=npeaks, peak_share=peak_share,
                    centroid=centroid, snapshot=None, first_valid=None)

    h = rd["kde_bandwidth"] * (T / 36.0) ** (-1.0 / 6.0)
    u, v = init_state(rd)
    k = rd["delta_lookback"]
    logG2 = np.log(G * G)

    for t in np.where(valid)[0]:
        x, y = px[t], py[t]
        s = kde_source(x, y, xs, h)
        for _ in range(rd["substeps"]):
            u, v = rd_step(u, v, s, rd)
        U[t] = u

        ux = np.gradient(u, axis=1)
        uy = np.gradient(u, axis=0)
        lap = _lap(u)
        u_i = bilinear(u, x, y)
        if t - k >= 0 and valid[t - k]:
            du_i = u_i - bilinear(U[t - k].astype(float), x, y)
        else:
            du_i = np.full(T, np.nan)

        feats[t, :, 0] = x
        feats[t, :, 1] = y
        feats[t, :, 2] = (x - 0.5) ** 2
        feats[t, :, 3] = (y - 0.5) ** 2
        feats[t, :, 4] = (x - 0.5) * (y - 0.5)
        feats[t, :, 5] = bilinear(s, x, y)
        feats[t, :, 6] = u_i
        feats[t, :, 7] = bilinear(v, x, y)
        feats[t, :, 8] = bilinear(ux, x, y)
        feats[t, :, 9] = bilinear(uy, x, y)
        feats[t, :, 10] = bilinear(lap, x, y)
        feats[t, :, 11] = du_i

        amp[t] = u.std() / (u.mean() + 1e-9)
        p_ = s / s.sum()
        conc[t] = 1.0 + float(np.sum(p_ * np.log(p_ + 1e-15))) / logG2
        pk = find_peaks(u, rd["peak_sigma"])
        npeaks[t] = len(pk)
        centroid[t] = (x.mean(), y.mean())
        if len(pk):
            strengths = u[pk[:, 0], pk[:, 1]]
            top = pk[int(np.argmax(strengths))]
            tx, ty = top[1] / (G - 1), top[0] / (G - 1)
            d = np.stack([np.hypot(x - c / (G - 1), y - r / (G - 1)) for r, c in pk], axis=1)
            peak_share[t] = float(np.mean(np.argmin(d, axis=1) == int(np.argmax(strengths))))

        if t == n - 1:
            pk_list = []
            for r, c in pk:
                pk_list.append({"x": float(c / (G - 1)), "y": float(r / (G - 1)), "strength": float(u[r, c])})
            snapshot = {"u": u.copy(), "v": v.copy(), "s": s.copy(), "peaks": pk_list}

    return dict(feats=feats, amp=amp, conc=conc, npeaks=npeaks, peak_share=peak_share,
                centroid=centroid, snapshot=snapshot, first_valid=int(np.where(valid)[0][0]))


# ---------------------------------------------------------------------------
# Regression machinery (cumulative cross-products => O(1) window fits)
# ---------------------------------------------------------------------------
def cs_zscore(a: np.ndarray) -> np.ndarray:
    """Cross-sectional (axis=1) z-score per day; NaN stays NaN, degenerate days become 0."""
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(a, axis=1, keepdims=True)
        sd = np.nanstd(a, axis=1, keepdims=True)
        out = np.where(sd > 1e-9, (a - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0)
    out[~np.isfinite(a)] = np.nan
    return out


def build_cross_products(Fz: np.ndarray, Yz: np.ndarray) -> Dict:
    n, T, p = Fz.shape
    feat_ok = np.isfinite(Fz).all(axis=(1, 2))
    y_ok = np.isfinite(Yz).all(axis=1)
    ok = feat_ok & y_ok
    F0 = np.where(ok[:, None, None], np.nan_to_num(Fz), 0.0)
    Y0 = np.where(ok[:, None], np.nan_to_num(Yz), 0.0)
    XtX = np.einsum("dtp,dtq->dpq", F0, F0)
    Xty = np.einsum("dtp,dt->dp", F0, Y0)
    return {
        "cXtX": np.concatenate([np.zeros((1, p, p)), np.cumsum(XtX, axis=0)], axis=0),
        "cXty": np.concatenate([np.zeros((1, p)), np.cumsum(Xty, axis=0)], axis=0),
        "cN": np.concatenate([[0], np.cumsum(ok.astype(int))]),
        "feat_ok": feat_ok,
    }


def ridge_solve(XtX: np.ndarray, Xty: np.ndarray, lam_rel: float) -> np.ndarray:
    p = XtX.shape[0]
    lam = lam_rel * np.trace(XtX) / p + 1e-12
    return np.linalg.solve(XtX + lam * np.eye(p), Xty)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def walk_forward(Fz: np.ndarray, cp: Dict, fwd: np.ndarray, horizon: int, window: int,
                 top_n: int, cfg: Dict) -> Optional[Dict]:
    """
    Evaluate the morpho model and the baseline on non-overlapping test points
    spaced `horizon` days apart. At test row i the model is fit on training
    rows [i-window, i-horizon] (targets fully resolved by i; no leakage).
    """
    n, T, p = Fz.shape
    base_idx = list(range(N_BASE))
    lam = cfg["ridge_lambda"]
    min_train = cfg["min_train_days"]
    cost = cfg["cost_bps"] / 1e4

    rows, ic_m, ic_b, sg_m, sg_b, sn_m, sn_b = [], [], [], [], [], [], []
    prev_m, prev_b = set(), set()
    i = 0
    while i <= n - 1 - horizon:
        a, b = max(0, i - window), i - horizon
        if b < a or not cp["feat_ok"][i]:
            i += 1 if b < a else horizon
            continue
        if cp["cN"][b + 1] - cp["cN"][a] < min_train:
            i += horizon
            continue
        XtX = cp["cXtX"][b + 1] - cp["cXtX"][a]
        Xty = cp["cXty"][b + 1] - cp["cXty"][a]
        try:
            beta_m = ridge_solve(XtX, Xty, lam)
            beta_b = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], lam)
        except np.linalg.LinAlgError:
            i += horizon
            continue
        f = Fz[i]
        s_m = f @ beta_m
        s_b = f[:, base_idx] @ beta_b
        r = fwd[i]
        if not np.isfinite(r).all():
            i += horizon
            continue
        top_m = set(np.argsort(-s_m)[:top_n].tolist())
        top_b = set(np.argsort(-s_b)[:top_n].tolist())
        g_m = r[list(top_m)].mean() - r.mean()
        g_b = r[list(top_b)].mean() - r.mean()
        c_m = (top_n - len(top_m & prev_m)) / top_n * cost
        c_b = (top_n - len(top_b & prev_b)) / top_n * cost
        rows.append(i)
        ic_m.append(spearman(s_m, r))
        ic_b.append(spearman(s_b, r))
        sg_m.append(g_m)
        sg_b.append(g_b)
        sn_m.append(g_m - c_m)
        sn_b.append(g_b - c_b)
        prev_m, prev_b = top_m, top_b
        i += horizon

    if len(rows) < 10:
        return None
    return {k: np.array(v) for k, v in dict(rows=rows, ic_m=ic_m, ic_b=ic_b, sg_m=sg_m, sg_b=sg_b,
                                              sn_m=sn_m, sn_b=sn_b).items()}


def summarize(wf: Dict, lo: int, hi: int, horizon: int) -> Dict:
    """Statistics over test periods [lo, hi)."""
    sl = slice(lo, hi)
    ic_m, ic_b = wf["ic_m"][sl], wf["ic_b"][sl]
    sn_m, sn_b, sg_m = wf["sn_m"][sl], wf["sn_b"][sl], wf["sg_m"][sl]
    n = len(ic_m)
    if n < 3:
        return {"n": n}
    ann = np.sqrt(252.0 / horizon)
    sd_net = sn_m.std() + 1e-12
    sd_gross = sg_m.std() + 1e-12
    return {
        "n": int(n),
        "ic_mean": float(ic_m.mean()), "ic_t": tstat(ic_m),
        "ic_base_mean": float(ic_b.mean()), "ic_base_t": tstat(ic_b),
        "net_spread_mean": float(sn_m.mean()), "net_spread_t": tstat(sn_m),
        "gross_spread_mean": float(sg_m.mean()),
        "base_net_spread_mean": float(sn_b.mean()),
        "hit_rate": float((sn_m > 0).mean()),
        "sharpe_net": float(sn_m.mean() / sd_net * ann),
        "sharpe_gross": float(sg_m.mean() / sd_gross * ann),
        "d_ic_mean": float((ic_m - ic_b).mean()), "d_ic_t": tstat(ic_m - ic_b),
        "d_spread_mean": float((sn_m - sn_b).mean()), "d_spread_t": tstat(sn_m - sn_b),
    }


def live_scores(Fz: np.ndarray, cp: Dict, fwd: np.ndarray, horizon: int, window: int,
                cfg: Dict) -> Optional[Dict]:
    """Fit on the latest resolved training window and score the final row (today)."""
    n = Fz.shape[0]
    i = n - 1
    a, b = max(0, i - window), i - horizon
    if b < a or not cp["feat_ok"][i] or cp["cN"][b + 1] - cp["cN"][a] < cfg["min_train_days"]:
        return None
    XtX = cp["cXtX"][b + 1] - cp["cXtX"][a]
    Xty = cp["cXty"][b + 1] - cp["cXty"][a]
    base_idx = list(range(N_BASE))
    beta_m = ridge_solve(XtX, Xty, cfg["ridge_lambda"])
    beta_b = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], cfg["ridge_lambda"])
    f = Fz[i]
    disp = np.nanmean(np.nanstd(fwd[a:b + 1], axis=1))
    return {
        "score_m": f @ beta_m,
        "score_b": f[:, base_idx] @ beta_b,
        "beta_m": beta_m,
        "dispersion": float(disp) if np.isfinite(disp) else 0.0,
    }


# ---------------------------------------------------------------------------
# Regime-lead diagnostic
# ---------------------------------------------------------------------------
def rolling_z(x: np.ndarray, window: int, min_periods: int = 63) -> np.ndarray:
    s = pd.Series(x)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return ((s - mu) / sd.replace(0, np.nan)).values


def _onsets(flag: np.ndarray, dedup: int) -> np.ndarray:
    out, last = [], -10 ** 9
    for t in np.where(flag)[0]:
        if t - last > dedup:
            out.append(t)
        last = t
    return np.array(out, dtype=int)


def regime_lead_diagnostic(conc: np.ndarray, vix: np.ndarray, dates: List[str], lp: Dict) -> Dict:
    """
    Does the pattern's risk-concentration index spike BEFORE a conventional
    VIX stress trigger?

      conventional trigger : 5d-mean VIX crosses above its trailing 252d 75th pctile
      morphogenetic trigger: concentration z-score (trailing 252d) exceeds 1.5
      (both de-duplicated: an onset needs 21 quiet days before it; thresholds fixed in advance)

    Reports how often a morphogenetic trigger sat in the 20 days before a VIX
    onset, against the hit rate expected if the same number of triggers fell at
    random, plus the false-alarm rate.
    """
    n = len(conc)
    z = rolling_z(conc, lp["z_window"])
    vix5 = pd.Series(vix).rolling(lp["vix_smooth"], min_periods=lp["vix_smooth"]).mean()
    thr = vix5.rolling(lp["z_window"], min_periods=126).quantile(lp["vix_pctl"] / 100.0)
    conv_flag = (vix5 > thr).fillna(False).values.astype(bool)
    morph_flag = np.nan_to_num(z, nan=-9.0) > lp["morph_z"]
    ok = np.isfinite(thr.values) & np.isfinite(z)
    if not ok.any():
        return {"available": False}
    start = int(np.argmax(ok))
    conv_flag[:start] = False
    morph_flag[:start] = False
    conv = _onsets(conv_flag, lp["dedup_days"])
    morph = _onsets(morph_flag, lp["dedup_days"])
    LW = lp["lead_window"]
    n_valid = n - start

    leads = []
    for d in conv:
        prior = morph[(morph >= d - LW) & (morph < d)]
        if len(prior):
            leads.append(int(d - prior.min()))
    false_alarms = sum(1 for m in morph if not ((conv > m) & (conv <= m + LW)).any())

    p_win = min(LW / max(n_valid, 1), 1.0)
    return {
        "available": True,
        "n_vix_onsets": int(len(conv)),
        "n_morph_triggers": int(len(morph)),
        "hit_rate": float(len(leads) / len(conv)) if len(conv) else None,
        "chance_hit_rate": float(1 - (1 - p_win) ** len(morph)) if len(morph) else 0.0,
        "median_lead_days": float(np.median(leads)) if leads else None,
        "false_alarm_rate": float(false_alarms / len(morph)) if len(morph) else None,
        "chance_false_alarm_rate": float((1 - p_win) ** len(conv)) if len(conv) else None,
        "vix_onset_dates": [dates[i] for i in conv],
        "morph_trigger_dates": [dates[i] for i in morph],
    }
