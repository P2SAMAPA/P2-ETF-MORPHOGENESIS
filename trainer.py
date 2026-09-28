"""
trainer.py  —  P2 MORPHOGENESIS trainer

Maps each universe onto 2D market "state surfaces", runs a persistent
reaction-diffusion field over them, and asks whether reading that field at each
ETF's location helps rank ETFs by their forward return relative to the universe
average — over and above a baseline that knows the ETF's position and the static
crowding density but none of the dynamics.

Design choices (each exists to keep the result honest):
  * Cross-sectional target. Features and the forward H-day return are z-scored
    across ETFs each day, so market drift/beta cancels and volatile ETFs cannot
    win just by being volatile. Picks are "best relative to the universe".
  * Baseline ablation. Every configuration is evaluated for the morpho model AND
    a no-dynamics baseline on the same test dates; only the difference is credit.
  * Non-overlapping test points, spaced `horizon` days apart; Sharpe annualized
    by sqrt(252/horizon); trading cost charged on the fraction of the top-N
    replaced at each rebalance.
  * Selection/holdout split. The first 70% of test periods choose the
    configuration; the last 30% is a holdout used for the reported numbers and
    the confidence label.
  * The product is what is tested: the metric is the top-N minus universe-average
    return, not a long/short of every ETF.
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import morpho_model as mm
from data_manager import load_master_data, validate_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _f(x):
    """JSON-safe float (NaN/inf -> None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(d: Dict) -> Dict:
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
            for k, v in d.items()}


def confidence_label(hold: Dict) -> str:
    """
    Graded on the HOLDOUT of the selected configuration (a single pre-selected
    config, so a one-sided 5% test is legitimate), and only upward if the field
    beats the no-dynamics baseline.
      High   : holdout rank-IC t >= IC_T_HIGH, net top-N spread > 0, and the
               morpho model beats the baseline on both IC and net spread.
      Medium : IC > 0, net spread > 0, and it beats the baseline on IC.
      Low    : anything else, or fewer than MIN_HOLDOUT_PERIODS test periods.
    """
    if hold.get("n", 0) < config.MIN_HOLDOUT_PERIODS:
        return "Low"
    ic, sp = hold["ic_mean"], hold["net_spread_mean"]
    d_ic, d_sp = hold["d_ic_mean"], hold["d_spread_mean"]
    if hold["ic_t"] >= config.IC_T_HIGH and sp > 0 and d_ic > 0 and d_sp > 0:
        return "High"
    if ic > 0 and sp > 0 and d_ic > 0:
        return "Medium"
    return "Low"


def select_config(grid: List[Dict], surface: Optional[str] = None):
    """
    Choose a configuration from the SELECTION segment only: eligible if rank-IC
    and net top-N spread are both positive there; among eligible, rank-sum of
    the two t-stats. If nothing is eligible, fall back to all configs and
    report gate_passed=False.
    """
    pool = [g for g in grid if (surface is None or g["surface"] == surface) and g["selection"].get("n", 0) >= 20]
    if not pool:
        return None, False
    elig = [g for g in pool if g["selection"]["ic_mean"] > 0 and g["selection"]["net_spread_mean"] > 0]
    gate = bool(elig)
    cand = elig if gate else pool
    ic_order = {id(g): r for r, g in enumerate(sorted(cand, key=lambda g: g["selection"]["ic_t"]))}
    sp_order = {id(g): r for r, g in enumerate(sorted(cand, key=lambda g: g["selection"]["net_spread_t"]))}
    best = max(cand, key=lambda g: (ic_order[id(g)] + sp_order[id(g)], g["selection"]["ic_t"]))
    return best, gate


def make_picks(scores: np.ndarray, tickers: List[str], dispersion: float, horizon: int,
               confidence: str, top_n: int) -> List[Dict]:
    order = np.argsort(-scores)[:top_n]
    return [{
        "ticker": tickers[j],
        "expected_return": round(float(scores[j] * dispersion * 100), 3),
        "score_z": round(float(scores[j]), 4),
        "horizon_days": int(horizon),
        "confidence": confidence,
    } for j in order]


def pattern_diagnostics(surface: str, sim: Dict, px: np.ndarray, py: np.ndarray, tickers: List[str],
                        dates: List[str], vix: Optional[np.ndarray]) -> Dict:
    n = len(dates)
    snap = sim["snapshot"]
    conc, amp = sim["conc"], sim["amp"]
    valid = np.isfinite(conc)
    out = {"axis_labels": list(config.AXIS_LABELS[surface])}
    if snap is None or not valid.any():
        return out

    def pctile(series):
        s = series[valid]
        return float((s <= s[-1]).mean()) if len(s) else None

    last = n - 1
    k = config.MIGRATION_DAYS
    movers = []
    if last - k >= 0 and np.isfinite(px[last - k]).all() and np.isfinite(px[last]).all():
        dx, dy = px[last] - px[last - k], py[last] - py[last - k]
        mag = np.hypot(dx, dy)
        for j in np.argsort(-mag)[:6]:
            movers.append({"ticker": tickers[j], "dx": round(float(dx[j]), 3), "dy": round(float(dy[j]), 3),
                           "magnitude": round(float(mag[j]), 3)})
    path_idx = [i for i in range(max(0, last - 60), n) if np.isfinite(sim["centroid"][i]).all()]
    step = config.SERIES_STEP
    idx = np.arange(0, n, step)
    idx = idx[valid[idx]]
    out.update({
        "snapshot": {
            "u": np.round(snap["u"], 3).tolist(),
            "v": np.round(snap["v"], 3).tolist(),
            "s": np.round(snap["s"], 3).tolist(),
            "peaks": snap["peaks"],
            "tickers": tickers,
            "x": [round(float(v), 4) for v in px[last]],
            "y": [round(float(v), 4) for v in py[last]],
        },
        "n_peaks": _f(sim["npeaks"][last]),
        "peak_share": _f(sim["peak_share"][last]),
        "concentration": _f(conc[last]),
        "concentration_pctile": pctile(conc),
        "amplitude": _f(amp[last]),
        "amplitude_pctile": pctile(amp),
        "migration": movers,
        "centroid_path": [[round(float(sim["centroid"][i][0]), 4), round(float(sim["centroid"][i][1]), 4)] for i in path_idx],
        "series": {
            "dates": [dates[i] for i in idx],
            "concentration": [_f(conc[i]) for i in idx],
            "amplitude": [_f(amp[i]) for i in idx],
            "n_peaks": [_f(sim["npeaks"][i]) for i in idx],
            "vix": [_f(vix[i]) for i in idx] if vix is not None else None,
        },
    })
    if vix is not None:
        out["regime_lead"] = mm.regime_lead_diagnostic(np.where(valid, conc, np.nan), vix, dates, config.LEAD_PARAMS)
    return out


def analyze_universe(name: str, tickers: List[str], prices_df: pd.DataFrame,
                     macro_df: Optional[pd.DataFrame]) -> Optional[Dict]:
    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        return None
    px_df = prices_df[available]
    px_df = px_df[~px_df.isna().any(axis=1)]
    if len(px_df) < 400:
        logger.warning(f"Not enough data for {name}")
        return None

    values = px_df.values
    returns = np.diff(np.log(values), axis=0)
    log_prices = np.log(values[1:])
    dates = [d.strftime("%Y-%m-%d") for d in px_df.index[1:]]
    n = len(dates)

    vix = dvix = None
    if macro_df is not None and "VIX" in macro_df.columns:
        v = macro_df["VIX"].reindex(px_df.index).ffill().values
        if np.isfinite(v).all() and (v > 0).all():
            vix = v[1:]
            dvix = np.diff(np.log(v))

    cfg = dict(ridge_lambda=config.RIDGE_LAMBDA, min_train_days=config.MIN_TRAIN_DAYS, cost_bps=config.TRADING_COST_BPS)
    fwd_by_h = {h: mm.compute_forward_returns(returns, h) for h in config.HORIZONS}
    yz_by_h = {h: mm.cs_zscore(fwd_by_h[h]) for h in config.HORIZONS}

    grid, wf_store, live_ctx, patterns = [], {}, {}, {}
    for surface in config.SURFACES:
        if surface == "macro_risk" and dvix is None:
            logger.info(f"  [{name}] skipping macro_risk surface (no VIX column)")
            continue
        logger.info(f"  [{name}] {surface}: building surface + running reaction-diffusion field...")
        px, py = mm.build_surface_positions(surface, returns, log_prices, dvix, config.SURFACE_PARAMS)
        sim = mm.simulate_morphogenesis(px, py, config.RD_PARAMS)
        if sim["first_valid"] is None:
            continue
        Fz = mm.cs_zscore(sim["feats"])
        patterns[surface] = pattern_diagnostics(surface, sim, px, py, available, dates, vix)

        for h in config.HORIZONS:
            cp = mm.build_cross_products(Fz, yz_by_h[h])
            for window in config.WINDOWS:
                wf = mm.walk_forward(Fz, cp, fwd_by_h[h], h, window, config.TOP_N, cfg)
                key = (surface, h, window)
                live_ctx[key] = (Fz, cp)
                if wf is None:
                    continue
                m = len(wf["rows"])
                split = int(m * config.SELECTION_FRACTION)
                entry = {
                    "surface": surface, "horizon": h, "window": window, "n_periods": m,
                    "selection": _clean(mm.summarize(wf, 0, split, h)),
                    "holdout": _clean(mm.summarize(wf, split, m, h)),
                    "full": _clean(mm.summarize(wf, 0, m, h)),
                }
                grid.append(entry)
                wf_store[key] = (wf, split)
        logger.info(f"  [{name}] {surface}: done ({len([g for g in grid if g['surface'] == surface])} configs)")

    if not grid:
        return None

    def live_for(g):
        Fz, cp = live_ctx[(g["surface"], g["horizon"], g["window"])]
        return mm.live_scores(Fz, cp, fwd_by_h[g["horizon"]], g["horizon"], g["window"], cfg)

    # ---- selection (or pin) ----
    selected, gate = select_config(grid)
    pinned = False
    pin = config.PINNED_CONFIG
    if pin:
        match = [g for g in grid if g["surface"] == pin["surface"] and g["horizon"] == pin["horizon"] and g["window"] == pin["window"]]
        if match:
            selected, pinned = match[0], True
    if selected is None:
        return None

    ls = live_for(selected)
    conf = confidence_label(selected["holdout"])
    picks = base_picks = []
    if ls is not None:
        picks = make_picks(ls["score_m"], available, ls["dispersion"], selected["horizon"], conf, config.TOP_N)
        base_picks = make_picks(ls["score_b"], available, ls["dispersion"], selected["horizon"], conf, config.TOP_N)

    # ---- best config per surface ----
    surface_picks = {}
    for surface in patterns:
        g, sgate = select_config(grid, surface)
        if g is None:
            continue
        lss = live_for(g)
        surface_picks[surface] = {
            "config": {"surface": surface, "horizon": g["horizon"], "window": g["window"]},
            "gate_passed": sgate,
            "holdout": g["holdout"],
            "confidence": confidence_label(g["holdout"]),
            "picks": make_picks(lss["score_m"], available, lss["dispersion"], g["horizon"],
                                confidence_label(g["holdout"]), config.TOP_N) if lss is not None else [],
        }

    # ---- cumulative net excess curve of the selected config ----
    wf, split = wf_store[(selected["surface"], selected["horizon"], selected["window"])]
    curves = {
        "dates": [dates[i] for i in wf["rows"]],
        "morpho_cum": [round(float(v) * 100, 4) for v in np.cumsum(wf["sn_m"])],
        "baseline_cum": [round(float(v) * 100, 4) for v in np.cumsum(wf["sn_b"])],
        "split_index": int(split),
    }

    top = {
        "surface": selected["surface"], "horizon": selected["horizon"], "window": selected["window"],
        "gate_passed": gate, "pinned": pinned, "confidence": conf,
        "selection": selected["selection"], "holdout": selected["holdout"], "full": selected["full"],
        "n_periods": selected["n_periods"],
    }
    feature_weights = None
    if ls is not None:
        feature_weights = {nm: round(float(w), 5) for nm, w in zip(mm.FEATURE_NAMES, ls["beta_m"])}

    return {
        "tickers": available, "n_days": n, "last_date": dates[-1],
        "selected": top, "picks": picks, "baseline_picks": base_picks,
        "grid": grid, "surface_picks": surface_picks, "curves": curves,
        "patterns": patterns, "feature_weights": feature_weights,
    }


def run_trainer() -> Dict:
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date, "algorithm": "MORPHOGENESIS",
        "top_picks": {}, "selected": {}, "grid": {}, "surface_picks": {}, "baseline_picks": {},
        "curves": {}, "patterns": {}, "feature_weights": {}, "universe_meta": {},
        "settings": {"trading_cost_bps": config.TRADING_COST_BPS, "top_n": config.TOP_N,
                     "selection_fraction": config.SELECTION_FRACTION},
    }

    for name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {name}...")
        out = analyze_universe(name, tickers, prices_df, macro_df)
        if out is None:
            continue
        results["top_picks"][name] = out["picks"]
        results["selected"][name] = out["selected"]
        results["grid"][name] = out["grid"]
        results["surface_picks"][name] = out["surface_picks"]
        results["baseline_picks"][name] = out["baseline_picks"]
        results["curves"][name] = out["curves"]
        results["patterns"][name] = out["patterns"]
        results["feature_weights"][name] = out["feature_weights"]
        results["universe_meta"][name] = {"tickers": out["tickers"], "n_days": out["n_days"], "last_date": out["last_date"]}

        s = out["selected"]
        h = s["holdout"]
        logger.info(f"  ✅ Selected: {s['surface']} / {s['horizon']}d / window {s['window']} "
                    f"(selection gate {'passed' if s['gate_passed'] else 'NOT passed'}) | holdout n={h.get('n')} "
                    f"IC={h.get('ic_mean', 0):+.4f} (t={h.get('ic_t', 0):+.2f}) "
                    f"vs baseline IC={h.get('ic_base_mean', 0):+.4f} | net top-{config.TOP_N} excess "
                    f"{h.get('net_spread_mean', 0) * 1e4:+.1f}bps/period | confidence {s['confidence']}")
        for p in out["picks"]:
            logger.info(f"     {p['ticker']}: {p['expected_return']:+.3f}% excess over {p['horizon_days']}d ({p['confidence']})")

    output_path = f"morpho_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"\n💾 Saved: {output_path}")

    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    run_trainer()
