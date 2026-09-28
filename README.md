# P2-ETF-MORPHOGENESIS

Reaction-diffusion (developmental-biology) patterns on a 2D map of the market,
used to rank ETFs. Unconventional on purpose, and built so it can fail visibly.

## The idea

Turing/Meinhardt systems turn a nearly uniform medium into spatial patterns: a
short-range self-reinforcing **activator** `u` and a long-range **inhibitor**
`v` amplify small inhomogeneities into distinct peaks (nuclei) with suppressed
surroundings:

```
du/dt = Du·∇²u + rho·u²/((v+eps)(1+kappa·u²)) − mu·u + sigma·s(x,y,t)
dv/dt = Dv·∇²v + rho·u²/(1+kappa·u²)          − nu·v          (Dv ≫ Du)
```

The "medium" is a **state surface** of the market. Every day each ETF is a point
on it, and the kernel density `s` of that cloud of points is the morphogen
source. The field is a persistent state advanced a few steps per trading day, so
it has memory.

State surfaces (`config.SURFACES`):

| surface | x axis | y axis |
|---|---|---|
| `return_vol` | 21d return | 21d realized vol |
| `momentum_value` | 63d momentum | cheapness = −z-score of log price vs own 252d mean |
| `macro_risk` | 63d beta to daily changes in log VIX | 21d realized vol |

Axes become positions in (0,1) via the normal CDF of the z-score against the
trailing pooled (all ETFs × 756d) distribution, so absolute level is kept (a
crisis pushes the whole cloud toward high vol).

## What it does and does not claim

* **Not literally spontaneous patterning.** The kinetics run near threshold and
  are *forced* by the market's own occupancy of the surface. The field is a
  nonlinear, memory-carrying spatial filter of the cloud (center-surround
  amplification). What is read out is how that filter responds.
* **"Valuation" is a price-only proxy** (distance from the ETF's own history).
  The master dataset has no fundamentals. Treat `momentum_value` accordingly.
* **The forecast is small and linear.** Features read from the field at each
  ETF's location (`u`, `v`, ∂u/∂x, ∂u/∂y, ∇²u, 5-day change in `u`) enter a
  pooled ridge regression of the cross-sectionally z-scored forward H-day return.
  Picks are the top 3 by predicted return **relative to the universe average**.

## How the result is kept honest

Lessons from the sibling barycenter engine are built in from the start:

1. **Baseline ablation.** Every configuration is run twice on the same dates:
   the morpho model, and a baseline seeing only the ETF's own position
   (`x, y, x², y², xy`) plus the *static* crowding density there, with none of
   the dynamics. The field is credited only for the difference.
2. **Cross-sectional target.** Features and forward returns are z-scored across
   ETFs each day, so drift/beta cancel and high-vol ETFs cannot win by being
   volatile.
3. **The product is what is tested**: mean forward return of the top-3 minus the
   equal-weight universe, after cost (`TRADING_COST_BPS` round trip on the
   fraction of picks replaced), plus cross-sectional rank-IC.
4. **Non-overlapping test points** spaced one horizon apart; Sharpe annualized by
   `sqrt(252/horizon)`. Training rows are used only if their forward window has
   resolved by the test date.
5. **Selection/holdout split.** The first 70% of test periods choose among
   surfaces × horizons (1/5/20d) × windows (252–1008d). The last 30% is a
   holdout used for the headline numbers and the confidence label. The full
   grid is shown; only the selected row's holdout drives confidence.
6. **Confidence** (`trainer.confidence_label`): *High* needs holdout IC t ≥ 1.65,
   positive net spread, and beating the baseline on both IC and spread; *Medium*
   needs positive IC and spread and beating the baseline on IC; otherwise *Low*
   (also *Low* below 30 holdout periods).

## Pattern readouts (dashboard, Tab 1)

* **Regime nucleation**: number of pattern peaks and pattern amplitude vs history.
* **Clustering**: share of ETFs in the strongest peak's basin.
* **Risk concentration**: entropy-based crowding index vs its own history.
* **Sector migration**: ETFs that moved furthest across the surface in 20 days,
  and the drift of the whole cloud.

## The "before conventional regime models" claim is tested, not assumed

`morpho_model.regime_lead_diagnostic` compares two triggers fixed in advance: a
VIX stress onset (5d-mean VIX above its trailing 75th percentile) and a
concentration-index spike (trailing z-score above 1.5), both de-duplicated to one
per 21 quiet days. It reports how often a concentration spike sat in the 20 days
*before* a VIX onset against what random placement would give, the median lead,
and the false-alarm rate against its chance level. Needs the VIX column; skipped
otherwise.

## Repo structure

```
config.py         universes, surfaces, reaction-diffusion parameters, grid, thresholds
data_manager.py   master parquet from HF (same as sibling repos); VIX optional
morpho_model.py   surfaces, reaction-diffusion field, features, O(1) window ridge fits,
                  walk-forward evaluation, pattern + regime-lead diagnostics
trainer.py        per-universe orchestration, selection/holdout, live picks, JSON output
push_results.py   HfApi.upload_file to the results dataset
us_calendar.py    trading-day helpers
streamlit_app.py  Tab 1 live picks + morphogenetic map; Tab 2 backtest grid, curves, diagnostics
.github/workflows/daily_run.yml   weekday cron after the US close
```

## Running

```bash
pip install -r requirements.txt
python trainer.py                 # writes morpho_results_YYYY-MM-DD.json, pushes to HF
streamlit run streamlit_app.py
```

Set `HF_TOKEN`; create the dataset repo `P2SAMAPA/p2-etf-morphogenesis-results`
or change `config.RESULTS_REPO`. To stop the live configuration re-selecting
daily, set `config.PINNED_CONFIG = {"surface": "return_vol", "horizon": 5, "window": 756}`.

## Validation status (please read)

Tested on synthetic data only; this sandbox cannot reach `huggingface.co`.
* Pure noise: all three universes ended "Low" confidence, rank-IC ≈ 0, the
  field's extra IC over baseline ≈ 0, and the regime-lead hit rate matched
  chance (20% vs 19%). The pipeline does not manufacture edge.
* A signal planted in a field feature was recovered (holdout IC t ≈ 16, gain
  over baseline t ≈ 6.6).
* Full three-universe run at realistic size (≈4,500 days, up to 43 ETFs) takes
  about 70 seconds.

Nothing here says it works on real markets. The first real run is the test; read
the holdout numbers and the baseline comparison before the picks. Realistic
rank-IC for a daily-run cross-sectional signal is small (order 0.01–0.05), and
with 36 configurations searched, a single strong-looking cell is expected by
chance. This is not financial advice.
