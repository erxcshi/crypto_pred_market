# CLAUDE.md — crypto_pred_market

> Living notes for Claude. Update this file as understanding improves. Last verified: 2026-07-07.

## What this is
Probability-calibration and trading system for crypto 15-minute prediction markets.
End-to-end system that (1) scrapes crypto spot + prediction-market data, (2) builds a per-second feature
table, (3) trains a logistic-regression / Bayesian-logistic model to estimate P(YES) for Kalshi `KX{coin}15M`
up/down markets, and (4) runs a live async paper/real trading engine against Kalshi that trades when the
model's probability diverges from the market price.

Coins: **BTC (primary), ETH, XRP, SOL**. BTC is the traded asset; ETH/XRP/SOL Kalshi quotes are cross-asset
features. Markets are 15-minute binary "will price be above strike at close" contracts.

## Environment
- **Python**: repo targets 3.13 (`.python-version`, README). The machine's default `python` is **3.11.9**
  with `numpy`/`pandas` but **no `polars`, no `.venv`, no `.env`**. Analysis of committed CSV logs works with
  stock pandas; running scrapers/filter/engine needs the venv + deps in `requirements.txt` (note: that file is
  UTF-16 encoded).
- Run everything as a module from repo root, e.g. `python -m data_gather.scrapers.run_all` (see `to_run.txt`).
- Bash tool = Git Bash; primary shell is PowerShell. Set `PYTHONIOENCODING=utf-8` for scripts that print.

## Data collector v2 (built + verified 2026-07-08) — the current collection path
`DATA_SINK_MODE=parquet python -m data_gather.scrapers.run_all` runs 12 supervised child processes
(per-child restart w/ backoff) writing **local hourly parquet** to `data/{table}/date=/hour=/part-*.parquet`
(zstd, file-per-flush + atomic rename — hard-kill-safe, verified with kill -9). Key modules:
- `data_gather/parquet_sink.py` — `ParquetDataSink`, drop-in `DataSink` mode (writer thread, bounded queue,
  explicit schemas from `parquet_schemas.py` with string→number coercion, heartbeat to `data/_status/*.json`)
- `data_gather/monitor.py` — staleness/error watcher (thresholds per table; optional `ALERT_WEBHOOK_URL`)
- `data_gather/compact.py` — daily merge of closed dates into `compacted-D.parquet` (verified row counts) +
  `--retention-days`; schedule per `deploy/` (NSSM/Task Scheduler on Windows, systemd on Linux)
- `data_gather/scrapers/ws_base.py` — shared `WsScraper`/`StreamBuffer` base for all new WS scrapers
- Scrapers/tables: kalshi_ws (deltas+trades+ticker_v2, 15M+hourly `KX{coin}D` series, 5-min REST re-snapshots;
  needs Kalshi creds), kalshi.py (5s market snapshots; `KALSHI_OB_OVERRIDE=1`+`KALSHI_POLL_SECONDS=1` only for
  the trading engine — rate-limit hog), kalshi_settlements (ground-truth labels incl. `expiration_value` BRTI
  print — verified), polymarket_ws (CLOB WS book/price_change/trades; book explosion throttled to 30s,
  price_change carries per-change asset_id), coinbase_ws + coinbase_l2 (BTC/ETH full depth `level2_batch`,
  500-level snapshots + 30-min resubscribe), kraken_ws, bitstamp_ws, binanceus_ws (trades sparse — thin venue,
  quotes are the signal), deribit_ws (DVOL + near-spot option tickers + perps + index), alpaca_ws (SPY/QQQ,
  needs free Alpaca keys; VIX poll runs keyless).
- Read data: `pl.scan_parquet('data/{table}/**/*.parquet')`. Env template: `.env.example`.
- Kalshi public API rate limit is shared per-IP across all Kalshi modules — keep request gaps (429s verified
  gone at current settings). Kalshi has 15M series for DOGE/ADA/BCH/ZEC/NEAR/TON/BNB/HYPE too —
  `KALSHI_EXTRA_SERIES` env adds them (easy sample-size multiplier).
- Venv: `.venv` (py3.11) with pyarrow/websockets/aiohttp/dotenv/polars/cryptography installed.

## Architecture / data flow
```
data_gather/scrapers/*  --(async, 1s poll or WS)-->  Supabase (Postgres tables)
   coinbase_ws.py   BTC/ETH/XRP/SOL trades+quotes (WebSocket)   -> coinbase_trades / coinbase_quotes
   kalshi.py        KX{coin}15M REST poll + live orderbook       -> kalshi_markets
   kalshi_ws.py     Kalshi WS orderbook deltas + trades (auth)   -> kalshi_orderbook_deltas / kalshi_trades
   deribit_vol.py   BTC/ETH near-spot option IV (25d/ATM)        -> deribit_option_vols
   polymarket.py    Polymarket up/down CLOB + Binance strike     -> polymarket_markets

Supabase --> data_gather/postgres_extract.py OR supabase_extract.py --> data_gather/raw_data/*.csv
raw_data --> data_gather/filter.py (filter_data class + build_btc_feature_tables) --> filtered_data/ , final_data/
final_data --> models/*.ipynb  (train) --> trading/model_artifacts.npz  (via save_model.py / save_log_reg.py)
model_artifacts.npz --> trading/engine.py (live inference + trading)
```
- `config.py` (repo root): Supabase client + `DataSink` (mode `'actual'`=Supabase, `'test'`=append CSV).
- Two DB extract paths exist: `postgres_extract.py` (SQLAlchemy/psycopg2, needs `PASSWORD/HOST/PORT/DATABASE`)
  and `supabase_extract.py` (supabase-py REST, needs `SUPABASE_URL/SUPABASE_KEY`). They produce the same CSVs.
- The real feature engineering lives in **`data_gather/filter.py`** (`filter_data` class + `build_btc_feature_tables`).
  Run `python -m data_gather.filter` to regenerate `filtered_data/` then `final_data/final_data.csv`.

## The model
- `trading/model.py`: `BayesLogRegModel` — vectorised posterior-predictive logistic regression over MCMC draws.
  Loads `model_artifacts.npz` = `posterior_draws (n_draws, N+1)`, `feature_means (N,)`, `feature_stds (N,)`.
  `predict(feats)` returns `(p_hat, p_std)`. The same class also serves a plain sklearn logreg (saved as a
  single "draw" so `p_std=0`) via `save_log_reg.py`.
- **N_FEATURES = 58** (verified). Layout: 8 global/BTC (`time_to_close` + 7 BTC microstructure, note BTC
  ask/bid/mid are NOT model inputs) + 13×3 cross-asset (ETH/XRP/SOL) + 11 `btc_spot_*`. `FEATURE_NAMES` order
  in `model.py` must exactly match training columns (save_* scripts assert this).
  ⚠️ Doc drift: `model.py` docstrings mention "64/65" — stale; the real count is 58 (+1 intercept = 59).
- `trading/features.py`: `FeatureEngine` re-implements `filter.py`'s feature math in streaming form
  (`KalshiMarketBuffer` per-event rolling windows + `SpotBuffer` 1s bars). Must stay in sync with `filter.py`.
- Training notebooks split with **`GroupShuffleSplit` grouped by market event** (good — avoids row-level
  leakage across the ~250 autocorrelated per-second rows of a single 15M market). CV uses `TimeSeriesSplit`.
  It is a *random* group split, not a forward/temporal split.

## ⚠️ Model quality verdict (independently verified 2026-07-07)
Verified from committed logs `trading/bayes_pnl_log.csv` (6,381 fills, 134 markets) and
`trading/feature_log_bayes.csv` (57,158 per-second scores, 226 markets). Outcomes reconstructed as
`last yes_mid > 0.5` (99.2% agreement with the ground-truth `result` on traded tickers).
My numbers (Brier 0.070, LogLoss 0.235, AUC 0.968) match the notebook's own holdout (0.073 / 0.233 / 0.969),
so the reconstruction is trustworthy.

**Bottom line: the model does NOT beat the market price where it matters. Aggregate metrics are inflated by
already-decided markets; there is no demonstrated forecasting edge.**

Evidence:
1. **No edge over the market mid.** Model vs. using `yes_mid` as the probability: Brier 0.0703 vs 0.0710,
   LogLoss 0.2354 vs **0.2317 (market better)**, AUC 0.9680 vs 0.9676 — statistically a tie.
2. **Coin-flip on hard cases.** The notebook itself reports *"Accuracy on hard cases (0.35 < p_hat < 0.65):
   0.512 (n=6418)."* All the headline 0.89 accuracy / 0.97 AUC comes from easy, near-settled markets.
3. **Trivial-sample inflation.** 68% of scored rows have mid <0.1 or >0.9 (already decided). Only ~226 distinct
   markets, ~253 autocorrelated rows each. Terminal-snapshot AUC = 1.000 (meaningless). In the *hard* early
   window (ttc 200–280s): AUC model 0.920 vs mid 0.921, Brier model 0.1216 vs **mid 0.1135 (market better)**.
4. **Mid-range miscalibration.** For model bins 0.4–0.6, empirical YES rate is 0.66–0.74 (under-confident by
   ~15–20 pts); well-calibrated only at the extremes.
5. **PnL is a longshot artifact.** Paper PnL = +$1,388 over ~2.4 days, but **96% ($1,338) comes from entries
   priced ≤ 10¢** (buying cheap near-sure NO/YES). Every mid-price bucket is ~break-even or negative. This is a
   pennies-in-front-of-a-steamroller profile that likely won't survive real fills / slippage / adverse
   selection in thin 15M books. It is paper-only (simulated VWAP fills at the touch).
6. **The engine's bias-correction hurts calibration.** `_apply_p_hat_bias_correction` (config `P_HAT_BIAS_*`)
   was tuned to reduce over-firing, but degrades probability quality: Brier 0.0758 / LogLoss 0.2845 (worse than
   raw). Any measured directional "edge" (Q1: when p_hat>mid, realized YES rate > mid) is concentrated in the
   late window and is plausibly a **stale-orderbook latency pickoff**, not superior forecasting.

**Is it good enough?** As an academic pipeline + engineering artifact: yes, genuinely impressive breadth.
As a probability model with edge over the market: **no** — it's ~tied with reading the price, and worse where
uncertainty is real. The single biggest missing analysis: the notebooks never benchmark `p_hat` against the
`yes_mid` baseline.

### Follow-up experiment (2026-07-07): can *any* model beat the mid out-of-sample?
Trained pure-numpy ridge logistic regression on the 58 logged features, **forward split by market**
(train = first 158 markets by time, test = last 68), evaluated on the test set (`scratchpad/experiment.py`):
- **All test rows:** raw mid Brier 0.0849 ≈ recalibrated mid 0.0848; `mid + 58 features` **worse** (0.0914);
  58 features w/o mid worst (0.0990). Adding features hurts.
- **Hard region (0.2<mid<0.8):** raw mid LogLoss **0.699 ≈ random (0.693)** — even the *market* is barely
  better than a coin flip here. Features don't help (Brier 0.2525 → 0.2640 worse).
- **Core (0.3<mid<0.7):** features give a small AUC bump (0.42→0.56) but calibration still worsens.

**Deeper conclusion:** the 15-minute crypto outcome is **intrinsically near-unpredictable when the market is
uncertain** (this is close to a random walk over a strike). This is a *fundamental ceiling*, not just a model
flaw — there may be very little genuine edge to extract. Effort is better spent on (a) latency/microstructure
pickoff of the stale thin book (the one place features showed faint directional signal), or (b) accepting the
market price as the probability and competing on execution/fees, than on trying to out-forecast the mid.

## Strategy revamp plan (agreed 2026-07-08) — supersedes the roadmap below where they conflict
Premise from the audit: forecasting the 15-min outcome is near-impossible where the market is uncertain.
The model's job is a **fast, well-calibrated fair value**, not outcome prediction. Three real edges:
(1) speed — fair value updates ~100ms off the Coinbase tick while thin Kalshi books lag seconds;
(2) maker economics — taker fee is ceil(7·p·(1−p))¢ ≈ 2¢ at p=0.5 (~4% of a 50¢ contract) while maker pays
none; the engine currently always takes — quote fair ± margin instead; (3) cross-venue divergence vs the
identical Polymarket 15M markets (already scraped, join commented out in filter.py).

Modeling: baseline `P(YES) = Φ(distance_from_strike / (S·σ·√ttc))` with EWMA realized vol from Coinbase 1s
bars (+ Deribit ATM IV blend). ML only on the residual: include `logit(mid)` + `logit(fair_bs)` as features so
the model can only learn increments (flow imbalance, spot lead-lag, depth asymmetry, cross-venue gaps).
Evaluation: forward-in-time split by market, hard region (0.2<mid<0.8) only, always vs the mid baseline,
isotonic calibration fit on validation.

Data layer: **store raw events, never pre-aggregated 1s stats** — features are derived offline by replaying
events (same code drives the backtest). `kalshi_ws.py` already does snapshot+delta+trade capture; changes:
(a) sink high-rate streams to local hourly Parquet partitioned by date/stream (DuckDB to query), not Supabase;
(b) periodic snapshots (~5 min/market) so a dropped delta only corrupts one segment; (c) keep both exchange
`ts_ms` and local receive time (latency measurement); (d) log own orders/quotes/fills for fill-sim and
adverse-selection measurement.

Sequence: 1) 24/7 collector w/ Parquet + gap monitoring — **sample size is the binding constraint**
(226 markets now; ~384/day possible → ~11.5k/month, so start collection before modeling). 2) event replayer +
backtest harness (fills at far touch, latency, real fees) — gates everything. 3) Φ(d) fair-value model,
verify calibration, measure |fair−mid| vs fees. 4) residual ML, maker-quoting, Polymarket divergence.
5) small live capital only after fee-adjusted edge shows in the hard region.

## Improvement roadmap (highest leverage first)
1. **Benchmark against the market, always.** Add `yes_mid` (and a mid→logit recalibration) as a baseline in the
   notebooks and in the trading gate. Only trade when the model beats the market baseline out-of-sample on a
   *per-market, forward-in-time* split — not a random group split.
2. **Fix evaluation to a forward/temporal split** and report metrics *restricted to the hard region* (e.g.
   0.2<mid<0.8) and to a fixed early ttc bucket. Headline accuracy on all rows is not informative.
3. **Recalibrate rather than bias-shift.** Replace the hand-tuned `P_HAT_BIAS_*` offsets + dead-zone with a
   proper post-hoc calibrator (isotonic / Platt) fit on held-out data; validate it lowers Brier/LogLoss.
4. **Model the residual, not the price.** The market mid already encodes most signal. Train on
   `outcome` with `logit(yes_mid)` as a feature (or predict `outcome - mid`) so the model can only add value
   on top of the price. This directly targets edge.
5. **Realistic backtest before more live capital.** Replay the WS/orderbook history with fills at the *far*
   touch, real Kalshi taker fees (`KalshiClient.taker_fee`, already implemented), and latency; measure PnL
   excluding ≤10¢ entries to see whether any genuine edge remains.
6. **Scraping/data next steps:** the Deribit option-IV and Polymarket feeds are collected but the Polymarket
   join is commented out in `filter.py` (`attach_btc_polymarket_features`) and options aren't used as features
   at all. Cross-venue signals (Polymarket implied prob, IV/vol term structure) are the most plausible source
   of *new* information beyond the Kalshi book — wire them in and test if they improve the hard-region metrics.
7. **Portability/cleanup:** (done 2026-07-07) `filter.py` `__main__` now uses `final_data_dir`; the dead
   `build_features/` stub was removed; `model.py` feature-count comments corrected to 58/59.

## Known gotchas
- `raw_data/`, `filtered_data/`, `final_data/` and `trading/model_artifacts.npz` are **gitignored and absent**.
  You cannot run `filter.py`, training, or live inference without regenerating data from Supabase (needs creds)
  or re-training. The two big `trading/*.csv` logs ARE committed and are the best available ground truth.
- Live inference vs training skew: training features were built from cached `/markets` quotes; the engine
  overrides them with the live `/orderbook` top-of-book. Intentional, but a train/serve mismatch to watch.
- `distance_from_strike` depends on `floor_strike`; if Kalshi returns strike=0 the feature becomes ~spot price
  (garbage). Engine warns once per coin.
- Kalshi taker fee is variance-proportional `ceil(7·p·(1-p))¢`, not flat — already handled everywhere.

## Verification / how to reproduce the audit
Stock pandas + numpy is enough (no sklearn/polars on the default interpreter). Scratchpad scripts used:
`verify.py` (PnL + calibration + reliability tables), `verify2.py` (info-over-mid, PnL attribution,
early/terminal-window AUC), `experiment.py` (forward-split ridge-logreg beat-the-mid test, pure numpy IRLS).
Outcome labels are reconstructed as last `yes_mid>0.5`.
Only `trading/bayes_pnl_log.csv` is still committed (PnL analysis reproduces from it). The 30MB
`trading/feature_log_bayes.csv` used for the calibration analysis was removed from tracking on 2026-07-07 as
runtime bloat — recover it from git history, or regenerate by running the engine, to re-run the calibration
parts of the audit.

## Secrets / credentials needed (deferred — set up only when running live components)
Create a `.env` at repo root. Nothing below is committed. **Ask the user for these when a task needs them:**
- `SUPABASE_URL`, `SUPABASE_KEY` — used by `config.py` / `supabase_extract.py` (data sink + REST extract).
- `PASSWORD`, `HOST`, `PORT`, `DATABASE` — direct Postgres extract (`postgres_extract.py`).
- `KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH` — Kalshi RSA API auth for `kalshi_ws.py` and the trading engine.
  The private key `.pem/.p8` is gitignored; the trader signs requests with RSA-PSS-SHA256.
- Trading real orders: run `python -m trading.engine --dry-run` first; live mode places real Kalshi orders.
