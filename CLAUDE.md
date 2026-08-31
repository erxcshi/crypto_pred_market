# CLAUDE.md — crypto_pred_market

> Living notes for Claude. Update this file as understanding improves. Last verified: 2026-07-18.

<!-- COLLECTOR-HEALTH:START -->
## 🩺 Collector health (auto-checked hourly — this block is replaced whole each hour; don't hand-edit)
**Last check: 2026-08-31 02:30 local / 06:30 UTC — 🔄 RESTARTED (now HEALTHY)**
- Collector was **DOWN** — died ~08-30 21:20 UTC (17:20 local) after ~23 days up as PID 32148; 0 processes, heartbeats ~9.2h stale. Caught by this hourly check and relaunched. Root `run_all` now **PID 27288**, up since 2026-08-31 02:30 local. **12/12** child scrapers alive (13 `data_gather` procs incl. root).
- All 11 table heartbeats fresh (max age 8s in `data/_status/*.json`). `collector.err.log` empty (0 B). `collector.log` rotated to `.log.1` (was 1.25 GB), growing fresh.
- Outage while down: **2026-08-30 21:20 → 2026-08-31 06:30 UTC (~9h10m, all venues)** — logged under "Known data gaps".
- Hourly auto-check loop `70fc8ba0` (re-armed 08-25, fires :23; session-only, 7-day expiry ~09-01).
- Recent checks (newest first): `08-31 02:30 🔄 restarted` · `08-29 02:39–40 ✅` · `08-25 14:47 ✅ (loop re-armed)` · `08-07 22:57 🔄 restarted` · `07-28 17:44 ✅` · `16:35–39 ✅` · `12:21 ✅` · `11:03 ✅`.
<!-- COLLECTOR-HEALTH:END -->

## What this is
Data-collection + trading system for crypto 15-minute prediction markets (Kalshi `KX{coin}15M` up/down
binaries). The v1 model pipeline was audited (2026-07-07), found to have **no edge over the market price**,
and **removed from the repo on 2026-07-18** — the model is being rebuilt from scratch on top of the raw-event
parquet data. Current state: (1) a 24/7 parquet data collector (the thing that must keep running), and
(2) a kept trading-engine skeleton (`trading/`) to be rewired for the new model.

Coins: **BTC (primary), ETH, XRP, SOL**. BTC is the traded asset; ETH/XRP/SOL Kalshi quotes are cross-asset
features. Markets are 15-minute binary "will price be above strike at close" contracts.

## Repo cleanup 2026-07-18 (what was removed, where it went)
Everything below is recoverable from git history (pre-cleanup commit; the deletion commit message marks it):
- `models/` (bayes_log_reg / log_reg notebooks, R script) — old training.
- `trading/model.py`, `trading/features.py`, `trading/save_model.py`, `trading/save_log_reg.py` — old
  58-feature model layer. ⚠️ `trading/engine.py` still imports `trading.features` / `trading.model`
  (lines ~138–140) — **engine.py is intentionally non-runnable** until the new model layer replaces those
  imports. Kept as the skeleton: async loop, order management, dry-run mode, WS handling.
- `trading/bayes_pnl_log.csv` — the 6,381-fill paper-trading log used for the audit (ground truth for the
  verdict below). In git history if the audit ever needs re-running.
- `data_gather/filter.py`, `postgres_extract.py`, `supabase_extract.py` — the Supabase-era CSV feature
  pipeline. Feature engineering will be done by replaying parquet events instead.
- `data_gather/scrapers/polymarket.py`, `deribit_vol.py` — retired REST pollers (superseded by
  `polymarket_ws.py` / `deribit_ws.py`).
- `test.py`, `test/` — old smoke/unit tests (test.py imported the removed REST pollers).
- Root `config.py` **stays** (sink factory used by every scraper); its legacy Supabase `'actual'` mode is
  dead code but harmless (lazy import).

## Environment
- **Python**: repo targets 3.13 (`.python-version`, README); actual venv `.venv` is py3.11 with
  pyarrow/websockets/aiohttp/dotenv/polars/cryptography. Machine default `python` is 3.11.9 without those.
- Run everything as a module from repo root, e.g. `python -m data_gather.scrapers.run_all` (see `to_run.txt`).
- Bash tool = Git Bash; primary shell is PowerShell. Set `PYTHONIOENCODING=utf-8` for scripts that print.
- `requirements.txt` is a UTF-16 pip freeze (includes stale model deps); `requirements-collector.txt` adds
  pyarrow. Env template: `.env.example`.

## Data collector v2 — the live workload
`DATA_SINK_MODE=parquet python -m data_gather.scrapers.run_all` runs 12 supervised child processes
(per-child restart w/ backoff) writing **local hourly parquet** to `data/{table}/date=/hour=/part-*.parquet`
(zstd, file-per-flush + atomic rename — hard-kill-safe, verified with kill -9). Key modules:
- `data_gather/parquet_sink.py` — `ParquetDataSink`, drop-in `DataSink` mode (writer thread, bounded queue,
  explicit schemas from `parquet_schemas.py` with string→number coercion, heartbeat to `data/_status/*.json`)
- `data_gather/monitor.py` — staleness/error watcher (thresholds per table; optional `ALERT_WEBHOOK_URL`).
  Equity tables (alpaca_*, vix) only checked Mon–Fri 14:30–20:00 UTC (EST/EDT session intersection).
- `data_gather/compact.py` — daily merge of closed dates into `compacted-D.parquet` (verified row counts) +
  `--retention-days`; schedule per `deploy/` (NSSM/Task Scheduler on Windows, systemd on Linux).
  Tables >5M rows/day stream through a ParquetWriter — `to_table()` on a full day OOM-killed a 16GB box;
  streamed output is ~time-sorted (flush order), small tables get an exact curr_time sort.
- `data_gather/scrapers/ws_base.py` — shared `WsScraper`/`StreamBuffer` base for all WS scrapers
- Scrapers/tables: kalshi_ws (deltas+trades+ticker_v2, 15M+hourly `KX{coin}D` series, 5-min REST re-snapshots;
  needs Kalshi creds), kalshi.py (5s market snapshots; `KALSHI_OB_OVERRIDE=1`+`KALSHI_POLL_SECONDS=1` only for
  the trading engine — rate-limit hog), kalshi_settlements (ground-truth labels incl. `expiration_value` BRTI
  print — verified), polymarket_ws (CLOB WS book/price_change/trades; book explosion throttled to 30s,
  price_change carries per-change asset_id), coinbase_ws + coinbase_l2 (BTC/ETH full depth `level2_batch`,
  500-level snapshots + 30-min resubscribe), kraken_ws, bitstamp_ws, binanceus_ws (trades sparse — thin venue,
  quotes are the signal), deribit_ws (DVOL + near-spot option tickers + perps + index), alpaca_ws (SPY/QQQ,
  needs free Alpaca keys; VIX poll runs keyless).
- Read data: `pl.scan_parquet('data/{table}/**/*.parquet')`.
- Kalshi public API rate limit is shared per-IP across all Kalshi modules — keep request gaps (429s verified
  gone at current settings). Kalshi has 15M series for DOGE/ADA/BCH/ZEC/NEAR/TON/BNB/HYPE too —
  `KALSHI_EXTRA_SERIES` env adds them (easy sample-size multiplier).

### Known data gaps (exclude from replays/vol estimates)
- **2026-07-11 05:17 → 2026-07-12 17:32 UTC (~36h, all venues)** — machine slept. Sleep on AC disabled
  since (powercfg, 2026-07-12). Settlement labels REST-backfilled; book/trade data unrecoverable.
- **2026-07-13 04:24 → 07:26 UTC (~3h)** — unexpected shutdown (event 6008); manual restart.
- **2026-07-16 20:22 → 2026-07-17 03:31 UTC (~7h)** — machine died again; manual restart.
- **2026-07-17 ~19:32 → 19:41 UTC (~10 min)** — collector process died; restarted. Negligible.
- **2026-07-18 19:44 → 2026-07-23 00:33 UTC (~4.2 days, all venues)** — collector died (user reported a
  timeout) and went unnoticed; restarted 2026-07-22. Gap exceeds the settlements 24h lookback, so labels
  for 07-18→07-21 need a manual REST backfill (as done for the 07-11 gap).
- **2026-07-23 00:33 → 2026-07-27 03:15 UTC (~4.1 days, polymarket + deribit tables only)** — cleanup
  fallout: `polymarket_ws`/`deribit_ws` imported helpers from the deleted `polymarket.py`/`deribit_vol.py`
  and crashlooped (exit-on-import, 300s supervisor backoff) from the 07-22 restart until the helpers were
  inlined into the WS modules on 2026-07-26. All other tables unaffected.
- **2026-07-27 03:32 → 05:11 UTC (~1h39m, all venues)** — collector stopped (last heartbeat 03:32 UTC;
  `collector.log` cut off mid-write, no crash trace, `collector.err.log` empty → likely killed/OS event,
  not a Python crash). Went unnoticed until restarted 05:11 UTC. Well within settlements 24h lookback, so
  labels self-heal; book/trade data in the window is lost.
- **2026-08-08 01:19 → 02:25 UTC (~1h06m, all venues)** — root collector (PID 33972, up continuously since
  the 07-27 restart) died ~01:19 UTC; noticed on a user-requested restart and relaunched 02:25 UTC as PID
  32148. Within settlements 24h lookback, so labels self-heal; book/trade data in the window is lost.
- **2026-08-30 21:20 → 2026-08-31 06:30 UTC (~9h10m, all venues)** — root collector (PID 32148, up ~23 days
  continuously since the 08-07 restart) died ~21:20 UTC 08-30 (`collector.log` last write 17:20 local, no
  crash trace, `collector.err.log` empty → likely OS/kill event). Caught by the hourly auto-check and
  relaunched 06:30 UTC 08-31 as PID 27288. Within settlements 24h lookback, so labels self-heal; book/trade
  data in the window is lost.
- Settlement labels self-heal on restart within 24h (kalshi_settlements polls with a 24h lookback);
  book/trade data in a gap is lost. Root cause of the big three: **no service install** — NSSM setup per
  `deploy/windows_setup.md` is the standing top ops priority.

## ⚠️ Why the model was scrapped (audit verdict, verified 2026-07-07)
Full analysis in git history (this file's pre-2026-07-18 version + `trading/bayes_pnl_log.csv`). Bottom line:
- The v1 logistic/Bayes-logistic model was **statistically tied with just reading the market mid** (LogLoss
  mid 0.2317 vs model 0.2354); coin-flip accuracy (0.512) on hard cases (0.35<p̂<0.65). Headline AUC 0.97 was
  inflated by already-decided markets (68% of rows had mid <0.1 or >0.9).
- Paper PnL (+$1,388) was 96% from entries ≤10¢ — a longshot/steamroller artifact unlikely to survive real
  fills.
- Follow-up experiment: even a forward-split model on all 58 features **couldn't beat the mid**; in the hard
  region (0.2<mid<0.8) the *market itself* is barely better than a coin flip (LogLoss 0.699 vs 0.693 random).
  15-min outcome prediction where the market is uncertain is near a random walk — a fundamental ceiling.

## Strategy for the rebuild (agreed 2026-07-08)
The model's job is a **fast, well-calibrated fair value**, not outcome prediction. Three real edges:
(1) speed — fair value updates ~100ms off the Coinbase tick while thin Kalshi books lag seconds;
(2) maker economics — taker fee is ceil(7·p·(1−p))¢ ≈ 2¢ at p=0.5 (~4% of a 50¢ contract) while maker pays
none — quote fair ± margin instead of always taking; (3) cross-venue divergence vs the identical Polymarket
15M markets (already scraped).

Modeling: baseline `P(YES) = Φ(distance_from_strike / (S·σ·√ttc))` with EWMA realized vol from Coinbase 1s
bars (+ Deribit ATM IV blend). ML only on the residual: include `logit(mid)` + `logit(fair_bs)` as features so
the model can only learn increments (flow imbalance, spot lead-lag, depth asymmetry, cross-venue gaps).
Evaluation: forward-in-time split by market, hard region (0.2<mid<0.8) only, always vs the mid baseline,
isotonic calibration fit on validation.

Data layer principle: **store raw events, never pre-aggregated stats** — features are derived offline by
replaying events (same code drives the backtest). Keep exchange `ts_ms` and local receive time (latency
measurement); log own orders/quotes/fills for fill-sim and adverse-selection measurement.

Sequence: 1) 24/7 collector (running) — **sample size is the binding constraint** (~384 markets/day possible).
2) event replayer + backtest harness (fills at far touch, latency, real fees) — gates everything.
3) Φ(d) fair-value model, verify calibration, measure |fair−mid| vs fees. 4) residual ML, maker-quoting,
Polymarket divergence. 5) small live capital only after fee-adjusted edge shows in the hard region.

## Trading skeleton (`trading/`) — kept for the engine buildout
- `kalshi_client.py` — async Kalshi REST client: RSA-PSS-SHA256 request signing, order placement,
  variance-proportional taker fee `ceil(7·p·(1−p))¢` (`KalshiClient.taker_fee`). Reusable as-is.
- `engine.py` — async live loop (market discovery, WS, order/position management, dry-run mode). ⚠️ imports
  the removed `trading.features`/`trading.model` — rewire during buildout, don't run until then. Its old
  `_apply_p_hat_bias_correction` approach was shown to *hurt* calibration — replace with a proper post-hoc
  calibrator (isotonic/Platt), not hand-tuned offsets.
- `risk.py` + `config.py` — position/exposure limits and engine constants (many constants reference the old
  58-feature model; prune during rewire).

## Known gotchas
- `data/` and `trading/model_artifacts.npz` are gitignored; model artifacts no longer exist (removed layer).
- `distance_from_strike` depends on `floor_strike`; if Kalshi returns strike=0 the feature becomes ~spot price
  (garbage). The old engine warned once per coin — keep that behavior in the rebuild.
- Kalshi taker fee is variance-proportional, not flat — already handled in `kalshi_client.py`.

## Ops (collector)
Runs as a detached process (`Start-Process .venv\Scripts\python.exe -m data_gather.scrapers.run_all`,
stdout/err → `collector.log` / `collector.err.log`, rotated to `*.log.1` on restart). Survives the Claude
session but NOT a reboot (NSSM pending). Health: heartbeats in `data/_status/*.json` (60s cadence) or grep
`collector.log` for `MONITOR`. Clean restart: kill root run_all pid, sweep leftover `data_gather` python
processes, rotate logs, relaunch. Daily `python -m data_gather.compact` still needs scheduling.

## Secrets / credentials (in `.env`, nothing committed)
- `KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH` — Kalshi RSA auth for `kalshi_ws.py` and the engine; the
  `.pem` is gitignored at repo root.
- `ALPACA_API_KEY/SECRET` — free-tier SPY/QQQ quotes (wired, verified).
- Optional: `ALERT_WEBHOOK_URL` (monitor alerts), `KALSHI_EXTRA_SERIES`.
- Trading real orders: run `python -m trading.engine --dry-run` first; live mode places real Kalshi orders.
