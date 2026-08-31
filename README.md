# crypto_pred_market

Data collection and trading system for crypto **15-minute** prediction markets (Kalshi `KX{coin}15M`
up/down binaries).

The system runs a 24/7 collector that captures raw spot + prediction-market events (Kalshi, Polymarket,
Coinbase, Kraken, Bitstamp, Binance.US, Deribit, Alpaca) into local hourly parquet. A fair-value model and
trading engine are being rebuilt from scratch on top of this data — the v1 model was removed after an audit
showed no edge over the market price (see `CLAUDE.md`).

Coins: **BTC** (traded) with **ETH/XRP/SOL** as cross-asset features.

## Setup

Use Python `3.13`.

```bash
python3.13 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-collector.txt
```

Copy `.env.example` to `.env` and fill in what your task needs (Kalshi API keys for kalshi_ws / trading,
optional Alpaca keys). None are committed.

## Running

Run everything as a module from the repo root (see `to_run.txt` for the full list):

```bash
# the collector — writes local parquet to data/{table}/date=/hour=/
DATA_SINK_MODE=parquet python -m data_gather.scrapers.run_all

# daily compaction of closed dates
python -m data_gather.compact
```

Read collected data with `pl.scan_parquet('data/<table>/**/*.parquet')`.

The trading engine skeleton in `trading/` is not currently runnable — it awaits the new model layer.

See `CLAUDE.md` for architecture, ops procedures, known data gaps, and the rebuild strategy.
