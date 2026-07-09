# crypto_pred_market

Probability calibration and trading system for crypto **15-minute** prediction markets.

The system scrapes crypto spot and prediction-market data, builds a per-second feature table, trains a
logistic / Bayesian-logistic model to estimate P(YES) for Kalshi `KX{coin}15M` up/down markets, and runs a
live async engine that trades when the model's probability diverges from the market price.

Coins: **BTC** (traded) with **ETH/XRP/SOL** as cross-asset features.

## Setup

Use Python `3.13`.

```bash
python3.13 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` in the repo root with the credentials your task needs (Supabase, Postgres, Kalshi API). None
are committed.

## Running

Run everything as a module from the repo root:

```bash
# scrape spot + prediction-market data into Supabase (or CSV in test mode)
python -m data_gather.scrapers.run_all

# pull collected tables from Supabase to data_gather/raw_data/*.csv
python -m data_gather.postgres_extract        # or: python -m data_gather.supabase_extract

# build the feature table
python -m data_gather.filter

# paper-trade against Kalshi (no real orders); drop --dry-run to trade live
python -m trading.engine --dry-run
```

Model training lives in `models/*.ipynb`; artifacts are written to `trading/model_artifacts.npz`.

See `CLAUDE.md` for architecture, the data flow, and a verified assessment of model performance.
