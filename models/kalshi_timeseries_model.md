# BTC Kalshi Time-Series Model

This model answers:

```text
What will the BTC Kalshi YES contract cost n seconds from now?
```

It uses `data_gather/final_data/final_data.csv` and creates a regression target.
If that file is not present, it falls back to the older
`final_data/final_data.csv` location.

```text
yes_mid_dollars_t_plus_{n}s
```

That target is built by shifting `yes_mid_dollars` forward inside each Kalshi
event, grouped by `open_time` and `close_time`.

## Why This Model

The data is second-by-second market microstructure data with many tabular
signals from Kalshi, Coinbase spot trades, and cross-coin context. The script
therefore uses an autoregressive-with-exogenous-features setup: current and
past market values predict the future contract price.

If `scikit-learn` is installed, `--model-type auto` uses
`HistGradientBoostingRegressor`, which is a good fit for nonlinear tabular
time-series features. If sklearn is not installed, it falls back to a
dependency-free NumPy ridge ARX model.

## Leakage Handling

The script excludes future/outcome columns from features, including:

```text
outcome
next_price_dollars_lead1
any column starting with next_
any column containing lead
any generated t_plus target column
```

It also splits train/test chronologically by Kalshi event, rather than randomly
mixing rows from the same 15-minute market.

## Run

From the repo root:

```bash
source .venv/bin/activate
python -m models.kalshi_timeseries_model --horizon-seconds 30
```

Try other horizons:

```bash
python -m models.kalshi_timeseries_model --horizon-seconds 10
python -m models.kalshi_timeseries_model --horizon-seconds 60
python -m models.kalshi_timeseries_model --horizon-seconds 300
```

Outputs are written to:

```text
models/artifacts/kalshi_timeseries/
```

Each run saves:

```text
model_{n}s.pkl
metrics_{n}s.json
predictions_{n}s.csv
```
