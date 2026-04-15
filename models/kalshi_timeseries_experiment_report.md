# Kalshi Time-Series Experiment Report

## Goal

Forecast the BTC Kalshi YES contract midpoint price `n` seconds in the future,
then evaluate whether the forecast creates useful trading signals.

The modeled price is:

```text
yes_mid_dollars = (yes_ask_dollars + yes_bid_dollars) / 2
```

The future target is built as:

```text
yes_mid_dollars_t_plus_{n}s
```

using a time-based as-of lookup inside the same Kalshi event.

## Experiments Run

Horizons:

```text
1s, 2s, 5s, 10s, 30s, 60s, 120s, 300s
```

Model/input variants:

```text
persistence baseline
5-second trend extrapolation
ridge price prediction
ridge delta prediction
```

Feature groups:

```text
kalshi_core: BTC Kalshi features only
kalshi_spot: BTC Kalshi + BTC spot features
kalshi_cross_coin: BTC Kalshi + ETH/XRP/SOL Kalshi features
all: BTC Kalshi + BTC spot + cross-coin Kalshi features
```

Ridge regularization:

```text
alpha = 0.1, 10, 1000
```

Trading thresholds:

```text
0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2
```

## Metrics

Forecast metrics:

```text
RMSE
MAE
R2
directional accuracy
predicted-vs-actual delta correlation
```

Trading metrics:

```text
coverage: percent of rows where abs(predicted move) exceeds threshold
hit rate: percent of trades with positive gross PnL
average gross PnL per 1-contract trade
average PnL after approximate current-spread round-trip cost
```

The spread-adjusted metric is approximate because the future executable spread
is not in the target.

## Main Findings

Very short horizons are not attractive after spread costs:

```text
1s and 2s had high directional accuracy in some variants, but expected
spread-adjusted trading PnL was negative or near zero.
```

BTC spot features matter:

```text
kalshi_spot and all-feature models consistently beat kalshi_core on 5s-120s
horizons. Coinbase/BTC spot flow and returns add meaningful short-horizon signal.
```

Cross-coin Kalshi features are useful mainly at longer horizons:

```text
At 300s, kalshi_cross_coin produced the highest sparse-signal PnL.
For 30s-120s, all features or kalshi_spot were generally stronger.
```

Price-vs-delta targets were very similar:

```text
ridge_price and ridge_delta produced nearly identical performance for most
horizons. The final balanced model uses price prediction for simplicity.
```

Thresholding matters more than tiny model differences:

```text
The best trading results came from only trading larger predicted moves.
For 30s-120s, 10 cents was a better action threshold than 5 cents.
```

## Best Raw Edge Model

The highest average spread-adjusted PnL with at least 1 percent coverage was:

```text
horizon: 300s
model: ridge_delta_kalshi_cross_coin_alpha_1000
threshold: 0.10
coverage: 1.23%
hit rate: 61.34%
avg gross PnL: 19.47 cents
avg PnL after current spread: 17.99 cents
```

Saved at:

```text
models/artifacts/kalshi_timeseries/experiments_extended_thresholds/best_model.pkl
```

This is attractive on per-trade edge, but sparse. It may be more exposed to
selection noise and regime risk.

## Final Recommended Model

For a better tradeoff between edge, coverage, and stability, use:

```text
horizon: 60s
model: RidgeARXRegressor
target: future YES midpoint price
feature group: all
alpha: 10
trade threshold: 0.10
```

Test-set performance:

```text
RMSE: 0.12034
MAE: 0.07958
directional accuracy: 64.98%
coverage at 10c threshold: 4.34%
hit rate at 10c threshold: 83.69%
avg gross PnL at 10c threshold: 14.46 cents
avg PnL after current-spread cost: 13.19 cents
```

Saved at:

```text
models/artifacts/kalshi_timeseries/final_recommended/model_60s_balanced.pkl
models/artifacts/kalshi_timeseries/final_recommended/metadata_60s_balanced.json
models/artifacts/kalshi_timeseries/final_recommended/predictions_60s_balanced.csv
```

## Caveats

These are offline backtest-style metrics, not live-trading guarantees.

Important missing realism:

```text
actual executable bid/ask at entry and exit
order fill probability
market impact
fees
latency
position limits
overlap between repeated second-by-second signals
regime drift across trading days
```

Before live use, evaluate on a newer holdout day and simulate trade de-duplication
so the same market move is not counted hundreds of times.

