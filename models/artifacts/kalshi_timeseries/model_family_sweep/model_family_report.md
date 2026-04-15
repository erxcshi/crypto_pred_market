# Kalshi Time-Series Model Family Report

## EDA Inputs Used

The sweep was informed by `eda.ipynb`:

```text
rows: 1,017,826
windows: 1,184 15-minute markets
missing values: only next_price_dollars_lead1, 0.12%
delta_p std: 2.15 cents
delta_p kurtosis: 227.86, meaning heavy tails
top linear signals: spread, spread mean, BTC spot returns, cross-coin mid changes
cross-coin BTC/ETH/XRP/SOL levels and distances are highly correlated
spot volatility alone has weak correlation with |delta_p|
```

That led to testing:

```text
robust linear models for heavy-tailed features
tree models for nonlinear threshold/interactions
all-feature models because spot and cross-coin features both showed signal
30s, 60s, 120s, and 300s horizons because prior ridge runs were strongest there
```

## PnL Method

All model families were evaluated with realized executable PnL:

```text
long YES = future YES bid - current YES ask
long NO = future NO bid - current NO ask
```

Current `final_data.csv` does not contain raw bid/ask columns, so executable
prices are reconstructed from midpoint and spread.

## Coarse Sweep

Families tested:

```text
ridge
elastic_net
sgd_huber
hist_gbr
extra_trees
random_forest
mlp
```

The coarse winner was:

```text
family: hist_gbr
experiment: hist_gbr_delta_all
horizon: 60s
target mode: delta
feature group: all
params: max_iter=160, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1
threshold: 0.20
coverage: 2.00%
hit rate: 90.28%
avg realized PnL: 27.98 cents
total realized PnL: 966.941
RMSE: 0.11165
MAE: 0.07116
directional accuracy: 68.68%
```

Saved coarse data:

```text
models/artifacts/kalshi_timeseries/model_family_sweep/coarse_results.csv
models/artifacts/kalshi_timeseries/model_family_sweep/coarse_best_metadata.json
models/artifacts/kalshi_timeseries/model_family_sweep/coarse_best_model.pkl
models/artifacts/kalshi_timeseries/model_family_sweep/coarse_summary.md
```

## Fine Tune

The fine tune focused on histogram gradient boosting with all features:

```text
target modes: delta, price
horizons: 30s, 60s, 120s
max_iter: 160, 320
learning_rate: 0.03, 0.06
max_leaf_nodes: 15, 31
l2_regularization: 0.1
```

The best average-per-trade model was:

```text
family: hist_gbr
experiment: hist_gbr_delta_all_iter160_lr0.03_leaf15_l20.1
horizon: 60s
target mode: delta
feature group: all
threshold: 0.20
coverage: 1.35%
hit rate: 93.00%
YES trades: 1030
NO trades: 1314
avg realized PnL: 30.91 cents
total realized PnL: 724.579
RMSE: 0.11275
MAE: 0.07264
directional accuracy: 67.78%
```

Best total-PnL model in the fine grid:

```text
experiment: hist_gbr_price_all_iter320_lr0.03_leaf31_l20.1
horizon: 60s
threshold: 0.20
coverage: 2.35%
hit rate: 91.19%
avg realized PnL: 27.19 cents
total realized PnL: 1108.143
RMSE: 0.11063
MAE: 0.07013
directional accuracy: 69.13%
```

Saved fine-tune data:

```text
models/artifacts/kalshi_timeseries/model_family_sweep/fine_results.csv
models/artifacts/kalshi_timeseries/model_family_sweep/fine_best_metadata.json
models/artifacts/kalshi_timeseries/model_family_sweep/fine_best_model.pkl
models/artifacts/kalshi_timeseries/model_family_sweep/fine_summary.md
```

## Conclusion

Ridge is no longer the best tested model family. In this broader sweep,
histogram gradient boosting was the strongest model type under realized
ask-to-bid PnL. It beat ridge by a large margin on the same executable PnL
metric.

The recommended next candidate depends on objective:

```text
maximize average edge per covered trade:
  hist_gbr_delta_all_iter160_lr0.03_leaf15_l20.1

maximize total realized PnL with still-high per-trade edge:
  hist_gbr_price_all_iter320_lr0.03_leaf31_l20.1
```

Before live use, the remaining big issue is still signal de-duplication:
second-by-second rows can count many overlapping positions inside the same
market move.
