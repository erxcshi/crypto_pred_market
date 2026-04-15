# Kalshi Model Family Coarse Results

Selection metric: average realized executable PnL per 1-contract covered trade.

PnL method: enter at current ask, exit at future bid.

EDA-informed choices: include spot/cross-coin features, use robust linear models for heavy-tailed features,
and compare nonlinear tree models for threshold/interactions.

## Best

- horizon_seconds: 60
- experiment: extra_trees_delta_all
- family: extra_trees
- target_mode: delta
- feature_group: all
- params_json: {"max_depth": 12, "min_samples_leaf": 50, "n_estimators": 80}
- rmse: 0.11633416850578254
- mae: 0.07422458957409565
- directional_accuracy: 0.6778307984677616
- best_threshold_by_realized_pnl: 0.1
- best_coverage_by_realized_pnl: 0.060984697502729494
- best_hit_rate_by_realized_pnl: 0.8356540683906413
- best_avg_realized_pnl: 0.1699463862839822
- best_total_realized_pnl: 1794.124
- best_yes_trade_count: 4544
- best_no_trade_count: 6013

## Top 25

| horizon_seconds | experiment | family | target_mode | feature_group | rmse | mae | directional_accuracy | best_threshold_by_realized_pnl | best_coverage_by_realized_pnl | best_hit_rate_by_realized_pnl | best_avg_realized_pnl | best_total_realized_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | extra_trees_delta_all | extra_trees | delta | all | 0.116334 | 0.074225 | 0.677831 | 0.100000 | 0.060985 | 0.835654 | 0.169946 | 1794.124000 |
| 60 | hist_gbr_delta_all | hist_gbr | delta | all | 0.111655 | 0.071162 | 0.686803 | 0.100000 | 0.102918 | 0.831051 | 0.156507 | 2788.334000 |
| 60 | random_forest_delta_all | random_forest | delta | all | 0.112426 | 0.071090 | 0.701952 | 0.100000 | 0.102664 | 0.823711 | 0.154052 | 2737.814000 |
| 60 | hist_gbr_price_all | hist_gbr | price | all | 0.111075 | 0.070393 | 0.689126 | 0.100000 | 0.116366 | 0.824563 | 0.148767 | 2996.769000 |
| 60 | ridge_price_all_alpha_10 | ridge | price | all | 0.120340 | 0.079581 | 0.649783 | 0.100000 | 0.043383 | 0.805859 | 0.131802 | 989.833000 |
| 60 | ridge_delta_all_alpha_10 | ridge | delta | all | 0.120341 | 0.079581 | 0.649789 | 0.100000 | 0.043377 | 0.805700 | 0.131758 | 989.373000 |
| 60 | sgd_huber_delta_all | sgd_huber | delta | all | 0.123539 | 0.078934 | 0.684032 | 0.100000 | 0.011137 | 0.830913 | 0.131520 | 253.570000 |
| 60 | elastic_net_delta_all | elastic_net | delta | all | 0.120688 | 0.079671 | 0.647913 | 0.100000 | 0.040275 | 0.805651 | 0.129147 | 900.412000 |
| 60 | mlp_delta_all | mlp | delta | all | 0.122353 | 0.076385 | 0.671659 | 0.100000 | 0.155815 | 0.747266 | 0.103063 | 2779.917000 |
| 60 | ridge_delta_cross_alpha_1000 | ridge | delta | kalshi_cross_coin | 0.128036 | 0.084315 | 0.586135 | 0.050000 | 0.047369 | 0.645854 | 0.070472 | 577.870000 |