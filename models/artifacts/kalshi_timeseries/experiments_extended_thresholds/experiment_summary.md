# Kalshi Time-Series Experiment Summary

Selection objective: maximize average realized executable PnL per 1-contract trade,
requiring at least the configured minimum test-set trade coverage.

PnL enters at the current ask and exits at the future bid:
long YES = future YES bid - current YES ask; long NO = future NO bid - current NO ask.

## Best Overall

- Horizon: 300 seconds
- Experiment: ridge_delta_kalshi_cross_coin_alpha_1000
- Feature group: kalshi_cross_coin
- Target mode: delta
- Alpha: 1000.0
- RMSE: 0.275552
- MAE: 0.216284
- Directional accuracy: 0.5411
- Best threshold: 0.1
- Coverage at best threshold: 0.0123
- Hit rate at best threshold: 0.6023
- YES trades at best threshold: 401
- NO trades at best threshold: 1050
- Avg realized PnL at best threshold: 0.181463
- Total realized PnL at best threshold: 263.303000
- Saved model: `models\artifacts\kalshi_timeseries\experiments_extended_thresholds\best_model.pkl`

## Top 3 By Horizon

| horizon_seconds | experiment | rmse | mae | directional_accuracy | best_threshold_by_realized_pnl | best_coverage_by_realized_pnl | best_avg_realized_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | persistence | 0.016658 | 0.001898 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | ridge_delta_kalshi_spot_alpha_0.1 | 0.016749 | 0.003043 | 0.763285 | 0.005000 | 0.019840 | -0.007529 |
| 1 | ridge_price_kalshi_spot_alpha_0.1 | 0.016749 | 0.003043 | 0.763285 | 0.005000 | 0.019840 | -0.007529 |
| 2 | persistence | 0.023571 | 0.003798 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | ridge_price_kalshi_spot_alpha_0.1 | 0.023542 | 0.005825 | 0.763485 | 0.010000 | 0.016301 | -0.001636 |
| 2 | ridge_delta_kalshi_spot_alpha_0.1 | 0.023542 | 0.005825 | 0.763485 | 0.010000 | 0.016301 | -0.001636 |
| 5 | ridge_delta_kalshi_spot_alpha_1000 | 0.036509 | 0.013486 | 0.758974 | 0.020000 | 0.027275 | 0.012737 |
| 5 | ridge_price_kalshi_spot_alpha_10 | 0.036506 | 0.013508 | 0.761120 | 0.020000 | 0.027834 | 0.012601 |
| 5 | ridge_delta_kalshi_spot_alpha_0.1 | 0.036506 | 0.013509 | 0.760670 | 0.020000 | 0.027877 | 0.012571 |
| 10 | ridge_delta_all_alpha_1000 | 0.049409 | 0.024739 | 0.760197 | 0.050000 | 0.018934 | 0.052772 |
| 10 | ridge_delta_all_alpha_10 | 0.049404 | 0.024789 | 0.760249 | 0.050000 | 0.019341 | 0.052451 |
| 10 | ridge_delta_all_alpha_0.1 | 0.049404 | 0.024789 | 0.760301 | 0.050000 | 0.019358 | 0.052403 |
| 30 | ridge_price_all_alpha_10 | 0.078696 | 0.049915 | 0.715426 | 0.100000 | 0.034414 | 0.124320 |
| 30 | ridge_delta_all_alpha_10 | 0.078695 | 0.049915 | 0.715461 | 0.100000 | 0.034408 | 0.124203 |
| 30 | ridge_price_all_alpha_0.1 | 0.078695 | 0.049915 | 0.715473 | 0.100000 | 0.034408 | 0.124177 |
| 60 | ridge_price_all_alpha_10 | 0.120340 | 0.079581 | 0.649783 | 0.100000 | 0.043383 | 0.131802 |
| 60 | ridge_delta_all_alpha_1000 | 0.120376 | 0.079587 | 0.649944 | 0.100000 | 0.042707 | 0.131776 |
| 60 | ridge_delta_all_alpha_10 | 0.120341 | 0.079581 | 0.649789 | 0.100000 | 0.043377 | 0.131758 |
| 120 | ridge_price_all_alpha_1000 | 0.173927 | 0.121943 | 0.590841 | 0.100000 | 0.046376 | 0.128103 |
| 120 | ridge_delta_all_alpha_1000 | 0.173947 | 0.121910 | 0.591084 | 0.100000 | 0.045816 | 0.128038 |
| 120 | ridge_delta_all_alpha_10 | 0.173919 | 0.121923 | 0.590924 | 0.100000 | 0.046590 | 0.127880 |
| 300 | ridge_delta_kalshi_cross_coin_alpha_1000 | 0.275552 | 0.216284 | 0.541070 | 0.100000 | 0.012342 | 0.181463 |
| 300 | ridge_price_kalshi_cross_coin_alpha_1000 | 0.275531 | 0.216317 | 0.539670 | 0.100000 | 0.012606 | 0.178868 |
| 300 | ridge_price_kalshi_cross_coin_alpha_10 | 0.275492 | 0.216275 | 0.542505 | 0.100000 | 0.013669 | 0.164803 |

## Notes

- `persistence` means predicting the current YES midpoint as the future YES midpoint.
- `trend_5s` extrapolates the last 5 seconds of YES midpoint movement.
- `ridge_price_*` directly predicts the future price.
- `ridge_delta_*` predicts the future price change and adds it back to the current price.
- Trading PnL uses executable ask/bid prices, while RMSE/MAE still evaluate future YES midpoint forecasts.