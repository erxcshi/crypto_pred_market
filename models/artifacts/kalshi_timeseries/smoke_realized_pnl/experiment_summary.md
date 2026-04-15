# Kalshi Time-Series Experiment Summary

Selection objective: maximize average realized executable PnL per 1-contract trade,
requiring at least the configured minimum test-set trade coverage.

PnL enters at the current ask and exits at the future bid:
long YES = future YES bid - current YES ask; long NO = future NO bid - current NO ask.

## Best Overall

- Horizon: 60 seconds
- Experiment: ridge_price_all_alpha_10
- Feature group: all
- Target mode: price
- Alpha: 10.0
- RMSE: 0.120340
- MAE: 0.079581
- Directional accuracy: 0.6498
- Best threshold: 0.1
- Coverage at best threshold: 0.0434
- Hit rate at best threshold: 0.8059
- YES trades at best threshold: 3316
- NO trades at best threshold: 4194
- Avg realized PnL at best threshold: 0.131802
- Total realized PnL at best threshold: 989.833000
- Saved model: `models\artifacts\kalshi_timeseries\smoke_realized_pnl\best_model.pkl`

## Top 3 By Horizon

| horizon_seconds | experiment | rmse | mae | directional_accuracy | best_threshold_by_realized_pnl | best_coverage_by_realized_pnl | best_avg_realized_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | ridge_price_all_alpha_10 | 0.120340 | 0.079581 | 0.649783 | 0.100000 | 0.043383 | 0.131802 |
| 60 | ridge_delta_all_alpha_1000 | 0.120376 | 0.079587 | 0.649944 | 0.100000 | 0.042707 | 0.131776 |
| 60 | ridge_delta_all_alpha_10 | 0.120341 | 0.079581 | 0.649789 | 0.100000 | 0.043377 | 0.131758 |

## Notes

- `persistence` means predicting the current YES midpoint as the future YES midpoint.
- `trend_5s` extrapolates the last 5 seconds of YES midpoint movement.
- `ridge_price_*` directly predicts the future price.
- `ridge_delta_*` predicts the future price change and adds it back to the current price.
- Trading PnL uses executable ask/bid prices, while RMSE/MAE still evaluate future YES midpoint forecasts.