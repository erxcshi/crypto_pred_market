# Kalshi Model Family Coarse Results

Selection metric: average realized executable PnL per 1-contract covered trade.

PnL method: enter at current ask, exit at future bid.

EDA-informed choices: include spot/cross-coin features, use robust linear models for heavy-tailed features,
and compare nonlinear tree models for threshold/interactions.

## Best

- horizon_seconds: 60
- experiment: hist_gbr_delta_all
- family: hist_gbr
- target_mode: delta
- feature_group: all
- params_json: {"l2_regularization": 0.1, "learning_rate": 0.05, "max_iter": 160, "max_leaf_nodes": 31}
- rmse: 0.11165452969177614
- mae: 0.07116228399448454
- directional_accuracy: 0.6868026140987377
- best_threshold_by_realized_pnl: 0.2
- best_coverage_by_realized_pnl: 0.019964299949742647
- best_hit_rate_by_realized_pnl: 0.9027777777777778
- best_avg_realized_pnl: 0.27978616898148145
- best_total_realized_pnl: 966.9409999999999
- best_yes_trade_count: 1598
- best_no_trade_count: 1858

## Top 25

| horizon_seconds | experiment | family | target_mode | feature_group | rmse | mae | directional_accuracy | best_threshold_by_realized_pnl | best_coverage_by_realized_pnl | best_hit_rate_by_realized_pnl | best_avg_realized_pnl | best_total_realized_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | hist_gbr_delta_all | hist_gbr | delta | all | 0.111655 | 0.071162 | 0.686803 | 0.200000 | 0.019964 | 0.902778 | 0.279786 | 966.941000 |
| 60 | extra_trees_delta_all | extra_trees | delta | all | 0.116334 | 0.074225 | 0.677831 | 0.150000 | 0.012420 | 0.906512 | 0.274009 | 589.119000 |
| 30 | hist_gbr_delta_all | hist_gbr | delta | all | 0.062452 | 0.038278 | 0.767254 | 0.200000 | 0.019478 | 0.953191 | 0.273975 | 965.763000 |
| 30 | hist_gbr_price_all | hist_gbr | price | all | 0.061177 | 0.037005 | 0.773700 | 0.200000 | 0.020887 | 0.959524 | 0.272251 | 1029.108000 |
| 30 | random_forest_delta_all | random_forest | delta | all | 0.065079 | 0.039166 | 0.772601 | 0.200000 | 0.016218 | 0.942419 | 0.272003 | 798.330000 |
| 60 | hist_gbr_price_all | hist_gbr | price | all | 0.111075 | 0.070393 | 0.689126 | 0.200000 | 0.023419 | 0.910705 | 0.271984 | 1102.625000 |
| 60 | random_forest_delta_all | random_forest | delta | all | 0.112426 | 0.071090 | 0.701952 | 0.200000 | 0.017579 | 0.896155 | 0.271919 | 827.448000 |
| 30 | extra_trees_delta_all | extra_trees | delta | all | 0.071528 | 0.043258 | 0.743847 | 0.150000 | 0.010134 | 0.965649 | 0.267929 | 491.381000 |
| 120 | extra_trees_delta_all | extra_trees | delta | all | 0.172269 | 0.117829 | 0.632045 | 0.150000 | 0.013556 | 0.791086 | 0.250765 | 540.147000 |
| 120 | hist_gbr_delta_all | hist_gbr | delta | all | 0.171162 | 0.116685 | 0.627021 | 0.200000 | 0.018830 | 0.754345 | 0.235604 | 704.928000 |
| 30 | mlp_delta_all | mlp | delta | all | 0.064709 | 0.038859 | 0.730575 | 0.200000 | 0.025286 | 0.928322 | 0.232549 | 1064.145000 |
| 120 | random_forest_delta_all | random_forest | delta | all | 0.171790 | 0.116577 | 0.640594 | 0.200000 | 0.017307 | 0.766182 | 0.232197 | 638.543000 |
| 120 | hist_gbr_price_all | hist_gbr | price | all | 0.171124 | 0.115920 | 0.620289 | 0.200000 | 0.024947 | 0.766398 | 0.223253 | 884.973000 |
| 10 | mlp_delta_all | mlp | delta | all | 0.038139 | 0.016879 | 0.804683 | 0.150000 | 0.014302 | 0.786116 | 0.182406 | 486.113000 |
| 300 | ridge_delta_cross_alpha_1000 | ridge | delta | kalshi_cross_coin | 0.275552 | 0.216284 | 0.541070 | 0.100000 | 0.012342 | 0.602343 | 0.181463 | 263.303000 |
| 300 | hist_gbr_delta_all | hist_gbr | delta | all | 0.276692 | 0.213933 | 0.568908 | 0.200000 | 0.011339 | 0.512378 | 0.171656 | 228.817000 |
| 300 | hist_gbr_price_all | hist_gbr | price | all | 0.277238 | 0.214171 | 0.560275 | 0.200000 | 0.016817 | 0.601416 | 0.167493 | 331.133000 |
| 300 | random_forest_delta_all | random_forest | delta | all | 0.276006 | 0.212139 | 0.588523 | 0.150000 | 0.025246 | 0.616914 | 0.161684 | 479.877000 |
| 10 | hist_gbr_delta_all | hist_gbr | delta | all | 0.041727 | 0.021968 | 0.805915 | 0.100000 | 0.016325 | 0.634122 | 0.159467 | 485.098000 |
| 60 | mlp_delta_all | mlp | delta | all | 0.122353 | 0.076385 | 0.671659 | 0.200000 | 0.035891 | 0.747626 | 0.155524 | 966.270000 |
| 10 | random_forest_delta_all | random_forest | delta | all | 0.044026 | 0.022703 | 0.813778 | 0.100000 | 0.011758 | 0.586490 | 0.148690 | 325.780000 |
| 10 | hist_gbr_price_all | hist_gbr | price | all | 0.041418 | 0.022340 | 0.817267 | 0.100000 | 0.018917 | 0.603972 | 0.147945 | 521.507000 |
| 300 | ridge_price_all_alpha_10 | ridge | price | all | 0.273794 | 0.214766 | 0.552282 | 0.150000 | 0.012385 | 0.643544 | 0.142392 | 207.323000 |
| 300 | ridge_delta_all_alpha_10 | ridge | delta | all | 0.273794 | 0.214765 | 0.552291 | 0.150000 | 0.012410 | 0.642906 | 0.142093 | 207.313000 |
| 300 | elastic_net_delta_all | elastic_net | delta | all | 0.274009 | 0.214822 | 0.550514 | 0.150000 | 0.010088 | 0.649241 | 0.133535 | 158.372000 |