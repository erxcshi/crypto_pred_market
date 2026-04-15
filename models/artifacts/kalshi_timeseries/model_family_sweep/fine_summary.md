# Kalshi Model Family Fine Results

Selection metric: average realized executable PnL per 1-contract covered trade.

PnL method: enter at current ask, exit at future bid.

EDA-informed choices: include spot/cross-coin features, use robust linear models for heavy-tailed features,
and compare nonlinear tree models for threshold/interactions.

## Best

- horizon_seconds: 60
- experiment: hist_gbr_delta_all_iter160_lr0.03_leaf15_l20.1
- family: hist_gbr
- target_mode: delta
- feature_group: all
- params_json: {"l2_regularization": 0.1, "learning_rate": 0.03, "max_iter": 160, "max_leaf_nodes": 15}
- rmse: 0.11274976546078971
- mae: 0.07264329179920459
- directional_accuracy: 0.6778307984677616
- best_threshold_by_realized_pnl: 0.2
- best_coverage_by_realized_pnl: 0.013540601586283786
- best_hit_rate_by_realized_pnl: 0.9300341296928327
- best_avg_realized_pnl: 0.3091207337883959
- best_total_realized_pnl: 724.579
- best_yes_trade_count: 1030
- best_no_trade_count: 1314

## Top 25

| horizon_seconds | experiment | family | target_mode | feature_group | rmse | mae | directional_accuracy | best_threshold_by_realized_pnl | best_coverage_by_realized_pnl | best_hit_rate_by_realized_pnl | best_avg_realized_pnl | best_total_realized_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | hist_gbr_delta_all_iter160_lr0.03_leaf15_l20.1 | hist_gbr | delta | all | 0.112750 | 0.072643 | 0.677831 | 0.200000 | 0.013541 | 0.930034 | 0.309121 | 724.579000 |
| 60 | hist_gbr_delta_all_iter160_lr0.03_leaf31_l20.1 | hist_gbr | delta | all | 0.112028 | 0.071763 | 0.682603 | 0.200000 | 0.016874 | 0.913043 | 0.291158 | 850.472000 |
| 30 | hist_gbr_delta_all_iter160_lr0.03_leaf15_l20.1 | hist_gbr | delta | all | 0.065632 | 0.040835 | 0.753649 | 0.200000 | 0.014765 | 0.956587 | 0.288889 | 771.912000 |
| 60 | hist_gbr_delta_all_iter320_lr0.03_leaf15_l20.1 | hist_gbr | delta | all | 0.112113 | 0.071933 | 0.680738 | 0.200000 | 0.018514 | 0.914821 | 0.287760 | 922.271000 |
| 60 | hist_gbr_delta_all_iter160_lr0.06_leaf15_l20.1 | hist_gbr | delta | all | 0.112100 | 0.071936 | 0.681596 | 0.200000 | 0.017810 | 0.909179 | 0.286683 | 883.845000 |
| 30 | hist_gbr_delta_all_iter160_lr0.03_leaf31_l20.1 | hist_gbr | delta | all | 0.063820 | 0.039406 | 0.759750 | 0.200000 | 0.016704 | 0.956666 | 0.283545 | 857.157000 |
| 60 | hist_gbr_price_all_iter160_lr0.03_leaf15_l20.1 | hist_gbr | price | all | 0.110672 | 0.071457 | 0.656348 | 0.200000 | 0.020386 | 0.923774 | 0.282625 | 997.385000 |
| 60 | hist_gbr_price_all_iter320_lr0.03_leaf15_l20.1 | hist_gbr | price | all | 0.110572 | 0.070408 | 0.689382 | 0.200000 | 0.021784 | 0.925749 | 0.280427 | 1057.489000 |
| 60 | hist_gbr_price_all_iter160_lr0.03_leaf31_l20.1 | hist_gbr | price | all | 0.110789 | 0.071109 | 0.660388 | 0.200000 | 0.022113 | 0.915622 | 0.278596 | 1066.466000 |
| 30 | hist_gbr_price_all_iter160_lr0.03_leaf31_l20.1 | hist_gbr | price | all | 0.061530 | 0.037927 | 0.745849 | 0.200000 | 0.019705 | 0.961582 | 0.276744 | 986.870000 |
| 60 | hist_gbr_price_all_iter160_lr0.06_leaf15_l20.1 | hist_gbr | price | all | 0.110938 | 0.070573 | 0.688483 | 0.200000 | 0.021605 | 0.920321 | 0.276648 | 1034.662000 |
| 60 | hist_gbr_delta_all_iter320_lr0.06_leaf15_l20.1 | hist_gbr | delta | all | 0.111841 | 0.071194 | 0.687792 | 0.200000 | 0.020952 | 0.901020 | 0.275746 | 1000.131000 |
| 60 | hist_gbr_delta_all_iter320_lr0.03_leaf31_l20.1 | hist_gbr | delta | all | 0.111585 | 0.071124 | 0.687100 | 0.200000 | 0.021177 | 0.899618 | 0.275730 | 1010.825000 |
| 120 | hist_gbr_delta_all_iter160_lr0.03_leaf15_l20.1 | hist_gbr | delta | all | 0.171230 | 0.116741 | 0.631516 | 0.200000 | 0.012278 | 0.787801 | 0.274848 | 536.229000 |
| 60 | hist_gbr_delta_all_iter160_lr0.06_leaf31_l20.1 | hist_gbr | delta | all | 0.111743 | 0.071162 | 0.688965 | 0.200000 | 0.021663 | 0.900267 | 0.274467 | 1029.251000 |
| 30 | hist_gbr_delta_all_iter320_lr0.06_leaf31_l20.1 | hist_gbr | delta | all | 0.060899 | 0.036754 | 0.777989 | 0.200000 | 0.020815 | 0.966021 | 0.273349 | 1029.704000 |
| 30 | hist_gbr_delta_all_iter320_lr0.03_leaf15_l20.1 | hist_gbr | delta | all | 0.063552 | 0.039222 | 0.763660 | 0.200000 | 0.018605 | 0.952777 | 0.272950 | 919.023000 |
| 30 | hist_gbr_price_all_iter160_lr0.03_leaf15_l20.1 | hist_gbr | price | all | 0.062767 | 0.039121 | 0.741233 | 0.200000 | 0.018854 | 0.959848 | 0.272927 | 931.226000 |
| 30 | hist_gbr_delta_all_iter320_lr0.06_leaf15_l20.1 | hist_gbr | delta | all | 0.061817 | 0.037720 | 0.771721 | 0.200000 | 0.019981 | 0.959624 | 0.272820 | 986.518000 |
| 30 | hist_gbr_price_all_iter320_lr0.06_leaf31_l20.1 | hist_gbr | price | all | 0.060346 | 0.036389 | 0.780229 | 0.200000 | 0.021119 | 0.964940 | 0.272401 | 1041.117000 |
| 60 | hist_gbr_delta_all_iter320_lr0.06_leaf31_l20.1 | hist_gbr | delta | all | 0.111635 | 0.070705 | 0.692540 | 0.200000 | 0.022477 | 0.904395 | 0.272115 | 1058.801000 |
| 60 | hist_gbr_price_all_iter320_lr0.03_leaf31_l20.1 | hist_gbr | price | all | 0.110628 | 0.070130 | 0.691324 | 0.200000 | 0.023540 | 0.911902 | 0.271937 | 1108.143000 |
| 30 | hist_gbr_price_all_iter320_lr0.06_leaf15_l20.1 | hist_gbr | price | all | 0.061267 | 0.037097 | 0.777044 | 0.200000 | 0.020102 | 0.958494 | 0.271843 | 988.966000 |
| 30 | hist_gbr_delta_all_iter320_lr0.03_leaf31_l20.1 | hist_gbr | delta | all | 0.061868 | 0.037811 | 0.770771 | 0.200000 | 0.019826 | 0.954571 | 0.271460 | 973.999000 |
| 30 | hist_gbr_price_all_iter320_lr0.03_leaf15_l20.1 | hist_gbr | price | all | 0.061601 | 0.037579 | 0.773034 | 0.200000 | 0.019578 | 0.959639 | 0.271147 | 960.674000 |