# data set up and globs
DATA_PATH <- file.path("final_data", "final_data.csv")
ARTIFACT_DIR <- file.path("models", "artifacts", "timeseries")
dir.create(ARTIFACT_DIR, recursive = TRUE, showWarnings = FALSE)

TIME_COLUMN <- "curr_time"
OPEN_COLUMN <- "open_time"
CLOSE_COLUMN <- "close_time"
PRICE_COLUMN <- "yes_mid_dollars"
SPREAD_COLUMN <- "yes_spread_dollars"

# two-minute bars (1lag - 2m)
RESAMPLE_SECONDS <- 120
EVENT_SECONDS <- 15 * 60
MAX_EVENT_LAG <- floor(EVENT_SECONDS / RESAMPLE_SECONDS)

# model iteration params
TRAIN_POINTS <- 480
VALID_POINTS <- 180
TEST_POINTS <- 240
MAX_FIT_POINTS <- 480

VALID_REFIT_EVERY <- 10
TEST_REFIT_EVERY <- 1

THRESHOLDS <- c(0.00, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20)
MIN_COVERAGE <- 0.01


# util
parse_time <- function(x) {
  # dates come in as strings so just forcing them into utc
  as.POSIXct(x, format = "%Y-%m-%dT%H:%M:%OS%z", tz = "UTC")
}

clip01 <- function(x) {
  # prices are probabilities so keeping them in bounds
  pmin(pmax(x, 0), 1)
}

lag_table <- function(x, kind = c("acf", "pacf"), max_lag = MAX_EVENT_LAG) {
  # helper so we can use the actual acf/pacf values, not just the plots
  kind <- match.arg(kind)

  if (kind == "acf") {
    obj <- acf(x, lag.max = max_lag, plot = FALSE, na.action = na.pass)
    data.frame(lag = as.numeric(obj$lag[-1]), value = as.numeric(obj$acf[-1]))
  } else {
    obj <- pacf(x, lag.max = max_lag, plot = FALSE, na.action = na.pass)
    data.frame(lag = as.numeric(obj$lag), value = as.numeric(obj$acf))
  }
}

largest_visible_lags <- function(tab, n = 3, n_obs) {
  # quick rule for which spikes are big enough to care about
  cutoff <- 1.96 / sqrt(n_obs)
  visible <- tab[tab$lag <= MAX_EVENT_LAG & abs(tab$value) > cutoff, ]
  visible <- visible[order(-abs(visible$value)), ]
  unique(head(visible$lag, n))
}

model_row <- function(family, label, p, d, q, P = 0, D = 0, Q = 0, period = NA) {
  data.frame(
    family = family,
    model = label,
    p = p,
    d = d,
    q = q,
    P = P,
    D = D,
    Q = Q,
    period = period
  )
}


# loading data
load_market_series <- function(path) {
  # only grabbing the columns needed for this classical time series section
  raw <- read.csv(path)

  keep <- raw[, c(TIME_COLUMN, OPEN_COLUMN, CLOSE_COLUMN, PRICE_COLUMN, SPREAD_COLUMN)]
  names(keep) <- c("time", "open_time", "close_time", "yes_mid", "yes_spread")

  keep$time <- parse_time(keep$time)
  keep$open_time<- parse_time(keep$open_time)
  keep$close_time <- parse_time(keep$close_time)
  keep$yes_mid <- as.numeric(keep$yes_mid)
  keep$yes_spread <- as.numeric(keep$yes_spread)

  keep <- keep[complete.cases(keep[, c("time", "open_time", "close_time", "yes_mid")]), ]
  keep <- keep[order(keep$time), ]

  # putting uneven updates into simple 2 minute buckets
  keep$time_bin <- as.POSIXct (
    floor(as.numeric(keep$time) / RESAMPLE_SECONDS) * RESAMPLE_SECONDS,
    origin = "1970-01-01",
    tz = "UTC"
  )

  bars <- aggregate(
    keep[, c("yes_mid", "yes_spread")],
    by = list(time = keep$time_bin, open_time = keep$open_time, close_time = keep$close_time),
    FUN = function(x) mean(x, na.rm = TRUE)
  )

  bars <- bars[order(bars$time), ]
  bars <- bars[complete.cases(bars), ]
  # midpoint should stay between 0 and 1, spread should not be negative
  bars$yes_mid <- clip01(bars$yes_mid)
  bars$yes_spread <- pmax(bars$yes_spread, 0)
  rownames(bars) <- NULL
  bars
}

market <- load_market_series(DATA_PATH)

needed <- TRAIN_POINTS + VALID_POINTS +TEST_POINTS
if (nrow(market) < needed) {
  stop("Not enough observations after resampling.")
}

window <- tail(market, needed)
# simple chronological split so there is no shuffling in the time series
train <- window[1:TRAIN_POINTS, ]
valid <- window[(TRAIN_POINTS + 1):(TRAIN_POINTS + VALID_POINTS), ]
test <- window[(TRAIN_POINTS + VALID_POINTS + 1):needed, ]


# taking a firs tlook at the series and generating acf/pacf
png(file.path(ARTIFACT_DIR, "01_series_acf_pacf.png"), width = 1300, height = 900, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(
  market$time, market$yes_mid,
  type = "l", col = "black",
  main = "YES Mid Price Over Time",
  xlab = "Time", ylab = "YES midpoint"
)
acf(train$yes_mid, lag.max = MAX_EVENT_LAG, main = "ACF:Training ")
pacf(train$yes_mid, lag.max = MAX_EVENT_LAG, main = "PACF: Training ")
hist(
  diff(train$yes_mid),
  breaks = 35, col = "gray ",
  main = "One-Step Price Changes",
  xlab = "Change in YES midpoint"
)

png(file.path(ARTIFACT_DIR, "02_differenced_acf_pacf.png"), width =1200, height = 600, res = 130)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
acf(diff(train$yes_mid), lag.max = MAX_EVENT_LAG, main = "ACF - Differenced Series")
pacf(diff(train$yes_mid), lag.max = MAX_EVENT_LAG, main = "PACF: Differenced Series")
dev.off()


# using acf nad pacf to inform model choices
acf_price <- lag_table(train$yes_mid, "acf")
pacf_price <- lag_table(train$yes_mid, "pacf")
acf_diff <- lag_table(diff(train$yes_mid), "acf")

# pacf points us toward ar lags, acf points us toward ma lags
ar_lags <- sort(unique(c(1, largest_visible_lags(pacf_price, n = 3, n_obs = nrow(train)))))
ma_lags <- sort(unique(c(1, largest_visible_lags(acf_price, n = 3, n_obs = nrow(train)))))
arima_ma_lags <- sort(unique(c(1, largest_visible_lags(acf_diff, n = 2, n_obs = nrow(train) - 1))))

# these are short within-event seasonal-ish lags, not day/week seasonality
seasonal_candidates <-  c(2, 3, 5, 7)
seasonal_candidates <-seasonal_candidates[seasonal_candidates <= MAX_EVENT_LAG]
seasonal_scan <- acf_price[acf_price$lag %in% seasonal_candidates, ]
seasonal_scan <- seasonal_scan[order(-abs(seasonal_scan$value)), ]
seasonal_lags <- head(seasonal_scan$lag, 2)

lag_notes <- data.frame(
  diagnostic = c(
    "PACF-selected AR lags",
    "ACF-selected MA lags",
    "Differenced ACF lags for ARIMA",
    "Within-event seasonal lags"
  ),
  selected_values = c(
    paste(ar_lags, collapse = ", "),
    paste(ma_lags, collapse = ", "),
    paste(arima_ma_lags, collapse = ", ") ,
    paste(seasonal_lags, collapse = ", ")
  )
)

candidate_rows <- list()

# start with pure ar and pure ma models
for (p in ar_lags) {
  candidate_rows[[length(candidate_rows) + 1]] <- model_row("AR", sprintf("AR(%d)", p), p, 0, 0)
}

for (q in ma_lags) {
  candidate_rows[[length(candidate_rows) + 1]] <- model_row("MA", sprintf("MA(%d)", q), 0, 0, q)
}

# then combine the best looking ar and ma lag ideas
for (p in head(ar_lags, 3)) {
  for (q in head(ma_lags, 3)) {
    candidate_rows[[length(candidate_rows) + 1]] <- model_row(
      "ARMA", sprintf("ARMA(%d,%d)", p, q), p, 0, q
    )
  }
}

for (p in head(ar_lags, 3)) {
  for (q in head(arima_ma_lags, 3)) {
    candidate_rows[[length(candidate_rows) + 1]] <- model_row(
      "ARIMA", sprintf("ARIMA(%d,1,%d)", p, q), p, 1, q
    )
  }
}

for (period in seasonal_lags) {
  # sarma lets us try repeated short patterns inside the same event
  candidate_rows[[length(candidate_rows) + 1]] <- model_row(
    "SARMA", sprintf("SARMA(0,0,0)x(1,0,0,%d)", period),
    0, 0, 0, 1, 0, 0, period
  )

  candidate_rows[[length(candidate_rows) + 1]] <- model_row(
    "SARMA", sprintf("SARMA(0,0,0)x(0,0,1,%d)", period),
    0, 0, 0, 0, 0, 1, period
  )

  candidate_rows[[length(candidate_rows) + 1]] <- model_row(
    "SARMA", sprintf("SARMA(1,0,1)x(1,0,0,%d)", period),
    1, 0, 1, 1, 0, 0, period
  )
}

candidates <- unique(do.call(rbind, candidate_rows))
rownames(candidates) <- NULL

write.csv(lag_notes, file.path(ARTIFACT_DIR, "diagnostic_lag_choices.csv"), row.names = FALSE)
write.csv(candidates, file.path(ARTIFACT_DIR, "informed_candidate_grid.csv"), row.names = FALSE)


# fitting the actual model - this is the code referneced in the writeup
fit_one_model <- function(y, spec) {
  # stats::arima handles all five classical model types here
  y <- tail(as.numeric(y), MAX_FIT_POINTS)
  include_mean <- spec$d == 0 && spec$D == 0

  if (spec$family == "SARMA") {
    arima(
      y,
      order = c(spec$p, spec$d, spec$q),
      seasonal = list(order = c(spec$P, spec$D, spec$Q), period = spec$period),
      include.mean = include_mean,
      method = "ML",
      optim.control = list(maxit = 100)
    )
  } else {
    arima(
      y,
      order = c(spec$p, spec$d, spec$q),
      include.mean = include_mean,
      method = "ML",
      optim.control = list(maxit = 100)
    )
  }
}

walk_forward <- function(spec, starting_history, actual_values, refit_every) {
  # refit as we move forward so the model only sees past information
  predictions  <- rep(NA_real_, length(actual_values))
  history <- as.numeric(starting_history)
  i <- 1

  while (i <= length(actual_values)) {
    n_ahead <- min(refit_every, length(actual_values) - i + 1)
    fit <- fit_one_model(history, spec)
    predictions[i:(i + n_ahead - 1)]<- clip01(as.numeric(predict(fit, n.ahead = n_ahead)$pred))
    history <- c(history, actual_values[i:(i + n_ahead - 1)])
    i <- i + n_ahead
  }

  predictions
}

forecast_error <- function(actual, predicted) {
  err <- actual - predicted
  data.frame(
    mae = mean(abs(err), na.rm = TRUE),
    rmse = sqrt(mean(err^2, na.rm = TRUE)),
    bias = mean(err, na.rm = TRUE)
  )
}

make_trade_sheet <- function(predicted_times, predicted_prices) {
  # align each forecast with the next observed market price
  row_next  <- match(predicted_times, market$time)
  keep <- !is.na(row_next) & row_next > 1 & !is.na(predicted_prices)
  row_next <- row_next[keep]
  row_now <- row_next - 1
  pred <- predicted_prices[keep]

  current <- market[row_now, ]
  future <- market[row_next, ]

  # only count trades that stay in the same 15 minute btc market
  same_event <- current$open_time ==future$open_time & current$close_time == future$close_time
  still_open <- current$time < current$close_time & future$time <= future$close_time
  tradable <- same_event & still_open

  current <- current[tradable, ]
  future <- future[tradable, ]
  pred <- pred[tradable]

  current_half_spread  <- current$yes_spread / 2
  future_half_spread <- future$yes_spread / 2

  yes_bid_now <- clip01(current$yes_mid - current_half_spread)
  yes_ask_now <-clip01(current$yes_mid + current_half_spread)
  yes_bid_future <- clip01(future$yes_mid - future_half_spread)
  yes_ask_future <- clip01(future$yes_mid + future_half_spread)

  data.frame(
    current = current$yes_mid,
    actual = future$yes_mid,
    predicted = pred,
    yes_ask = yes_ask_now,
    no_ask = 1 - yes_bid_now,
    future_yes_bid = yes_bid_future,
    future_no_bid = 1 - yes_ask_future
  )
}

score_trades <- function(trade_sheet) {
  # try a few signal cutoffs and keep the one with the best gross pnl
  empty_curve <- data.frame(
    threshold = THRESHOLDS,
    coverage = 0,
    trade_count = 0,
    yes_trade_count = 0,
    no_trade_count =0,
    hit_rate = 0,
    avg_gross_pnl = 0,
    total_gross_pnl = 0
  )

  if (nrow(trade_sheet) == 0) {
    return(list(
      summary = data.frame(
        pnl_directional_accuracy = NA_real_,
        best_threshold = 0,
        best_coverage = 0,
        best_hit_rate = 0,
        best_avg_gross_pnl = 0,
        best_total_gross_pnl = 0,
        best_yes_trade_count = 0,
        best_no_trade_count = 0
      ),
      curve = empty_curve
    ))
  }

  predicted_move <- trade_sheet$predicted - trade_sheet$current
  actual_move <- trade_sheet$actual -trade_sheet$current
  moved <- actual_move != 0
  direction_acc <- if (any(moved)) mean(sign(predicted_move[moved]) == sign(actual_move[moved])) else NA_real_

  curve <- do.call(rbind, lapply(THRESHOLDS, function(threshold) {
    # positive signal buys yes, negative signal buys no
    side <- ifelse(abs(predicted_move) >= threshold, sign(predicted_move), 0)
    traded <- side != 0

    if (!any(traded)) {
      return(data.frame(
        threshold =threshold,
        coverage = 0,
        trade_count = 0,
        yes_trade_count = 0,
        no_trade_count = 0,
        hit_rate = 0,
        avg_gross_pnl = 0,
        total_gross_pnl = 0
      ))
    }

    pnl <- ifelse(
      side[traded] > 0,
      trade_sheet$future_yes_bid[traded]- trade_sheet$yes_ask[traded],
      trade_sheet$future_no_bid[traded] - trade_sheet$no_ask[traded]
    )

    data.frame(
      threshold = threshold,
      coverage =mean(traded),
      trade_count = sum(traded),
      yes_trade_count = sum(side[traded] > 0),
      no_trade_count = sum(side[traded] < 0),
      hit_rate = mean(pnl > 0),
      avg_gross_pnl = mean(pnl),
      total_gross_pnl = sum(pnl)
    )
  }))

  eligible <- curve[curve$coverage >= MIN_COVERAGE, ]
  if (nrow(eligible) == 0) eligible <- curve
  best <- eligible[order(-eligible$total_gross_pnl, -eligible$avg_gross_pnl), ][1, ]

  list(
    summary = data.frame(
      pnl_directional_accuracy = direction_acc,
      best_threshold = best$threshold,
      best_coverage = best$coverage,
      best_hit_rate = best$hit_rate,
      best_avg_gross_pnl = best$avg_gross_pnl,
      best_total_gross_pnl = best$total_gross_pnl,
      best_yes_trade_count = best$yes_trade_count,
      best_no_trade_count = best$no_trade_count
    ),
    curve = curve
  )
}

evaluate_spec <- function(spec, history, evaluation_set, refit_every)  {
  # one wrapper so each model gets scored the exact same way
  pred <- tryCatch(
   walk_forward(spec, history, evaluation_set$yes_mid, refit_every),
    error = function(e) rep(NA_real_, nrow(evaluation_set))
  )

  if (all(is.na(pred))) return(NULL)

  errors <- forecast_error(evaluation_set$yes_mid, pred)
  trade_sheet <- make_trade_sheet(evaluation_set$time, pred)
  pnl <- score_trades(trade_sheet)

  list(
    result = cbind(spec, errors, pnl$summary),
    predictions = pred,
    threshold_curve = pnl$curve
  )
}


# validation first and then holdout
validation_results <- list()

# first pass: use validation to narrow the list down
for (i in seq_len(nrow(candidates))) {
  scored <- evaluate_spec(candidates[i, ], train$yes_mid, valid, VALID_REFIT_EVERY)
  if (!is.null(scored)) {
    validation_results[[length(validation_results) + 1]] <- scored$result
  }
}

validation_results <- do.call(rbind, validation_results)
validation_results <- validation_results[order(-validation_results$best_total_gross_pnl), ]
rownames(validation_results) <- NULL
write.csv(validation_results, file.path(ARTIFACT_DIR, "validation_results.csv"), row.names = FALSE)

# The holdout round is intentionally smaller: the best six validation models,
# plus the best model from each family so AR, MA, ARMA, ARIMA, and SARMA all get
# a fair final look.
top_six <- head(validation_results,  6)
family_winners <- do.call(rbind, lapply(split(validation_results, validation_results$family), function(x) x[1, ]))
holdout_candidates <- unique(rbind(top_six, family_winners)[, names(candidates)])

holdout_results <-list()
holdout_predictions <- list()
holdout_curves <- list()
train_valid_history <- c(train$yes_mid, valid$yes_mid)

# final pass: only the short list gets checked on holdout
for (i in seq_len(nrow(holdout_candidates))) {
  scored <- evaluate_spec(holdout_candidates[i, ], train_valid_history, test, TEST_REFIT_EVERY)
  if (!is.null(scored)) {
    holdout_results[[length(holdout_results) + 1]] <- scored$result
    holdout_predictions[[scored$result$model]] <- scored$predictions
    holdout_curves[[scored$result$model]] <- scored$threshold_curve
  }
}

holdout_results <- do.call(rbind, holdout_results)
holdout_results <- holdout_results[order(-holdout_results$best_total_gross_pnl), ]
rownames(holdout_results) <- NULL
write.csv(holdout_results, file.path(ARTIFACT_DIR, "holdout_results.csv"), row.names = FALSE)

best <-holdout_results[1, ]
# keeping the best model objects so the last plots match the winner
best_spec <- holdout_candidates[holdout_candidates$model == best$model, ][1, ]
best_predictions <- holdout_predictions[[best$model]]
best_curve <- holdout_curves[[best$model]]
write.csv(best_curve, file.path(ARTIFACT_DIR, "best_threshold_curve.csv"), row.names = FALSE)

# final figures
top_for_plot <- holdout_results[order(-holdout_results$best_total_gross_pnl), ]
top_for_plot <- top_for_plot[seq_len(min(10, nrow(top_for_plot))), ]
top_for_plot <- top_for_plot[order(top_for_plot$best_total_gross_pnl), ]

png(file.path(ARTIFACT_DIR, "04_informed_model_comparison.png"), width = 1450, height =760, res = 130)
par(mar = c(5, 13, 4, 2))
barplot(
  top_for_plot$best_total_gross_pnl,
  names.arg = top_for_plot$model,
  horiz = TRUE,
  las = 1,
  cex.names = 0.8,
  col = "#2A9D8F",
  main = "Informed Candi date Models by Gross PnL ",
  xlab = "Total gross PnL, one-contract units"
)
abline(v = 0, lty = 2)

best_fit <- fit_one_model(train_valid_history, best_spec)
best_residuals <- na.omit(best_fit$residuals)

png(file.path(ARTIFACT_DIR, "05_best_model_residual_diagnostics.png"), width = 1200, height = 850, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(best_residuals, type = "l", main = "Residuals Over  Time", ylab = "Residual", xlab = "Index")
abline(h = 0, col = "red", lty = 2)
hist(best_residuals, breaks = 30, col = " gray", probability = TRUE, main = "Residual Histogram", xlab = "Residual")
lines(density(best_residuals), col = "blue", lwd = 2)
qqnorm(best_residuals, main = "Q-Q Plot")
qqline(best_residuals , col = "red", lwd = 2)
acf(best_residuals, main = "Residual ACF")
dev.off()

png(file.path(ARTIFACT_DIR, "06_best_forecast_and_thresholds.png"), width =1300, height = 850, res = 130)
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
zoom <- seq_len(min(140, nrow(test)))
plot(
  test$time[zoom], test$yes_mid[zoom],
  type = "l", col = "black", lwd = 2,
  main = paste("Actual vs Forecast:", best$model),
  xlab = "Time", ylab = "YES midpoint"
)
lines(test$time[zoom], best_predictions[zoom], col = "#D55E00", lwd  = 2)
legend("topleft", legend = c("Actual", "Forecast"), col = c("black", "#D55E00"), lwd = 2, bty = "n")

plot(
  best_curve$threshold, best_curve$total_gross_pnl,
  type = "b", pch = 19, col = "#457B9D",
  main = "Trading Threshold vs Gross PnL",
  xlab = "Predicted move threshold",
  ylab = "Total gross PnL"
)
abline(h = 0, lty = 2)
