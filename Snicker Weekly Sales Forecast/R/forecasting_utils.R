load_sales_data <- function(path = "data/snickers.csv") {
  if (!file.exists(path)) {
    stop(sprintf("Data file not found: %s", path), call. = FALSE)
  }

  data <- read.csv(path, stringsAsFactors = FALSE)
  required_columns <- c("Week", "Our.price", "Comp.price", "Display", "Sales")
  missing_columns <- setdiff(required_columns, names(data))

  if (length(missing_columns) > 0) {
    stop(
      sprintf("Missing required columns: %s", paste(missing_columns, collapse = ", ")),
      call. = FALSE
    )
  }

  data <- data[order(data$Week), ]
  data$Display <- as.numeric(data$Display)
  data
}

compute_accuracy_metrics <- function(actual, predicted) {
  actual <- as.numeric(actual)
  predicted <- as.numeric(predicted)
  valid <- is.finite(actual) & is.finite(predicted) & actual != 0

  if (!any(valid)) {
    return(list(rmse = NA_real_, mape = NA_real_, mae = NA_real_))
  }

  actual <- actual[valid]
  predicted <- predicted[valid]

  list(
    rmse = sqrt(mean((predicted - actual) ^ 2)),
    mape = mean(abs(predicted - actual) / actual),
    mae = mean(abs(predicted - actual))
  )
}

build_future_frame <- function(data, horizon, our_price, comp_price, display) {
  start_week <- max(data$Week, na.rm = TRUE) + 1

  data.frame(
    Week = seq.int(start_week, by = 1, length.out = horizon),
    Our.price = rep(our_price, horizon),
    Comp.price = rep(comp_price, horizon),
    Display = rep(display, horizon)
  )
}

naive_forecast <- function(history, horizon) {
  rep(tail(history$Sales, 1), horizon)
}

moving_average_forecast <- function(history, horizon, window = 4) {
  width <- min(window, nrow(history))
  rep(mean(tail(history$Sales, width)), horizon)
}

ses_fit_forecast <- function(history, horizon, alpha = 0.2) {
  alpha <- max(min(alpha, 0.99), 0.01)
  level <- history$Sales[1]

  for (idx in 2:nrow(history)) {
    level <- alpha * history$Sales[idx] + (1 - alpha) * level
  }

  rep(level, horizon)
}

fit_model_by_id <- function(model_id, history) {
  switch(
    model_id,
    log_linear = lm(log(Sales) ~ log(Our.price) + log(Comp.price) + Display, data = history),
    arimax = lm(
      log(Sales) ~ log(lag_sales) + log(Our.price) + log(Comp.price) + Display,
      data = transform(history, lag_sales = c(NA, head(Sales, -1)))[-1, ]
    ),
    arima = stats::arima(history$Sales, order = c(1, 0, 0)),
    ets = list(alpha = 0.2),
    naive = list(type = "naive"),
    moving_average = list(type = "moving_average"),
    stop(sprintf("Unsupported model id: %s", model_id), call. = FALSE)
  )
}

predict_model_by_id <- function(model_id, fit, history, future_data) {
  horizon <- nrow(future_data)

  switch(
    model_id,
    log_linear = {
      pmax(exp(stats::predict(fit, newdata = future_data)), 0)
    },
    arimax = {
      forecasts <- numeric(horizon)
      augmented_history <- history

      for (idx in seq_len(horizon)) {
        lag_sales <- tail(augmented_history$Sales, 1)
        new_row <- future_data[idx, , drop = FALSE]
        new_row$lag_sales <- lag_sales
        forecasts[idx] <- exp(stats::predict(fit, newdata = new_row))
        augmented_history <- rbind(
          augmented_history,
          data.frame(
            Week = new_row$Week,
            Our.price = new_row$Our.price,
            Comp.price = new_row$Comp.price,
            Display = new_row$Display,
            Sales = forecasts[idx]
          )
        )
      }

      pmax(forecasts, 0)
    },
    arima = {
      as.numeric(stats::predict(fit, n.ahead = horizon)$pred)
    },
    ets = {
      ses_fit_forecast(history, horizon = horizon, alpha = fit$alpha)
    },
    naive = {
      naive_forecast(history, horizon = horizon)
    },
    moving_average = {
      moving_average_forecast(history, horizon = horizon)
    },
    stop(sprintf("Unsupported model id: %s", model_id), call. = FALSE)
  )
}

generate_intervals <- function(predictions, residual_sd) {
  margin <- 1.96 * residual_sd
  data.frame(
    lower = pmax(predictions - margin, 0),
    upper = pmax(predictions + margin, 0)
  )
}

backtest_model <- function(model_id, data, initial_window = 36) {
  if (nrow(data) <= initial_window + 1) {
    return(data.frame(Week = numeric(), Actual = numeric(), Predicted = numeric()))
  }

  records <- vector("list", length = nrow(data) - initial_window)
  counter <- 1

  for (idx in seq.int(initial_window + 1, nrow(data))) {
    history <- data[1:(idx - 1), ]
    actual <- data[idx, , drop = FALSE]
    future_frame <- actual[, c("Week", "Our.price", "Comp.price", "Display")]
    fit <- tryCatch(fit_model_by_id(model_id, history), error = function(e) NULL)

    if (is.null(fit)) {
      next
    }

    predicted <- tryCatch(
      predict_model_by_id(model_id, fit, history, future_frame)[1],
      error = function(e) NA_real_
    )

    records[[counter]] <- data.frame(
      Week = actual$Week,
      Actual = actual$Sales,
      Predicted = predicted
    )
    counter <- counter + 1
  }

  records <- Filter(Negate(is.null), records)
  if (length(records) == 0) {
    return(data.frame(Week = numeric(), Actual = numeric(), Predicted = numeric()))
  }

  do.call(rbind, records)
}

build_model_catalog <- function() {
  data.frame(
    id = c("log_linear", "arimax", "arima", "ets"),
    label = c(
      "Log-Linear Regression",
      "Dynamic Regression / ARIMAX",
      "ARIMA(1,0,0)",
      "ETS / Exponential Smoothing"
    ),
    category = c("recommended", "recommended", "recommended", "recommended"),
    price_sensitive = c(TRUE, TRUE, FALSE, FALSE),
    stringsAsFactors = FALSE
  )
}

is_price_sensitive_model <- function(model_id, model_catalog) {
  match_idx <- match(model_id, model_catalog$id)
  if (is.na(match_idx)) {
    return(FALSE)
  }

  isTRUE(model_catalog$price_sensitive[match_idx])
}

score_models <- function(data, model_catalog, initial_window = 36) {
  rows <- lapply(seq_len(nrow(model_catalog)), function(idx) {
    model_id <- model_catalog$id[idx]
    backtest <- backtest_model(model_id, data, initial_window = initial_window)
    metrics <- compute_accuracy_metrics(backtest$Actual, backtest$Predicted)

    data.frame(
      id = model_id,
      label = model_catalog$label[idx],
      category = model_catalog$category[idx],
      rmse = metrics$rmse,
      mape = metrics$mape,
      mae = metrics$mae,
      stringsAsFactors = FALSE
    )
  })

  leaderboard <- do.call(rbind, rows)
  leaderboard <- leaderboard[order(leaderboard$rmse, leaderboard$mape), ]
  leaderboard$rank <- seq_len(nrow(leaderboard))
  leaderboard
}

resolve_model_choice <- function(selected_model_id, leaderboard) {
  if (is.null(selected_model_id) || !nzchar(selected_model_id)) {
    return(leaderboard$id[1])
  }

  selected_model_id
}

run_forecast <- function(data, selected_model_id, model_catalog, horizon, our_price, comp_price, display) {
  leaderboard <- score_models(data, model_catalog)
  model_id <- resolve_model_choice(selected_model_id, leaderboard)
  history <- data
  future_data <- build_future_frame(data, horizon, our_price, comp_price, display)
  fit <- fit_model_by_id(model_id, history)
  predictions <- predict_model_by_id(model_id, fit, history, future_data)
  residual_reference <- backtest_model(model_id, data)
  residual_sd <- stats::sd(residual_reference$Actual - residual_reference$Predicted, na.rm = TRUE)
  if (!is.finite(residual_sd)) {
    residual_sd <- stats::sd(history$Sales, na.rm = TRUE) * 0.1
  }

  forecast <- cbind(
    future_data,
    Forecast = predictions,
    generate_intervals(predictions, residual_sd)
  )

  list(
    model_id = model_id,
    leaderboard = leaderboard,
    forecast = forecast,
    backtest = residual_reference
  )
}

find_optimal_price <- function(data, selected_model_id, model_catalog, horizon, our_price, comp_price, display, unit_cost, display_cost, price_grid = NULL) {
  if (is.null(price_grid)) {
    observed_min <- min(data$Our.price, na.rm = TRUE)
    observed_max <- max(data$Our.price, na.rm = TRUE)
    lower_bound <- max(0.01, observed_min * 0.8)
    upper_bound <- max(lower_bound + 0.05, observed_max * 1.2)
    price_grid <- seq(lower_bound, upper_bound, by = 0.01)
  }

  leaderboard <- score_models(data, model_catalog)
  model_id <- resolve_model_choice(selected_model_id, leaderboard)
  display_weeks <- horizon * as.numeric(display)

  if (!is_price_sensitive_model(model_id, model_catalog)) {
    baseline_result <- run_forecast(
      data = data,
      selected_model_id = model_id,
      model_catalog = model_catalog,
      horizon = horizon,
      our_price = our_price,
      comp_price = comp_price,
      display = display
    )

    total_units <- sum(baseline_result$forecast$Forecast, na.rm = TRUE)
    unit_profit <- our_price - unit_cost
    gross_profit <- total_units * unit_profit - display_cost * display_weeks

    return(list(
      model_id = model_id,
      best_price = our_price,
      best_units = total_units,
      best_gross_profit = gross_profit,
      grid = data.frame(
        price = our_price,
        total_units = total_units,
        gross_profit = gross_profit
      ),
      optimized = FALSE
    ))
  }

  optimization_rows <- lapply(price_grid, function(candidate_price) {
    result <- run_forecast(
      data = data,
      selected_model_id = model_id,
      model_catalog = model_catalog,
      horizon = horizon,
      our_price = candidate_price,
      comp_price = comp_price,
      display = display
    )

    total_units <- sum(result$forecast$Forecast, na.rm = TRUE)
    unit_profit <- candidate_price - unit_cost
    gross_profit <- total_units * unit_profit - display_cost * display_weeks

    data.frame(
      price = candidate_price,
      total_units = total_units,
      gross_profit = gross_profit
    )
  })

  optimization_table <- do.call(rbind, optimization_rows)
  best_row <- optimization_table[which.max(optimization_table$gross_profit), , drop = FALSE]

  list(
    model_id = model_id,
    best_price = best_row$price[1],
    best_units = best_row$total_units[1],
    best_gross_profit = best_row$gross_profit[1],
    grid = optimization_table,
    optimized = TRUE
  )
}
