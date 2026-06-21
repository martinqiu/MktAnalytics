planning_model_label <- "Lagged Regression model"

build_default_scheme <- function(start_week = 43, end_week = 52) {
  data.frame(
    Week = seq.int(start_week, end_week),
    Our.price = rep(1, end_week - start_week + 1),
    Comp.price = rep(1, end_week - start_week + 1),
    Display = rep(1, end_week - start_week + 1),
    stringsAsFactors = FALSE
  )
}

normalize_scheme <- function(scheme) {
  cleaned <- scheme
  cleaned$Week <- as.integer(cleaned$Week)
  cleaned$Our.price <- pmax(as.numeric(cleaned$Our.price), 0.01)
  cleaned$Comp.price <- pmax(as.numeric(cleaned$Comp.price), 0.01)
  cleaned$Display <- ifelse(as.numeric(cleaned$Display) >= 0.5, 1, 0)
  cleaned[order(cleaned$Week), ]
}

fit_planning_model <- function(history) {
  model_data <- history
  model_data$lag1_sales <- c(NA, head(model_data$Sales, -1))
  model_data$lag1_our <- c(NA, head(model_data$Our.price, -1))
  model_data$lag1_comp <- c(NA, head(model_data$Comp.price, -1))
  model_data <- model_data[complete.cases(model_data), ]

  lm(
    log(Sales) ~ log(lag1_sales) + log(Our.price) + log(Comp.price) + Display + log(lag1_our) + log(lag1_comp),
    data = model_data
  )
}

predict_scheme_sales <- function(history, scheme) {
  fit <- fit_planning_model(history)
  planned <- normalize_scheme(scheme)
  predictions <- numeric(nrow(planned))

  last_sales <- tail(history$Sales, 1)
  last_our <- tail(history$Our.price, 1)
  last_comp <- tail(history$Comp.price, 1)

  for (idx in seq_len(nrow(planned))) {
    newdata <- data.frame(
      lag1_sales = last_sales,
      Our.price = planned$Our.price[idx],
      Comp.price = planned$Comp.price[idx],
      Display = planned$Display[idx],
      lag1_our = last_our,
      lag1_comp = last_comp
    )

    predictions[idx] <- exp(predict(fit, newdata = newdata))
    last_sales <- predictions[idx]
    last_our <- planned$Our.price[idx]
    last_comp <- planned$Comp.price[idx]
  }

  pmax(as.numeric(predictions), 0)
}

calculate_markup_financials <- function(scheme, forecast_sales, markup_rate, weekly_display_cost) {
  unit_cost <- scheme$Our.price / (1 + markup_rate)
  unit_profit <- scheme$Our.price - unit_cost
  revenue <- forecast_sales * scheme$Our.price
  display_cost <- scheme$Display * weekly_display_cost
  profit <- forecast_sales * unit_profit - display_cost

  cbind(
    normalize_scheme(scheme),
    Forecast = forecast_sales,
    Unit.cost = unit_cost,
    Unit.profit = unit_profit,
    Revenue = revenue,
    Display.cost = display_cost,
    Profit = profit
  )
}

summarize_scheme_totals <- function(weekly_results, scheme_name, markup_rate, weekly_display_cost) {
  data.frame(
    Scheme = scheme_name,
    Forecast.engine = planning_model_label,
    Markup = markup_rate,
    Weekly.display.cost = weekly_display_cost,
    Total.forecast.sales = sum(weekly_results$Forecast, na.rm = TRUE),
    Total.revenue = sum(weekly_results$Revenue, na.rm = TRUE),
    Total.display.cost = sum(weekly_results$Display.cost, na.rm = TRUE),
    Total.profit = sum(weekly_results$Profit, na.rm = TRUE),
    Avg.our.price = mean(weekly_results$Our.price, na.rm = TRUE),
    Avg.comp.price = mean(weekly_results$Comp.price, na.rm = TRUE),
    Display.weeks = sum(weekly_results$Display, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

best_model_metrics <- function(history) {
  bt <- planning_model_backtest(history)
  metrics <- compute_accuracy_metrics(bt$Actual, bt$Predicted)
  data.frame(
    id = "planning_lagged_log_linear",
    label = planning_model_label,
    rmse = metrics$rmse,
    mape = metrics$mape,
    mae = metrics$mae,
    rank = 1,
    stringsAsFactors = FALSE
  )
}

planning_model_backtest <- function(data, initial_window = 36) {
  records <- list()

  for (idx in seq.int(initial_window + 1, nrow(data))) {
    history <- data[1:(idx - 1), ]
    actual <- data[idx, , drop = FALSE]
    fit <- fit_planning_model(history)

    newdata <- data.frame(
      lag1_sales = tail(history$Sales, 1),
      Our.price = actual$Our.price,
      Comp.price = actual$Comp.price,
      Display = actual$Display,
      lag1_our = tail(history$Our.price, 1),
      lag1_comp = tail(history$Comp.price, 1)
    )

    predicted <- exp(predict(fit, newdata = newdata))
    records[[length(records) + 1]] <- data.frame(
      Week = actual$Week,
      Actual = actual$Sales,
      Predicted = predicted
    )
  }

  do.call(rbind, records)
}
