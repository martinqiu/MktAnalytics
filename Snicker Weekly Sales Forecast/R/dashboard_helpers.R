format_number <- function(x, digits = 0) {
  format(round(x, digits), big.mark = ",", scientific = FALSE, nsmall = digits)
}

format_percent <- function(x, digits = 1) {
  ifelse(is.na(x), "NA", paste0(round(x * 100, digits), "%"))
}

model_summary_text <- function(model_id) {
  switch(
    model_id,
    log_linear = "Uses price and display drivers to forecast sales with a log-linear regression.",
    arimax = "Uses lagged sales plus price and display drivers in a dynamic regression.",
    arima = "Uses autoregressive time-series structure from past sales only.",
    ets = "Uses exponential smoothing to estimate level from past sales.",
    naive = "Uses the latest observed sales value as the next forecast.",
    moving_average = "Uses the recent rolling average of sales as the forecast.",
    "Uses the current champion selected from rolling backtest performance."
  )
}

build_kpi_cards <- function(latest_sales, next_forecast, rmse, mape) {
  list(
    latest_sales = format_number(latest_sales),
    next_forecast = format_number(next_forecast),
    rmse = format_number(rmse),
    mape = format_percent(mape)
  )
}
