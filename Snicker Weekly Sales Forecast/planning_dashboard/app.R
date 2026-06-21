app_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
project_root <- normalizePath(file.path(app_dir, ".."), winslash = "/", mustWork = TRUE)

source(file.path(project_root, "R", "forecasting_utils.R"))
source(file.path(project_root, "R", "planning_dashboard_utils.R"))
source(file.path(project_root, "R", "dashboard_helpers.R"))

library(shiny)
library(bslib)
library(plotly)
library(DT)

sales_data <- load_sales_data(file.path(project_root, "data", "snickers.csv"))
model_metrics <- best_model_metrics(sales_data)
default_scheme <- build_default_scheme()

resizable_card <- function(..., min_height = "300px") {
  card(
    style = paste0(
      "resize: both; overflow: auto; min-height: ", min_height,
      "; min-width: 340px;"
    ),
    ...
  )
}

arrow_button <- function(id, label, title) {
  actionButton(
    inputId = id,
    label = label,
    title = title,
    class = "arrow-button"
  )
}

price_cell <- function(week, prefix, value) {
  tags$div(
    class = "adjust-cell",
    arrow_button(paste0(prefix, "_up_", week), "\u25B2", "Increase by $0.05"),
    tags$div(class = "adjust-value", paste0("$", format_number(value, 2))),
    arrow_button(paste0(prefix, "_down_", week), "\u25BC", "Decrease by $0.05")
  )
}

display_cell <- function(week, value) {
  tags$div(
    class = "adjust-cell",
    arrow_button(paste0("display_up_", week), "\u25B2", "Set display to 1"),
    tags$div(class = "adjust-value", as.character(value)),
    arrow_button(paste0("display_down_", week), "\u25BC", "Set display to 0")
  )
}

adjustment_grid <- function(scheme) {
  weeks <- scheme$Week

  tags$div(
    class = "adjust-grid",
    tags$div(class = "grid-label header-label", "Horizon"),
    lapply(weeks, function(week) tags$div(class = "week-header", paste("Week", week))),
    tags$div(class = "grid-label", "Our price"),
    lapply(seq_along(weeks), function(i) price_cell(weeks[i], "our", scheme$Our.price[i])),
    tags$div(class = "grid-label", "Comp price"),
    lapply(seq_along(weeks), function(i) price_cell(weeks[i], "comp", scheme$Comp.price[i])),
    tags$div(class = "grid-label", "Display"),
    lapply(seq_along(weeks), function(i) display_cell(weeks[i], scheme$Display[i]))
  )
}

ui <- page_fluid(
  title = "Snickers Ten-Week Scheme Planner",
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly",
    primary = "#1f4e5f",
    secondary = "#c2611a",
    success = "#2a7f62",
    bg = "#f7f4ed",
    fg = "#183642",
    base_font = font_google("Source Sans 3"),
    heading_font = font_google("Libre Baskerville")
  ),
  tags$style(HTML("
    body { background: #f7f4ed; }
    .app-title { margin: 18px 0 8px; }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto auto auto;
      gap: 12px;
      align-items: end;
      margin-bottom: 14px;
    }
    .toolbar .form-group { margin-bottom: 0; }
    .adjust-grid {
      display: grid;
      grid-template-columns: 120px repeat(10, minmax(92px, 1fr));
      gap: 1px;
      background: #cfc8bd;
      border: 1px solid #cfc8bd;
      overflow-x: auto;
    }
    .grid-label, .week-header {
      background: #efeae0;
      font-weight: 700;
      padding: 12px 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 74px;
      text-align: center;
    }
    .header-label { background: #e4ddd0; }
    .adjust-cell {
      background: #fffaf2;
      min-height: 112px;
      display: grid;
      grid-template-rows: 32px 1fr 32px;
      align-items: center;
      justify-items: center;
      padding: 8px 6px;
    }
    .adjust-value {
      font-size: 1.25rem;
      font-weight: 700;
      color: #183642;
      line-height: 1;
    }
    .arrow-button {
      width: 34px;
      height: 30px;
      padding: 0;
      border-radius: 4px;
      border: 1px solid #9d9282;
      background: #f4eee4;
      color: #183642;
      font-weight: 700;
      line-height: 1;
    }
    .arrow-button:hover {
      background: #e8dccd;
      border-color: #1f4e5f;
    }
    .card {
      border-radius: 6px;
    }
    .adjustment-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      width: 100%;
    }
    .assumption-panel {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(220px, 1fr);
      gap: 24px;
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid #d8d0c3;
    }
    .assumption-panel .form-group {
      margin-bottom: 0;
    }
    @media (max-width: 1200px) {
      .toolbar { grid-template-columns: repeat(2, minmax(220px, 1fr)); }
    }
  ")),
  tags$h2(class = "app-title", "Snickers Ten-Week Scheme Planner"),
  tags$div(
    class = "toolbar",
    textInput("scheme_name", "Scheme name", value = "Base scheme"),
    actionButton("reset_scheme", "Reset", class = "btn btn-outline-secondary"),
    actionButton("all_display_on", "All display = 1", class = "btn btn-outline-secondary"),
    actionButton("all_display_off", "All display = 0", class = "btn btn-outline-secondary"),
    actionButton("clear_schemes", "Clear comparison", class = "btn btn-outline-danger")
  ),
  layout_column_wrap(
    width = 1 / 4,
    value_box(title = "Forecast Engine", value = planning_model_label, theme = "primary"),
    value_box(title = "Rolling MAPE", value = textOutput("kpi_mape", inline = TRUE), theme = "secondary"),
    value_box(title = "Total 10-Week Sales", value = textOutput("kpi_sales", inline = TRUE), theme = "success"),
    value_box(title = "Total 10-Week Profit", value = textOutput("kpi_profit", inline = TRUE), theme = "warning")
  ),
  resizable_card(
    card_header(
      tags$div(
        class = "adjustment-header",
        tags$span("Adjustment Panel"),
        actionButton("submit_scheme", "Submit scheme", class = "btn btn-primary")
      )
    ),
    uiOutput("adjustment_panel"),
    tags$div(
      class = "assumption-panel",
      sliderInput("markup_rate", "Markup", min = 30, max = 50, value = 40, step = 1, post = "%"),
      sliderInput("display_cost", "Weekly display cost", min = 500, max = 2000, value = 1000, step = 100, pre = "$")
    ),
    min_height = "480px"
  ),
  layout_columns(
    resizable_card(
      card_header("Current Scheme Totals"),
      tableOutput("totals_table"),
      min_height = "280px"
    ),
    resizable_card(
      card_header("Recorded Scheme Comparison"),
      DTOutput("scheme_comparison_table"),
      min_height = "340px"
    ),
    col_widths = c(4, 8)
  ),
  layout_columns(
    resizable_card(
      card_header("Historical Sales Context"),
      plotlyOutput("history_plot", height = 360),
      min_height = "420px"
    ),
    col_widths = c(12)
  )
)

server <- function(input, output, session) {
  current_scheme <- reactiveVal(default_scheme)
  recorded_schemes <- reactiveVal(data.frame())
  last_submitted <- reactiveVal(NULL)
  scheme_counter <- reactiveVal(0)

  update_price <- function(week, column, delta) {
    scheme <- current_scheme()
    row <- which(scheme$Week == week)
    scheme[row, column] <- min(max(scheme[row, column] + delta, 0.50), 2.00)
    current_scheme(normalize_scheme(scheme))
  }

  update_display <- function(week, value) {
    scheme <- current_scheme()
    row <- which(scheme$Week == week)
    scheme$Display[row] <- value
    current_scheme(normalize_scheme(scheme))
  }

  lapply(default_scheme$Week, function(week) {
    local({
      wk <- week
      observeEvent(input[[paste0("our_up_", wk)]], update_price(wk, "Our.price", 0.05), ignoreInit = TRUE)
      observeEvent(input[[paste0("our_down_", wk)]], update_price(wk, "Our.price", -0.05), ignoreInit = TRUE)
      observeEvent(input[[paste0("comp_up_", wk)]], update_price(wk, "Comp.price", 0.05), ignoreInit = TRUE)
      observeEvent(input[[paste0("comp_down_", wk)]], update_price(wk, "Comp.price", -0.05), ignoreInit = TRUE)
      observeEvent(input[[paste0("display_up_", wk)]], update_display(wk, 1), ignoreInit = TRUE)
      observeEvent(input[[paste0("display_down_", wk)]], update_display(wk, 0), ignoreInit = TRUE)
    })
  })

  observeEvent(input$reset_scheme, {
    current_scheme(default_scheme)
  })

  observeEvent(input$all_display_on, {
    scheme <- current_scheme()
    scheme$Display <- 1
    current_scheme(normalize_scheme(scheme))
  })

  observeEvent(input$all_display_off, {
    scheme <- current_scheme()
    scheme$Display <- 0
    current_scheme(normalize_scheme(scheme))
  })

  output$adjustment_panel <- renderUI({
    adjustment_grid(current_scheme())
  })

  calculate_scheme_results <- function(scheme, scheme_name) {
    weekly_forecast <- predict_scheme_sales(sales_data, scheme)
    weekly_results <- calculate_markup_financials(
      scheme = scheme,
      forecast_sales = weekly_forecast,
      markup_rate = input$markup_rate / 100,
      weekly_display_cost = input$display_cost
    )

    totals <- summarize_scheme_totals(
      weekly_results = weekly_results,
      scheme_name = scheme_name,
      markup_rate = input$markup_rate / 100,
      weekly_display_cost = input$display_cost
    )

    list(
      scheme = scheme,
      weekly = weekly_results,
      totals = totals
    )
  }

  observeEvent(input$submit_scheme, {
    next_number <- scheme_counter() + 1
    scheme_counter(next_number)

    default_name <- paste("Scheme", next_number)
    typed_name <- trimws(input$scheme_name)
    scheme_name <- if (nzchar(typed_name) && typed_name != "Base scheme") typed_name else default_name
    results <- calculate_scheme_results(current_scheme(), scheme_name)
    last_submitted(results)

    totals <- results$totals
    totals$Recorded.at <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    existing <- recorded_schemes()
    recorded_schemes(if (nrow(existing) == 0) totals else rbind(existing, totals))
  })

  observeEvent(input$clear_schemes, {
    recorded_schemes(data.frame())
    last_submitted(NULL)
    scheme_counter(0)
  })

  output$kpi_mape <- renderText(format_percent(model_metrics$mape[1], 1))
  output$kpi_sales <- renderText({
    results <- last_submitted()
    if (is.null(results)) return("Not submitted")
    format_number(results$totals$Total.forecast.sales[1], 0)
  })
  output$kpi_profit <- renderText({
    results <- last_submitted()
    if (is.null(results)) return("Not submitted")
    paste0("$", format_number(results$totals$Total.profit[1], 2))
  })

  output$totals_table <- renderTable({
    results <- last_submitted()
    if (is.null(results)) {
      return(data.frame(
        Metric = "Status",
        Value = "Adjust the scheme, then click Submit scheme.",
        check.names = FALSE,
        stringsAsFactors = FALSE
      ))
    }

    totals <- results$totals
    data.frame(
      Metric = c(
        "Planning horizon",
        "Total forecast sales",
        "Total revenue",
        "Total display cost",
        "Total profit",
        "Display weeks",
        "Average planned price",
        "Average competitor price"
      ),
      Value = c(
        "Weeks 43 to 52",
        paste0(format_number(totals$Total.forecast.sales, 0), " units"),
        paste0("$", format_number(totals$Total.revenue, 2)),
        paste0("$", format_number(totals$Total.display.cost, 2)),
        paste0("$", format_number(totals$Total.profit, 2)),
        format_number(totals$Display.weeks, 0),
        paste0("$", format_number(totals$Avg.our.price, 2)),
        paste0("$", format_number(totals$Avg.comp.price, 2))
      ),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  }, striped = TRUE, bordered = FALSE, spacing = "m", width = "100%")

  output$scheme_comparison_table <- renderDT({
    saved <- recorded_schemes()

    if (nrow(saved) == 0) {
      return(datatable(
        data.frame(Message = "No schemes recorded yet."),
        rownames = FALSE,
        options = list(dom = "t", paging = FALSE, searching = FALSE, info = FALSE)
      ))
    }

    display_table <- saved
    display_table$Markup <- paste0(round(display_table$Markup * 100, 0), "%")

    currency_columns <- c("Weekly.display.cost", "Total.revenue", "Total.display.cost", "Total.profit", "Avg.our.price", "Avg.comp.price")
    count_columns <- c("Total.forecast.sales", "Display.weeks")

    display_table[currency_columns] <- lapply(display_table[currency_columns], function(x) paste0("$", format_number(x, 2)))
    display_table[count_columns] <- lapply(display_table[count_columns], function(x) format_number(x, 0))

    datatable(
      display_table[, c(
        "Scheme",
        "Forecast.engine",
        "Markup",
        "Weekly.display.cost",
        "Total.forecast.sales",
        "Total.display.cost",
        "Total.profit",
        "Recorded.at"
      )],
      rownames = FALSE,
      options = list(
        paging = FALSE,
        searching = FALSE,
        info = FALSE,
        scrollX = TRUE,
        dom = "t"
      )
    )
  })

  output$history_plot <- renderPlotly({
    plot <- plot_ly(
      data = sales_data,
      x = ~Week,
      y = ~Sales,
      type = "scatter",
      mode = "lines+markers",
      name = "Actual sales",
      line = list(color = "#002FA7", width = 3),
      marker = list(size = 7, color = "#002FA7")
    )

    results <- last_submitted()
    if (!is.null(results)) {
      plot <- plot |>
        add_lines(
          data = results$weekly,
          x = ~Week,
          y = ~Forecast,
          name = "Predicted sales",
          line = list(color = "#f28c28", width = 3, dash = "dot"),
          marker = list(size = 7, color = "#f28c28")
        )
    }

    plot |>
      layout(
        xaxis = list(title = "Historical week"),
        yaxis = list(title = "Sales"),
        legend = list(orientation = "h", x = 0, y = 1.1)
      )
  })
}

shinyApp(ui, server)
