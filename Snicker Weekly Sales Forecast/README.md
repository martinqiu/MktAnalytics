# Snicker Weekly Sales Forecast

This project is a local R Shiny dashboard for planning Snickers weekly sales schemes from week 43 to week 52. Users can adjust planned price, competitor price, display status, markup, and weekly display cost, then submit a scheme to calculate total ten-week sales and profit.

## What The Dashboard Does

- Uses a lagged regression model:
  `log(Sales) ~ log(lagged Sales) + log(Our.price) + log(Comp.price) + Display + log(lagged Our.price) + log(lagged Comp.price)`
- Trains the model using the historical data in `data/snickers.csv`.
- Forecasts total sales for weeks 43 through 52.
- Lets users adjust each week's:
  - `Our.price`
  - `Comp.price`
  - `Display`
- Lets users adjust:
  - markup from 30% to 50%
  - weekly display cost from $500 to $2,000
- Records submitted schemes for comparison.

## Project Structure

- `launch_dashboard.bat`: one-click Windows launcher.
- `install_dependencies.bat`: installs required R packages.
- `planning_dashboard/app.R`: main Shiny app.
- `planning_dashboard/run_app.R`: R launcher for the app.
- `R/forecasting_utils.R`: shared forecasting and backtesting helpers.
- `R/planning_dashboard_utils.R`: lagged regression forecast and scheme profit logic.
- `R/dashboard_helpers.R`: display formatting helpers.
- `data/snickers.csv`: input data file.
- `scripts/install_packages.R`: R package installation script.

## Requirements

Install R before running the dashboard.

Recommended:

- R 4.3 or later
- Windows 10 or later
- Internet access for first-time package installation

Required R packages:

- `shiny`
- `bslib`
- `plotly`
- `DT`

## First-Time Setup

1. Download this project folder or the zip file from GitHub.
2. Extract the zip file if needed.
3. Open the extracted folder.
4. Double-click `install_dependencies.bat`.
5. Wait until R finishes installing packages.

## Launch The Dashboard

After setup, double-click:

```text
launch_dashboard.bat
```

The dashboard opens in your default browser.

If the browser does not open automatically, copy the local URL shown in the command window into your browser.

## Data Location

The dashboard reads data from:

```text
data/snickers.csv
```

To use your own data, replace that file with another CSV using the same column names.

Required columns:

```text
Week
Our.price
Comp.price
Display
Sales
```

Expected format:

- `Week`: numeric week index.
- `Our.price`: numeric selling price.
- `Comp.price`: numeric competitor price.
- `Display`: `1` for feature display on, `0` for off.
- `Sales`: numeric weekly sales.

The included sample data has weeks 1 through 42. The dashboard uses those historical weeks to forecast weeks 43 through 52.

## How To Use

1. Adjust week 43 to week 52 using the arrow controls.
2. Use the up/down arrows to change prices by $0.05.
3. Use display arrows to switch display between `1` and `0`.
4. Adjust markup and weekly display cost under the adjustment panel.
5. Click `Submit scheme`.
6. Review total ten-week sales and profit.
7. Submit additional schemes to compare alternatives.

## Model Estimation Notes

The regression is trained using all available historical observations in `data/snickers.csv`. With the included sample data, the model has weeks 1 through 42 available. Because lagged variables are required, week 1 is used only as a lag source and the regression estimates are fit on weeks 2 through 42.

For the forecast:

- Week 43 uses actual week 42 values as lag inputs.
- Weeks 44 through 52 use recursively forecasted prior sales and the prior planned prices from the submitted scheme.

## Troubleshooting

### Rscript Was Not Found

Install R from:

```text
https://cran.r-project.org/
```

Then either add R to your system PATH or edit `launch_dashboard.bat` and `install_dependencies.bat` to point to your local `Rscript.exe`.

### Missing Packages

Run:

```text
install_dependencies.bat
```

### Data File Not Found

Make sure the data file is located at:

```text
data/snickers.csv
```

### Dashboard Opens But Values Do Not Change

Price and display changes update the proposed scheme only. Sales and profit are calculated only after clicking `Submit scheme`.
