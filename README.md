# Time-Series Forecasting Project

**Daily Minimum Temperature Forecasting for Heating Planning**

Group: Taha Shabbir & Sivakumar Vijayan

## Dataset
- Melbourne daily minimum temperatures, 1981–1990
- Source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
- `daily_min_temp.csv` is the raw file (3,650 rows). `daily_min_temp_clean.csv` adds the two missing dates
  (1984-12-31 and 1988-12-31) by linear interpolation, giving 3,652 rows on a regular daily index.

## How to reproduce
```bash
pip install -r requirements.txt
python full_analysis.py      # about 5 minutes on a laptop CPU; writes plots/ and results/
node build_report.js         # optional: rebuilds the Word report (needs `npm install docx`)
```
The run is seeded (`SEED = 42`, TensorFlow op determinism on), so it regenerates the same numbers.

## Project structure
- `full_analysis.py` – data analysis, all models, tuning, quantile LSTMs, cost analysis, plots
- `build_report.js` – builds `Time_Series_Forecasting_Report.docx` from `results/summary.json` and `plots/`
- `plots/` – all figures used in the report
- `results/model_metrics.csv` – test-set metrics for every model
- `results/decision_costs.csv` – heating cost for every planning rule
- `results/test_predictions.csv` – actual values and every model's forecast for each test day
- `results/summary.json` – every number quoted in the report
- `Time_Series_Forecasting_Report.docx` – the report

## Method
- Chronological split: 70% train (2,556 days), 15% validation (548), 15% test (548). No shuffling.
- Every model makes one-step-ahead forecasts on the same test window. Classical models are fitted on the
  training set and their coefficients are then frozen.
- RNN and LSTM settings (lag, units, layers, learning rate, batch size) are chosen on the validation set.

## Key results (test set, one-step-ahead)
| Model | RMSE (°C) | MAE (°C) | MAPE | R² |
|---|---|---|---|---|
| Climatology | 3.90 | 3.24 | 38.1% | -0.000 |
| Persistence | 2.57 | 2.04 | 22.4% | 0.565 |
| MA(7) | 2.45 | 1.95 | 22.4% | 0.605 |
| AR(14) | 2.24 | 1.75 | 19.6% | 0.671 |
| ARIMA(2,1,1) | 2.23 | 1.75 | 19.3% | 0.672 |
| SARIMA(1,1,1)(1,0,1,7) | 2.25 | 1.76 | 19.5% | 0.666 |
| ARIMA(2,0,1) + annual Fourier | 2.19 | 1.72 | 19.0% | 0.684 |
| RNN | 2.24 | 1.74 | 19.2% | 0.669 |
| LSTM | 2.24 | 1.75 | 19.1% | 0.671 |

- 95% prediction interval (two quantile LSTMs, pinball loss): PICP = 96.2%, MPIW = 10.08 °C
- Heating cost over the test period: $7,275 with no forecast,
  $6,380 with the recommended model
  (saving $895)

## Decision framing
Each afternoon the facility books heating for the coming night from the forecast minimum temperature.
Planned heat costs $12 per degree below 8 °C; any shortfall is covered at 2.5 times that rate.
The ARIMA(2,0,1) model with annual sine and cosine terms is recommended for the point forecast.
Lower quantiles of the LSTM interval are for nights when a shortfall would be much more costly than usual.

## AI disclaimer
[Edit to match what you actually did.] AI tools were used to write and debug the analysis code, to produce
the figures, and to help draft and edit the report. The authors chose the dataset, decision problem and cost
assumptions, reviewed the results, and are responsible for the conclusions.
