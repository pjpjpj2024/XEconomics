# Data Directory

The original datasets used in this paper are not included in this repository due to data privacy and copyright restrictions on the scraped news content. To run the pipeline, populate this directory with your own data conforming to the formats below.

## `All3econnews2017_2026cleaned.csv`
Scraped economic news articles with a publication date column and a text body or summary column.

## `2017-2025.csv`
Monthly macroeconomic time-series: `date`, `cci`, `cpi`, `gdp`, `unemployment_rate`, `impi`, `expi`, plus the eight ABSA-derived news sentiment columns.

## `pred_direction.csv`
Forecast output. Columns: `date` (`YYYY-MM`), `predicted` (float), `direction` (string, e.g. increase/decrease/stable).

`direction` is computed by comparing the forecast against a reference point: for H=1, compare against the previous month's actual CCI value; for H=2 and H=3, compare against the previous month's forecast rather than an actual value, since the actual is not yet available at inference time.

## `shap_top3_unique.csv`
SHAP feature importance per month. Columns: a date column (`date`/`month`/`target_month`), a feature name column, and a SHAP value column (`SHAP_Value` or `shap_value`).

---

Any forecasting model or feature-attribution method can be substituted here, as XEconomics' components are independently interchangeable — the downstream pipeline only depends on the column formats above, not on how they were produced.