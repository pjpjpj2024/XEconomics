"""Smoke test for the pinned forecasting stack on mock data (CPU only).

Mock CSV follows data/readme.md: date, cci, cpi, gdp, unemployment_rate,
impi, expi, plus the eight news sentiment columns.
Reuses the real functions from forecasting/3.backtest.py.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MOCK_CSV = Path(__file__).resolve().parent / "2017-2025.csv"

sys.path.insert(0, str(REPO))  # for config.py imported by 3.backtest.py
spec = importlib.util.spec_from_file_location("backtest", REPO / "forecasting/3.backtest.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)

from darts import TimeSeries
from darts.models import XGBModel
from darts.metrics import mape
from darts.explainability import ShapExplainer

# ---- 1. mock data ----------------------------------------------------------
rng = np.random.default_rng(0)
dates = pd.date_range("2017-01-01", "2025-12-01", freq="MS")
n = len(dates)
df = pd.DataFrame({"date": dates.strftime("%Y-%m")})
df["cci"] = 50 + np.cumsum(rng.normal(0, 1, n))
for c in bt.MACRO_COLS:
    df[c] = 100 + np.cumsum(rng.normal(0, 1, n))
for c in bt.NEWS_COLS:
    df[c] = rng.uniform(-1, 1, n).round(4)
df.to_csv(MOCK_CSV, index=False, encoding="utf-8-sig")
print(f"mock csv: {MOCK_CSV} rows={len(df)} cols={len(df.columns)}")

# ---- 2. real loader from 3.backtest ----------------------------------------
bt.DATA_CSV = MOCK_CSV
data = bt.load_data()
assert len(data) == n, "loader dropped rows unexpectedly"

feature_cols = bt.MACRO_COLS + bt.NEWS_COLS
target, past_cov = bt.make_series(data, feature_cols)

# ---- 3. darts models, one horizon, one param combo each --------------------
horizon = 2
checks = {
    "XGBModel": {"lags": 3, "lags_past_covariates": 3, "n_estimators": 50,
                 "max_depth": 3, "learning_rate": 0.1, "output_chunk_length": horizon},
    "BlockRNNModel": {"model": "LSTM", "input_chunk_length": 3, "hidden_dim": 8,
                      "n_rnn_layers": 1},
    "NHiTSModel": {"input_chunk_length": 3, "layer_widths": 8},
}
for name, params in checks.items():
    full = bt.build_dl_params(name, params, horizon) if name in bt.DL_MODELS else params
    if name in bt.DL_MODELS:
        full["n_epochs"] = 2
    res = bt.evaluate_darts(name, full, target, past_cov, horizon, name in bt.DL_MODELS)
    assert res is not None, f"{name}: evaluate_darts returned None"
    metrics, pred_df = res
    print(f"{name:14s} OK  mape={metrics['mape']} rmse={metrics['rmse']} rows={len(pred_df)}")

# ---- 4. ARIMAX (statsmodels SARIMAX) ---------------------------------------
y, exog, dates_np = bt.make_numpy(data, feature_cols)
res = bt.evaluate_arimax({"p": 1, "d": 1, "q": 0}, y, exog, dates_np, horizon)
assert res[0] is not None, "ARIMAX returned no metrics"
print(f"ARIMAX         OK  mape={res[0]['mape']} rmse={res[0]['rmse']} rows={len(res[1])}")

# ---- 5. script 4 path: XGB gridsearch + ShapExplainer (mirrors 4.predict_latest.py)
_, best_params, metric = XGBModel.gridsearch(
    parameters={"lags": [2, 3], "lags_past_covariates": [2], "n_estimators": [50],
                "max_depth": [3], "learning_rate": [0.1], "output_chunk_length": [3]},
    series=target, past_covariates=past_cov, forecast_horizon=3, stride=1,
    start=0.8, last_points_only=True, metric=mape, verbose=False,
)
model = XGBModel(**best_params)
model.fit(series=target, past_covariates=past_cov)
pred = model.predict(n=3, series=target, past_covariates=past_cov)
explainer = ShapExplainer(model=model, background_series=target, background_past_covariates=past_cov)
explaination = explainer.explain(foreground_series=target, foreground_past_covariates=past_cov)
shap_exp = explaination.get_shap_explanation_object(horizon=1)
print(f"script4 path   OK  best={best_params} forecast={pred.values().ravel().round(2).tolist()} "
      f"shap_shape={shap_exp.values.shape}")

print("\nSMOKE TEST PASSED")
