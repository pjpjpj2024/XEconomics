"""
Estimate total runtime for forecasting/3.backtest.py on this machine.
Runs one sample fit per model type, then extrapolates.
"""

import os
import sys
import time
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import BlockRNNModel, NHiTSModel, XGBModel
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config

warnings.filterwarnings("ignore")
seed_everything(42, workers=True)

# ── Load data (same as 3.backtest.py) ──

DATE_COL = "date"
TARGET_COL = "cci"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
MACRO_COLS = ["cpi", "gdp", "unemployment_rate", "impi", "expi"]
NEWS_COLS = [
    "การเมือง", "ภัยพิบัติ/โรคระบาด", "มาตรการของรัฐ",
    "ราคาน้ำมันเชื้อเพลิง", "ราคาสินค้าเกษตร", "สังคม/ความมั่นคง",
    "เศรษฐกิจโลก", "เศรษฐกิจไทย",
]

df = pd.read_csv(config.FORECAST_DATA_CSV)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)
for col in df.columns:
    if col != DATE_COL:
        df[col] = pd.to_numeric(df[col], errors="coerce")
all_cols = [DATE_COL, TARGET_COL] + MACRO_COLS + NEWS_COLS
df = df[[c for c in all_cols if c in df.columns]].dropna().reset_index(drop=True)
news_in_df = [c for c in NEWS_COLS if c in df.columns]
if news_in_df:
    df = df[(df[news_in_df] != 0).any(axis=1)].reset_index(drop=True)

n = len(df)
feature_cols = [c for c in MACRO_COLS + NEWS_COLS if c in df.columns]
print(f"Data: {n} rows")

# ── Prepare series ──

target = TimeSeries.from_dataframe(df, time_col=DATE_COL, value_cols=TARGET_COL, freq="MS")
past_cov = TimeSeries.from_dataframe(df, time_col=DATE_COL, value_cols=feature_cols, freq="MS")

y = df[TARGET_COL].values.astype(float)
exog = df[feature_cols].values.astype(float)
dates = df[DATE_COL].values

test_start_idx = int(n * (TRAIN_RATIO + VAL_RATIO))
n_test_points = n - test_start_idx
horizon = 3

FEATURE_SETS = {
    "cci_only": [],
    "macro": MACRO_COLS,
    "news": NEWS_COLS,
    "macro_plus_news": MACRO_COLS + NEWS_COLS,
}
N_FEATURE_SETS = len(FEATURE_SETS)
N_HORIZONS = 6

# ── Grid sizes ──

GRIDS = {
    "ARIMAX": 13 * 3 * 13,  # 507
    "XGBModel_cov": 12 * 12 * 2 * 2 * 2,  # 1152
    "XGBModel_nocov": 12 * 2 * 2 * 2,  # 96
    "BlockRNNModel": 12 * 2 * 1,  # 24
    "NHiTSModel": 12 * 2,  # 24
}


def bench_arimax():
    """Time one ARIMAX expanding-window backtest (one param combo, one horizon)."""
    t0 = time.time()
    for t in range(test_start_idx, len(y) - horizon + 1):
        try:
            fit = SARIMAX(
                y[:t], exog=exog[:t], order=(1, 1, 1),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            fit.forecast(steps=horizon, exog=exog[t:t + horizon])
        except Exception:
            pass
    return time.time() - t0


def bench_xgb():
    """Time one XGBModel historical_forecasts (one param combo, one horizon)."""
    model = XGBModel(
        lags=6, lags_past_covariates=6,
        n_estimators=100, max_depth=3, learning_rate=0.1,
        output_chunk_length=horizon, random_state=42,
    )
    t0 = time.time()
    model.historical_forecasts(
        series=target, past_covariates=past_cov,
        forecast_horizon=horizon, stride=1,
        start=target.time_index[test_start_idx],
        retrain=True, last_points_only=True, verbose=False, show_warnings=False,
    )
    return time.time() - t0


def bench_lstm():
    """Time one BlockRNNModel historical_forecasts (one param combo, one horizon)."""
    model = BlockRNNModel(
        model="LSTM", input_chunk_length=6, hidden_dim=32, n_rnn_layers=1,
        output_chunk_length=horizon, n_epochs=100, dropout=0.1,
        use_reversible_instance_norm=True, random_state=42,
        pl_trainer_kwargs={
            "accelerator": config.LIGHTNING_ACCELERATOR, "devices": 1,
            "strategy": "auto", "enable_progress_bar": False,
            "enable_model_summary": False, "logger": False,
            "enable_checkpointing": False,
            "callbacks": [EarlyStopping(monitor="val_loss", patience=5, min_delta=0.001, mode="min")],
        },
    )
    t0 = time.time()
    model.historical_forecasts(
        series=target, past_covariates=past_cov,
        forecast_horizon=horizon, stride=1,
        start=target.time_index[test_start_idx],
        retrain=True, last_points_only=True, verbose=False, show_warnings=False,
        val_length=int(n * VAL_RATIO),
        data_transformers={"series": Scaler(), "past_covariates": Scaler()},
    )
    return time.time() - t0


def bench_nhits():
    """Time one NHiTSModel historical_forecasts (one param combo, one horizon)."""
    model = NHiTSModel(
        input_chunk_length=6, layer_widths=32,
        output_chunk_length=horizon, n_epochs=100, dropout=0.1,
        use_reversible_instance_norm=True, random_state=42,
        pl_trainer_kwargs={
            "accelerator": config.LIGHTNING_ACCELERATOR, "devices": 1,
            "strategy": "auto", "enable_progress_bar": False,
            "enable_model_summary": False, "logger": False,
            "enable_checkpointing": False,
            "callbacks": [EarlyStopping(monitor="val_loss", patience=5, min_delta=0.001, mode="min")],
        },
    )
    t0 = time.time()
    model.historical_forecasts(
        series=target, past_covariates=past_cov,
        forecast_horizon=horizon, stride=1,
        start=target.time_index[test_start_idx],
        retrain=True, last_points_only=True, verbose=False, show_warnings=False,
        val_length=int(n * VAL_RATIO),
        data_transformers={"series": Scaler(), "past_covariates": Scaler()},
    )
    return time.time() - t0


# ── Run benchmarks ──

print(f"\nTest window: {n_test_points} points, sample horizon: {horizon}")
print(f"Feature sets: {N_FEATURE_SETS}, Horizons: {N_HORIZONS}")
print("=" * 60)

results = {}

print("\nBenchmarking ARIMAX (1 combo)...")
t = bench_arimax()
results["ARIMAX"] = t
print(f"  1 combo = {t:.1f}s")

print("\nBenchmarking XGBModel (1 combo)...")
t = bench_xgb()
results["XGBModel"] = t
print(f"  1 combo = {t:.1f}s")

print("\nBenchmarking BlockRNNModel/LSTM (1 combo)...")
t = bench_lstm()
results["BlockRNNModel"] = t
print(f"  1 combo = {t:.1f}s")

print("\nBenchmarking NHiTSModel (1 combo)...")
t = bench_nhits()
results["NHiTSModel"] = t
print(f"  1 combo = {t:.1f}s")

# ── Extrapolate ──

print("\n" + "=" * 60)
print("ESTIMATED TOTAL RUNTIME")
print("=" * 60)

# ARIMAX: 507 combos × 4 feature sets × 6 horizons
arimax_total = results["ARIMAX"] * GRIDS["ARIMAX"] * N_FEATURE_SETS * N_HORIZONS

# XGBModel: 3 feature sets with cov (1152 each) + 1 without (96) × 6 horizons
xgb_total = results["XGBModel"] * (
    GRIDS["XGBModel_cov"] * 3 + GRIDS["XGBModel_nocov"]
) * N_HORIZONS

# DL models: 24 combos × 4 feature sets × 6 horizons
lstm_total = results["BlockRNNModel"] * GRIDS["BlockRNNModel"] * N_FEATURE_SETS * N_HORIZONS
nhits_total = results["NHiTSModel"] * GRIDS["NHiTSModel"] * N_FEATURE_SETS * N_HORIZONS

grand_total = arimax_total + xgb_total + lstm_total + nhits_total

def fmt(secs):
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    if secs < 86400:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 86400:.1f}d"

rows = [
    ("ARIMAX", GRIDS["ARIMAX"] * N_FEATURE_SETS * N_HORIZONS, results["ARIMAX"], arimax_total),
    ("XGBModel", (GRIDS["XGBModel_cov"] * 3 + GRIDS["XGBModel_nocov"]) * N_HORIZONS, results["XGBModel"], xgb_total),
    ("LSTM", GRIDS["BlockRNNModel"] * N_FEATURE_SETS * N_HORIZONS, results["BlockRNNModel"], lstm_total),
    ("NHiTS", GRIDS["NHiTSModel"] * N_FEATURE_SETS * N_HORIZONS, results["NHiTSModel"], nhits_total),
]

print(f"\n{'Model':<12} {'Combos':>8} {'Per combo':>10} {'Total':>10}")
print("-" * 44)
for name, combos, per, total in rows:
    print(f"{name:<12} {combos:>8,} {fmt(per):>10} {fmt(total):>10}")
print("-" * 44)
print(f"{'GRAND TOTAL':<12} {'':>8} {'':>10} {fmt(grand_total):>10}")
print(f"\nNote: early stopping may reduce DL times. Cache skips completed combos.")
print(f"      This estimate assumes no cache hits (fresh run).")
