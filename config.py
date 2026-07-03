# ============================================================
# config.py — XEconomics Pipeline Configuration
#
# All scripts import their paths and date settings from here.
# ============================================================

from pathlib import Path

# ----------------------------------------------------------------
# ROOT: Automatically detects your local project folder!
# ----------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------
# Shared Pipeline Configurations
# ----------------------------------------------------------------
START_MONTH  = "2025-06"
END_MONTH    = "2025-08"

start_year = int(START_MONTH[:4])
end_year   = int(END_MONTH[:4])
TARGET_YEARS = list(range(start_year, end_year + 1))

# ----------------------------------------------------------------
# Script 1 — Text Summarization
# ----------------------------------------------------------------
RAW_NEWS_CSV        = DATA_ROOT / "data/All3econnews2017_2026cleaned.csv"
SUMMARIZED_NEWS_CSV = DATA_ROOT / "outputs/news_analysis/All3econnews2017_2026_full_sum.csv"

# ----------------------------------------------------------------
# Script 2 — Aspect-Based Sentiment Analysis (ABSA)
# ----------------------------------------------------------------
ABSA_OUTPUT_DIR     = DATA_ROOT / "outputs/news_analysis"
ABSA_FINAL_CSV      = ABSA_OUTPUT_DIR / "absa_results.csv"

# ----------------------------------------------------------------
# Script 3 — Forecasting Backtest
# ----------------------------------------------------------------
FORECAST_DATA_CSV      = DATA_ROOT / "data/2017-2025.csv"
FORECAST_RESULTS_DIR   = DATA_ROOT / "outputs/forecasting/results_backtest"
FORECAST_ARTIFACTS_DIR = DATA_ROOT / "outputs/forecasting/artifacts_backtest"

# ----------------------------------------------------------------
# Script 4 — Latest Forecast + SHAP (H1-H3)
# ----------------------------------------------------------------
PREDICT_HORIZON       = 3
PREDICT_OUTPUT_DIR    = DATA_ROOT / "outputs/forecasting/predict_latest"
PRED_LATEST_CSV       = PREDICT_OUTPUT_DIR / "pred_latest.csv"
SHAP_IMPORTANCE_CSV   = PREDICT_OUTPUT_DIR / "shap_importance.csv"
SHAP_VALUES_CSV       = PREDICT_OUTPUT_DIR / "shap_values.csv"

# ----------------------------------------------------------------
# Script 5 — Monthly Summary
# ----------------------------------------------------------------
MONTHLY_DIR         = DATA_ROOT / "outputs/monthly_summary"
MONTHLY_CSV         = MONTHLY_DIR / "monthly_summary.csv"

# ----------------------------------------------------------------
# Script 6 — Three-Month Evidence Summary
# ----------------------------------------------------------------
SHAP_CSV            = DATA_ROOT / "data/shap_top3_unique.csv"
PRED_CSV            = DATA_ROOT / "data/pred_direction.csv"
THREEM_DIR          = DATA_ROOT / "outputs/3m_summary"
THREEM_CSV          = THREEM_DIR / "3m_summary.csv"

# ----------------------------------------------------------------
# Script 7 — LLM Reasoning Generation
# ----------------------------------------------------------------
REASONING_DIR       = DATA_ROOT / "outputs/reasoning"
REASONING_CSV       = REASONING_DIR / "reasoning.csv"
REASONING_JSON      = REASONING_DIR / "reasoning.json"