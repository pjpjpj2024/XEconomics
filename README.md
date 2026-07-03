# XEconomics: An Explainable System for News-Driven Economic Forecasting

> Jarusawee, P. et al. (2026) — manuscript under review.

## Requirements

```bash
pip install -r requirement.txt
```

This default framework requires [Ollama](https://ollama.com) running locally for LLM inference (scripts 2, 5, 6, 7) using the `gemma2:27b` model.

---

## Data Replication Schema

To comply with data privacy and copyright restrictions regarding web-scraped economic news, the original evaluation datasets are omitted from this repository.

To replicate this forecasting architecture or train the pipeline on an alternative market, populate your local `data/` directory with your own datasets conforming to the following structural naming signatures:

1. `data/All3econnews2017_2026cleaned.csv` — Scraped economic news articles containing at least a text body or summary column.
2. `data/2017-2025.csv` — Historical macroeconomic time-series tracking indicators (e.g., CPI, GDP, Unemployment) aligned by month.
3. `data/shap_top3_unique.csv` — Logged SHAP framework local feature weights calculated from your forecasting model.
4. `data/pred_direction.csv` — Categorical directional forecasting targets (`direction`) and raw predicted values (`predicted`).

See `data/README.md` for exact column schemas and formatting requirements for each file.

---

## Pipeline

Run the scripts sequentially in order:

| Script | Description |
| --- | --- |
| `news_analysis/1.text_summarization.py` | Summarize raw news articles using local translation architectures. |
| `news_analysis/2.absa.py` | Aspect-Based Sentiment Analysis categorization using LLM. |
| `forecasting/3.backtest.py` | Train and evaluate time-series forecasting models. |
| `forecasting/4.predict_latest.py` | Generate the latest forecast and SHAP feature importances. |
| `explainable/5.monthly_summary.py` | Cluster positive and negative monthly narratives per economic domain aspect. |
| `explainable/6.3m_summary.py` | Generate consolidated 3-month lookback evidence text per feature. |
| `explainable/7.llm_reasoning_oneshot.py` | Synthesize final structured senior economist reasoning reports. |

---

## Configuration

All local data input mappings, pipeline checkpoint dumps, target calculation directories, and global evaluation date parameters are centrally managed within `config.py`.

The configuration layer utilizes dynamic resolution logic (`DATA_ROOT = Path(__file__).resolve().parent`). You do not need to manually configure directory absolute string variations unless adjusting baseline training frames or targeting alternative text window spans.

---

## Citation

```bibtex
@article{jarusawee2026xeconomics,
  title={XEconomics: An Explainable System for News-Driven Economic Forecasting},
  author={Jarusawee, P. and et al.},
  journal={Manuscript under review},
  year={2026}
}
```