# XEconomics: An Explainable System for News-Driven Economic Forecasting

> Jarusawee, P. et al. (2026) — manuscript under review.

## Requirements

### Environment

- Python 3.10, 3.11, 3.12 or 3.13 (tested on all four)
- [Ollama](https://ollama.com) (for local LLM inference)
- A machine with enough GPU VRAM or system RAM to run your local LLM model

> **Python 3.14 is not supported.** `darts` depends on `llvmlite`, which has no Python 3.14 wheel and fails to build from source. Use one of the versions above.

#### Install with uv (recommended)

Install [uv](https://docs.astral.sh/uv/), then create the environment. uv downloads the requested Python version if it is not installed:

| Python | Command |
| --- | --- |
| 3.10 | `uv sync --python 3.10` |
| 3.11 | `uv sync --python 3.11` |
| 3.12 | `uv sync --python 3.12` |
| 3.13 | `uv sync --python 3.13` |

This creates `.venv/` from `uv.lock`. Run scripts with `uv run`, for example `uv run python forecasting/3.backtest.py`.

#### Install with pip

Use a Python interpreter you have already installed. Replace `python3.10` with `python3.11`, `python3.12` or `python3.13` as needed:

| Python | Command |
| --- | --- |
| 3.10 | `python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| 3.11 | `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| 3.12 | `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| 3.13 | `python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |

`requirements.txt` and `pyproject.toml` pin the same versions for each Python version.

Pull the required Ollama model (e.g. `gemma2:27b`):

```bash
ollama pull gemma2:27b
```

This is used for LLM inference in scripts 2, 5, 6, and 7.

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
@unpublished{jarusawee2026xeconomics,
  title={XEconomics: An Explainable System for News-Driven Economic Forecasting},
  author={Jarusawee, P. and et al.},
  notes={Manuscript under review},
  year={2026}
}
```
