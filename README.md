# Event-Driven AI Financial Advisor Bot

An earnings-event-driven stock spike predictor (XGBoost) with a FastAPI
backend and React frontend, plus a local-LLM (Ollama) explanation layer.
Given a ticker, it looks up that company's most recent earnings event and
predicts whether the following trading day was/would be a "Spike" (a
price move past a configured threshold), with a SHAP-based feature
breakdown and a plain-English explanation.

The full experimental record behind the current model choice - including
several results that looked promising at first and turned out to be
artefacts once checked properly - lives in
[`notebooks/experiment_report.ipynb`](notebooks/experiment_report.ipynb).
That notebook, not this README, is the place to look for performance
numbers, methodology, and known limitations.

## Prerequisites

- Python 3.14 (a `venv/` is expected at the project root - see Setup)
- Node.js 24+ / npm 11+ (for the frontend)
- An [EODHD](https://eodhd.com/) API key (historical price data)
- A SEC EDGAR-compliant `User-Agent` string (sector metadata lookup) -
  see [SEC's fair access guidance](https://www.sec.gov/os/webmaster-faq#developers)
- [Ollama](https://ollama.com/) running locally, with the `llama3.2` model
  pulled - **optional**: if unreachable, the app falls back to a
  deterministic template explanation instead of an LLM one, so the rest
  of the app still works without it

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

cd frontend && npm install && cd ..
```

Create a `.env` file at the project root (this file is gitignored - never
commit real API keys):

```
EODHD_API_KEY=<your key>
SEC_EDGAR_USER_AGENT=<your name/organisation> <your email>
```

## 1. Build the dataset

`data/` is gitignored project-wide (raw price/earnings data is large and,
in EODHD's case, licensed - it must never be committed), so nothing under
it survives a `git clone`. It has to be regenerated locally:

```bash
./venv/bin/python scripts/run_ingest_pipeline.py
```

This downloads full daily price history (1990-present) for every current
and historical S&P 500 constituent (~600+ tickers) via EODHD, plus sector
metadata and earnings history. **This can take a while** depending on
EODHD's rate limits - each stage is independently resumable, so re-running
after a partial failure skips files already downloaded rather than
re-fetching them.

## 2. Train and save the production model

```bash
./venv/bin/python scripts/train_and_save_model.py
```

Regenerates the engineered feature table, trains the current production
model configuration, backtests it, and saves the fitted model to
`data/5_models/production_model.joblib` (plus a human-readable
`production_model_metadata.json` sidecar next to it). The web API loads
this file **once at startup** - see "Verifying the model in use" below.

## 3. Run the app

**Simplest path (single server, what a marker should normally use):**

```bash
cd frontend && npm run build && cd ..
./venv/bin/uvicorn src.web.app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` - the backend serves the built frontend
directly.

**Frontend dev mode (hot reload, for active development):** run the
backend as above, then in a second terminal:

```bash
cd frontend && npm run dev
```

Vite's dev server proxies `/api/*` requests to `http://localhost:8000`
(see `frontend/vite.config.ts`), so the backend must already be running
on port 8000 for this to work.

## Verifying the model in use

The backend only loads `data/5_models/production_model.joblib` **once, at
process startup** - overwriting that file (e.g. by re-running step 2)
does **not** update an already-running server; it must be restarted.

To confirm which model a running server actually has loaded:

```bash
curl http://localhost:8000/api/health
```

Check that `model_run_id` and `model_trained_at` match the `Run <run_id>`
line printed by `train_and_save_model.py` when it last saved a model. You
can also inspect `data/5_models/production_model_metadata.json` directly
(no server required) to see the full config/metrics of whatever is
currently saved on disk.

## Running the tests

```bash
./venv/bin/python -m pytest tests/ -v
```

Coverage is measured and enforced automatically (`pytest.ini`'s `addopts`
runs `pytest-cov` against `src/` on every invocation, `.coveragerc`
excludes script `if __name__ == "__main__":` entry points from the count
so the number reflects real logic rather than CLI boilerplate). The run
fails if coverage drops below 90% (`--cov-fail-under=90`) - currently at
~98%. No separate command needed; this runs as part of the command above.

## Reproducing the experiments / report

`scripts/experiment_*.py` are one-off, ad-hoc experiment scripts (each
with a docstring explaining its hypothesis and how to run it) - they are
not part of the app's runtime path and are safe to ignore unless you want
to reproduce a specific result. `notebooks/experiment_report.ipynb`
collects the full experiment log (`data/4_experiments/experiment_log.csv`)
into tables/charts and is the canonical write-up of what was tried, what
worked, and what didn't.
