import os
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from src.ai import model_persistence, explainability
from src.transform import feature_engineering
from src.web import recommendation, llm_explainer, scenario_analysis

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")

FEATURES_FILE = os.environ.get(
    "FEATURES_FILE", os.path.join("data", "3_features", "event_driven_features.csv"))
MODEL_FILE = os.environ.get("MODEL_FILE", model_persistence.DEFAULT_MODEL_FILE)
RAW_EARNINGS_FILE = os.environ.get(
    "RAW_EARNINGS_FILE", os.path.join("data", "1_raw", "earnings", "sp500_historical_earnings.csv"))
PRICES_DIR = os.environ.get("PRICES_DIR", os.path.join("data", "1_raw", "prices"))
METADATA_FILE = os.environ.get(
    "METADATA_FILE", os.path.join("data", "2_processed", "stock_metadata.csv"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.features_df = pd.read_csv(FEATURES_FILE)
    app.state.model_bundle = model_persistence.load_model(MODEL_FILE)
    app.state.raw_earnings_df = pd.read_csv(RAW_EARNINGS_FILE)
    app.state.snapshot_df = feature_engineering.generate_latest_price_snapshot(
        prices_dir=PRICES_DIR, metadata_file=METADATA_FILE)
    yield


app = FastAPI(lifespan=lifespan)


def get_features_df(request: Request):
    return request.app.state.features_df


def get_model_bundle(request: Request):
    return request.app.state.model_bundle


def get_raw_earnings_df(request: Request):
    return request.app.state.raw_earnings_df


def get_snapshot_df(request: Request):
    return request.app.state.snapshot_df


@app.get("/api/health")
def health(bundle=Depends(get_model_bundle), features_df=Depends(get_features_df)):
    return {
        "status": "ok",
        "model_run_id": bundle['run_id'],
        "model_trained_at": bundle['saved_at'],
        "spike_threshold": bundle['config'].get('spike_threshold'),
        "features_rows": len(features_df),
        "features_date_range": [str(features_df['Date'].min()), str(features_df['Date'].max())],
    }


@app.get("/api/tickers")
def list_tickers(features_df=Depends(get_features_df)):
    latest_sector = features_df.sort_values('Date').groupby('Symbol')['Sector'].last()
    return sorted(
        [{"symbol": sym, "sector": sec} for sym, sec in latest_sector.items()],
        key=lambda r: r["symbol"])


@app.get("/api/recommend/{ticker}")
def recommend(ticker: str, features_df=Depends(get_features_df), bundle=Depends(get_model_bundle)):
    row = recommendation.find_latest_event_row(features_df, ticker)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No earnings-event data found for '{ticker.strip().upper()}' in the current dataset.")

    rec = recommendation.build_recommendation(
        row, bundle['model'], bundle['feature_cols'],
        spike_threshold=bundle['config'].get('spike_threshold'))
    explanation, source = llm_explainer.generate_explanation(rec)

    X = recommendation.feature_row_to_frame(row, bundle['feature_cols'])
    shap_values = explainability.compute_shap_values(bundle['model'], X)

    return {**rec, "explanation": explanation, "explanation_source": source,
            "shap_values": shap_values}


@app.get("/api/upcoming-earnings")
def upcoming_earnings(raw_earnings_df=Depends(get_raw_earnings_df), snapshot_df=Depends(get_snapshot_df)):
    upcoming = scenario_analysis.find_upcoming_earnings(raw_earnings_df, snapshot_df)
    sector_by_symbol = dict(zip(snapshot_df['Symbol'], snapshot_df['Sector']))

    return [
        {
            "symbol": row['Symbol'],
            "date": str(pd.Timestamp(row['Date']).date()),
            "sector": sector_by_symbol.get(row['Symbol']),
            "eps_estimate": row['EPS Estimate'],
        }
        for _, row in upcoming.iterrows()
    ]


@app.get("/api/upcoming-earnings/{ticker}/scenario")
def upcoming_earnings_scenario(ticker: str, raw_earnings_df=Depends(get_raw_earnings_df),
                                snapshot_df=Depends(get_snapshot_df), bundle=Depends(get_model_bundle)):
    ticker = ticker.strip().upper()
    eps_row = scenario_analysis.find_upcoming_earnings_row(raw_earnings_df, ticker)
    snapshot_row = scenario_analysis.find_snapshot_row(snapshot_df, ticker)

    if eps_row is None or snapshot_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No upcoming-earnings scenario available for '{ticker}'.")

    result = scenario_analysis.build_scenario_analysis(
        snapshot_row, float(eps_row['EPS Estimate']), bundle['model'], bundle['feature_cols'],
        spike_threshold=bundle['config'].get('spike_threshold'))

    return {
        "ticker": ticker,
        "upcoming_earnings_date": str(pd.Timestamp(eps_row['Date']).date()),
        "as_of_date": str(pd.Timestamp(snapshot_row['Date']).date()),
        "sector": snapshot_row.get('Sector'),
        **result,
    }


# Mounted last so it never shadows the /api/* routes above. Guarded since
# frontend/dist only exists after `npm run build` - tests and API-only dev
# workflows shouldn't require it.
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
