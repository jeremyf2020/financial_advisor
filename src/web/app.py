import os
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from src.ai import model_persistence, explainability
from src.web import recommendation, llm_explainer

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")

FEATURES_FILE = os.environ.get(
    "FEATURES_FILE", os.path.join("data", "3_features", "event_driven_features.csv"))
MODEL_FILE = os.environ.get("MODEL_FILE", model_persistence.DEFAULT_MODEL_FILE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.features_df = pd.read_csv(FEATURES_FILE)
    app.state.model_bundle = model_persistence.load_model(MODEL_FILE)
    yield


app = FastAPI(lifespan=lifespan)


def get_features_df(request: Request):
    return request.app.state.features_df


def get_model_bundle(request: Request):
    return request.app.state.model_bundle


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


# Mounted last so it never shadows the /api/* routes above. Guarded since
# frontend/dist only exists after `npm run build` - tests and API-only dev
# workflows shouldn't require it.
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
