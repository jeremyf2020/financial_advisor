import pytest
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient
from xgboost import XGBClassifier
from src.ai import model_persistence
from src.web.app import app, get_features_df, get_model_bundle, get_raw_earnings_df, get_snapshot_df

FEATURE_COLS = ['EPS Estimate', 'Reported EPS', 'Surprise(%)', 'Return_60d', 'Sector_Rank_60d']


def make_raw_earnings_df():
    # Far-future dates so these stay "upcoming" regardless of when the test suite runs
    return pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT'],
        'Date': ['2099-12-01', '2099-12-15'],
        'EPS Estimate': [1.8, 2.0],
        'Reported EPS': [float('nan'), float('nan')],
        'Surprise(%)': [float('nan'), float('nan')],
    })


def make_snapshot_df():
    return pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT'],
        'Date': ['2099-11-20', '2099-11-20'],
        'Sector': ['Technology', 'Technology'],
        'Return_60d': [0.05, 0.02],
        'Sector_Rank_60d': [0.8, 0.5],
    })


def make_features_df():
    return pd.DataFrame({
        'Date': ['2024-01-01', '2024-07-01', '2023-01-01'],
        'Symbol': ['AAPL', 'AAPL', 'MSFT'],
        'Sector': ['Technology', 'Technology', 'Technology'],
        'EPS Estimate': [1.0, 1.2, 2.0],
        'Reported EPS': [1.1, 1.3, 2.1],
        'Surprise(%)': [10.0, 8.0, 5.0],
        'Return_60d': [0.05, 0.02, 0.01],
        'Sector_Rank_60d': [0.8, 0.6, 0.5],
        'Target_T1_Close_Ret': [0.03, 0.09, 0.01],
        'Target_T1_High_Ret': [0.05, 0.12, 0.02],
    })


def make_model_bundle():
    X = pd.DataFrame({
        'EPS Estimate': [1.0, 1.2, 2.0, 1.5],
        'Reported EPS': [1.1, 1.3, 2.1, 1.4],
        'Surprise(%)': [10.0, 8.0, 5.0, -5.0],
        'Return_60d': [0.05, 0.02, 0.01, -0.02],
        'Sector_Rank_60d': [0.8, 0.6, 0.5, 0.2],
    })
    y = [1, 1, 0, 0]
    model = XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    return {
        'model': model, 'feature_cols': FEATURE_COLS, 'target_col': 'Target_Spike_Class',
        'config': {'spike_threshold': 0.07}, 'run_id': 'test_run', 'saved_at': '2026-01-01T00:00:00',
        'metrics': {'accuracy': 0.9},
    }


@pytest.fixture
def client():
    app.dependency_overrides[get_features_df] = make_features_df
    app.dependency_overrides[get_model_bundle] = make_model_bundle
    app.dependency_overrides[get_raw_earnings_df] = make_raw_earnings_df
    app.dependency_overrides[get_snapshot_df] = make_snapshot_df
    with patch('src.web.app.llm_explainer.generate_explanation',
               return_value=("mocked explanation", "template_fallback")):
        yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body['model_run_id'] == 'test_run'
    assert body['spike_threshold'] == 0.07
    assert body['features_rows'] == 3


def test_list_tickers(client):
    response = client.get("/api/tickers")

    assert response.status_code == 200
    symbols = [row['symbol'] for row in response.json()]
    assert symbols == ['AAPL', 'MSFT']


def test_recommend_known_ticker(client):
    response = client.get("/api/recommend/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body['ticker'] == 'AAPL'
    assert body['as_of_date'] == '2024-07-01'  # the later of AAPL's two rows
    assert body['predicted_label'] in ('No Spike', 'Spike')
    assert 'explanation' in body
    assert body['explanation_source'] == 'template_fallback'
    assert set(body['shap_values'].keys()) == set(FEATURE_COLS)


def test_recommend_unknown_ticker_returns_404(client):
    response = client.get("/api/recommend/NOPE")

    assert response.status_code == 404
    assert 'NOPE' in response.json()['detail']


def test_list_upcoming_earnings(client):
    response = client.get("/api/upcoming-earnings")

    assert response.status_code == 200
    body = response.json()
    assert [row['symbol'] for row in body] == ['AAPL', 'MSFT']
    assert body[0]['date'] == '2099-12-01'
    assert body[0]['sector'] == 'Technology'
    assert body[0]['eps_estimate'] == 1.8


def test_upcoming_earnings_scenario(client):
    response = client.get("/api/upcoming-earnings/AAPL/scenario")

    assert response.status_code == 200
    body = response.json()
    assert body['ticker'] == 'AAPL'
    assert body['upcoming_earnings_date'] == '2099-12-01'
    assert body['as_of_date'] == '2099-11-20'
    assert body['eps_estimate'] == 1.8
    assert body['spike_threshold'] == 0.07
    assert body['context_features']['Return_60d'] == 0.05
    assert len(body['scenarios']) > 1
    for scenario in body['scenarios']:
        assert scenario['predicted_label'] in ('No Spike', 'Spike')
        assert 0.0 <= scenario['spike_probability'] <= 1.0


def test_upcoming_earnings_scenario_unknown_ticker_returns_404(client):
    response = client.get("/api/upcoming-earnings/NOPE/scenario")

    assert response.status_code == 404
    assert 'NOPE' in response.json()['detail']


def test_lifespan_loads_real_features_and_model_from_disk(tmp_path, monkeypatch):
    """ Every other test bypasses the real lifespan() startup via
    dependency_overrides (get_features_df/get_model_bundle/etc. are swapped
    out entirely). This exercises the actual startup path - reading every
    *_FILE/*_DIR global from disk into app.state - the same way the real
    app does when uvicorn boots it, using real (tiny) files on disk rather
    than mocks. These are module-level globals read once at import time, so
    they're monkeypatched directly rather than via environment variables,
    which lifespan() would no longer see post-import. """
    features_path = tmp_path / "features.csv"
    make_features_df().to_csv(features_path, index=False)

    model_path = tmp_path / "model.joblib"
    bundle = model_persistence.build_model_bundle(
        model=make_model_bundle()['model'], feature_cols=FEATURE_COLS,
        target_col='Target_Spike_Class', config={'spike_threshold': 0.07},
        run_id='real_lifespan_test')
    model_persistence.save_model(
        bundle, model_file=str(model_path),
        metadata_file=str(tmp_path / "model_metadata.json"))

    raw_earnings_path = tmp_path / "raw_earnings.csv"
    make_raw_earnings_df().to_csv(raw_earnings_path, index=False)

    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    closes = [100.0 + i * 0.5 for i in range(70)]
    pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': closes, 'high': [c + 1 for c in closes],
        'low': [c - 1 for c in closes], 'close': closes,
        'adjusted_close': closes, 'volume': [1000000] * 70,
    }).to_csv(prices_dir / "AAPL.csv", index=False)

    metadata_path = tmp_path / "stock_metadata.csv"
    pd.DataFrame({'Symbol': ['AAPL'], 'Sector': ['Technology']}).to_csv(metadata_path, index=False)

    monkeypatch.setattr('src.web.app.FEATURES_FILE', str(features_path))
    monkeypatch.setattr('src.web.app.MODEL_FILE', str(model_path))
    monkeypatch.setattr('src.web.app.RAW_EARNINGS_FILE', str(raw_earnings_path))
    monkeypatch.setattr('src.web.app.PRICES_DIR', str(prices_dir))
    monkeypatch.setattr('src.web.app.METADATA_FILE', str(metadata_path))

    with TestClient(app) as real_client:
        response = real_client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body['model_run_id'] == 'real_lifespan_test'
    assert body['features_rows'] == 3
