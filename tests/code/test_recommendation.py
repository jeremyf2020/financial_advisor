import pytest
import pandas as pd
from xgboost import XGBClassifier
from src.web import recommendation as rec

FEATURE_COLS = ['EPS Estimate', 'Reported EPS', 'Surprise(%)', 'Return_60d', 'Sector_Rank_60d']


def make_features_df():
    """ Small hand-built event-driven feature table, mirroring event_driven_features.csv's shape """
    return pd.DataFrame({
        'Date': ['2024-01-01', '2024-04-01', '2024-07-01', '2023-01-01'],
        'Symbol': ['AAPL', 'AAPL', 'AAPL', 'MSFT'],
        'Sector': ['Technology', 'Technology', 'Technology', 'Technology'],
        'EPS Estimate': [1.0, 1.1, 1.2, 2.0],
        'Reported EPS': [1.1, 1.0, 1.3, 2.1],
        'Surprise(%)': [10.0, -9.0, 8.0, 5.0],
        'Return_60d': [0.05, -0.03, 0.02, 0.01],
        'Sector_Rank_60d': [0.8, 0.2, 0.6, 0.5],
        'Target_T1_Close_Ret': [0.03, -0.02, 0.09, 0.01],
        'Target_T1_High_Ret': [0.05, -0.01, 0.12, 0.02],
    })


def make_fitted_model():
    X = pd.DataFrame({
        'EPS Estimate': [1.0, 1.1, 1.2, 2.0],
        'Reported EPS': [1.1, 1.0, 1.3, 2.1],
        'Surprise(%)': [10.0, -9.0, 8.0, 5.0],
        'Return_60d': [0.05, -0.03, 0.02, 0.01],
        'Sector_Rank_60d': [0.8, 0.2, 0.6, 0.5],
    })
    y = [1, 0, 1, 0]
    model = XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    return model


def test_find_latest_event_row_picks_max_date():
    """ Should return the most recent row for a ticker with multiple events, regardless of row order """
    features_df = make_features_df()

    row = rec.find_latest_event_row(features_df, 'AAPL')

    assert row['Date'] == '2024-07-01'


def test_find_latest_event_row_case_and_whitespace_insensitive():
    """ Ticker lookup should tolerate lowercase/whitespace input from the UI """
    features_df = make_features_df()

    row = rec.find_latest_event_row(features_df, '  aapl  ')

    assert row['Symbol'] == 'AAPL'


def test_find_latest_event_row_unknown_ticker_returns_none():
    """ A ticker with zero rows should return None, not raise """
    features_df = make_features_df()

    row = rec.find_latest_event_row(features_df, 'NOPE')

    assert row is None


def test_build_recommendation_shape():
    """ Should return predicted class/label/confidence plus feature values and historical outcome """
    features_df = make_features_df()
    row = rec.find_latest_event_row(features_df, 'AAPL')
    model = make_fitted_model()

    result = rec.build_recommendation(row, model, FEATURE_COLS, spike_threshold=0.07)

    assert result['ticker'] == 'AAPL'
    assert result['as_of_date'] == '2024-07-01'
    assert result['predicted_label'] in ('No Spike', 'Spike')
    assert 0.0 <= result['confidence'] <= 1.0
    assert result['spike_threshold'] == 0.07
    assert result['feature_values']['Surprise(%)'] == 8.0


def test_build_recommendation_historical_outcome_is_the_rows_actual_realized_return():
    """ historical_outcome must pass through the row's actual past outcome unchanged - it is not a forecast """
    features_df = make_features_df()
    row = rec.find_latest_event_row(features_df, 'AAPL')
    model = make_fitted_model()

    result = rec.build_recommendation(row, model, FEATURE_COLS)

    assert result['historical_outcome']['target_t1_close_ret'] == pytest.approx(0.09)
    assert result['historical_outcome']['target_t1_high_ret'] == pytest.approx(0.12)


def test_build_recommendation_missing_historical_outcome_is_none():
    """ A row with no forward-return columns at all (e.g. the very last day in the dataset) should be None, not crash """
    features_df = make_features_df().drop(columns=['Target_T1_Close_Ret', 'Target_T1_High_Ret'])
    row = rec.find_latest_event_row(features_df, 'AAPL')
    model = make_fitted_model()

    result = rec.build_recommendation(row, model, FEATURE_COLS)

    assert result['historical_outcome']['target_t1_close_ret'] is None
