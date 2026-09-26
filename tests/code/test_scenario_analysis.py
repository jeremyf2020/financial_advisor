import pytest
import pandas as pd
from xgboost import XGBClassifier
from src.web import scenario_analysis as sa

FEATURE_COLS = ['EPS Estimate', 'Reported EPS', 'Surprise(%)', 'Return_60d', 'Sector_Rank_60d']


def make_raw_earnings_df():
    """ Symbol/Date/EPS Estimate/Reported EPS/Surprise(%), mirroring sp500_historical_earnings.csv's shape """
    return pd.DataFrame({
        'Symbol': ['AAPL', 'AAPL', 'MSFT', 'NOPE'],
        'Date': ['2024-01-01', '2024-06-01', '2024-06-15', '2024-06-01'],
        'EPS Estimate': [1.5, 1.8, 2.0, float('nan')],
        'Reported EPS': [1.6, float('nan'), float('nan'), float('nan')],
        'Surprise(%)': [6.67, float('nan'), float('nan'), float('nan')],
    })


def make_snapshot_df():
    return pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT'],
        'Date': ['2024-05-20', '2024-05-20'],
        'Sector': ['Technology', 'Technology'],
        'Return_60d': [0.05, 0.02],
        'Sector_Rank_60d': [0.8, 0.5],
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


def test_find_upcoming_earnings_keeps_only_future_unreported_rows_with_a_snapshot():
    """ Should exclude past/reported rows, and rows for tickers with no price snapshot """
    raw_earnings_df = make_raw_earnings_df()
    snapshot_df = make_snapshot_df()

    result = sa.find_upcoming_earnings(raw_earnings_df, snapshot_df, as_of='2024-05-25')

    # AAPL's 2024-01-01 row is past and already reported -> excluded
    # AAPL's 2024-06-01 row is future and unreported -> included
    # MSFT's 2024-06-15 row is future and unreported -> included
    # NOPE has no snapshot row and no EPS Estimate -> excluded
    assert list(result['Symbol']) == ['AAPL', 'MSFT']
    assert list(result['Date']) == ['2024-06-01', '2024-06-15']


def test_find_upcoming_earnings_row_returns_soonest_future_row():
    raw_earnings_df = make_raw_earnings_df()

    row = sa.find_upcoming_earnings_row(raw_earnings_df, 'AAPL', as_of='2024-05-25')

    assert row['Date'] == '2024-06-01'
    assert row['EPS Estimate'] == 1.8


def test_find_upcoming_earnings_row_no_future_event_returns_none():
    raw_earnings_df = make_raw_earnings_df()

    row = sa.find_upcoming_earnings_row(raw_earnings_df, 'AAPL', as_of='2024-12-01')

    assert row is None


def test_find_upcoming_earnings_row_case_and_whitespace_insensitive():
    raw_earnings_df = make_raw_earnings_df()

    row = sa.find_upcoming_earnings_row(raw_earnings_df, '  aapl  ', as_of='2024-05-25')

    assert row['Symbol'] == 'AAPL'


def test_find_snapshot_row_known_and_unknown_ticker():
    snapshot_df = make_snapshot_df()

    assert sa.find_snapshot_row(snapshot_df, 'MSFT')['Return_60d'] == 0.02
    assert sa.find_snapshot_row(snapshot_df, 'NOPE') is None


def test_build_surprise_grid_shape_and_inverse_formula():
    """ Reported EPS should follow Reported EPS = EPS Estimate * (1 + Surprise%/100) """
    grid = sa.build_surprise_grid(eps_estimate=2.0, surprise_range=(-10, 10), step=5)

    assert list(grid['Surprise(%)']) == [-10, -5, 0, 5, 10]
    zero_row = grid[grid['Surprise(%)'] == 0].iloc[0]
    assert zero_row['Reported EPS'] == pytest.approx(2.0)
    plus_ten = grid[grid['Surprise(%)'] == 10].iloc[0]
    assert plus_ten['Reported EPS'] == pytest.approx(2.2)


def test_build_scenario_analysis_holds_snapshot_features_fixed_across_the_sweep():
    """ Every scenario row should use the same Return_60d/Sector_Rank_60d - only Surprise(%)/Reported EPS vary """
    snapshot_row = make_snapshot_df().iloc[0]  # AAPL
    model = make_fitted_model()

    result = sa.build_scenario_analysis(
        snapshot_row, eps_estimate=1.8, model=model, feature_cols=FEATURE_COLS,
        spike_threshold=0.07, surprise_range=(-10, 10), step=10)

    assert result['eps_estimate'] == 1.8
    assert result['spike_threshold'] == 0.07
    assert result['context_features']['Return_60d'] == 0.05
    assert result['context_features']['Sector_Rank_60d'] == 0.8
    assert len(result['scenarios']) == 3  # -10, 0, 10
    for scenario in result['scenarios']:
        assert scenario['predicted_label'] in ('No Spike', 'Spike')
        assert 0.0 <= scenario['spike_probability'] <= 1.0


def test_build_scenario_analysis_reported_eps_tracks_surprise_pct():
    snapshot_row = make_snapshot_df().iloc[0]
    model = make_fitted_model()

    result = sa.build_scenario_analysis(
        snapshot_row, eps_estimate=2.0, model=model, feature_cols=FEATURE_COLS,
        surprise_range=(-10, 10), step=10)

    by_surprise = {s['surprise_pct']: s['reported_eps'] for s in result['scenarios']}
    assert by_surprise[0.0] == pytest.approx(2.0)
    assert by_surprise[10.0] == pytest.approx(2.2)
    assert by_surprise[-10.0] == pytest.approx(1.8)
