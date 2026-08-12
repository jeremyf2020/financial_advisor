import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.ingest import earnings


def test_fetch_ticker_earnings_eodhd_success():
    """
    A mocked EODHD Earnings::History response should be returned as-is
    (raw dict keyed by date). This path requires a paid Fundamentals
    subscription and isn't used by build_earnings_table() by default.
    """
    # Arrange: mock EODHD's Earnings::History filter response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "2023-01-26": {"date": "2023-01-26", "epsActual": 1.88, "epsEstimate": 1.94,
                        "epsDifference": -0.06, "surprisePercent": -3.09},
    }

    # Act
    with patch('src.ingest.earnings.requests.get', return_value=mock_response) as mock_get:
        history = earnings.fetch_ticker_earnings_eodhd(
            "AAPL", api_key="fake_key")

    # Assert: raw dict passed through untouched
    assert "2023-01-26" in history
    assert history["2023-01-26"]["epsActual"] == 1.88
    mock_get.assert_called_once()


def test_fetch_ticker_earnings_eodhd_api_error():
    """ A non-200 response should return None, not raise """
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch('src.ingest.earnings.requests.get', return_value=mock_response):
        history = earnings.fetch_ticker_earnings_eodhd(
            "DELISTED", api_key="fake_key")

    assert history is None


def test_fetch_ticker_earnings_eodhd_empty_response():
    """ An empty JSON object (no earnings history on record) should return None """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}

    with patch('src.ingest.earnings.requests.get', return_value=mock_response):
        history = earnings.fetch_ticker_earnings_eodhd(
            "EMPTY", api_key="fake_key")

    assert history is None


def _mock_yfinance_earnings_df():
    """ Shared helper: a small realistic yfinance get_earnings_dates() result """
    df = pd.DataFrame({
        'EPS Estimate': [1.94, 1.27],
        'Reported EPS': [1.88, 1.29],
        'Surprise(%)': [-3.09, 1.57],
    }, index=pd.to_datetime(['2023-01-26', '2022-10-27'], utc=True))
    df.index.name = 'Earnings Date'
    return df


def test_fetch_ticker_earnings_yfinance_success():
    """ A mocked yfinance response should be returned as the raw DataFrame """
    # Arrange: mock yf.Ticker(...).get_earnings_dates(), no real network call
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates.return_value = _mock_yfinance_earnings_df()

    # Act
    with patch('src.ingest.earnings.yf.Ticker', return_value=mock_ticker):
        raw_df = earnings.fetch_ticker_earnings_yfinance("AAPL")

    # Assert
    assert raw_df is not None
    assert len(raw_df) == 2
    assert 'EPS Estimate' in raw_df.columns


def test_fetch_ticker_earnings_yfinance_empty():
    """ An empty result (no earnings history on record) should return None """
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates.return_value = pd.DataFrame()

    with patch('src.ingest.earnings.yf.Ticker', return_value=mock_ticker):
        raw_df = earnings.fetch_ticker_earnings_yfinance("EMPTY")

    assert raw_df is None


def test_fetch_ticker_earnings_yfinance_error():
    """ yfinance raising (e.g. delisted/unresolvable ticker) should return None, not raise """
    mock_ticker = MagicMock()
    mock_ticker.get_earnings_dates.side_effect = Exception("no data found")

    with patch('src.ingest.earnings.yf.Ticker', return_value=mock_ticker):
        raw_df = earnings.fetch_ticker_earnings_yfinance("DELISTED")

    assert raw_df is None


def test_clean_earnings_df_shape():
    """
    yfinance's raw earnings-dates DataFrame should become a Symbol/Date +
    EPS-columns DataFrame, with a tz-naive Date column
    """
    # Act
    df = earnings.clean_earnings_df(_mock_yfinance_earnings_df(), "AAPL")

    # Assert
    assert list(df.columns) == ['Symbol', 'Date',
                                 'EPS Estimate', 'Reported EPS', 'Surprise(%)']
    assert len(df) == 2
    assert (df['Symbol'] == 'AAPL').all()
    # tz-naive plain dates, not tz-aware Timestamps
    assert df['Date'].iloc[0] == pd.Timestamp('2023-01-26').date()


def test_clean_earnings_df_empty_history():
    """ None/empty history should produce an empty DataFrame with the right columns, not crash """
    # Act
    df = earnings.clean_earnings_df(None, "AAPL")

    # Assert
    assert df.empty
    assert list(df.columns) == ['Symbol', 'Date',
                                 'EPS Estimate', 'Reported EPS', 'Surprise(%)']


def test_build_earnings_table_concats_multiple_tickers():
    """ Orchestrator should fetch + clean + concat earnings for every ticker """
    # Act
    with patch('src.ingest.earnings.fetch_ticker_earnings_yfinance',
               return_value=_mock_yfinance_earnings_df()):
        df = earnings.build_earnings_table(
            ["AAPL", "MSFT"], rate_limit_seconds=0)

    # Assert
    assert sorted(df['Symbol'].unique().tolist()) == ['AAPL', 'MSFT']
    assert len(df) == 4  # 2 tickers x 2 earnings rows each


def test_build_earnings_table_skips_tickers_with_no_history():
    """ A ticker with no earnings history on record should be skipped, not error """
    # Arrange
    def fake_fetch(ticker, limit=80):
        return None if ticker == "NODATA" else _mock_yfinance_earnings_df()

    # Act
    with patch('src.ingest.earnings.fetch_ticker_earnings_yfinance', side_effect=fake_fetch):
        df = earnings.build_earnings_table(
            ["AAPL", "NODATA"], rate_limit_seconds=0)

    # Assert
    assert df['Symbol'].unique().tolist() == ['AAPL']
