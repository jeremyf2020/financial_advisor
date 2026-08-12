import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.ingest import fetch_prices


def test_download_single_ticker_success(tmp_path):
    """
    A mocked EODHD response with raw OHLCV fields should get saved as CSV
    """
    # Arrange: mock EODHD's JSON response shape, no real network call
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"date": "2024-01-02", "open": 100, "high": 105, "low": 99,
         "close": 104, "adjusted_close": 104, "volume": 1000000},
        {"date": "2024-01-03", "open": 104, "high": 106, "low": 103,
         "close": 105, "adjusted_close": 105, "volume": 900000},
    ]

    # Act: download AAPL with the mocked HTTP call
    with patch('src.ingest.fetch_prices.requests.get', return_value=mock_response) as mock_get:
        result = fetch_prices.download_single_ticker(
            "AAPL", "2024-01-01", "2024-01-03", str(tmp_path), api_key="fake_key")

    # Assert: CSV saved with the expected raw columns, one call made
    assert result is True
    saved_file = tmp_path / "AAPL.csv"
    assert saved_file.exists()
    df = pd.read_csv(saved_file)
    assert list(df.columns) == [
        'date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
    assert len(df) == 2
    mock_get.assert_called_once()


def test_download_single_ticker_surfaces_free_tier_warning(tmp_path, capsys):
    """
    If EODHD's response includes a 'warning' field (e.g. free-tier history
    limit silently truncating the requested range to the latest day only),
    it should be printed loudly rather than silently ignored, since the
    caller otherwise has no way to know the data they got isn't what they
    asked for
    """
    # Arrange: mock the real free-tier warning shape seen from EODHD
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"date": "2025-08-11", "open": 227.92, "high": 229.56, "low": 224.76,
         "close": 227.18, "adjusted_close": 226.3439, "volume": 61806100,
         "warning": "Data is limited by one year as you have free subscription"},
    ]

    # Act: request a range EODHD silently can't honour on this plan
    with patch('src.ingest.fetch_prices.requests.get', return_value=mock_response):
        result = fetch_prices.download_single_ticker(
            "AAPL", "2024-01-01", "2024-01-31", str(tmp_path), api_key="fake_key")

    # Assert: still saves whatever it got, but prints the warning so it's not missed
    assert result is True
    captured = capsys.readouterr()
    assert "Data is limited by one year" in captured.out

    # Assert: the 'warning' key itself doesn't pollute the saved CSV's columns
    df = pd.read_csv(tmp_path / "AAPL.csv")
    assert 'warning' not in df.columns


def test_download_single_ticker_api_error(tmp_path):
    """
    A non-200 response (e.g. bad ticker) should fail cleanly, no file written
    """
    # Arrange: mock a 404 response
    mock_response = MagicMock()
    mock_response.status_code = 404

    # Act
    with patch('src.ingest.fetch_prices.requests.get', return_value=mock_response):
        result = fetch_prices.download_single_ticker(
            "DELISTED", "2024-01-01", "2024-01-03", str(tmp_path), api_key="fake_key")

    # Assert: reports failure, nothing saved
    assert result is False
    assert not (tmp_path / "DELISTED.csv").exists()


def test_download_single_ticker_network_timeout(tmp_path):
    """
    A network-level failure (timeout, connection reset, etc.) should fail
    cleanly and return False, not raise - so a single flaky ticker doesn't
    crash a batch download_all_tickers() run partway through
    """
    # Arrange: mock requests.get raising a timeout, as it does on a slow/dead connection
    import requests as requests_module

    # Act
    with patch('src.ingest.fetch_prices.requests.get',
               side_effect=requests_module.exceptions.ReadTimeout("timed out")):
        result = fetch_prices.download_single_ticker(
            "SLOW", "2024-01-01", "2024-01-03", str(tmp_path), api_key="fake_key")

    # Assert: reports failure, nothing saved, no exception propagated
    assert result is False
    assert not (tmp_path / "SLOW.csv").exists()


def test_download_single_ticker_empty_response(tmp_path):
    """
    An empty JSON array (ticker exists but no data in the requested range)
    should fail cleanly rather than save an empty CSV
    """
    # Arrange: mock a 200 response with no rows
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []

    # Act
    with patch('src.ingest.fetch_prices.requests.get', return_value=mock_response):
        result = fetch_prices.download_single_ticker(
            "EMPTY", "2024-01-01", "2024-01-03", str(tmp_path), api_key="fake_key")

    # Assert: reports failure, nothing saved
    assert result is False
    assert not (tmp_path / "EMPTY.csv").exists()


def test_is_file_valid_not_exist(tmp_path):
    """ No CSV on disk yet -> not valid """
    assert fetch_prices.is_file_valid("AAPL", str(tmp_path)) is False


def test_is_file_valid_exists(tmp_path):
    """ Non-empty CSV already on disk -> valid, counts as already downloaded """
    dummy = tmp_path / "AAPL.csv"
    dummy.write_text(
        "date,open,high,low,close,adjusted_close,volume\n2024-01-02,100,105,99,104,104,1000000")
    assert fetch_prices.is_file_valid("AAPL", str(tmp_path)) is True


def test_is_file_valid_empty_file(tmp_path):
    """ A zero-byte file (interrupted/failed download) should NOT count as valid """
    dummy = tmp_path / "AAPL.csv"
    dummy.write_text("")
    assert fetch_prices.is_file_valid("AAPL", str(tmp_path)) is False


def test_download_all_tickers_skips_existing(tmp_path):
    """
    Orchestrator should skip tickers that already have a valid CSV,
    making no API call for them at all
    """
    # Arrange: AAPL already has a valid CSV on disk
    existing = tmp_path / "AAPL.csv"
    existing.write_text(
        "date,open,high,low,close,adjusted_close,volume\n2024-01-02,100,105,99,104,104,1000000")

    # Act: run the orchestrator with download_single_ticker mocked out
    with patch('src.ingest.fetch_prices.download_single_ticker') as mock_download:
        count = fetch_prices.download_all_tickers(
            ["AAPL"], "2024-01-01", "2024-01-03", str(tmp_path),
            api_key="fake_key", rate_limit_seconds=0)

    # Assert: counted as success, but never actually downloaded again
    assert count == 1
    mock_download.assert_not_called()


def test_download_all_tickers_downloads_missing(tmp_path):
    """
    Orchestrator should call download_single_ticker for tickers that don't
    have a file yet
    """
    # Arrange: no MSFT.csv exists in tmp_path

    # Act: run the orchestrator with download_single_ticker mocked out
    with patch('src.ingest.fetch_prices.download_single_ticker', return_value=True) as mock_download:
        count = fetch_prices.download_all_tickers(
            ["MSFT"], "2024-01-01", "2024-01-03", str(tmp_path),
            api_key="fake_key", rate_limit_seconds=0)

    # Assert: one success, download was actually attempted
    assert count == 1
    mock_download.assert_called_once_with(
        "MSFT", "2024-01-01", "2024-01-03", str(tmp_path), "fake_key")
