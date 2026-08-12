import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.ingest import stock_metadata


def test_fetch_ticker_sector_success():
    """
    A mocked EODHD General-section response should return the raw
    Sector/Industry/Name fields
    """
    # Arrange: mock EODHD's General filter response, no real network call
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "Sector": "Technology", "Industry": "Consumer Electronics", "Name": "Apple Inc"}

    # Act
    with patch('src.ingest.stock_metadata.requests.get', return_value=mock_response) as mock_get:
        info = stock_metadata.fetch_ticker_sector("AAPL", api_key="fake_key")

    # Assert: raw fields passed through as-is
    assert info == {"Sector": "Technology",
                     "Industry": "Consumer Electronics", "Name": "Apple Inc"}
    mock_get.assert_called_once()


def test_fetch_ticker_sector_api_error():
    """ A non-200 response should return None, not raise """
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch('src.ingest.stock_metadata.requests.get', return_value=mock_response):
        info = stock_metadata.fetch_ticker_sector(
            "DELISTED", api_key="fake_key")

    assert info is None


def test_fetch_ticker_sector_empty_response():
    """ An empty JSON object (ticker exists but no fundamentals data) should return None """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}

    with patch('src.ingest.stock_metadata.requests.get', return_value=mock_response):
        info = stock_metadata.fetch_ticker_sector("EMPTY", api_key="fake_key")

    assert info is None


def test_classify_metadata_row_valid():
    """ A ticker with a known sector should be classified as valid """
    # Arrange
    info = {"Sector": "Healthcare", "Industry": "Biotechnology", "Name": "Some Co"}

    # Act
    result = stock_metadata.classify_metadata_row(
        "SOME", info, "2015-01-01", "present")

    # Assert
    assert result["valid"] is True
    assert result["row"]["Sector"] == "Healthcare"
    assert result["row"]["Symbol"] == "SOME"


def test_classify_metadata_row_missing_info():
    """ No info at all (fetch failed / delisted) should be classified as rejected """
    # Act
    result = stock_metadata.classify_metadata_row(
        "GONE", None, "2001-01-01", "2020-05-01")

    # Assert
    assert result["valid"] is False
    assert result["row"]["Symbol"] == "GONE"
    assert "Sector" not in result["row"]


def test_classify_metadata_row_empty_sector():
    """ Info present but with an empty Sector string should also be rejected """
    # Arrange
    info = {"Sector": "", "Industry": None, "Name": "Weird Co"}

    # Act
    result = stock_metadata.classify_metadata_row(
        "WEIRD", info, "2010-01-01", "present")

    # Assert
    assert result["valid"] is False


def test_fetch_cik_lookup_success():
    """ A mocked SEC company_tickers.json response should return a ticker->CIK dict """
    # Arrange
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    }

    # Act
    with patch('src.ingest.stock_metadata.requests.get', return_value=mock_response) as mock_get:
        lookup = stock_metadata.fetch_cik_lookup(user_agent="test test@example.com")

    # Assert: dict keyed by ticker, and a descriptive User-Agent was sent (SEC requires this)
    assert lookup == {"AAPL": 320193, "MSFT": 789019}
    assert mock_get.call_args.kwargs["headers"]["User-Agent"] == "test test@example.com"


def test_fetch_cik_lookup_error():
    """ A non-200 response should return an empty dict, not raise """
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch('src.ingest.stock_metadata.requests.get', return_value=mock_response):
        lookup = stock_metadata.fetch_cik_lookup(user_agent="test test@example.com")

    assert lookup == {}


def test_fetch_ticker_sector_edgar_success():
    """ A mocked SEC submissions response should return sicDescription as Sector """
    # Arrange
    cik_lookup = {"AAPL": 320193}
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "sic": "3571", "sicDescription": "Electronic Computers", "name": "Apple Inc."}

    # Act
    with patch('src.ingest.stock_metadata.requests.get', return_value=mock_response):
        info = stock_metadata.fetch_ticker_sector_edgar(
            "AAPL", cik_lookup, user_agent="test test@example.com")

    # Assert: SIC description used as the Sector fallback value
    assert info == {"Sector": "Electronic Computers",
                     "Industry": None, "Name": "Apple Inc."}


def test_fetch_ticker_sector_edgar_ticker_not_in_lookup():
    """ A ticker missing from the CIK lookup should return None without an HTTP call """
    # Act
    with patch('src.ingest.stock_metadata.requests.get') as mock_get:
        info = stock_metadata.fetch_ticker_sector_edgar(
            "UNKNOWN", {}, user_agent="test test@example.com")

    # Assert
    assert info is None
    mock_get.assert_not_called()


def test_fetch_ticker_sector_edgar_api_error():
    """ A non-200 response should return None, not raise """
    cik_lookup = {"AAPL": 320193}
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch('src.ingest.stock_metadata.requests.get', return_value=mock_response):
        info = stock_metadata.fetch_ticker_sector_edgar(
            "AAPL", cik_lookup, user_agent="test test@example.com")

    assert info is None


def test_build_metadata_tables_uses_wiki_sector_map_first():
    """
    A ticker present in the free Wikipedia sector map should be classified
    without ever calling SEC EDGAR
    """
    # Arrange
    tickers_df = pd.DataFrame({
        "Symbol": ["AAPL"],
        "Date_added": ["1980-01-01"],
        "Date_removed": ["present"],
    })
    wiki_sector_map = {
        "AAPL": {"Sector": "Information Technology", "Industry": "Hardware", "Name": "Apple Inc."}}

    # Act
    with patch('src.ingest.stock_metadata.fetch_ticker_sector_edgar') as mock_edgar:
        valid_df, rejected_df = stock_metadata.build_metadata_tables(
            tickers_df, wiki_sector_map, cik_lookup={"AAPL": 320193})

    # Assert: classified from the wiki map, EDGAR never touched
    assert valid_df['Symbol'].tolist() == ['AAPL']
    assert valid_df.iloc[0]['Sector'] == 'Information Technology'
    mock_edgar.assert_not_called()


def test_build_metadata_tables_falls_back_to_edgar():
    """
    A ticker missing from the Wikipedia sector map (e.g. delisted) should
    fall back to SEC EDGAR
    """
    # Arrange: 'OLDCO' is not a current constituent, so it's not in wiki_sector_map
    tickers_df = pd.DataFrame({
        "Symbol": ["OLDCO"],
        "Date_added": ["2001-01-01"],
        "Date_removed": ["2015-01-01"],
    })

    def fake_edgar(ticker, cik_lookup, user_agent=None):
        return {"Sector": "Old Industry Inc", "Industry": None, "Name": "Old Co"}

    # Act
    with patch('src.ingest.stock_metadata.fetch_ticker_sector_edgar', side_effect=fake_edgar):
        valid_df, rejected_df = stock_metadata.build_metadata_tables(
            tickers_df, wiki_sector_map={}, cik_lookup={"OLDCO": 1}, rate_limit_seconds=0)

    # Assert
    assert valid_df['Symbol'].tolist() == ['OLDCO']
    assert valid_df.iloc[0]['Sector'] == 'Old Industry Inc'


def test_build_metadata_tables_rejects_when_no_source_has_it():
    """
    A ticker missing from both the Wikipedia map and SEC EDGAR should be
    routed to the rejected (manual-review) table
    """
    # Arrange
    tickers_df = pd.DataFrame({
        "Symbol": ["GHOST"],
        "Date_added": ["1999-01-01"],
        "Date_removed": ["2005-01-01"],
    })

    # Act: no cik_lookup provided at all, so EDGAR fallback is skipped entirely
    valid_df, rejected_df = stock_metadata.build_metadata_tables(
        tickers_df, wiki_sector_map={}, cik_lookup=None)

    # Assert
    assert valid_df.empty
    assert rejected_df['Symbol'].tolist() == ['GHOST']
