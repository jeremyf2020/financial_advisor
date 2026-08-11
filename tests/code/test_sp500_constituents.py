import pytest
import pandas as pd
import io
from unittest.mock import patch
from src.ingest import sp500_constituents


def test_check_wiki_connection():
    """
    Test connection to Wikipedia and ensure that the function
    should retrieve S&P 500 tickers (HTML)
    """
    # Arrange & Act: call the function to check connection and retrieve HTML
    is_connected, html_content = sp500_constituents.check_wiki_connection()

    # Assert: connection succeeded and the HTML looks like the right page
    assert is_connected == True, "Failed to connect to Wikipedia"
    assert html_content is not None, "No HTML content retrieved from Wikipedia"
    assert "S&P 500" in html_content, "HTML content does not contain expected S&P 500 information"


def test_parse_sp500_mock_tables():
    """
    Test dummy html conversion,
    ensuring the parsing logic correctly extracts tickers and changes
    and handles the '.' to '-' conversion for tickers
    """
    # Arrange: build a small dummy HTML page with a constituents table and a changes table
    dummy_html = """
        <html>
        <body>
            <table id="constituents">
            <tr><th>Symbol</th><th>Security</th></tr>
            <tr><td>AAPL</td><td>Apple Inc.</td></tr>
            <tr><td>BRK.B</td><td>Berkshire</td></tr>
            </table>
            <table id="changes">
            <tr>
                <th>Date</th>
                <th>Added Ticker</th>
                <th>Added Security</th>
                <th>Removed Ticker</th>
                <th>Removed Security</th>
                <th>Reason</th>
            </tr>
            <tr>
                <td>January 1, 2023</td>
                <td>B</td>
                <td>Company B</td>
                <td>C</td>
                <td>Company C</td>
                <td>Merger</td>
            </tr>
            </table>
        </body>
        </html>
    """

    # Act: parse the dummy HTML into current tickers (set) + changes (DataFrame)
    current, changes = sp500_constituents.parse_sp500_tables(dummy_html)

    # Assert: tickers parsed correctly (incl. '.' -> '-' conversion) and the changes row is correct
    assert 'AAPL' in current
    # confirm "." is replaced with "-" (align with yfinance format)
    assert 'BRK-B' in current
    assert '' not in current   # confirm no empty tickers

    assert len(changes) == 1
    assert changes.iloc[0]['Added_Ticker'] == 'B'
    assert changes.iloc[0]['Removed_Ticker'] == 'C'


def test_parse_sp500_tables():
    """
    Integration Test: Test parsing HTML into SET/DF with real Wikipedia content,
    test 2 functions: check_wiki_connection() + parse_sp500_tables()
    """
    # Arrange: fetch the real Wikipedia HTML to parse against
    is_connected, html_content = sp500_constituents.check_wiki_connection()
    assert is_connected is True, "Failed to connect to Wikipedia, Test cannot proceed"

    # Act: parse the live HTML into current tickers (set) + changes (DataFrame)
    current_tickers, changes_df = sp500_constituents.parse_sp500_tables(
        html_content)

    # Assert: current tickers form a ~500-ticker set, changes_df has the expected shape
    assert isinstance(current_tickers, set), "The list should be a Set"
    assert 490 <= len(
        current_tickers) <= 510, "The List should be about 500 stocks"
    assert 'AAPL' in current_tickers, "AAPL exists from 1980 till now, should be in the list"

    assert isinstance(
        changes_df, pd.DataFrame), "The history records must be a DataFrame"
    assert 'Date' in changes_df.columns, "Missing Date column"
    assert 'Added_Ticker' in changes_df.columns, "Missing Added_Ticker column"
    assert 'Removed_Ticker' in changes_df.columns, "Missing Removed_Ticker column"


def test_get_historical_sp500_add():
    """
    Test add-only scenario in the historical reconstruction logic
    """
    # Arrange: mock current constituents + a single "add" change
    mock_current = {'A', 'B'}
    mock_changes = pd.DataFrame({
        'Date': pd.to_datetime(['2023-01-01']),
        'Added_Ticker': ['B'],
        'Removed_Ticker': ['']
    })

    # Act: reconstruct the constituent list as of a date before the change happened
    result_2022 = sp500_constituents.get_historical_sp500(
        '2022-01-01', mock_current, mock_changes)

    # Assert: the ticker added after the target date should not appear in the earlier list
    assert result_2022 == [
        'A'], "Add logic failed: 'B' should not be in the 2022 list"


def test_get_historical_sp500_remove():
    """
    Test remove-only scenario in the historical reconstruction logic
    """
    # Arrange: mock current constituents + a single "remove" change
    mock_current = {'A', 'B'}
    mock_changes = pd.DataFrame({
        'Date': pd.to_datetime(['2021-01-01']),
        'Added_Ticker': [''],
        'Removed_Ticker': ['C']
    })

    # Act: reconstruct the constituent list as of a date before the change happened
    result_2020 = sp500_constituents.get_historical_sp500(
        '2020-01-01', mock_current, mock_changes)

    # Assert: the ticker removed after the target date should still appear in the earlier list
    assert result_2020 == [
        'A', 'B', 'C'], "Remove logic failed: 'C' should be in the 2020 list"


def test_get_historical_sp500_integration_logic():
    """
    Test the core logic of "Time Machine"
    ensuring the add/remove logic is accurate
    """
    # Arrange: mock current constituents + two changes (one add, one remove) on different dates
    mock_current = {'A', 'B'}

    mock_changes = pd.DataFrame({
        'Date': pd.to_datetime(['2023-01-01', '2021-01-01']),
        'Added_Ticker': ['B', 'A'],
        'Removed_Ticker': ['C', 'D']
    })

    # Act: reconstruct the 2022 list
    # 'A' was added in 2021, 'B' was added in 2023, 'C'/'D' were removed in 2021
    result_2022 = sp500_constituents.get_historical_sp500(
        '2022-01-01', mock_current, mock_changes)
    # Assert: 'A' should be in (added before 2022), 'C' should be back in (removed after 2022)
    assert result_2022 == ['A', 'C'], "Fail to get 2022 list"

    # Act: reconstruct the 2020 list
    # 'A' was added in 2021, so not in 2020; 'B' was added in 2023, so not in 2020
    result_2020 = sp500_constituents.get_historical_sp500(
        '2020-01-01', mock_current, mock_changes)
    # Assert: both 'C' and 'D' should be back in (removed after 2020), 'A'/'B' should be out
    assert result_2020 == ['C', 'D'], "Fail to get 2020 list"


def test_get_sp500_tickers_by_date_integration():
    """
    Integration Test
    test the entire flow of getting S&P 500 tickers by date
    """
    # Arrange: pick a target date to query
    test_date = "2020-01-01"

    # Act: run the full end-to-end flow (connect -> parse -> reconstruct)
    tickers_2020 = sp500_constituents.get_sp500_tickers_by_date(test_date)

    # Assert: result is a list of ~500 tickers, and TSLA (added later) is excluded
    assert isinstance(
        tickers_2020, list), "The output should be a list of tickers"

    assert 490 <= len(
        tickers_2020) <= 510, f"Expected around 500 tickers in 2020, got {len(tickers_2020)}"

    assert 'TSLA' not in tickers_2020, "TSLA was added to S&P 500 in 2020, should not be in the list on Jan 1, 2020"


def test_save_raw_wiki_html(tmp_path):
    """
    Stage 1: save whatever check_wiki_connection() returns as-is to disk,
    no parsing applied
    """
    # Arrange: mock the network call so this test doesn't depend on live Wikipedia
    output_file = tmp_path / "sp500_wikipedia_page.html"
    fake_html = "<html>fake S&P 500 page</html>"

    with patch('src.ingest.sp500_constituents.check_wiki_connection') as mock_check:
        mock_check.return_value = (True, fake_html)

        # Act: run stage 1
        result = sp500_constituents.save_raw_wiki_html(str(output_file))

    # Assert: file saved with the exact raw content, no parsing applied
    assert result is True
    assert output_file.read_text(encoding='utf-8') == fake_html


def test_save_raw_wiki_html_connection_failure(tmp_path):
    """
    Stage 1 should fail cleanly (no file written) if Wikipedia is unreachable
    """
    # Arrange: mock a failed connection
    output_file = tmp_path / "sp500_wikipedia_page.html"

    with patch('src.ingest.sp500_constituents.check_wiki_connection') as mock_check:
        mock_check.return_value = (False, None)

        # Act: run stage 1
        result = sp500_constituents.save_raw_wiki_html(str(output_file))

    # Assert: reports failure, no file written
    assert result is False
    assert not output_file.exists()
