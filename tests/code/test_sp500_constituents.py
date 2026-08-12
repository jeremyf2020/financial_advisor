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


def test_parse_sp500_tables_missing_changes_table():
    """
    A page that renders without the changes table (Wikipedia does this
    intermittently) should raise ValueError, not silently misparse a
    different table as if it were the changes table
    """
    # Arrange: constituents table present, but no Ticker+Date changes table
    dummy_html = """
        <html>
        <body>
            <table id="constituents">
            <tr><th>Symbol</th><th>Security</th></tr>
            <tr><td>AAPL</td><td>Apple Inc.</td></tr>
            </table>
            <table id="sector-nav">
            <tr><th>Energy</th><th>Materials</th></tr>
            <tr><td>Some Company</td><td>Another Company</td></tr>
            </table>
        </body>
        </html>
    """

    # Act / Assert: parsing raises rather than mis-reading the nav table as changes
    with pytest.raises(ValueError, match="changes table"):
        sp500_constituents.parse_sp500_tables(dummy_html)


def test_parse_sp500_tables_missing_constituents_table():
    """ A page with no Symbol column at all should raise ValueError """
    # Arrange: only a changes-shaped table, no constituents table
    dummy_html = """
        <html>
        <body>
            <table id="changes">
            <tr><th>Date</th><th>Added Ticker</th><th>Removed Ticker</th></tr>
            <tr><td>January 1, 2023</td><td>B</td><td>C</td></tr>
            </table>
        </body>
        </html>
    """

    # Act / Assert
    with pytest.raises(ValueError, match="constituents table"):
        sp500_constituents.parse_sp500_tables(dummy_html)


def test_extract_current_sector_map():
    """
    GICS Sector / Sub-Industry should be parsed from the constituents table
    for free, for tickers that are current S&P 500 members
    """
    # Arrange: dummy HTML with the real GICS Sector / Sub-Industry columns
    dummy_html = """
        <html>
        <body>
            <table id="constituents">
            <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
            <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>Technology Hardware, Storage &amp; Peripherals</td></tr>
            <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector Holdings</td></tr>
            </table>
            <table id="changes">
            <tr>
                <th>Date</th><th>Added Ticker</th><th>Added Security</th>
                <th>Removed Ticker</th><th>Removed Security</th><th>Reason</th>
            </tr>
            <tr><td>January 1, 2023</td><td>B</td><td>Company B</td><td>C</td><td>Company C</td><td>Merger</td></tr>
            </table>
        </body>
        </html>
    """

    # Act: extract the sector map
    sector_map = sp500_constituents.extract_current_sector_map(dummy_html)

    # Assert: correct sector/industry, and '.' -> '-' conversion applied to the key
    assert sector_map['AAPL']['Sector'] == 'Information Technology'
    assert sector_map['BRK-B']['Sector'] == 'Financials'
    assert sector_map['BRK-B']['Industry'] == 'Multi-Sector Holdings'
    assert sector_map['BRK-B']['Name'] == 'Berkshire Hathaway'


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


VALID_DUMMY_WIKI_HTML = """
    <html>
    <body>
        <table id="constituents">
        <tr><th>Symbol</th><th>Security</th></tr>
        <tr><td>AAPL</td><td>Apple Inc.</td></tr>
        </table>
        <table id="changes">
        <tr><th>Date</th><th>Added Ticker</th><th>Added Security</th>
            <th>Removed Ticker</th><th>Removed Security</th><th>Reason</th></tr>
        <tr><td>January 1, 2023</td><td>B</td><td>Company B</td><td>C</td><td>Company C</td><td>Merger</td></tr>
        </table>
    </body>
    </html>
"""

INCOMPLETE_DUMMY_WIKI_HTML = """
    <html>
    <body>
        <table id="constituents">
        <tr><th>Symbol</th><th>Security</th></tr>
        <tr><td>AAPL</td><td>Apple Inc.</td></tr>
        </table>
        <table id="sector-nav">
        <tr><th>Energy</th><th>Materials</th></tr>
        <tr><td>Some Company</td><td>Another Company</td></tr>
        </table>
    </body>
    </html>
"""


def test_save_raw_wiki_html(tmp_path):
    """
    Stage 1: save whatever check_wiki_connection() returns as-is to disk,
    no parsing applied to the saved content itself (only used to validate
    the page is complete before saving)
    """
    # Arrange: mock the network call so this test doesn't depend on live Wikipedia
    output_file = tmp_path / "sp500_wikipedia_page.html"

    with patch('src.ingest.sp500_constituents.check_wiki_connection') as mock_check:
        mock_check.return_value = (True, VALID_DUMMY_WIKI_HTML)

        # Act: run stage 1
        result = sp500_constituents.save_raw_wiki_html(str(output_file))

    # Assert: file saved with the exact raw content, no parsing applied to it
    assert result is True
    assert output_file.read_text(encoding='utf-8') == VALID_DUMMY_WIKI_HTML


def test_save_raw_wiki_html_connection_failure(tmp_path):
    """
    Stage 1 should fail cleanly (no file written) if Wikipedia is unreachable
    """
    # Arrange: mock a failed connection
    output_file = tmp_path / "sp500_wikipedia_page.html"

    with patch('src.ingest.sp500_constituents.check_wiki_connection') as mock_check:
        mock_check.return_value = (False, None)

        # Act: run stage 1
        result = sp500_constituents.save_raw_wiki_html(str(output_file), max_retries=2)

    # Assert: reports failure, no file written
    assert result is False
    assert not output_file.exists()


def test_save_raw_wiki_html_retries_on_incomplete_page(tmp_path):
    """
    If the first fetch comes back without a changes table (Wikipedia does
    this intermittently), stage 1 should retry rather than save a broken page
    """
    # Arrange: first call returns an incomplete page, second call a valid one
    output_file = tmp_path / "sp500_wikipedia_page.html"

    with patch('src.ingest.sp500_constituents.check_wiki_connection') as mock_check:
        mock_check.side_effect = [
            (True, INCOMPLETE_DUMMY_WIKI_HTML),
            (True, VALID_DUMMY_WIKI_HTML),
        ]

        # Act
        result = sp500_constituents.save_raw_wiki_html(str(output_file), max_retries=3)

    # Assert: succeeded on the second attempt, saved the valid page
    assert result is True
    assert output_file.read_text(encoding='utf-8') == VALID_DUMMY_WIKI_HTML
    assert mock_check.call_count == 2


def test_save_raw_wiki_html_gives_up_after_max_retries(tmp_path):
    """
    If every attempt comes back incomplete, stage 1 should give up after
    max_retries and not save anything
    """
    # Arrange: every call returns an incomplete page
    output_file = tmp_path / "sp500_wikipedia_page.html"

    with patch('src.ingest.sp500_constituents.check_wiki_connection') as mock_check:
        mock_check.return_value = (True, INCOMPLETE_DUMMY_WIKI_HTML)

        # Act
        result = sp500_constituents.save_raw_wiki_html(str(output_file), max_retries=3)

    # Assert: gave up, nothing saved, tried exactly max_retries times
    assert result is False
    assert not output_file.exists()
    assert mock_check.call_count == 3


def test_build_ticker_lifespans_current_only():
    """
    Current constituent with no changes history should have no known
    Date_added and be marked as still 'present'
    """
    # Arrange: one current ticker, no changes at all
    mock_current = {'A'}
    mock_changes = pd.DataFrame({
        'Date': pd.to_datetime([]),
        'Added_Ticker': [],
        'Removed_Ticker': []
    })

    # Act: build the lifespan dict
    lifespans = sp500_constituents.build_ticker_lifespans(
        mock_current, mock_changes)

    # Assert: no add date on record, still present
    assert lifespans['A']['Date_added'] is None
    assert lifespans['A']['Date_removed'] == 'present'


def test_build_ticker_lifespans_historical_removal():
    """
    A ticker that was removed and is not a current constituent should
    keep its removal date instead of 'present'
    """
    # Arrange: 'B' is not in current_tickers, was removed in the past
    mock_current = {'A'}
    mock_changes = pd.DataFrame({
        'Date': pd.to_datetime(['2019-06-01']),
        'Added_Ticker': [''],
        'Removed_Ticker': ['B']
    })

    # Act: build the lifespan dict
    lifespans = sp500_constituents.build_ticker_lifespans(
        mock_current, mock_changes)

    # Assert: 'B' recorded with its removal date, not 'present'
    assert lifespans['B']['Date_removed'] == pd.Timestamp('2019-06-01')


def test_build_ticker_lifespans_keeps_earliest_add_date():
    """
    If a ticker shows up as 'added' more than once in the changes history,
    the earliest date should win (first time it actually joined)
    """
    # Arrange: 'A' added twice, in 2015 then again in 2021
    mock_current = {'A'}
    mock_changes = pd.DataFrame({
        'Date': pd.to_datetime(['2021-01-01', '2015-03-01']),
        'Added_Ticker': ['A', 'A'],
        'Removed_Ticker': ['', '']
    })

    # Act: build the lifespan dict
    lifespans = sp500_constituents.build_ticker_lifespans(
        mock_current, mock_changes)

    # Assert: Date_added is the earliest of the two dates, not the latest
    assert lifespans['A']['Date_added'] == pd.Timestamp('2015-03-01')


def test_build_ticker_lifespans_keeps_latest_removal_date():
    """
    If a historical (non-current) ticker was removed more than once,
    the latest removal date should win (most recent time it actually left)
    """
    # Arrange: 'C' is not current, removed once in 2018 then again in 2022
    mock_current = {'A'}
    mock_changes = pd.DataFrame({
        'Date': pd.to_datetime(['2018-05-01', '2022-09-01']),
        'Added_Ticker': ['', ''],
        'Removed_Ticker': ['C', 'C']
    })

    # Act: build the lifespan dict
    lifespans = sp500_constituents.build_ticker_lifespans(
        mock_current, mock_changes)

    # Assert: Date_removed is the latest of the two dates, not the earliest
    assert lifespans['C']['Date_removed'] == pd.Timestamp('2022-09-01')


def test_build_ticker_lifespans_readded_stays_present():
    """
    A ticker that was removed in the past but is a current constituent
    (i.e. it was re-added later) should stay marked 'present', not be
    overwritten by its earlier removal date
    """
    # Arrange: 'A' is current, but was removed once back in 2017 before rejoining
    mock_current = {'A'}
    mock_changes = pd.DataFrame({
        'Date': pd.to_datetime(['2017-01-01']),
        'Added_Ticker': [''],
        'Removed_Ticker': ['A']
    })

    # Act: build the lifespan dict
    lifespans = sp500_constituents.build_ticker_lifespans(
        mock_current, mock_changes)

    # Assert: current constituency wins over the stale historical removal event
    assert lifespans['A']['Date_removed'] == 'present'


def test_build_ticker_lifespans_added_then_removed_not_current():
    """
    Regression test: a ticker that was added and later removed, and is NOT
    a current constituent, must NOT end up as 'present'. Real-world case:
    AAL was added 2015-03-23 and removed 2024-09-23 (replaced by PLTR) -
    the add event used to default Date_removed to 'present', which then
    blocked the later remove event from ever updating it.
    """
    # Arrange: 'AAL' is not current; added in 2015, removed in 2024
    mock_current = {'PLTR'}
    mock_changes = pd.DataFrame({
        'Date': pd.to_datetime(['2015-03-23', '2024-09-23']),
        'Added_Ticker': ['AAL', 'PLTR'],
        'Removed_Ticker': ['AGN', 'AAL']
    })

    # Act: build the lifespan dict
    lifespans = sp500_constituents.build_ticker_lifespans(
        mock_current, mock_changes)

    # Assert: the later removal actually took effect, not stuck on 'present'
    assert lifespans['AAL']['Date_added'] == pd.Timestamp('2015-03-23')
    assert lifespans['AAL']['Date_removed'] == pd.Timestamp('2024-09-23')


def test_lifespans_to_dataframe():
    """
    Convert the {ticker: {Date_added, Date_removed}} dict into the
    master_ticker_list.csv shape: Symbol/Date_added/Date_removed, sorted
    """
    # Arrange: a small unsorted lifespans dict
    lifespans = {
        'B': {'Date_added': pd.Timestamp('2010-01-01'), 'Date_removed': 'present'},
        'A': {'Date_added': None, 'Date_removed': pd.Timestamp('2019-01-01')},
    }

    # Act: convert to DataFrame
    df = sp500_constituents.lifespans_to_dataframe(lifespans)

    # Assert: correct columns, sorted by Symbol, values preserved
    assert list(df.columns) == ['Symbol', 'Date_added', 'Date_removed']
    assert df['Symbol'].tolist() == ['A', 'B']
    # pandas coerces a mixed None/Timestamp column to datetime64, turning
    # None into NaT - pd.isna() is the correct way to check for it
    assert pd.isna(df.iloc[0]['Date_added'])
    assert df.iloc[1]['Date_removed'] == 'present'


def test_save_master_ticker_list_from_raw_html(tmp_path):
    """
    Stage 2: reads a previously-saved raw HTML file from disk and produces
    master_ticker_list.csv, with no network call needed
    """
    # Arrange: a small dummy raw HTML file, as if stage 1 already saved it
    raw_html_file = tmp_path / "sp500_wikipedia_page.html"
    raw_html_file.write_text("""
        <html>
        <body>
            <table id="constituents">
            <tr><th>Symbol</th><th>Security</th></tr>
            <tr><td>AAPL</td><td>Apple Inc.</td></tr>
            </table>
            <table id="changes">
            <tr>
                <th>Date</th><th>Added Ticker</th><th>Added Security</th>
                <th>Removed Ticker</th><th>Removed Security</th><th>Reason</th>
            </tr>
            <tr>
                <td>January 1, 2023</td><td></td><td></td>
                <td>OLDCO</td><td>Old Co.</td><td>Removed</td>
            </tr>
            </table>
        </body>
        </html>
    """, encoding='utf-8')
    output_file = tmp_path / "master_ticker_list.csv"

    # Act: run stage 2 purely from the local raw file, no network involved
    result = sp500_constituents.save_master_ticker_list(
        str(raw_html_file), str(output_file))

    # Assert: master list produced with both the current and historical ticker
    assert result is True
    df = pd.read_csv(output_file)
    assert set(df['Symbol']) == {'AAPL', 'OLDCO'}
    assert df[df['Symbol'] == 'AAPL'].iloc[0]['Date_removed'] == 'present'


def test_save_master_ticker_list_missing_raw_file(tmp_path):
    """
    Stage 2 should fail cleanly if stage 1's raw HTML hasn't been saved yet
    """
    # Arrange: point at a raw HTML file that doesn't exist
    missing_file = tmp_path / "does_not_exist.html"
    output_file = tmp_path / "master_ticker_list.csv"

    # Act: run stage 2 without a stage-1 file present
    result = sp500_constituents.save_master_ticker_list(
        str(missing_file), str(output_file))

    # Assert: fails cleanly, no output written
    assert result is False
    assert not output_file.exists()
