import os
import requests
import pandas as pd
import io

CONSTITUENTS_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# Wikipedia split the historical add/remove changes table out into its own
# article at some point - the main list page no longer reliably includes it.
HISTORICAL_CHANGES_URL = 'https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500'


def check_wiki_connection(url=CONSTITUENTS_URL):
    """
    Check connection to a Wikipedia page and retrieve its HTML content,
    making sure not get banned from wiki. Defaults to the current-
    constituents page; pass HISTORICAL_CHANGES_URL for the changes page.
    """
    # Add User-Agent to avoid 403 Forbidden
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        # return true & text if the request was successful
        if response.status_code == 200:
            return True, response.text

        # otherwise, return false & None
        return False, None

    except requests.exceptions.RequestException:
        # Catch any request-related exceptions (e.g., connection errors, timeouts)
        return False, None


def _find_constituents_table(tables):
    """
    Pure: locate the current-constituents table by column signature (a
    'Symbol' column) rather than by fixed position - Wikipedia's page
    layout is not guaranteed to keep tables in the same order.
    """
    for table in tables:
        if any('Symbol' in str(col) for col in table.columns):
            return table
    return None


def _find_changes_table(tables):
    """
    Pure: locate the historical add/remove changes table by column
    signature (both a Ticker-like and a Date-like column present) rather
    than by fixed position.
    """
    for table in tables:
        flat_cols = []
        for col in table.columns:
            parts = col if isinstance(col, tuple) else (col,)
            flat_cols.extend(str(part) for part in parts)
        has_ticker = any('Ticker' in c for c in flat_cols)
        has_date = any('Date' in c for c in flat_cols)
        if has_ticker and has_date:
            return table
    return None


def parse_current_constituents(html_content):
    """
    Pure: the current-constituents page -> current tickers (set).
    Raises ValueError if the expected table can't be found.
    """
    tables = pd.read_html(io.StringIO(html_content))

    current_df = _find_constituents_table(tables)
    if current_df is None:
        raise ValueError(
            "Could not find the constituents table (no 'Symbol' column) in the Wikipedia page")

    # replace '.' with '-' in tickers to match Yahoo Finance format (e.g., BRK.B -> BRK-B)
    current_tickers = set(current_df['Symbol'].str.replace(
        '.', '-', regex=False).tolist())

    if '' in current_tickers:
        current_tickers.remove('')

    return current_tickers


def parse_changes_table(html_content):
    """
    Pure: the historical-changes page -> changes as a DataFrame
    (Date/Added_Ticker/Removed_Ticker). Raises ValueError if the expected
    table can't be found - Wikipedia occasionally renders a page
    incompletely, and silently misreading the wrong table as if it were
    the changes table would corrupt master_ticker_list.csv downstream.
    """
    tables = pd.read_html(io.StringIO(html_content))

    changes_df = _find_changes_table(tables)
    if changes_df is None:
        raise ValueError(
            "Could not find the changes table (no Ticker+Date columns) in the "
            "Wikipedia page - it may have rendered incompletely")

    clean_changes = pd.DataFrame({
        'Date': pd.to_datetime(changes_df.iloc[:, 0]),
        'Added_Ticker': changes_df.iloc[:, 1].fillna('').astype(str).str.replace('.', '-', regex=False),
        'Removed_Ticker': changes_df.iloc[:, 3].fillna('').astype(str).str.replace('.', '-', regex=False)
    })

    return clean_changes


def parse_sp500_tables(constituents_html, changes_html):
    """
    Combine both pages' parsing into the (current_tickers, changes_df)
    shape the rest of the module expects. The two pieces of data now live
    on two separate Wikipedia articles (see HISTORICAL_CHANGES_URL).
    """
    current_tickers = parse_current_constituents(constituents_html)
    changes_df = parse_changes_table(changes_html)
    return current_tickers, changes_df


def extract_current_sector_map(html_content):
    """
    Pure: parse the current-constituents table for GICS Sector / Sub-Industry,
    which Wikipedia provides for free. Only covers tickers that are *current*
    S&P 500 members - historically delisted tickers aren't in this table and
    need a different source (see stock_metadata.py's SEC EDGAR fallback).
    Returns {ticker: {'Sector': ..., 'Industry': ..., 'Name': ...}}
    """
    tables = pd.read_html(io.StringIO(html_content))
    current_df = _find_constituents_table(tables)
    if current_df is None:
        raise ValueError(
            "Could not find the constituents table (no 'Symbol' column) in the Wikipedia page")

    sector_map = {}
    for _, row in current_df.iterrows():
        ticker = str(row['Symbol']).replace('.', '-')
        if not ticker:
            continue
        sector_map[ticker] = {
            'Sector': row.get('GICS Sector'),
            'Industry': row.get('GICS Sub-Industry'),
            'Name': row.get('Security'),
        }

    return sector_map


def get_historical_sp500(target_date, current_tickers, changes_df):
    """
    get historical S&P 500 constituents as of the target date
    return a sorted list of tickers
    """
    target_date = pd.to_datetime(target_date)
    historical_tickers = current_tickers.copy()

    # find all changes that happened after the target date, and reverse them to reconstruct the historical list
    post_changes = changes_df[changes_df['Date'] > target_date]

    for index, row in post_changes.iterrows():
        added = row['Added_Ticker']
        removed = row['Removed_Ticker']

        if added != '' and added in historical_tickers:
            historical_tickers.remove(added)

        if removed != '':
            historical_tickers.add(removed)

    return sorted(list(historical_tickers))


def _fetch_and_validate(url, parse_fn, max_retries, label):
    """
    I/O: fetch `url` up to max_retries times, keeping only a page that
    `parse_fn` can successfully parse. Returns the raw HTML, or None if
    every attempt failed.
    """
    for attempt in range(max_retries):
        is_connected, html_content = check_wiki_connection(url)
        if not is_connected:
            print(
                f"Failed to connect to Wikipedia ({label}, attempt {attempt + 1}/{max_retries})")
            continue

        try:
            parse_fn(html_content)
        except ValueError as e:
            print(
                f"Incomplete Wikipedia page ({label}, attempt {attempt + 1}/{max_retries}): {e}")
            continue

        return html_content

    print(f"Failed to fetch a complete {label} page after {max_retries} attempts")
    return None


def save_raw_wiki_html(
    output_file=os.path.join("data", "1_raw", "wiki", "sp500_wikipedia_page.html"),
    changes_output_file=os.path.join(
        "data", "1_raw", "wiki", "sp500_historical_changes.html"),
    max_retries=3
):
    """
    Stage 1 (raw ingest / Bronze layer): hit both the current-constituents
    page and the historical changes page, save each exactly as returned
    (no parsing beyond validating the expected table is present).
    Downstream steps re-read these local files instead of hitting the
    network again.
    """
    constituents_html = _fetch_and_validate(
        CONSTITUENTS_URL, parse_current_constituents, max_retries, "constituents")
    if constituents_html is None:
        return False

    changes_html = _fetch_and_validate(
        HISTORICAL_CHANGES_URL, parse_changes_table, max_retries, "historical changes")
    if changes_html is None:
        return False

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(constituents_html)

    os.makedirs(os.path.dirname(changes_output_file), exist_ok=True)
    with open(changes_output_file, 'w', encoding='utf-8') as f:
        f.write(changes_html)

    return True


def build_ticker_lifespans(current_tickers, changes_df):
    """
    Combine current constituents with historical add/remove changes into
    {ticker: {'Date_added': Timestamp or None, 'Date_removed': Timestamp or 'present'}}.
    Date_added keeps the earliest add event; Date_removed keeps the latest
    remove event, unless the ticker is a current constituent, in which case
    it always stays 'present'.
    """
    lifespans = {
        ticker: {'Date_added': None, 'Date_removed': 'present'}
        for ticker in current_tickers
    }

    sorted_changes = changes_df.sort_values('Date')

    for _, row in sorted_changes.iterrows():
        date = row['Date']
        added = row['Added_Ticker']
        removed = row['Removed_Ticker']

        if added:
            if added not in lifespans:
                # 'present' is reserved for tickers actually in current_tickers -
                # a ticker only seen here via an add event might still get
                # removed by a later change event, so start it as unknown/None
                # rather than falsely claiming it's still a live constituent
                lifespans[added] = {'Date_added': date,
                                     'Date_removed': None}
            elif lifespans[added]['Date_added'] is None:
                lifespans[added]['Date_added'] = date

        if removed:
            if removed not in lifespans:
                lifespans[removed] = {'Date_added': None, 'Date_removed': date}
            elif lifespans[removed]['Date_removed'] != 'present':
                lifespans[removed]['Date_removed'] = date

    return lifespans


def lifespans_to_dataframe(lifespans):
    """
    Convert {ticker: {Date_added, Date_removed}} into the master_ticker_list.csv
    shape: Symbol/Date_added/Date_removed, sorted by Symbol.
    """
    rows = [
        {'Symbol': ticker, 'Date_added': dates['Date_added'],
         'Date_removed': dates['Date_removed']}
        for ticker, dates in lifespans.items()
    ]
    return pd.DataFrame(rows, columns=['Symbol', 'Date_added', 'Date_removed']).sort_values('Symbol').reset_index(drop=True)


def save_master_ticker_list(
    raw_html_file=os.path.join(
        "data", "1_raw", "wiki", "sp500_wikipedia_page.html"),
    changes_html_file=os.path.join(
        "data", "1_raw", "wiki", "sp500_historical_changes.html"),
    output_file=os.path.join(
        "data", "2_processed", "master_ticker_list.csv")
):
    """
    Stage 2 (process raw -> Silver layer): read the raw HTML saved by
    save_raw_wiki_html(), parse + merge into master_ticker_list.csv.
    No network call, so it can be re-run freely to debug/fix the merge
    logic without re-downloading.
    """
    if not os.path.exists(raw_html_file):
        print(
            f"Raw HTML not found: {raw_html_file}. Run save_raw_wiki_html() first.")
        return False

    if not os.path.exists(changes_html_file):
        print(
            f"Raw changes HTML not found: {changes_html_file}. Run save_raw_wiki_html() first.")
        return False

    with open(raw_html_file, 'r', encoding='utf-8') as f:
        constituents_html = f.read()

    with open(changes_html_file, 'r', encoding='utf-8') as f:
        changes_html = f.read()

    current_tickers, changes_df = parse_sp500_tables(
        constituents_html, changes_html)
    lifespans = build_ticker_lifespans(current_tickers, changes_df)
    master_df = lifespans_to_dataframe(lifespans)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    master_df.to_csv(output_file, index=False)

    return True


def get_sp500_tickers_by_date(target_date):
    """
    Main function: Get S&P 500 tickers for a specific date
    Input: target_date (str in 'YYYY-MM-DD' format)
    Output: List of tickers that were in the S&P 500 on that date
    """

    is_connected, constituents_html = check_wiki_connection(CONSTITUENTS_URL)
    if not is_connected:
        print("Failed to connect to Wikipedia (constituents page)")
        return []

    is_connected, changes_html = check_wiki_connection(HISTORICAL_CHANGES_URL)
    if not is_connected:
        print("Failed to connect to Wikipedia (historical changes page)")
        return []

    # parse both pages to get current tickers (Set) and historical changes (DF)
    current_tickers, changes_df = parse_sp500_tables(
        constituents_html, changes_html)

    # get historical tickers based on the target date, current tickers, and changes
    historical_tickers = get_historical_sp500(
        target_date, current_tickers, changes_df)

    return historical_tickers


if __name__ == "__main__":
    if save_raw_wiki_html():
        save_master_ticker_list()
