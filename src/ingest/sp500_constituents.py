import os
import requests
import pandas as pd
import io


def check_wiki_connection():
    """
    Check connection to Wikipedia and retrieve S&P 500 HTML content,
    making sure not get banned from wiki
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

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


def parse_sp500_tables(html_content):
    """
    Parse the HTML content to extract current S&P 500 tickers and historical changes
    should return current constituents as a set, historical changes as a DataFrame
    """

    tables = pd.read_html(io.StringIO(html_content))

    current_df = tables[0]

    # replace '.' with '-' in tickers to match Yahoo Finance format (e.g., BRK.B -> BRK-B)
    current_tickers = set(current_df['Symbol'].str.replace(
        '.', '-', regex=False).tolist())

    if '' in current_tickers:
        current_tickers.remove('')

    changes_df = tables[1]

    clean_changes = pd.DataFrame({
        'Date': pd.to_datetime(changes_df.iloc[:, 0]),
        'Added_Ticker': changes_df.iloc[:, 1].fillna('').astype(str).str.replace('.', '-', regex=False),
        'Removed_Ticker': changes_df.iloc[:, 3].fillna('').astype(str).str.replace('.', '-', regex=False)
    })

    return current_tickers, clean_changes


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


def save_raw_wiki_html(output_file=os.path.join("data", "1_raw", "wiki", "sp500_wikipedia_page.html")):
    """
    Stage 1 (raw ingest / Bronze layer): hit Wikipedia and save the HTML
    exactly as returned, no parsing. Downstream steps re-read this local
    file instead of hitting the network again.
    """
    is_connected, html_content = check_wiki_connection()
    if not is_connected:
        print("Failed to connect to Wikipedia")
        return False

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

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
                lifespans[added] = {'Date_added': date,
                                     'Date_removed': 'present'}
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

    with open(raw_html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    current_tickers, changes_df = parse_sp500_tables(html_content)
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

    is_connected, html_content = check_wiki_connection()
    if not is_connected:
        print("Failed to connect to Wikipedia")
        return []

    # parse the HTML to get current tickers (Set) and historical changes (DF)
    current_tickers, changes_df = parse_sp500_tables(html_content)

    # get historical tickers based on the target date, current tickers, and changes
    historical_tickers = get_historical_sp500(
        target_date, current_tickers, changes_df)

    return historical_tickers
