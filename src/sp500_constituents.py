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
