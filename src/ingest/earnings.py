import os
import time
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

EODHD_API_KEY = os.getenv("EODHD_API_KEY")
EODHD_BASE_URL = "https://eodhd.com/api"

EARNINGS_COLUMNS = ['Symbol', 'Date',
                     'EPS Estimate', 'Reported EPS', 'Surprise(%)']


def fetch_ticker_earnings_eodhd(ticker, api_key=None):
    """
    I/O: EODHD /api/fundamentals/{TICKER}.US, filtered to Earnings::History.
    Requires a paid EODHD Fundamentals subscription - kept available but not
    used by build_earnings_table() by default, since yfinance covers the
    same need for free (see fetch_ticker_earnings_yfinance).
    Raw fields pulled: date, epsActual, epsEstimate, epsDifference, surprisePercent
    """
    api_key = api_key or EODHD_API_KEY
    url = f"{EODHD_BASE_URL}/fundamentals/{ticker}.US"
    params = {"api_token": api_key, "fmt": "json", "filter": "Earnings::History"}

    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        return None

    history = response.json()
    if not history:
        return None

    return history


def fetch_ticker_earnings_yfinance(ticker, limit=80):
    """
    I/O: yfinance's earnings calendar, free. Raw fields pulled:
    Earnings Date, EPS Estimate, Reported EPS, Surprise(%). Returns the raw
    DataFrame as yfinance gives it (indexed by Earnings Date), or None if
    there's nothing on record / the lookup failed.
    """
    try:
        raw_df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    except Exception:
        return None

    if raw_df is None or raw_df.empty:
        return None

    return raw_df


def clean_earnings_df(raw_df, ticker):
    """
    Pure: yfinance's raw earnings-dates DataFrame -> a clean DataFrame with
    Symbol/Date + EPS Estimate/Reported EPS/Surprise(%), tz-naive dates.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=EARNINGS_COLUMNS)

    df = raw_df.reset_index()
    df['Earnings Date'] = pd.to_datetime(
        df['Earnings Date'], utc=True).dt.tz_localize(None).dt.date
    df = df.rename(columns={'Earnings Date': 'Date'})
    df.insert(0, 'Symbol', ticker)

    return df[[c for c in EARNINGS_COLUMNS if c in df.columns]]


def build_earnings_table(tickers, rate_limit_seconds=0.5):
    """
    Orchestrator: loop over tickers, fetch + clean earnings history for
    each via yfinance (free), concat into one master earnings table.
    """
    all_earnings = []

    for ticker in tickers:
        raw_df = fetch_ticker_earnings_yfinance(ticker)
        cleaned = clean_earnings_df(raw_df, ticker)
        if not cleaned.empty:
            all_earnings.append(cleaned)
        time.sleep(rate_limit_seconds)

    if not all_earnings:
        return pd.DataFrame(columns=EARNINGS_COLUMNS)

    return pd.concat(all_earnings, ignore_index=True)


if __name__ == "__main__":
    master_list = os.path.join("data", "2_processed", "master_ticker_list.csv")
    tickers_df = pd.read_csv(master_list)
    tickers = tickers_df['Symbol'].tolist()

    print(f"Fetching earnings history for {len(tickers)} tickers via yfinance...")
    earnings_df = build_earnings_table(tickers)

    output_dir = os.path.join("data", "1_raw", "earnings")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sp500_historical_earnings.csv")
    earnings_df.to_csv(output_file, index=False)

    print(f"Saved {len(earnings_df)} earnings records to {output_file}")
