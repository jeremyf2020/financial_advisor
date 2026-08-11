import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

EODHD_API_KEY = os.getenv("EODHD_API_KEY")
EODHD_BASE_URL = "https://eodhd.com/api"

RAW_PRICE_COLUMNS = ['date', 'open', 'high',
                     'low', 'close', 'adjusted_close', 'volume']


def download_single_ticker(ticker, start_date, end_date, output_dir, api_key=None):
    """
    I/O: EODHD /api/eod/{TICKER}.US -> CSV.
    Raw fields pulled: date, open, high, low, close, adjusted_close, volume
    """
    api_key = api_key or EODHD_API_KEY
    url = f"{EODHD_BASE_URL}/eod/{ticker}.US"
    params = {
        "api_token": api_key,
        "from": start_date,
        "to": end_date,
        "period": "d",
        "fmt": "json",
    }

    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        return False

    data = response.json()
    if not data:
        return False

    for row in data:
        if isinstance(row, dict) and 'warning' in row:
            print(f"!! [{ticker}] EODHD warning: {row['warning']}")
            break

    df = pd.DataFrame(data)
    df = df[[c for c in RAW_PRICE_COLUMNS if c in df.columns]]

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(save_path, index=False)

    return True


def is_file_valid(ticker, output_dir):
    """
    I/O (light): resume/skip check - a non-empty CSV already on disk for
    this ticker counts as already downloaded.
    """
    save_path = os.path.join(output_dir, f"{ticker}.csv")
    return os.path.exists(save_path) and os.path.getsize(save_path) > 0


def download_all_tickers(tickers, start_date, end_date, output_dir, api_key=None, rate_limit_seconds=0.5):
    """
    Orchestrator: loop over tickers, skip ones already downloaded, rate
    limit between live calls to stay within EODHD's request limits.
    """
    success_count = 0

    for ticker in tickers:
        if is_file_valid(ticker, output_dir):
            success_count += 1
            continue

        if download_single_ticker(ticker, start_date, end_date, output_dir, api_key):
            success_count += 1

        time.sleep(rate_limit_seconds)

    return success_count
