import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

EODHD_API_KEY = os.getenv("EODHD_API_KEY")
EODHD_BASE_URL = "https://eodhd.com/api"

SEC_USER_AGENT = os.getenv(
    "SEC_EDGAR_USER_AGENT", "financial_advisor_fyp contact@example.com")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def fetch_ticker_sector(ticker, api_key=None):
    """
    I/O: EODHD /api/fundamentals/{TICKER}.US, filtered to the General
    section. Raw fields pulled: Sector, Industry, Name. Requires a paid
    EODHD Fundamentals subscription - kept available but not used by
    build_metadata_tables() by default, since the free Wikipedia + SEC
    EDGAR sources below cover the same need at no cost.
    """
    api_key = api_key or EODHD_API_KEY
    url = f"{EODHD_BASE_URL}/fundamentals/{ticker}.US"
    params = {"api_token": api_key, "fmt": "json", "filter": "General"}

    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        return None

    general = response.json()
    if not general:
        return None

    return {
        "Sector": general.get("Sector"),
        "Industry": general.get("Industry"),
        "Name": general.get("Name"),
    }


def fetch_cik_lookup(user_agent=None):
    """
    I/O: SEC's free ticker -> CIK mapping (company_tickers.json).
    Raw fields pulled: ticker, cik_str. No API key required, but SEC
    requires a descriptive User-Agent identifying the requester.
    """
    user_agent = user_agent or SEC_USER_AGENT
    response = requests.get(SEC_TICKERS_URL, headers={
                             "User-Agent": user_agent}, timeout=10)
    if response.status_code != 200:
        return {}

    data = response.json()
    return {entry["ticker"]: entry["cik_str"] for entry in data.values()}


def fetch_ticker_sector_edgar(ticker, cik_lookup, user_agent=None):
    """
    I/O: SEC EDGAR submissions API, filtered to sic/sicDescription - a free
    fallback for tickers no longer in the S&P 500 (missing from Wikipedia's
    current-constituents GICS table). SIC is a coarser, differently
    structured scheme than GICS Sector, so this is a fallback
    classification for delisted names, not a like-for-like replacement.
    """
    cik = cik_lookup.get(ticker)
    if cik is None:
        return None

    user_agent = user_agent or SEC_USER_AGENT
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    response = requests.get(url, headers={
                             "User-Agent": user_agent}, timeout=10)
    if response.status_code != 200:
        return None

    data = response.json()
    if not data.get("sicDescription"):
        return None

    return {
        "Sector": data["sicDescription"],
        "Industry": None,
        "Name": data.get("name"),
    }


def classify_metadata_row(ticker, info, date_added, date_removed):
    """
    Pure: decide whether this ticker gets a valid metadata row or gets
    routed to the manual-review (rejected) list. A ticker with no sector
    on record from any source is rejected.
    """
    if not info or not info.get("Sector"):
        return {
            "valid": False,
            "row": {
                "Symbol": ticker,
                "Official_Name": (info or {}).get("Name") or "Unknown / Missing Sector",
                "Date_added": date_added,
                "Date_removed": date_removed,
            },
        }

    return {
        "valid": True,
        "row": {
            "Symbol": ticker,
            "Official_Name": info.get("Name") or ticker,
            "Sector": info["Sector"],
            "Industry": info.get("Industry"),
            "Date_added": date_added,
            "Date_removed": date_removed,
        },
    }


def build_metadata_tables(tickers_df, wiki_sector_map, cik_lookup=None, sec_user_agent=None, rate_limit_seconds=0.2):
    """
    Orchestrator: for each ticker, use the free Wikipedia GICS sector map
    first (covers current S&P 500 constituents, no network call needed -
    it's already an in-memory dict). Falls back to SEC EDGAR's SIC
    classification for tickers missing from it (historically delisted
    names), otherwise routes to the manual-review (rejected) table.
    """
    valid_rows = []
    rejected_rows = []

    for _, row in tickers_df.iterrows():
        ticker = row["Symbol"]
        info = wiki_sector_map.get(ticker)

        if info is None and cik_lookup is not None:
            info = fetch_ticker_sector_edgar(
                ticker, cik_lookup, sec_user_agent)
            time.sleep(rate_limit_seconds)

        classified = classify_metadata_row(
            ticker, info, row["Date_added"], row["Date_removed"])

        if classified["valid"]:
            valid_rows.append(classified["row"])
        else:
            rejected_rows.append(classified["row"])

    valid_df = pd.DataFrame(valid_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    return valid_df, rejected_df


if __name__ == "__main__":
    from src.ingest import sp500_constituents as constituents

    master_list = os.path.join("data", "2_processed", "master_ticker_list.csv")
    raw_html_file = os.path.join(
        "data", "1_raw", "wiki", "sp500_wikipedia_page.html")

    tickers_df = pd.read_csv(master_list)

    with open(raw_html_file, encoding="utf-8") as f:
        wiki_sector_map = constituents.extract_current_sector_map(f.read())

    print("Fetching SEC EDGAR CIK lookup...")
    cik_lookup = fetch_cik_lookup()

    print(f"Classifying {len(tickers_df)} tickers "
          f"({len(wiki_sector_map)} covered by Wikipedia for free)...")
    valid_df, rejected_df = build_metadata_tables(
        tickers_df, wiki_sector_map, cik_lookup=cik_lookup)

    output_dir = os.path.join("data", "2_processed")
    valid_df.to_csv(os.path.join(output_dir, "stock_metadata.csv"), index=False)
    rejected_df.to_csv(os.path.join(
        output_dir, "delisted_stocks_to_scan.csv"), index=False)

    print(f"Valid: {len(valid_df)}, Manual review needed: {len(rejected_df)}")
