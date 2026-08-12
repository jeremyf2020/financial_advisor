"""
Run the full raw-data ingestion pipeline end-to-end, one command:

    ./venv/bin/python scripts/run_ingest_pipeline.py

Stages (each is independently resumable - safe to re-run after a partial
failure, already-downloaded files are skipped, not re-fetched):
  1. S&P 500 constituent history (Wikipedia)    -> data/2_processed/master_ticker_list.csv
  2. Historical daily prices, every ticker (EODHD) -> data/1_raw/prices/*.csv
  3. Sector metadata (Wikipedia + SEC EDGAR)     -> data/2_processed/stock_metadata.csv
  4. Earnings history (yfinance)                 -> data/1_raw/earnings/sp500_historical_earnings.csv

Each stage is also runnable on its own via `python -m src.ingest.<module>`
if you only need to redo one step.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
from src.ingest import sp500_constituents as constituents  # noqa: E402
from src.ingest import fetch_prices as prices  # noqa: E402
from src.ingest import stock_metadata as metadata  # noqa: E402
from src.ingest import earnings as earn  # noqa: E402

MASTER_LIST = os.path.join("data", "2_processed", "master_ticker_list.csv")
RAW_HTML = os.path.join("data", "1_raw", "wiki", "sp500_wikipedia_page.html")


def step_1_constituents():
    print("=== Step 1/4: S&P 500 constituent history (Wikipedia) ===")
    if not constituents.save_raw_wiki_html():
        raise SystemExit(
            "Failed to fetch a complete Wikipedia page - aborting pipeline")
    constituents.save_master_ticker_list()
    tickers_df = pd.read_csv(MASTER_LIST)
    print(f"{len(tickers_df)} tickers in {MASTER_LIST}")


def step_2_prices():
    print("\n=== Step 2/4: Historical daily prices (EODHD) ===")
    tickers_df = pd.read_csv(MASTER_LIST)
    tickers = tickers_df['Symbol'].tolist()
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    output_dir = os.path.join("data", "1_raw", "prices")
    count = prices.download_all_tickers(
        tickers, "1990-01-01", end_date, output_dir)
    print(f"Downloaded {count}/{len(tickers)} tickers")


def step_3_metadata():
    print("\n=== Step 3/4: Sector metadata (Wikipedia + SEC EDGAR) ===")
    tickers_df = pd.read_csv(MASTER_LIST)

    with open(RAW_HTML, encoding="utf-8") as f:
        wiki_sector_map = constituents.extract_current_sector_map(f.read())

    cik_lookup = metadata.fetch_cik_lookup()
    valid_df, rejected_df = metadata.build_metadata_tables(
        tickers_df, wiki_sector_map, cik_lookup=cik_lookup)

    output_dir = os.path.join("data", "2_processed")
    valid_df.to_csv(os.path.join(
        output_dir, "stock_metadata.csv"), index=False)
    rejected_df.to_csv(os.path.join(
        output_dir, "delisted_stocks_to_scan.csv"), index=False)
    print(f"Valid: {len(valid_df)}, Manual review needed: {len(rejected_df)}")


def step_4_earnings():
    print("\n=== Step 4/4: Earnings history (yfinance) ===")
    tickers_df = pd.read_csv(MASTER_LIST)
    tickers = tickers_df['Symbol'].tolist()
    earnings_df = earn.build_earnings_table(tickers)

    output_dir = os.path.join("data", "1_raw", "earnings")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sp500_historical_earnings.csv")
    earnings_df.to_csv(output_file, index=False)
    print(f"Saved {len(earnings_df)} earnings records")


if __name__ == "__main__":
    step_1_constituents()
    step_2_prices()
    step_3_metadata()
    step_4_earnings()
    print("\n=== Ingestion pipeline complete ===")
