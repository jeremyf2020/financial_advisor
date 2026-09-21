import os
import pandas as pd

PRICE_PANEL_COLUMNS = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']


def adjust_ohlc_for_splits(df):
    """
    Pure: EODHD only natively split/dividend-adjusts Close - Open/High/Low
    are left on the raw, unadjusted scale. Mixing an adjusted Close against
    unadjusted Open/High/Low would produce huge artificial jumps in return
    calculations around any stock split, so Open/High/Low are scaled by the
    same (adjusted_close / close) factor to put all four on a consistent
    basis. No-op if adjusted_close/close aren't both present.
    """
    if 'adjusted_close' not in df.columns or 'close' not in df.columns:
        return df

    result = df.copy()
    adj_factor = result['adjusted_close'] / result['close']
    for raw_col in ['open', 'high', 'low']:
        if raw_col in result.columns:
            result[raw_col] = result[raw_col] * adj_factor

    return result


def load_price_panel(prices_dir):
    """
    I/O: read every ticker's EODHD price CSV in prices_dir into one
    long-format panel DataFrame.
    """
    all_prices = []

    for filename in os.listdir(prices_dir):
        if not filename.endswith(".csv"):
            continue

        ticker = filename.replace(".csv", "")
        filepath = os.path.join(prices_dir, filename)

        try:
            df = pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            continue

        if df.empty or 'date' not in df.columns:
            continue

        df = adjust_ohlc_for_splits(df)

        df = df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'adjusted_close': 'Close', 'volume': 'Volume',
        })
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df['Symbol'] = ticker

        df = df[[c for c in PRICE_PANEL_COLUMNS if c in df.columns]]
        all_prices.append(df)

    if not all_prices:
        return pd.DataFrame(columns=PRICE_PANEL_COLUMNS)

    return pd.concat(all_prices, ignore_index=True)


def merge_sector_map(price_df, metadata_df):
    """
    Pure: attach Sector to the price panel via metadata_df's Symbol->Sector
    mapping, dropping rows with no known sector or no Close price.
    """
    sector_map = dict(zip(metadata_df['Symbol'], metadata_df['Sector']))

    result = price_df.copy()
    result['Sector'] = result['Symbol'].map(sector_map)
    result = result.dropna(subset=['Sector', 'Close'])

    return result.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)


def compute_momentum_features(price_df, windows=(14, 60)):
    """
    Pure: add a Return_Nd column for each window in `windows` - the
    percentage change in Close over N rows, computed separately per ticker.
    """
    result = price_df.copy()

    for window in windows:
        result[f'Return_{window}d'] = result.groupby(
            'Symbol')['Close'].pct_change(periods=window)

    return result


def compute_sector_rank(price_df, window=60):
    """
    Pure: cross-sectional percentile rank of Return_{window}d within each
    (Date, Sector) group - relative strength vs. sector peers on the same
    day, not the broad market.
    """
    result = price_df.copy()
    return_col = f'Return_{window}d'

    result[f'Sector_Rank_{window}d'] = result.groupby(
        ['Date', 'Sector'])[return_col].rank(pct=True, ascending=True)

    return result


def compute_forward_returns(price_df, horizon=1):
    """
    Pure: T+horizon OHLC returns relative to the current Close - the
    target variables for the event-driven prediction task.
    """
    result = price_df.sort_values(by=['Symbol', 'Date']).copy()

    for col in ['Open', 'High', 'Low', 'Close']:
        shifted = result.groupby('Symbol')[col].shift(-horizon)
        result[f'Target_T{horizon}_{col}_Ret'] = (
            shifted - result['Close']) / result['Close']

    return result


def label_spike_event(price_df, threshold=0.02, horizon=1, price_col='High'):
    """
    Pure: binary label - 1 if the T+horizon `price_col` return exceeds
    `threshold` (a "spike"), 0 otherwise. Crude fixed-threshold labeling;
    compute_triple_barrier_labels is a richer alternative that accounts
    for downside risk, not just whether price ever touched the threshold.

    price_col='High' (default) preserves the original production label -
    whether price touched the threshold at any point intraday.
    price_col='Close' instead labels on the T+horizon Close return, which
    is what backtesting.py's DEFAULT_RETURN_COL actually trades on - use
    this to align the training target with close-to-close execution
    (see the target/execution alignment experiment; the High-based
    default and Close-based backtest disagree on what "a spike" means).
    """
    result = price_df.copy()
    ret_col = f'Target_T{horizon}_{price_col}_Ret'

    result['Target_Spike_Class'] = (result[ret_col] > threshold).astype(int)

    return result


def compute_triple_barrier_labels(price_df, take_profit=0.05, stop_loss=0.03, max_holding_days=5):
    """
    Pure: alternative to label_spike_event. For each row, scans forward up
    to max_holding_days trading days (same Symbol) and labels by whichever
    barrier is touched first: take_profit (a later day's High return
    reaches it - label 1), stop_loss (a later day's Low return breaches it
    - label 0), or neither, in which case the label falls back to the sign
    of the return at the max_holding_days close (the "vertical" time
    barrier). On a day both barriers are touched, stop_loss takes priority
    - a conservative assumption, since daily OHLC alone can't say which was
    actually hit first intraday. Writes the same 'Target_Spike_Class'
    column as label_spike_event, so it's a drop-in swap for the rest of
    the pipeline.
    """
    result = price_df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True).copy()
    entry_close = result['Close']

    label = pd.Series(float('nan'), index=result.index)
    touched = pd.Series(False, index=result.index)

    for horizon in range(1, max_holding_days + 1):
        fwd_high = result.groupby('Symbol')['High'].shift(-horizon)
        fwd_low = result.groupby('Symbol')['Low'].shift(-horizon)

        high_ret = (fwd_high - entry_close) / entry_close
        low_ret = (fwd_low - entry_close) / entry_close

        hits_stop_loss = (low_ret <= -stop_loss) & ~touched
        hits_take_profit = (high_ret >= take_profit) & ~touched & ~hits_stop_loss

        label[hits_stop_loss] = 0
        label[hits_take_profit] = 1
        touched |= hits_stop_loss | hits_take_profit

    fwd_close_final = result.groupby('Symbol')['Close'].shift(-max_holding_days)
    final_ret = (fwd_close_final - entry_close) / entry_close
    hits_vertical = ~touched & fwd_close_final.notna()
    label[hits_vertical] = (final_ret[hits_vertical] > 0).astype(float)

    result['Target_Spike_Class'] = label

    return result


def merge_events(price_features_df, earnings_df):
    """
    Pure: left-join earnings data onto the price/feature panel on
    (Symbol, Date), then keep only rows with an actual earnings
    announcement that day - the defining step of the event-driven design,
    reducing a full daily panel to only the days the model may act on.
    """
    earnings_clean = earnings_df.drop_duplicates(
        subset=['Symbol', 'Date']).copy()
    earnings_clean['Date'] = pd.to_datetime(earnings_clean['Date']).dt.date

    merged = pd.merge(price_features_df, earnings_clean,
                       on=['Symbol', 'Date'], how='left')

    return merged.dropna(subset=['Surprise(%)'])


def compute_sue(event_df, window=8, min_periods=4):
    """
    Pure: Standardized Unexpected Earnings - each event's raw Surprise(%)
    divided by that company's own trailing standard deviation of its prior
    surprises (excluding the current one), over up to `window` prior
    earnings events. This is the normalisation PEAD's original literature
    actually uses (Bernard & Thomas, 1989, 1990), not the raw Surprise(%)
    this pipeline otherwise uses as a feature - the same magnitude of
    surprise means something very different for a company with a history
    of volatile earnings vs. a stable one, and raw Surprise(%) can't tell
    them apart. Rows with fewer than min_periods prior events for that
    Symbol, or a prior surprise history with zero variance, are NaN -
    there isn't enough (or varied enough) history yet to standardise
    against, not a legitimate SUE of 0 or infinity.
    """
    result = event_df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True).copy()

    prior_std = result.groupby('Symbol')['Surprise(%)'].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).std())

    result['SUE'] = result['Surprise(%)'] / prior_std.replace(0, float('nan'))

    return result


def generate_earnings_driven_features(
    prices_dir=os.path.join("data", "1_raw", "prices"),
    metadata_file=os.path.join("data", "2_processed", "stock_metadata.csv"),
    earnings_file=os.path.join(
        "data", "1_raw", "earnings", "sp500_historical_earnings.csv"),
    output_file=os.path.join(
        "data", "3_features", "event_driven_features.csv"),
    momentum_windows=(14, 60),
    sector_rank_window=60,
    forward_horizon=1,
    spike_threshold=0.02,
    label_price_col='High',
    use_triple_barrier=False,
    take_profit=0.05,
    stop_loss=0.03,
    max_holding_days=5,
    use_sue=False,
    sue_window=8,
    sue_min_periods=4,
):
    """
    Orchestrator: wire load_price_panel -> merge_sector_map ->
    compute_momentum_features -> compute_sector_rank ->
    compute_forward_returns -> (label_spike_event or
    compute_triple_barrier_labels) -> merge_events -> (compute_sue, if
    use_sue) -> write the final event-driven feature table to
    output_file. use_triple_barrier swaps the labeling step; take_profit/
    stop_loss/max_holding_days only apply when it's set. label_price_col
    only applies to label_spike_event - see its docstring for the
    target/execution alignment rationale. use_sue adds a 'SUE' column
    (see compute_sue) alongside the existing raw 'Surprise(%)' column,
    rather than replacing it - callers choose which one to actually train
    on via their own feature list.
    """
    metadata_df = pd.read_csv(metadata_file)
    earnings_df = pd.read_csv(earnings_file)

    price_df = load_price_panel(prices_dir)
    price_df = merge_sector_map(price_df, metadata_df)
    price_df = compute_momentum_features(price_df, momentum_windows)
    price_df = compute_sector_rank(price_df, sector_rank_window)
    price_df = compute_forward_returns(price_df, forward_horizon)

    if use_triple_barrier:
        price_df = compute_triple_barrier_labels(
            price_df, take_profit, stop_loss, max_holding_days)
    else:
        price_df = label_spike_event(
            price_df, spike_threshold, forward_horizon, label_price_col)

    event_df = merge_events(price_df, earnings_df)
    if use_sue:
        event_df = compute_sue(event_df, window=sue_window, min_periods=sue_min_periods)
    event_df = event_df.dropna(subset=[
        f'Return_{sector_rank_window}d', f'Target_T{forward_horizon}_High_Ret'])

    final_cols = [
        'Date', 'Symbol', 'Sector', 'Close', 'Volume',
        'EPS Estimate', 'Reported EPS', 'Surprise(%)', 'SUE',
        'Return_14d', f'Return_{sector_rank_window}d', f'Sector_Rank_{sector_rank_window}d',
        f'Target_T{forward_horizon}_Open_Ret', f'Target_T{forward_horizon}_High_Ret',
        f'Target_T{forward_horizon}_Low_Ret', f'Target_T{forward_horizon}_Close_Ret',
        'Target_Spike_Class',
    ]
    final_cols = [c for c in final_cols if c in event_df.columns]
    final_df = event_df[final_cols].sort_values(by=['Date', 'Symbol'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)

    return final_df


if __name__ == "__main__":
    df = generate_earnings_driven_features()
    print(f"Generated {len(df)} event-driven feature rows")
