import pytest
import pandas as pd
from src.transform import feature_engineering as fe


def test_load_price_panel(tmp_path):
    """
    EODHD-shaped CSVs (lowercase columns, adjusted_close) should load into
    one long panel, with adjusted_close renamed to Close
    """
    # Arrange: one ticker's EODHD-shaped price CSV
    (tmp_path / "AAPL.csv").write_text(
        "date,open,high,low,close,adjusted_close,volume\n"
        "2024-01-02,100,105,99,104,103.5,1000000\n"
        "2024-01-03,104,106,103,105,104.5,900000\n"
    )

    # Act
    df = fe.load_price_panel(str(tmp_path))

    # Assert: Title-Case columns, adjusted_close used as Close, Symbol tagged
    assert list(df.columns) == ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
    assert len(df) == 2
    assert (df['Symbol'] == 'AAPL').all()
    assert df.iloc[0]['Close'] == 103.5  # adjusted_close, not raw close


def test_adjust_ohlc_for_splits():
    """
    Regression test: EODHD only natively adjusts Close for splits/dividends.
    If Open/High/Low were left unadjusted while Close is adjusted, forward
    return calculations would show huge artificial jumps around any split
    (observed for real: 89% of rows flagged as a 2%+ "spike" instead of the
    expected ~5-10%, with some T+1 returns over 200%). Open/High/Low must
    be scaled by the same adjustment factor as Close.
    """
    # Arrange: adjusted_close is half of close - simulating a 2:1 split
    # sometime after this date, the way EODHD represents it
    df = pd.DataFrame({
        'open': [100], 'high': [105], 'low': [99],
        'close': [100], 'adjusted_close': [50], 'volume': [1000000],
    })

    # Act
    result = fe.adjust_ohlc_for_splits(df)

    # Assert: Open/High/Low scaled down by the same 0.5 factor as Close
    row = result.iloc[0]
    assert row['close'] == 100  # untouched - Close comes from adjusted_close later
    assert row['open'] == pytest.approx(50)
    assert row['high'] == pytest.approx(52.5)
    assert row['low'] == pytest.approx(49.5)


def test_adjust_ohlc_for_splits_noop_without_adjusted_close():
    """ Missing adjusted_close/close should pass the DataFrame through unchanged """
    df = pd.DataFrame({'open': [100], 'high': [105], 'low': [99]})

    result = fe.adjust_ohlc_for_splits(df)

    assert result['open'].iloc[0] == 100


def test_load_price_panel_skips_empty_files(tmp_path):
    """ An empty CSV (failed/interrupted download) should be skipped, not crash """
    # Arrange: one good file, one empty file
    (tmp_path / "AAPL.csv").write_text(
        "date,open,high,low,close,adjusted_close,volume\n"
        "2024-01-02,100,105,99,104,103.5,1000000\n"
    )
    (tmp_path / "EMPTY.csv").write_text("")

    # Act
    df = fe.load_price_panel(str(tmp_path))

    # Assert
    assert df['Symbol'].unique().tolist() == ['AAPL']


def test_merge_sector_map():
    """ Rows with a known sector should keep it; rows with no sector match should be dropped """
    # Arrange
    price_df = pd.DataFrame({
        'Date': ['2024-01-02', '2024-01-02'],
        'Symbol': ['AAPL', 'UNKNOWN'],
        'Close': [104.0, 50.0],
    })
    metadata_df = pd.DataFrame({
        'Symbol': ['AAPL'],
        'Sector': ['Information Technology'],
    })

    # Act
    result = fe.merge_sector_map(price_df, metadata_df)

    # Assert: AAPL kept with its sector, UNKNOWN dropped (no sector match)
    assert result['Symbol'].tolist() == ['AAPL']
    assert result.iloc[0]['Sector'] == 'Information Technology'


def test_compute_momentum_features():
    """ Return_Nd should be the N-period pct_change of Close, per ticker """
    # Arrange: 3 rows of a rising price series for one ticker
    price_df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Symbol': ['AAPL', 'AAPL', 'AAPL'],
        'Close': [100.0, 110.0, 121.0],
    })

    # Act
    result = fe.compute_momentum_features(price_df, windows=(1, 2))

    # Assert
    assert 'Return_1d' in result.columns
    assert 'Return_2d' in result.columns
    assert result.iloc[1]['Return_1d'] == pytest.approx(0.10)
    assert result.iloc[2]['Return_2d'] == pytest.approx(0.21)


def test_compute_momentum_features_does_not_leak_across_tickers():
    """ pct_change must be computed per ticker, not across the whole panel """
    # Arrange: two tickers interleaved in the panel
    price_df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
        'Symbol': ['AAPL', 'MSFT', 'AAPL', 'MSFT'],
        'Close': [100.0, 200.0, 110.0, 190.0],
    })

    # Act
    result = fe.compute_momentum_features(price_df, windows=(1,))

    # Assert: first row of each ticker has no prior value -> NaN, not a cross-ticker ratio
    first_rows = result[result['Date'] == '2024-01-01']
    assert first_rows['Return_1d'].isna().all()


def test_compute_sector_rank():
    """ Sector_Rank_Nd should percentile-rank Return_Nd within the same (Date, Sector) group """
    # Arrange: 3 stocks, same sector, same day, different Return_60d
    price_df = pd.DataFrame({
        'Date': ['2024-01-02'] * 3,
        'Symbol': ['A', 'B', 'C'],
        'Sector': ['Tech', 'Tech', 'Tech'],
        'Return_60d': [0.01, 0.05, 0.03],
    })

    # Act
    result = fe.compute_sector_rank(price_df, window=60)

    # Assert: B has the highest return -> rank 1.0, A the lowest -> rank ~0.33
    assert result[result['Symbol'] == 'B']['Sector_Rank_60d'].iloc[0] == 1.0
    assert result[result['Symbol'] == 'A']['Sector_Rank_60d'].iloc[0] == pytest.approx(1 / 3)


def test_compute_forward_returns():
    """ Target_T1_*_Ret should be T+1 OHLC relative to today's Close """
    # Arrange: two consecutive days for one ticker
    price_df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Symbol': ['AAPL', 'AAPL'],
        'Open': [100.0, 106.0],
        'High': [105.0, 110.0],
        'Low': [95.0, 104.0],
        'Close': [100.0, 108.0],
    })

    # Act
    result = fe.compute_forward_returns(price_df, horizon=1)

    # Assert: day 1's target is computed from day 2's OHLC vs. day 1's Close
    day1 = result[result['Date'] == '2024-01-01'].iloc[0]
    assert day1['Target_T1_Close_Ret'] == pytest.approx(0.08)
    assert day1['Target_T1_High_Ret'] == pytest.approx(0.10)
    # the last row has no future day -> NaN target
    day2 = result[result['Date'] == '2024-01-02'].iloc[0]
    assert pd.isna(day2['Target_T1_Close_Ret'])


def test_label_spike_event():
    """ Target_Spike_Class should be 1 only when the T+1 High return exceeds the threshold """
    # Arrange
    price_df = pd.DataFrame({
        'Target_T1_High_Ret': [0.03, 0.01, 0.02000001],
    })

    # Act
    result = fe.label_spike_event(price_df, threshold=0.02)

    # Assert
    assert result['Target_Spike_Class'].tolist() == [1, 0, 1]


def test_merge_events_keeps_only_announcement_days():
    """ merge_events should drop every row without a matching earnings row that day """
    # Arrange
    price_features_df = pd.DataFrame({
        'Date': [pd.Timestamp('2024-01-02').date(), pd.Timestamp('2024-01-03').date()],
        'Symbol': ['AAPL', 'AAPL'],
        'Close': [104.0, 105.0],
    })
    earnings_df = pd.DataFrame({
        'Symbol': ['AAPL'],
        'Date': ['2024-01-02'],
        'Surprise(%)': [-3.09],
    })

    # Act
    result = fe.merge_events(price_features_df, earnings_df)

    # Assert: only the announcement day survives
    assert len(result) == 1
    assert result.iloc[0]['Date'] == pd.Timestamp('2024-01-02').date()


def test_generate_earnings_driven_features_smoke(tmp_path):
    """
    Orchestrator smoke test: wires every step together end-to-end against
    small on-disk fixtures, no mocking - real integration, not exhaustive
    """
    # Arrange: a price history long enough to clear the 60d momentum window,
    # with one earnings announcement in the middle of it
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()

    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    closes = [100.0 + i * 0.5 for i in range(70)]
    price_df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': closes, 'high': [c + 1 for c in closes],
        'low': [c - 1 for c in closes], 'close': closes,
        'adjusted_close': closes, 'volume': [1000000] * 70,
    })
    price_df.to_csv(prices_dir / "AAPL.csv", index=False)

    metadata_file = tmp_path / "stock_metadata.csv"
    pd.DataFrame({'Symbol': ['AAPL'], 'Sector': ['Information Technology']}).to_csv(
        metadata_file, index=False)

    earnings_file = tmp_path / "earnings.csv"
    event_date = dates[65].strftime('%Y-%m-%d')
    pd.DataFrame({
        'Symbol': ['AAPL'], 'Date': [event_date],
        'EPS Estimate': [1.5], 'Reported EPS': [1.6], 'Surprise(%)': [6.67],
    }).to_csv(earnings_file, index=False)

    output_file = tmp_path / "event_driven_features.csv"

    # Act
    result_df = fe.generate_earnings_driven_features(
        prices_dir=str(prices_dir), metadata_file=str(metadata_file),
        earnings_file=str(earnings_file), output_file=str(output_file))

    # Assert: the one earnings day survives, with momentum + target columns populated
    assert output_file.exists()
    assert len(result_df) == 1
    row = result_df.iloc[0]
    assert row['Symbol'] == 'AAPL'
    assert row['Sector'] == 'Information Technology'
    assert not pd.isna(row['Return_60d'])
    assert not pd.isna(row['Target_Spike_Class'])
