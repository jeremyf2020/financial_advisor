import pytest
import pandas as pd
from src.ai import backtesting as bt
from src.utils import experiment_log


def make_predictions_df():
    """ Small hand-built prediction table: Date, Symbol, actual T+1 return, confidence """
    return pd.DataFrame({
        'Date': ['2022-01-01', '2022-01-01', '2022-01-02', '2022-01-03', '2022-01-03'],
        'Symbol': ['A', 'B', 'C', 'D', 'E'],
        'Target_T1_Close_Ret': [0.05, -0.02, 0.03, -0.04, 0.06],
        'y_pred_proba': [0.9, 0.8, 0.3, 0.7, 0.6],
    })


def test_filter_trade_signals():
    """ Only rows with confidence above the threshold should survive """
    # Arrange
    df = make_predictions_df()

    # Act
    trades_df = bt.filter_trade_signals(df, confidence_threshold=0.5)

    # Assert: 'C' (0.3) is filtered out, the rest (>= 0.6) remain
    assert sorted(trades_df['Symbol']) == ['A', 'B', 'D', 'E']

def test_compute_limit_order_returns_fills_at_threshold_when_high_touches_it():
    """ If High return reaches the threshold, Realized_Return is capped at
    exactly threshold - not the (possibly much larger) actual High return """
    trades_df = pd.DataFrame({
        'Target_T1_High_Ret': [0.09],   # touches and overshoots the 7% target
        'Target_T1_Close_Ret': [0.02],  # closed up only 2% by end of day
    })

    result = bt.compute_limit_order_returns(trades_df, threshold=0.07)

    assert result.iloc[0]['Realized_Return'] == pytest.approx(0.07)


def test_compute_limit_order_returns_falls_back_to_close_when_not_touched():
    """ If High never reaches the threshold, Realized_Return falls back to
    the actual Close return - the order was never filled """
    trades_df = pd.DataFrame({
        'Target_T1_High_Ret': [0.04],    # never reaches the 7% target
        'Target_T1_Close_Ret': [-0.01],  # closed down 1%
    })

    result = bt.compute_limit_order_returns(trades_df, threshold=0.07)

    assert result.iloc[0]['Realized_Return'] == pytest.approx(-0.01)


def test_compute_payoff_stats():
    """ avg_win/avg_loss/payoff_ratio should reflect the win/loss split of the raw returns """
    trades_df = pd.DataFrame({
        'Target_T1_Close_Ret': [0.10, 0.20, -0.05, -0.05],  # 2 wins avg 0.15, 2 losses avg -0.05
    })

    stats = bt.compute_payoff_stats(trades_df)

    assert stats['avg_win'] == pytest.approx(0.15)
    assert stats['avg_loss'] == pytest.approx(-0.05)
    assert stats['payoff_ratio'] == pytest.approx(3.0)


def test_compute_payoff_stats_no_losses_gives_none_ratio():
    """ payoff_ratio should be None (undefined), not raise, when there are no losing trades """
    trades_df = pd.DataFrame({'Target_T1_Close_Ret': [0.10, 0.05]})

    stats = bt.compute_payoff_stats(trades_df)

    assert stats['payoff_ratio'] is None


def test_exclude_top_n_trades():
    """ Should drop exactly the n highest-return rows, keeping the rest """
    trades_df = pd.DataFrame({
        'Target_T1_Close_Ret': [0.50, 0.01, -0.02, 0.30, -0.10],
    })

    result = bt.exclude_top_n_trades(trades_df, n=2)

    # the 0.50 and 0.30 rows (the two highest) should be gone
    assert sorted(result['Target_T1_Close_Ret'].tolist()) == [-0.10, -0.02, 0.01]



def test_compute_daily_portfolio_returns_averages_same_day_trades():
    """ Trades on the same Date should average into one equal-weighted return, net of cost """
    # Arrange
    trades_df = pd.DataFrame({
        'Date': ['2022-01-01', '2022-01-01'],
        'Target_T1_Close_Ret': [0.05, -0.03],
    })

    # Act
    daily_df = bt.compute_daily_portfolio_returns(
        trades_df, transaction_cost=0.001)

    # Assert: mean(0.05, -0.03) - 0.001 cost
    assert len(daily_df) == 1
    assert daily_df.iloc[0]['Daily_Return'] == pytest.approx(0.01 - 0.001)


def test_compute_daily_portfolio_returns_caps_single_trade_day():
    """
    A single-trade day would otherwise put 100% of capital on one name -
    max_position_weight should cap that trade's contribution, leaving the
    rest as idle (0%) capital.
    """
    # Arrange: one trade, one day - a big loss that would otherwise be the
    # entire day's return
    trades_df = pd.DataFrame({
        'Date': ['2022-01-01'],
        'Target_T1_Close_Ret': [-0.20],
    })

    # Act
    daily_df = bt.compute_daily_portfolio_returns(
        trades_df, transaction_cost=0.0, max_position_weight=0.2)

    # Assert: only 20% weight on the -0.20 trade, not the full 100%
    assert daily_df.iloc[0]['Daily_Return'] == pytest.approx(-0.20 * 0.2)


def test_compute_daily_portfolio_returns_cap_does_not_affect_diversified_days():
    """ A day with enough trades to already be under the cap should be unaffected """
    # Arrange: 5 trades on one day, cap = 0.2 (i.e. 1/5) - equal weight
    # already meets the cap exactly
    trades_df = pd.DataFrame({
        'Date': ['2022-01-01'] * 5,
        'Target_T1_Close_Ret': [0.05, -0.03, 0.02, -0.01, 0.04],
    })

    # Act
    uncapped = bt.compute_daily_portfolio_returns(trades_df, transaction_cost=0.0)
    capped = bt.compute_daily_portfolio_returns(
        trades_df, transaction_cost=0.0, max_position_weight=0.2)

    # Assert
    assert capped.iloc[0]['Daily_Return'] == pytest.approx(uncapped.iloc[0]['Daily_Return'])


def test_build_equity_curve_compounds_from_initial_capital():
    """ Equity should compound day over day starting from initial_capital """
    # Arrange
    daily_df = pd.DataFrame({
        'Date': ['2022-01-01', '2022-01-02'],
        'Daily_Return': [0.10, -0.05],
    })

    # Act
    equity_df = bt.build_equity_curve(daily_df, initial_capital=1000)

    # Assert
    assert equity_df.iloc[0]['Equity'] == pytest.approx(1100)
    assert equity_df.iloc[1]['Equity'] == pytest.approx(1100 * 0.95)


def test_compute_backtest_kpis():
    """ total_return/win_rate/max_drawdown/sharpe should reflect a simple two-day equity path """
    # Arrange: equity rises then falls, giving a known drawdown
    equity_df = pd.DataFrame({
        'Date': ['2022-01-01', '2022-01-02'],
        'Daily_Return': [0.10, -0.20],
        'Equity': [1100.0, 880.0],
    })
    trades_df = pd.DataFrame({
        'Target_T1_Close_Ret': [0.05, -0.03, 0.02],
    })

    # Act
    metrics = bt.compute_backtest_kpis(equity_df, trades_df)

    # Assert
    assert metrics['total_return'] == pytest.approx(880 / 1100 - 1)
    assert metrics['win_rate'] == pytest.approx(2 / 3)  # 2 of 3 trades positive
    assert metrics['max_drawdown'] == pytest.approx(880 / 1100 - 1)  # peak was day 1


def test_compute_backtest_kpis_empty_input():
    """ Empty trades/equity should return zeroed metrics, not raise """
    # Act
    metrics = bt.compute_backtest_kpis(pd.DataFrame(), pd.DataFrame())

    # Assert
    assert metrics == {'total_return': 0.0, 'win_rate': 0.0,
                        'max_drawdown': 0.0, 'sharpe': 0.0}


def test_run_event_driven_backtest_smoke(tmp_path):
    """
    Orchestrator smoke test: wires filter -> daily returns -> equity curve
    -> kpis, then upserts the result onto an existing training run's row
    """
    # Arrange: a training run must already be logged for this run_id
    log_file = tmp_path / "experiment_log.csv"
    run_id = experiment_log.new_run_id()
    experiment_log.log_training_run(
        run_id, {}, {'accuracy': 0.6, 'precision': 0.5}, log_file=str(log_file))
    predictions_df = make_predictions_df()

    # Act
    metrics = bt.run_event_driven_backtest(
        predictions_df, run_id, log_file=str(log_file))

    # Assert
    assert set(metrics.keys()) == {
        'total_return', 'win_rate', 'max_drawdown', 'sharpe'}
    logged = experiment_log.load_experiment_log(log_file=str(log_file))
    row = logged[logged['run_id'] == run_id].iloc[0]
    assert row['total_return'] == pytest.approx(metrics['total_return'])
