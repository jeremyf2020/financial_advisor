import os
import pandas as pd
from src.utils import experiment_log

DEFAULT_RETURN_COL = 'Target_T1_Close_Ret'


def filter_trade_signals(predictions_df, confidence_threshold=0.5, proba_col='y_pred_proba'):
    """
    Pure: keep only predictions whose confidence exceeds confidence_threshold -
    the trades the strategy would actually place, out of every prediction made.
    """
    return predictions_df[predictions_df[proba_col] > confidence_threshold].reset_index(drop=True)


def compute_daily_portfolio_returns(trades_df, transaction_cost=0.001,
                                     return_col=DEFAULT_RETURN_COL, max_position_weight=None):
    """
    Pure: aggregate trade-level returns into one daily portfolio return per
    Date - equal-weighted across every signal firing that day by default.
    If max_position_weight is set, no single trade can contribute more than
    that share of the day's return; unused capital sits idle (0% return)
    rather than concentrating into one name. This only changes days where
    too few signals fire to naturally diversify (e.g. a single-trade day
    would otherwise put 100% of capital on one name) - days with enough
    signals to already be under the cap are unaffected. transaction_cost is
    a single round-trip figure, subtracted once per trade.
    """
    result = trades_df.copy()
    result['Net_Return'] = result[return_col] - transaction_cost

    if max_position_weight is None:
        daily = result.groupby('Date')['Net_Return'].mean().reset_index()
        daily = daily.rename(columns={'Net_Return': 'Daily_Return'})
    else:
        trades_per_day = result.groupby('Date')['Net_Return'].transform('count')
        weight = (1 / trades_per_day).clip(upper=max_position_weight)
        result['Daily_Return'] = result['Net_Return'] * weight
        daily = result.groupby('Date')['Daily_Return'].sum().reset_index()

    return daily.sort_values('Date').reset_index(drop=True)


def build_equity_curve(daily_returns_df, initial_capital=100000):
    """
    Pure: cumulative compounding of daily portfolio returns into an equity
    curve starting at initial_capital. Only covers days a trade actually
    fired - this is an event-driven strategy, not a daily-rebalanced one.
    """
    result = daily_returns_df.sort_values('Date').reset_index(drop=True).copy()
    result['Equity'] = initial_capital * (1 + result['Daily_Return']).cumprod()

    return result


def compute_backtest_kpis(equity_curve_df, trades_df, return_col=DEFAULT_RETURN_COL):
    """
    Pure: total_return/max_drawdown/sharpe from the cost-adjusted equity
    curve; win_rate from the raw (pre-cost) per-trade outcomes - whether the
    underlying event moved favorably, independent of the execution-cost
    assumption. Sharpe is annualized assuming 252 trading days.
    """
    if equity_curve_df.empty or trades_df.empty:
        return {'total_return': 0.0, 'win_rate': 0.0,
                'max_drawdown': 0.0, 'sharpe': 0.0}

    equity = equity_curve_df['Equity']
    drawdown = equity / equity.cummax() - 1

    daily_returns = equity_curve_df['Daily_Return']
    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)

    return {
        'total_return': equity.iloc[-1] / equity.iloc[0] - 1,
        'win_rate': (trades_df[return_col] > 0).mean(),
        'max_drawdown': drawdown.min(),
        'sharpe': sharpe,
    }


def run_event_driven_backtest(
    predictions_df,
    run_id,
    confidence_threshold=0.5,
    transaction_cost=0.001,
    initial_capital=100000,
    return_col=DEFAULT_RETURN_COL,
    max_position_weight=None,
    log_file=experiment_log.DEFAULT_LOG_FILE,
):
    """
    Orchestrator: filter_trade_signals -> compute_daily_portfolio_returns ->
    build_equity_curve -> compute_backtest_kpis, then upserts the result
    onto run_id's row via experiment_log.log_backtest_result(). run_id must
    already have a log_training_run() row - a backtest with no matching
    training run is a bug, not a new experiment.
    """
    trades_df = filter_trade_signals(predictions_df, confidence_threshold)
    daily_returns_df = compute_daily_portfolio_returns(
        trades_df, transaction_cost, return_col, max_position_weight)
    equity_curve_df = build_equity_curve(daily_returns_df, initial_capital)
    metrics = compute_backtest_kpis(equity_curve_df, trades_df, return_col)

    experiment_log.log_backtest_result(run_id, metrics, log_file)

    return metrics


if __name__ == "__main__":
    from src.ai import ai_training

    features_df = pd.read_csv(os.path.join(
        "data", "3_features", "event_driven_features.csv"))

    run_id = experiment_log.new_run_id()
    model_config = {}  # baseline: plain XGBoost defaults

    model, metrics, extras = ai_training.train_xgboost_event_model(
        features_df=features_df, config=model_config)
    X_test, y_test, y_pred, y_pred_proba = extras
    experiment_log.log_training_run(run_id, model_config, metrics)

    predictions_df = features_df.loc[
        X_test.index, ['Date', 'Symbol', DEFAULT_RETURN_COL]].copy()
    predictions_df['y_pred_proba'] = y_pred_proba
    predictions_df = predictions_df.dropna(subset=[DEFAULT_RETURN_COL])

    backtest_metrics = run_event_driven_backtest(predictions_df, run_id)

    print(f"Run {run_id}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Total Return: {backtest_metrics['total_return']:.4%}")
    print(f"  Win Rate: {backtest_metrics['win_rate']:.4%}")
    print(f"  Max Drawdown: {backtest_metrics['max_drawdown']:.4%}")
    print(f"  Sharpe: {backtest_metrics['sharpe']:.4f}")
