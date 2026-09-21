import os
import numpy as np
import pandas as pd
from src.utils import experiment_log

DEFAULT_RETURN_COL = 'Target_T1_Close_Ret'


def filter_trade_signals(predictions_df, confidence_threshold=0.5, proba_col='y_pred_proba'):
    """
    Pure: keep only predictions whose confidence exceeds confidence_threshold -
    the trades the strategy would actually place, out of every prediction made.
    """
    return predictions_df[predictions_df[proba_col] > confidence_threshold].reset_index(drop=True)


def compute_limit_order_returns(trades_df, threshold,
                                 high_col='Target_T1_High_Ret',
                                 close_col='Target_T1_Close_Ret'):
    """
    Pure: an alternative to always exiting at Close - simulates a limit
    sell order placed at entry price * (1 + threshold), matching what a
    High-based Target_Spike_Class label actually promised. If the day's
    High return reaches threshold, the trade is assumed filled there
    (capped at exactly threshold); otherwise, since the target was never
    touched, the position is assumed closed at the day's actual Close
    return instead. This is the alignment fix applied to execution
    rather than to the label (compare against
    feature_engineering.label_spike_event's price_col='Close' option,
    which instead aligns the label to Close). Adds a 'Realized_Return'
    column, usable as compute_daily_portfolio_returns'/
    compute_backtest_kpis' return_col.
    """
    result = trades_df.copy()
    hit = result[high_col] >= threshold
    result['Realized_Return'] = result[close_col].where(~hit, threshold)
    return result


def compute_payoff_stats(trades_df, return_col=DEFAULT_RETURN_COL):
    """
    Pure: average winning-trade return, average losing-trade return, and
    the payoff ratio (avg_win / abs(avg_loss)) - the piece win_rate alone
    can't show. A low win rate can still be profitable if avg_win is
    large enough relative to avg_loss (and vice versa); §5.4's debunked
    triple-barrier result showed this ratio, not win rate, was what
    actually drove that config's return. payoff_ratio is None if there
    are no losing trades (division undefined, not "infinitely good").
    """
    wins = trades_df[trades_df[return_col] > 0][return_col]
    losses = trades_df[trades_df[return_col] <= 0][return_col]

    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else None

    return {'avg_win': avg_win, 'avg_loss': avg_loss, 'payoff_ratio': payoff_ratio}


def exclude_top_n_trades(trades_df, n, return_col=DEFAULT_RETURN_COL):
    """
    Pure: removes the n trades with the highest return_col values - a
    robustness check for whether a backtest's total return depends on a
    handful of extreme winners rather than a broad, repeatable edge
    (generalising the controlled comparison §5.4 used to debunk the
    triple-barrier result to any configuration, not just that one).
    """
    return trades_df.sort_values(return_col, ascending=False).iloc[n:].reset_index(drop=True)


def compute_tranche_portfolio_returns(trades_df, horizon, n_tranches=None,
                                       transaction_cost=0.001, return_col=DEFAULT_RETURN_COL):
    """
    Pure: the correct alternative to compute_daily_portfolio_returns for
    any horizon > 1. compute_daily_portfolio_returns implicitly assumes a
    trade's capital is free again the very next trading day - true for a
    T+1 strategy, but wrong for a longer holding period, where the same
    capital would otherwise be double-counted as open in multiple
    overlapping positions at once (an implicit, unlimited-leverage
    assumption). Here, capital is split into n_tranches (defaulting to
    horizon) equal slices, each able to hold one open position at a time;
    trades are processed in chronological Date order and assigned to
    whichever tranche is free (its previous position, if any, has already
    resolved `horizon` business days after it opened) - if every tranche
    is occupied, the trade is dropped (no capital available), not
    force-fitted, matching what a capital-constrained strategy would
    actually do. horizon is treated as business days, an approximation of
    the trading-day horizon compute_forward_returns actually used - this
    function only sees the event/signal days already filtered into
    trades_df, not a full trading calendar to count exact trading days
    against. Returns (daily_returns_df, accepted_trades_df) - the second
    df (the trades tranche capital actually allowed to be taken) is what
    compute_backtest_kpis' win_rate should be computed against, not every
    signal filter_trade_signals produced.
    """
    n_tranches = n_tranches or horizon
    sorted_df = trades_df.sort_values('Date').reset_index(drop=True).copy()

    if sorted_df.empty:
        return (pd.DataFrame(columns=['Date', 'Daily_Return']),
                sorted_df.iloc[0:0])

    dates = pd.to_datetime(sorted_df['Date'])
    free_at = [pd.Timestamp.min] * n_tranches
    accepted_mask = []

    for date in dates:
        free_idx = next((i for i, t in enumerate(free_at) if t <= date), None)
        if free_idx is None:
            accepted_mask.append(False)
            continue
        free_at[free_idx] = date + pd.tseries.offsets.BDay(horizon)
        accepted_mask.append(True)

    accepted_df = sorted_df[accepted_mask].reset_index(drop=True)

    if accepted_df.empty:
        return pd.DataFrame(columns=['Date', 'Daily_Return']), accepted_df

    weight = 1 / n_tranches
    net_return = accepted_df[return_col] - transaction_cost
    accepted_df = accepted_df.assign(Weighted_Return=net_return * weight)

    daily = accepted_df.groupby('Date')['Weighted_Return'].sum().reset_index()
    daily = daily.rename(columns={'Weighted_Return': 'Daily_Return'})

    return daily.sort_values('Date').reset_index(drop=True), accepted_df


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


def bootstrap_backtest_metrics(trades_df, n_bootstrap=2000, seed=42, transaction_cost=0.001,
                                return_col=DEFAULT_RETURN_COL, max_position_weight=None,
                                initial_capital=100000):
    """
    Pure: bootstrap resampling (with replacement, i.i.d. at the trade level)
    of trades_df, re-run through compute_daily_portfolio_returns ->
    build_equity_curve -> compute_backtest_kpis for each of n_bootstrap
    resamples, to get an empirical distribution of
    total_return/win_rate/max_drawdown/sharpe under trade-level resampling
    uncertainty. Each resample keeps the same trade count as trades_df but
    reassigns which specific trades occurred (sampling with replacement),
    then re-groups by Date exactly as the real backtest does - a Date drawn
    twice contributes twice to that day's aggregate, same as if two
    independent signals had actually fired that day. This is an i.i.d.
    assumption at the trade level - it does not model day-to-day return
    autocorrelation, and it only captures "how much would the result move
    under a different random draw of these same trades", not the broader
    question of whether the whole strategy-selection process was itself
    overfit (see Bailey, Borwein, Lopez de Prado & Zhu 2014, "The
    Probability of Backtest Overfitting", for that separate question).
    Returns a DataFrame with one row per resample and columns
    ['total_return', 'win_rate', 'max_drawdown', 'sharpe'].
    """
    rng = np.random.default_rng(seed)
    n = len(trades_df)
    rows = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        resampled = trades_df.iloc[sample_idx].reset_index(drop=True)
        daily_returns_df = compute_daily_portfolio_returns(
            resampled, transaction_cost, return_col, max_position_weight)
        equity_curve_df = build_equity_curve(daily_returns_df, initial_capital)
        rows.append(compute_backtest_kpis(equity_curve_df, resampled, return_col))

    return pd.DataFrame(rows)


def compute_confidence_interval(values, ci=0.95):
    """
    Pure: percentile confidence interval - the (lower, upper) bounds
    spanning the central `ci` fraction of the empirical distribution
    `values` (e.g. ci=0.95 keeps the 2.5th-97.5th percentiles, dropping the
    2.5% most extreme values on each tail). Works on any array-like of
    values, most commonly one column of bootstrap_backtest_metrics'
    output.
    """
    alpha = (1 - ci) / 2
    lower = float(np.percentile(values, alpha * 100))
    upper = float(np.percentile(values, (1 - alpha) * 100))
    return lower, upper


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


def run_tranche_backtest(
    predictions_df,
    run_id,
    horizon,
    confidence_threshold=0.5,
    transaction_cost=0.001,
    initial_capital=100000,
    return_col=DEFAULT_RETURN_COL,
    n_tranches=None,
    log_file=experiment_log.DEFAULT_LOG_FILE,
):
    """
    Orchestrator: filter_trade_signals -> compute_tranche_portfolio_returns
    -> build_equity_curve -> compute_backtest_kpis, then upserts the result
    onto run_id's row via experiment_log.log_backtest_result(). The
    capital-constrained counterpart to run_event_driven_backtest - use this
    instead whenever horizon > 1, since run_event_driven_backtest's
    day-by-day compounding assumes a position resolves the very next
    trading day, which only holds for horizon=1. compute_backtest_kpis'
    win_rate is scored against the trades tranche capital actually
    accepted, not every signal filter_trade_signals produced - a trade
    dropped for lack of free capital was never actually taken.
    """
    trades_df = filter_trade_signals(predictions_df, confidence_threshold)
    daily_returns_df, accepted_df = compute_tranche_portfolio_returns(
        trades_df, horizon, n_tranches, transaction_cost, return_col)
    equity_curve_df = build_equity_curve(daily_returns_df, initial_capital)
    metrics = compute_backtest_kpis(equity_curve_df, accepted_df, return_col)

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
