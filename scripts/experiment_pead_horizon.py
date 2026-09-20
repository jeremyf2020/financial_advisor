"""
PEAD horizon alignment: the production model predicts and trades on a
next-day (T+1) horizon, but PEAD (Post-Earnings Announcement Drift) in the
literature (Foster, Olsen & Shevlin, 1984; Bernard & Thomas, 1989, 1990)
refers to a drift that persists for roughly 60 trading days after an
earnings surprise, not a next-day effect. Does the project's result hold
up once the label and the execution/holding horizon are both realigned to
a horizon closer to the literature - here, T+20 (about one month, a
partial step toward the full ~60-day drift window) - instead of T+1?

    ./venv/bin/python scripts/experiment_pead_horizon.py

Two configs, same 7% threshold, same universe, same model hyperparameters -
only the horizon and label/execution alignment change. No src changes were
needed for this - generate_earnings_driven_features/label_spike_event/
compute_forward_returns already take an arbitrary forward_horizon, and
backtesting's functions already take a return_col override, so a T+20
config is just a different set of arguments to already-tested code.

  A. T+1 (current production) - High-based label, Close-based execution
     (the existing mismatch, unchanged here - this config is the
     reference point, not the subject of this experiment).
  D. T+20 aligned - label AND execution both use the T+20 Close return,
     the closest approximation to the literature's drift window this
     project's data supports without adding a dedicated longer-horizon
     feature set.

For each: Tier 1 (precision/gap), Tier 2 (backtest), and the same
payoff/outlier-robustness check used throughout this notebook - a config
this far from the production setup needs the same scrutiny before its
result is trusted, not less.

Each run is logged to data/4_experiments/experiment_log.csv
(experiment='pead_horizon').
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
from src.transform import feature_engineering  # noqa: E402
from src.ai import ai_training, backtesting  # noqa: E402
from src.utils import experiment_log  # noqa: E402

SPIKE_THRESHOLD = 0.07
SPLIT_DATE = "2023-01-01"
MAX_POSITION_WEIGHT = 0.2
OUTLIER_CHECK_NS = (5, 10)
HORIZON_DAYS = 20


def build_universe(forward_horizon, label_price_col):
    features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD, forward_horizon=forward_horizon,
        label_price_col=label_price_col)
    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))
    _, universe_df = ai_training.split_by_universe(features_df, master_ticker_df)
    return universe_df


def run_config(label, forward_horizon, universe_df):
    """ Trains the production model config on universe_df, backtests it on
    that horizon's own Close return column, and runs the outlier-robustness
    check - same methodology as every other config in this notebook. """
    return_col = f'Target_T{forward_horizon}_Close_Ret'
    run_id = experiment_log.new_run_id()

    X_train, y_train, X_test, y_test = ai_training.split_train_test(
        universe_df, SPLIT_DATE)
    model = ai_training.build_model({})
    model, y_pred, y_pred_proba = ai_training.train_model(
        model, X_train, y_train, X_test, y_test)
    metrics = ai_training.evaluate_predictions(y_test, y_pred, y_pred_proba)

    train_pred = model.predict(X_train)
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    train_metrics = ai_training.evaluate_predictions(y_train, train_pred, train_pred_proba)
    metrics['train_accuracy'] = train_metrics['accuracy']
    metrics['train_precision'] = train_metrics['precision']

    log_config = {'universe': 'all_incl_delisted', 'n_rows': len(universe_df),
                  'spike_threshold': SPIKE_THRESHOLD, 'experiment': 'pead_horizon',
                  'label': label, 'forward_horizon': forward_horizon}
    experiment_log.log_training_run(run_id, log_config, metrics)

    predictions_df = universe_df.loc[X_test.index, ['Date', 'Symbol', return_col]].copy()
    predictions_df['y_pred_proba'] = y_pred_proba
    predictions_df = predictions_df.dropna(subset=[return_col])

    n_trades = int((predictions_df['y_pred_proba'] > 0.5).sum())
    if n_trades == 0:
        backtest_metrics = {'total_return': 0.0, 'win_rate': 0.0,
                             'max_drawdown': 0.0, 'sharpe': 0.0}
        experiment_log.log_backtest_result(run_id, backtest_metrics)
        payoff = {'avg_win': float('nan'), 'avg_loss': float('nan'), 'payoff_ratio': None}
        robustness = {f'return_excl_top{n}': float('nan') for n in OUTLIER_CHECK_NS}
    else:
        backtest_metrics = backtesting.run_event_driven_backtest(
            predictions_df, run_id, return_col=return_col,
            max_position_weight=MAX_POSITION_WEIGHT, log_file=experiment_log.DEFAULT_LOG_FILE)
        trades_df = backtesting.filter_trade_signals(predictions_df)
        payoff = backtesting.compute_payoff_stats(trades_df, return_col=return_col)

        robustness = {}
        for n in OUTLIER_CHECK_NS:
            trimmed_df = backtesting.exclude_top_n_trades(trades_df, n, return_col=return_col)
            if trimmed_df.empty:
                robustness[f'return_excl_top{n}'] = float('nan')
                continue
            daily_df = backtesting.compute_daily_portfolio_returns(
                trimmed_df, return_col=return_col, max_position_weight=MAX_POSITION_WEIGHT)
            equity_df = backtesting.build_equity_curve(daily_df)
            kpis = backtesting.compute_backtest_kpis(equity_df, trimmed_df, return_col=return_col)
            robustness[f'return_excl_top{n}'] = kpis['total_return']

    gap = metrics['train_precision'] - metrics['precision']
    return {'config': label, 'test_precision': metrics['precision'],
            'train_precision': metrics['train_precision'], 'gap': gap,
            'n_trades': n_trades, **backtest_metrics, **payoff, **robustness}


def main():
    universe_a = build_universe(forward_horizon=1, label_price_col='High')
    universe_d = build_universe(forward_horizon=HORIZON_DAYS, label_price_col='Close')

    results = [
        run_config('A. T+1 (current production, High-based label)', 1, universe_a),
        run_config(f'D. T+{HORIZON_DAYS} aligned (Close-based label + T+{HORIZON_DAYS} execution)',
                   HORIZON_DAYS, universe_d),
    ]

    def fmt(x, pct=False):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return f"{'n/a':>9s}"
        return f"{x:9.4%}" if pct else f"{x:9.4f}"

    print("--- Summary: precision / gap / return / risk ---")
    header = (f"{'config':52s}  {'precision':>9s}  {'gap':>7s}  {'n_trades':>8s}  "
              f"{'return':>9s}  {'win_rate':>9s}  {'max_dd':>9s}  {'sharpe':>7s}")
    print(header)
    for r in results:
        print(f"{r['config']:52s}  {fmt(r['test_precision'])}  {fmt(r['gap'])[:7]:>7s}  "
              f"{r['n_trades']:8d}  {r['total_return']:9.4%}  {r['win_rate']:9.4%}  "
              f"{r['max_drawdown']:9.4%}  {r['sharpe']:7.4f}")

    print("\n--- Summary: payoff structure + outlier robustness ---")
    header2 = (f"{'config':52s}  {'avg_win':>9s}  {'avg_loss':>9s}  {'payoff':>7s}  " +
               "  ".join(f"{f'excl_top{n}':>10s}" for n in OUTLIER_CHECK_NS))
    print(header2)
    for r in results:
        payoff_str = fmt(r['payoff_ratio'])
        excl_str = "  ".join(fmt(r[f'return_excl_top{n}'], pct=True) for n in OUTLIER_CHECK_NS)
        print(f"{r['config']:52s}  {fmt(r['avg_win'], pct=True)}  {fmt(r['avg_loss'], pct=True)}  "
              f"{payoff_str[:7]:>7s}  {excl_str}")


if __name__ == "__main__":
    main()
