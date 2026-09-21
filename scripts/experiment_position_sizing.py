"""
Portfolio sizing justification: every other experiment in this notebook
fixes MAX_POSITION_WEIGHT=0.2 without ever testing whether that specific
cap is actually a good choice - it was carried over unexamined from the
report's earlier drawdown-mitigation experiment (which tested it on a
different, pre-7%-threshold configuration). Does 0.2 hold up as the right
choice for the actual production model, or would a tighter/looser cap do
better?

    ./venv/bin/python scripts/experiment_position_sizing.py

Trains the production model config exactly once (position sizing is a
backtest-time decision, not a training-time one, so the same
predictions_df is reused for every position-weight level - the model
itself, and therefore win_rate/precision/payoff_ratio, cannot change
across this sweep). Sweeps max_position_weight over
{0.05, 0.1, 0.2, 0.33, 0.5, 1.0} (1.0 is equivalent to no cap at all,
since a day's natural equal weight 1/n_trades is already <= 1.0) and
reports Tier 2 (return/drawdown/Sharpe) at each level.

Each level is logged to data/4_experiments/experiment_log.csv
(experiment='position_sizing').
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
POSITION_WEIGHTS = [0.05, 0.1, 0.2, 0.33, 0.5, 1.0]


def run_weight(weight, predictions_df, n_rows):
    run_id = experiment_log.new_run_id()

    log_config = {'universe': 'all_incl_delisted', 'n_rows': n_rows,
                  'spike_threshold': SPIKE_THRESHOLD, 'experiment': 'position_sizing',
                  'max_position_weight': weight}
    experiment_log.log_training_run(run_id, log_config, {})

    backtest_metrics = backtesting.run_event_driven_backtest(
        predictions_df, run_id, max_position_weight=weight,
        log_file=experiment_log.DEFAULT_LOG_FILE)

    return {'max_position_weight': weight, **backtest_metrics}


def main():
    features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD)
    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))
    _, universe_df = ai_training.split_by_universe(features_df, master_ticker_df)

    X_train, y_train, X_test, y_test = ai_training.split_train_test(universe_df, SPLIT_DATE)
    model = ai_training.build_model({})
    model, y_pred, y_pred_proba = ai_training.train_model(
        model, X_train, y_train, X_test, y_test)

    predictions_df = universe_df.loc[
        X_test.index, ['Date', 'Symbol', 'Target_T1_Close_Ret']].copy()
    predictions_df['y_pred_proba'] = y_pred_proba
    predictions_df = predictions_df.dropna(subset=['Target_T1_Close_Ret'])

    trades_df = backtesting.filter_trade_signals(predictions_df)
    trades_per_day = trades_df.groupby('Date').size()
    print(f"{len(trades_df)} trades across {trades_df['Date'].nunique()} unique days "
          f"(median {trades_per_day.median():.0f}, max {trades_per_day.max()} trades/day)\n")

    payoff = backtesting.compute_payoff_stats(trades_df)
    print("Payoff structure (identical at every position-weight level - computed "
          "from raw trade outcomes, not the weighted equity curve):")
    print(f"  avg_win={payoff['avg_win']:.4%}  avg_loss={payoff['avg_loss']:.4%}  "
          f"payoff_ratio={payoff['payoff_ratio']:.4f}\n")

    results = [run_weight(w, predictions_df, len(universe_df)) for w in POSITION_WEIGHTS]

    print(f"{'max_position_weight':>20s}  {'return':>9s}  {'win_rate':>9s}  "
          f"{'max_dd':>9s}  {'sharpe':>7s}")
    for r in results:
        label = f"{r['max_position_weight']:.2f}" + (" (no cap)" if r['max_position_weight'] == 1.0 else "")
        print(f"{label:>20s}  {r['total_return']:9.4%}  {r['win_rate']:9.4%}  "
              f"{r['max_drawdown']:9.4%}  {r['sharpe']:7.4f}")


if __name__ == "__main__":
    main()
