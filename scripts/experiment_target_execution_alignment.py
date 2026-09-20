"""
Target/execution alignment experiment: compares three ways of handling the
mismatch identified in review - the production model labels a "spike" on
the T+1 intraday High return, but the backtest always exits at the T+1
Close, regardless of whether that High was ever reached.

    ./venv/bin/python scripts/experiment_target_execution_alignment.py

Three configs, same 7% threshold, same universe (all incl. delisted),
same model hyperparameters (plain XGBoost defaults) as the production
model - only the label/execution alignment changes:

  A. mismatched (current production)
     Label: High >= 7%.  Execution: always exit at Close.
     This is what's currently deployed and reported.

  B. close_aligned
     Label: Close >= 7%.  Execution: exit at Close.
     Fixes the mismatch by changing the LABEL - the model is trained to
     predict exactly the return the backtest realizes.

  C. limit_order_aligned
     Label: High >= 7% (unchanged).  Execution: a simulated limit sell
     at +7%, filled if High touches it, else exit at Close.
     Fixes the mismatch by changing EXECUTION - the backtest now trades
     what the label actually promised, instead of the label's target.

Each is trained/backtested independently and logged to
data/4_experiments/experiment_log.csv (experiment='target_execution_alignment')
so results stay traceable. Does not touch the persisted production model.
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


def run_config(label, features_df, return_col, threshold_for_limit_order=None):
    run_id = experiment_log.new_run_id()

    model, metrics, extras = ai_training.train_xgboost_event_model(
        features_df=features_df, split_date=SPLIT_DATE, config={})
    X_test, y_test, y_pred, y_pred_proba = extras

    predictions_df = features_df.loc[
        X_test.index, ['Date', 'Symbol', 'Target_T1_Close_Ret', 'Target_T1_High_Ret']].copy()
    predictions_df['y_pred_proba'] = y_pred_proba

    trades_df = backtesting.filter_trade_signals(predictions_df)

    if threshold_for_limit_order is not None:
        trades_df = backtesting.compute_limit_order_returns(
            trades_df, threshold=threshold_for_limit_order)

    trades_df = trades_df.dropna(subset=[return_col])
    daily_df = backtesting.compute_daily_portfolio_returns(
        trades_df, return_col=return_col, max_position_weight=MAX_POSITION_WEIGHT)
    equity_df = backtesting.build_equity_curve(daily_df)
    backtest_metrics = backtesting.compute_backtest_kpis(equity_df, trades_df, return_col=return_col)

    log_config = {
        'universe': 'all_incl_delisted', 'n_rows': len(features_df),
        'spike_threshold': SPIKE_THRESHOLD, 'max_position_weight': MAX_POSITION_WEIGHT,
        'experiment': 'target_execution_alignment', 'label': label,
        'return_col': return_col,
    }
    experiment_log.log_training_run(run_id, log_config, metrics)
    experiment_log.log_backtest_result(run_id, backtest_metrics)

    gap = metrics['train_precision'] - metrics['precision']
    print(f"\n[{label}]")
    print(f"  test_precision={metrics['precision']:.4f}  train_precision={metrics['train_precision']:.4f}  gap={gap:.4f}")
    print(f"  total_return={backtest_metrics['total_return']:.4%}  win_rate={backtest_metrics['win_rate']:.4%}  "
          f"max_drawdown={backtest_metrics['max_drawdown']:.4%}  sharpe={backtest_metrics['sharpe']:.4f}")

    return {'config': label, **metrics, **backtest_metrics, 'gap': gap}


def main():
    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))

    # A. mismatched (current production): High-based label
    high_features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD, label_price_col='High')
    _, high_universe_df = ai_training.split_by_universe(high_features_df, master_ticker_df)

    # B. close_aligned: Close-based label
    close_features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD, label_price_col='Close')
    _, close_universe_df = ai_training.split_by_universe(close_features_df, master_ticker_df)

    results = []
    results.append(run_config(
        "A. mismatched (current production)", high_universe_df,
        return_col='Target_T1_Close_Ret'))
    results.append(run_config(
        "B. close_aligned (label changed)", close_universe_df,
        return_col='Target_T1_Close_Ret'))
    results.append(run_config(
        "C. limit_order_aligned (execution changed)", high_universe_df,
        return_col='Realized_Return', threshold_for_limit_order=SPIKE_THRESHOLD))

    print("\n--- Summary ---")
    header = f"{'config':45s}  {'precision':>9s}  {'gap':>7s}  {'return':>9s}  {'win_rate':>9s}  {'max_dd':>9s}  {'sharpe':>7s}"
    print(header)
    for r in results:
        print(f"{r['config']:45s}  {r['precision']:9.4f}  {r['gap']:7.4f}  "
              f"{r['total_return']:9.4%}  {r['win_rate']:9.4%}  {r['max_drawdown']:9.4%}  {r['sharpe']:7.4f}")


if __name__ == "__main__":
    main()
