"""
Gap vs. Tier 2 tradeoff: across every run logged so far, has any
configuration achieved both a low train/test precision gap AND strong
Tier 2 (win rate / return / Sharpe) results at once - or has closing the
gap and improving financial performance been in tension throughout this
project's whole experiment history?

    ./venv/bin/python scripts/experiment_gap_robustness_check.py

Part 1 re-reads data/4_experiments/experiment_log.csv (every run logged by
every earlier experiment in this notebook) and joins each run's gap
against its own backtest result, to answer the question empirically from
the whole logged history rather than from a single sweep or assumption.

Part 2 takes whichever config currently has the lowest gap in that history
(as of writing, 'regularised+tree_complexity+early_stopping' from the
overfitting-gap sweep - reg_alpha/reg_lambda/scale_pos_weight +
max_depth=3/min_child_weight=5/subsample=0.7/colsample_bytree=0.7 + early
stopping) and applies the same outlier-robustness check (§5.4's
methodology, generalised in the baseline-comparison experiment) that
already flagged the production baseline's return as fragile - testing
directly whether this config's apparently-better result is any more
trustworthy, or the same artefact in a broader disguise.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
from src.transform import feature_engineering  # noqa: E402
from src.ai import ai_training, backtesting  # noqa: E402
from src.utils import experiment_log, experiment_charts as ec  # noqa: E402

SPIKE_THRESHOLD = 0.07
SPLIT_DATE = "2023-01-01"
VALIDATION_DATE = "2022-01-01"
MAX_POSITION_WEIGHT = 0.2
OUTLIER_CHECK_NS = (5, 10, 20)


def main():
    # --- Part 1: does gap vs. Tier 2 trade off across every logged run? ---
    log_df = experiment_log.load_experiment_log()
    gap_vs_backtest_df = ec.compute_gap_vs_backtest(log_df, metric='precision')

    print(f"{len(gap_vs_backtest_df)} logged runs have both a gap and a Tier 2 result\n")
    print("--- Top 5 lowest-gap runs (across every experiment run so far) ---")
    print(gap_vs_backtest_df.sort_values('gap').head(5).to_string(index=False))
    print()
    print("--- Correlation between gap and each Tier 2 metric, across all logged runs ---")
    for col in ['win_rate', 'total_return', 'sharpe']:
        corr = gap_vs_backtest_df['gap'].corr(gap_vs_backtest_df[col])
        print(f"  gap vs {col}: {corr:.4f}")
    print()

    # --- Part 2: is the lowest-gap config's return genuine, or fragile? ---
    features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD)
    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))
    _, universe_df = ai_training.split_by_universe(features_df, master_ticker_df)

    _, y_train, _, _ = ai_training.split_train_test(universe_df, SPLIT_DATE)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    low_gap_config = {
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'scale_pos_weight': scale_pos_weight,
        'max_depth': 3, 'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'n_estimators': 500, 'early_stopping_rounds': 20,
    }

    run_id = experiment_log.new_run_id()
    model, metrics, extras = ai_training.train_xgboost_event_model(
        features_df=universe_df, split_date=SPLIT_DATE,
        validation_date=VALIDATION_DATE, config=low_gap_config)
    X_test, y_test, y_pred, y_pred_proba = extras
    gap = metrics['train_precision'] - metrics['precision']

    log_config = {**low_gap_config, 'universe': 'all_incl_delisted', 'n_rows': len(universe_df),
                  'spike_threshold': SPIKE_THRESHOLD, 'experiment': 'gap_robustness_check',
                  'label': 'lowest_gap_config'}
    experiment_log.log_training_run(run_id, log_config, metrics)

    predictions_df = universe_df.loc[
        X_test.index, ['Date', 'Symbol', 'Target_T1_Close_Ret']].copy()
    predictions_df['y_pred_proba'] = y_pred_proba
    predictions_df = predictions_df.dropna(subset=['Target_T1_Close_Ret'])

    backtest_metrics = backtesting.run_event_driven_backtest(
        predictions_df, run_id, max_position_weight=MAX_POSITION_WEIGHT,
        log_file=experiment_log.DEFAULT_LOG_FILE)
    trades_df = backtesting.filter_trade_signals(predictions_df)
    payoff = backtesting.compute_payoff_stats(trades_df)

    print(f"Lowest-gap config: test_precision={metrics['precision']:.4f}  "
          f"train_precision={metrics['train_precision']:.4f}  gap={gap:.4f}")
    print(f"  n_trades={len(trades_df)}  total_return={backtest_metrics['total_return']:.4%}  "
          f"win_rate={backtest_metrics['win_rate']:.4%}  sharpe={backtest_metrics['sharpe']:.4f}")
    print(f"  payoff_ratio={payoff['payoff_ratio']:.4f}  "
          f"avg_win={payoff['avg_win']:.4%}  avg_loss={payoff['avg_loss']:.4%}")
    print()
    print("  Outlier-robustness (return after excluding the top N highest-return trades):")
    for n in OUTLIER_CHECK_NS:
        trimmed_df = backtesting.exclude_top_n_trades(trades_df, n)
        if trimmed_df.empty:
            print(f"    excl top {n}: n/a (no trades left)")
            continue
        daily_df = backtesting.compute_daily_portfolio_returns(
            trimmed_df, max_position_weight=MAX_POSITION_WEIGHT)
        equity_df = backtesting.build_equity_curve(daily_df)
        kpis = backtesting.compute_backtest_kpis(equity_df, trimmed_df)
        print(f"    excl top {n}: total_return={kpis['total_return']:.4%}  sharpe={kpis['sharpe']:.4f}")


if __name__ == "__main__":
    main()
