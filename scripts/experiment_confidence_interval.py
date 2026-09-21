"""
Confidence intervals: every Tier 2 metric reported so far in this project
(total_return/win_rate/max_drawdown/sharpe) has been a single point
estimate - +14.18% return, 2.59 Sharpe, etc. - with no sense of how much
that number would move under a different draw of the same underlying
trades. Is the production model's headline result a precise estimate, or
does it sit inside a wide band that could plausibly include zero (or
negative) as well?

    ./venv/bin/python scripts/experiment_confidence_interval.py

Trains the production model config exactly once (n_bootstrap=2000
resamples all draw from the same fixed trades_df - the model itself does
not change across this sweep, only which trades happen to be sampled).
Reports the actual point estimate from the real backtest, then the 95%
bootstrap confidence interval for each Tier 2 metric via
backtesting.bootstrap_backtest_metrics/compute_confidence_interval (i.i.d.
resampling at the trade level - see that function's docstring for the
caveats this does and does not cover).

Logs the single actual (point-estimate) run to
data/4_experiments/experiment_log.csv (experiment='confidence_interval');
the bootstrap distribution itself is not persisted, matching the pattern
used by scripts/experiment_gap_robustness_check.py and
scripts/experiment_position_sizing.py for statistics outside the log's
fixed schema.
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
N_BOOTSTRAP = 2000
CI = 0.95


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
    print(f"{len(trades_df)} trades in the point-estimate backtest\n")

    run_id = experiment_log.new_run_id()
    log_config = {'universe': 'all_incl_delisted', 'n_rows': len(universe_df),
                  'spike_threshold': SPIKE_THRESHOLD, 'experiment': 'confidence_interval',
                  'max_position_weight': MAX_POSITION_WEIGHT, 'n_bootstrap': N_BOOTSTRAP}
    experiment_log.log_training_run(run_id, log_config, {})

    point_estimate = backtesting.run_event_driven_backtest(
        predictions_df, run_id, max_position_weight=MAX_POSITION_WEIGHT,
        log_file=experiment_log.DEFAULT_LOG_FILE)

    bootstrap_df = backtesting.bootstrap_backtest_metrics(
        trades_df, n_bootstrap=N_BOOTSTRAP, max_position_weight=MAX_POSITION_WEIGHT)

    print(f"{'metric':>13s}  {'point est.':>11s}  {(f'{CI:.0%} CI lower'):>13s}  "
          f"{(f'{CI:.0%} CI upper'):>13s}  {'includes 0?':>11s}")
    for metric in ['total_return', 'win_rate', 'max_drawdown', 'sharpe']:
        lower, upper = backtesting.compute_confidence_interval(bootstrap_df[metric], ci=CI)
        includes_zero = "yes" if lower <= 0 <= upper else "no"
        is_ratio = metric == 'sharpe'
        fmt = "{:13.4f}" if is_ratio else "{:13.4%}"
        point_str = f"{point_estimate[metric]:11.4f}" if is_ratio else f"{point_estimate[metric]:11.4%}"
        print(f"{metric:>13s}  {point_str}  {fmt.format(lower)}  "
              f"{fmt.format(upper)}  {includes_zero:>11s}")


if __name__ == "__main__":
    main()
