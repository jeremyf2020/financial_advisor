"""
Probability calibration: is the production XGBoost model's predict_proba
output actually a trustworthy probability, or just a confidence-ordering
score - and does explicitly calibrating it change trading behaviour?

    ./venv/bin/python scripts/experiment_probability_calibration.py

XGBoost's predict_proba is optimised for log-loss on the training
objective, not for calibration - there is no guarantee a predicted
probability of 0.7 actually means "70% of these fire" empirically. This
matters here specifically because filter_trade_signals() fires a trade
whenever predict_proba > 0.5: if the raw score is systematically over- or
under-confident, the 0.5 threshold selects a different, wrong set of
trades regardless of how good the model's underlying ranking ability is.

Three configs, same 7% threshold, same universe, same train/test split as
the production model:
  A. XGBoost (current production, uncalibrated)
  B. XGBoost + Platt scaling  (CalibratedClassifierCV(method='sigmoid'))
  C. XGBoost + isotonic calibration (CalibratedClassifierCV(method='isotonic'))

For each: Tier 1 precision, Brier score (lower is better-calibrated), a
reliability-diagram table (mean predicted probability vs. actual observed
frequency per bin), and a full Tier 2 backtest - calibration only matters
here if it changes which trades fire and/or the financial result, not
just the Brier score in isolation.

Each run is logged to data/4_experiments/experiment_log.csv
(experiment='probability_calibration').
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
CALIBRATION_CV = 5
N_BINS = 10


def run_config(label, build_fn, config, universe_df):
    """ Trains build_fn(config), evaluates calibration + backtests it, logs it. """
    run_id = experiment_log.new_run_id()

    X_train, y_train, X_test, y_test = ai_training.split_train_test(
        universe_df, SPLIT_DATE)
    model = build_fn(config)
    model, y_pred, y_pred_proba = ai_training.train_model(
        model, X_train, y_train, X_test, y_test)
    metrics = ai_training.evaluate_predictions(y_test, y_pred, y_pred_proba)
    brier = ai_training.compute_brier_score(y_test, y_pred_proba)
    calibration_df = ai_training.compute_calibration_curve(
        y_test, y_pred_proba, n_bins=N_BINS)

    log_config = {'universe': 'all_incl_delisted', 'n_rows': len(universe_df),
                  'spike_threshold': SPIKE_THRESHOLD, 'experiment': 'probability_calibration',
                  'label': label, 'brier_score': brier}
    experiment_log.log_training_run(run_id, log_config, metrics)

    predictions_df = universe_df.loc[
        X_test.index, ['Date', 'Symbol', 'Target_T1_Close_Ret']].copy()
    predictions_df['y_pred_proba'] = y_pred_proba
    predictions_df = predictions_df.dropna(subset=['Target_T1_Close_Ret'])

    n_trades = int((predictions_df['y_pred_proba'] > 0.5).sum())
    if n_trades == 0:
        backtest_metrics = {'total_return': 0.0, 'win_rate': 0.0,
                             'max_drawdown': 0.0, 'sharpe': 0.0}
        experiment_log.log_backtest_result(run_id, backtest_metrics)
    else:
        backtest_metrics = backtesting.run_event_driven_backtest(
            predictions_df, run_id, max_position_weight=MAX_POSITION_WEIGHT,
            log_file=experiment_log.DEFAULT_LOG_FILE)

    result = {'config': label, 'test_precision': metrics['precision'],
              'brier_score': brier, 'n_trades': n_trades, **backtest_metrics}
    return result, calibration_df


def main():
    features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD)
    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))
    _, universe_df = ai_training.split_by_universe(features_df, master_ticker_df)

    configs = [
        ('A. XGBoost (uncalibrated, production)', ai_training.build_model, {}),
        ('B. XGBoost + Platt scaling (sigmoid)', ai_training.build_calibrated_model,
         {'method': 'sigmoid', 'cv': CALIBRATION_CV}),
        ('C. XGBoost + isotonic calibration', ai_training.build_calibrated_model,
         {'method': 'isotonic', 'cv': CALIBRATION_CV}),
    ]

    results = []
    for label, build_fn, config in configs:
        r, calibration_df = run_config(label, build_fn, config, universe_df)
        results.append(r)

        print(f"{label}")
        print(f"  test_precision={r['test_precision']:.4f}  brier_score={r['brier_score']:.4f}  "
              f"n_trades={r['n_trades']}")
        print(f"  total_return={r['total_return']:.4%}  win_rate={r['win_rate']:.4%}  "
              f"max_drawdown={r['max_drawdown']:.4%}  sharpe={r['sharpe']:.4f}")
        print("  Reliability diagram (mean_predicted vs. fraction_positive):")
        print(calibration_df.to_string(index=False))
        print()

    print("--- Summary ---")
    header = (f"{'config':42s}  {'precision':>9s}  {'brier':>7s}  {'n_trades':>8s}  "
              f"{'return':>9s}  {'win_rate':>9s}  {'max_dd':>9s}  {'sharpe':>7s}")
    print(header)
    for r in results:
        print(f"{r['config']:42s}  {r['test_precision']:9.4f}  {r['brier_score']:7.4f}  "
              f"{r['n_trades']:8d}  {r['total_return']:9.4%}  {r['win_rate']:9.4%}  "
              f"{r['max_drawdown']:9.4%}  {r['sharpe']:7.4f}")


if __name__ == "__main__":
    main()
