"""
Train the placeholder "production" model and persist it to disk, one command:

    ./venv/bin/python scripts/train_and_save_model.py

Regenerates data/3_features/event_driven_features.csv at SPIKE_THRESHOLD
(do not trust the file's on-disk state - it may have been left at a
different threshold by an earlier experiment), trains on the
all-including-delisted universe, backtests, logs both to
data/4_experiments/experiment_log.csv, and saves the fitted model to
data/5_models/ via src.ai.model_persistence. The web API loads this file
at startup - it is never retrained per request.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
from src.transform import feature_engineering  # noqa: E402
from src.ai import ai_training, backtesting, model_persistence  # noqa: E402
from src.utils import experiment_log  # noqa: E402

SPIKE_THRESHOLD = 0.07
MAX_POSITION_WEIGHT = 0.2


def main():
    features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD)

    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))
    _, universe_df = ai_training.split_by_universe(features_df, master_ticker_df)

    run_id = experiment_log.new_run_id()
    log_config = {
        'universe': 'all_incl_delisted', 'n_rows': len(universe_df),
        'spike_threshold': SPIKE_THRESHOLD, 'max_position_weight': MAX_POSITION_WEIGHT,
    }

    model, metrics, extras = ai_training.train_xgboost_event_model(
        features_df=universe_df, config={})
    X_test, y_test, y_pred, y_pred_proba = extras
    experiment_log.log_training_run(run_id, log_config, metrics)

    predictions_df = universe_df.loc[
        X_test.index, ['Date', 'Symbol', backtesting.DEFAULT_RETURN_COL]].copy()
    predictions_df['y_pred_proba'] = y_pred_proba
    predictions_df = predictions_df.dropna(subset=[backtesting.DEFAULT_RETURN_COL])

    backtest_metrics = backtesting.run_event_driven_backtest(
        predictions_df, run_id, max_position_weight=MAX_POSITION_WEIGHT)

    print(f"Run {run_id}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}  Precision: {metrics['precision']:.4f}")
    print(f"  Total Return: {backtest_metrics['total_return']:.4%}")
    print(f"  Win Rate: {backtest_metrics['win_rate']:.4%}")
    print(f"  Max Drawdown: {backtest_metrics['max_drawdown']:.4%}")
    print(f"  Sharpe: {backtest_metrics['sharpe']:.4f}")

    bundle = model_persistence.build_model_bundle(
        model=model, feature_cols=ai_training.DEFAULT_FEATURES,
        target_col=ai_training.TARGET_COLUMN, config={**log_config, **backtest_metrics},
        run_id=run_id, metrics={**metrics, **backtest_metrics})
    model_file = model_persistence.save_model(bundle)
    print(f"Saved production model to {model_file}")


if __name__ == "__main__":
    main()
