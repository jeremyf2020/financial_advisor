"""
Train the "production" model and persist it to disk, one command:

    ./venv/bin/python scripts/train_and_save_model.py

Regenerates data/3_features/event_driven_features.csv at SPIKE_THRESHOLD
(do not trust the file's on-disk state - it may have been left at a
different threshold by an earlier experiment), trains on the
all-including-delisted universe, backtests, logs both to
data/4_experiments/experiment_log.csv, and saves the fitted model to
data/5_models/ via src.ai.model_persistence. The web API loads this file
at startup - it is never retrained per request.

Uses the 'regularised+tree_complexity+early_stopping' config identified in
scripts/experiment_overfitting_gap.py as the lowest train/test precision
gap (0.004 vs. the plain-XGBoost baseline's 0.678) among every config
tried, while also having the highest total return and win rate of any
backtested run logged so far (see scripts/experiment_gap_robustness_check.py).
This trades away some of the baseline's Sharpe and trades far more often
(~2400 vs. 56 trades) - see the notebook write-up for that config
('Gap vs. Tier 2 tradeoff' section) for the full tradeoff before assuming
this is a strictly better model. scale_pos_weight is computed from the
actual training-set class balance, exactly as experiment_overfitting_gap.py
does, rather than hard-coded, so it stays correct if the feature panel
changes.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  
from src.transform import feature_engineering  
from src.ai import ai_training, backtesting, model_persistence  
from src.utils import experiment_log  

SPIKE_THRESHOLD = 0.07
SPLIT_DATE = "2023-01-01"
VALIDATION_DATE = "2022-01-01"  # carved out of the training period, before SPLIT_DATE
MAX_POSITION_WEIGHT = 0.2


def main():
    features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD)

    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))
    _, universe_df = ai_training.split_by_universe(features_df, master_ticker_df)

    _, y_train, _, _ = ai_training.split_train_test(universe_df, SPLIT_DATE)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model_config = {
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'scale_pos_weight': scale_pos_weight,
        'max_depth': 3, 'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'n_estimators': 500, 'early_stopping_rounds': 20,
    }

    run_id = experiment_log.new_run_id()
    log_config = {
        **model_config, 'universe': 'all_incl_delisted', 'n_rows': len(universe_df),
        'spike_threshold': SPIKE_THRESHOLD, 'max_position_weight': MAX_POSITION_WEIGHT,
        'label': 'regularised+tree_complexity+early_stopping',
    }

    model, metrics, extras = ai_training.train_xgboost_event_model(
        features_df=universe_df, split_date=SPLIT_DATE,
        validation_date=VALIDATION_DATE, config=model_config)
    X_test, y_test, y_pred, y_pred_proba = extras
    experiment_log.log_training_run(run_id, log_config, metrics)

    predictions_df = universe_df.loc[
        X_test.index, ['Date', 'Symbol', backtesting.DEFAULT_RETURN_COL]].copy()
    predictions_df['y_pred_proba'] = y_pred_proba
    predictions_df = predictions_df.dropna(subset=[backtesting.DEFAULT_RETURN_COL])

    backtest_metrics = backtesting.run_event_driven_backtest(
        predictions_df, run_id, max_position_weight=MAX_POSITION_WEIGHT)

    gap = metrics['train_precision'] - metrics['precision']
    print(f"Run {run_id}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}  Precision: {metrics['precision']:.4f}  "
          f"Train Precision: {metrics['train_precision']:.4f}  Gap: {gap:.4f}")
    print(f"  Total Return: {backtest_metrics['total_return']:.4%}")
    print(f"  Win Rate: {backtest_metrics['win_rate']:.4%}")
    print(f"  Max Drawdown: {backtest_metrics['max_drawdown']:.4%}")
    print(f"  Sharpe: {backtest_metrics['sharpe']:.4f}")

    bundle = model_persistence.build_model_bundle(
        model=model, feature_cols=ai_training.DEFAULT_FEATURES,
        target_col=ai_training.TARGET_COLUMN, config={**log_config, **backtest_metrics},
        run_id=run_id, metrics={**metrics, **backtest_metrics, 'gap': gap})
    model_file = model_persistence.save_model(bundle)
    print(f"Saved production model to {model_file}")


if __name__ == "__main__":
    main()
