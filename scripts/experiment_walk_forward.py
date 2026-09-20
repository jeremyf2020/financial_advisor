"""
Full rolling (expanding-window) walk-forward analysis: does the production
model's performance hold up when retrained and re-tested across multiple
sequential out-of-sample periods, instead of the single static
2023-01-01 train/test split used 

    ./venv/bin/python scripts/experiment_walk_forward.py

Each fold trains on every event strictly before that fold's test window
(an expanding window - the same information a periodically-retrained
production model would actually have at that point in time) and tests on
the following FOLD_MONTHS-month window, then rolls forward. This is the
walk-forward validation the Evaluation chapter's marker feedback asked
for as a follow-up to the single chronological split already used
(López de Prado, 2018): a single split can't show whether performance is
stable across different market periods, or just an artefact of exactly
where that one split happened to land.

Each fold is logged to data/4_experiments/experiment_log.csv
(experiment='walk_forward') individually, plus a final aggregate summary
(mean/std/min/max of test precision, gap, total_return, win_rate,
max_drawdown, sharpe across folds) - the spread across folds is the whole
point: a model whose Sharpe swings wildly fold to fold is not
"validated", even if its single-split number looked good.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
from src.transform import feature_engineering  # noqa: E402
from src.ai import ai_training, backtesting  # noqa: E402
from src.utils import experiment_log  # noqa: E402

SPIKE_THRESHOLD = 0.07
FIRST_TEST_START = "2023-01-01"  # matches the production model's single split, as fold 1's start
FOLD_MONTHS = 6
MAX_POSITION_WEIGHT = 0.2


def run_fold(fold_idx, test_start, test_end, universe_df):
    """
    Trains a fresh model on every event strictly before test_start,
    evaluates it on [test_start, test_end), logs both the training run
    and (if any trades fire) the backtest, and returns a result dict.
    Returns None if the fold has no usable rows on either side of the
    split after dropping rows with missing features/target.
    """
    clean_df = universe_df.dropna(
        subset=ai_training.DEFAULT_FEATURES + [ai_training.TARGET_COLUMN])
    dates = pd.to_datetime(clean_df['Date'])
    train_df = clean_df[dates < test_start]
    test_df = clean_df[(dates >= test_start) & (dates < test_end)]

    if train_df.empty or test_df.empty:
        return None

    X_train, y_train = train_df[ai_training.DEFAULT_FEATURES], train_df[ai_training.TARGET_COLUMN]
    X_test, y_test = test_df[ai_training.DEFAULT_FEATURES], test_df[ai_training.TARGET_COLUMN]

    run_id = experiment_log.new_run_id()
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
                  'spike_threshold': SPIKE_THRESHOLD, 'experiment': 'walk_forward',
                  'fold': fold_idx, 'test_start': str(test_start.date()),
                  'test_end': str(test_end.date()), 'n_train': len(X_train),
                  'n_test': len(X_test)}
    experiment_log.log_training_run(run_id, log_config, metrics)

    predictions_df = test_df[['Date', 'Symbol', 'Target_T1_Close_Ret']].copy()
    predictions_df['y_pred_proba'] = y_pred_proba
    predictions_df = predictions_df.dropna(subset=['Target_T1_Close_Ret'])

    if predictions_df.empty or (predictions_df['y_pred_proba'] > 0.5).sum() == 0:
        backtest_metrics = {'total_return': 0.0, 'win_rate': 0.0,
                             'max_drawdown': 0.0, 'sharpe': 0.0}
        experiment_log.log_backtest_result(run_id, backtest_metrics)
    else:
        backtest_metrics = backtesting.run_event_driven_backtest(
            predictions_df, run_id, max_position_weight=MAX_POSITION_WEIGHT,
            log_file=experiment_log.DEFAULT_LOG_FILE)

    gap = metrics['train_precision'] - metrics['precision']
    return {'fold': fold_idx, 'test_start': test_start.date(), 'test_end': test_end.date(),
            'n_train': len(X_train), 'n_test': len(X_test),
            'test_precision': metrics['precision'], 'train_precision': metrics['train_precision'],
            'gap': gap, **backtest_metrics}


def main():
    features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD)
    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))
    _, universe_df = ai_training.split_by_universe(features_df, master_ticker_df)

    dates = pd.to_datetime(universe_df['Date'])
    folds = ai_training.generate_walk_forward_folds(
        dates, FIRST_TEST_START, fold_months=FOLD_MONTHS)

    print(f"{len(folds)} walk-forward folds, {FOLD_MONTHS} months each, "
          f"expanding training window, starting {FIRST_TEST_START}\n")

    results = []
    for i, (test_start, test_end) in enumerate(folds, start=1):
        r = run_fold(i, test_start, test_end, universe_df)
        if r is None:
            print(f"Fold {i}: {test_start.date()} to {test_end.date()} - skipped (no usable rows)")
            continue

        results.append(r)
        print(f"Fold {i}: {r['test_start']} to {r['test_end']}  "
              f"(train={r['n_train']}, test={r['n_test']})")
        print(f"  test_precision={r['test_precision']:.4f}  "
              f"train_precision={r['train_precision']:.4f}  gap={r['gap']:.4f}")
        print(f"  total_return={r['total_return']:.4%}  win_rate={r['win_rate']:.4%}  "
              f"max_drawdown={r['max_drawdown']:.4%}  sharpe={r['sharpe']:.4f}")

    results_df = pd.DataFrame(results)
    print("\n--- Aggregate across folds (mean / std / min / max) ---")
    for col in ['test_precision', 'gap', 'total_return', 'win_rate', 'max_drawdown', 'sharpe']:
        print(f"{col:16s}  mean={results_df[col].mean():9.4f}  std={results_df[col].std():9.4f}  "
              f"min={results_df[col].min():9.4f}  max={results_df[col].max():9.4f}")


if __name__ == "__main__":
    main()
