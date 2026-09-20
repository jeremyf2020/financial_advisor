"""
Ad-hoc experiment: does regularisation, tree-complexity control, or early
stopping narrow the train/test precision gap diagnosed in the Evaluation
chapter (§5.5)? [to check]

    ./venv/bin/python scripts/experiment_overfitting_gap.py

Regenerates data/3_features/event_driven_features.csv at the production
model's SPIKE_THRESHOLD (0.07) so the "baseline" row here is directly
comparable to the report's own production-model row (train precision
0.946, test precision 0.268, gap 0.678). Trains five configurations on
the all-including-delisted universe and prints train/test precision and
the gap for each, side by side. Each run is also logged to
data/4_experiments/experiment_log.csv via the normal experiment_log
machinery, so it stays part of the traceable experiment history - no
run here overwrites the persisted production model in data/5_models/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
from src.transform import feature_engineering  # noqa: E402
from src.ai import ai_training  # noqa: E402
from src.utils import experiment_log  # noqa: E402

SPIKE_THRESHOLD = 0.07
SPLIT_DATE = "2023-01-01"
VALIDATION_DATE = "2022-01-01"  # carved out of the training period, before SPLIT_DATE


def main():
    features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD)

    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))
    _, universe_df = ai_training.split_by_universe(features_df, master_ticker_df)

    # scale_pos_weight is computed from the actual training-set class
    # balance, once, and reused across every config below so the
    # comparison isolates the effect of each intervention.
    _, y_train, _, _ = ai_training.split_train_test(universe_df, SPLIT_DATE)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Training set class balance: {pos} positive / {neg} negative "
          f"(scale_pos_weight={scale_pos_weight:.3f})\n")

    regularisation = {'reg_alpha': 0.1, 'reg_lambda': 1.0,
                       'scale_pos_weight': scale_pos_weight}
    tree_complexity = {'max_depth': 3, 'min_child_weight': 5,
                        'subsample': 0.7, 'colsample_bytree': 0.7}

    configs = {
        'baseline': {},
        'regularised': regularisation,
        'tree_complexity': tree_complexity,
        'regularised+tree_complexity': {**regularisation, **tree_complexity},
        'regularised+tree_complexity+early_stopping': {
            **regularisation, **tree_complexity,
            'n_estimators': 500, 'early_stopping_rounds': 20,
        },
    }

    results = []
    for label, model_config in configs.items():
        run_id = experiment_log.new_run_id()
        validation_date = VALIDATION_DATE if 'early_stopping_rounds' in model_config else None

        model, metrics, _ = ai_training.train_xgboost_event_model(
            features_df=universe_df, split_date=SPLIT_DATE,
            validation_date=validation_date, config=model_config)

        gap = metrics['train_precision'] - metrics['precision']
        results.append({
            'config': label, 'test_precision': metrics['precision'],
            'train_precision': metrics['train_precision'], 'gap': gap,
        })

        log_config = {**model_config, 'universe': 'all_incl_delisted',
                      'n_rows': len(universe_df), 'spike_threshold': SPIKE_THRESHOLD,
                      'experiment': 'overfitting_gap_sweep', 'label': label}
        experiment_log.log_training_run(run_id, log_config, metrics)

        print(f"{label:45s}  test_precision={metrics['precision']:.4f}  "
              f"train_precision={metrics['train_precision']:.4f}  gap={gap:.4f}")

    baseline_gap = results[0]['gap']
    print("\n--- Summary (gap vs baseline) ---")
    for r in results:
        delta = r['gap'] - baseline_gap
        print(f"{r['config']:45s}  gap={r['gap']:.4f}  "
              f"({'+' if delta >= 0 else ''}{delta:.4f} vs baseline)")


if __name__ == "__main__":
    main()
