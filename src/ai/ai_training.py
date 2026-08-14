import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score

# Surprise(%) is deliberately excluded: this feature set is used to test
# whether survivorship bias affects the price/momentum signal itself,
# without that comparison being confounded by earnings-data coverage
# differences between current and delisted tickers. EPS Estimate/Reported
# EPS are kept - only Surprise(%) is dropped.
DEFAULT_FEATURES = ['EPS Estimate', 'Reported EPS',
                     'Return_60d', 'Sector_Rank_60d']
TARGET_COLUMN = 'Target_Spike_Class'


def split_train_test(df, split_date, feature_cols=DEFAULT_FEATURES, target_col=TARGET_COLUMN):
    """
    Pure: chronological train/test split - all rows before split_date are
    train, everything from split_date onward is held-out test. Avoids the
    future-data leakage a random shuffle would introduce (López de Prado,
    2018 - already the anchor citation for this project's validation
    approach).
    """
    clean_df = df.dropna(subset=feature_cols + [target_col]).copy()
    clean_df['Date'] = pd.to_datetime(clean_df['Date'])
    split_date = pd.to_datetime(split_date)

    train_df = clean_df[clean_df['Date'] < split_date]
    test_df = clean_df[clean_df['Date'] >= split_date]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    return X_train, y_train, X_test, y_test


def build_model(config=None):
    """
    Pure: config dict -> XGBClassifier. An empty/None config is the
    baseline - plain XGBoost defaults, no scale_pos_weight, no tuning -
    the reference point every later sweep step (TDD_REWRITE_PLAN.md §4)
    is measured against.
    """
    config = config or {}
    return XGBClassifier(random_state=42, n_jobs=-1, **config)


def train_model(model, X_train, y_train, X_test, y_test):
    """
    I/O (training): fit the model, return it plus predictions on the
    held-out test set.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return model, y_pred, y_pred_proba


def evaluate_predictions(y_test, y_pred, y_pred_proba=None):
    """
    Pure: accuracy + precision on the held-out set. Precision on the
    "Buy" class is the primary metric (TDD_REWRITE_PLAN.md §6.2) - under
    this dataset's class imbalance, accuracy alone is misleading.
    """
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
    }


def split_by_universe(features_df, master_ticker_df):
    """
    Pure: split the event-driven feature panel into (path_a, path_b) by
    ticker universe - path_a keeps only rows for tickers that are current
    S&P 500 constituents (the survivorship-biased universe a yfinance-only
    pipeline could ever see), path_b keeps every row (current + delisted).
    Comparing results trained on each is a direct empirical test of
    whether survivorship bias actually matters for this model.
    """
    current_symbols = set(
        master_ticker_df[master_ticker_df['Date_removed'] == 'present']['Symbol'])

    path_a = features_df[features_df['Symbol'].isin(
        current_symbols)].reset_index(drop=True)
    path_b = features_df.reset_index(drop=True)

    return path_a, path_b


def train_xgboost_event_model(
    features_file=os.path.join(
        "data", "3_features", "event_driven_features.csv"),
    features_df=None,
    split_date="2023-01-01",
    config=None,
    feature_cols=DEFAULT_FEATURES,
    target_col=TARGET_COLUMN,
):
    """
    Orchestrator: load features -> split_train_test -> build_model ->
    train_model -> evaluate_predictions. Returns (model, metrics, extras)
    - extras carries the test set + predictions, and metrics/config are
    what the caller passes to experiment_log.log_training_run().
    Pass a pre-loaded features_df (e.g. from split_by_universe()) to skip
    reading features_file from disk.
    """
    df = features_df if features_df is not None else pd.read_csv(
        features_file)

    X_train, y_train, X_test, y_test = split_train_test(
        df, split_date, feature_cols, target_col)

    model = build_model(config)
    model, y_pred, y_pred_proba = train_model(
        model, X_train, y_train, X_test, y_test)

    metrics = evaluate_predictions(y_test, y_pred, y_pred_proba)

    return model, metrics, (X_test, y_test, y_pred, y_pred_proba)


if __name__ == "__main__":
    from src.utils import experiment_log

    features_df = pd.read_csv(os.path.join(
        "data", "3_features", "event_driven_features.csv"))
    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))

    path_a_df, path_b_df = split_by_universe(features_df, master_ticker_df)

    # Survivorship-bias comparison: same model config, same features, only
    # the ticker universe differs between the two runs.
    for label, universe_df in [("current_only", path_a_df), ("all_incl_delisted", path_b_df)]:
        run_id = experiment_log.new_run_id()
        model_config = {}  # baseline: plain XGBoost defaults
        log_config = {**model_config, 'universe': label,
                      'n_rows': len(universe_df)}

        model, metrics, _ = train_xgboost_event_model(
            features_df=universe_df, config=model_config)

        print(f"Run {run_id} [{label}] (rows={len(universe_df)})")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")

        experiment_log.log_training_run(run_id, log_config, metrics)

    print(f"Logged to data/4_experiments/experiment_log.csv")
