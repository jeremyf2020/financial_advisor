import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score

DEFAULT_FEATURES = ['EPS Estimate', 'Reported EPS', 'Surprise(%)',
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
    is measured against. Regularisation (reg_alpha, reg_lambda,
    scale_pos_weight) and tree-complexity controls (max_depth,
    min_child_weight, subsample, colsample_bytree) all pass straight
    through here with no extra wiring needed - only early stopping
    (early_stopping_rounds) needs eval_set support in train_model below.
    """
    config = config or {}
    return XGBClassifier(random_state=42, n_jobs=-1, **config)


def build_logistic_model(config=None):
    """
    Pure: config dict -> a StandardScaler + LogisticRegression pipeline -
    a deliberately low-capacity baseline to compare against build_model's
    XGBoost (see report §5.6/Conclusion: "a simpler baseline model...
    logistic regression"). Features are scaled first because logistic
    regression's coefficients and convergence are sensitive to differing
    feature scales (e.g. Surprise(%) spans tens of points, Return_60d
    spans a fraction of 1) - unlike XGBoost's tree splits, which are
    scale-invariant, so build_model has no equivalent step. Exposes the
    same fit/predict/predict_proba interface, so it's a drop-in
    replacement for build_model wherever train_model is called.
    """
    config = config or {}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, random_state=42, **config)),
    ])


def build_calibrated_model(config=None):
    """
    Pure: config dict -> a CalibratedClassifierCV wrapping build_model's
    XGBClassifier. XGBoost's raw predict_proba is optimised for the
    training objective (log-loss on the label), not for calibration -
    there's no guarantee a predicted probability of 0.7 actually means
    "70% of these fire" empirically, and filter_trade_signals() fires a
    trade on nothing more than predict_proba > 0.5, so a systematically
    over- or under-confident raw score silently selects the wrong set of
    trades. CalibratedClassifierCV fits a secondary mapping from raw
    scores to calibrated probabilities using internal cross-validation on
    the training set, so the mapping isn't fit on the same data it's
    evaluated on. config supports 'method' ('sigmoid' for Platt scaling,
    the default, or 'isotonic' for a non-parametric monotonic fit), 'cv'
    (number of internal folds, default 3), and 'base_config' (passed
    straight through to build_model for the underlying XGBClassifier's
    hyperparameters, default {}). Exposes the same fit/predict/
    predict_proba interface as build_model, so it's a drop-in replacement
    wherever train_model is called.
    """
    config = config or {}
    base_model = build_model(config.get('base_config', {}))
    return CalibratedClassifierCV(
        estimator=base_model,
        method=config.get('method', 'sigmoid'),
        cv=config.get('cv', 3),
    )


def carve_validation_slice(X_train, y_train, dates, validation_date):
    """
    Pure: split a chronologically-ordered training set into (fit set,
    validation set) using validation_date as the boundary - rows before
    it are used to fit, rows from validation_date onward form the
    held-out validation slice used for early stopping. `dates` must be a
    Series of datetimes aligned to X_train/y_train's index (e.g. the
    source DataFrame's 'Date' column). Carving this slice out of the
    training period, chronologically prior to the test period, is the
    early-stopping intervention scoped in the Design chapter's
    overfitting-reduction plan.
    """
    validation_date = pd.to_datetime(validation_date)
    fit_mask = dates.loc[X_train.index] < validation_date
    return (X_train[fit_mask], y_train[fit_mask],
            X_train[~fit_mask], y_train[~fit_mask])


def generate_walk_forward_folds(dates, first_test_start, fold_months=6):
    """
    Pure: split a chronologically-ordered date range into a sequence of
    expanding-window walk-forward folds. Fold i trains on every date
    strictly before that fold's test_start (all history that would
    actually be available to a periodically-retrained production model at
    that point in time) and tests on the following fold_months-month
    window, then rolls forward. A single chronological split (as used
    everywhere else in this project) can only show performance at one
    point in time; walking forward through several folds instead shows
    whether that performance is stable across different market periods,
    or an artefact of exactly where that one split happened to land.
    Returns a list of (test_start, test_end) Timestamp pairs, stopping
    once test_start passes the last available date - a trailing fold
    shorter than fold_months is included as long as it still covers at
    least one date, so the tail of the data isn't silently dropped.
    """
    dates = pd.to_datetime(dates)
    max_date = dates.max()
    test_start = pd.to_datetime(first_test_start)

    folds = []
    while test_start <= max_date:
        test_end = test_start + pd.DateOffset(months=fold_months)
        if ((dates >= test_start) & (dates < test_end)).any():
            folds.append((test_start, test_end))
        test_start = test_end

    return folds


def train_model(model, X_train, y_train, X_test, y_test, eval_set=None):
    """
    I/O (training): fit the model, return it plus predictions on the
    held-out test set. If eval_set is given (a validation slice, see
    carve_validation_slice) and the model was built with
    early_stopping_rounds in its config, training stops once validation
    performance plateaus instead of always running the full
    n_estimators rounds - the direct defence against fitting
    training-set noise that the Design chapter scopes.
    """
    if eval_set is not None:
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    else:
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


def compute_brier_score(y_true, y_pred_proba):
    """
    Pure: mean squared error between predicted probabilities and actual
    binary outcomes - the standard scalar summary of calibration quality
    (0.0 = perfect, 0.25 = what a constant 0.5 prediction scores on a
    balanced set). Complements precision/accuracy, which only look at the
    thresholded 0/1 decision and say nothing about whether the underlying
    confidence score itself is trustworthy.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    return float(np.mean((y_pred_proba - y_true) ** 2))


def compute_calibration_curve(y_true, y_pred_proba, n_bins=10):
    """
    Pure: buckets predictions into n_bins equal-width probability bins
    and computes, per bin, the mean predicted probability vs. the actual
    observed fraction of positives - the reliability-diagram data used to
    judge calibration. A perfectly calibrated model has mean_predicted ==
    fraction_positive in every bin. Bins with zero predictions in them
    are dropped rather than returned as NaN rows, since there's nothing
    to plot or compare for an empty bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_pred_proba, bin_edges[1:-1])

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        rows.append({
            'bin_low': bin_edges[b], 'bin_high': bin_edges[b + 1],
            'mean_predicted': float(y_pred_proba[mask].mean()),
            'fraction_positive': float(y_true[mask].mean()),
            'count': int(mask.sum()),
        })

    return pd.DataFrame(rows)


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
    validation_date=None,
    config=None,
    feature_cols=DEFAULT_FEATURES,
    target_col=TARGET_COLUMN,
):
    """
    load features -> split_train_test -> build_model ->
    train_model -> evaluate_predictions. Returns (model, metrics, extras)
    - extras carries the test set + predictions, and metrics/config are
    what the caller passes to experiment_log.log_training_run(). metrics
    also carries train_accuracy/train_precision (same model, scored on
    X_train) alongside the test-set accuracy/precision - a large train-vs-
    test gap is a classic overfitting signal.
    Pass a pre-loaded features_df (e.g. from split_by_universe()) to skip
    reading features_file from disk. Pass validation_date to carve an
    early-stopping validation slice out of the training period (requires
    config to include early_stopping_rounds to have any effect) -
    train_accuracy/train_precision are then scored on the narrowed
    training set actually fitted, not the full pre-split_date period.
    """
    df = features_df if features_df is not None else pd.read_csv(
        features_file)

    X_train, y_train, X_test, y_test = split_train_test(
        df, split_date, feature_cols, target_col)

    eval_set = None
    if validation_date is not None:
        dates = pd.to_datetime(df['Date'])
        X_train, y_train, X_val, y_val = carve_validation_slice(
            X_train, y_train, dates, validation_date)
        eval_set = [(X_val, y_val)]

    model = build_model(config)
    model, y_pred, y_pred_proba = train_model(
        model, X_train, y_train, X_test, y_test, eval_set=eval_set)

    metrics = evaluate_predictions(y_test, y_pred, y_pred_proba)

    train_pred = model.predict(X_train)
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    train_metrics = evaluate_predictions(y_train, train_pred, train_pred_proba)
    metrics['train_accuracy'] = train_metrics['accuracy']
    metrics['train_precision'] = train_metrics['precision']

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
