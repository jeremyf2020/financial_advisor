import pytest
import pandas as pd
from src.ai import ai_training as ai


def make_fixture_df():
    """ Small hand-built event-driven feature table, both classes present """
    return pd.DataFrame({
        'Date': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01',
                 '2023-01-01', '2023-02-01'],
        'Symbol': ['A', 'B', 'C', 'D', 'E', 'F'],
        'EPS Estimate': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        'Reported EPS': [1.1, 1.0, 1.3, 1.2, 1.5, 1.4],
        'Surprise(%)': [10.0, -9.0, 8.0, -7.0, 6.0, -5.0],
        'Return_60d': [0.05, -0.03, 0.02, -0.01, 0.04, -0.02],
        'Sector_Rank_60d': [0.8, 0.2, 0.6, 0.4, 0.9, 0.1],
        'Target_Spike_Class': [1, 0, 1, 0, 1, 0],
    })


def test_split_train_test():
    """ Rows before split_date go to train, on/after go to test """
    # Arrange
    df = make_fixture_df()

    # Act
    X_train, y_train, X_test, y_test = ai.split_train_test(
        df, split_date="2023-01-01")

    # Assert
    assert len(X_train) == 4
    assert len(X_test) == 2
    assert list(X_train.columns) == ai.DEFAULT_FEATURES


def test_split_train_test_drops_rows_with_missing_features():
    """ A row missing a required feature should be dropped, not passed to XGBoost as NaN """
    # Arrange
    df = make_fixture_df()
    df.loc[0, 'Return_60d'] = None

    # Act
    X_train, y_train, X_test, y_test = ai.split_train_test(
        df, split_date="2023-01-01")

    # Assert: one fewer training row than the full fixture
    assert len(X_train) == 3


def test_build_model_default_config():
    """ An empty/None config should still produce a valid, seeded model """
    # Act
    model = ai.build_model()

    # Assert
    assert model.get_params()['random_state'] == 42


def test_build_model_applies_config_overrides():
    """ Config dict entries should pass straight through to XGBClassifier """
    # Act
    model = ai.build_model({'max_depth': 3, 'n_estimators': 50})

    # Assert
    params = model.get_params()
    assert params['max_depth'] == 3
    assert params['n_estimators'] == 50

def test_build_logistic_model_default_config():
    """ An empty/None config should still produce a valid, seeded pipeline """
    # Act
    model = ai.build_logistic_model()

    # Assert
    assert model.named_steps['logreg'].get_params()['random_state'] == 42


def test_build_logistic_model_applies_config_overrides():
    """ Config dict entries should pass straight through to LogisticRegression """
    # Act
    model = ai.build_logistic_model({'C': 0.5, 'penalty': 'l1', 'solver': 'liblinear'})

    # Assert
    params = model.named_steps['logreg'].get_params()
    assert params['C'] == 0.5
    assert params['penalty'] == 'l1'


def test_build_logistic_model_fits_and_predicts():
    """ The pipeline should fit and produce valid predictions/probabilities,
    exercising the same interface train_model() relies on """
    # Arrange
    df = make_fixture_df()
    X_train, y_train, X_test, y_test = ai.split_train_test(
        df, split_date="2023-01-01")
    model = ai.build_logistic_model()

    # Act
    model, y_pred, y_pred_proba = ai.train_model(
        model, X_train, y_train, X_test, y_test)

    # Assert
    assert len(y_pred) == len(y_test)
    assert all(p in (0, 1) for p in y_pred)
    assert all(0.0 <= p <= 1.0 for p in y_pred_proba)



def test_train_model_and_evaluate_predictions():
    """ train_model should return predictions the same length as the test set,
    and evaluate_predictions should produce valid metric values """
    # Arrange
    df = make_fixture_df()
    X_train, y_train, X_test, y_test = ai.split_train_test(
        df, split_date="2023-01-01")
    model = ai.build_model({'n_estimators': 10, 'max_depth': 2})

    # Act
    model, y_pred, y_pred_proba = ai.train_model(
        model, X_train, y_train, X_test, y_test)
    metrics = ai.evaluate_predictions(y_test, y_pred, y_pred_proba)

    # Assert
    assert len(y_pred) == len(y_test)
    assert len(y_pred_proba) == len(y_test)
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['precision'] <= 1.0


def test_train_xgboost_event_model_smoke(tmp_path):
    """
    Orchestrator smoke test: wires load -> split -> build -> train ->
    evaluate together against a small on-disk fixture
    """
    # Arrange
    df = make_fixture_df()
    features_file = tmp_path / "features.csv"
    df.to_csv(features_file, index=False)

    # Act
    model, metrics, extras = ai.train_xgboost_event_model(
        features_file=str(features_file), split_date="2023-01-01",
        config={'n_estimators': 10, 'max_depth': 2})

    # Assert
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    X_test, y_test, y_pred, y_pred_proba = extras
    assert len(y_test) == 2


def test_train_xgboost_event_model_accepts_preloaded_df():
    """ Passing features_df directly should skip reading from disk entirely """
    # Arrange
    df = make_fixture_df()

    # Act
    model, metrics, extras = ai.train_xgboost_event_model(
        features_df=df, split_date="2023-01-01",
        config={'n_estimators': 10, 'max_depth': 2})

    # Assert
    assert 'accuracy' in metrics

def test_carve_validation_slice():
    """ Rows before validation_date go to the fit set, on/after go to the validation set """
    # Arrange
    df = make_fixture_df()
    X_train, y_train, X_test, y_test = ai.split_train_test(
        df, split_date="2023-01-01")
    dates = pd.to_datetime(df['Date'])

    # Act
    X_fit, y_fit, X_val, y_val = ai.carve_validation_slice(
        X_train, y_train, dates, validation_date="2022-03-01")

    # Assert
    assert len(X_fit) == 2   # 2022-01-01, 2022-02-01
    assert len(X_val) == 2   # 2022-03-01, 2022-04-01
    assert len(X_fit) + len(X_val) == len(X_train)



def test_generate_walk_forward_folds_covers_full_range_without_overlap():
    """ Folds should be contiguous (each fold's end is the next fold's start)
    and stop once the dates run out """
    # Arrange: dates span 2022-06-01 to 2023-10-15
    dates = pd.to_datetime(['2022-06-01', '2022-09-01', '2023-01-15',
                             '2023-06-01', '2023-10-15'])

    # Act
    folds = ai.generate_walk_forward_folds(
        dates, first_test_start="2023-01-01", fold_months=6)

    # Assert: fold 1 is Jan-Jul 2023, fold 2 is Jul 2023-Jan 2024 (covers 10-15)
    assert folds[0] == (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-01"))
    assert folds[1] == (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-01-01"))
    for i in range(len(folds) - 1):
        assert folds[i][1] == folds[i + 1][0]


def test_generate_walk_forward_folds_skips_empty_trailing_fold():
    """ A fold window with no dates in it at all (e.g. past the last
    available date) should not be included """
    # Arrange: last date is well inside the first fold's window
    dates = pd.to_datetime(['2022-06-01', '2023-02-01'])

    # Act
    folds = ai.generate_walk_forward_folds(
        dates, first_test_start="2023-01-01", fold_months=6)

    # Assert: exactly one fold (2023-01-01 to 2023-07-01), no empty tail fold
    assert len(folds) == 1


def test_generate_walk_forward_folds_returns_empty_list_when_no_data_after_start():
    """ If every date is before first_test_start, there is nothing to
    walk forward through """
    # Arrange
    dates = pd.to_datetime(['2020-01-01', '2020-06-01'])

    # Act
    folds = ai.generate_walk_forward_folds(
        dates, first_test_start="2023-01-01", fold_months=6)

    # Assert
    assert folds == []



def test_train_model_with_eval_set_enables_early_stopping():
    """ Passing eval_set with early_stopping_rounds in the config should
    still produce valid, test-set-length predictions """
    # Arrange
    df = make_fixture_df()
    X_train, y_train, X_test, y_test = ai.split_train_test(
        df, split_date="2023-01-01")
    X_fit, y_fit, X_val, y_val = ai.carve_validation_slice(
        X_train, y_train, pd.to_datetime(df['Date']), validation_date="2022-03-01")
    model = ai.build_model(
        {'n_estimators': 50, 'max_depth': 2, 'early_stopping_rounds': 5})

    # Act
    model, y_pred, y_pred_proba = ai.train_model(
        model, X_fit, y_fit, X_test, y_test, eval_set=[(X_val, y_val)])

    # Assert
    assert len(y_pred) == len(y_test)
    assert len(y_pred_proba) == len(y_test)


def test_train_xgboost_event_model_with_validation_date(tmp_path):
    """ Orchestrator smoke test: validation_date should narrow the fitted
    training set and still return a valid metrics dict """
    # Arrange
    df = make_fixture_df()
    features_file = tmp_path / "features.csv"
    df.to_csv(features_file, index=False)

    # Act
    model, metrics, extras = ai.train_xgboost_event_model(
        features_file=str(features_file), split_date="2023-01-01",
        validation_date="2022-03-01",
        config={'n_estimators': 20, 'max_depth': 2, 'early_stopping_rounds': 5})

    # Assert
    assert 'accuracy' in metrics
    assert 0.0 <= metrics['train_precision'] <= 1.0
    


def test_split_by_universe():
    """
    path_a should keep only rows for tickers marked 'present' in
    master_ticker_df; path_b should keep every row (current + delisted)
    """
    # Arrange: 'A' and 'B' are current, 'C' is delisted
    features_df = make_fixture_df()  # Symbols A-F
    master_ticker_df = pd.DataFrame({
        'Symbol': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Date_added': ['2010-01-01'] * 6,
        'Date_removed': ['present', 'present', '2019-01-01',
                          '2019-01-01', 'present', '2019-01-01'],
    })

    # Act
    path_a, path_b = ai.split_by_universe(features_df, master_ticker_df)

    # Assert: path_a only has the 'present' tickers, path_b has everything
    assert sorted(path_a['Symbol'].unique()) == ['A', 'B', 'E']
    assert len(path_b) == len(features_df)


def test_split_by_universe_path_a_is_subset_of_path_b():
    """ Every row in path_a must also appear in path_b - path_a is strictly a subset """
    # Arrange
    features_df = make_fixture_df()
    master_ticker_df = pd.DataFrame({
        'Symbol': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Date_added': ['2010-01-01'] * 6,
        'Date_removed': ['present', '2019-01-01', 'present',
                          '2019-01-01', 'present', '2019-01-01'],
    })

    # Act
    path_a, path_b = ai.split_by_universe(features_df, master_ticker_df)

    # Assert
    assert len(path_a) < len(path_b)
    assert set(path_a['Symbol']).issubset(set(path_b['Symbol']))
