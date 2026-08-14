import pytest
import pandas as pd
import json
from src.utils import experiment_log as log


def test_new_run_id_is_unique():
    """ Two run ids generated back-to-back should never collide """
    # Act
    id1 = log.new_run_id()
    id2 = log.new_run_id()

    # Assert
    assert id1 != id2
    assert isinstance(id1, str)


def test_log_training_run_creates_file(tmp_path):
    """ Logging the first run should create the CSV with that run's config + metrics """
    # Arrange
    log_file = tmp_path / "experiment_log.csv"
    config = {'n_estimators': 100, 'max_depth': 4}
    metrics = {'accuracy': 0.6, 'precision': 0.55}

    # Act
    log.log_training_run("test_run_1", config, metrics, log_file=str(log_file))

    # Assert: config round-trips through JSON, metrics stored correctly
    assert log_file.exists()
    df = pd.read_csv(log_file)
    assert df.iloc[0]['run_id'] == "test_run_1"
    assert json.loads(df.iloc[0]['config_json']) == config
    assert df.iloc[0]['accuracy'] == 0.6
    assert df.iloc[0]['precision'] == 0.55


def test_log_training_run_appends_to_existing_log(tmp_path):
    """ A second run should be appended as a new row, not overwrite the first """
    # Arrange
    log_file = tmp_path / "experiment_log.csv"

    # Act
    log.log_training_run(
        "run_1", {}, {'accuracy': 0.5, 'precision': 0.5}, log_file=str(log_file))
    log.log_training_run(
        "run_2", {}, {'accuracy': 0.6, 'precision': 0.6}, log_file=str(log_file))

    # Assert
    df = pd.read_csv(log_file)
    assert len(df) == 2
    assert df['run_id'].tolist() == ["run_1", "run_2"]


def test_log_backtest_result_upserts_onto_existing_run(tmp_path):
    """ Backtest metrics should land on the same row as the training run, not a new one """
    # Arrange
    log_file = tmp_path / "experiment_log.csv"
    log.log_training_run(
        "run_1", {}, {'accuracy': 0.5, 'precision': 0.5}, log_file=str(log_file))

    # Act
    log.log_backtest_result("run_1", {
        'total_return': -0.10, 'win_rate': 0.45, 'max_drawdown': -0.30, 'sharpe': -0.1,
    }, log_file=str(log_file))

    # Assert: still one row, training metrics untouched, backtest metrics added
    df = pd.read_csv(log_file)
    assert len(df) == 1
    assert df.iloc[0]['total_return'] == -0.10
    assert df.iloc[0]['win_rate'] == 0.45
    assert df.iloc[0]['accuracy'] == 0.5


def test_log_backtest_result_missing_run_raises(tmp_path):
    """ Logging a backtest result for a run_id that was never trained should raise """
    # Arrange
    log_file = tmp_path / "experiment_log.csv"
    log.log_training_run(
        "run_1", {}, {'accuracy': 0.5, 'precision': 0.5}, log_file=str(log_file))

    # Act / Assert
    with pytest.raises(ValueError, match="run_2"):
        log.log_backtest_result(
            "run_2", {'total_return': 0.1}, log_file=str(log_file))


def test_log_backtest_result_missing_file_raises(tmp_path):
    """ Logging a backtest result before any training run has ever been logged should raise """
    # Arrange
    log_file = tmp_path / "does_not_exist.csv"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        log.log_backtest_result(
            "run_1", {'total_return': 0.1}, log_file=str(log_file))


def test_load_experiment_log_empty(tmp_path):
    """ Reading a log that doesn't exist yet should return an empty, correctly-shaped DataFrame """
    # Arrange
    log_file = tmp_path / "does_not_exist.csv"

    # Act
    df = log.load_experiment_log(log_file=str(log_file))

    # Assert
    assert df.empty
    assert list(df.columns) == log.LOG_COLUMNS


def test_load_experiment_log_sorted_chronologically(tmp_path):
    """ Runs should come back sorted by timestamp, regardless of write order """
    # Arrange: write two rows out of chronological order
    log_file = tmp_path / "experiment_log.csv"
    pd.DataFrame([
        {'run_id': 'run_2', 'timestamp': '2024-02-01T00:00:00', 'config_json': '{}',
         'accuracy': 0.6, 'precision': 0.6, 'total_return': None, 'win_rate': None,
         'max_drawdown': None, 'sharpe': None},
        {'run_id': 'run_1', 'timestamp': '2024-01-01T00:00:00', 'config_json': '{}',
         'accuracy': 0.5, 'precision': 0.5, 'total_return': None, 'win_rate': None,
         'max_drawdown': None, 'sharpe': None},
    ]).to_csv(log_file, index=False)

    # Act
    df = log.load_experiment_log(log_file=str(log_file))

    # Assert
    assert df['run_id'].tolist() == ['run_1', 'run_2']
