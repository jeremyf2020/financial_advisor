import os
import json
import uuid
from datetime import datetime, timezone
import pandas as pd

LOG_COLUMNS = ['run_id', 'timestamp', 'config_json', 'accuracy', 'precision',
               'total_return', 'win_rate', 'max_drawdown', 'sharpe']

DEFAULT_LOG_FILE = os.path.join(
    "data", "4_experiments", "experiment_log.csv")


def new_run_id():
    """
    A chronologically sortable, unique run id - timestamp plus a short
    random suffix to avoid collisions if two runs start the same second.
    """
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    suffix = uuid.uuid4().hex[:6]
    return f"{timestamp}_{suffix}"


def log_training_run(run_id, config, metrics, log_file=DEFAULT_LOG_FILE):
    """
    I/O: append one row for this run - the config that produced it (as
    JSON, so every run is fully reconstructable) plus its training
    metrics. Backtest metrics start empty; log_backtest_result() fills
    them in later once a backtest has actually been run for this run_id.
    """
    row = {
        'run_id': run_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config_json': json.dumps(config),
        'accuracy': metrics.get('accuracy'),
        'precision': metrics.get('precision'),
        'total_return': None,
        'win_rate': None,
        'max_drawdown': None,
        'sharpe': None,
    }

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([row], columns=LOG_COLUMNS)

    log_df.to_csv(log_file, index=False)
    return row


def log_backtest_result(run_id, metrics, log_file=DEFAULT_LOG_FILE):
    """
    I/O: upsert backtest metrics (total_return/win_rate/max_drawdown/
    sharpe) onto the existing row for run_id. Requires log_training_run()
    to have been called for this run_id first - a backtest result with no
    matching training run is a bug, not a new experiment.
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(
            f"{log_file} not found - call log_training_run() for run_id "
            f"'{run_id}' first")

    log_df = pd.read_csv(log_file)

    if run_id not in log_df['run_id'].values:
        raise ValueError(
            f"run_id '{run_id}' not found in {log_file} - call "
            f"log_training_run() for it first")

    for key in ['total_return', 'win_rate', 'max_drawdown', 'sharpe']:
        if key in metrics:
            log_df.loc[log_df['run_id'] == run_id, key] = metrics[key]

    log_df.to_csv(log_file, index=False)
    return log_df[log_df['run_id'] == run_id].iloc[0].to_dict()


def load_experiment_log(log_file=DEFAULT_LOG_FILE):
    """
    I/O: read the experiment log back as a DataFrame, chronologically
    sorted. Returns an empty (but correctly-shaped) DataFrame if no runs
    have been logged yet.
    """
    if not os.path.exists(log_file):
        return pd.DataFrame(columns=LOG_COLUMNS)

    log_df = pd.read_csv(log_file)
    return log_df.sort_values('timestamp').reset_index(drop=True)
