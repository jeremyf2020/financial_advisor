import matplotlib
matplotlib.use('Agg')

import pytest
import pandas as pd
from src.utils import experiment_charts as ec


def make_log_df():
    """ Small hand-built experiment log, mirroring experiment_log.LOG_COLUMNS """
    return pd.DataFrame({
        'run_id': ['run_1', 'run_2', 'run_3'],
        'timestamp': ['2026-01-01T00:00:00', '2026-01-02T00:00:00', '2026-01-03T00:00:00'],
        'config_json': ['{}', '{"scale_pos_weight": 5}', '{"scale_pos_weight": 10}'],
        'accuracy': [0.56, 0.58, 0.55],
        'precision': [0.50, 0.55, 0.60],
        'total_return': [-0.10, 0.05, 0.20],
        'win_rate': [0.50, 0.51, 0.53],
        'max_drawdown': [-0.70, -0.50, -0.40],
        'sharpe': [0.3, 0.8, 1.2],
    })


def test_label_runs_sequentially_assigns_chronological_labels():
    """ Run_Label should be Run 1/2/3... in timestamp order, not run_id order """
    # Arrange: rows deliberately out of timestamp order
    log_df = make_log_df().iloc[[2, 0, 1]].reset_index(drop=True)

    # Act
    labeled = ec.label_runs_sequentially(log_df)

    # Assert
    assert labeled[labeled['run_id'] == 'run_1']['Run_Label'].iloc[0] == 'Run 1'
    assert labeled[labeled['run_id'] == 'run_3']['Run_Label'].iloc[0] == 'Run 3'


def test_report_table_hides_run_id_and_timestamp():
    """ The report-facing table must not leak run_id (which embeds a real
    timestamp) or the raw timestamp column """
    # Arrange
    log_df = make_log_df()

    # Act
    table = ec.report_table(log_df)

    # Assert
    assert 'run_id' not in table.columns
    assert 'timestamp' not in table.columns
    assert list(table['Run_Label']) == ['Run 1', 'Run 2', 'Run 3']


def test_expand_config_parses_json_into_columns():
    """ config_json should unpack into config_-prefixed columns per key """
    # Arrange
    log_df = make_log_df()

    # Act
    expanded = ec.expand_config(log_df)

    # Assert
    assert 'config_scale_pos_weight' in expanded.columns
    assert pd.isna(expanded.iloc[0]['config_scale_pos_weight'])
    assert expanded.iloc[1]['config_scale_pos_weight'] == 5


def test_diff_config_between_runs_flags_changed_keys_and_metric_deltas():
    """ Each row after the first should report which config key changed and by how much metrics moved """
    # Arrange
    log_df = make_log_df()

    # Act
    diff_df = ec.diff_config_between_runs(log_df)

    # Assert: one diff row per run after the first
    assert len(diff_df) == 2
    row = diff_df.iloc[0]
    assert row['run_label'] == 'Run 2'
    changed = row['config_changed']
    assert set(changed.keys()) == {'scale_pos_weight'}
    prev_val, curr_val = changed['scale_pos_weight']
    assert pd.isna(prev_val)
    assert curr_val == pytest.approx(5)
    assert row['precision_delta'] == pytest.approx(0.05)
    assert row['sharpe_delta'] == pytest.approx(0.5)


def test_plot_metric_trend_returns_figure_with_data():
    """ Should plot one point per run, in run order """
    # Arrange
    log_df = make_log_df()

    # Act
    fig = ec.plot_metric_trend('sharpe', log_df=log_df)

    # Assert
    line = fig.axes[0].lines[0]
    assert list(line.get_ydata()) == [0.3, 0.8, 1.2]


def test_plot_metric_comparison_returns_figure_with_one_bar_group_per_metric():
    """ Should draw len(metrics) * len(runs) bars total """
    # Arrange
    log_df = make_log_df()

    # Act
    fig = ec.plot_metric_comparison(['precision', 'sharpe'], log_df=log_df)

    # Assert
    assert len(fig.axes[0].patches) == 2 * len(log_df)


def test_plot_metric_comparison_respects_top_n():
    """ top_n should limit to the N best runs by the first metric """
    # Arrange
    log_df = make_log_df()

    # Act
    fig = ec.plot_metric_comparison(['sharpe'], log_df=log_df, top_n=2)

    # Assert: 2 runs * 1 metric = 2 bars
    assert len(fig.axes[0].patches) == 2


def test_plot_tradeoff_returns_figure_with_one_point_per_run():
    """ Should scatter one point per run, x/y matching the requested metrics """
    # Arrange
    log_df = make_log_df()

    # Act
    fig = ec.plot_tradeoff('precision', 'sharpe', log_df=log_df)

    # Assert
    offsets = fig.axes[0].collections[0].get_offsets()
    assert len(offsets) == len(log_df)
