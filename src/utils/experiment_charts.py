import json
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import experiment_log


def label_runs_sequentially(log_df):
    """
    Pure: chronological "Run 1", "Run 2", ... labels. run_id embeds the
    real timestamp it was created at (see experiment_log.new_run_id), which
    is convenient internally but not something a report should expose -
    it'd reveal exactly which calendar days the work was done on. Every
    report-facing table/chart uses Run_Label instead of run_id.
    """
    result = log_df.sort_values('timestamp').reset_index(drop=True).copy()
    result['Run_Label'] = [f'Run {i + 1}' for i in range(len(result))]

    return result


def expand_config(log_df):
    """
    Pure: parse each row's config_json into its own columns, prefixed
    `config_` - lets the notebook table show exactly what changed between
    runs without hand-parsing JSON.
    """
    config_rows = [json.loads(c) if c else {} for c in log_df['config_json']]
    config_df = pd.json_normalize(config_rows).add_prefix('config_')

    return pd.concat([log_df.reset_index(drop=True), config_df], axis=1)


def report_table(log_df):
    """
    Pure: expand_config, labeled Run 1/2/3/..., with run_id and timestamp
    dropped - the report-safe view of the log that doesn't reveal real
    run dates.
    """
    labeled = label_runs_sequentially(log_df)
    expanded = expand_config(labeled)
    expanded = expanded.drop(columns=['run_id', 'timestamp'])

    return expanded[['Run_Label'] + [c for c in expanded.columns if c != 'Run_Label']]


def diff_config_between_runs(log_df):
    """
    Pure: for each run (after the first), which config keys changed vs. the
    immediately preceding run, plus how much each metric moved - the
    "what changed, what happened" record the §4 sweep needs for the report.
    """
    expanded = expand_config(label_runs_sequentially(log_df))
    config_cols = [c for c in expanded.columns
                   if c.startswith('config_') and c != 'config_json']
    metric_cols = ['accuracy', 'precision', 'total_return',
                   'win_rate', 'max_drawdown', 'sharpe']

    rows = []
    for i in range(1, len(expanded)):
        prev, curr = expanded.iloc[i - 1], expanded.iloc[i]
        changed = {
            col.replace('config_', ''): (prev[col], curr[col])
            for col in config_cols if prev[col] != curr[col]
        }
        deltas = {
            f'{m}_delta': curr[m] - prev[m]
            for m in metric_cols
            if pd.notna(curr[m]) and pd.notna(prev[m])
        }
        rows.append({
            'run_label': curr['Run_Label'],
            'prev_run_label': prev['Run_Label'],
            'config_changed': changed,
            **deltas,
        })

    return pd.DataFrame(rows)


def compute_train_test_gap(log_df, metric='precision'):
    """
    Pure: train-minus-test gap for `metric`, per run - the same fitted
    model scored on its own training data vs. held-out test data. A large
    positive gap (much better on data it was trained on) is a classic
    overfitting signal. Runs logged before train_accuracy/train_precision
    existed show NaN here, not zero - there's no way to compute their gap
    retroactively.
    """
    labeled = label_runs_sequentially(log_df)
    train_col = f'train_{metric}'

    gap_df = labeled[['Run_Label', train_col, metric]].copy()
    gap_df['gap'] = gap_df[train_col] - gap_df[metric]

    return gap_df


def compute_gap_vs_backtest(log_df, metric='precision'):
    """
    Pure: joins each run's train/test gap (see compute_train_test_gap) with
    its Tier 2 backtest results into one table, so a single query can check
    whether any run across the whole logged history has achieved both a low
    gap and strong financial performance at once - or whether closing the
    gap and improving Tier 2 have been in tension across every
    configuration actually tried so far. Runs missing either side (no
    train_precision recorded, or Tier 2 never computed for that run) are
    dropped, since there's nothing to compare for them.
    """
    labeled = label_runs_sequentially(log_df)
    train_col = f'train_{metric}'

    result = labeled[['Run_Label', train_col, metric, 'total_return',
                       'win_rate', 'max_drawdown', 'sharpe']].copy()
    result['gap'] = result[train_col] - result[metric]

    return result.dropna(subset=['gap', 'total_return']).reset_index(drop=True)


def plot_scatter(x_metric, y_metric, df, label_col='Run_Label'):
    """
    Scatter of x_metric vs y_metric from an already-prepared DataFrame
    (e.g. compute_gap_vs_backtest's output), each point labeled from
    label_col. Unlike plot_tradeoff, this does not re-derive Run_Label
    from a raw experiment log - the input here is already a derived/joined
    table, not the log itself, so it has no timestamp column to sort by.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x_metric], df[y_metric])
    for _, row in df.iterrows():
        ax.annotate(row[label_col], (row[x_metric], row[y_metric]), fontsize=8)
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    fig.tight_layout()

    return fig


def summarize_run_counts(log_df):
    """
    Pure: how many runs are logged in total, and how many of those have
    an actual backtest (Tier 2) result. The more times Tier 2 gets
    consulted while iterating toward a "winning" config, the less that
    win can be trusted - this is the raw count behind that caveat, not a
    judgment call on whether it's been consulted too many times.
    """
    total_runs = len(log_df)
    backtested_runs = int(log_df['total_return'].notna().sum())

    return {'total_runs': total_runs, 'backtested_runs': backtested_runs}


def plot_metric_trend(metric, log_df=None, log_file=experiment_log.DEFAULT_LOG_FILE):
    """
    Line chart of `metric` across every logged run, in chronological order -
    shows whether each successive change in the §4 sweep made the metric
    better or worse over time.
    """
    if log_df is None:
        log_df = experiment_log.load_experiment_log(log_file)
    log_df = label_runs_sequentially(log_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log_df['Run_Label'], log_df[metric], marker='o')
    ax.set_xlabel('Run')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} across runs')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    fig.tight_layout()

    return fig


def plot_metric_comparison(metrics, log_df=None, top_n=None, log_file=experiment_log.DEFAULT_LOG_FILE):
    """
    Grouped bar chart comparing each metric in `metrics` across runs -
    optionally only the top_n runs, ranked by the first metric in the list.
    """
    if log_df is None:
        log_df = experiment_log.load_experiment_log(log_file)
    df = label_runs_sequentially(log_df)

    if top_n is not None:
        df = df.sort_values(metrics[0], ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(metrics)
    x = range(len(df))
    for i, metric in enumerate(metrics):
        ax.bar([xi + i * width for xi in x], df[metric], width=width, label=metric)

    ax.set_xticks([xi + width * (len(metrics) - 1) / 2 for xi in x])
    ax.set_xticklabels(df['Run_Label'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_tradeoff(x_metric, y_metric, log_df=None, log_file=experiment_log.DEFAULT_LOG_FILE):
    """
    Scatter of x_metric vs y_metric, one point per run, labeled Run 1/2/3/... -
    e.g. precision vs total_return, to check whether a Tier 1 improvement
    actually paid off at Tier 2 (§6.4).
    """
    if log_df is None:
        log_df = experiment_log.load_experiment_log(log_file)
    log_df = label_runs_sequentially(log_df)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(log_df[x_metric], log_df[y_metric])
    for _, row in log_df.iterrows():
        ax.annotate(row['Run_Label'], (row[x_metric], row[y_metric]), fontsize=8)
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    fig.tight_layout()

    return fig
