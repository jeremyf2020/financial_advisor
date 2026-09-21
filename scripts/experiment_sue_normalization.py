"""
SUE normalization: does standardizing the earnings surprise by each
company's own historical surprise volatility - the normalisation PEAD's
original literature actually uses (Bernard & Thomas, 1989, 1990) - improve
on the raw Surprise(%) this pipeline otherwise trains on? The same
magnitude of surprise means something very different for a company with a
history of volatile earnings vs. a stable one; raw Surprise(%) can't tell
them apart, SUE can.

    ./venv/bin/python scripts/experiment_sue_normalization.py

Two configs, same 7% threshold, same T+1 horizon, same universe, same
model hyperparameters - only the surprise feature changes:

  A. Raw Surprise(%) (current production) - ['EPS Estimate', 'Reported EPS',
     'Surprise(%)', 'Return_60d', 'Sector_Rank_60d']
  B. SUE-normalized - ['EPS Estimate', 'Reported EPS', 'SUE', 'Return_60d',
     'Sector_Rank_60d']. Rows without enough prior earnings history for
     that Symbol (see feature_engineering.compute_sue) have SUE = NaN and
     are dropped by split_train_test like any other missing feature - B's
     usable sample size is expected to be smaller than A's for this reason
     alone, independent of whether SUE is a better feature.

For each: Tier 1 (precision/gap), Tier 2 (backtest), and the same
payoff/outlier-robustness check used throughout this notebook.

Each run is logged to data/4_experiments/experiment_log.csv
(experiment='sue_normalization').
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402
from src.transform import feature_engineering  # noqa: E402
from src.ai import ai_training, backtesting  # noqa: E402
from src.utils import experiment_log  # noqa: E402

SPIKE_THRESHOLD = 0.07
SPLIT_DATE = "2023-01-01"
MAX_POSITION_WEIGHT = 0.2
OUTLIER_CHECK_NS = (5, 10)

FEATURES_RAW = ['EPS Estimate', 'Reported EPS', 'Surprise(%)', 'Return_60d', 'Sector_Rank_60d']
FEATURES_SUE = ['EPS Estimate', 'Reported EPS', 'SUE', 'Return_60d', 'Sector_Rank_60d']


def run_config(label, feature_cols, universe_df):
    """ Trains the production model config with feature_cols swapped in,
    backtests it, and runs the outlier-robustness check - same methodology
    as every other config in this notebook. """
    run_id = experiment_log.new_run_id()

    X_train, y_train, X_test, y_test = ai_training.split_train_test(
        universe_df, SPLIT_DATE, feature_cols=feature_cols)
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
                  'spike_threshold': SPIKE_THRESHOLD, 'experiment': 'sue_normalization',
                  'label': label, 'features': feature_cols,
                  'n_train': len(X_train), 'n_test': len(X_test)}
    experiment_log.log_training_run(run_id, log_config, metrics)

    predictions_df = universe_df.loc[
        X_test.index, ['Date', 'Symbol', 'Target_T1_Close_Ret']].copy()
    predictions_df['y_pred_proba'] = y_pred_proba
    predictions_df = predictions_df.dropna(subset=['Target_T1_Close_Ret'])

    n_trades = int((predictions_df['y_pred_proba'] > 0.5).sum())
    if n_trades == 0:
        backtest_metrics = {'total_return': 0.0, 'win_rate': 0.0,
                             'max_drawdown': 0.0, 'sharpe': 0.0}
        experiment_log.log_backtest_result(run_id, backtest_metrics)
        payoff = {'avg_win': float('nan'), 'avg_loss': float('nan'), 'payoff_ratio': None}
        robustness = {f'return_excl_top{n}': float('nan') for n in OUTLIER_CHECK_NS}
    else:
        backtest_metrics = backtesting.run_event_driven_backtest(
            predictions_df, run_id, max_position_weight=MAX_POSITION_WEIGHT,
            log_file=experiment_log.DEFAULT_LOG_FILE)
        trades_df = backtesting.filter_trade_signals(predictions_df)
        payoff = backtesting.compute_payoff_stats(trades_df)

        robustness = {}
        for n in OUTLIER_CHECK_NS:
            trimmed_df = backtesting.exclude_top_n_trades(trades_df, n)
            if trimmed_df.empty:
                robustness[f'return_excl_top{n}'] = float('nan')
                continue
            daily_df = backtesting.compute_daily_portfolio_returns(
                trimmed_df, max_position_weight=MAX_POSITION_WEIGHT)
            equity_df = backtesting.build_equity_curve(daily_df)
            kpis = backtesting.compute_backtest_kpis(equity_df, trimmed_df)
            robustness[f'return_excl_top{n}'] = kpis['total_return']

    gap = metrics['train_precision'] - metrics['precision']
    return {'config': label, 'test_precision': metrics['precision'],
            'train_precision': metrics['train_precision'], 'gap': gap,
            'n_train': len(X_train), 'n_test': len(X_test), 'n_trades': n_trades,
            **backtest_metrics, **payoff, **robustness}


def main():
    features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD, use_sue=True)
    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))
    _, universe_df = ai_training.split_by_universe(features_df, master_ticker_df)

    results = [
        run_config('A. Raw Surprise(%) (current production)', FEATURES_RAW, universe_df),
        run_config('B. SUE-normalized', FEATURES_SUE, universe_df),
    ]

    def fmt(x, pct=False):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return f"{'n/a':>9s}"
        return f"{x:9.4%}" if pct else f"{x:9.4f}"

    print("--- Summary: precision / gap / return / risk ---")
    header = (f"{'config':38s}  {'precision':>9s}  {'gap':>7s}  {'n_train':>7s}  {'n_test':>6s}  "
              f"{'n_trades':>8s}  {'return':>9s}  {'win_rate':>9s}  {'max_dd':>9s}  {'sharpe':>7s}")
    print(header)
    for r in results:
        print(f"{r['config']:38s}  {fmt(r['test_precision'])}  {fmt(r['gap'])[:7]:>7s}  "
              f"{r['n_train']:7d}  {r['n_test']:6d}  {r['n_trades']:8d}  {r['total_return']:9.4%}  "
              f"{r['win_rate']:9.4%}  {r['max_drawdown']:9.4%}  {r['sharpe']:7.4f}")

    print("\n--- Summary: payoff structure + outlier robustness ---")
    header2 = (f"{'config':38s}  {'avg_win':>9s}  {'avg_loss':>9s}  {'payoff':>7s}  " +
               "  ".join(f"{f'excl_top{n}':>10s}" for n in OUTLIER_CHECK_NS))
    print(header2)
    for r in results:
        payoff_str = fmt(r['payoff_ratio'])
        excl_str = "  ".join(fmt(r[f'return_excl_top{n}'], pct=True) for n in OUTLIER_CHECK_NS)
        print(f"{r['config']:38s}  {fmt(r['avg_win'], pct=True)}  {fmt(r['avg_loss'], pct=True)}  "
              f"{payoff_str[:7]:>7s}  {excl_str}")


if __name__ == "__main__":
    main()
