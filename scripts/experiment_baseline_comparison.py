"""
Baseline comparison: does the production XGBoost model actually add value
over (a) a much simpler model with the same features, and (b) simply being
invested on the same days/at the same rate, regardless of which stocks are
picked?

    ./venv/bin/python scripts/experiment_baseline_comparison.py

Four configs, same 7% threshold, same universe (all incl. delisted), same
train/test split as the production model:

  A. XGBoost (current production)
     The existing model - the one being benchmarked against.

  B. Logistic regression
     Same 5 features, same split, but a deliberately low-capacity linear
     model (~6 free parameters vs XGBoost's thousands of tree splits).
     If it performs similarly to A, XGBoost's extra capacity isn't
     earning its keep and the ceiling is set by feature/signal quality,
     not model capacity - the same diagnosis the overfitting-gap sweep
     pointed at, from a different angle.

  C. Simple PEAD rule (no ML at all)
     Predicts "Spike" whenever Surprise(%) > 0, "No Spike" otherwise.
     Tests whether the literature's core theoretical claim (PEAD:
     positive surprises drift up) has any raw signal before any model
     is layered on top of it.

  D. Exposure-matched random benchmark
     Keeps XGBoost's label untouched, but replaces WHICH events are
     traded each day with a random draw of the same COUNT of events
     from that day's full candidate pool. Isolates whether A's
     financial performance comes from picking the right stocks, or
     merely from being invested on the same days at the same rate.
     Tier 1 (precision) is not meaningful for this config - it doesn't
     make a "prediction" in the same sense, so only Tier 2 is reported.
     Averaged over 20 random seeds to avoid one lucky/unlucky draw.

Each is logged to data/4_experiments/experiment_log.csv
(experiment='baseline_comparison') so results stay traceable. Does not
touch the persisted production model.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.transform import feature_engineering  # noqa: E402
from src.ai import ai_training, backtesting  # noqa: E402
from src.utils import experiment_log  # noqa: E402

SPIKE_THRESHOLD = 0.07
SPLIT_DATE = "2023-01-01"
MAX_POSITION_WEIGHT = 0.2
N_RANDOM_SEEDS = 20
OUTLIER_CHECK_NS = (5, 10)


def compute_robustness_check(predictions_df, return_col='Target_T1_Close_Ret'):
    """
    Payoff stats (avg_win/avg_loss/payoff_ratio) plus total_return after
    excluding the top-5 and top-10 highest-return trades - tests whether
    a config's headline return depends on a handful of extreme winners
    rather than a broad, repeatable edge (the same check §5.4 applied to
    the triple-barrier result, generalised to every config here).
    """
    trades_df = backtesting.filter_trade_signals(predictions_df)
    payoff = backtesting.compute_payoff_stats(trades_df, return_col=return_col)

    robustness = {}
    for n in OUTLIER_CHECK_NS:
        trimmed_df = backtesting.exclude_top_n_trades(trades_df, n, return_col=return_col)
        if trimmed_df.empty:
            robustness[f'return_excl_top{n}'] = float('nan')
            continue
        daily_df = backtesting.compute_daily_portfolio_returns(
            trimmed_df, return_col=return_col, max_position_weight=MAX_POSITION_WEIGHT)
        equity_df = backtesting.build_equity_curve(daily_df)
        kpis = backtesting.compute_backtest_kpis(equity_df, trimmed_df, return_col=return_col)
        robustness[f'return_excl_top{n}'] = kpis['total_return']

    return {**payoff, **robustness}


def run_model_config(label, build_fn, config, universe_df):
    """ Trains build_fn(config), backtests it, logs it, returns a result dict. """
    run_id = experiment_log.new_run_id()

    X_train, y_train, X_test, y_test = ai_training.split_train_test(
        universe_df, SPLIT_DATE)
    model = build_fn(config)
    model, y_pred, y_pred_proba = ai_training.train_model(
        model, X_train, y_train, X_test, y_test)
    metrics = ai_training.evaluate_predictions(y_test, y_pred, y_pred_proba)

    train_pred = model.predict(X_train)
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    train_metrics = ai_training.evaluate_predictions(y_train, train_pred, train_pred_proba)
    metrics['train_accuracy'] = train_metrics['accuracy']
    metrics['train_precision'] = train_metrics['precision']

    predictions_df = universe_df.loc[
        X_test.index, ['Date', 'Symbol', 'Target_T1_Close_Ret']].copy()
    predictions_df['y_pred_proba'] = y_pred_proba
    predictions_df = predictions_df.dropna(subset=['Target_T1_Close_Ret'])

    log_config = {'universe': 'all_incl_delisted', 'n_rows': len(universe_df),
                  'spike_threshold': SPIKE_THRESHOLD, 'max_position_weight': MAX_POSITION_WEIGHT,
                  'experiment': 'baseline_comparison', 'label': label}
    experiment_log.log_training_run(run_id, log_config, metrics)

    backtest_metrics = backtesting.run_event_driven_backtest(
        predictions_df, run_id, max_position_weight=MAX_POSITION_WEIGHT,
        log_file=experiment_log.DEFAULT_LOG_FILE)
    robustness = compute_robustness_check(predictions_df)

    gap = metrics['train_precision'] - metrics['precision']
    return {'config': label, 'test_precision': metrics['precision'],
            'train_precision': metrics['train_precision'], 'gap': gap,
            **backtest_metrics, **robustness,
            'n_trades': (predictions_df['y_pred_proba'] > 0.5).sum()}


def run_simple_pead_rule(universe_df):
    """
    No model at all: predict Spike whenever Surprise(%) > 0. y_pred_proba
    is set to 1.0/0.0 (not a real probability) purely so the existing
    filter_trade_signals(threshold=0.5) machinery selects exactly the
    rows the rule flags as Spike - reusing the same backtest pipeline as
    every other config for a fair, apples-to-apples comparison.
    """
    run_id = experiment_log.new_run_id()

    _, _, X_test, y_test = ai_training.split_train_test(universe_df, SPLIT_DATE)
    test_df = universe_df.loc[X_test.index]

    y_pred = (test_df['Surprise(%)'] > 0).astype(int)
    y_pred_proba = y_pred.astype(float)
    metrics = ai_training.evaluate_predictions(y_test, y_pred, y_pred_proba)
    metrics['train_accuracy'] = float('nan')
    metrics['train_precision'] = float('nan')

    predictions_df = universe_df.loc[
        X_test.index, ['Date', 'Symbol', 'Target_T1_Close_Ret']].copy()
    predictions_df['y_pred_proba'] = y_pred_proba.values
    predictions_df = predictions_df.dropna(subset=['Target_T1_Close_Ret'])

    log_config = {'universe': 'all_incl_delisted', 'n_rows': len(universe_df),
                  'spike_threshold': SPIKE_THRESHOLD, 'max_position_weight': MAX_POSITION_WEIGHT,
                  'experiment': 'baseline_comparison', 'label': 'simple_pead_rule'}
    experiment_log.log_training_run(run_id, log_config, metrics)

    backtest_metrics = backtesting.run_event_driven_backtest(
        predictions_df, run_id, max_position_weight=MAX_POSITION_WEIGHT,
        log_file=experiment_log.DEFAULT_LOG_FILE)
    robustness = compute_robustness_check(predictions_df)

    return {'config': 'C. Simple PEAD rule (Surprise%>0)', 'test_precision': metrics['precision'],
            'train_precision': float('nan'), 'gap': float('nan'),
            **backtest_metrics, **robustness, 'n_trades': int(y_pred.sum())}


def run_exposure_matched_random(universe_df, real_predictions_df, n_seeds=N_RANDOM_SEEDS):
    """
    For each Date in the test period, draws the same COUNT of trades that
    the real model actually took that day, but chosen at random from that
    day's full candidate pool, instead of the model's actual picks.
    Averaged over n_seeds independent draws.
    """
    _, _, X_test, _ = ai_training.split_train_test(universe_df, SPLIT_DATE)
    candidates_df = universe_df.loc[X_test.index, ['Date', 'Symbol', 'Target_T1_Close_Ret']].dropna(
        subset=['Target_T1_Close_Ret'])

    real_trades_df = backtesting.filter_trade_signals(real_predictions_df)
    trades_per_day = real_trades_df.groupby('Date').size()

    seed_results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        picked_rows = []
        for date, day_candidates in candidates_df.groupby('Date'):
            n = int(trades_per_day.get(date, 0))
            if n == 0:
                continue
            n = min(n, len(day_candidates))
            picked_rows.append(day_candidates.sample(n=n, random_state=int(rng.integers(1e9))))
        if not picked_rows:
            continue
        random_trades_df = pd.concat(picked_rows, ignore_index=True)
        random_trades_df['y_pred_proba'] = 1.0

        daily_df = backtesting.compute_daily_portfolio_returns(
            random_trades_df, return_col='Target_T1_Close_Ret',
            max_position_weight=MAX_POSITION_WEIGHT)
        equity_df = backtesting.build_equity_curve(daily_df)
        metrics = backtesting.compute_backtest_kpis(
            equity_df, random_trades_df, return_col='Target_T1_Close_Ret')
        seed_results.append(metrics)

    avg_metrics = {k: float(np.mean([r[k] for r in seed_results])) for k in
                   ['total_return', 'win_rate', 'max_drawdown', 'sharpe']}

    run_id = experiment_log.new_run_id()
    log_config = {'universe': 'all_incl_delisted', 'n_rows': len(universe_df),
                  'spike_threshold': SPIKE_THRESHOLD, 'max_position_weight': MAX_POSITION_WEIGHT,
                  'experiment': 'baseline_comparison', 'label': 'exposure_matched_random',
                  'n_seeds': n_seeds}
    experiment_log.log_training_run(run_id, log_config, {'accuracy': float('nan'), 'precision': float('nan')})
    experiment_log.log_backtest_result(run_id, avg_metrics)

    nan_robustness = {'avg_win': float('nan'), 'avg_loss': float('nan'), 'payoff_ratio': float('nan'),
                       **{f'return_excl_top{n}': float('nan') for n in OUTLIER_CHECK_NS}}
    return {'config': f'D. Exposure-matched random (avg of {n_seeds} seeds)',
            'test_precision': float('nan'), 'train_precision': float('nan'), 'gap': float('nan'),
            **avg_metrics, **nan_robustness, 'n_trades': len(real_trades_df)}


def main():
    features_df = feature_engineering.generate_earnings_driven_features(
        spike_threshold=SPIKE_THRESHOLD)
    master_ticker_df = pd.read_csv(os.path.join(
        "data", "2_processed", "master_ticker_list.csv"))
    _, universe_df = ai_training.split_by_universe(features_df, master_ticker_df)

    results = []

    # A. XGBoost (production)
    xgb_result = run_model_config(
        'A. XGBoost (current production)', ai_training.build_model, {}, universe_df)
    results.append(xgb_result)

    # Rebuild the real model's actual test-period predictions, for the
    # exposure-matched benchmark to measure "how many trades did it take".
    X_train, y_train, X_test, y_test = ai_training.split_train_test(universe_df, SPLIT_DATE)
    xgb_model = ai_training.build_model({})
    xgb_model, _, xgb_proba = ai_training.train_model(xgb_model, X_train, y_train, X_test, y_test)
    real_predictions_df = universe_df.loc[
        X_test.index, ['Date', 'Symbol', 'Target_T1_Close_Ret']].copy()
    real_predictions_df['y_pred_proba'] = xgb_proba
    real_predictions_df = real_predictions_df.dropna(subset=['Target_T1_Close_Ret'])

    # B. Logistic regression
    results.append(run_model_config(
        'B. Logistic regression', ai_training.build_logistic_model, {}, universe_df))

    # C. Simple PEAD rule
    results.append(run_simple_pead_rule(universe_df))

    # D. Exposure-matched random benchmark
    results.append(run_exposure_matched_random(universe_df, real_predictions_df))

    def fmt(x, pct=False):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return f"{'n/a':>9s}"
        return f"{x:9.4%}" if pct else f"{x:9.4f}"

    print("\n--- Summary: precision / return / risk ---")
    header = (f"{'config':45s}  {'precision':>9s}  {'gap':>7s}  {'n_trades':>8s}  "
              f"{'return':>9s}  {'win_rate':>9s}  {'max_dd':>9s}  {'sharpe':>7s}")
    print(header)
    for r in results:
        print(f"{r['config']:45s}  {fmt(r['test_precision'])}  {fmt(r['gap'])[:7]:>7s}  "
              f"{r['n_trades']:8d}  {r['total_return']:9.4%}  {r['win_rate']:9.4%}  "
              f"{r['max_drawdown']:9.4%}  {r['sharpe']:7.4f}")

    print("\n--- Summary: payoff structure + outlier robustness ---")
    header2 = (f"{'config':45s}  {'avg_win':>9s}  {'avg_loss':>9s}  {'payoff':>7s}  "
               f"{'ret_excl5':>10s}  {'ret_excl10':>10s}")
    print(header2)
    for r in results:
        payoff_str = fmt(r['payoff_ratio'])
        print(f"{r['config']:45s}  {fmt(r['avg_win'], pct=True)}  {fmt(r['avg_loss'], pct=True)}  "
              f"{payoff_str[:7]:>7s}  {fmt(r['return_excl_top5'], pct=True):>10s}  "
              f"{fmt(r['return_excl_top10'], pct=True):>10s}")


if __name__ == "__main__":
    main()
