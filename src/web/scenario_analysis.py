import numpy as np
import pandas as pd


def _unreported_future_mask(df, as_of):
    """ Pure: rows dated after `as_of` with an EPS Estimate but no Reported EPS yet - i.e. scheduled but not yet announced. """
    dates = pd.to_datetime(df['Date'])
    return (dates > as_of) & df['Reported EPS'].isna() & df['EPS Estimate'].notna()


def find_upcoming_earnings(raw_earnings_df, snapshot_df, as_of=None):
    """
    Pure: upcoming (scheduled, not-yet-reported) earnings events, restricted
    to tickers that also have a price-feature snapshot available (both are
    needed to run a scenario analysis on them - see build_scenario_analysis).
    as_of defaults to now; overridable for tests/reproducibility. Sorted by
    Date ascending (soonest first).
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    mask = _unreported_future_mask(raw_earnings_df, as_of)
    upcoming = raw_earnings_df[mask & raw_earnings_df['Symbol'].isin(snapshot_df['Symbol'])]
    return upcoming.sort_values('Date').reset_index(drop=True)


def find_upcoming_earnings_row(raw_earnings_df, ticker, as_of=None):
    """
    Pure: the single next unreported earnings row for `ticker`, or None if
    it has no scheduled event on file. If somehow more than one future row
    exists for the ticker, the soonest is used.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    ticker = ticker.strip().upper()
    ticker_df = raw_earnings_df[raw_earnings_df['Symbol'] == ticker]
    upcoming = ticker_df[_unreported_future_mask(ticker_df, as_of)]

    if upcoming.empty:
        return None
    return upcoming.sort_values('Date').iloc[0]


def find_snapshot_row(snapshot_df, ticker):
    """ Pure: the latest price-only feature row for `ticker` (Return_Nd/Sector_Rank_Nd, no Surprise(%)/Reported EPS), or None if absent. """
    ticker = ticker.strip().upper()
    rows = snapshot_df[snapshot_df['Symbol'] == ticker]
    return rows.iloc[-1] if not rows.empty else None


def build_surprise_grid(eps_estimate, surprise_range=(-100, 100), step=5):
    """
    Pure: a grid of hypothetical Surprise(%) values and their implied
    Reported EPS (Reported EPS = EPS Estimate * (1 + Surprise%/100), the
    standard convention this project's earnings data already follows - see
    src/ingest/earnings.py). Approximate: yfinance's own Surprise(%) figures
    are computed from a marginally different-precision EPS Estimate than
    the rounded one stored here, so this round-trip isn't exact - acceptable
    for a conditional "what if" sweep, not a claim about a specific number.
    """
    surprises = np.arange(surprise_range[0], surprise_range[1] + step, step)
    reported_eps = eps_estimate * (1 + surprises / 100)
    return pd.DataFrame({'Surprise(%)': surprises, 'Reported EPS': reported_eps})


def build_scenario_analysis(snapshot_row, eps_estimate, model, feature_cols,
                             spike_threshold=None, surprise_range=(-100, 100), step=5,
                             class_labels=('No Spike', 'Spike')):
    """
    Pure: for each hypothetical Surprise(%) in the grid, the model's
    predicted class and Spike-class probability, holding every other
    feature (Return_Nd, Sector_Rank_Nd, ...) at its real, already-known
    current value from snapshot_row. This is a conditional "if the surprise
    turns out to be X%, what would the model say" curve - not a forecast of
    what the surprise will actually be, which is unknown until the event is
    reported (see recommendation.build_recommendation's historical_outcome
    docstring for why this project treats that distinction as important).
    """
    grid = build_surprise_grid(eps_estimate, surprise_range, step)
    n = len(grid)

    base_values = {}
    for col in feature_cols:
        if col == 'EPS Estimate':
            base_values[col] = eps_estimate
        elif col in ('Reported EPS', 'Surprise(%)'):
            continue
        else:
            base_values[col] = snapshot_row.get(col)

    rows = pd.DataFrame([base_values] * n)
    rows['Surprise(%)'] = grid['Surprise(%)'].values
    rows['Reported EPS'] = grid['Reported EPS'].values
    X = rows[feature_cols].astype(float)

    pred_classes = model.predict(X)
    spike_proba = model.predict_proba(X)[:, 1]

    return {
        'eps_estimate': float(eps_estimate),
        'spike_threshold': spike_threshold,
        'context_features': {
            col: _safe_float(base_values.get(col)) for col in feature_cols
            if col not in ('Reported EPS', 'Surprise(%)')
        },
        'scenarios': [
            {
                'surprise_pct': float(s),
                'reported_eps': float(r),
                'predicted_label': class_labels[int(p)],
                'spike_probability': float(pr),
            }
            for s, r, p, pr in zip(
                grid['Surprise(%)'], grid['Reported EPS'], pred_classes, spike_proba)
        ],
    }


def _safe_float(value):
    """ Pure: None-safe float conversion for NaN/missing values """
    if value is None or pd.isna(value):
        return None
    return float(value)
