import pandas as pd


def find_latest_event_row(features_df, ticker):
    """
    Pure: most recent (max Date) row for `ticker` in the event-driven
    features table. Returns None if the ticker has no row at all - an
    unknown ticker and a known ticker with no surviving earnings-event row
    look the same from here, which is fine: both mean "nothing to
    recommend."
    """
    ticker = ticker.strip().upper()
    ticker_df = features_df[features_df['Symbol'] == ticker]

    if ticker_df.empty:
        return None

    ticker_df = ticker_df.assign(_sort_date=pd.to_datetime(ticker_df['Date']))
    return ticker_df.sort_values('_sort_date').iloc[-1]


def feature_row_to_frame(row, feature_cols):
    """
    Pure: a looked-up Series -> the single-row DataFrame shape XGBoost
    (and SHAP's TreeExplainer, which needs the same input) expects -
    feature_cols selected in training order, cast to float.
    """
    return row[feature_cols].to_frame().T.astype(float)


def build_recommendation(row, model, feature_cols, spike_threshold=None,
                          class_labels=('No Spike', 'Spike')):
    """
    Pure: a looked-up feature row + a fitted model -> a structured
    recommendation dict. No disk/network I/O - caller supplies an
    already-loaded row and model.

    The model only predicts a binary spike/no-spike class and its
    probability - it does NOT predict a return magnitude. The row's
    Target_T1_*_Ret columns are the ACTUAL, ALREADY-REALIZED outcome of
    that specific past earnings event (historical fact, not a live
    forecast), so they're returned separately as `historical_outcome` -
    never merged with the model's own prediction - so callers (the LLM
    prompt, the UI) don't present a known historical fact as if it were a
    live forward-looking number.
    """
    X = feature_row_to_frame(row, feature_cols)
    pred_class = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    return {
        'ticker': row['Symbol'],
        'as_of_date': str(pd.Timestamp(row['Date']).date()),
        'sector': row.get('Sector'),
        'predicted_class': pred_class,
        'predicted_label': class_labels[pred_class],
        'confidence': float(proba[pred_class]),
        'spike_threshold': spike_threshold,
        'feature_values': {
            c: _safe_float(row.get(c)) for c in feature_cols
        },
        'historical_outcome': {
            'target_t1_close_ret': _safe_float(row.get('Target_T1_Close_Ret')),
            'target_t1_high_ret': _safe_float(row.get('Target_T1_High_Ret')),
        },
    }


def _safe_float(value):
    """ Pure: None-safe float conversion for NaN/missing values """
    if value is None or pd.isna(value):
        return None
    return float(value)
