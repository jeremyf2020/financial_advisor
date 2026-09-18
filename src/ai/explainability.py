import shap


def compute_shap_values(model, X):
    """
    Pure-ish: SHAP TreeExplainer feature contributions for X's row - how
    much each feature pushed this specific prediction's log-odds toward
    the positive ("Spike") class. Positive = pushed toward Spike, negative
    = pushed toward No Spike. No I/O beyond the explainer's internal use
    of the already-fitted model; caller supplies model + an already-
    selected single-row feature DataFrame (same feature_cols/order used
    to train it).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return dict(zip(X.columns, shap_values[0].tolist()))
