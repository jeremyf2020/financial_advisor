import pytest
import pandas as pd
from xgboost import XGBClassifier
from src.ai import explainability as xai


def make_fitted_model():
    """ A model where feature 'a' is obviously the deciding factor - lets us assert direction, not just shape """
    X = pd.DataFrame({
        'a': [10, 10, -10, -10, 10, -10, 10, -10],
        'b': [1, -1, 1, -1, -1, 1, 1, -1],
    })
    y = [1, 1, 0, 0, 1, 0, 1, 0]
    model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)
    return model


def test_compute_shap_values_returns_one_value_per_feature():
    model = make_fitted_model()
    X = pd.DataFrame({'a': [10], 'b': [1]})

    result = xai.compute_shap_values(model, X)

    assert set(result.keys()) == {'a', 'b'}
    assert isinstance(result['a'], float)


def test_compute_shap_values_direction_matches_the_dominant_feature():
    """ A row with a strongly positive 'a' (the class-1-favoring value) should get a positive shap value for 'a' """
    model = make_fitted_model()
    X_positive = pd.DataFrame({'a': [10], 'b': [1]})
    X_negative = pd.DataFrame({'a': [-10], 'b': [1]})

    result_positive = xai.compute_shap_values(model, X_positive)
    result_negative = xai.compute_shap_values(model, X_negative)

    assert result_positive['a'] > 0
    assert result_negative['a'] < 0
