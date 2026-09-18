import pytest
import pandas as pd
from xgboost import XGBClassifier
from src.ai import model_persistence as mp


def make_fitted_model():
    """ A tiny real fitted XGBClassifier - persistence must round-trip a real model object """
    X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [4, 3, 2, 1]})
    y = [0, 1, 0, 1]
    model = XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    return model


def test_build_model_bundle():
    """ Bundle should carry the model plus everything needed to call and trace it """
    model = make_fitted_model()

    bundle = mp.build_model_bundle(
        model=model, feature_cols=['a', 'b'], target_col='y',
        config={'spike_threshold': 0.07}, run_id='run_1',
        metrics={'accuracy': 0.9})

    assert bundle['model'] is model
    assert bundle['feature_cols'] == ['a', 'b']
    assert bundle['run_id'] == 'run_1'
    assert bundle['config'] == {'spike_threshold': 0.07}
    assert bundle['metrics'] == {'accuracy': 0.9}
    assert 'saved_at' in bundle


def test_save_and_load_model_round_trip(tmp_path):
    """ A saved bundle should load back with an equivalent, usable model """
    model = make_fitted_model()
    bundle = mp.build_model_bundle(
        model=model, feature_cols=['a', 'b'], target_col='y',
        config={}, run_id='run_1')
    model_file = tmp_path / "model.joblib"
    metadata_file = tmp_path / "metadata.json"

    mp.save_model(bundle, model_file=str(model_file), metadata_file=str(metadata_file))
    loaded = mp.load_model(model_file=str(model_file))

    assert loaded['run_id'] == 'run_1'
    assert loaded['feature_cols'] == ['a', 'b']
    X = pd.DataFrame({'a': [1], 'b': [4]})
    assert loaded['model'].predict(X)[0] == model.predict(X)[0]


def test_save_model_writes_readable_metadata_sidecar(tmp_path):
    """ The JSON sidecar should be readable without joblib/xgboost, and exclude the model object """
    import json

    model = make_fitted_model()
    bundle = mp.build_model_bundle(
        model=model, feature_cols=['a', 'b'], target_col='y',
        config={'spike_threshold': 0.07}, run_id='run_1', metrics={'accuracy': 0.9})
    model_file = tmp_path / "model.joblib"
    metadata_file = tmp_path / "metadata.json"

    mp.save_model(bundle, model_file=str(model_file), metadata_file=str(metadata_file))

    with open(metadata_file) as f:
        metadata = json.load(f)
    assert 'model' not in metadata
    assert metadata['run_id'] == 'run_1'
    assert metadata['config'] == {'spike_threshold': 0.07}


def test_load_model_missing_file_raises_actionable_error(tmp_path):
    """ A missing model file should point the caller at the training script, not just fail silently """
    missing_file = tmp_path / "does_not_exist.joblib"

    with pytest.raises(FileNotFoundError, match="train_and_save_model.py"):
        mp.load_model(model_file=str(missing_file))
