import os
import json
from datetime import datetime, timezone
import joblib

DEFAULT_MODEL_DIR = os.path.join("data", "5_models")
DEFAULT_MODEL_FILE = os.path.join(DEFAULT_MODEL_DIR, "production_model.joblib")
DEFAULT_METADATA_FILE = os.path.join(DEFAULT_MODEL_DIR, "production_model_metadata.json")


def build_model_bundle(model, feature_cols, target_col, config, run_id, metrics=None):
    """
    Pure: assemble the dict save_model() persists - the fitted model plus
    everything needed to (a) call it correctly (feature_cols, in order)
    and (b) trace it back to its row in data/4_experiments/experiment_log.csv
    via run_id.
    """
    return {
        'model': model,
        'feature_cols': list(feature_cols),
        'target_col': target_col,
        'config': config,
        'run_id': run_id,
        'metrics': metrics or {},
        'saved_at': datetime.now(timezone.utc).isoformat(),
    }


def save_model(bundle, model_file=DEFAULT_MODEL_FILE, metadata_file=DEFAULT_METADATA_FILE):
    """
    I/O: persist the bundle via joblib (the model object needs pickle-based
    serialization). Also writes a JSON sidecar with everything except the
    model object itself, so the config/metrics/run_id are readable without
    importing joblib/xgboost.
    """
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    joblib.dump(bundle, model_file)

    metadata = {k: v for k, v in bundle.items() if k != 'model'}
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    return model_file


def load_model(model_file=DEFAULT_MODEL_FILE):
    """
    I/O: load a bundle saved by save_model(). Raises FileNotFoundError with
    an actionable message if missing - data/ is gitignored project-wide, so
    the artifact never arrives via git clone and must be generated locally.
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"No saved model found at {model_file}. Run "
            f"'./venv/bin/python scripts/train_and_save_model.py' first.")

    return joblib.load(model_file)
