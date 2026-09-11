"""Every setting in one place.

Both scripts read from here, so there is one answer to any question about how
the model is configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
PREDICTION_TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_INPUT_PATH = DATA_DIR / "december_chart_inputs.csv"

# score.py wants the submission under exactly this name, at the repo root.
SUBMISSION_PATH = PROJECT_ROOT / "validation_predictions.csv"
# The December answers go into a copy so data/ is never written to.
DECEMBER_OUTPUT_PATH = OUTPUT_DIR / "december_predictions.csv"
MODEL_PATH = OUTPUT_DIR / "model.joblib"
METRICS_PATH = OUTPUT_DIR / "cv_metrics.json"
DIAGNOSTIC_PATH = OUTPUT_DIR / "seasonal_extrapolation.png"

TARGET = "posted_rate"
DATE = "date"

#: The only columns december_chart_inputs.csv provides.
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]

#: Coordinates are a fixed property of each city, so the December rows can have
#: theirs filled in by name. See data.city_coordinates.
COORDINATE_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


@dataclass(frozen=True)
class FeatureConfig:
    """Which groups of features to build.

    ``use_macro`` is off because those two columns make the model worse, by
    about 16% on mean absolute error. ``use_geo`` is on because coordinates make
    it better, by about 2% overall and 7% on rows whose city never appears in
    training. Both figures come from ``scripts/train.py --compare``, which
    rebuilds them on demand.
    """

    use_macro: bool = False
    use_geo: bool = True
    #: How many sine waves describe the yearly cycle. Waves repeat, so they keep
    #: working in November and December. A plain day counter would freeze at its
    #: last training value.
    annual_harmonics: int = 3
    weekly_harmonics: int = 2
    #: How hard to pull a thinly seen city back towards the overall average.
    #: Higher means more pull.
    lane_smoothing: float = 50.0


@dataclass(frozen=True)
class ModelConfig:
    """Settings for the gradient boosting model.

    This is scikit-learn's HistGradientBoostingRegressor rather than LightGBM or
    XGBoost. It needs no extra install, it treats a blank value as its own case,
    and it splits on equipment as a real category.
    """

    learning_rate: float = 0.05
    max_iter: int = 600
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 40
    random_state: int = 42
    #: How far a price has to sit from normal before it counts as corrupted and
    #: gets dropped. Applies to training only, never to scoring.
    outlier_mad_threshold: float = 5.0
    #: Duan's smearing correction. Undoing a log gives the middle value rather
    #: than the average, and this nudges it back towards the average. It helps
    #: RMSE and costs a little on MAE. Spotter has not said which one they
    #: grade on, so it stays a switch instead of a decision.
    apply_smearing: bool = True


@dataclass(frozen=True)
class SplitConfig:
    """Shape of the backtest.

    validation.csv begins the day after train_test.csv ends and runs 61 days,
    from 2025-11-01 to 2025-12-31. Every fold copies that shape. Train up to a
    date, then score the next ``horizon_days``.
    """

    horizon_days: int = 61
    n_folds: int = 4
    #: Gap between folds. Defaults to the horizon, so the scored windows do not
    #: overlap.
    step_days: int | None = None


@dataclass(frozen=True)
class PipelineConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
