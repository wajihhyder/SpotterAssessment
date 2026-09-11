"""Freight rate prediction, for the Spotter assessment.

``scripts/train.py`` and ``scripts/predict.py`` drive this package:

* :mod:`freight_rate.config`   every setting, in one place
* :mod:`freight_rate.data`     reading the files and fixing what is broken
* :mod:`freight_rate.features` turning raw columns into model input
* :mod:`freight_rate.splits`   deciding what trains and what tests
* :mod:`freight_rate.model`    the model itself
* :mod:`freight_rate.evaluate` scoring, and the backtest loop
"""

from .config import FeatureConfig, ModelConfig, PipelineConfig, SplitConfig
from .model import RateModel

__all__ = [
    "FeatureConfig",
    "ModelConfig",
    "PipelineConfig",
    "SplitConfig",
    "RateModel",
]
