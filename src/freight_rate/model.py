"""The model.

It predicts the price per mile, then multiplies the distance back in.

Two reasons for that rather than predicting the price directly. Distance already
explains most of what the price does, at 0.91, so a model aimed at the total
spends its effort rediscovering that long trips cost more. Dividing it out first
leaves the model free to work on equipment, route and season, which is where the
accuracy actually comes from.

The second reason is that freight error is proportional. Missing by $100 on a
$700 run is a different failure from missing by $100 on a $7,000 run, and
working in logs makes the model treat them that way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import TARGET, FeatureConfig, ModelConfig
from .data import attach_coordinates, city_coordinates, outlier_mask, rate_per_mile
from .features import (
    CATEGORICAL_FEATURES,
    LaneEncoder,
    align_categories,
    build_features,
    fit_encoder,
)


@dataclass
class FitReport:
    rows_available: int
    rows_used: int
    outliers_dropped: int
    smearing_factor: float

    def summary(self) -> str:
        pct = 100.0 * self.outliers_dropped / max(self.rows_available, 1)
        return (
            f"trained on {self.rows_used:,} of {self.rows_available:,} rows "
            f"({self.outliers_dropped:,} corrupted prices dropped, {pct:.2f}%); "
            f"smearing factor {self.smearing_factor:.4f}"
        )


class RateModel:
    """Holds the model, the lane encoder and the city coordinate lookup.

    Keeping all three together is what makes the backtest honest. Every fold
    builds its own RateModel, so the lane scores and the coordinate table are
    rebuilt on that fold's training rows and never see anything later.
    """

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.feature_config = feature_config or FeatureConfig()
        self.model_config = model_config or ModelConfig()
        self.encoder_: LaneEncoder | None = None
        self.coordinates_: pd.DataFrame | None = None
        self.regressor_: HistGradientBoostingRegressor | None = None
        self.columns_: pd.DataFrame | None = None
        self.smearing_factor_: float = 1.0
        self.report_: FitReport | None = None

    def _build_regressor(self) -> HistGradientBoostingRegressor:
        config = self.model_config
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=config.learning_rate,
            max_iter=config.max_iter,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_leaf=config.min_samples_leaf,
            l2_regularization=config.l2_regularization,
            early_stopping=config.early_stopping,
            validation_fraction=config.validation_fraction,
            n_iter_no_change=config.n_iter_no_change,
            random_state=config.random_state,
            categorical_features=CATEGORICAL_FEATURES,
        )

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.feature_config.use_geo and self.coordinates_ is not None:
            return attach_coordinates(frame, self.coordinates_)
        return frame

    def fit(self, frame: pd.DataFrame) -> "RateModel":
        if TARGET not in frame.columns:
            raise ValueError("fit requires the labelled target column")

        # Corrupted prices come out of training only. They stay in any scoring
        # set, because the graded data probably holds the same junk, and cutting
        # the hard rows out of the measurement would only flatter the result.
        outliers = outlier_mask(frame, self.model_config.outlier_mad_threshold)
        clean_frame = frame[~outliers]

        self.coordinates_ = city_coordinates(frame)
        self.encoder_ = fit_encoder(clean_frame, self.feature_config)
        features = build_features(clean_frame, self.encoder_, self.feature_config)
        self.columns_ = features.iloc[:0].copy()

        target = np.log(rate_per_mile(clean_frame))

        self.regressor_ = self._build_regressor()
        self.regressor_.fit(features, target)

        # Duan's smearing estimator. Undoing a log gives the middle price rather
        # than the average one, and multiplying by the average of the undone
        # residuals corrects for that.
        residuals = target.to_numpy() - self.regressor_.predict(features)
        self.smearing_factor_ = float(np.mean(np.exp(residuals)))

        self.report_ = FitReport(
            rows_available=len(frame),
            rows_used=len(clean_frame),
            outliers_dropped=int(outliers.sum()),
            smearing_factor=self.smearing_factor_,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return predicted prices in dollars."""
        if self.regressor_ is None or self.encoder_ is None or self.columns_ is None:
            raise RuntimeError("predict called before fit")

        prepared = self._prepare(frame)
        features = build_features(prepared, self.encoder_, self.feature_config)
        features = align_categories(self.columns_, features)

        log_rpm = self.regressor_.predict(features)
        rpm = np.exp(log_rpm)
        if self.model_config.apply_smearing:
            rpm = rpm * self.smearing_factor_

        rates = rpm * frame["distance"].to_numpy(dtype=float)
        # score.py rejects anything at or below zero. Working in logs makes that
        # impossible, and the floor keeps the guarantee visible.
        return np.maximum(rates, 1.0)

    def feature_importance(self, frame: pd.DataFrame, n_repeats: int = 5) -> pd.Series:
        """Rank features by how much the error grows when each is scrambled."""
        from sklearn.inspection import permutation_importance

        if self.regressor_ is None or self.encoder_ is None or self.columns_ is None:
            raise RuntimeError("feature_importance called before fit")

        prepared = self._prepare(frame)
        features = align_categories(
            self.columns_, build_features(prepared, self.encoder_, self.feature_config)
        )
        target = np.log(rate_per_mile(frame))
        result = permutation_importance(
            self.regressor_,
            features,
            target,
            n_repeats=n_repeats,
            random_state=self.model_config.random_state,
            scoring="neg_mean_absolute_error",
        )
        return pd.Series(
            result.importances_mean, index=features.columns
        ).sort_values(ascending=False)
