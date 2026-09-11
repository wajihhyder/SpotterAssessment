"""Turning raw columns into things the model can use.

One rule shapes this file. december_chart_inputs.csv provides six columns:
pickup, delivery, distance, equipment, weight and date. Anything the December
chart needs has to come from those six, or from a lookup keyed on them.
Coordinates qualify, because every city has one fixed pair. market_index and
quote_signal do not, so the model never touches them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DATE, TARGET, FeatureConfig
from .data import rate_per_mile

EQUIPMENT_COLUMN = "equipment"
CATEGORICAL_FEATURES = [EQUIPMENT_COLUMN]

#: Fixed starting point for the seasonal maths, so training, validation and
#: December all sit on the same part of the wave.
_EPOCH = pd.Timestamp("2025-01-01")


@dataclass
class LaneEncoder:
    """Scores each city and each route by how its prices sit against average.

    Eight cities appear in the scoring file and never in training. Charlotte,
    Chicago, Allentown, Norfolk, Knoxville, San Diego, Laredo and Jackson. They
    touch 1,447 rows, which is 12.1% of the graded set.

    A column-per-city scheme has no column for Chicago and either breaks or
    emits nonsense. Numbering the cities is worse, because the model would read
    city 65 as larger than city 64. This encoder stores how far above or below
    average each city prices, and returns zero for anything it has not seen. An
    unknown city becomes an average city, and distance, equipment, weight and
    season carry the prediction instead.

    Scores are on price per mile, so a city's score does not depend on how long
    its trips happen to be.
    """

    smoothing: float
    prior_: float = 0.0
    pickup_: pd.Series | None = None
    delivery_: pd.Series | None = None
    lane_: pd.Series | None = None
    pickup_counts_: pd.Series | None = None
    delivery_counts_: pd.Series | None = None

    @staticmethod
    def _lane_key(frame: pd.DataFrame) -> pd.Series:
        return frame["pickup"].astype(str) + " -> " + frame["delivery"].astype(str)

    def _encode_group(self, keys: pd.Series, target: pd.Series) -> pd.Series:
        """Distance from average, shrunk towards zero when evidence is thin."""
        stats = target.groupby(keys).agg(["mean", "count"])
        weight = stats["count"] / (stats["count"] + self.smoothing)
        return (stats["mean"] - self.prior_) * weight

    def fit(self, frame: pd.DataFrame) -> "LaneEncoder":
        log_rpm = np.log(rate_per_mile(frame))
        self.prior_ = float(log_rpm.mean())

        self.pickup_ = self._encode_group(frame["pickup"], log_rpm)
        self.delivery_ = self._encode_group(frame["delivery"], log_rpm)
        self.lane_ = self._encode_group(self._lane_key(frame), log_rpm)

        self.pickup_counts_ = frame["pickup"].value_counts()
        self.delivery_counts_ = frame["delivery"].value_counts()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.pickup_ is None:  # pragma: no cover - guarded by callers
            raise RuntimeError("LaneEncoder.transform called before fit")
        lane_keys = self._lane_key(frame)
        return pd.DataFrame(
            {
                "pickup_te": frame["pickup"].map(self.pickup_).fillna(0.0).to_numpy(),
                "delivery_te": frame["delivery"].map(self.delivery_).fillna(0.0).to_numpy(),
                "lane_te": lane_keys.map(self.lane_).fillna(0.0).to_numpy(),
                # How many rows back each score. Lets the model discount a score
                # it has little evidence for, and marks unseen cities with zero.
                "pickup_support": np.log1p(
                    frame["pickup"].map(self.pickup_counts_).fillna(0.0).to_numpy()
                ),
                "delivery_support": np.log1p(
                    frame["delivery"].map(self.delivery_counts_).fillna(0.0).to_numpy()
                ),
            },
            index=frame.index,
        )


def _seasonal_terms(dates: pd.Series, config: FeatureConfig) -> pd.DataFrame:
    """Sine waves for the yearly and weekly cycles.

    Training stops on 2025-10-31 and predictions run to 2025-12-31. A feature
    that counts time forwards, such as a day number, has no values past October
    and simply holds its last one. Waves do not have that problem. They repeat,
    so December sits at a real point on the yearly cycle and gets a real answer.

    There is a cycle to capture. Median price per mile runs $2.01 in January,
    peaks near $2.30 in late June, and settles at $2.16 by October.
    """
    day_of_year = dates.dt.dayofyear.to_numpy(dtype=float)
    day_of_week = dates.dt.dayofweek.to_numpy(dtype=float)

    columns: dict[str, np.ndarray] = {}
    for harmonic in range(1, config.annual_harmonics + 1):
        angle = 2.0 * np.pi * harmonic * day_of_year / 365.25
        columns[f"annual_sin_{harmonic}"] = np.sin(angle)
        columns[f"annual_cos_{harmonic}"] = np.cos(angle)
    for harmonic in range(1, config.weekly_harmonics + 1):
        angle = 2.0 * np.pi * harmonic * day_of_week / 7.0
        columns[f"weekly_sin_{harmonic}"] = np.sin(angle)
        columns[f"weekly_cos_{harmonic}"] = np.cos(angle)
    return pd.DataFrame(columns, index=dates.index)


def _geo_terms(frame: pd.DataFrame) -> pd.DataFrame:
    """Where the trip sits on the map, and which way it runs.

    The supplied coordinates are not real geography. Richmond is not at 38.09
    north. They are, however, a consistent map of their own. Straight-line
    distance between two cities tracks the distance column at 0.9995, and the
    ratio between road and straight line sits between 1.16 and 1.26 for 99.8%
    of city pairs, which is how real road networks behave.

    That consistency is all the model needs, because it only ever compares
    positions to each other. Adding these cuts mean absolute error by about 2%
    overall, and by 7% on rows whose city never appears in training. Those rows
    carry their own coordinates in validation.csv even though the city is
    unknown, so the model can place the city and read across from its
    neighbours.
    """
    pickup_lat = frame["pickup_lat"].astype(float)
    pickup_lon = frame["pickup_lon"].astype(float)
    delivery_lat = frame["delivery_lat"].astype(float)
    delivery_lon = frame["delivery_lon"].astype(float)
    return pd.DataFrame(
        {
            "pickup_lat": pickup_lat,
            "pickup_lon": pickup_lon,
            "delivery_lat": delivery_lat,
            "delivery_lon": delivery_lon,
            # Which way the load travels. Hauls into one part of the map price
            # differently from hauls out of it, and distance alone cannot say.
            "delta_lat": delivery_lat - pickup_lat,
            "delta_lon": delivery_lon - pickup_lon,
            # Roughly where the whole trip sits.
            "mid_lat": (delivery_lat + pickup_lat) / 2.0,
            "mid_lon": (delivery_lon + pickup_lon) / 2.0,
        },
        index=frame.index,
    )


def build_features(
    frame: pd.DataFrame,
    encoder: LaneEncoder,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Assemble the table the model reads.

    Column order is fixed so training and prediction always line up.
    """
    dates = frame[DATE]
    distance = frame["distance"].astype(float)
    weight = frame["weight"].astype(float)

    features = pd.DataFrame(index=frame.index)

    # Distance is the biggest single driver, at 0.91 against the raw price.
    # Price per mile falls as trips get longer, from $2.74 under 250 miles to
    # $1.91 past 2,000. The log version gives the model a clean axis to split
    # that curve on.
    features["distance"] = distance
    features["log_distance"] = np.log(distance)

    features["weight"] = weight
    features["weight_missing"] = frame["weight"].isna().astype(float)
    # Heavy freight on a short run prices differently from the same tonnage
    # spread over a long one.
    features["weight_per_100mi"] = weight / (distance / 100.0)

    features["day_of_year"] = dates.dt.dayofyear.astype(float)
    features["day_of_week"] = dates.dt.dayofweek.astype(float)
    features["month"] = dates.dt.month.astype(float)
    features["is_weekend"] = (dates.dt.dayofweek >= 5).astype(float)
    features["day_of_month"] = dates.dt.day.astype(float)

    features = pd.concat([features, _seasonal_terms(dates, config)], axis=1)
    features = pd.concat([features, encoder.transform(frame)], axis=1)

    if config.use_geo:
        features = pd.concat([features, _geo_terms(frame)], axis=1)

    if config.use_macro:
        # Only reachable on validation.csv. The December file has neither
        # column, and both make the model worse anyway. See FeatureConfig.
        features["market_index"] = frame["market_index"].astype(float)
        features["quote_signal"] = frame["quote_signal"].astype(float)
        features["market_index_missing"] = frame["market_index"].isna().astype(float)

    features[EQUIPMENT_COLUMN] = frame[EQUIPMENT_COLUMN].astype("category")
    return features


def fit_encoder(frame: pd.DataFrame, config: FeatureConfig) -> LaneEncoder:
    """Fit the lane encoder. It must only ever see training rows."""
    if TARGET not in frame.columns:
        raise ValueError("fit_encoder requires the labelled target column")
    return LaneEncoder(smoothing=config.lane_smoothing).fit(frame)


def align_categories(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Give ``other`` the same category order as ``train``.

    The model stores categories as numbers. If the order differs between fit and
    predict, the equipment types quietly swap meaning and nothing complains.
    """
    aligned = other.copy()
    for column in CATEGORICAL_FEATURES:
        aligned[column] = pd.Categorical(
            aligned[column], categories=train[column].cat.categories
        )
    return aligned[train.columns]
