"""Reading the files and fixing what is broken in them.

Every rule below came out of exploring the data before any of this was written.
The counts quoted are what the supplied files actually contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    COORDINATE_COLUMNS,
    DATE,
    DECEMBER_INPUT_PATH,
    TARGET,
    TRAIN_PATH,
    VALIDATION_PATH,
)


@dataclass
class CleaningReport:
    """A record of what the cleaner changed, so it can be printed and checked."""

    rows: int = 0
    negative_weights_fixed: int = 0
    missing_weight: int = 0
    missing_market_index: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"rows                    : {self.rows:,}",
            f"negative weights fixed  : {self.negative_weights_fixed:,}",
            f"missing weight (kept NaN)      : {self.missing_weight:,}",
            f"missing market_index (kept NaN): {self.missing_market_index:,}",
        ]
        lines += [f"{key:24}: {value:,}" for key, value in self.counts.items()]
        return "\n".join(lines)


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found at {path}. The CSVs belong in data/. "
            "See README.md for the layout."
        )
    return pd.read_csv(path, parse_dates=[DATE])


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Fix the two problems that appear in both the training and scoring files.

    Some weights are negative. A truck cannot weigh minus 30,000 lb, and the
    sizes themselves are normal, so the minus sign is a typing error. The sign is
    flipped back rather than the row being thrown away.

    Some weights and market_index values are missing. Those stay blank. The
    model treats blank as its own case and learns which way to lean, which beats
    filling in a made-up average.
    """
    result = frame.copy()
    report = CleaningReport(rows=len(result))

    negative = result["weight"] < 0
    report.negative_weights_fixed = int(negative.sum())
    result.loc[negative, "weight"] = result.loc[negative, "weight"].abs()

    report.missing_weight = int(result["weight"].isna().sum())
    if "market_index" in result.columns:
        report.missing_market_index = int(result["market_index"].isna().sum())

    zero_weight = int((result["weight"] == 0).sum())
    if zero_weight:
        report.counts["zero weight -> NaN"] = zero_weight
        result.loc[result["weight"] == 0, "weight"] = np.nan

    bad_distance = int((result["distance"] <= 0).sum())
    if bad_distance:
        report.counts["non-positive distance"] = bad_distance

    return result, report


def rate_per_mile(frame: pd.DataFrame) -> pd.Series:
    return frame[TARGET] / frame["distance"]


def outlier_mask(frame: pd.DataFrame, threshold: float) -> pd.Series:
    """Find rows whose price is corrupted.

    The judgement is made on price per mile rather than the total. The total runs from $57 to
    $25,533 only because trips run from 70 to 3,440 miles, so any fixed cut-off
    on the total would just flag long trips for being long. Dividing by distance
    first leaves the pricing itself, and there the normal figure is about $2.15
    a mile while the worst rows reach $14.

    The cut-off uses the median absolute deviation rather than the standard
    deviation. The bad rows are extreme enough to stretch a standard deviation
    wide enough to miss them.
    """
    log_rpm = np.log(rate_per_mile(frame))
    median = log_rpm.median()
    mad = (log_rpm - median).abs().median()
    if mad == 0:  # pragma: no cover - degenerate input
        return pd.Series(False, index=frame.index)
    # 0.6745 puts MAD on the same scale as a standard deviation for normal data.
    robust_z = 0.6745 * (log_rpm - median) / mad
    return robust_z.abs() > threshold


def city_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a city to coordinate lookup.

    Each city carries exactly one coordinate pair across all 48,000 rows, so
    this table is a plain fact about the data rather than an estimate.

    It exists because december_chart_inputs.csv gives city names but no
    coordinates. Both Lexington and Fort Wayne appear in the training file, so
    their coordinates are known and can be filled in by name.
    """
    pickup = frame[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    delivery = frame[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    both = pd.concat([pickup, delivery], ignore_index=True)
    return both.dropna(subset=["city"]).drop_duplicates("city").set_index("city")


def attach_coordinates(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Make sure a frame carries the four coordinate columns.

    validation.csv already has them on every row, including the eight cities
    that never appear in training, so those rows pass through untouched. The
    December file has none, so the lookup fills them in. A city the lookup
    does not know stays blank, and the model handles blank.
    """
    if all(column in frame.columns for column in COORDINATE_COLUMNS):
        return frame

    result = frame.copy()
    result["pickup_lat"] = result["pickup"].map(lookup["lat"])
    result["pickup_lon"] = result["pickup"].map(lookup["lon"])
    result["delivery_lat"] = result["delivery"].map(lookup["lat"])
    result["delivery_lon"] = result["delivery"].map(lookup["lon"])
    return result


def load_train(path: Path = TRAIN_PATH) -> tuple[pd.DataFrame, CleaningReport]:
    frame = _read(path, "training data")
    return clean(frame)


def load_validation(path: Path = VALIDATION_PATH) -> tuple[pd.DataFrame, CleaningReport]:
    frame = _read(path, "validation data")
    return clean(frame)


def load_december_inputs(path: Path = DECEMBER_INPUT_PATH) -> pd.DataFrame:
    """Load the 31 fixed December rows.

    This file holds six usable columns and nothing else. No coordinates, no
    market_index, no quote_signal. Coordinates get filled in by city name at
    predict time. The two market columns are not used by the model at all.
    """
    frame = pd.read_csv(path, parse_dates=[DATE])
    cleaned, _ = clean(frame)
    return cleaned
