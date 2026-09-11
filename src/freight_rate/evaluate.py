"""Scoring, and the loop that runs the backtest.

Spotter works out the final mark after submission and has not said which measure
they use, so several are reported rather than one being tuned for. MAE and WAPE are the
usual freight measures. RMSE is here because it reacts hardest to the extreme
rows, and the gap between MAE and RMSE says what the deliberately
kept corrupted prices cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET, FeatureConfig, ModelConfig, SplitConfig
from .model import RateModel
from .splits import apply_fold, rolling_origin_folds


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    absolute = np.abs(error)
    percentage = absolute / np.maximum(np.abs(actual), 1e-9)

    total_variance = np.sum((actual - actual.mean()) ** 2)
    r_squared = 1.0 - np.sum(error**2) / total_variance if total_variance else float("nan")

    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        # WAPE is total error over total value. MAPE leans too hard on cheap
        # short trips, where a few dollars off reads as a large percentage.
        "wape": float(absolute.sum() / np.maximum(actual.sum(), 1e-9)),
        "mape": float(percentage.mean()),
        "median_ape": float(np.median(percentage)),
        "r2": float(r_squared),
        "bias": float(error.mean()),
        "n": int(actual.size),
    }


def backtest(
    frame: pd.DataFrame,
    feature_config: FeatureConfig,
    model_config: ModelConfig,
    split_config: SplitConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the backtest and return the numbers for each fold.

    Every fold builds a new RateModel, so the lane scores, the coordinate table,
    the outlier cut-off and the smearing factor are all rebuilt on that fold's
    training rows and never see anything later.
    """
    folds = rolling_origin_folds(frame, split_config)
    if not folds:
        raise ValueError("no folds produced. Check SplitConfig against the date range")

    rows = []
    for fold in folds:
        train, valid = apply_fold(frame, fold)
        if train.empty or valid.empty:
            continue

        model = RateModel(feature_config, model_config).fit(train)
        predicted = model.predict(valid)
        fold_metrics = metrics(valid[TARGET].to_numpy(), predicted)

        row = {
            "fold": fold.index,
            "train_end": fold.train_end.date().isoformat(),
            "valid_start": fold.valid_start.date().isoformat(),
            "valid_end": fold.valid_end.date().isoformat(),
            "train_rows": len(train),
            **fold_metrics,
        }
        rows.append(row)
        if verbose:
            print(
                f"  {fold.label()} | train {len(train):>6,} rows | "
                f"MAE ${row['mae']:>7.2f} | WAPE {row['wape']:.4f} | "
                f"R2 {row['r2']:.4f}"
            )

    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> dict[str, float]:
    """Average and spread across folds.

    The spread earns as much attention as the average. A model whose error
    climbs fold by fold is drifting, and the November and December window sits
    further out than any fold reaches.
    """
    numeric = ["mae", "rmse", "wape", "mape", "median_ape", "r2", "bias"]
    summary = {f"{name}_mean": float(results[name].mean()) for name in numeric}
    summary.update({f"{name}_std": float(results[name].std(ddof=0)) for name in numeric})
    summary["folds"] = int(len(results))
    return summary
