"""Deciding which rows train the model and which rows test it.

train_test.csv covers 1 January to 31 October 2025. validation.csv starts the
next day and runs to 31 December. Every row being graded sits in the future
relative to every row available to learn from.

That rules out shuffling. Each day in the file carries about 158 loads, so a
shuffled split hands the model most of a day and asks it to fill in the rest.
That reads the answer off the row next door instead of forecasting, and it
looks far better than it is. Shuffled folds report $84 mean absolute error,
against $123 for the real thing.

The check is one line. Count the dates that appear in both halves. Anything
other than zero is a leak.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DATE, SplitConfig


@dataclass(frozen=True)
class Fold:
    """One round of the backtest."""

    index: int
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    def label(self) -> str:
        return (
            f"fold {self.index}: train to {self.train_end.date()}, "
            f"score {self.valid_start.date()} to {self.valid_end.date()}"
        )


def rolling_origin_folds(frame: pd.DataFrame, config: SplitConfig) -> list[Fold]:
    """Build folds that each forecast ``horizon_days`` past their cutoff.

    The training window grows rather than slides. The final model learns from
    all ten months, so each fold should learn from everything up to its own
    cutoff. That keeps the backtest a rehearsal of the model actually shipped.

    The last fold ends on the final training date, which makes it the closest
    available analogue to the real November and December job.
    """
    dates = frame[DATE]
    last_date = dates.max()
    step = pd.Timedelta(days=config.step_days or config.horizon_days)
    horizon = pd.Timedelta(days=config.horizon_days)

    folds: list[Fold] = []
    for offset in range(config.n_folds):
        valid_end = last_date - step * offset
        valid_start = valid_end - horizon + pd.Timedelta(days=1)
        train_end = valid_start - pd.Timedelta(days=1)
        if train_end <= dates.min():
            break
        folds.append(
            Fold(
                index=config.n_folds - offset,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )

    folds.sort(key=lambda fold: fold.train_end)
    # Renumber so fold 1 is the earliest.
    return [
        Fold(position + 1, fold.train_end, fold.valid_start, fold.valid_end)
        for position, fold in enumerate(folds)
    ]


def apply_fold(frame: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut a frame into training rows and scoring rows for one fold."""
    dates = frame[DATE]
    train = frame[dates <= fold.train_end]
    valid = frame[(dates >= fold.valid_start) & (dates <= fold.valid_end)]
    return train, valid


def holdout_split(
    frame: pd.DataFrame, horizon_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent window only. Faster, for quick checks."""
    dates = frame[DATE]
    valid_start = dates.max() - pd.Timedelta(days=horizon_days - 1)
    return frame[dates < valid_start], frame[dates >= valid_start]
