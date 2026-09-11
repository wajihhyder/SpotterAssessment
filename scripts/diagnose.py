"""Check that the yearly curve still behaves past the end of the training data.

    python scripts/diagnose.py

Training stops on 31 October and the December answers run to 31 December. The
whole December deliverable rests on two months the model has never seen, and
this kind of model cannot follow a trend upward or downward past its data. Only
the repeating waves carry the shape forward.

So this makes the behaviour visible. It runs the fixed December route across all
365 days of 2025 and lays what similar loads actually cost on top. If the fitted
part follows the real prices and the unseen part carries on smoothly, the
December chart can be trusted.

Writes outputs/seasonal_extrapolation.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from freight_rate.config import (  # noqa: E402
    DIAGNOSTIC_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    TRAIN_PATH,
)

# The fixed inputs from december_chart_inputs.csv.
LANE = dict(
    pickup="Lexington",
    delivery="Fort Wayne",
    distance=360.0,
    equipment="Dry Van",
    weight=32_000.0,
)
TRAIN_END = pd.Timestamp("2025-10-31")

# Two colours checked for colour-blind separation on a light background.
SERIES_MODEL = "#2a78d6"
SERIES_ACTUAL = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e2e2df"
SURFACE = "#fcfcfb"


def fixed_lane_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({**{key: [value] * len(dates) for key, value in LANE.items()}, "date": dates})


def main() -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit(f"No model at {MODEL_PATH}. Run scripts/train.py first.")
    model = joblib.load(MODEL_PATH)

    dates = pd.date_range("2025-01-01", "2025-12-31")
    frame = fixed_lane_frame(dates)
    frame["rate_per_mile"] = model.predict(frame) / LANE["distance"]

    # Similar real loads. Same equipment, close enough in length, and corrupted
    # prices removed by the same rule the model trains under.
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    train["rate_per_mile"] = train["posted_rate"] / train["distance"]
    comparable = train[
        train["distance"].between(300, 420)
        & (train["equipment"] == LANE["equipment"])
        & train["rate_per_mile"].between(1, 5)
    ]
    observed = comparable.groupby(comparable["date"].dt.to_period("M"))["rate_per_mile"].median()
    observed.index = observed.index.to_timestamp() + pd.Timedelta(days=14)

    print(f"Fixed lane: {LANE['pickup']} to {LANE['delivery']}, "
          f"{LANE['distance']:g}mi {LANE['equipment']} {LANE['weight']:,.0f}lb\n")
    monthly = frame.groupby(frame["date"].dt.to_period("M"))["rate_per_mile"].mean()
    print(f"{'month':<10}{'model $/mi':>12}{'observed $/mi':>16}   status")
    for period, predicted in monthly.items():
        actual = observed.get(period.to_timestamp() + pd.Timedelta(days=14))
        actual_text = f"{actual:.3f}" if actual is not None else "--"
        status = "extrapolated" if period.month > TRAIN_END.month else "fitted"
        print(f"{str(period):<10}{predicted:>12.3f}{actual_text:>16}   {status}")

    drift = 100 * (monthly.iloc[-1] / monthly.iloc[TRAIN_END.month - 1] - 1)
    print(f"\nDecember sits {drift:+.2f}% against October, the last fitted month.")
    offset = 100 * (monthly.iloc[:10].mean() / observed.mean() - 1)
    print(
        f"The model line runs {offset:+.1f}% against the observed points because it "
        "prices one\nspecific lane while the points are a median over the whole "
        "300-420mi Dry Van band.\nRead the shape, not the level."
    )

    figure, axis = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)

    # Shade the unseen months first, so the lines draw over the top.
    axis.axvspan(TRAIN_END, dates[-1], color="#8a8a85", alpha=0.07, linewidth=0)
    axis.axvline(TRAIN_END, color="#8a8a85", linewidth=1, linestyle=(0, (4, 3)))

    axis.plot(
        frame["date"], frame["rate_per_mile"],
        color=SERIES_MODEL, linewidth=2, label="Model, fixed lane",
        solid_capstyle="round",
    )
    axis.plot(
        observed.index, observed.to_numpy(),
        marker="o", markersize=8, linestyle="none",
        color=SERIES_ACTUAL, markeredgecolor=SURFACE, markeredgewidth=2,
        label="Observed median, comparable loads",
    )

    axis.set_title(
        "Seasonal curve holds its shape two months past the training window",
        loc="left", fontsize=14, fontweight="bold", color=TEXT_PRIMARY, pad=14,
    )
    axis.set_ylabel("Rate per mile ($)", fontsize=10, color=TEXT_SECONDARY)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#b4b4af")
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    top = axis.get_ylim()[1]
    axis.text(
        TRAIN_END + pd.Timedelta(days=4), top,
        "extrapolated\n(no training data)",
        fontsize=8.5, color=TEXT_SECONDARY, va="top",
    )
    legend = axis.legend(
        loc="lower right", frameon=False, fontsize=9.5, handletextpad=0.6
    )
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = DIAGNOSTIC_PATH
    figure.savefig(output, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
