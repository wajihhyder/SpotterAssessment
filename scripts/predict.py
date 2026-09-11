"""Write the two submission files.

    python scripts/predict.py

  validation_predictions.csv        12,000 rows of load_id,predicted_rate
  outputs/december_predictions.csv  the December file with the column filled

Both are written to pass score.py's checks exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from freight_rate.config import (  # noqa: E402
    DECEMBER_INPUT_PATH,
    DECEMBER_OUTPUT_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    PREDICTION_TEMPLATE_PATH,
    SUBMISSION_PATH,
)
from freight_rate.data import load_december_inputs, load_validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def write_validation_predictions(model) -> pd.DataFrame:
    validation, report = load_validation()
    print("Loaded validation data")
    print(report.summary())

    predictions = model.predict(validation)
    frame = pd.DataFrame(
        {"load_id": validation["load_id"], "predicted_rate": predictions.round(2)}
    )

    # Line the rows up against the template rather than trusting the two files
    # to already agree on order and contents.
    template = pd.read_csv(PREDICTION_TEMPLATE_PATH)
    frame = template[["load_id"]].merge(frame, on="load_id", how="left")
    if frame["predicted_rate"].isna().any():
        missing = int(frame["predicted_rate"].isna().sum())
        raise RuntimeError(f"{missing} template load_ids got no prediction")

    frame.to_csv(SUBMISSION_PATH, index=False)
    print(
        f"\nWrote {len(frame):,} predictions -> {SUBMISSION_PATH.name} "
        f"(median ${frame['predicted_rate'].median():,.2f}, "
        f"range ${frame['predicted_rate'].min():,.2f} to ${frame['predicted_rate'].max():,.2f})"
    )
    return frame


def write_december_predictions(model) -> pd.DataFrame:
    december = load_december_inputs()
    predictions = model.predict(december)

    # score.py wants the original seven columns in the original order, so the
    # raw file is re-read and one column filled in.
    original = pd.read_csv(DECEMBER_INPUT_PATH)
    original["predicted_rate"] = predictions.round(2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original.to_csv(DECEMBER_OUTPUT_PATH, index=False)
    print(
        f"Wrote 31 December predictions -> {DECEMBER_OUTPUT_PATH.name} "
        f"(range ${original['predicted_rate'].min():,.2f} to "
        f"${original['predicted_rate'].max():,.2f})"
    )
    return original


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"No model at {args.model}. Run scripts/train.py first.")

    model = joblib.load(args.model)
    write_validation_predictions(model)
    write_december_predictions(model)

    print("\nNext: python score.py --predictions validation_predictions.csv \\")
    print("        --december-predictions outputs/december_predictions.csv")


if __name__ == "__main__":
    main()
