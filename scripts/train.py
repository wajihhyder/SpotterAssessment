"""Backtest the model, then fit it on everything and save it.

    python scripts/train.py
    python scripts/train.py --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402

from freight_rate.config import (  # noqa: E402
    METRICS_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    FeatureConfig,
    ModelConfig,
    SplitConfig,
)
from freight_rate.data import load_train  # noqa: E402
from freight_rate.evaluate import backtest, summarise  # noqa: E402
from freight_rate.model import RateModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="backtest the feature choices as well, so the numbers behind them "
        "can be checked rather than taken on trust",
    )
    parser.add_argument("--folds", type=int, default=SplitConfig.n_folds)
    parser.add_argument("--horizon-days", type=int, default=SplitConfig.horizon_days)
    parser.add_argument(
        "--no-smearing",
        action="store_true",
        help="skip the back-transform correction, which suits MAE and costs RMSE",
    )
    parser.add_argument("--importance", action="store_true", help="rank the features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train, report = load_train()
    print("Loaded training data")
    print(report.summary())
    print(
        f"\ndate range: {train['date'].min().date()} to {train['date'].max().date()} "
        f"({train['date'].nunique()} days)"
    )

    split_config = SplitConfig(horizon_days=args.horizon_days, n_folds=args.folds)
    model_config = ModelConfig(apply_smearing=not args.no_smearing)
    feature_config = FeatureConfig()

    print(
        f"\nBacktest: {args.folds} folds, {args.horizon_days}-day horizon, "
        "matching the 1 Nov to 31 Dec window being graded"
    )
    results = backtest(train, feature_config, model_config, split_config)
    summary = summarise(results)
    print(
        f"\n  mean MAE ${summary['mae_mean']:.2f} (sd ${summary['mae_std']:.2f}), "
        f"mean WAPE {summary['wape_mean']:.4f}, mean R2 {summary['r2_mean']:.4f}"
    )

    payload = {
        "default": {
            "feature_config": asdict(feature_config),
            "model_config": asdict(model_config),
            "split_config": asdict(split_config),
            "folds": results.to_dict(orient="records"),
            "summary": summary,
        }
    }

    if args.compare:
        variants = {
            "without_geo": (
                replace(feature_config, use_geo=False),
                "Without coordinates",
            ),
            "with_macro": (
                replace(feature_config, use_macro=True),
                "With market_index and quote_signal",
            ),
        }
        for key, (config, label) in variants.items():
            print(f"\n{label}")
            variant_results = backtest(train, config, model_config, split_config)
            variant_summary = summarise(variant_results)
            delta = variant_summary["mae_mean"] - summary["mae_mean"]
            print(
                f"\n  mean MAE ${variant_summary['mae_mean']:.2f}, "
                f"WAPE {variant_summary['wape_mean']:.4f}, "
                f"R2 {variant_summary['r2_mean']:.4f}"
            )
            print(
                f"  against the default: ${delta:+.2f} "
                f"({100 * delta / summary['mae_mean']:+.2f}%)"
            )
            payload[key] = {
                "feature_config": asdict(config),
                "folds": variant_results.to_dict(orient="records"),
                "summary": variant_summary,
            }

    print("\nFitting the final model on all ten months")
    model = RateModel(feature_config, model_config).fit(train)
    assert model.report_ is not None
    print(f"  {model.report_.summary()}")

    if args.importance:
        print("\nFeature ranking, top 12:")
        for name, value in model.feature_importance(train).head(12).items():
            print(f"  {name:22} {value:.5f}")

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(payload, indent=2))

    print(f"\nSaved model   -> {MODEL_PATH.relative_to(MODEL_PATH.parents[1])}")
    print(f"Saved metrics -> {METRICS_PATH.relative_to(METRICS_PATH.parents[1])}")


if __name__ == "__main__":
    main()
