# Freight Rate Prediction

Predicts `posted_rate` for the 12,000 loads in `data/validation.csv`, and fills
the 31 December rows in `data/december_chart_inputs.csv`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 or newer.

## Run

Two commands, in this order.

```bash
python scripts/train.py
python scripts/predict.py
```

`train.py` backtests the model, fits it on all ten months of training data, and
saves it to `outputs/model.joblib`. Takes about two minutes.

`predict.py` loads that model and writes both answer files.

| File | What it is |
| --- | --- |
| `validation_predictions.csv` | the submission, 12,000 rows of `load_id,predicted_rate` |
| `outputs/december_predictions.csv` | the December file with `predicted_rate` filled in |
| `outputs/model.joblib` | the trained model |
| `outputs/cv_metrics.json` | the score for every backtest fold |

To produce the December chart, pass both files to the supplied scorer.

```bash
python score.py \
    --predictions validation_predictions.csv \
    --december-predictions outputs/december_predictions.csv
```

That writes `scorer_results/candidate_december.png`.

### Checking the December extrapolation

Training stops on 31 October while the December answers run to 31 December, so
the whole December chart rests on two months the model has never seen.

```bash
python scripts/diagnose.py
```

This prices the fixed December route across all 365 days of 2025 and lays what
similar loads actually cost on top, so the fitted months can be compared against
real prices and the extrapolated ones read as a continuation of them. It writes
`outputs/seasonal_extrapolation.png`.

### Options

```bash
python scripts/train.py --compare       # also backtest the two feature decisions
python scripts/train.py --importance    # rank the features
python scripts/train.py --no-smearing   # drop the back-transform correction
python scripts/train.py --folds 6 --horizon-days 30
```

Everything is seeded at `random_state=42`, so a rerun reproduces the same
numbers.

## Layout

```
score.py                  the supplied scorer, unchanged
data/                     the four supplied CSVs, unchanged
scripts/
  train.py                backtest, then fit and save
  predict.py              write the two answer files
  diagnose.py             check the December extrapolation
src/freight_rate/
  config.py               every setting, in one place
  data.py                 reading the files and fixing what is broken
  features.py             turning raw columns into model input
  splits.py               deciding what trains and what tests
  model.py                the model
  evaluate.py             scoring and the backtest loop
```

## How it works

**The split is by date, never shuffled.** `train_test.csv` covers 1 January to
31 October 2025 and `validation.csv` picks up the next day, so every load being
graded happened after every load available to learn from. Each backtest fold
copies that shape: train up to a cutoff, then score the next 61 days, which is
the length of the real November and December window. Four folds, and everything
fitted is rebuilt inside each one so no fold sees anything later than its own
cutoff.

**The model predicts price per mile**, then multiplies distance back in.
Distance explains most of what the price does, so aiming at the total would
spend the model's capacity rediscovering that long trips cost more. Working in
logs also makes the error proportional.

**Gradient boosting**, specifically scikit-learn's `HistGradientBoostingRegressor`.
It needs no extra install, treats a blank value as its own case, and splits on
equipment as a real category.

**Cleaning.** 292 negative weights get their sign flipped. Missing weights are
left blank for the model to handle. 674 corrupted prices, judged on price per
mile rather than the total, are dropped from training and kept in scoring.

**Unseen cities.** Eight appear only in `validation.csv` and touch 12% of the
graded rows. Each city carries a score for how far above or below average it
prices, and an unseen one scores as average.

**Season** is encoded as repeating waves rather than a day counter, because
training stops two months before the predictions end and a counter would freeze
at its last value. `diagnose.py` is what checks that this holds up.
