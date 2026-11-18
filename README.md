# Freight Rate Prediction

Predicts `posted_rate` for 12,000 freight loads dated Nov–Dec 2025. Training uses 48,000 labelled loads from Jan–Oct 2025.

**Final model:** a log-additive structural ridge regression plus a gradient-boosted model of its residuals.
**Time-forward backtest MAPE:** ~1.5% on folds that replicate the real Oct → Nov/Dec setup (lane-median baseline: 3.5–6%).

The full write-up is in [`reports/Freight_Rate_Model_Report.docx`](reports/Freight_Rate_Model_Report.docx), with a PDF copy alongside it.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
python -m pip install -r requirements.txt

python train.py        # data-quality report, backtests, final fit, writes predictions (~30 s)
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv   # validates + December chart
python eda.py          # figures -> reports/figures/
python make_report.py  # report -> reports/Freight_Rate_Model_Report.docx
```

Outputs:

| File | What |
|---|---|
| `validation_predictions.csv` | `load_id,predicted_rate` for all 12,000 validation loads (submission) |
| `data/december_chart_inputs.csv` | the 31 fixed December rows with `predicted_rate` filled |
| `scorer_results/candidate_december.png` | December chart produced by the provided `score.py` |
| `reports/validation_metrics.csv` | every model × fold × metric |
| `reports/data_quality.json` | data-quality counts |
| `reports/figures/*.png` | EDA and validation figures |
| `models/final_model.pkl` | fitted final model (recreated by `train.py`) |

## Repository layout

```
data/                      provided datasets (canonical names expected by score.py)
src/freight_rate/
  config.py                paths and constants
  data.py                  loading, daily market table, corrupted-label detection, data-quality report
  features.py              cleaning of features, calendar features, unseen-city geographic encoding
  models.py                baseline, structural ridge, GBM-only, hybrid (final)
  validation.py            time-forward fold definitions and metrics
  pipeline.py              end-to-end orchestration (called by train.py)
train.py  eda.py  make_report.py  score.py (provided)
docs/                      original assignment README
```

## Approach in brief

**Key data findings**
- `log(rate)` is close to additive in its drivers: distance curve, equipment, weight, pickup city and delivery city effects. Genuine labels have only ~3% noise around that structure.
- Each quarter has its own base level, and rates ramp up about 4–5% over the last ~5 weeks of every quarter, then reset on day 1. Nov–Dec is the end of Q4, so the forecast must include that ramp.
- `market_index` is a daily market level (with a 7-day cycle) plus row noise. Its daily mean is the useful signal.

**Data quality**
- **Rate outliers:** 677 rate labels (1.4%) are off by 2–6×. They are detected with a robust baseline and removed from training.
- **Weight:** negative weights are sign errors, so absolute values are used. Missing weights get the training median plus a flag; the 47,500 lb cap gets its own flag.
- **Market index:** missing values are filled with that day's mean.
- **Unseen cities:** 8 validation cities (1,447 rows) never appear in training. They get geographic interpolation of nearby cities' effects.

**Split and validation**
- All splits are **time-forward**, because the task is a forecast of a later period.
- The **quarter-analog** folds are primary. They train through the first month of a quarter and predict its last two months (Jan–Apr → May–Jun; Jan–Jul → Aug–Sep), exactly like Oct → Nov/Dec.
- Two extra checks: a last-2-months holdout (Jan–Aug → Sep–Oct) and a cold-city test (8 cities removed from training).
- Outlier flags, imputation values, splines and city effects are fitted inside each training window only.

**Why the hybrid**
- The ridge part keeps the time structure additive (quarter level + quarter-end ramp), so it extrapolates correctly to new dates and new cities.
- The GBM adds non-linear interactions but has no absolute-date feature, so it cannot drift on future dates.

**December chart**
- `december_chart_inputs.csv` has no market columns, so each day's `market_index` and `quote_signal` are set to that day's mean across `validation.csv`. These are feature values only, no labels.
