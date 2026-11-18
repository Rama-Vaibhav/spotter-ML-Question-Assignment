"""Time-forward validation design.

The final task is: train on Jan 1 - Oct 31, predict Nov 1 - Dec 31 (a strictly later
period). A random K-fold split would leak future market levels into training and
overstate accuracy, so every fold here is a forward-in-time split.

Because rates have a quarterly structure (a new base level each quarter + a ramp into
quarter end), the most faithful replica of the real task is a "quarter-analog" fold:
train up to the end of a quarter's first month, predict its remaining two months,
including the quarter-end ramp. That is exactly Oct -> Nov/Dec. We also report a plain
"last two months" holdout (harder: its test quarter is entirely unseen) and a cold-start
test with cities removed from training (12% of final rows involve unseen cities).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FOLDS = [
    {"fold": "Q2-analog", "description": "train Jan-Apr -> predict May-Jun",
     "train_end": "2025-04-30", "test_start": "2025-05-01", "test_end": "2025-06-30"},
    {"fold": "Q3-analog", "description": "train Jan-Jul -> predict Aug-Sep",
     "train_end": "2025-07-31", "test_start": "2025-08-01", "test_end": "2025-09-30"},
    {"fold": "Last-2-months", "description": "train Jan-Aug -> predict Sep-Oct",
     "train_end": "2025-08-31", "test_start": "2025-09-01", "test_end": "2025-10-31"},
]
COLD_CITY_FOLD = FOLDS[1]
N_COLD_CITIES = 8


def split(df: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df.date <= fold["train_end"]]
    test = df[(df.date >= fold["test_start"]) & (df.date <= fold["test_end"])]
    return train, test


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "n": int(len(y_true)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_pct": float(100 * ape.mean()),
        "MedAPE_pct": float(100 * np.median(ape)),
        "bias_pct": float(100 * np.mean(err / y_true)),
    }
