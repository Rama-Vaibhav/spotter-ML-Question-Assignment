"""Loading, cleaning and data-quality handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from .config import OUTLIER_LOG_THRESHOLD, TARGET


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One lat/lon per city, pooled from pickup and delivery columns of every frame."""
    parts = []
    for f in frames:
        if {"pickup_lat", "pickup_lon"}.issubset(f.columns):
            parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
            parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    coords = pd.concat(parts).groupby("city")[["lat", "lon"]].median()
    return coords


def daily_market(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-day mean of market_index / quote_signal over all available feature rows.

    market_index is a common daily market level plus row-level noise (within-day std
    ~0.025), so the daily mean is a de-noised market signal and is also how missing
    values and the December chart inputs (which lack these columns) are filled.
    Only feature columns are used, never labels.
    """
    rows = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    return rows.groupby("date").agg(
        mi_daily=("market_index", "mean"), qs_daily=("quote_signal", "mean")
    )


def complete_december_inputs(dec: pd.DataFrame, coords: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """The December chart file only has lane/equipment/weight/date. Add the missing
    columns the model needs: city coordinates and that day's market conditions."""
    out = dec.copy()
    out["date"] = pd.to_datetime(out["date"])
    for role in ("pickup", "delivery"):
        out[f"{role}_lat"] = out[role].map(coords["lat"])
        out[f"{role}_lon"] = out[role].map(coords["lon"])
    out["market_index"] = out["date"].map(market["mi_daily"])
    out["quote_signal"] = out["date"].map(market["qs_daily"])
    return out


def flag_rate_outliers(train: pd.DataFrame, threshold: float = OUTLIER_LOG_THRESHOLD) -> pd.Series:
    """Flag corrupted posted_rate labels using an iteratively re-fit robust baseline.

    Baseline: log(rate) ~ log-distance polynomial + equipment + |weight| + pickup and
    delivery city effects. ~1.4% of rows are off by roughly 2-6x (unit / keying errors);
    these are removed from training only (they are not predictable from features).
    """
    df = train
    w = df["weight"].abs().fillna(df["weight"].abs().median())
    ld = np.log(df["distance"].to_numpy())
    numeric = np.column_stack([ld, ld**2, ld**3, w / 1e4])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cats = ohe.fit_transform(df[["equipment", "pickup", "delivery"]])
    X = np.hstack([numeric, cats])
    y = np.log(df[TARGET].to_numpy())
    keep = np.ones(len(df), dtype=bool)
    for _ in range(4):
        model = Ridge(alpha=1e-3).fit(X[keep], y[keep])
        resid = y - model.predict(X)
        new_keep = np.abs(resid) < threshold
        if (new_keep == keep).all():
            break
        keep = new_keep
    return pd.Series(~keep, index=df.index, name="is_outlier")


def data_quality_report(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    tr_cities = set(train.pickup) | set(train.delivery)
    va_cities = set(validation.pickup) | set(validation.delivery)
    new_cities = sorted(va_cities - tr_cities)
    rpm = train[TARGET] / train["distance"]
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_date_range": [str(train.date.min().date()), str(train.date.max().date())],
        "validation_date_range": [str(validation.date.min().date()), str(validation.date.max().date())],
        "missing_weight_train": int(train.weight.isna().sum()),
        "missing_weight_validation": int(validation.weight.isna().sum()),
        "negative_weight_train": int((train.weight < 0).sum()),
        "negative_weight_validation": int((validation.weight < 0).sum()),
        "weight_at_cap_train": int((train.weight.abs() == 47_500).sum()),
        "missing_market_index_train": int(train.market_index.isna().sum()),
        "missing_market_index_validation": int(validation.market_index.isna().sum()),
        "rate_per_mile_min": float(rpm.min()),
        "rate_per_mile_max": float(rpm.max()),
        "duplicate_load_ids": int(train.load_id.duplicated().sum()),
        "cities_train": len(tr_cities),
        "cities_validation": len(va_cities),
        "unseen_cities_in_validation": new_cities,
        "validation_rows_with_unseen_city": int(
            (validation.pickup.isin(new_cities) | validation.delivery.isin(new_cities)).sum()
        ),
    }
