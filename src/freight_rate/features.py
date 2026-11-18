"""Feature engineering shared by all models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WEIGHT_CAP

EARTH_RADIUS_MILES = 3958.8


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features. The key one is days_to_quarter_end: rates ramp up ~4-5% over
    the last ~5 weeks of every quarter (seen in Mar, Jun, Sep) and reset on day 1."""
    out = df
    d = out["date"]
    period = d.dt.to_period("Q")
    q_start = period.dt.start_time
    q_end = period.dt.end_time.dt.normalize()
    out["quarter_key"] = period.astype(str)
    out["days_since_quarter_start"] = (d - q_start).dt.days
    out["days_to_quarter_end"] = (q_end - d).dt.days
    out["days_to_month_end"] = d.dt.days_in_month - d.dt.day
    out["dow"] = d.dt.dayofweek
    return out


def prepare(df: pd.DataFrame, market: pd.DataFrame, weight_median: float) -> pd.DataFrame:
    """Clean features (never labels) and derive model inputs. Safe for train and inference."""
    out = df.copy()
    # Weight: negative values are sign errors (their rates match |weight|); missing are
    # imputed with the training median and flagged; 47,500 is a reporting cap - flagged.
    w = out["weight"].abs()
    out["weight_missing"] = w.isna().astype(float)
    out["weight_clean"] = w.fillna(weight_median)
    out["weight_capped"] = (out["weight_clean"] >= WEIGHT_CAP).astype(float)
    # Market index: fill missing with that day's mean; split into daily level + row deviation.
    out = out.join(market, on="date")
    out["market_index_clean"] = out["market_index"].fillna(out["mi_daily"])
    out["mi_dev"] = out["market_index_clean"] - out["mi_daily"]
    out["log_mi_daily"] = np.log(out["mi_daily"])
    out["log_distance"] = np.log(out["distance"])
    out["crow_miles"] = haversine(out.pickup_lat, out.pickup_lon, out.delivery_lat, out.delivery_lon)
    out["equipment_code"] = out["equipment"].map({"Dry Van": 0, "Reefer": 1, "Flatbed": 2}).astype(float)
    return add_calendar(out)


def city_design(names: pd.Series, lats: pd.Series, lons: pd.Series, known: dict, k: int = 4) -> np.ndarray:
    """One-hot city encoding over known cities. A city never seen in training gets
    inverse-distance weights over its k nearest known cities ("soft one-hot"), so a
    linear model predicts an interpolated geographic city effect for it."""
    known_names = list(known)
    k_lat = np.array([known[c][0] for c in known_names])
    k_lon = np.array([known[c][1] for c in known_names])
    index = {c: i for i, c in enumerate(known_names)}
    mat = np.zeros((len(names), len(known_names)))
    cache: dict[str, np.ndarray] = {}
    for row, (name, lat, lon) in enumerate(zip(names, lats, lons)):
        if name in index:
            mat[row, index[name]] = 1.0
            continue
        if name not in cache:
            dist = haversine(lat, lon, k_lat, k_lon)
            nearest = np.argsort(dist)[:k]
            wts = 1.0 / np.maximum(dist[nearest], 1.0)
            vec = np.zeros(len(known_names))
            vec[nearest] = wts / wts.sum()
            cache[name] = vec
        mat[row] = cache[name]
    return mat
