"""Models. All models predict log(posted_rate); predictions are exp() of that.

* LaneMedianBaseline   - naive: median rate-per-mile of the lane (fallbacks below).
* StructuralModel      - ridge regression on interpretable, additive log-rate effects.
* GBMModel             - HistGradientBoosting on raw-ish features (pure tree model).
* HybridModel          - StructuralModel + gradient-boosted residual correction (final).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from .config import RANDOM_STATE
from .features import city_design

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]


class LaneMedianBaseline:
    name = "lane_median_baseline"

    def fit(self, df: pd.DataFrame, y_log: np.ndarray):
        rpm = np.exp(y_log) / df["distance"].to_numpy()
        tmp = df.assign(rpm=rpm, dband=pd.cut(df.distance, [0, 300, 600, 1000, 1500, 2200, 5000]))
        self.lane = tmp.groupby(["pickup", "delivery", "equipment"]).rpm.median()
        self.band = tmp.groupby(["equipment", "dband"], observed=True).rpm.median()
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        dband = pd.cut(df.distance, [0, 300, 600, 1000, 1500, 2200, 5000])
        lane_key = pd.MultiIndex.from_frame(df[["pickup", "delivery", "equipment"]])
        band_key = pd.MultiIndex.from_arrays([df.equipment, dband])
        rpm = pd.Series(self.lane.reindex(lane_key).to_numpy(), index=df.index)
        rpm = rpm.fillna(pd.Series(self.band.reindex(band_key).to_numpy(), index=df.index))
        return np.log(rpm.to_numpy() * df["distance"].to_numpy())


class StructuralModel:
    """Additive model of log(rate):

    distance curve + equipment + weight curve + pickup city + delivery city
    + market (daily level, row deviation, quote signal) + day-of-week
    + quarter base level + quarter-end ramp + quarter-start / month-end effects.

    Unseen quarters inherit the most recent quarter's level; unseen cities get a
    geographic interpolation of nearby city effects (see features.city_design).
    """

    name = "structural_ridge"
    RAMP_KNOTS = (7, 14, 21, 28, 35, 42, 56)
    START_KNOTS = (3, 7, 14)
    MONTH_END_KNOTS = (2, 5)

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    # -- design matrix ---------------------------------------------------------
    def _blocks(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        if fit:
            self.dist_spline = SplineTransformer(n_knots=10, degree=3, extrapolation="linear").fit(df[["log_distance"]])
            self.weight_spline = SplineTransformer(n_knots=6, degree=3, extrapolation="constant").fit(df[["weight_clean"]])
            self.qs_spline = SplineTransformer(n_knots=5, degree=3, extrapolation="constant").fit(df[["quote_signal"]])
            self.quarters = sorted(df["quarter_key"].unique())
            coords = pd.concat([
                df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["c", "lat", "lon"], axis=1),
                df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["c", "lat", "lon"], axis=1),
            ]).groupby("c")[["lat", "lon"]].median()
            self.known_cities = {c: (r.lat, r.lon) for c, r in coords.iterrows()}

        quarter = df["quarter_key"].where(df["quarter_key"].isin(self.quarters), self.quarters[-1])
        q_onehot = (quarter.to_numpy()[:, None] == np.array(self.quarters)[None, :]).astype(float)
        dtqe = df["days_to_quarter_end"].to_numpy()[:, None]
        dsqs = df["days_since_quarter_start"].to_numpy()[:, None]
        dtme = df["days_to_month_end"].to_numpy()[:, None]
        blocks = [
            self.dist_spline.transform(df[["log_distance"]]),
            (df["equipment"].to_numpy()[:, None] == np.array(EQUIPMENT)[None, :]).astype(float),
            self.weight_spline.transform(df[["weight_clean"]]),
            df[["weight_missing", "weight_capped"]].to_numpy(),
            city_design(df.pickup, df.pickup_lat, df.pickup_lon, self.known_cities),
            city_design(df.delivery, df.delivery_lat, df.delivery_lon, self.known_cities),
            df[["log_mi_daily", "mi_dev"]].to_numpy(),
            self.qs_spline.transform(df[["quote_signal"]]),
            (df["dow"].to_numpy()[:, None] == np.arange(7)[None, :]).astype(float),
            q_onehot,
            np.maximum(0, np.array(self.RAMP_KNOTS)[None, :] - dtqe) / np.array(self.RAMP_KNOTS)[None, :],
            np.maximum(0, np.array(self.START_KNOTS)[None, :] - dsqs) / np.array(self.START_KNOTS)[None, :],
            np.maximum(0, np.array(self.MONTH_END_KNOTS)[None, :] - dtme) / np.array(self.MONTH_END_KNOTS)[None, :],
        ]
        if fit:
            self._widths = [b.shape[1] for b in blocks]
        return np.hstack(blocks)

    def fit(self, df: pd.DataFrame, y_log: np.ndarray):
        X = self._blocks(df, fit=True)
        self.ridge = Ridge(alpha=self.alpha).fit(X, y_log)
        n_cities = len(self.known_cities)
        start = sum(self._widths[:4])  # blocks before the pickup-city block
        coef = self.ridge.coef_
        names = list(self.known_cities)
        self.pickup_effect = pd.Series(coef[start:start + n_cities], index=names)
        self.delivery_effect = pd.Series(coef[start + n_cities:start + 2 * n_cities], index=names)
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return self.ridge.predict(self._blocks(df, fit=False))

    def city_effects(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pu = city_design(df.pickup, df.pickup_lat, df.pickup_lon, self.known_cities) @ self.pickup_effect.to_numpy()
        de = city_design(df.delivery, df.delivery_lat, df.delivery_lon, self.known_cities) @ self.delivery_effect.to_numpy()
        return pu, de


TREE_FEATURES = [
    "log_distance", "equipment_code", "weight_clean", "weight_missing", "weight_capped",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "crow_miles",
    "log_mi_daily", "mi_dev", "quote_signal", "dow",
    "days_to_quarter_end", "days_since_quarter_start", "days_to_month_end",
]


def _hgb(**kw) -> HistGradientBoostingRegressor:
    params = dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE)
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


class GBMModel:
    """Pure gradient boosting on log(rate). A month index lets it learn level shifts,
    but trees can only carry the last level forward - they cannot add a new quarter's
    level to the quarter-end ramp, which is why the hybrid wins on time-forward folds."""

    name = "gbm_only"

    def fit(self, df, y_log):
        self.t0 = df["date"].min()
        X = self._X(df)
        self.model = _hgb(categorical_features=[TREE_FEATURES.index("equipment_code")]).fit(X, y_log)
        return self

    def _X(self, df):
        X = df[TREE_FEATURES].copy()
        X["month_index"] = (df["date"].dt.year - self.t0.year) * 12 + df["date"].dt.month - self.t0.month
        return X

    def predict_log(self, df):
        return self.model.predict(self._X(df))


class HybridModel:
    """Structural ridge model + a shallow gradient-boosted model of its residuals.

    The ridge part carries the level/time structure (extrapolates cleanly to new dates
    and new cities); the GBM adds non-linear interactions it cannot express. The GBM
    gets no absolute-time feature, so it cannot drift on future dates."""

    name = "hybrid_structural_plus_gbm"

    def __init__(self, alpha: float = 1.0, gbm_params: dict | None = None):
        self.structural = StructuralModel(alpha=alpha)
        self.gbm_params = gbm_params or dict(learning_rate=0.04, max_iter=400, max_depth=6,
                                             min_samples_leaf=80, l2_regularization=5.0)

    def _X(self, df):
        X = df[TREE_FEATURES].copy()
        X["pickup_effect"], X["delivery_effect"] = self.structural.city_effects(df)
        return X

    def fit(self, df, y_log):
        self.structural.fit(df, y_log)
        resid = y_log - self.structural.predict_log(df)
        self.gbm = _hgb(categorical_features=[TREE_FEATURES.index("equipment_code")], **self.gbm_params)
        self.gbm.fit(self._X(df), resid)
        return self

    def predict_log(self, df):
        return self.structural.predict_log(df) + self.gbm.predict(self._X(df))


MODEL_FACTORIES = {
    "lane_median_baseline": LaneMedianBaseline,
    "structural_ridge": StructuralModel,
    "gbm_only": GBMModel,
    "hybrid_structural_plus_gbm": HybridModel,
}
