"""End-to-end pipeline: load -> clean -> validate (time-forward) -> fit final -> predict."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from . import config as C
from .data import (complete_december_inputs, city_coordinates, daily_market, data_quality_report,
                   flag_rate_outliers, load_csv)
from .features import prepare
from .models import MODEL_FACTORIES, HybridModel
from .validation import COLD_CITY_FOLD, FOLDS, N_COLD_CITIES, metrics, split

FINAL_MODEL = "hybrid_structural_plus_gbm"


def load_all():
    train = load_csv(C.TRAIN_PATH)
    validation = load_csv(C.VALIDATION_PATH)
    december = pd.read_csv(C.DECEMBER_PATH)
    market = daily_market(train, validation)
    coords = city_coordinates(train, validation)
    return train, validation, december, market, coords


def fit_model(name: str, raw_train: pd.DataFrame, market: pd.DataFrame):
    """Fit `name` on raw labelled rows: outlier labels and feature stats come from these rows only."""
    outlier = flag_rate_outliers(raw_train)
    clean = raw_train[~outlier]
    weight_median = float(clean["weight"].abs().median())
    X = prepare(clean, market, weight_median)
    model = MODEL_FACTORIES[name]().fit(X, np.log(X["posted_rate"].to_numpy()))
    return model, weight_median, int(outlier.sum())


def predict(model, weight_median: float, df: pd.DataFrame, market: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict_log(prepare(df, market, weight_median)))


def run_validation(train: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every model on every time-forward fold (+ cold-start cities)."""
    is_outlier_all = flag_rate_outliers(train)  # used only to report metrics on clean labels
    rows, daily_frames = [], []
    for fold in FOLDS:
        fold_train, fold_test = split(train, fold)
        clean_mask = ~is_outlier_all.loc[fold_test.index].to_numpy()
        for name in MODEL_FACTORIES:
            model, wmed, n_out = fit_model(name, fold_train, market)
            pred = predict(model, wmed, fold_test, market)
            y = fold_test["posted_rate"].to_numpy()
            for subset, mask in (("clean_labels", clean_mask), ("all_labels", np.ones_like(clean_mask))):
                rows.append({"fold": fold["fold"], "description": fold["description"], "model": name,
                             "eval_rows": subset, "train_rows": len(fold_train),
                             "train_outliers_removed": n_out, **metrics(y[mask], pred[mask])})
            if name == FINAL_MODEL:
                daily_frames.append(pd.DataFrame({"fold": fold["fold"], "date": fold_test.date.to_numpy()[clean_mask],
                                                  "actual": y[clean_mask], "pred": pred[clean_mask]}))

    # Cold-start: remove N random cities from training entirely, score test rows that touch them.
    rng = np.random.default_rng(C.RANDOM_STATE)
    cities = sorted(set(train.pickup) | set(train.delivery))
    cold = set(rng.choice(cities, N_COLD_CITIES, replace=False))
    fold_train, fold_test = split(train, COLD_CITY_FOLD)
    fold_train = fold_train[~(fold_train.pickup.isin(cold) | fold_train.delivery.isin(cold))]
    touches = (fold_test.pickup.isin(cold) | fold_test.delivery.isin(cold)).to_numpy()
    clean_mask = ~is_outlier_all.loc[fold_test.index].to_numpy()
    for name in MODEL_FACTORIES:
        model, wmed, n_out = fit_model(name, fold_train, market)
        pred = predict(model, wmed, fold_test, market)
        y = fold_test["posted_rate"].to_numpy()
        for subset, mask in (("unseen_city_rows", touches & clean_mask), ("seen_city_rows", ~touches & clean_mask)):
            rows.append({"fold": "Cold-city", "description": f"Q3-analog with {N_COLD_CITIES} cities removed from training: "
                         + ", ".join(sorted(cold)), "model": name, "eval_rows": subset,
                         "train_rows": len(fold_train), "train_outliers_removed": n_out,
                         **metrics(y[mask], pred[mask])})
    return pd.DataFrame(rows), pd.concat(daily_frames)


def fit_final_and_predict(train, validation, december, market, coords):
    model, wmed, n_out = fit_model(FINAL_MODEL, train, market)
    val_pred = predict(model, wmed, validation, market)

    template = pd.read_csv(C.TEMPLATE_PATH)
    out = template[["load_id"]].merge(
        pd.DataFrame({"load_id": validation.load_id, "predicted_rate": np.round(val_pred, 2)}),
        on="load_id", how="left", validate="one_to_one")
    assert out.predicted_rate.notna().all() and len(out) == len(template)
    out.to_csv(C.PREDICTIONS_PATH, index=False)

    dec_inputs = complete_december_inputs(december, coords, market)
    december = december.copy()
    december["predicted_rate"] = np.round(predict(model, wmed, dec_inputs, market), 2)
    december.to_csv(C.DECEMBER_PATH, index=False)

    C.MODELS_DIR.mkdir(exist_ok=True)
    with open(C.MODELS_DIR / "final_model.pkl", "wb") as fh:
        pickle.dump({"model": model, "weight_median": wmed, "market": market}, fh)
    return model, out, december, n_out


def main():
    C.REPORTS_DIR.mkdir(exist_ok=True)
    train, validation, december, market, coords = load_all()

    quality = data_quality_report(train, validation)
    quality["rate_outliers_removed_from_training"] = int(flag_rate_outliers(train).sum())
    (C.REPORTS_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps(quality, indent=2))

    print("\nRunning time-forward validation ...")
    results, daily = run_validation(train, market)
    results.to_csv(C.REPORTS_DIR / "validation_metrics.csv", index=False)
    daily.to_csv(C.REPORTS_DIR / "validation_daily_predictions.csv", index=False)
    view = results[results.eval_rows.isin(["clean_labels", "unseen_city_rows"])]
    print(view.pivot_table(index=["fold", "eval_rows"], columns="model", values="MAPE_pct").round(2).to_string())

    print("\nFitting final model on all of Jan-Oct ...")
    model, preds, dec, n_out = fit_final_and_predict(train, validation, december, market, coords)
    print(f"Removed {n_out} corrupted labels. Wrote {C.PREDICTIONS_PATH} ({len(preds)} rows) and {C.DECEMBER_PATH}.")
    print(dec[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
