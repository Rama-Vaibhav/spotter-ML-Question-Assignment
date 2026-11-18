"""Exploratory analysis figures used in the report (writes reports/figures/*.png).

Run after train.py (the validation figures read reports/validation_*.csv)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

from freight_rate import config as C  # noqa: E402
from freight_rate.data import flag_rate_outliers  # noqa: E402
from freight_rate.features import prepare  # noqa: E402
from freight_rate.pipeline import load_all  # noqa: E402

# Validated reference palette (categorical slots in fixed order) + recessive ink.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
DIVERGING = LinearSegmentedColormap.from_list("div", [BLUE, "#d9d8d3", ORANGE])

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 170, "font.size": 9.5, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def residuals(train, market, with_cities=True):
    """Residual of log(rate) after distance/equipment/weight/market (and optionally city)
    effects, with no calendar terms. Its daily mean exposes the calendar structure;
    without city terms its per-city mean is the city effect."""
    clean = train[~flag_rate_outliers(train)]
    X = prepare(clean, market, float(clean.weight.abs().median()))
    ld = X.log_distance.to_numpy()
    num = np.column_stack([ld, ld**2, ld**3, X.weight_clean / 1e4, X.log_mi_daily, X.mi_dev])
    cat_cols = ["equipment", "pickup", "delivery"] if with_cities else ["equipment"]
    cats = OneHotEncoder(sparse_output=False).fit_transform(X[cat_cols])
    design = np.hstack([num, cats])
    y = np.log(X.posted_rate.to_numpy())
    return X.assign(resid=y - Ridge(alpha=1e-3).fit(design, y).predict(design))


def fig_rate_vs_distance(train, outlier):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    rpm = train.posted_rate / train.distance
    sample = train[~outlier].sample(6000, random_state=0)
    for eq, col in (("Dry Van", BLUE), ("Reefer", ORANGE), ("Flatbed", AQUA)):
        s = sample[sample.equipment == eq]
        ax.scatter(s.distance, s.posted_rate / s.distance, s=4, alpha=0.35, color=col, label=eq, linewidths=0)
    ax.scatter(train.distance[outlier], rpm[outlier], s=10, color=INK, marker="x", linewidths=0.8,
               label=f"Corrupted label ({outlier.sum()} rows)")
    ax.set_yscale("log")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Rate per mile ($, log scale)")
    ax.set_title("Rate per mile falls with distance; ~1.4% of labels are off by 2-6x")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, markerscale=3, fontsize=8.5)
    for h in leg.legend_handles:
        h.set_alpha(1)
    save(fig, "01_rate_per_mile_vs_distance.png")


def fig_outlier_residuals(train):
    w = train.weight.abs().fillna(train.weight.abs().median())
    ld = np.log(train.distance.to_numpy())
    cats = OneHotEncoder(sparse_output=False).fit_transform(train[["equipment", "pickup", "delivery"]])
    X = np.hstack([np.column_stack([ld, ld**2, ld**3, w / 1e4]), cats])
    y = np.log(train.posted_rate.to_numpy())
    keep = np.ones(len(y), bool)
    for _ in range(4):
        r = y - Ridge(alpha=1e-3).fit(X[keep], y[keep]).predict(X)
        keep = np.abs(r) < C.OUTLIER_LOG_THRESHOLD
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.hist(r, bins=np.arange(-2.0, 2.0, 0.04), color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    for t in (-C.OUTLIER_LOG_THRESHOLD, C.OUTLIER_LOG_THRESHOLD):
        ax.axvline(t, color=INK2, linestyle="--", linewidth=1)
    ax.text(C.OUTLIER_LOG_THRESHOLD + 0.05, 3000, f"removal threshold\n|residual| > {C.OUTLIER_LOG_THRESHOLD}",
            color=INK2, fontsize=8.5)
    ax.set_xlabel("log(posted rate) minus robust baseline")
    ax.set_ylabel("Loads (log scale)")
    ax.set_title("Corrupted labels separate cleanly from genuine rates")
    save(fig, "02_outlier_residuals.png")


def fig_market(train, validation):
    daily_tr = train.groupby("date").market_index.mean()
    daily_va = validation.groupby("date").market_index.mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(daily_tr.index, daily_tr, color=BLUE, linewidth=1.4, label="Training (Jan-Oct)")
    ax.plot(daily_va.index, daily_va, color=ORANGE, linewidth=1.4, label="Prediction period (Nov-Dec)")
    ax.set_ylabel("Daily mean market_index")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.set_title("market_index: a 7-day cycle on a seasonal swell; Nov-Dec sits at the low end")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "03_market_index.png")


def fig_time_structure(res):
    daily = res.groupby("date").resid.mean()
    weekly = daily.rolling(7, center=True, min_periods=4).mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for qe in ("2025-03-31", "2025-06-30", "2025-09-30"):
        end = pd.Timestamp(qe)
        ax.axvspan(end - pd.Timedelta(days=34), end + pd.Timedelta(hours=23), color=YELLOW, alpha=0.18, linewidth=0)
    ax.plot(daily.index, 100 * daily, color=MUTED, linewidth=0.7, label="Daily mean")
    ax.plot(weekly.index, 100 * weekly, color=BLUE, linewidth=2, label="7-day average")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylabel("Rate level vs. average (%)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.set_title("Each quarter: a higher base level plus a ~4-5% ramp into quarter end")
    ax.text(pd.Timestamp("2025-07-10"), ax.get_ylim()[0] + 0.4, "shaded = last 5 weeks of quarter", color=INK2, fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "04_quarter_end_effect.png")


def fig_city_effects(res, coords, unseen):
    eff = (res.groupby("pickup").resid.mean() + res.groupby("delivery").resid.mean()) / 2 * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    known = coords.loc[eff.index]
    lim = np.abs(eff).max()
    sc = ax.scatter(known.lon, known.lat, c=eff, cmap=DIVERGING, vmin=-lim, vmax=lim, s=60,
                    edgecolors="white", linewidths=1)
    new = coords.loc[unseen]
    ax.scatter(new.lon, new.lat, s=70, facecolors="none", edgecolors=INK, linewidths=1.2, marker="D",
               label="Unseen in training (validation only)")
    for c in unseen:
        ax.annotate(c, (coords.loc[c].lon, coords.loc[c].lat), xytext=(5, 3), textcoords="offset points",
                    fontsize=7.5, color=INK2)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("City effect on rate (%)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_xlabel("Longitude (as given)")
    ax.set_ylabel("Latitude (as given)")
    ax.set_title("City effects are geographic: South/Texas higher, Great Lakes/Northeast lower")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "05_city_effects.png")


def fig_weight(train):
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = train.weight.dropna()
    ax.hist(w, bins=np.arange(-48_000, 49_000, 1000), color=BLUE, edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Reported weight (lb)")
    ax.set_ylabel("Loads (log scale)")
    ax.annotate(f"{(train.weight < 0).sum()} negative (sign errors)", (-30_000, 30), xytext=(-46_000, 400),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.annotate(f"{(train.weight.abs() == 47_500).sum()} at 47,500 cap", (47_500, 1100), xytext=(20_000, 3000),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_title(f"Weight: sign errors, a reporting cap, and {train.weight.isna().sum()} missing values")
    save(fig, "06_weight_distribution.png")


def fig_validation():
    m = pd.read_csv(C.REPORTS_DIR / "validation_metrics.csv")
    m = m[m.eval_rows.isin(["clean_labels", "unseen_city_rows"])]
    order = ["Q2-analog", "Q3-analog", "Last-2-months", "Cold-city"]
    models = [("lane_median_baseline", "Lane-median baseline", YELLOW), ("gbm_only", "GBM only", ORANGE),
              ("structural_ridge", "Structural ridge", AQUA), ("hybrid_structural_plus_gbm", "Hybrid (final)", BLUE)]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    width = 0.2
    for i, (key, label, col) in enumerate(models):
        vals = [m[(m.fold == f) & (m.model == key)].MAPE_pct.iloc[0] for f in order]
        x = np.arange(len(order)) + (i - 1.5) * width
        ax.bar(x, vals, width=width - 0.02, color=col, label=label)
        if key == "hybrid_structural_plus_gbm":
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(np.arange(len(order)), ["Q2-analog\nJan-Apr -> May-Jun", "Q3-analog\nJan-Jul -> Aug-Sep",
                                          "Last 2 months\nJan-Aug -> Sep-Oct", "Cold-city\n(unseen-city rows)"])
    ax.set_ylabel("MAPE (%) on test window")
    ax.grid(axis="x", visible=False)
    ax.set_title("Time-forward backtests: hybrid model wins on the folds that mirror the task")
    ax.legend(ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, max(7, ax.get_ylim()[1]))
    save(fig, "07_validation_mape.png")

    d = pd.read_csv(C.REPORTS_DIR / "validation_daily_predictions.csv", parse_dates=["date"])
    d = d[d.fold == "Q3-analog"].groupby("date")[["actual", "pred"]].mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(d.index, d.actual, color=MUTED, linewidth=1.6, label="Actual (daily mean)")
    ax.plot(d.index, d.pred, color=BLUE, linewidth=2, label="Predicted (daily mean)")
    ax.set_ylabel("Mean rate ($)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.set_title("Q3-analog fold: forecast tracks daily level incl. September quarter-end ramp")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "08_q3_fold_daily.png")


def main():
    train, validation, _, market, coords = load_all()
    outlier = flag_rate_outliers(train)
    unseen = sorted((set(validation.pickup) | set(validation.delivery)) - (set(train.pickup) | set(train.delivery)))
    res = residuals(train, market, with_cities=True)
    res_no_city = residuals(train, market, with_cities=False)
    fig_rate_vs_distance(train, outlier)
    fig_outlier_residuals(train)
    fig_market(train, validation)
    fig_time_structure(res)
    fig_city_effects(res_no_city, coords, unseen)
    fig_weight(train)
    fig_validation()
    print(f"Figures written to {C.FIGURES_DIR}")


if __name__ == "__main__":
    main()
