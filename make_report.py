"""Build reports/Freight_Rate_Model_Report.docx from pipeline outputs.

Run after: train.py, score.py (creates scorer_results/candidate_december.png), eda.py."""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
CHART = ROOT / "scorer_results" / "candidate_december.png"
OUT = REPORTS / "Freight_Rate_Model_Report.docx"

MODEL_LABELS = {
    "lane_median_baseline": "Lane-median baseline",
    "gbm_only": "Gradient boosting only",
    "structural_ridge": "Structural ridge (log-additive)",
    "hybrid_structural_plus_gbm": "Hybrid: structural + GBM residual (final)",
}


def table(doc, header, rows, widths=None, bold_last_col_max=False):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(9)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(item, tuple):
            p.add_run(item[0]).bold = True
            p.add_run(item[1])
        else:
            p.add_run(item)


def figure(doc, path, caption, width=6.3):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.italic = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)


def main():
    q = json.loads((REPORTS / "data_quality.json").read_text())
    m = pd.read_csv(REPORTS / "validation_metrics.csv")
    dec = pd.read_csv(ROOT / "data" / "december_chart_inputs.csv")

    def get(fold, model, rows="clean_labels", col="MAPE_pct"):
        return m[(m.fold == fold) & (m.model == model) & (m.eval_rows == rows)][col].iloc[0]

    H = "hybrid_structural_plus_gbm"
    analog = [get("Q2-analog", H), get("Q3-analog", H)]

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9)
        s.top_margin = s.bottom_margin = Inches(0.8)

    doc.add_heading("Freight Rate Prediction — Validation & Modeling Report", 0)
    doc.add_paragraph("Machine Learning Engineer assessment · Spotter")

    # 1. Summary ------------------------------------------------------------------
    doc.add_heading("1. Summary", 1)
    bullets(doc, [
        ("Task. ", f"Predict posted_rate for {q['validation_rows']:,} loads dated "
                   f"{q['validation_date_range'][0]} to {q['validation_date_range'][1]} using "
                   f"{q['train_rows']:,} labelled loads from {q['train_date_range'][0]} to {q['train_date_range'][1]}. "
                   "This is a forecast into a later time period, so validation is strictly time-forward."),
        ("Final model. ", "A log-additive structural ridge regression (distance curve, equipment, weight, "
                          "pickup/delivery city effects, market signals, quarterly level and quarter-end ramp) "
                          "plus a gradient-boosted model of its residuals."),
        ("Validation result. ", f"On the two 'quarter-analog' backtests that replicate the real Oct → Nov/Dec setup, "
                                f"MAPE is {analog[0]:.2f}% and {analog[1]:.2f}% (MAE ≈ ${get('Q3-analog', H, col='MAE'):.0f}), "
                                f"versus {get('Q2-analog', 'lane_median_baseline'):.1f}% / "
                                f"{get('Q3-analog', 'lane_median_baseline'):.1f}% for a lane-median baseline. "
                                f"Rows involving cities never seen in training score {get('Cold-city', H, 'unseen_city_rows'):.2f}%."),
        ("Data quality. ", f"{q['rate_outliers_removed_from_training']} corrupted rate labels (~1.4%), "
                           f"{q['negative_weight_train']} sign-flipped weights, {q['missing_weight_train']} missing weights, "
                           f"a 47,500 lb weight cap, {q['missing_market_index_train']} missing market_index values and "
                           f"{len(q['unseen_cities_in_validation'])} validation cities absent from training — all handled explicitly."),
    ])

    # 2. Key findings ------------------------------------------------------------------
    doc.add_heading("2. Key findings from exploring the data", 1)
    bullets(doc, [
        ("Rates are multiplicative. ", "log(rate) is close to additive in its drivers, so all models work in log space "
                                       "(this also makes errors relative, which suits rates ranging from ~$200 to ~$7,000). "
                                       "Rate per mile falls steeply with distance; Reefer > Flatbed > Dry Van."),
        ("A clean signal. ", "With distance, equipment, weight and city effects, genuine labels have only ~3% residual "
                             "noise — so the time-level structure below dominates achievable accuracy."),
        ("Quarterly structure. ", "Each quarter has its own base level (rising through the year) and rates ramp up "
                                  "~4–5% over the last ~5 weeks of every quarter, resetting on the 1st (Figure 3). "
                                  "Nov–Dec are the end of Q4, so the forecast must include that ramp into Dec 31."),
        ("market_index. ", "A daily market level (strong 7-day cycle on a seasonal swell) plus row-level noise "
                           "(within-day std ≈ 0.025). Its daily mean is the useful signal. Nov–Dec values sit "
                           "at the low end of the training range (Figure 2)."),
        ("Geography. ", "Pickup and delivery city effects (±3%) are stable over time (early-vs-late correlation 0.94) "
                        "and smooth in space: South/Texas higher, Great Lakes/Northeast lower (Figure 4). Lane-specific "
                        "effects beyond the two city effects are mostly noise."),
        ("quote_signal ", "carries a weak, non-linear signal (top decile ≈ +1.5%); distance in the data is "
                          "consistent within each lane, but coordinates are jittered, so distance — not haversine — is used."),
    ])
    figure(doc, FIG / "01_rate_per_mile_vs_distance.png", "Figure 1. Rate per mile vs distance by equipment; corrupted labels marked ×.")
    figure(doc, FIG / "03_market_index.png", "Figure 2. Daily mean market_index in training (Jan–Oct) and prediction period (Nov–Dec).")
    figure(doc, FIG / "04_quarter_end_effect.png", "Figure 3. Daily rate level after removing lane, equipment, weight, distance and market effects.")
    figure(doc, FIG / "05_city_effects.png", "Figure 4. City effects on rate; diamonds are validation-only cities (predicted by geographic interpolation).")

    # 3. Data quality ------------------------------------------------------------------
    doc.add_heading("3. Data-quality issues and how they were addressed", 1)
    table(doc, ["Issue", "Evidence", "Treatment"], [
        ["Corrupted rate labels", f"{q['rate_outliers_removed_from_training']} rows (1.4%) off by 2–6× "
         f"(rate/mile {q['rate_per_mile_min']:.2f}–{q['rate_per_mile_max']:.1f}); residuals are cleanly bimodal (Fig. 5)",
         "Flag with an iteratively re-fit robust baseline (|log residual| > 0.45) and drop from training. Fitted on the "
         "training window only inside each backtest."],
        ["Negative weights", f"{q['negative_weight_train']} train / {q['negative_weight_validation']} validation; "
         "their rates match |weight|", "Use absolute value (sign error)."],
        ["Missing weights", f"{q['missing_weight_train']} train / {q['missing_weight_validation']} validation",
         "Impute training median + missing-indicator feature."],
        ["Weight cap", f"{q['weight_at_cap_train']} rows exactly 47,500 lb", "Keep value, add a capped-indicator feature."],
        ["Missing market_index", f"{q['missing_market_index_train']} train / {q['missing_market_index_validation']} validation",
         "Fill with that day's mean market_index (feature data only, no labels)."],
        ["Unseen cities", f"{len(q['unseen_cities_in_validation'])} cities ({', '.join(q['unseen_cities_in_validation'])}) "
         f"in {q['validation_rows_with_unseen_city']:,} validation rows", "Inverse-distance interpolation of the 4 nearest known "
         "cities' effects; trees use coordinates. Verified with a cold-city backtest."],
        ["December chart inputs", "File lacks lat/lon, market_index, quote_signal", "Coordinates from the city lookup; "
         "market_index and quote_signal = that day's mean across validation.csv rows."],
    ], widths=[1.3, 2.6, 2.8])
    figure(doc, FIG / "02_outlier_residuals.png", "Figure 5. Residuals from the robust baseline: genuine labels within ±0.2, corrupted beyond ±0.7.")
    figure(doc, FIG / "06_weight_distribution.png", "Figure 6. Reported weight: sign errors, a reporting cap and missing values.")

    # 4. Split & validation ------------------------------------------------------------------
    doc.add_heading("4. Train/test split and validation approach", 1)
    doc.add_paragraph(
        "The labelled data covers Jan 1 – Oct 31, 2025 and the loads to predict are Nov 1 – Dec 31, 2025: the final "
        "task is a forecast of a later period. A random K-fold split would place future market conditions and the "
        "same days' quarter-end effects in training, and would materially overstate accuracy. Every split is "
        "therefore forward in time: the model is trained only on loads dated before the test window.")
    doc.add_paragraph(
        "Because rates follow a quarterly pattern, the split was designed to replicate the real task as closely "
        "as possible. In the real task, training ends after the first month of Q4 (October) and the model must "
        "predict the remaining two months, including the quarter-end ramp. The 'quarter-analog' folds do exactly "
        "this for earlier quarters:")
    table(doc, ["Fold", "Train window", "Test window", "Purpose"], [
        ["Q2-analog", "Jan 1 – Apr 30", "May 1 – Jun 30", "Replica of the task (primary)"],
        ["Q3-analog", "Jan 1 – Jul 31", "Aug 1 – Sep 30", "Replica of the task (primary)"],
        ["Last-2-months", "Jan 1 – Aug 31", "Sep 1 – Oct 31", "Stress test: test quarter (Q4) has no training data"],
        ["Cold-city", "Q3-analog, 8 random cities removed", "Aug 1 – Sep 30", "Unseen-city robustness (12% of final rows)"],
        ["Final fit", "Jan 1 – Oct 31 (all labels)", "Nov 1 – Dec 31", "Submission"],
    ], widths=[1.1, 1.8, 1.3, 2.5])
    bullets(doc, [
        ("Leakage controls. ", "Outlier detection, imputation statistics, splines and city effects are fitted on each "
                               "fold's training window only. Daily market means are computed from feature columns, which "
                               "are available at prediction time, never from labels."),
        ("Metrics. ", "MAPE (primary — model trained on log rate), MAE, RMSE, median APE and bias, reported on "
                      "clean test labels. Metrics on all labels (including corrupted ones) are also recorded in "
                      "reports/validation_metrics.csv. Those are dominated by the ~1.4% corrupted labels that no "
                      "model can predict."),
        ("Model selection. ", "Choices were made on the quarter-analog folds. Ridge strength and GBM capacity were "
                              "checked over a small grid, and differences were within ±0.1 MAPE points, so defaults were kept "
                              "to avoid overfitting the backtests. A linear time-trend term was tested and did not help."),
    ])

    # 5. Results ------------------------------------------------------------------
    doc.add_heading("5. Model comparison", 1)
    rows = []
    for key, label in MODEL_LABELS.items():
        rows.append([label] + [f"{get(f, key, r):.2f}%" for f, r in [
            ("Q2-analog", "clean_labels"), ("Q3-analog", "clean_labels"),
            ("Last-2-months", "clean_labels"), ("Cold-city", "unseen_city_rows")]])
    table(doc, ["Model (MAPE)", "Q2-analog", "Q3-analog", "Last-2-months", "Cold-city (unseen)"], rows,
          widths=[2.6, 1.0, 1.0, 1.1, 1.1])
    hyb = m[(m.model == H) & m.eval_rows.isin(["clean_labels", "unseen_city_rows"])]
    table(doc, ["Final model fold", "Rows", "MAE ($)", "RMSE ($)", "MAPE", "Median APE", "Bias"],
          [[r.fold, f"{r.n:,}", f"{r.MAE:.1f}", f"{r.RMSE:.1f}", f"{r.MAPE_pct:.2f}%", f"{r.MedAPE_pct:.2f}%",
            f"{r.bias_pct:+.2f}%"] for r in hyb.itertuples()])
    figure(doc, FIG / "07_validation_mape.png", "Figure 7. MAPE by model and backtest fold.")
    figure(doc, FIG / "08_q3_fold_daily.png", "Figure 8. Q3-analog fold: daily mean actual vs predicted rate.")
    doc.add_paragraph(
        "Why the hybrid model: the structural part encodes the time structure additively (quarter base level + "
        "quarter-end ramp), so it extrapolates correctly to dates later than the training data: Q4's base is learned "
        "from October and the ramp shape from March, June and September. A pure tree model can only carry the last "
        "level forward and cannot add a ramp on top of a new quarter's base, which is why it loses on the "
        "quarter-analog folds. The GBM residual stage adds non-linear interactions (e.g. weight × distance) and "
        "has no absolute-time feature, so it cannot drift on future dates. The Last-2-months fold is the one place "
        "GBM-only is better: there the test quarter (Q4) is completely unseen, which is not the situation for the "
        "final forecast because October is in training.")

    # 6. Features ------------------------------------------------------------------
    doc.add_heading("6. Final model and features", 1)
    bullets(doc, [
        ("Target: ", "log(posted_rate); prediction = exp(model output)."),
        ("Structural ridge (α = 1): ", "cubic spline of log distance; equipment; spline of cleaned weight + missing/cap "
                                       "flags; pickup and delivery city effects (soft one-hot for unseen cities); log daily "
                                       "market_index and row deviation; spline of quote_signal; day of week; quarter base "
                                       "level (unseen quarter → most recent level); hinge ramps on days-to-quarter-end "
                                       "(7–56 days), days-since-quarter-start and days-to-month-end."),
        ("GBM residual model: ", "HistGradientBoosting (depth 6, 400 trees, lr 0.04, min leaf 80, L2 5) on distance, "
                                 "equipment, weight + flags, coordinates, crow-fly miles, market features, calendar "
                                 "position (no absolute date) and the learned city effects."),
        ("Fitting: ", "on all Jan–Oct loads after removing corrupted labels; model saved to models/final_model.pkl."),
    ])

    # 7. December chart ------------------------------------------------------------------
    doc.add_heading("7. Fixed December prediction chart (score.py)", 1)
    figure(doc, CHART, "Figure 9. scorer_results/candidate_december.png produced by the provided score.py.", width=6.5)
    lo, hi = dec.predicted_rate.min(), dec.predicted_rate.max()
    first, last = dec.predicted_rate.iloc[0], dec.predicted_rate.iloc[-1]
    doc.add_paragraph(
        f"Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb. Predictions range ${lo:,.0f}–${hi:,.0f} "
        f"(${first:,.0f} on Dec 1 to ${last:,.0f} on Dec 31, +{100 * (last / first - 1):.1f}%). Two effects drive the shape. "
        "First, a weekly oscillation that follows the 7-day cycle in December's market_index. Second, an upward "
        "drift from the Q4/year-end ramp, which peaks on Dec 31. The level is consistent with historical Dry Van "
        "loads on this lane ($790–$870 for ~360 miles). Because the chart file has no market columns, each day's "
        "market_index and quote_signal were set to that day's average across validation.csv.")

    # 8. Limitations ------------------------------------------------------------------
    doc.add_heading("8. Limitations and next steps", 1)
    bullets(doc, [
        "Only one prior example of each quarter-end exists in the training data, and none of a year-end. A "
        "year-end or holiday effect (e.g. Christmas capacity) beyond the regular quarter-end ramp cannot be "
        "learned from this data and is the main forecast risk.",
        "Q4's base level is estimated from a single month (October). With more history, a seasonal "
        "(year-over-year) term would replace the per-quarter level.",
        "The final metric Spotter uses is unknown; the model is tuned for relative error. A mean-optimal "
        "correction (exp of the log-residual variance) would shift predictions by under 0.1%.",
        "Next: add prediction intervals (quantile GBM on residuals), monitor for drift in market_index, and "
        "retrain weekly as new labelled loads arrive.",
    ])

    doc.add_heading("Appendix: reproduce", 1)
    for line in ["python -m pip install -r requirements.txt", "python train.py",
                 "python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv",
                 "python eda.py", "python make_report.py"]:
        p = doc.add_paragraph(line)
        p.runs[0].font.name = "Menlo"
        p.runs[0].font.size = Pt(9)

    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
