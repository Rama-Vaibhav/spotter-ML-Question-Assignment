from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

TARGET = "posted_rate"
RANDOM_STATE = 42

# Rows whose log-rate is further than this from a robust baseline are treated as
# corrupted labels. On this data the residuals are cleanly bimodal: genuine rows sit
# within +/-0.2 and corrupted rows beyond +/-0.7, so anything in between is safe.
OUTLIER_LOG_THRESHOLD = 0.45

WEIGHT_CAP = 47_500.0
