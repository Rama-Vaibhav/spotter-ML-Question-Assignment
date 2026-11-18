"""Run the full pipeline: data-quality report, time-forward validation, final fit,
validation_predictions.csv and the filled data/december_chart_inputs.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from freight_rate.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
