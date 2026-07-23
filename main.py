"""
Main Entry Point for the Sales Revenue Forecasting System.

Usage:
    python main.py             Launch the Streamlit dashboard
    python main.py --train     Train models (70/30 split) and evaluate, then launch dashboard
    python main.py --app       Launch dashboard only (alias for default)
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SalesForecastingMain")


def launch_dashboard():
    """Launches the Streamlit dashboard."""
    app_path = PROJECT_ROOT / "app" / "app.py"
    logger.info(f"Launching Streamlit Dashboard from {app_path}...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user.")
    except Exception as e:
        logger.error(f"Failed to launch dashboard: {e}")


def run_training():
    """Runs the training & evaluation pipeline."""
    logger.info("Starting training pipeline...")
    from src.models.train_and_evaluate import run_full_pipeline
    run_full_pipeline()
    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Revenue Forecasting System")
    parser.add_argument("--train", action="store_true", help="Train models (70/30 split) then launch dashboard")
    parser.add_argument("--app", action="store_true", help="Launch dashboard only")
    parser.add_argument("--train-only", action="store_true", help="Train models without launching dashboard")

    args = parser.parse_args()

    if args.train_only:
        run_training()
    elif args.train:
        run_training()
        launch_dashboard()
    else:
        # Default: just launch the dashboard
        launch_dashboard()
