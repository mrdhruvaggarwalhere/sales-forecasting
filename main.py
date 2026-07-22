import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path to ensure absolute imports work properly
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
# Future imports when modules are implemented:
# from src.data_ingestion.data_loader import DataLoader
# from src.preprocessing.cleaner import DataCleaner
# from src.preprocessing.feature_engineering import FeatureEngineer
# from src.models.xgboost_model import XGBoostModel
# from src.evaluation.metrics import evaluate_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline(train_models=False):
    logging.info("Starting Sales Revenue Forecasting Pipeline...")
    
    # 1. Data Ingestion
    logging.info("Ingesting data...")
    # data = DataLoader.load_data(RAW_DATA_DIR / 'sales_train_validation.csv')
    
    # 2. Preprocessing & Feature Engineering
    logging.info("Preprocessing and engineering features...")
    # cleaned_data = DataCleaner.clean(data)
    # features = FeatureEngineer.create_features(cleaned_data)
    
    # 3. Model Training & Evaluation
    if train_models:
        logging.info("Training XGBoost Model...")
        # model = XGBoostModel()
        # model.train(features)
        # predictions = model.predict(test_features)
        
        logging.info("Evaluating Model...")
        # metrics = evaluate_model(actual, predictions)
        # logging.info(f"Model Metrics: {metrics}")
        
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML Pipeline")
    parser.add_argument("--train", action="store_true", help="Flag to train models")
    args = parser.parse_args()
    
    run_pipeline(train_models=args.train)
