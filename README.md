# Sales Revenue Forecasting System 📈

A production-quality Sales Revenue Forecasting System predicting future sales using historical data, built on the Kaggle M5 Forecasting Accuracy dataset.

## 🚀 Overview

This project focuses on building a robust time-series forecasting pipeline to predict daily sales across multiple products and stores. We employ modular architecture, comparing multiple models (ARIMA, Prophet, XGBoost) and present the results in an interactive Streamlit dashboard.

## 🛠️ Tech Stack & Architecture
- **Language**: Python 3.10+
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Statsmodels (ARIMA/SARIMA), Prophet, XGBoost
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **Database**: MySQL (for data ingestion simulations)

### Architecture
1. **Data Ingestion**: Loading raw Kaggle datasets and simulated exports from MySQL.
2. **Preprocessing & Feature Engineering**: Missing value imputation, lag generation, rolling statistics, calendar variable integration.
3. **Modeling**: Hyperparameter tuning, cross-validation, and training multiple algorithms to compare errors.
4. **Dashboard**: An interactive UI to visualize forecasts and business metrics.

## 📁 Folder Structure

```
sales-forecasting/
├── data/                  # Git-ignored. Raw & processed CSV files
├── notebooks/             # EDA and experimental notebooks
├── src/                   # Core ML pipeline modules
│   ├── config.py
│   ├── data_ingestion/
│   ├── preprocessing/
│   ├── models/
│   └── evaluation/
├── tests/                 # Unit tests (pytest)
├── app/                   # Streamlit application logic
├── main.py                # Pipeline execution entry point
├── requirements.txt       # Dependencies
└── README.md
```

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd sales-forecasting
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Data Preparation**:
   - Download the Kaggle M5 dataset and place it inside `data/raw/`.
   - Optional: Configure `DB_HOST`, `DB_USER` in `.env` for MySQL connectivity.

## 🏃 Usage

**Run the end-to-end Data Pipeline (Preprocess -> Train -> Evaluate):**
```bash
python main.py --train
```

**Launch the Streamlit Dashboard:**
```bash
streamlit run app/main_app.py
```

## 📊 Methodology (Planned)

- **Handling large data**: Optimized memory usage via dtype downcasting.
- **Feature Engineering**: Heavy reliance on rolling windows (7, 28 days) and lag features to capture autocorrelation.
- **Models**:
  - `ARIMA`: Traditional statistical baseline.
  - `Prophet`: Handling multiple seasonalities and holidays.
  - `XGBoost`: Gradient boosting taking advantage of exogenous features.

## 📈 Evaluation Metrics

The system measures forecast accuracy primarily using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSSE** (Root Mean Squared Scaled Error - standard for the M5 competition)

## 👤 Author
- **[Your Name/Handle]**
- [LinkedIn Profile]
- [Portfolio Link]
