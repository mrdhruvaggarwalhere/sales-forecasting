# 📈 Sales Revenue Forecasting System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Prophet-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-red" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

## 🚀 Project Overview

The **Sales Revenue Forecasting System** is a production-quality time-series forecasting pipeline designed to predict future daily sales across multiple products and retail stores. Built on top of the robust [Kaggle M5 Forecasting Accuracy dataset](https://www.kaggle.com/c/m5-forecasting-accuracy), this project leverages advanced machine learning techniques to help retail businesses optimize inventory management, reduce waste, and maximize revenue.

By combining an extensible machine learning pipeline with an interactive Streamlit dashboard, stakeholders can visualize predictions and gain actionable insights into business performance.

---

## ✨ Features

- **End-to-end ML Pipeline**: Modular architecture for data ingestion, preprocessing, feature engineering, and model training.
- **Advanced Feature Engineering**: Automatic generation of rolling statistics, time-based lag features, and calendar variables to capture seasonality and trends.
- **Multiple Forecasting Models**: Comprehensive comparison across traditional statistical models (ARIMA) and advanced ML algorithms (Prophet, XGBoost).
- **Interactive Dashboard**: A sleek, user-friendly Streamlit web interface for visualizing historical sales data, future forecasts, and performance metrics.
- **Scalable Design**: Optimized memory usage and well-structured configuration for handling large-scale retail datasets.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, Prophet, Statsmodels (ARIMA/SARIMA)
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Web Application:** Streamlit
- **Testing:** Pytest

---

## 📁 Folder Structure

```text
sales-forecasting/
├── data/                  # Ignored by Git. Place raw datasets here
│   ├── raw/               # Raw CSV files
│   └── processed/         # Processed features and engineered data
├── notebooks/             # Jupyter notebooks for EDA and experiments
├── src/                   # Core ML pipeline modules
│   ├── config.py          # Centralized configuration and parameters
│   ├── data_ingestion/    # Data loaders and database connectors
│   ├── preprocessing/     # Data cleaning and feature engineering
│   ├── models/            # Model training, evaluation, and utilities
│   └── evaluation/        # Metrics tracking (RMSE, MAE, RMSSE)
├── tests/                 # Unit tests for pipeline components
├── app/                   # Streamlit dashboard application
│   ├── app.py             # Dashboard entry point
│   ├── components/        # Reusable UI components
│   └── styles.css         # Custom UI styling
├── main.py                # Command-line entry point for pipeline execution
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 📊 Dataset

This project utilizes the **M5 Forecasting Dataset** from Kaggle, which includes hierarchical sales data from Walmart. 
- **Timeframe:** 5+ years of historical daily sales data.
- **Scope:** 30,000+ products across 10 stores and 3 states.
- **Information:** Includes unit sales, calendar events (holidays), and sell prices.

*Note: Ensure you download the dataset and place the CSV files inside the `data/raw/` directory before running the pipeline.*

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/sales-forecasting.git
   cd sales-forecasting
   ```

2. **Set up a virtual environment:**
   ```bash
   # On macOS and Linux
   python -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables (Optional):**
   Copy the `.env.example` file to `.env` and update your database or environment configurations if simulating MySQL data ingestion.

---

## 🏃 Usage

### 1. Run the ML Pipeline
Execute the end-to-end data processing and modeling pipeline from the root directory:
```bash
# To run preprocessing and optionally train the models
python main.py --train
```

### 2. Launch the Dashboard
Start the Streamlit application to visualize the sales predictions:
```bash
streamlit run app/app.py
```
Navigate to `http://localhost:8501` in your web browser.

---

## 🖼️ Dashboard Screenshots

> **[Placeholder]** Add screenshots of your Streamlit dashboard here to showcase the UI to visitors.

<details>
<summary>Click to view screenshots</summary>

![Dashboard Home](https://via.placeholder.com/800x400.png?text=Dashboard+Home+Screen)
*Overview of historical sales and top-level metrics.*

![Forecast View](https://via.placeholder.com/800x400.png?text=Forecast+View+Screen)
*Detailed 28-day revenue forecast and trend analysis.*

</details>

---

## 📈 Model Performance

The forecasting models are primarily evaluated using the following metrics to ensure reliability against the M5 standard:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSSE** (Root Mean Squared Scaled Error)

*Currently, **XGBoost** utilizing rolling 7/28 day window features yields the most optimal balance between training speed and RMSSE accuracy.*

---

## 🔮 Future Scope

- [ ] **Automated Hyperparameter Tuning:** Integrate `Optuna` for advanced model tuning.
- [ ] **Deep Learning Models:** Experiment with LSTM or Temporal Fusion Transformers (TFT) for capturing complex temporal patterns.
- [ ] **Cloud Deployment:** Containerize the application using Docker and deploy the pipeline/dashboard to AWS/GCP.
- [ ] **Real-time Ingestion:** Integrate Kafka for streaming data ingestion updates.

---

## 🤝 Contributors

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/mrdhruvaggarwal/sales-forecasting/issues).

- **Your Name** - *Initial work* - [Dhruv Aggarwal](https://github.com/mrdhruvaggarwalhere)
- [LinkedIn Profile](https://linkedin.com/in/mrdhruvaggarwal)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
