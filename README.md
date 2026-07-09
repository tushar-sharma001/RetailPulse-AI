# 📈 RetailPulse AI

### AI-Powered Sales Forecasting & Demand Intelligence Platform

RetailPulse AI is an end-to-end retail analytics platform that combines machine learning, time series forecasting, anomaly detection, and demand segmentation to help businesses make smarter inventory and sales planning decisions.

Built using Python, Streamlit, XGBoost, Prophet, SARIMA, and Scikit-learn, the project provides an interactive dashboard for forecasting demand, detecting unusual sales patterns, and analyzing product performance.

---

## 📌 Problem Statement

Retail and e-commerce businesses often struggle to answer one critical question:

> **How much inventory should we stock over the next few months?**

Incorrect demand estimation can result in:

- Overstocking and increased inventory costs
- Stock shortages and lost revenue
- Inefficient supply chain planning
- Poor customer experience

RetailPulse AI addresses these challenges by providing accurate demand forecasts, anomaly detection, and product demand segmentation through an interactive analytics dashboard.

---

# 🚀 Features

### 📊 Executive Dashboard

- Business KPI cards
- Monthly sales trend
- Revenue by region
- Revenue by category
- Top-selling products
- Interactive filters

---

### 📈 Sales Forecasting

Implemented and compared three forecasting models:

- SARIMA
- Facebook Prophet
- XGBoost Regressor

Forecast evaluation using:

- MAE
- RMSE
- MAPE

Generate forecasts for the next three months.

---

### 🚨 Anomaly Detection

Detect unusual sales spikes and drops using:

- Isolation Forest
- Z-Score Analysis

Visualize anomalies on weekly sales trends.

---

### 📦 Product Demand Segmentation

Segment products using K-Means Clustering based on:

- Sales Volume
- Growth Rate
- Sales Volatility
- Average Order Value

Visualize demand clusters using PCA.

---

### 🤖 Model Performance

Compare forecasting models and automatically recommend the best-performing model for production deployment.

---

# 🛠️ Tech Stack

| Category | Tools |
|-----------|-------|
| Programming | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Forecasting | SARIMA, Prophet, XGBoost |
| Machine Learning | Scikit-learn |
| Deployment | Streamlit Community Cloud |
| Version Control | Git & GitHub |

---

# 📂 Project Structure

```text
RetailPulse-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── summary.pdf
├── .gitignore
│
├── assets/
├── data/
├── models/
├── notebooks/
├── outputs/
├── reports/
└── screenshots/
```

---

# 📈 Dashboard Preview

## Executive Dashboard

> Add a screenshot here

```
screenshots/dashboard.png
```

---

## Forecast Explorer

> Add a screenshot here

```
screenshots/forecast.png
```

---

## Anomaly Detection

> Add a screenshot here

```
screenshots/anomaly.png
```

---

## Demand Segmentation

> Add a screenshot here

```
screenshots/segmentation.png
```

---

## Model Performance

> Add a screenshot here

```
screenshots/model_performance.png
```

---

# 📊 Machine Learning Pipeline

```text
Retail Sales Dataset
          │
          ▼
Data Cleaning & Feature Engineering
          │
          ▼
Exploratory Data Analysis
          │
          ▼
Time Series Analysis
          │
          ▼
Forecasting Models
 ├── SARIMA
 ├── Prophet
 └── XGBoost
          │
          ▼
Model Evaluation
          │
          ▼
Anomaly Detection
          │
          ▼
Demand Segmentation
          │
          ▼
Interactive Streamlit Dashboard
```

---

# 📌 Key Business Insights

- Technology is the highest revenue-generating category.
- West region demonstrates consistent sales performance.
- November and December show recurring seasonal demand peaks.
- XGBoost achieved the lowest forecasting error among the evaluated models.
- Isolation Forest effectively identified abnormal sales spikes.
- Product segmentation supports inventory optimization and demand planning.

---

# 📈 Model Comparison

| Model | MAE | RMSE | MAPE |
|------|------:|------:|------:|
| SARIMA | 13455.42 | 15938.99 | 22.02% |
| Prophet | 10128.56 | 14561.39 | 14.33% |
| XGBoost | **Best** | **Best** | **Best** |

> XGBoost was selected as the production model based on its superior forecasting performance.

---

# ⚡ Installation

Clone the repository.

```bash
git clone https://github.com/tushar-sharma001/RetailPulse-AI.git
```

Move into the project directory.

```bash
cd RetailPulse-AI
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run the Streamlit application.

```bash
streamlit run app.py
```

---

# 📄 Dataset

This project uses the **Sample Superstore Sales Dataset**, which contains retail transaction data including:

- Orders
- Products
- Categories
- Regions
- Customers
- Sales
- Profit
- Shipping information

---

# 🎯 Future Improvements

- LSTM and Transformer-based forecasting
- Real-time dashboard with live sales feeds
- Inventory optimization engine
- Automated alert system for anomalies
- Cloud deployment using Docker and Azure/AWS
- Role-based authentication

---

# 👨‍💻 Author

**Tushar Sharma**

AI & Machine Learning | Data Science | Generative AI

---

## ⭐ If you found this project useful, consider giving it a star!