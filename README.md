# Sales Forecasting & Demand Intelligence System 📊

An end-to-end machine learning system that predicts future product 
demand, detects unusual sales patterns, segments products by demand 
behavior, and presents everything through a live interactive dashboard.

## 🌐 Live Dashboard
👉 https://salesforecasting-em8watydvx4fccvh97wsoe.streamlit.app/

## 📦 Dataset
- Source: Kaggle — Superstore Sales Dataset
- Records: 9,800 orders across 4 years (2015–2018)
- Features: 24 columns (order date, category, region, sales, etc.)

## 🛠️ Tools & Libraries Used
- Python 3
- Google Colab
- Pandas & NumPy
- Statsmodels (SARIMA)
- Prophet (Facebook)
- XGBoost
- Scikit-learn (Isolation Forest, K-Means, PCA)
- Matplotlib & Seaborn
- Streamlit

## ✅ Tasks Completed
- Task 1: Data Loading, Merging & Deep Exploration
- Task 2: Time Series Analysis & Decomposition
- Task 3: Sales Forecasting using 3 Models (SARIMA, Prophet, XGBoost)
- Task 4: Category & Region Level Forecasting
- Task 5: Anomaly Detection (Isolation Forest + Z-Score)
- Task 6: Product Demand Segmentation (K-Means Clustering)
- Task 7: Interactive Dashboard Deployment (Streamlit)
- Task 8: Executive Business Report

## 📊 Model Performance Comparison
| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| SARIMA | $15,448 | $18,566 | 18.26% |
| Prophet | $14,501 | $19,156 | 17.76% |
| **XGBoost ✓** | **$11,551** | **$14,494** | **14.18%** |

**Recommended Model: XGBoost** — best accuracy across all metrics!

## 🔍 Key Findings
- Sales doubled from 2015 to 2018 — consistent business growth
- Technology is the highest revenue category
- November & December spike every year — Black Friday & Christmas
- Sales Representatives have highest attrition in HR dataset
- Copiers behave uniquely — high volatility, explosive growth

## 📁 Repository Structure
SalesForecasting/
├── analysis.ipynb          ← Complete Jupyter Notebook
├── app.py                  ← Streamlit dashboard code
├── requirements.txt        ← Python dependencies
├── train.csv               ← Superstore sales dataset
├── monthly_sales.csv       ← Aggregated monthly data
├── weekly_sales.csv        ← Aggregated weekly data
├── cluster_features.csv    ← Product cluster data
├── forecast_results.json   ← Model forecast outputs
└── charts/                 ← All visualization images

## 🚀 How to Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run app.py
```

## 👩‍💻 Developed By
Sujithra S | BCA 3rd Year | Stella Maris College, Chennai
sujithra.s290407@gmail.com
