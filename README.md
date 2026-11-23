# 🎯 Telco Customer Churn & Retention Command Center

Connected directly to the Machine Learning Stacking Ensemble model developed in `Customer_churn_retention.ipynb` and trained on `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

---

## ⚡ Quick Start (1-Click Run)

### Method 1: Double-Click Launcher (Windows)
Double-click **`run_dashboard.bat`** in this folder:
- **`[1]`**: Launches the **Full Streamlit Machine Learning Dashboard** (`app.py`) on `http://localhost:8501`.
- **`[2]`**: Opens the **Standalone Zero-Install Web Dashboard** (`dashboard.html`) directly in your default browser.
- **`[3]`**: Retrains and updates the model pipeline (`train_model.py`).

### Method 2: Command Line (Streamlit App)
```bash
python -m streamlit run app.py
```

### Method 3: Standalone Instant Web Dashboard (Zero Install)
Double-click **`dashboard.html`** or open it in Chrome, Edge, or Firefox. No active Python server required.

---

## 🏗️ Architecture & Model Pipeline

The dashboard connects directly to the **Stacked Generalization Ensemble** trained on the customer dataset:

1. **Feature Engineering**:
   - `NumServices`: Active digital services count.
   - `NoSupportServices`: Flag for accounts without Online Security and Tech Support.
   - `AvgMonthlySpend`: Historical average monthly spending (`TotalCharges / tenure`).
   - `BillingSpike`: Difference between current bill and historical average.
   - `IsMonthToMonth`: Critical risk factor identifying short-term contracts.
   - `AutoPay`: Payment convenience indicator.
   - `NewCustomer`: Onboarding vulnerability flag (`tenure <= 6` months).

2. **Model Benchmark & Stacking Ensemble**:
   - **Logistic Regression (Balanced)**: High recall on churning segment.
   - **Decision Tree**: Captures non-linear contract splits.
   - **Random Forest (300 Trees)**: Ensemble variance reduction.
   - **Gradient Boosting (180 Trees)**: Sequentially optimizes pseudo-residuals.
   - **Stacked Meta-Learner**: Blends out-of-fold probability distributions via 5-Fold Stratified Cross-Validation (`ROC-AUC: 0.847`, `Recall: 0.789`).

---

## 🌟 Dashboard Capabilities

1. **🎯 Priority At-Risk Queue (Tab 1)**:
   - Real-time KPIs: Critical accounts count, total queue, mean churn probability, and Monthly Revenue at Risk ($ MRR).
   - Search by Customer ID, filter by Risk Tier (High / Medium / Low), Contract, or custom probability threshold.
   - 1-Click CSV export of the prioritized intervention list (`retention_dashboard.csv`).

2. **🔍 Customer Deep-Dive & Action Desk (Tab 2)**:
   - Interactive Churn Probability Speedometer Gauge.
   - Top-3 root cause drivers per customer with exact SHAP/feature weights.
   - Prescribed retention playbook (e.g. Contract lock, Rate discount, Tech support setup).
   - Conversational customer care call scripts (toggle between **English** and **Hinglish**).

3. **🔮 Live What-If Retention Simulator (Tab 3)**:
   - Powered by the live trained model.
   - Adjust contract term (Month-to-month ➔ 1 Year / 2 Year), discount % (0-40%), add Tech Support, or switch to AutoPay.
   - Model recalculates post-intervention churn risk on the fly and displays the net risk reduction.

4. **📊 Portfolio Analytics & Risk Breakdown (Tab 4)**:
   - Risk Tier distribution donut chart.
   - Mean churn probability by contract type.
   - Top 10 most prevalent churn drivers across the customer base.

5. **🧠 Model Explainability & Candidate Benchmark (Tab 5)**:
   - Model performance comparison matrix (Accuracy, Precision, Recall, F1, ROC-AUC, Train Time).
   - Ensemble architecture documentation.

---

## 📁 Project File Manifest

| File | Description |
| :--- | :--- |
| **`app.py`** | Main Streamlit application with live model scoring & What-If simulator. |
| **`dashboard.html`** | Standalone zero-dependency web dashboard (Chart.js + Tailwind CSS). |
| **`run_dashboard.bat`** | 1-Click Windows launcher menu. |
| **`retention_dashboard.csv`** | Prioritized at-risk customer queue output with top drivers and actions. |
| **`train_model.py`** | Standalone training and model artifact export script. |
| **`Customer_churn_retention.ipynb`** | Original model development notebook. |
| **`WA_Fn-UseC_-Telco-Customer-Churn.csv`** | Telco customer churn dataset (7,043 rows). |
