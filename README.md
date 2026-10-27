# 🏭 Industrial IoT Predictive Maintenance & Explainable AI (XAI) Suite

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange?style=for-the-badge)
![Plotly Dash](https://img.shields.io/badge/Dash-Plotly-blueviolet?style=for-the-badge&logo=plotly&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

## 📌 Executive Summary
This project is an end-to-end Machine Learning ecosystem designed for **Industry 4.0** predictive maintenance. Using the UCI AI4I 2020 sensor telemetry dataset, the system predicts imminent machine failures, mitigating unplanned downtime while preventing false alarms. 

By employing a cost-sensitive **XGBoost** architecture, the model successfully overcomes a severe class imbalance (only a 3.39% historical failure rate). It achieves a **79.4% Recall** on actual machine breakdowns with a **76.1% Precision** rate on a validation set of 2,000 machines, translating to significant operational cost savings for heavy manufacturing facilities.

## 🛠️ Core Technologies
* **Machine Learning:** Python, Scikit-Learn, XGBoost
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)
* **Web Dashboard & UI:** Plotly, Dash, Bootstrap (Premium Slate Theme)
* **Data Processing:** Pandas, NumPy, Joblib, Matplotlib

## 📂 Project Directory Structure
```text
Predictive_Maintenance_AI/
├── data/
│   └── ai4i2020.csv                          # Auto-downloaded UCI dataset (generated on run)
├── notebooks/
│   └── predictive_maintenance.ipynb          # Jupyter notebook for EDA & model training
├── app.py                                    # Main Dash interactive web application
├── predictive_maintenance_model.joblib       # Serialized XGBoost model (generated on run)
├── shap_summary.png                          # Global SHAP dependency visualization
└── README.md                                 # Project documentation

🧠 Architecture & Methodology
Data Preprocessing & Leakage Prevention: Removed post-failure identifying columns (TWF, HDF, PWF, OSF, RNF) to ensure the algorithm learns strictly from pre-failure physical telemetry, guaranteeing authentic predictive capability.

Class Imbalance Handling: Implemented the scale_pos_weight parameter derived from the exact class distribution to heavily penalize missed failures, establishing robust minority-class detection.

Explainability: Integrated shap.TreeExplainer to extract root-cause diagnostics. The analysis mathematically proves Torque [Nm] and Tool Wear [min] as the primary mechanical failure drivers, preventing "black-box" decision making and building trust with floor engineers.

🚀 Interactive Digital Twin App (Dash)
The system is deployed via a production-ready local web application featuring 4 core modules:

Live Digital Twin Simulator: Adjust sensor telemetry via what-if sliders to observe real-time failure probabilities and local SHAP diagnostics.

Validation Metrics: Executive summary of ROC-AUC, F1-Score, and interactive Confusion Matrix.

Global SHAP Diagnostics: Feature importance and interaction plots (e.g., Torque vs. RPM).

Automated CMMS Queue: Risk-filtered work-order generation for ERP/SAP integration with one-click CSV export capabilities.

💻 Installation & Local Deployment (Zero-Config)
The application is designed with fault tolerance. If the dataset or the pre-trained model is missing, the system will automatically download the telemetry data and train the model upon the first execution.

Bash
# 1. Clone the repository
git clone [https://github.com/GorkemKutukculer/Predictive_Maintenance_AI.git](https://github.com/GorkemKutukculer/Predictive_Maintenance_AI.git)
cd Predictive_Maintenance_AI

# 2. Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Install required dependencies
pip install pandas numpy scikit-learn xgboost shap plotly dash dash-bootstrap-components joblib matplotlib

# 4. Run the local Dash application
python app.py

Navigate to http://127.0.0.1:8050/ in your local web browser to access the suite. 