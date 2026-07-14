# 📊 Customer Churn Prediction with Explainable AI

An end-to-end machine learning project that predicts telecom customer churn and explains
*why* each prediction was made, deployed as an interactive web app.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)

---

## 🚀 Live Demo

🔗 **Try it here:** [https://churn-prediction-vaishnavi.streamlit.app](https://churn-prediction-vaishnavi.streamlit.app)

---

## 📌 Overview

Customer churn — when a customer stops using a service — is one of the most costly
problems for subscription-based businesses. This project builds a machine learning
pipeline that:

- Predicts whether a customer is likely to churn
- Compares multiple ML models to pick the best performer
- Explains *individual* predictions using SHAP (not just a black-box output)
- Is deployed as a live, interactive web app

  ## 🔄 Pipeline

![Pipeline FlowChart] 
<img width="715" height="622" alt="Screenshot 2026-07-14 160214" src="https://github.com/user-attachments/assets/a7d91421-9280-429f-87dc-ce2b2a831ce5" />

![Pipeline Graph]
<img width="810" height="315" alt="model_comparison_chart" src="https://github.com/user-attachments/assets/0c19b56f-ef85-4f56-b81f-cb2f84ba6b64" />

## 🧠 Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.80 | 0.64 | 0.55 | 0.59 | **0.84** |
| Random Forest | 0.79 | 0.63 | 0.49 | 0.55 | 0.82 |
| XGBoost | 0.78 | 0.60 | 0.52 | 0.56 | 0.82 |

Logistic Regression was selected as the best model based on ROC-AUC — a good reminder
that simpler models can outperform more complex ones depending on the dataset.

## 🔍 Key Insights from EDA

- Month-to-month contract customers churn far more than annual contract customers
- New customers (low tenure) are the highest churn-risk group
- Customers paying by electronic check churn more than those on automatic payments
- Fiber optic internet customers show higher churn than DSL customers

## 🛠️ Tech Stack

- **Data processing:** pandas, numpy
- **Modeling:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Deployment:** Streamlit
- **Visualization:** matplotlib, seaborn

## 📁 Project Structure
churn_project/
├── data/                  # dataset
├── models/                # trained model artifacts
├── eda_charts/            # generated EDA visualizations
├── app/
│   └── streamlit_app.py   # deployable Streamlit web app
├── train_model.py         # training pipeline
├── eda.py                 # EDA chart generation script
├── requirements.txt
└── README.md

## 📊 Dataset

This project uses the **Telco Customer Churn** dataset from Kaggle:
🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## ⚙️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/vaishnavi27082005/churn-prediction.git
cd churn-prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run EDA (optional, generates charts)
python eda.py

# Train the model
python train_model.py

# Launch the web app
streamlit run app/streamlit_app.py
```

## 🎯 Why This Project

Most churn prediction tutorials stop at a model with an accuracy score. This project
goes further by:
1. Comparing multiple models with proper evaluation metrics (not just accuracy, which
   is misleading on imbalanced data)
2. Adding **explainability** — showing exactly which features drove each individual
   prediction, using SHAP
3. **Deploying** the model as something an actual non-technical user could interact
   with, rather than leaving it in a notebook

## 🔮 Future Improvements

- Handle class imbalance with SMOTE or class weighting to improve churner recall
- Add a model retraining pipeline for new data
- Add authentication for a production-style deployment

## 📄 License

This project is for educational purposes as part of an academic mini-project.

## 🙋 Author

**Vaishnavi**
[LinkedIn](https://www.linkedin.com/in/vaishnavi-k-7b6506291)
