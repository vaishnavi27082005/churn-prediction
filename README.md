\# 📊 Customer Churn Prediction with Explainable AI



An end-to-end machine learning project that predicts telecom customer churn and explains

\*why\* each prediction was made, deployed as an interactive web app.



!\[Python](https://img.shields.io/badge/Python-3.10+-blue)

!\[scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)

!\[XGBoost](https://img.shields.io/badge/XGBoost-Model-green)

!\[Streamlit](https://img.shields.io/badge/Streamlit-App-red)

!\[SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)



\---



\## 🚀 Demo



<!-- Replace with your actual screenshot or Streamlit Community Cloud link -->

🔗 \*\*Live App:\*\* \[Add your deployed Streamlit link here]



!\[App Screenshot](docs/screenshot\_app.png)

!\[SHAP Explanation](docs/screenshot\_shap.png)



\---



\## 📌 Overview



Customer churn — when a customer stops using a service — is one of the most costly

problems for subscription-based businesses. This project builds a machine learning

pipeline that:



\- Predicts whether a customer is likely to churn

\- Compares multiple ML models to pick the best performer

\- Explains \*individual\* predictions using SHAP (not just a black-box output)

\- Is deployed as a live, interactive web app



\## 🧠 Key Results



| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |

|---|---|---|---|---|---|

| \*\*Logistic Regression\*\* | 0.80 | 0.64 | 0.55 | 0.59 | \*\*0.84\*\* |

| Random Forest | 0.79 | 0.63 | 0.49 | 0.55 | 0.82 |

| XGBoost | 0.78 | 0.60 | 0.52 | 0.56 | 0.82 |



Logistic Regression was selected as the best model based on ROC-AUC — a good reminder

that simpler models can outperform more complex ones depending on the dataset.



\## 🔍 Key Insights from EDA



\- Month-to-month contract customers churn far more than annual contract customers

\- New customers (low tenure) are the highest churn-risk group

\- Customers paying by electronic check churn more than those on automatic payments

\- Fiber optic internet customers show higher churn than DSL customers



\## 🛠️ Tech Stack



\- \*\*Data processing:\*\* pandas, numpy

\- \*\*Modeling:\*\* scikit-learn, XGBoost

\- \*\*Explainability:\*\* SHAP

\- \*\*Deployment:\*\* Streamlit

\- \*\*Visualization:\*\* matplotlib, seaborn



\## 📁 Project Structure



```

churn\_project/

├── data/                  # dataset (not included — see Dataset section below)

├── models/                # trained model artifacts (generated, not committed)

├── eda\_charts/            # generated EDA visualizations

├── app/

│   └── streamlit\_app.py   # deployable Streamlit web app

├── train\_model.py         # training pipeline

├── eda.py                 # EDA chart generation script

├── requirements.txt

└── README.md

```



\## 📊 Dataset



This project uses the \*\*Telco Customer Churn\*\* dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn



Download it and place it at `data/churn.csv` before running the training script

(not included in this repo — see `.gitignore`).



\## ⚙️ Setup \& Usage



```bash

\# Clone the repo

git clone https://github.com/<your-username>/churn-prediction.git

cd churn-prediction



\# Create virtual environment

python -m venv venv

venv\\Scripts\\activate        # Windows

\# source venv/bin/activate   # Mac/Linux



\# Install dependencies

pip install -r requirements.txt



\# Add dataset to data/churn.csv (see Dataset section above)



\# Run EDA (optional, generates charts)

python eda.py



\# Train the model

python train\_model.py



\# Launch the web app

streamlit run app/streamlit\_app.py

```



\## 🎯 Why This Project



Most churn prediction tutorials stop at a model with an accuracy score. This project

goes further by:

1\. Comparing multiple models with proper evaluation metrics (not just accuracy, which

&#x20;  is misleading on imbalanced data)

2\. Adding \*\*explainability\*\* — showing exactly which features drove each individual

&#x20;  prediction, using SHAP

3\. \*\*Deploying\*\* the model as something an actual non-technical user could interact

&#x20;  with, rather than leaving it in a notebook



\## 🔮 Future Improvements



\- Handle class imbalance with SMOTE or class weighting to improve churner recall

\- Add a model retraining pipeline for new data

\- Deploy permanently on Streamlit Community Cloud / cloud hosting

\- Add authentication for a production-style deployment



\## 📄 License



This project is for educational purposes as part of an academic mini-project.



\## 🙋 Author



<!-- Add your name, LinkedIn, and portfolio link here -->

\*\*\[vaishnavi27082005]\*\*

\[LinkedIn](www.linkedin.com/in/vaishnavi-k-7b6506291) 

