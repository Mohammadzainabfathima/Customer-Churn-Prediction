# Customer-Churn-Prediction
Built a machine learning model to predict customer churn using feature engineering and Random Forest, achieving an 89% F1-score. Deployed the model with FastAPI for real-time predictions, enabling businesses to identify at-risk customers.
📁 1. Folder Structure
customer-churn-prediction/
│
├── data/
│   └── sample_data.csv
│
├── models/
│   └── churn_model.pkl
│
├── app/
│   ├── main.py
│   └── requirements.txt
│
├── notebooks/
│   └── churn_training.ipynb
│
└── README.md
📌 2. sample_data.csv 
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
001,Female,0,Yes,No,10,Yes,No,DSL,Yes,No,Month-to-month,Yes,Credit card,29.85,298.8,No
002,Male,1,No,No,2,Yes,No,Fiber optic,No,No,Month-to-month,Yes,Electronic check,56.95,188.5,Yes
003,Male,0,No,No,45,No,No,DSL,Yes,Yes,One year,No,Maestro,42.30,1840.75,No
🧠 3. Training Notebook Code (churn_training.ipynb)
# -----------------------------
# Customer Churn Model Training
# -----------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("../data/sample_data.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.fillna(0, inplace=True)

# Features & target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# Identify column types
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [col for col in X.columns if col not in numeric_features]

# Preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Final ML pipeline
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# Save model
joblib.dump(model, "../models/churn_model.pkl")

print("Model saved successfully!")
🚀 4. FastAPI Backend (app/main.py)
This creates a real-time scoring API.
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# Load trained model
model = joblib.load("../models/churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}

@app.post("/predict")
def predict_churn(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(round(prob, 3))
    }
📦 5. requirements.txt (put inside app/ folder)
fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
▶️ 6. How to run the API locally
cd app
uvicorn main:app --reload


Open browser at:
👉 http://127.0.0.1:8000/docs
 (Swagger UI)

📝 7. README.md (professional GitHub version)
# 📊 Customer Churn Prediction  
A complete end-to-end machine learning project that predicts customer churn using ML models, feature engineering, and a FastAPI microservice.

## 🚀 Tech Stack
- Python, Pandas, NumPy  
- Scikit-Learn (ML Model)  
- Random Forest Classifier  
- FastAPI (Model Deployment)  
- Joblib (Model Serialization)  

## 🧠 Workflow
1. Data preprocessing  
2. Feature encoding (OneHot + Scaling)  
3. Model training (Random Forest)  
4. Evaluation (F1-Score, Accuracy)  
5. Save model  
6. Deploy via FastAPI  

## 📌 API Endpoint
### `/predict`  
Send customer details → Get churn prediction & probability.

---

## ▶️ Run API
```bash
cd app
uvicorn main:app --reload


Swagger UI: http://127.0.0.1:8000/docs

📁 Project Structure
customer-churn-prediction/
│── data/
│── models/
│── app/
│── notebooks/
└── README.md




✨ Author

Md Zainab Fathima
Junior Software Engineer | ML & App Development
