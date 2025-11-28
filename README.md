# Customer Churn Prediction

This repository contains a complete end-to-end **Customer Churn Prediction** project suitable for showcasing to recruiters.

## What’s included
- `data/generate_dataset.py` — creates a realistic synthetic telecom-style churn dataset (CSV).
- `src/preprocess.py` — preprocessing utilities (encoding, feature engineering).
- `src/train_model.py` — trains multiple models (Logistic Regression, Random Forest), evaluates them, and saves the best model.
- `src/predict.py` — loads a saved model and runs predictions on new data (CSV or single example).
- `requirements.txt` — Python dependencies.
- `.gitignore`

## How to run
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS / Linux
   venv\Scripts\activate     # Windows PowerShell
   pip install -r requirements.txt
   ```

2. Generate dataset:
   ```bash
   python data/generate_dataset.py --output data/customer_churn.csv --n 10000
   ```

3. Train models:
   ```bash
   python src/train_model.py --data data/customer_churn.csv --out models/
   ```

4. Predict using saved model:
   ```bash
   python src/predict.py --model models/best_model.joblib --input data/sample_input.csv
   ```

## Notes for recruiters
- The dataset is synthetic but designed to reflect realistic churn drivers (contract type, tenure, payment method, internet type).
- Code is modular: preprocessing, training and inference are separated and easy to extend.
- Model artifacts are saved with joblib.
- Evaluation prints classification report and confusion matrix.

