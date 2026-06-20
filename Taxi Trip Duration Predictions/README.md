# NYC Taxi Trip Duration Prediction

## Overview

This project aims to predict the duration of New York City taxi trips using machine learning techniques. The prediction system was developed to support transportation decision-making, route planning, and operational efficiency.

The final model uses XGBoost and is deployed through FastAPI and Streamlit to provide an interactive prediction dashboard.

---

## Objectives

- Predict taxi trip duration before the trip starts.
- Support transportation operational efficiency.
- Build an end-to-end machine learning project from data understanding to deployment.
- Develop an interactive dashboard for users.

---

## Dataset

Dataset: NYC Taxi Trip Dataset

Target variable:

- `duration_min`

Final features used:

1. `VendorID`
2. `passenger_count`
3. `trip_distance`
4. `RatecodeID`
5. `PULocationID`
6. `DOLocationID`
7. `pickup_hour`
8. `day_of_week`
9. `is_weekend`

---

## Project Workflow

1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Machine Learning Modeling
5. Hyperparameter Optimization
6. Model Export
7. API Development (FastAPI)
8. Dashboard Development (Streamlit)

---

## Machine Learning Models

The following models were evaluated:

- Linear Regression
- Random Forest
- XGBoost

Final model:

- XGBoost

Reason for selection:

- High predictive performance
- Efficient computation time
- Suitable for large datasets
- More practical for real-world implementation

---

## Model Performance

| Model | MAE | RMSE | R² |
|------|------|------|------|
| Linear Regression | 5.22 | 7.62 | 0.68 |
| Random Forest | 3.60 | 5.74 | 0.82 |
| XGBoost | 3.70 | 5.84 | 0.81 |

Hyperparameter tuning result:

- MAE: 3.43 minutes
- RMSE: 5.50 minutes
- R²: 0.83

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib

### Deployment Tools

- FastAPI
- Uvicorn
- Streamlit

---

## Project Structure

```text
Taxi Trip Duration Predictions/

├── API/
│   └── main.py
│
├── Data/
│   ├── Cleaned NYC Taxi Trip Data.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── Models/
│   ├── xgboost_model.pkl
│   └── feature_names.pkl
│
├── Notebook/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Machine_Learning.ipynb
│   ├── 05_Model_Optimization.ipynb
│   └── 06_Model_Export.ipynb
│
├── Streamlit/
│   └── app.py
│
└── requirements.txt
```

---

## Run FastAPI

```bash
uvicorn API.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Run Streamlit

```bash
streamlit run Streamlit/app.py
```

---

## Dashboard Features

The dashboard provides:

- Taxi trip duration prediction
- Estimated arrival time
- Traffic condition indicator
- Operational recommendations

---

## Key Insights

- Trip distance is the most influential feature in predicting trip duration.
- Time-related features (pickup hour, day of week, weekend) improve prediction performance.
- Using only features available before the trip starts makes the model more realistic and prevents data leakage.
- XGBoost provides a good balance between predictive performance and computational efficiency.

---

## Author

Retno Lintang
Dibimbing-DSML40