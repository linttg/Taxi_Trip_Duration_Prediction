from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib

from pathlib import Path

app = FastAPI(
    title="NYC Taxi Duration Prediction API"
)

# =====================
# Load model
# =====================

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(
    BASE_DIR / "Models" / "xgboost_model.pkl"
)

# =====================
# Input schema
# =====================

class TaxiInput(BaseModel):
    VendorID: int
    passenger_count: int
    trip_distance: float
    RatecodeID: int
    PULocationID: int
    DOLocationID: int
    pickup_hour: int
    day_of_week: int
    is_weekend: int


# =====================
# Home endpoint
# =====================

@app.get("/")
def home():
    return {
        "message": "NYC Taxi Duration Prediction API"
    }


# =====================
# Prediction endpoint
# =====================

@app.post("/predict")

def predict(data: TaxiInput):

    input_data = pd.DataFrame({
        "VendorID":[data.VendorID],
        "passenger_count":[data.passenger_count],
        "trip_distance":[data.trip_distance],
        "RatecodeID":[data.RatecodeID],
        "PULocationID":[data.PULocationID],
        "DOLocationID":[data.DOLocationID],
        "pickup_hour":[data.pickup_hour],
        "day_of_week":[data.day_of_week],
        "is_weekend":[data.is_weekend]
    })
    prediction = model.predict(input_data)
    return {

        "estimated_duration": round(
            float(prediction[0]),
            2
        )
    }