from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import joblib
import pandas as pd
import sys
import os

# -------------------------------------------------
# 1. FastAPI setup
# -------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# 2. Import (nodig voor joblib pickle restore)
# -------------------------------------------------
from features import column_ratio, ClusterSimilarity


# -------------------------------------------------
# 3. Enum
# -------------------------------------------------
class OceanProximity(str, Enum):
    NEAR_BAY = "NEAR BAY"
    INLAND = "INLAND"
    ISLAND = "ISLAND"
    NEAR_OCEAN = "NEAR OCEAN"
    LESS_THAN_1H_OCEAN = "<1H OCEAN"


# -------------------------------------------------
# 4. Input schema
# -------------------------------------------------
class HousingInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: OceanProximity


# -------------------------------------------------
# 5. Model laden (complete pipeline)
# -------------------------------------------------
sys.path.append(os.path.dirname(__file__))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../model/california_housing_model.pkl")

model = joblib.load(model_path)


# -------------------------------------------------
# 6. Safety function (BELANGRIJK voor log(0) issues)
# -------------------------------------------------
def safe(value):
    if value is None:
        return 1
    return max(value, 1)


# -------------------------------------------------
# 7. Root endpoint
# -------------------------------------------------
@app.get("/")
def root():
    return {"message": "Housing price API is running 🚀"}


# -------------------------------------------------
# 8. Predict endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(data: HousingInput):

    df = pd.DataFrame([{
        "longitude": data.longitude,
        "latitude": data.latitude,
        "housing_median_age": data.housing_median_age,
        "total_rooms": safe(data.total_rooms),
        "total_bedrooms": safe(data.total_bedrooms),
        "population": safe(data.population),
        "households": safe(data.households),
        "median_income": data.median_income,
        "ocean_proximity": data.ocean_proximity.value
    }])

    prediction = model.predict(df)

    return {
        "prediction": round(float(prediction[0]), 2)
    }