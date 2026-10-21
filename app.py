
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Initialize the FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer is likely to churn.",
    version="1.0.0"
)

# Load the saved preprocessing and model pipeline with correct filename
model = joblib.load("customer_churn_model.joblib")


# Define the expected customer input
class CustomerData(BaseModel):
    Territory: str
    DeviceType: str
    Promotion: str
    PlanType: str
    BaseFee: float = Field(ge=0)
    AdditionalFeatureSpend: float = Field(ge=0)
    NumberOfPenalties: int = Field(ge=0)
    CurrentPeriodUsage: float = Field(ge=0)
    PriorPeriodUsage: float = Field(ge=0)
    ServiceRating: float = Field(ge=0)
    ServiceTickets: int = Field(ge=0)
    CustomerTenure: int = Field(ge=0)
    UsageDifference: float
    AdditionalSpendRatio: float = Field(ge=0)
    TotalSpend: float = Field(ge=0)
    StartMonthNumber: int = Field(ge=1, le=12)
    StartWeekNumber: int = Field(ge=1, le=53)


# Health check endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }



# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Convert input data into a DataFrame
    input_data = pd.DataFrame([customer.model_dump()])

    # Generate prediction
    prediction = int(model.predict(input_data)[0])

    # Get probability specifically for Churn = Yes (class 1)
    probabilities = model.predict_proba(input_data)[0]
    classes = list(model.classes_)
    churn_index = classes.index(1)
    probability = float(probabilities[churn_index])

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
        "churn_probability_percent": round(probability * 100, 2)
    }
