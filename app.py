```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Initialize the FastAPI application
app = FastAPI(
    title="Customer Cancellation Prediction API",
    description="Predict whether a customer is likely to cancel the service.",
    version="1.0.0"
)

# Load the saved preprocessing and model pipeline
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
        "message": "Customer Cancellation Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert the received customer data into a DataFrame
    input_data = pd.DataFrame([customer.model_dump()])

    # Feature engineering: create the same features used during training

    # 1. Categorize the number of penalties
    input_data["PenaltyLevel"] = pd.cut(
        input_data["NumberOfPenalties"],
        bins=[-1, 3, 100],
        labels=["Low (0-3)", "High (4+)"]
    )

    # 2. Categorize the base fee
    input_data["FeeBand"] = pd.cut(
        input_data["BaseFee"],
        bins=[0, 20, 30, 38, 100],
        labels=["<=20", "20-30", "30-38", ">38"]
    )

    # 3. Categorize customer tenure
    input_data["TenureGroup"] = pd.cut(
        input_data["CustomerTenure"],
        bins=[-1, 60, 90, 120, 1000],
        labels=["New", "Mid", "Established", "Long-term"]
    )

    # 4. Interaction between Red Plan and penalties
    input_data["RedPlan_x_Penalties"] = (
        (input_data["PlanType"] == "Red Plan").astype(int)
        * input_data["NumberOfPenalties"]
    )

    # 5. Penalties relative to customer tenure
    input_data["Penalties_per_Tenure"] = (
        input_data["NumberOfPenalties"]
        / (input_data["CustomerTenure"] + 1)
    )

    # 6. Service tickets relative to customer tenure
    input_data["Tickets_per_Tenure"] = (
        input_data["ServiceTickets"]
        / (input_data["CustomerTenure"] + 1)
    )

    # Arrange features in the same order used during model training
    if hasattr(model, "feature_names_in_"):
        input_data = input_data.reindex(
            columns=model.feature_names_in_
        )

    # Generate the cancellation prediction
    prediction = int(model.predict(input_data)[0])

    # Get the probability for class 1 (cancellation)
    probabilities = model.predict_proba(input_data)[0]
    classes = list(model.classes_)
    cancellation_index = classes.index(1)
    probability = float(probabilities[cancellation_index])

    return {
        "cancellation_prediction": (
            "Yes" if prediction == 1 else "No"
        ),
        "cancellation_probability": round(probability, 4),
        "cancellation_probability_percent": round(
            probability * 100, 2
        )
    }
```
