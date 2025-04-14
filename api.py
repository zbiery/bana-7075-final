import datetime
import io
from typing import List
import mlflow.pyfunc
import pandas as pd
from dataclasses import dataclass
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Set experiment name
mlflow.set_experiment("Forecasting Apple Demand")

# Load the trained model from MLflow
MODEL_URI = "models:/apple_demand@champion"  # Replace with your model name and alias
model = mlflow.pyfunc.load_model(MODEL_URI)


# Define the expected input schema for a single prediction
@dataclass
class InputData(BaseModel):
    LeadTime: int
    Adults: int
    Children: int
    PreviousCancellations: int
    PreviousBookingsNotCanceled: int
    DaysInWaitingList: int
    CustomerType: str
    DepositType: str
    DistributionChannel: str
    StayType: str
    TotalNights: int
    HasBabies: int
    HasMeals: int
    HasParking: int
    IsCanceled: int


@app.post("/predict")
def predict_single(input_data: List[InputData]):
    """Endpoint for real-time predictions with a single input."""

    # Convert input to DataFrame
    df = pd.DataFrame([data.model_dump() for data in input_data])

    try:
        # Make predictions
        predictions = model.predict(df)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Endpoint for batch predictions using a CSV file."""
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Validate required columns
        required_features = [
            "LeadTime",
            "Adults",
            "Children",
            "PreviousCancellations",
            "PreviousBookingsNotCanceled",
            "DaysInWaitingList",
            "CustomerType",
            "DepositType",
            "DistributionChannel",
            "StayType",
            "TotalNights",
            "HasBabies",
            "HasMeals",
            "HasParking",
            "IsCanceled"
        ]

        if not all(feature in df.columns for feature in required_features):
            missing_cols = set(required_features) - set(df.columns)
            raise HTTPException(
                status_code=400, detail=f"Missing columns: {missing_cols}"
            )

        # Make batch predictions
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))