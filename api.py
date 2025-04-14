import datetime
import io
from typing import List
import mlflow.pyfunc
import pandas as pd
# from dataclasses import dataclass
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from src.preprocessing import encode_data

# Initialize FastAPI app
app = FastAPI()

# Set experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5000") 
mlflow.set_experiment("hotel_cancellation_lr")

# Load the trained model from MLflow
MODEL_URI = "models:/lr_champion/1"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Define the expected input schema for a single prediction
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

@app.post("/predict")
def predict_single(input_data: List[InputData]):
    """Endpoint for real-time predictions with a single input."""
    try:
        df = pd.DataFrame([data.model_dump() for data in input_data])
        
        # One-hot encode to match training pipeline
        df_encoded = encode_data(df)

        # Fix boolean and int32 typing
        df_encoded["HasBabies"] = df_encoded["HasBabies"].astype("int32")
        df_encoded["HasMeals"] = df_encoded["HasMeals"].astype("int32")
        df_encoded["HasParking"] = df_encoded["HasParking"].astype("int32")

        # Add missing columns with default values (False)
        expected_cols = model.metadata.get_input_schema().input_names()
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = False

        # ⬅️ Ensure correct column order
        df_encoded = df_encoded[expected_cols]

        predictions = model.predict(df_encoded)
        return {"predictions": predictions.tolist()}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


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
        try:
            predictions = model.predict(df)
            return {"predictions": predictions.tolist()}
        except Exception as e:
            import traceback
            traceback.print_exc()  # Logs full traceback to console
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))