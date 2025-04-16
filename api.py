import io
import traceback
from typing import List
import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from src.predicting import predict_glm, predict_nnet

# Initialize FastAPI app
app = FastAPI()

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Registry of models
MODEL_REGISTRY = {
    "glm": {
        "uri": "models:/lr_champion/1",
        "type": "pyfunc",
        "requires_scaler": False,
    },
    "nnet": {
        "uri": "models:/nn_champion/1",
        "type": "pytorch",
        "requires_scaler": True,
    }
}

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

def load_model_and_scaler(model_key: str):
    config = MODEL_REGISTRY.get(model_key)
    if not config:
        raise ValueError(f"Unsupported model: {model_key}")

    if config["type"] == "pyfunc":
        model = mlflow.pyfunc.load_model(config["uri"])
        return model, None

    elif config["type"] == "pytorch":
        model = mlflow.pytorch.load_model(config["uri"])
        model_name, model_version = config["uri"].split("/")[1:]
        client = MlflowClient()
        model_info = client.get_model_version(name=model_name, version=model_version)
        run_id = model_info.run_id
        scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")
        return model, scaler

@app.post("/predict")
def predict_single(input_data: List[InputData], model: str = "glm"):
    """Endpoint for real-time predictions with dynamic model selection."""
    try:
        model = model.lower()
        loaded_model, scaler = load_model_and_scaler(model)

        df = pd.DataFrame([data.model_dump() for data in input_data])

        if model == "glm":
            preds, probs = predict_glm(df, loaded_model)
        elif model == "nnet":
            preds, probs = predict_nnet(df, loaded_model, scaler)
        else:
            raise ValueError(f"Unsupported model: {model}")

        return {"predictions": preds, "probabilities": probs}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...), model: str = "glm"):
    """Endpoint for batch predictions with CSV file and dynamic model selection."""
    try:
        model = model.lower()
        loaded_model, scaler = load_model_and_scaler(model)

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        if model == "glm":
            preds, probs = predict_glm(df, loaded_model)
        elif model == "nnet":
            preds, probs = predict_nnet(df, loaded_model, scaler)
        else:
            raise ValueError(f"Unsupported model: {model}")

        return {"predictions": preds, "probabilities": probs}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
