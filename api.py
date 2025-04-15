import datetime
import io
import traceback
from typing import List

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from mlflow.models import Model
from src.preprocessing import encode_data

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

def get_input_columns_from_pytorch_model(model_uri: str) -> list:
    model_path = mlflow.artifacts.download_artifacts(model_uri)
    m = Model.load(model_path)
    return [col.name for col in m.signature.inputs]

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
        expected_cols = model.metadata.get_input_schema().input_names()
        return model, None, expected_cols

    elif config["type"] == "pytorch":
        model = mlflow.pytorch.load_model(config["uri"])

        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        model_name, model_version = config["uri"].split("/")[1:]
        model_info = client.get_model_version(name=model_name, version=model_version)
        run_id = model_info.run_id

        scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")
        expected_cols = get_input_columns_from_pytorch_model(config["uri"])
        return model, scaler, expected_cols

@app.post("/predict")
def predict_single(input_data: List[InputData], model: str = "glm"):
    try:
        model = model.lower()
        loaded_model, scaler, expected_cols = load_model_and_scaler(model)

        df = pd.DataFrame([data.model_dump() for data in input_data])
        print("[DEBUG] Columns BEFORE encoding:", df.columns.tolist())

        df = encode_data(df)
        print("[DEBUG] Columns AFTER encoding:", df.columns.tolist())

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        bool_cols = [col for col in df.columns if col.startswith(("CustomerType_", "DistributionChannel_", "StayType_", "DepositType_"))]
        df[bool_cols] = df[bool_cols].astype(bool)

        int_cols = ["HasBabies", "HasMeals", "HasParking"]
        present_int_cols = [col for col in int_cols if col in df.columns]
        df[present_int_cols] = df[present_int_cols].astype("int32")

        print(f"[DEBUG] Final shape before prediction: {df.shape}")
        print(f"[DEBUG] Final columns: {df.columns.tolist()}")

        if scaler:
            df_scaled = scaler.transform(df)
            tensor = torch.tensor(df_scaled, dtype=torch.float32)
            loaded_model.eval()
            with torch.no_grad():
                logits = loaded_model(tensor)
                probs = torch.sigmoid(logits).numpy().flatten()
                preds = (probs > 0.5).astype(int)
            return {"predictions": preds.tolist(), "probabilities": probs.tolist()}
        else:
            if hasattr(loaded_model, "predict_proba"):
                probs = loaded_model.predict_proba(df)[:, 1]
                preds = loaded_model.predict(df)
                return {"predictions": preds.tolist(), "probabilities": probs.tolist()}
            else:
                preds = loaded_model.predict(df)
                return {"predictions": preds.tolist()}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...), model: str = "glm"):
    try:
        model = model.lower()
        loaded_model, scaler, expected_cols = load_model_and_scaler(model)

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        df = encode_data(df)

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        bool_cols = [col for col in df.columns if col.startswith(("CustomerType_", "DistributionChannel_", "StayType_", "DepositType_"))]
        df[bool_cols] = df[bool_cols].astype(bool)

        int_cols = ["HasBabies", "HasMeals", "HasParking"]
        present_int_cols = [col for col in int_cols if col in df.columns]
        df[present_int_cols] = df[present_int_cols].astype("int32")

        if scaler:
            df_scaled = scaler.transform(df)
            tensor = torch.tensor(df_scaled, dtype=torch.float32)
            loaded_model.eval()
            with torch.no_grad():
                logits = loaded_model(tensor)
                probs = torch.sigmoid(logits).numpy().flatten()
                preds = (probs > 0.5).astype(int)
            return {"predictions": preds.tolist(), "probabilities": probs.tolist()}
        else:
            if hasattr(loaded_model, "predict_proba"):
                probs = loaded_model.predict_proba(df)[:, 1]
                preds = loaded_model.predict(df)
                return {"predictions": preds.tolist(), "probabilities": probs.tolist()}
            else:
                preds = loaded_model.predict(df)
                return {"predictions": preds.tolist()}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
