import torch
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.base import BaseEstimator
from src.preprocessing import encode_data

NNET_NUMERIC_INPUTS = [
    "LeadTime", "Adults", "Children", "PreviousCancellations",
    "PreviousBookingsNotCanceled", "DaysInWaitingList", "TotalNights"
]

NNET_EXPECTED_COLUMNS = [
    "LeadTime", "Adults", "Children", "PreviousCancellations", "PreviousBookingsNotCanceled",
    "DaysInWaitingList", "TotalNights", "HasBabies", "HasMeals", "HasParking",
    "CustomerType_Transient", "CustomerType_Transient-Party",
    "DistributionChannel_Direct", "DistributionChannel_TA/TO",
    "StayType_Weekend", "StayType_Mixed",
    "DepositType_Non Refund", "DepositType_Refundable"
]

GLM_INT_COLS = ["HasBabies", "HasMeals", "HasParking"]

def predict_glm(df: pd.DataFrame, model: BaseEstimator) -> Tuple[list, Optional[list]]:
    """
    Preprocess and predict using a GLM model.

    Args:
        df (pd.DataFrame): Input features.
        model (BaseEstimator): Loaded GLM model.

    Returns:
        Tuple of predictions and optional probabilities.
    """
    df = encode_data(df)

    expected_cols = model.metadata.get_input_schema().input_names()
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    # Ensure correct types
    bool_cols = [c for c in df.columns if any(c.startswith(prefix) for prefix in ["CustomerType_", "DistributionChannel_", "StayType_", "DepositType_"])]
    df[bool_cols] = df[bool_cols].astype(bool)

    for c in GLM_INT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("int32")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[:, 1]
        preds = model.predict(df)
        return preds.tolist(), probs.tolist()
    preds = model.predict(df)
    return preds.tolist(), None

def predict_nnet(df: pd.DataFrame, model: torch.nn.Module, scaler) -> Tuple[list, list]:
    """
    Preprocess and predict using a Neural Network model.

    Args:
        df (pd.DataFrame): Input features.
        model (torch.nn.Module): Trained PyTorch model.
        scaler: Fitted sklearn scaler.

    Returns:
        Tuple of predictions and probabilities.
    """
    df = encode_data(df)

    for col in NNET_EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[NNET_EXPECTED_COLUMNS]

    # Type coercion
    bool_cols = [c for c in df.columns if any(c.startswith(prefix)
                  for prefix in ["CustomerType_", "DistributionChannel_", "StayType_", "DepositType_"])]
    df[bool_cols] = df[bool_cols].astype(bool)
    df[["HasBabies", "HasMeals", "HasParking"]] = df[["HasBabies", "HasMeals", "HasParking"]].astype("int32")

    # Scale numeric features
    numeric_df = df[NNET_NUMERIC_INPUTS]
    non_numeric_df = df.drop(columns=NNET_NUMERIC_INPUTS)

    scaled_numeric = scaler.transform(numeric_df)
    scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=NNET_NUMERIC_INPUTS, index=df.index)

    # Combine & ensure all dtypes are float32/bool
    final_df = pd.concat([scaled_numeric_df, non_numeric_df], axis=1)
    final_df = final_df.astype("float32")

    # Predict
    tensor = torch.tensor(final_df.values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).numpy().flatten()
        preds = (probs > 0.5).astype(int)

    return preds.tolist(), probs.tolist()

def predict_tree(df: pd.DataFrame, model: BaseEstimator) -> Tuple[list, Optional[list]]:
    """
    Preprocess and predict using a GLM model.

    Args:
        df (pd.DataFrame): Input features.
        model (BaseEstimator): Loaded GLM model.

    Returns:
        Tuple of predictions and optional probabilities.
    """
    df = encode_data(df)

    expected_cols = model.metadata.get_input_schema().input_names()
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    # Ensure correct types
    bool_cols = [c for c in df.columns if any(c.startswith(prefix) for prefix in ["CustomerType_", "DistributionChannel_", "StayType_", "DepositType_"])]
    df[bool_cols] = df[bool_cols].astype(bool)

    for c in GLM_INT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("int32")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[:, 1]
        preds = model.predict(df)
        return preds.tolist(), probs.tolist()
    preds = model.predict(df)
    return preds.tolist(), None

