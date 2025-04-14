# src/logistic_regression.py

import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Add the project root to sys.path so src can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_model(
        X,
        y,
        C=1.0,
        max_iter=100,
        solver='lbfgs',
        penalty='l2',
        return_model=False
    ):
    """
    Trains a logistic regression model on feature and target data.

    Args:
        X (ndarray): Features (already numeric and encoded)
        y (ndarray): Target labels (0 or 1)
        C (float): Inverse of regularization strength
        max_iter (int): Maximum number of iterations
        solver (str): Optimization algorithm (e.g., 'lbfgs', 'liblinear')
        penalty (str): Regularization type ('l2', 'l1', 'elasticnet', 'none')
        return_model (bool): Whether to return the full sklearn model object

    Returns:
        model (LogisticRegression): Trained model
    """

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train logistic regression model
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        penalty=penalty
    )
    model.fit(X_train, y_train)

    # Print training and validation accuracy
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    print(f"✅ Logistic Regression trained.")
    print(f"📊 Train Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f}")

    if return_model:
        return model
    return model


def lr_predict(model, X_new, threshold=0.5):
    """
    Generates predictions on new input data using the trained logistic regression model.

    Args:
        model (LogisticRegression): Trained sklearn logistic regression model
        X_new (np.ndarray): New input data (must match training features)
        threshold (float): Decision threshold for converting probabilities to 0/1

    Returns:
        probs (np.ndarray): Raw predicted probabilities
        preds (np.ndarray): Binary predictions (0 or 1)
    """

    # Get probabilities and binary predictions
    probs = model.predict_proba(X_new)[:, 1]
    preds = (probs >= threshold).astype(int)

    return probs, preds
