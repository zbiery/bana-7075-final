import os
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from typing import Union, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.logger import logger

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def save_training_data(df: pd.DataFrame, return_path: bool = True) -> Union[str,None]:
    """
    Save training data to a local subdirectory with the current MLflow run ID in the filename.

    Args:
        training_data (pd.DataFrame): DataFrame containing the training data.
        return_path (bool, optional): Whether to return the full path to the saved file. Defaults to True.

    Returns:
        str | None: Absolute path to the saved file, or None if return_path is False.
    """
    run_id = mlflow.active_run().info.run_id
    output_dir = os.path.join("data/temp", run_id)
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, "training_data.csv")
    df.to_csv(full_path, index=False)

    full_path = os.path.abspath(full_path).replace("\\", "/") 
    logger.info(f"Saved training data to: {full_path}")

    if return_path:
        return full_path
    return None

def train_lr_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    C: float = 1.0,
    max_iter: int = 100,
    solver: str = 'lbfgs',
    penalty: str = 'l2',
    return_model: bool = False,
    run_name: str = "Logistic Regression"
):
    """
    Trains a SKLearn logistic regression model, evaluates it, and logs parameters, metrics,
    training data, and the model to MLflow.

    Args:
        x_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training labels.
        x_test (pd.DataFrame): Testing feature set.
        y_test (pd.Series): Testing labels.
        C (float): Inverse regularization strength.
        max_iter (int): Maximum number of iterations for convergence.
        solver (str): Algorithm to use in the optimization problem.
        penalty (str): Regularization type.
        return_model (bool): Whether to return the trained model object.
        run_name (str): Name of the MLflow run.
        artifact_path (str): Path within MLflow to save the model.

    Returns:
        LogisticRegression (optional): The trained model if return_model=True.
    """
    mlflow.set_experiment("hotel_cancellation_lr")

    model_params = {
        "C": C,
        "max_iter": max_iter,
        "solver": solver,
        "penalty": penalty,
    }

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(model_params)

        # Save training data to temp file
        training_data = pd.concat([x_train, y_train.rename("IsCanceled")], axis=1)
        training_data_path = save_training_data(training_data)
        assert os.path.exists(training_data_path), f"File missing at path: {training_data_path}"

        # Log to MLflow
        mlflow.log_artifact(training_data_path, artifact_path="training_data")

        # Clean up the temp file
        try:
            os.remove(training_data_path)
            logger.info(f"Deleted temp file: {training_data_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")

        # Train model
        model = LogisticRegression(**model_params)
        model.fit(x_train, y_train)

        # Evaluate
        train_acc = model.score(x_train, y_train)
        test_acc = model.score(x_test, y_test)
        y_test_preds =  model.predict(x_test)
        acc = accuracy_score(y_test, y_test_preds)
        prec = precision_score(y_test, y_test_preds, zero_division=0)
        rec = recall_score(y_test, y_test_preds, zero_division=0)
        f1 = f1_score(y_test, y_test_preds, zero_division=0)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_lr",
            input_example=x_test.head(3)
        )

        logger.info(
            f"Model trained. Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
            f"Recall: {rec:.4f}, F1: {f1:.4f}"
        )

        if return_model:
            return model
        
def train_nn_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: Union[StandardScaler, MinMaxScaler],
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    hidden_dims: Tuple[int, ...] = (64, 32),
    return_model: bool = False,
    run_name: str = "Neural Network"
) -> Union[
    Tuple[nn.Module, Union[StandardScaler, MinMaxScaler]],
    Union[StandardScaler, MinMaxScaler]
]:
    """
    Trains a Pytorch feedforward neural network and logs the model, scaler, and metrics with MLflow.

    Args:
        x_train (pd.DataFrame): Scaled training features.
        y_train (pd.Series): Binary training labels.
        x_test (pd.DataFrame): Scaled test features.
        y_test (pd.Series): Binary test labels.
        scaler (StandardScaler or MinMaxScaler): Fitted scaler used during preprocessing.
        epochs (int): Number of training epochs. Defaults to 10.
        batch_size (int): Training batch size. Defaults to 32.
        lr (float): Learning rate. Defaults to 0.001.
        hidden_dims (tuple): Tuple of hidden layer sizes. Defaults to (64, 32).
        return_model (bool): Whether to return the trained model. Defaults to False.
        return_loss (bool): Whether to return training/validation loss history. Defaults to False.
        run_name (str): Custom MLflow run name. Defaults to "Neural Network".

    Returns:
        Depending on flags:
            - model, scaler
            - scaler
    """
    mlflow.set_experiment("hotel_cancellation_nn")

    hidden_str = "-".join(str(h) for h in hidden_dims)
    run_name = run_name or f"e{epochs}_lr{lr}_h{hidden_str}"

    with mlflow.start_run(run_name=run_name):
        # Save training data to temp file
        training_data = pd.concat([x_train, y_train.rename("IsCanceled")], axis=1)
        training_data_path = save_training_data(training_data)
        assert os.path.exists(training_data_path), f"File missing at path: {training_data_path}"

        # Log to MLflow
        mlflow.log_artifact(training_data_path, artifact_path="training_data")

        # Clean up the temp file
        try:
            os.remove(training_data_path)
            logger.info(f"Deleted temp file: {training_data_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")

        # Log parameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "hidden_dims": hidden_dims
        })

        # Define the model
        class SimpleNN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                layers = []
                dims = [input_dim] + list(hidden_dims)
                for i in range(len(dims) - 1):
                    layers.append(nn.Linear(dims[i], dims[i + 1]))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(dims[-1], 1))  # No sigmoid here
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        model = SimpleNN(x_train.shape[1])
        loss_fn = nn.BCEWithLogitsLoss()  # more stable than Sigmoid + BCELoss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_history = []

        # Convert to tensors
        x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        train_loader = DataLoader(
            TensorDataset(x_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True
        )

        # Training loop
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                y_pred = model(batch_x)
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                y_test_logits = model(x_test_tensor)
                test_loss = loss_fn(y_test_logits, y_test_tensor)

            loss_history.append((avg_train_loss, test_loss.item()))
            logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {test_loss.item():.4f}")

        # Final evaluation
        y_test_probs = torch.sigmoid(y_test_logits).numpy()
        y_test_preds = (y_test_probs > 0.5).astype(int)
        y_true = y_test_tensor.numpy()

        acc = accuracy_score(y_true, y_test_preds)
        prec = precision_score(y_true, y_test_preds, zero_division=0)
        rec = recall_score(y_true, y_test_preds, zero_division=0)
        f1 = f1_score(y_true, y_test_preds, zero_division=0)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Log model
        mlflow.pytorch.log_model(
            model, 
            artifact_path="pytorch_nn", 
            input_example=x_test.head(2))

        # Log scaler
        mlflow.sklearn.log_model(
            sk_model=scaler,
            artifact_path="scaler",
            input_example=x_train.head(2)
        )

        logger.info(
            f"Model trained. Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
            f"Recall: {rec:.4f}, F1: {f1:.4f}"
        )

        if return_model:
            return model, scaler
        else:
            return scaler

def train_rf_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = 'sqrt',
    return_model: bool = False,
    run_name: str = "Random Forest"
):
    """
    Trains a Random Forest model, evaluates it, and logs all relevant artifacts and metrics to MLflow.

    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        max_features (str): Number of features to consider when looking for the best split.
        return_model (bool): Whether to return the trained model.
        run_name (str): MLflow run name.

    Returns:
        RandomForestClassifier (optional): Trained model if return_model=True.
    """
    mlflow.set_experiment("hotel_cancellation_rf")

    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features
    }

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(model_params)

        # Save training data to temp file
        training_data = pd.concat([x_train, y_train.rename("IsCanceled")], axis=1)
        training_data_path = save_training_data(training_data)
        assert os.path.exists(training_data_path), f"File missing at path: {training_data_path}"

        # Log to MLflow
        mlflow.log_artifact(training_data_path, artifact_path="training_data")

        # Clean up the temp file
        try:
            os.remove(training_data_path)
            logger.info(f"Deleted temp file: {training_data_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")

        # Train the model
        model = RandomForestClassifier(**model_params, random_state=42)
        model.fit(x_train, y_train)

        # Evaluate
        y_test_preds = model.predict(x_test)
        acc = accuracy_score(y_test, y_test_preds)
        prec = precision_score(y_test, y_test_preds, zero_division=0)
        rec = recall_score(y_test, y_test_preds, zero_division=0)
        f1 = f1_score(y_test, y_test_preds, zero_division=0)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_rf",
            input_example=x_test.head(3)
        )

        logger.info(
            f"Random Forest trained. Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
            f"Recall: {rec:.4f}, F1: {f1:.4f}"
        )

        if return_model:
            return model
