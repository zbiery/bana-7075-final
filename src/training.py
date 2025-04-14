import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from src.logger import logger

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def save_training_data(df: pd.DataFrame, return_path: bool = True) -> str | None:
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

    full_path = os.path.abspath(full_path).replace("\\", "/")  # Normalize for MLflow on Windows
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
    run_name: str = "Logistic Regression",
    artifact_path: str = "artifacts"
):
    """
    Trains a logistic regression model, evaluates it, and logs parameters, metrics,
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

        # Combine and save training data
        training_data = pd.concat([x_train, y_train.rename("IsCanceled")], axis=1)
        training_data_path = save_training_data(training_data)
        assert os.path.exists(training_data_path), f"File missing at path: {training_data_path}"
        mlflow.log_artifact(training_data_path, artifact_path="training_data")

        # Train model
        model = LogisticRegression(**model_params)
        model.fit(x_train, y_train)

        # Evaluate
        train_acc = model.score(x_train, y_train)
        test_acc = model.score(x_test, y_test)
        f1 = f1_score(y_test, model.predict(x_test))

        # Log metrics
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "val_accuracy": test_acc,
            "f1_score": f1
        })

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            input_example=x_test.head(3)
        )

        logger.info(
            f"Model trained. Train acc: {train_acc:.4f}, "
            f"Val acc: {test_acc:.4f}, F1: {f1:.4f}"
        )

        if return_model:
            return model
