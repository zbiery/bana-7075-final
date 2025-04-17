import os
import sys

# Ensures src can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evidently.metric_preset import ClassificationPreset
from evidently.report import Report
from evidently import ColumnMapping
import pandas as pd
import mlflow
import mlflow.pyfunc
from src.pipeline import pipeline

# Load current model
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("hotel_cancellation_rf")
MODEL_URI = "models:/rf_champion/1"
model = mlflow.pyfunc.load_model(MODEL_URI)

x_train_base, x_test_base, y_train_base, y_test_base = pipeline(model="tree", filename="H1.csv")
x_train_drift, x_test_drift, y_train_drift, y_test_drift = pipeline(model="tree", filename="H2.csv")

baseline_df = pd.concat([x_train_base, y_train_base.rename("IsCanceled")], axis=1)
drifted_df = pd.concat([x_train_drift, y_train_drift.rename("IsCanceled")], axis=1)

# Ensure int32 conversion for these specific columns
int32_cols = ["HasBabies", "HasMeals", "HasParking"]
for col in int32_cols:
    if col in baseline_df.columns:
        baseline_df[col] = baseline_df[col].astype("int32")
    if col in drifted_df.columns:
        drifted_df[col] = drifted_df[col].astype("int32")

bool_cols = [c for c in baseline_df.columns if c.startswith(("CustomerType_", "DistributionChannel_", "StayType_", "DepositType_"))]
for col in bool_cols:
    baseline_df[col] = baseline_df[col].astype(bool)
    drifted_df[col] = drifted_df[col].astype(bool)

# Score baseline & drift data with model
baseline_df['predictions'] = model.predict(drifted_df.iloc[:27745])
drifted_df['predictions'] = model.predict(drifted_df)

# Add the prediction variable to our column mapping to compare to actuals
column_mapping = ColumnMapping()
column_mapping.target = 'IsCanceled'
column_mapping.prediction = 'predictions'

# Create a model performance monitoring report
performance_report = Report(metrics=[ClassificationPreset()])
performance_report.run(
    reference_data=baseline_df,
    current_data=drifted_df,
    column_mapping=column_mapping  # Explicitly map the target variable
)

# Save report
performance_report.save_html("rf_model_drift.html")

# from evidently.legacy.report import Report
# from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
# from evidently.legacy.test_suite import TestSuite
# from evidently.legacy.test_preset import DataStabilityTestPreset
# import pandas as pd
# from typing import Optional
# import os

# def run_drift_report(
#     reference: pd.DataFrame,
#     current: pd.DataFrame,
#     target_col: Optional[str] = None,
#     output_html: str = "reports/drift_report.html"
# ) -> None:
#     """
#     Generate and save an Evidently report for data and target drift.

#     Args:
#         reference (pd.DataFrame): Data used during training.
#         current (pd.DataFrame): Incoming or recent data.
#         target_col (Optional[str]): Column name of the target variable.
#         output_html (str): File path to save the HTML drift report.
#     """
#     metrics = [DataDriftPreset()]
#     if target_col and target_col in reference.columns and target_col in current.columns:
#         metrics.append(TargetDriftPreset())

#     report = Report(metrics=metrics)
#     report.run(reference_data=reference, current_data=current)

#     os.makedirs(os.path.dirname(output_html), exist_ok=True)
#     report.save_html(output_html)

# def run_data_tests(
#     reference: pd.DataFrame,
#     current: pd.DataFrame,
#     output_html: str = "reports/data_test_suite.html"
# ) -> None:
#     """
#     Run a stability test suite on the input data and export as HTML.

#     Args:
#         reference (pd.DataFrame): Baseline data for comparison.
#         current (pd.DataFrame): New incoming data.
#         output_html (str): File path to save the HTML test report.
#     """
#     suite = TestSuite(tests=[DataStabilityTestPreset()])
#     suite.run(reference_data=reference, current_data=current)

#     os.makedirs(os.path.dirname(output_html), exist_ok=True)
#     suite.save_html(output_html)
