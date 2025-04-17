from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.test_preset import DataStabilityTestPreset
import pandas as pd
from typing import Optional
import os

def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: Optional[str] = None,
    output_html: str = "reports/drift_report.html"
) -> None:
    """
    Generate and save an Evidently report for data and target drift.

    Args:
        reference (pd.DataFrame): Data used during training.
        current (pd.DataFrame): Incoming or recent data.
        target_col (Optional[str]): Column name of the target variable.
        output_html (str): File path to save the HTML drift report.
    """
    metrics = [DataDriftPreset()]
    if target_col and target_col in reference.columns and target_col in current.columns:
        metrics.append(TargetDriftPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    report.save_html(output_html)

def run_data_tests(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_html: str = "reports/data_test_suite.html"
) -> None:
    """
    Run a stability test suite on the input data and export as HTML.

    Args:
        reference (pd.DataFrame): Baseline data for comparison.
        current (pd.DataFrame): New incoming data.
        output_html (str): File path to save the HTML test report.
    """
    suite = TestSuite(tests=[DataStabilityTestPreset()])
    suite.run(reference_data=reference, current_data=current)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    suite.save_html(output_html)
