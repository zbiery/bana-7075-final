import pandas as pd
import subprocess
import logging
import great_expectations as gx
from great_expectations.expectations import ExpectColumnToExist, ExpectColumnValuesToBeOfType, ExpectColumnValuesToNotBeNull
from great_expectations.core import ExpectationSuiteValidationResult
from src.logger import logger
from src.utils import suppress_gx_logging


@suppress_gx_logging
def validate_data(df: pd.DataFrame) -> bool:
    """
    Validates the input DataFrame using Great Expectations via in-memory Data Context.

    Args:
        df (pd.DataFrame): The DataFrame to validate.

    Returns:
        bool: True if validation passed, False o.w.
    """

    logger.info("Initializing Great Expectations Data Context and running validation...")

    try:
        context = gx.get_context()
        datasource = context.data_sources.add_pandas(name="pandas")
        asset = datasource.add_dataframe_asset(name="hotel_asset")
        batch_definition = asset.add_batch_definition_whole_dataframe("hotel_batch")
        batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

        suite = gx.ExpectationSuite(name="hotel_booking_suite")
        suite = context.suites.add(suite)

        expected_columns = {
            "LeadTime": "int64",
            "Adults": "int64",
            "Children": "int64",
            "PreviousCancellations": "int64",
            "DaysInWaitingList": "int64",
            "CustomerType": "CategoricalDtypeType",
            "DistributionChannel": "CategoricalDtypeType",
            "StayType": "CategoricalDtypeType",
            "TotalNights": "int64",
            "HasBabies": "int32",
            "HasMeals": "int32",
            "HasParking": "int32",
            "IsCanceled": "int64"
        }

        for column, col_type in expected_columns.items():
            suite.add_expectation(ExpectColumnToExist(column=column))
            suite.add_expectation(ExpectColumnValuesToBeOfType(column=column, type_=col_type))
            suite.add_expectation(ExpectColumnValuesToNotBeNull(column=column))

        validation_results: ExpectationSuiteValidationResult = batch.validate(suite)

        if validation_results.success:
            logger.info("PASSED: Data validation passed.")
            df.to_csv("data/h1_hotel_clean.csv", index=False)
            return True
        else:
            logger.warning("FAILED: Data validation failed.")
            for res in validation_results.results:
                if not res.success:
                    column = res.expectation_config.kwargs.get("column", "N/A")
                    logger.warning(f"FAILED: {res.expectation_config.expectation_context} on column '{column}' -> {res.result}")
            return False

    except Exception as e:
        logger.error(f"Data validation error: {e}")
        return False

def version_data(filepath: str, commit_msg: str, tag: str, tag_msg: str):
    """
    Adds a dvc file to Git, commits the change, and tags it with a version.

    Args:
        filepath (str): The path to the file or .dvc file to add.
        commit_msg (str): The Git commit message.
        tag (str): The Git tag (e.g. 'v2.0').
        tag_msg (str): The message for the tag.

    Raises:
        RuntimeError: If any Git command fails.
    """
    try:
        logger.info(f"Adding file to Git: {filepath}")
        subprocess.run(["git", "add", filepath], check=True)

        logger.info(f"Committing with message: {commit_msg}")
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)

        logger.info(f"Tagging commit with {tag}: {tag_msg}")
        subprocess.run(["git", "tag", "-a", tag, "-m", tag_msg], check=True)

        logger.info("Git versioning complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        raise RuntimeError(f"Git command failed: {e}")
