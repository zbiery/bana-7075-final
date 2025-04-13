import pandas as pd
import numpy as np
from typing import Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.logger import logger

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the hotel dataset by handling types, missing values, duplicates, and filtering.
    Does NOT include feature engineering (handled in create_features).

    Args:
        df (pd.DataFrame): The dataframe to perform cleaning operations on

    Returns:
        pd.DataFrame: A dataframe containing the cleaned dataset
    """
    logger.info("Starting data cleaning...")

    try:
        # Convert string columns to categorical
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("category")
        logger.info("Converted object columns to categorical.")

        # Remove duplicates based on conditions
        original_len = len(df)
        df = df[~((df["IsCanceled"] == 1) & df.duplicated())]
        df = df[~(df["CustomerType"].isin(["Group", "Transient-Party"]) & df.duplicated())]
        logger.info(f"Removed duplicates. Rows reduced from {original_len} to {len(df)}.")

        # # Convert key columns to categorical
        # for col in ["IsCanceled", "IsRepeatedGuest"]:
        #     df[col] = df[col].astype("category")

        # Handle 'Meal'
        df["Meal"] = df["Meal"].astype("object").replace("Undefined", "SC")
        meal_order = ["Undefined", "SC", "BB", "HB", "FB"]
        df["Meal"] = pd.Categorical(df["Meal"], categories=meal_order, ordered=True)

        # Handle 'DistributionChannel' and 'MarketSegment'
        df["DistributionChannel"] = df["DistributionChannel"].astype("object").replace("Undefined", pd.NA)
        df["DistributionChannel"] = df["DistributionChannel"].astype("category")
        df["MarketSegment"] = df["MarketSegment"].astype("object").replace("Undefined", pd.NA)
        df["MarketSegment"] = df["MarketSegment"].astype("category")

        # Filter out unrealistic values
        before_filter = len(df)
        df = df[(df["Adults"] < 5) & (df["Children"] < 10) & (df["Babies"] < 9)]
        df = df[df["RequiredCarParkingSpaces"] <= (df["Adults"] + df["Children"])]
        df = df[df["LeadTime"] < 709]
        df = df[df["ADR"] >= 0]
        df = df[df["ADR"] < 5400]
        logger.info(f"Removed outliers. Rows reduced from {before_filter} to {len(df)}.")

        # Drop unnecessary columns
        df = df.drop(columns=[
            "ReservationStatus", "ReservationStatusDate",
            "ReservedRoomType", "AssignedRoomType", "Country",
            "Agent", "Company", "ADR", "TotalOfSpecialRequests", 
            "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateWeekNumber", 
            "ArrivalDateDayOfMonth", "BookingChanges"
        ])
        logger.info("Removed unimportant columns.")

        # # Drop remaining NAs
        # df = df.dropna()
        # logger.info("Removed NAs.")
        logger.info("Data cleaning completed.")
        return df

    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the cleaned hotel booking dataset.

    Args:
        df (pd.DataFrame): The cleaned dataframe to perform feature engineering on.
    
    Returns: 
        pd.DataFrame: A dataframe containing the final engineered dataset.
    """
    logger.info("Starting feature engineering...")

    try:
        # TotalNights = weekday + weekend nights
        df["TotalNights"] = df["StaysInWeekNights"] + df["StaysInWeekendNights"]

        # StayType
        df["StayType"] = pd.Categorical(
            np.select(
                [
                    (df["StaysInWeekNights"] >= 1) & (df["StaysInWeekendNights"] == 0),
                    (df["StaysInWeekNights"] == 0) & (df["StaysInWeekendNights"] >= 1)
                ],
                ["Weekday", "Weekend"],
                default="Mixed"
            ),
            categories=["Weekday", "Weekend", "Mixed"]
        )

        # Binary flags
        df["HasBabies"] = (df["Babies"] != 0).astype(int)
        df["HasMeals"] = (df["Meal"] != "SC").astype(int)
        df["HasParking"] = (df["RequiredCarParkingSpaces"] != 0).astype(int)

        # Re-map CustomerType and DistributionChannel
        df["CustomerType"] = df["CustomerType"].map({
            "Contract": "Non-Transient",
            "Group": "Non-Transient",
            "Transient": "Transient",
            "Transient-Party": "Transient-Party"
        }).astype("category")

        df["DistributionChannel"] = df["DistributionChannel"].map({
            "GDS": "TA/TO",
            "TA/TO": "TA/TO",
            "Corporate": "Corporate",
            "Direct": "Direct"
        }).astype("category")

        # Drop unused
        df = df.drop(columns=["MarketSegment", "IsRepeatedGuest", 
                              "StaysInWeekendNights", "StaysInWeekNights", 
                              "Meal", "Babies", "RequiredCarParkingSpaces"])

        df = df.dropna()
        logger.info("Feature engineering completed.")
        return df

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features in the dataset using one-hot encoding.
    Drops the first category to avoid multicollinearity.

    Args:
        df (pd.DataFrame): The input DataFrame with categorical columns.

    Returns:
        pd.DataFrame: One-hot encoded DataFrame.
    """
    logger.info("Starting one-hot encoding of categorical variables...")

    try:
        encoded_df = pd.get_dummies(df, columns=["CustomerType", "DistributionChannel", "StayType", "DepositType"], drop_first=True)
        logger.info(f"One-hot encoding completed.")
        return encoded_df

    except Exception as e:
        logger.error(f"Error during encoding: {e}")
        raise

def split_data(df: pd.DataFrame, target: str = "IsCanceled", test_size: float = 0.2, random_state: int = 42):
    """
    Splits the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The dataset to split.
        target (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        X_train, X_test, y_train, y_test: Split features and target sets.
    """
    logger.info(f"Splitting data with target = '{target}', test_size = {test_size}")

    try:
        x = df.drop(columns=[target])
        y = df[target]

        return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    
    except Exception as e:
        logger.error(f"Error during data split: {e}")
        raise

EXCLUDED_COLUMNS = ["HasBabies", "HasMeals", "HasParking", "IsCanceled"]

def scale_data(df: pd.DataFrame, how: str = "min-max") -> Tuple[pd.DataFrame, Union[MinMaxScaler, StandardScaler]]:
    """
    Fits a scaler on the training DataFrame and applies scaling to numeric features, 
    excluding predefined binary columns.

    Args:
        df (pd.DataFrame): Data to scale.
        how (str): Scaling method: 'min-max' or 'z-score'.

    Returns:
        Tuple: (scaled DataFrame, fitted scaler)
    """
    logger.info(f"Fitting and scaling data using {how} method.")

    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in EXCLUDED_COLUMNS]

        df_to_scale = df[cols_to_scale]
        df_excluded = df.drop(columns=cols_to_scale)

        scaler = MinMaxScaler() if how == "min-max" else StandardScaler()
        scaled = scaler.fit_transform(df_to_scale)
        scaled_df = pd.DataFrame(scaled, columns=cols_to_scale, index=df.index)

        result_df = pd.concat([scaled_df, df_excluded], axis=1)
        result_df = result_df[df.columns]

        logger.info("Data scaled successfully.")
        return result_df

    except Exception as e:
        logger.error(f"Error in fit_and_scale_train: {e}")
        raise

def inverse_scale(scaled_df: pd.DataFrame, scaler: Union[MinMaxScaler, StandardScaler]) -> pd.DataFrame:
    """
    Reverts scaled numeric columns back to their original scale using a fitted scaler,
    skipping predefined excluded columns.

    Args:
        df_scaled (pd.DataFrame): The scaled DataFrame.
        scaler: The fitted scaler.

    Returns:
        pd.DataFrame: DataFrame with scaled columns reverted.
    """
    logger.info("Reverting scaled data to original scale.")

    try:
        original_columns = [col for col in scaler.feature_names_in_ if col not in EXCLUDED_COLUMNS]
        scaled_part = scaled_df[original_columns]
        unscaled_part = scaled_df.drop(columns=original_columns)

        unscaled = scaler.inverse_transform(scaled_part)
        unscaled_df = pd.DataFrame(unscaled, columns=original_columns, index=scaled_df.index)

        result_df = pd.concat([unscaled_df, unscaled_part], axis=1)
        result_df = result_df[scaled_df.columns]

        logger.info("Inverse transformation completed.")
        return result_df

    except Exception as e:
        logger.error(f"Error during inverse scaling: {e}")
        raise
