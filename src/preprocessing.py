import pandas as pd
import numpy as np
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

        # Convert key columns to categorical
        for col in ["IsCanceled", "IsRepeatedGuest"]:
            df[col] = df[col].astype("category")

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
            "ArrivalDateDayOfMonth"
        ])
        logger.info("Removed unimportant columns.")

        # Drop remaining NAs
        df = df.dropna()
        logger.info("Removed NAs.")

        logger.info("Data cleaning completed.")
        return df

    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise

def create_features(df: pd.DataFrame, encode: bool = False) -> pd.DataFrame:
    """
    Performs feature engineering on the cleaned hotel booking dataset.

    Args:
        df (pd.DataFrame): The cleaned dataframe to perform feature engineering on
        encode (bool): Whether or not to encode categorical variables using one-hot encoding. Necessary for Logistic Regression or GAM modeling.
    
    Returns: 
        pd.DataFrame: A dataframe containing the final engineered dataset
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

        if encode:
            df = pd.get_dummies(df, columns=["CustomerType", "DistributionChannel", "StayType", "DepositType"], drop_first=True)
            logger.info("Applied one-hot encoding.")

        logger.info("Feature engineering completed.")
        return df

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise



# import pandas as pd
# import numpy as np
# import calendar

# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Cleans the hotel dataset by handling types, missing values, duplicates, and filtering.
#     Does NOT include feature engineering (handled in create_features).
#     """
#     # Convert string columns to categorical
#     for col in df.select_dtypes(include="object").columns:
#         df[col] = df[col].astype("category")

#     # Remove duplicates based on conditions
#     df = df[~((df["IsCanceled"] == 1) & df.duplicated())]
#     df = df[~(df["CustomerType"].isin(["Group", "Transient-Party"]) & df.duplicated())]

#     # Convert key columns to categorical
#     for col in ["IsCanceled", "IsRepeatedGuest"]:
#         df[col] = df[col].astype("category")

#     # Map ArrivalDateMonth to numeric month
#     month_lookup = {month: i for i, month in enumerate(calendar.month_name) if month}
#     df["ArrivalDateMonth"] = df["ArrivalDateMonth"].map(month_lookup)

#     # Handle ordered Meal categories
#     # Replace 'Undefined' → 'SC' in Meal safely
#     meal_order = ["Undefined", "SC", "BB", "HB", "FB"]
#     df["Meal"] = df["Meal"].astype("object").replace("Undefined", "SC")
#     df["Meal"] = pd.Categorical(df["Meal"], categories=meal_order, ordered=True)

#     # Replace 'Undefined' with NA in select columns
#     df["DistributionChannel"] = df["DistributionChannel"].astype("object").replace("Undefined", pd.NA)
#     df["DistributionChannel"] = df["DistributionChannel"].astype("category")
#     df["MarketSegment"] = df["MarketSegment"].astype("object").replace("Undefined", pd.NA)
#     df["MarketSegment"] = df["MarketSegment"].astype("category")

#     # Filter out unrealistic values
#     df = df[(df["Adults"] < 5) & (df["Children"] < 10) & (df["Babies"] < 9)]
#     df = df[df["RequiredCarParkingSpaces"] <= (df["Adults"] + df["Children"])]
#     df = df[df["LeadTime"] < 709]
#     df = df[df["ADR"] >= 0]
#     df = df[df["ADR"] < 5400]

#     # Drop columns not needed
#     df = df.drop(columns=[
#         "ReservationStatus", "ReservationStatusDate",
#         "ReservedRoomType", "AssignedRoomType", "Country",
#         "Agent", "Company", "ADR", "TotalOfSpecialRequests"
#     ])

#     return df

# def create_features(df: pd.DataFrame, encode: bool = False) -> pd.DataFrame:
#     """
#     Performs feature engineering on the cleaned hotel booking dataset.
#     Adds:
#     - ArrivalDate
#     - TotalNights
#     - StayType
#     - StayDuration
#     - Binary flags for modeling (as 0/1)
#     """
#     # # Create ArrivalDate from year, month, day
#     # df["ArrivalDate"] = pd.to_datetime(
#     #     dict(
#     #         year=df["ArrivalDateYear"],
#     #         month=df["ArrivalDateMonth"],
#     #         day=df["ArrivalDateDayOfMonth"]
#     #     ),
#     #     errors="coerce"
#     # )
#     # df = df.drop(columns=["ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth"])

#     # TotalNights = weekday + weekend nights
#     df["TotalNights"] = df["StaysInWeekNights"] + df["StaysInWeekendNights"]

#     # StayType
#     df["StayType"] = pd.Categorical(
#         np.select(
#             [
#                 (df["StaysInWeekNights"] >= 1) & (df["StaysInWeekendNights"] == 0),
#                 (df["StaysInWeekNights"] == 0) & (df["StaysInWeekendNights"] >= 1)
#             ],
#             ["Weekday", "Weekend"],
#             default="Mixed"
#         ),
#         categories=["Weekday", "Weekend", "Mixed"]
#     )

#     df = df.drop(columns=["StaysInWeekendNights","StaysInWeekNights"])

#     # Binary features (converted to 0/1 int)
#     #df["IsWaitListed"] = (df["DaysInWaitingList"] != 0).astype(int)
#     #df["HasNotCancelledPreviousBooking"] = (df["PreviousBookingsNotCanceled"] != 0).astype(int)
#     #df["IsPreviousCancellationRisk"] = (~df["PreviousCancellations"].isin(range(0, 12))).astype(int)
#     df["HasBabies"] = (df["Babies"] != 0).astype(int)
#     df["HasMeals"] = (df["Meal"] != "SC").astype(int)
#     df["HasParking"] = (df["RequiredCarParkingSpaces"] != 0).astype(int)

#     # Drop raw columns now represented by binary flags
#     df = df.drop(columns=[
#         #"DaysInWaitingList", 
#         #"PreviousBookingsNotCanceled", 
#         #"PreviousCancellations",
#         "Babies", 
#         "RequiredCarParkingSpaces", 
#         #"Meal"
#     ])

#     # Re-map CustomerType into simplified buckets
#     df["CustomerType"] = df["CustomerType"].map({
#         "Contract": "Non-Transient",
#         "Group": "Non-Transient",
#         "Transient": "Transient",
#         "Transient-Party": "Transient-Party"
#     }).astype("category")

#     # Collapse DistributionChannel
#     df["DistributionChannel"] = df["DistributionChannel"].map({
#         "GDS": "TA/TO",
#         "TA/TO": "TA/TO",
#         "Corporate": "Corporate",
#         "Direct": "Direct"
#     }).astype("category")

#     # StayDuration buckets
#     # df["StayDuration"] = pd.cut(
#     #     df["TotalNights"],
#     #     bins=[0, 1, 4, 7, 14, float("inf")],
#     #     labels=["1 Night", "2-4 Nights", "5-7 Nights", "8-14 Nights", "15+ Nights"],
#     #     right=True,
#     #     ordered=True
#     # )

#     # Collapse StayDuration categories
#     # duration_map = {
#     #     "8-14 Nights": "8+ Nights",
#     #     "15+ Nights": "8+ Nights",
#     #     "5-7 Nights": "5-7 Nights",
#     #     "2-4 Nights": "2-4 Nights",
#     #     "1 Night": "1 Night"
#     # }
#     # df["StayDuration"] = df["StayDuration"].map(duration_map).astype(pd.CategoricalDtype(
#     #     categories=["1 Night", "2-4 Nights", "5-7 Nights", "8+ Nights"],
#     #     ordered=True
#     # ))

#     # Drop columns with high multicollinearity
#     df = df.drop(columns=["MarketSegment", "IsRepeatedGuest"])

#     if encode:
#         df = pd.get_dummies(df, columns=["CustomerType", "DistributionChannel", "StayType", "DepositType"], drop_first=True)

#     return df

