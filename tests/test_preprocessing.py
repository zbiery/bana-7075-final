import pytest
import pandas as pd
from src.preprocessing import clean_data, create_features

@pytest.fixture
def raw_sample_df():
    return pd.DataFrame({
        "IsCanceled": [0, 1, 1, 0],
        "CustomerType": ["Transient", "Group", "Transient-Party", "Transient"],
        "IsRepeatedGuest": [0, 0, 1, 0],
        "Meal": ["SC", "BB", "Undefined", "FB"],
        "DistributionChannel": ["Direct", "TA/TO", "Undefined", "Corporate"],
        "MarketSegment": ["Online TA", "Offline TA/TO", "Undefined", "Online TA"],
        "Adults": [2, 4, 6, 1],
        "Children": [0, 1, 2, 0],
        "Babies": [0, 1, 0, 0],
        "RequiredCarParkingSpaces": [1, 0, 2, 1],
        "LeadTime": [100, 300, 800, 50],
        "ADR": [100.5, 200.0, -5, 130],
        "TotalOfSpecialRequests": [0, 1, 0, 2],
        "ReservationStatus": ["Check-Out"]*4,
        "ReservationStatusDate": ["2021-01-01"]*4,
        "ReservedRoomType": ["A"]*4,
        "AssignedRoomType": ["A"]*4,
        "Country": ["PRT"]*4,
        "Agent": [1]*4,
        "Company": [1]*4,
        "ArrivalDateYear": [2017]*4,
        "ArrivalDateMonth": ["July", "August", "September", "June"],
        "ArrivalDateWeekNumber": [27]*4,
        "ArrivalDateDayOfMonth": [15, 16, 17, 18],
        "StaysInWeekNights": [2, 0, 3, 0],
        "StaysInWeekendNights": [0, 2, 0, 3],
        "DepositType": ["No Deposit", "Refundable", "No Deposit", "Non Refund"]
    })

def test_clean_data_removes_unrealistic_values(raw_sample_df):
    cleaned = clean_data(raw_sample_df)
    assert isinstance(cleaned, pd.DataFrame)
    assert "ADR" not in cleaned.columns
    assert all(cleaned["Adults"] < 5)
    assert cleaned.isnull().sum().sum() == 0 

def test_create_features_adds_expected_columns(raw_sample_df):
    df_cleaned = clean_data(raw_sample_df)
    df_feat = create_features(df_cleaned)
    assert "TotalNights" in df_feat.columns
    assert "HasBabies" in df_feat.columns
    assert "HasMeals" in df_feat.columns
    assert "StayType" in df_feat.columns
    assert df_feat["HasMeals"].isin([0, 1]).all()

def test_create_features_encoding_flag(raw_sample_df):
    df_cleaned = clean_data(raw_sample_df)
    df_encoded = create_features(df_cleaned, encode=True)
    encoded_cols = [col for col in df_encoded.columns if "CustomerType_" in col or "StayType_" in col]
    assert any(encoded_cols)
