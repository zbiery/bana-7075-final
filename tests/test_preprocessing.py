import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.preprocessing import encode_data, split_data, scale_data
from src.preprocessing import clean_data, create_features

@pytest.fixture
def raw_sample_df():
    return pd.DataFrame({
        "IsCanceled": [0, 1, 1, 0, 0, 1, 0, 1],  
        "CustomerType": ["Transient", "Group", "Transient-Party", "Transient"] * 2,
        "IsRepeatedGuest": [0, 0, 1, 0, 1, 0, 1, 0],
        "Meal": ["SC", "BB", "Undefined", "FB", "SC", "BB", "SC", "FB"],
        "DistributionChannel": ["Direct", "TA/TO", "Undefined", "Corporate", "TA/TO", "Direct", "TA/TO", "Direct"],
        "MarketSegment": ["Online TA", "Offline TA/TO", "Undefined", "Online TA", "Online TA", "Offline TA/TO", "Offline TA/TO", "Online TA"],
        "Adults": [2, 4, 6, 1, 2, 3, 4, 1],
        "Children": [0, 1, 2, 4, 0, 2, 1, 0],
        "Babies": [0, 1, 0, 0, 1, 0, 0, 0],
        "BookingChanges": [0, 0, 2, 1, 1, 0, 1, 0],
        "RequiredCarParkingSpaces": [1, 0, 2, 1, 0, 1, 1, 0],
        "LeadTime": [100, 300, 800, 50, 200, 150, 400, 600],
        "ADR": [100.5, 200.0, 150.0, 130.0, 120.0, 90.0, 80.0, 110.0],
        "TotalOfSpecialRequests": [0, 1, 0, 2, 1, 0, 2, 1],
        "ReservationStatus": ["Check-Out"]*8,
        "ReservationStatusDate": ["2021-01-01"]*8,
        "ReservedRoomType": ["A"]*8,
        "AssignedRoomType": ["A"]*8,
        "Country": ["PRT"]*8,
        "Agent": [1]*8,
        "Company": [1]*8,
        "ArrivalDateYear": [2017]*8,
        "ArrivalDateMonth": ["July", "August", "September", "June"] * 2,
        "ArrivalDateWeekNumber": [27]*8,
        "ArrivalDateDayOfMonth": [15, 16, 17, 18, 19, 20, 21, 22],
        "StaysInWeekNights": [2, 0, 3, 0, 1, 2, 3, 1],
        "StaysInWeekendNights": [0, 2, 0, 3, 1, 1, 2, 0],
        "DepositType": ["No Deposit", "Refundable", "No Deposit", "Non Refund", "No Deposit", "Refundable", "Non Refund", "No Deposit"]
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

@pytest.fixture
def sample_df(raw_sample_df):
    df_cleaned = clean_data(raw_sample_df)
    df_engineered = create_features(df_cleaned)
    return df_engineered

def test_encode_data_columns(sample_df):
    df_encoded = encode_data(sample_df)
    expected_columns = {
        'LeadTime', 'Adults', 'Children',
        'TotalNights', 'HasBabies', 'HasMeals', 'HasParking',
        'CustomerType_Transient', 'CustomerType_Transient-Party',
        'DistributionChannel_Direct', 'DistributionChannel_TA/TO',
        'StayType_Weekend', 'StayType_Mixed',
        'DepositType_Non Refund', 'DepositType_Refundable'
    }

    assert expected_columns.issubset(set(df_encoded.columns))

    assert df_encoded.shape[0] == sample_df.shape[0]

def test_split_data_shapes(sample_df):
    x_train, x_test, y_train, y_test = split_data(sample_df, target="IsCanceled", test_size=0.2, random_state=42)

    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    assert len(x_train) + len(x_test) == len(sample_df)
    assert len(y_train) + len(y_test) == len(sample_df)


@pytest.mark.parametrize("method", ["min-max", "z-score"])
def test_scale_data_only_selected_columns_scaled(sample_df, method):
    df_encoded = encode_data(sample_df)
    scaled_df= scale_data(df_encoded, how=method)

    excluded = ["HasBabies", "HasMeals", "HasParking", "IsCanceled"]
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
    scaled_cols = [col for col in numeric_cols if col not in excluded]

    assert scaled_df.shape == df_encoded.shape

    for col in excluded:
        assert np.allclose(scaled_df[col], df_encoded[col]), f"{col} was modified"

    for col in scaled_cols:
        assert not np.allclose(scaled_df[col], df_encoded[col]), f"{col} was not scaled"

    if method == "min-max":
        for col in scaled_cols:
            assert scaled_df[col].min() >= 0
            assert scaled_df[col].max() <= 1

