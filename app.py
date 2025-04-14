import streamlit as st
import pandas as pd
import requests

FASTAPI_URL = "http://127.0.0.1:1000"

st.set_page_config(page_title="Hotel Booking Prediction Model", layout="centered")

st.sidebar.title("Select Input Mode")
mode = st.sidebar.radio("Choose input method:", ["Manual Entry", "Batch Upload"])

if mode == "Manual Entry":
    st.title("Hotel Booking Input Form")
    st.subheader("Fill in the booking features below:")

    lead_time = st.number_input("Lead Time (days)", min_value=0, value=100)
    adults = st.number_input("Number of Adults", min_value=0, value=2)
    children = st.number_input("Number of Children", min_value=0, value=0)
    previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
    previous_bookings = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)
    waiting_days = st.number_input("Days in Waiting List", min_value=0, value=0)
    total_nights = st.number_input("Total Nights", min_value=1, value=3)

    has_babies = st.selectbox("Has Babies?", [0, 1])
    has_meals = st.selectbox("Has Meals (not SC)?", [0, 1])
    has_parking = st.selectbox("Has Parking?", [0, 1])
    is_canceled = st.selectbox("Booking Canceled?", [0, 1])

    customer_type = st.selectbox("Customer Type", ["Transient", "Transient-Party", "Non-Transient"])
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
    distribution_channel = st.selectbox("Distribution Channel", ["Direct", "TA/TO", "Corporate"])
    stay_type = st.selectbox("Stay Type", ["Weekday", "Weekend", "Mixed"])

    if st.button("Make Single Prediction"):
        input_data = [{
            "LeadTime": lead_time,
            "Adults": adults,
            "Children": children,
            "PreviousCancellations": previous_cancellations,
            "PreviousBookingsNotCanceled": previous_bookings,
            "DaysInWaitingList": waiting_days,
            "CustomerType": customer_type,
            "DepositType": deposit_type,
            "DistributionChannel": distribution_channel,
            "StayType": stay_type,
            "TotalNights": total_nights,
            "HasBabies": has_babies,
            "HasMeals": has_meals,
            "HasParking": has_parking,
            "IsCanceled": is_canceled
        }]

        response = requests.post(f"{FASTAPI_URL}/predict", json=input_data)
        
        # The response is displayed on the Streamlit UI.
        if response.status_code == 200:
            prediction = response.json()["predictions"][0]
            if prediction == 0:
                msg = "Non-Cancellation"
            else:
                msg = "Cancellation"
            st.success(f"Prediction : {prediction}")
        else:
            st.error("Error fetching prediction. Check FastAPI logs.")

    elif mode == "Batch Upload":
        st.title("Batch Upload for Prediction")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:", df.head())

                if st.button("Make Batch Predictions"):
                    files = {"file": ("input.csv", uploaded_file.getvalue(), "text/csv")}
                    response = requests.post(f"{FASTAPI_URL}/predict_batch", files=files)

                    if response.status_code == 200:
                        predictions = response.json()["predictions"]
                        df["Predictions"] = predictions
                        st.subheader("Predictions:")
                        st.dataframe(df)
                    else:
                        st.error("Prediction failed.")
                        st.json(response.json())

            except Exception as e:
                st.error(f"Failed to process file: {e}")