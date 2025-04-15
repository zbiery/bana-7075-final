import streamlit as st
import pandas as pd
import requests

FASTAPI_URL = "http://127.0.0.1:1000"

st.set_page_config(page_title="Hotel Booking Prediction Model", layout="centered")

st.sidebar.title("Model & Input Mode")
model_type = st.sidebar.selectbox("Select Model Type", ["GLM", "Tree", "Neural Network"])
mode = st.sidebar.radio("Choose input method:", ["Manual Entry", "Batch Upload"])

model_map = {
    "GLM": "glm",
    "Tree": "tree",
    "Neural Network": "nnet"
}
model_param = model_map[model_type]

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
            "HasParking": has_parking
        }]

        try:
            response = requests.post(f"{FASTAPI_URL}/predict?model={model_param}", json=input_data)
            if response.status_code == 200:
                result = response.json()
                prediction = result["predictions"][0]
                probability = result.get("probabilities", [None])[0]
                msg = "Cancellation" if prediction > 0.5 else "Non-Cancellation"

                if probability is not None:
                    st.success(f"Prediction: {prediction:.0f} ({msg}) — Probability: {probability:.2%}")
                else:
                    st.success(f"Prediction: {prediction:.0f} ({msg})")
            else:
                st.error("Error fetching prediction. Check FastAPI logs.")
                st.json(response.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

elif mode == "Batch Upload":
    st.title("Batch Upload for Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", df.head())

            if st.button("Make Batch Predictions"):
                files = {"file": ("input.csv", uploaded_file.getvalue(), "text/csv")}
                response = requests.post(f"{FASTAPI_URL}/predict_batch?model={model_param}", files=files)

                if response.status_code == 200:
                    predictions = response.json()["predictions"]
                    probabilities = response.json().get("probabilities", [None] * len(predictions))
                    df["Predictions"] = predictions
                    df["Probabilities"] = probabilities

                    def highlight_risk(row):
                        prob = row["Probabilities"]
                        if prob is not None and prob > 0.7:
                            return ['background-color: #ffcccc'] * len(row)
                        return [''] * len(row)

                    st.subheader("Predictions:")
                    st.dataframe(df.style.apply(highlight_risk, axis=1))
                else:
                    st.error("Prediction failed.")
                    st.json(response.json())

        except Exception as e:
            st.error(f"Failed to process file: {e}")
