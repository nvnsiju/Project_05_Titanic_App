
import streamlit as st
import pandas as pd
import joblib
import os

FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger details to estimate their survival probability."
)

st.divider()

# Passenger inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.radio(
    "Gender",
    ["Female", "Male"],
    horizontal=True
)

age = st.slider(
    "Passenger Age",
    min_value=1,
    max_value=80,
    value=30
)

fare = st.slider(
    "Ticket Fare",
    min_value=0.0,
    max_value=500.0,
    value=30.0
)

# Prediction
if st.button("Predict Survival", type="primary"):

    try:
        model = load_model()

        sex_value = 1 if sex == "Female" else 0

        input_data = pd.DataFrame(
            [[pclass, sex_value, age, fare]],
            columns=FEATURES
        )

        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        st.metric(
            "Estimated Survival Probability",
            f"{probability:.1%}"
        )

        if prediction == 1:
            st.success("The model predicts that the passenger survived.")
        else:
            st.warning("The model predicts that the passenger did not survive.")

    except FileNotFoundError:
        st.error("Model file is missing. Please add model.joblib.")

    except Exception as e:
        st.error("Unable to make prediction. Please check your inputs.")

st.divider()

st.caption(
    "Educational machine learning project using the Titanic dataset. "
    "Predictions are historical model estimates, not real-world advice."
)
