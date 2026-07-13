import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL FILES

model = joblib.load("energy_model.pkl")

scaler = joblib.load("scaler.pkl")

label_encoders = joblib.load("label_encoders.pkl")

# TITLE

st.title("⚡ Energy Consumption Prediction App")

st.write("Enter details to predict energy consumption.")

# USER INPUTS

month = st.slider("Month", 1, 12, 1)

hour = st.slider("Hour", 0, 23, 12)

day_of_week = st.selectbox(
    "Day Of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

holiday = st.selectbox(
    "Holiday",
    ["Yes", "No"]
)

temperature = st.number_input("Temperature")

humidity = st.number_input("Humidity")

square_footage = st.number_input("Square Footage")

occupancy = st.number_input("Occupancy", step=1)

hvac_usage = st.selectbox(
    "HVAC Usage",
    ["On", "Off"]
)

lighting_usage = st.selectbox(
    "Lighting Usage",
    ["On", "Off"]
)

renewable_energy = st.number_input("Renewable Energy")

# ENCODE INPUTS


day_of_week_encoded = label_encoders["DayOfWeek"].transform([day_of_week])[0]

holiday_encoded = label_encoders["Holiday"].transform([holiday])[0]

hvac_encoded = label_encoders["HVACUsage"].transform([hvac_usage])[0]

lighting_encoded = label_encoders["LightingUsage"].transform([lighting_usage])[0]

# PREDICT BUTTON

if st.button("Predict Energy Consumption"):

    input_data = pd.DataFrame([[
        
        month,
        hour,
        day_of_week_encoded,
        holiday_encoded,
        temperature,
        humidity,
        square_footage,
        occupancy,
        hvac_encoded,
        lighting_encoded,
        renewable_energy

    ]], columns=[

        "Month",
        "Hour",
        "DayOfWeek",
        "Holiday",
        "Temperature",
        "Humidity",
        "SquareFootage",
        "Occupancy",
        "HVACUsage",
        "LightingUsage",
        "RenewableEnergy"

    ])

    # SCALE INPUT

    input_scaled = scaler.transform(input_data)

    # PREDICTION

    prediction = model.predict(input_scaled)

    # OUTPUT
    

    st.success(
        f"Predicted Energy Consumption: {round(prediction[0], 2)} kWh"
    )