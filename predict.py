import pandas as pd
import joblib
# LOAD SAVED FILES


model = joblib.load("energy_model.pkl")

scaler = joblib.load("scaler.pkl")

label_encoders = joblib.load("label_encoders.pkl")

print("\nENERGY CONSUMPTION PREDICTION\n")

# USER INPUT
month = int(input("Enter Month (1-12): "))

hour = int(input("Enter Hour (0-23): "))

day_of_week = input("Enter Day Of Week: ")

holiday = input("Holiday (Yes/No): ")

temperature = float(input("Enter Temperature: "))

humidity = float(input("Enter Humidity: "))

square_footage = float(input("Enter Square Footage: "))

occupancy = int(input("Enter Occupancy: "))

hvac_usage = input("HVAC Usage (On/Off): ")

lighting_usage = input("Lighting Usage (On/Off): ")

renewable_energy = float(input("Enter Renewable Energy: "))

# ENCODE INPUTS


day_of_week = label_encoders["DayOfWeek"].transform([day_of_week])[0]

holiday = label_encoders["Holiday"].transform([holiday])[0]

hvac_usage = label_encoders["HVACUsage"].transform([hvac_usage])[0]

lighting_usage = label_encoders["LightingUsage"].transform([lighting_usage])[0]
# CREATE DATAFRAME

input_data = pd.DataFrame([[
    
    month,
    hour,
    day_of_week,
    holiday,
    temperature,
    humidity,
    square_footage,
    occupancy,
    hvac_usage,
    lighting_usage,
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

# SCALE INPUT DATA


input_scaled = scaler.transform(input_data)

# PREDICTION


prediction = model.predict(input_scaled)

# OUTPUT


print("\n")

print(f"Predicted Energy Consumption: {round(prediction[0], 2)}")
