import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib

# LOAD DATASET


print("Loading Dataset...")

df = pd.read_csv("Energy_consumption_dataset.csv")

print("\nDataset Loaded Successfully!")
print(df.head())

# ENCODE CATEGORICAL COLUMNS


label_encoders = {}

categorical_columns = [
    "DayOfWeek",
    "Holiday",
    "HVACUsage",
    "LightingUsage"
]

for column in categorical_columns:
    
    le = LabelEncoder()
    
    df[column] = le.fit_transform(df[column])
    
    label_encoders[column] = le

print("\nCategorical Columns Encoded!")

# FEATURES AND TARGET


X = df.drop("EnergyConsumption", axis=1)

y = df["EnergyConsumption"]

# FEATURE SCALING


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed!")

# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nData Split Completed!")

# MODEL CREATION


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("\nTraining Model...")

# MODEL TRAINING


model.fit(X_train, y_train)

print("\nModel Training Completed!")

# PREDICTION


y_pred = model.predict(X_test)

# EVALUATION


mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"MAE  : {mae}")

print(f"MSE  : {mse}")

print(f"RMSE : {rmse}")

print(f"R2 Score : {r2}")

# SAVE MODEL


joblib.dump(model, "energy_model.pkl")

joblib.dump(scaler, "scaler.pkl")

joblib.dump(label_encoders, "label_encoders.pkl")

print("\nModel Saved Successfully!")

print("\nFiles Created:")
print("1. energy_model.pkl")
print("2. scaler.pkl")
print("3. label_encoders.pkl")