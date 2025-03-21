# model_training.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Load preprocessed data
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

# Separate features and target variable
X_train = train[["T1", "RH_1", "T_out", "RH_out", "Press_mm_hg", "Windspeed"]]
y_train = train["Appliances"]
X_test = test[["T1", "RH_1", "T_out", "RH_out", "Press_mm_hg", "Windspeed"]]
y_test = test["Appliances"]

# Train Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Random Forest Regressor Results:")
print(f"R² Score: {r2}")
print(f"RMSE: {rmse}")

# Save the trained model
import joblib
joblib.dump(rf, "models/random_forest_model.pkl")

print("Model training completed. Model saved in 'models/random_forest_model.pkl'.")
