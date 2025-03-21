# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("KAG_energydata_complete.csv")

# Drop unnecessary columns
columns_to_drop = [0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
df_new = df.drop(df.columns[columns_to_drop], axis=1)

# Convert date column to datetime
df_new['date'] = pd.to_datetime(df_new['date'])

# Split the data into training and testing sets
train, test = train_test_split(df_new, test_size=0.25, random_state=40)

# Separate features and target variable
col_temp = ["T1"]
col_hum = ["RH_1"]
col_weather = ["T_out", "RH_out", "Press_mm_hg", "Windspeed"]
col_target = ["Appliances"]

feature_vars = train[col_temp + col_hum + col_weather]
target_vars = train[col_target]

# Scale the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Save preprocessed data to CSV
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Data preprocessing completed. Preprocessed data saved in 'data/processed/'.")
