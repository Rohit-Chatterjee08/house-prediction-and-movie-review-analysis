# predict_house_price_kaggle.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. Load and Prepare the Data ---
# We load the data from the CSV file into a pandas DataFrame.
print("--- Loading and Preparing Data ---")
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please download it from Kaggle and place it in the same directory.")
    exit()

# For simplicity, we'll use only a few important numerical features.
# A real project would involve much more feature engineering.
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'
all_cols = features + [target]

# Drop rows where any of our selected columns have missing values.
df_clean = df[all_cols].dropna()

# Separate features (X) from the target variable (y)
X = df_clean[features]
y = df_clean[target]

print(f"Loaded {len(df_clean)} rows of clean data.")
print("Features we will use:", features)
print("-" * 100)

# --- 2. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Create and Train the Model ---
print("\n--- Training Model ---")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")
print("-" * 100)

# --- 4. Evaluate the Model ---
print("\n--- Evaluating Model ---")
y_pred = model.predict(X_test)
# We use Root Mean Squared Error (RMSE) here, which is common for price prediction.
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model Performance (Root Mean Squared Error): ${rmse:,.2f}")
print("This is the typical error margin of our model's price predictions.")
print("-" * 100)

# --- 5. Use the Model for a New Prediction ---
print("\n--- Making a New Prediction ---")
# Let's predict the price for a house with:
# - 1800 sq. ft. of living area
# - 3 bedrooms
# - 2 full bathrooms
new_house_data = pd.DataFrame([[1800, 3, 2]], columns=features)
predicted_price = model.predict(new_house_data)

print(f"Predicted price for the new house: ${predicted_price[0]:,.2f}")