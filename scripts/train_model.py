import joblib
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pathlib import Path

from data_prep import load_and_clean_data

# Load cleaned data
df = load_and_clean_data("data/raw/retail_sales.csv")

# Train-Test split (last 6 months for testing)
train = df.iloc[:-6]
test = df.iloc[-6:]

# Model training
model = ExponentialSmoothing(train["Sales"], trend="add", seasonal="add", seasonal_periods=12)
fitted_model = model.fit()

# Forecast
preds = fitted_model.forecast(len(test))
mse = mean_squared_error(test["Sales"], preds)
print("Test MSE:", mse)

# Ensure the folder exists
Path("../models").mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(fitted_model, "../models/final_model.pkl")
