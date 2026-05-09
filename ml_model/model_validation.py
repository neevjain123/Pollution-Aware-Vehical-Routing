import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load Model and Data
model = joblib.load("xgboost_aqi_model.pkl")
df1 = pd.read_csv("New Delhi January dataset hourly.csv")[['Timestamp(UTC)', 'US AQI']].rename(columns={'Timestamp(UTC)': 'Timestamp', 'US AQI': 'AQI'})
df2 = pd.read_csv("delhi_aqi_log.csv")[['Timestamp', 'AQI']]
df = pd.concat([df1, df2], ignore_index=True)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
df.dropna(inplace=True)
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

X = df[['Hour', 'DayOfWeek']]
y = df['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Generate Predictions
y_pred = model.predict(X_test)

# 3. Calculate Performance Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- VALIDATION METRICS ---")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"R-Squared (R2) Score: {r2:.2f}")

# 4. Generate Plot 1: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title("Actual vs. Predicted AQI (Validation Set)")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.grid(True)
plt.savefig("actual_vs_predicted.png")
print("Saved actual_vs_predicted.png")

# 5. Generate Plot 2: Feature Importance
plt.figure(figsize=(6, 4))
importances = model.feature_importances_
sns.barplot(x=['Hour', 'DayOfWeek'], y=importances, palette='viridis')
plt.title("XGBoost Feature Importance")
plt.ylabel("Relative Importance")
plt.savefig("feature_importance.png")
print("Saved feature_importance.png")