import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Splitting data into Train, Validation, and Test sets...")

# 1. Load the hybrid dataset
df1 = pd.read_csv("New Delhi January dataset hourly.csv")[['Timestamp(UTC)', 'US AQI']].rename(columns={'Timestamp(UTC)': 'Timestamp', 'US AQI': 'AQI'})
df2 = pd.read_csv("delhi_aqi_log.csv")[['Timestamp', 'AQI']]
df = pd.concat([df1, df2], ignore_index=True)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
df.dropna(inplace=True)
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

X = df[['Hour', 'DayOfWeek']]
y = df['AQI']

# 2. FIRST SPLIT: 70% Training, 30% Temporary (To be split again)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)

# 3. SECOND SPLIT: Divide the 30% Temp equally into 15% Validation and 15% Testing
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"✅ Data Split Complete:")
print(f"   - Training Set: {len(X_train)} rows (70%)")
print(f"   - Validation Set: {len(X_val)} rows (15%)")
print(f"   - Testing Set: {len(X_test)} rows (15%)")

# 4. Train XGBoost and track performance on BOTH Train and Validation sets simultaneously
print("Training XGBoost and monitoring for Overfitting...")
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,       # Reduced from 0.1 (Makes the AI learn slower and steadier)
    max_depth=3,              # Reduced from 5 (Stops the AI from thinking too deeply and memorizing)
    subsample=0.8,            # Uses only 80% of data per tree (Forces it to generalize)
    colsample_bytree=0.8,     # Adds randomness to the features
    min_child_weight=3,       # Adds a mathematical penalty for memorizing
    random_state=42
)

# The eval_set parameter tells XGBoost to evaluate these datasets after every single tree is built
evalset = [(X_train, y_train), (X_val, y_val)]
model.fit(X_train, y_train, eval_set=evalset, verbose=False)

# 5. Final Evaluation on the unseen Test Set
final_predictions = model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(f"🚀 Final Test RMSE (on completely unseen data): {final_rmse:.2f}")

# 6. Extract the training history to plot the Learning Curve
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

# 7. Plot the Overfitting Check (Learning Curve)
plt.figure(figsize=(9, 6))
plt.plot(x_axis, results['validation_0']['rmse'], label='Training Error (RMSE)', lw=2, color='blue')
plt.plot(x_axis, results['validation_1']['rmse'], label='Validation Error (RMSE)', lw=2, color='orange')
plt.legend(loc="upper right", fontsize=12)
plt.xlabel('Number of Iterations (Trees Built)', fontsize=12)
plt.ylabel('RMSE (Error Rate)', fontsize=12)
plt.title('XGBoost Learning Curve: Overfitting Check', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('learning_curve.png')
print("✅ Saved 'learning_curve.png' in your folder! Add this directly to your presentation.")