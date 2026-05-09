import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Loading and merging datasets...")

try:
    # 1. Load Dataset 1: The January Historical Data
    # We only take the columns we need and rename them so they match the second dataset
    df1 = pd.read_csv("New Delhi January dataset hourly.csv")
    df1_clean = df1[['Timestamp(UTC)', 'US AQI']].copy()
    df1_clean.rename(columns={'Timestamp(UTC)': 'Timestamp', 'US AQI': 'AQI'}, inplace=True)

    # 2. Load Dataset 2: Your Custom Scraped Data
    df2 = pd.read_csv("delhi_aqi_log.csv")
    df2_clean = df2[['Timestamp', 'AQI']].copy()

    # 3. Combine both datasets into one massive DataFrame
    df_combined = pd.concat([df1_clean, df2_clean], ignore_index=True)

    # 4. Clean the combined data
    # Convert Timestamp to actual datetime objects (ignoring timezone issues)
    df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'], utc=True, errors='coerce')
    df_combined['AQI'] = pd.to_numeric(df_combined['AQI'], errors='coerce')
    
    # Drop any rows where the data is missing or corrupted
    df_combined.dropna(subset=['Timestamp', 'AQI'], inplace=True)

    print(f"Data combined successfully! Total training rows: {len(df_combined)}")

    # 5. Feature Engineering (Extracting time patterns for the AI)
    df_combined['Hour'] = df_combined['Timestamp'].dt.hour
    df_combined['DayOfWeek'] = df_combined['Timestamp'].dt.dayofweek
    
    # 6. Define Features (X) and Target (y)
    X = df_combined[['Hour', 'DayOfWeek']]
    y = df_combined['AQI']
    
    # 7. Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 8. Initialize and Train XGBoost Model
    print("Training the XGBoost AI Model...")
    model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,       # Reduced from 0.1 (Makes the AI learn slower and steadier)
    max_depth=3,              # Reduced from 5 (Stops the AI from thinking too deeply and memorizing)
    subsample=0.8,            # Uses only 80% of data per tree (Forces it to generalize)
    colsample_bytree=0.8,     # Adds randomness to the features
    min_child_weight=3,       # Adds a mathematical penalty for memorizing
    random_state=42
    )
    model.fit(X_train, y_train)
    
    # 9. Evaluate the Model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"✅ Model Evaluation Complete. RMSE: {rmse:.2f}")
    
    # 10. Save the Model for the Backend to use
    model_output = "xgboost_aqi_model.pkl"
    joblib.dump(model, model_output)
    print(f"🚀 SUCCESS! Model saved to {model_output}")

except FileNotFoundError as e:
    print(f"❌ File Error: Could not find one of the CSV files.")
    print("Ensure BOTH 'New Delhi January dataset hourly.csv' and 'delhi_aqi_log.csv' are in the ml_model folder.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")