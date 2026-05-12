import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score

print("Loading and merging datasets...")

try:
    
    df1 = pd.read_csv("New Delhi January dataset hourly.csv")
    df1_clean = df1[['Timestamp(UTC)', 'US AQI']].copy()
    df1_clean.rename(columns={'Timestamp(UTC)': 'Timestamp', 'US AQI': 'AQI'}, inplace=True)

    df2 = pd.read_csv("delhi_aqi_log.csv")
    df2_clean = df2[['Timestamp', 'AQI']].copy()

    df_combined = pd.concat([df1_clean, df2_clean], ignore_index=True)

    
    df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'], utc=True, errors='coerce')
    df_combined['AQI'] = pd.to_numeric(df_combined['AQI'], errors='coerce')
    df_combined.dropna(subset=['Timestamp', 'AQI'], inplace=True)

    
    df_combined['Hour'] = df_combined['Timestamp'].dt.hour
    df_combined['DayOfWeek'] = df_combined['Timestamp'].dt.dayofweek
    
    
    X = df_combined[['Hour', 'DayOfWeek']]
    y = df_combined['AQI'] # Keeping it RAW for the A* Algorithm!
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    print("Training the XGBoost AI Regressor Model...")
    
    
    is_safe_train = (y_train <= 150)
    toxic_count = sum(~is_safe_train)
    safe_count = sum(is_safe_train)
    
    
    weight_multiplier = toxic_count / safe_count if safe_count > 0 else 1
    
    
    weights = np.where(is_safe_train, weight_multiplier, 1.0)

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,       
        max_depth=3,              
        subsample=0.8,            
        colsample_bytree=0.8,     
        min_child_weight=3,       
        random_state=42
    )
    
    
    model.fit(X_train, y_train, sample_weight=weights)
    
    
    raw_predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, raw_predictions))
    print(f"\n✅ Regression Engine Healthy. RMSE: {rmse:.2f}")
    
    
    print("\nGenerating Classification Metrics for Presentation...")
    
    
    y_test_binary = (y_test > 150).astype(int)
    predictions_binary = (raw_predictions > 150).astype(int)
    
    accuracy = accuracy_score(y_test_binary, predictions_binary)
    
    print("----------------------------------------")
    print(f"🚀 THRESHOLD CLASSIFICATION ACCURACY: {accuracy * 100:.2f}%")
    print("----------------------------------------")
    print(classification_report(y_test_binary, predictions_binary, target_names=["Safe (<=150)", "Toxic (>150)"]))
    
    
    model_output = "xgboost_aqi_model.pkl"
    joblib.dump(model, model_output)
    print(f"\n🚀 SUCCESS! Regression Model saved to {model_output} for A* Routing")

except Exception as e:
    print(f"❌ An error occurred: {e}")