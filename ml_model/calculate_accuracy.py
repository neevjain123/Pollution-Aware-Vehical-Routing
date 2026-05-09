import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Calculating Classification Accuracy...")

# 1. Load the same hybrid dataset
df1 = pd.read_csv("New Delhi January dataset hourly.csv")[['Timestamp(UTC)', 'US AQI']].rename(columns={'Timestamp(UTC)': 'Timestamp', 'US AQI': 'AQI'})
df2 = pd.read_csv("delhi_aqi_log.csv")[['Timestamp', 'AQI']]
df = pd.concat([df1, df2], ignore_index=True)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
df.dropna(inplace=True)
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

# 2. Binary Threshold: AQI > 150 is Toxic (1), else Safe (0)
df['Is_Toxic'] = (df['AQI'] > 150).astype(int)

X = df[['Hour', 'DayOfWeek']]
y = df['Is_Toxic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Classifier
classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
classifier.fit(X_train, y_train)

# 4. Generate Predictions and Calculate Accuracy
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 40)
print(f"🚀 FINAL MODEL ACCURACY: {accuracy * 100:.2f}%")
print("-" * 40)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe (<=150)', 'Toxic (>150)']))