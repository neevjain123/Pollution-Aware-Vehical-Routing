import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print("Engineering new features and generating Performance Matrix...")

# 1. Load the hybrid dataset
df1 = pd.read_csv("New Delhi January dataset hourly.csv")[['Timestamp(UTC)', 'US AQI']].rename(columns={'Timestamp(UTC)': 'Timestamp', 'US AQI': 'AQI'})
df2 = pd.read_csv("delhi_aqi_log.csv")[['Timestamp', 'AQI']]
df = pd.concat([df1, df2], ignore_index=True)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
df.dropna(inplace=True)

# 2. FEATURE ENGINEERING (The Fix)
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int) # 1 if Saturday/Sunday, 0 if Weekday

# Define peak traffic hours (8 AM - 11 AM, and 5 PM - 9 PM)
df['Traffic_Peak'] = df['Hour'].isin([8, 9, 10, 11, 17, 18, 19, 20, 21]).astype(int)

# 3. Binary Threshold: AQI > 150 is Toxic (1), else Safe (0)
df['Is_Toxic'] = (df['AQI'] > 150).astype(int)

# 4. Count the classes to handle imbalance
num_safe = len(df[df['Is_Toxic'] == 0])
num_toxic = len(df[df['Is_Toxic'] == 1])
scale_weight = num_safe / num_toxic if num_toxic > 0 else 1

X = df[['Hour', 'DayOfWeek', 'Is_Weekend', 'Traffic_Peak']]
y = df['Is_Toxic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. The Tuned Classifier with Class Balancing
classifier = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=3, 
    subsample=0.8,
    scale_pos_weight=scale_weight, # Forces the AI to pay equal attention to Safe and Toxic routes
    random_state=42
)
classifier.fit(X_train, y_train)

# 6. Generate Predictions and Plot
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Safe', 'Predicted Toxic'], 
            yticklabels=['Actually Safe', 'Actually Toxic'],
            annot_kws={"size": 14})

plt.title('Model Performance: Confusion Matrix (Engineered Features)')
plt.ylabel('Real-World Condition (Ground Truth)')
plt.xlabel('XGBoost AI Prediction')
plt.tight_layout()

plt.savefig("confusion_matrix.png")
print("✅ Saved NEW confusion_matrix.png in your folder!")