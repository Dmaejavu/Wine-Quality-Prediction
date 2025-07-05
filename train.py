import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('winequality-red-selected-missing.csv')

# Fill missing values with column means
df_cleaned = df.fillna(df.mean())

df_cleaned['good_quality'] = (df_cleaned['quality'] >= 7).astype(int)

# target quality
X = df_cleaned.drop(['quality', 'good_quality'], axis=1)
y = df_cleaned['good_quality']

# handle imbalance
neg, pos = np.bincount(y)
scale_pos_weight = neg / pos
print(f"Class 0 (bad): {neg}, Class 1 (good): {pos}")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train XGBoost model
xgb_model = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)
xgb_model.fit(X_train, y_train)

# predictions
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nXGBoost Accuracy: {accuracy:.4f}")
print("XGBoost Classification Report:")
print(report)

# saving model
try:
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_model, "models/xgb_model.joblib")
    print("\nSuccess: Model saved to models/xgb_model.joblib")
except Exception as e:
    print(f"\nFailed to save model: {e}")
