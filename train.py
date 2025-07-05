import pandas as pd
import os
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('winequality-red-selected-missing.csv')

# Fill missing values, could try knn
df_cleaned = df.fillna(df.mean())

# Create binary target column
df_cleaned['good_quality'] = (df_cleaned['quality'] >= 7).astype(int)

# Define features and target
X = df_cleaned.drop(['quality', 'good_quality'], axis=1)
y = df_cleaned['good_quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
report_xgb = classification_report(y_test, y_pred_xgb)

print(f"XGBoost Accuracy: {accuracy_xgb}")
print("XGBoost Classification Report:")
print(report_xgb)

# Save the model
try: 
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_model, "models/xgb_model.joblib")
    print("\nSuccess: Model saved!")
except Exception as e: 
    print(f"Failed to save model: {e}")
