import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv("heart.csv")   # <-- Make sure heart.csv is in project folder

print("Shape:", df.shape)
print(df.head())

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost
xgb_model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]))

# Save model
joblib.dump(xgb_model, "best_heart_model.pkl")
print("✅ Model saved as best_heart_model.pkl")
