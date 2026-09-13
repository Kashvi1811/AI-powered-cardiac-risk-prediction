"""
evaluate_threshold.py
---------------------
Evaluates the impact of different decision thresholds on the final deployed
Logistic Regression model. Run this from the project root directory.

Results proved that threshold tuning is vital for medical screening:
  - Threshold 0.50 (default): Recall = 0.47 (misses 53% of fatal cases)
  - Threshold 0.40: Recall = 0.84 (catches 84% of fatal cases)
  - Threshold 0.35 (deployed via F2 OOF): Recall = 0.95 (catches 95% of cases)

Usage:
    python evaluate_threshold.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import classification_report
import model_utils  # required for pickle to resolve the pipeline's transform step

# Load dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
if "time" in df.columns:
    df = df.drop(columns=["time"])

X_real = df.drop(columns=["DEATH_EVENT"])
y_real = df["DEATH_EVENT"]

# Use same seed as final model export
X_train, X_test, y_train, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

# Load the deployed model and threshold
artifact = joblib.load("logisticregression_pipeline.pkl")
if isinstance(artifact, dict):
    model = artifact["pipeline"]
    deployed_threshold = artifact["threshold"]
else:
    model = artifact
    deployed_threshold = 0.4 # fallback

probs = model.predict_proba(X_test)[:, 1]

thresholds = [0.5, 0.4, 0.35, 0.3, 0.2]
for t in thresholds:
    print(f"\n{'='*40}")
    print(f"Threshold = {t}{'  <-- CURRENTLY DEPLOYED (F2 Tuned)' if t == deployed_threshold else ''}")
    print(f"{'='*40}")
    preds = (probs >= t).astype(int)
    print(classification_report(y_test, preds, target_names=["Survived", "At Risk"]))
