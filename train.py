import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib
from model_utils import log_transform_func  # shared reference for stable pickling

# SDV imports for synthetic data
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdmetrics.reports.single_table import QualityReport
import warnings
warnings.filterwarnings('ignore')

print("Loading real dataset...")
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# ---------------------------------------------------------
# 1. FIX TARGET LEAKAGE
# ---------------------------------------------------------
if "time" in df.columns:
    df = df.drop(columns=["time"])
    print("Dropped 'time' feature to prevent target leakage.")

X_real = df.drop(columns=["DEATH_EVENT"])
y_real = df["DEATH_EVENT"]

# ---------------------------------------------------------
# 2. PREPROCESSING PIPELINE
# ---------------------------------------------------------
standard_scaler = ["age", "ejection_fraction", "serum_sodium"]
robust_scaler = ["platelets"]
log_robust = ["serum_creatinine", "creatinine_phosphokinase"]

log_transform = FunctionTransformer(log_transform_func)

preprocessor = ColumnTransformer(
    transformers=[
        ("standard_scaling", StandardScaler(), standard_scaler),
        ("robust_scaling", RobustScaler(), robust_scaler),
        ("log_robust", Pipeline([
            ("log_transform", log_transform),
            ("robust", RobustScaler())
        ]), log_robust)
    ],
    remainder="passthrough"
)

# ---------------------------------------------------------
# 3. MULTI-SEED ROBUSTNESS TESTING (Monte Carlo CV)
# ---------------------------------------------------------
n_iterations = 10
print(f"\n--- RUNNING MULTI-SEED ROBUSTNESS TESTING ({n_iterations} Iterations) ---")
print("Generating synthetic data and training XGBoost on multiple random train/test splits...")

roc_auc_scores = []

# Extract metadata once
print("Extracting metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(pd.concat([X_real, y_real], axis=1))
for col in ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT"]:
    if col in metadata.columns:
        metadata.update_column(column_name=col, sdtype='categorical')

for seed in range(n_iterations):
    # 1. Split data with a NEW seed
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.2, random_state=seed, stratify=y_real
    )
    df_train_real = pd.concat([X_train_real, y_train_real], axis=1)
    
    # 2. Train Synthetic Generator & Sample
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df_train_real)
    synthetic_data = synthesizer.sample(num_rows=2000) # Used 2000 for faster robust testing loop
    
    X_train_synth = synthetic_data.drop(columns=["DEATH_EVENT"])
    y_train_synth = synthetic_data["DEATH_EVENT"].astype(int)
    
    # 3. Train XGBoost strictly on synthetic data
    pipeline = Pipeline([
        ("Preprocessor", preprocessor),
        ("Model", XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.01,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
            eval_metric="logloss", random_state=42 # Fix model internal randomness
        ))
    ])
    pipeline.fit(X_train_synth, y_train_synth)
    
    # 4. Evaluate on the NEW X_test
    y_pred_prob = pipeline.predict_proba(X_test_real)[:, 1]
    score = roc_auc_score(y_test_real, y_pred_prob)
    roc_auc_scores.append(score)
    print(f"Iteration {seed + 1} (split seed={seed}): ROC-AUC = {score:.3f}")

mean_auc = np.mean(roc_auc_scores)
std_auc = np.std(roc_auc_scores)
ci_lower = mean_auc - 1.96 * std_auc
ci_upper = mean_auc + 1.96 * std_auc

print("\n--- ROBUSTNESS RESULTS ---")
print(f"Mean ROC-AUC across {n_iterations} random splits: {mean_auc:.3f}")
print(f"Standard Deviation: {std_auc:.3f}")
print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")

# ---------------------------------------------------------
# 4. FINAL MODEL EXPORT (Logistic Regression on Combined Data)
# ---------------------------------------------------------
print("\n--- TRAINING FINAL EXPORT MODEL (seed=42) ---")
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)
df_train_real = pd.concat([X_train_real, y_train_real], axis=1)

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df_train_real)
synthetic_data = synthesizer.sample(num_rows=5000) # Use 5000 for the final export

print("Evaluating final synthetic data quality...")
report = QualityReport()
report.generate(df_train_real, synthetic_data, metadata.to_dict())
print(f"Overall Quality Score: {report.get_score()*100:.2f}%")

X_train_synth = synthetic_data.drop(columns=["DEATH_EVENT"])
y_train_synth = synthetic_data["DEATH_EVENT"].astype(int)

# Combine Real and Synthetic Data
X_train_comb = pd.concat([X_train_real, X_train_synth], ignore_index=True)
y_train_comb = pd.concat([y_train_real, y_train_synth], ignore_index=True)

final_pipeline = Pipeline([
    ("Preprocessor", preprocessor),
    ("Model", LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ))
])

final_pipeline.fit(X_train_comb, y_train_comb)

# ---------------------------------------------------------
# 5. THRESHOLD TUNING (on training data only — no test leakage)
# Using cross_val_predict to get out-of-fold probabilities
# then maximizing F2-score (weights recall 2x over precision,
# appropriate for medical screening where missing a fatal
# case is far worse than a false alarm).
# ---------------------------------------------------------
print("\n--- THRESHOLD TUNING (Real Training Data OOF Only — no test leakage) ---")
print("Running 5-fold CV on real training data to calibrate threshold to real patient distribution...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Use real training data only for OOF — synthetic data has a slightly
# different probability distribution, so tuning on it produces
# thresholds that don't transfer to real patients.
oof_probs = cross_val_predict(
    final_pipeline, X_train_real, y_train_real,
    cv=cv, method="predict_proba"
)[:, 1]

best_threshold = 0.5
best_f2 = 0.0
for t in np.arange(0.10, 0.91, 0.01):
    preds = (oof_probs >= t).astype(int)
    f2 = fbeta_score(y_train_real, preds, beta=2, zero_division=0)
    if f2 > best_f2:
        best_f2 = f2
        best_threshold = round(t, 2)

print(f"Optimal threshold (F2-score on OOF training predictions): {best_threshold:.2f}")
print(f"Best F2-score on training OOF: {best_f2:.3f}")

# ---------------------------------------------------------
# 6. FINAL EVALUATION on the held-out test set (ONE TIME ONLY)
# Test set is touched here and only here.
# ---------------------------------------------------------
print("\n--- EVALUATION OF FINAL DEPLOYED MODEL (seed=42) ---")
y_pred_prob_real = final_pipeline.predict_proba(X_test_real)[:, 1]
y_pred_real = (y_pred_prob_real >= best_threshold).astype(int)

roc_auc = roc_auc_score(y_test_real, y_pred_prob_real)
print(f"Final Model ROC-AUC on Real Holdout Set: {roc_auc:.3f}")
print(f"Decision Threshold Used: {best_threshold:.2f} (tuned on training OOF, not test set)")
print("\nClassification Report on Real Data:")
print(classification_report(y_test_real, y_pred_real, target_names=["Survived", "At Risk"]))

print("\nSaving model and threshold to logisticregression_pipeline.pkl...")
joblib.dump({"pipeline": final_pipeline, "threshold": best_threshold}, "logisticregression_pipeline.pkl")
print(f"Done! Model saved with threshold={best_threshold:.2f}")
