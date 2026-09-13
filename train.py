import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
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

print("\nSaving model to logisticregression_pipeline.pkl...")
joblib.dump(final_pipeline, "logisticregression_pipeline.pkl")
print("Done!")
