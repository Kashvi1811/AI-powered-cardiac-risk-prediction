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
# The 'time' column (follow-up period) is target leakage.
# We must remove it completely.
if "time" in df.columns:
    df = df.drop(columns=["time"])
    print("Dropped 'time' feature to prevent target leakage.")

X_real = df.drop(columns=["DEATH_EVENT"])
y_real = df["DEATH_EVENT"]

# Split into Real Train and Real Test sets (80/20)
print("Splitting into Real Train and Real Test...")
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

df_train_real = pd.concat([X_train_real, y_train_real], axis=1)
df_test_real = pd.concat([X_test_real, y_test_real], axis=1)

# ---------------------------------------------------------
# 2. SYNTHETIC DATA GENERATION (SDV)
# ---------------------------------------------------------
print("\n--- SYNTHETIC DATA WORKFLOW ---")
print("Extracting metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_train_real)

# Treat binary variables properly
for col in ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT"]:
    if col in metadata.columns:
        metadata.update_column(column_name=col, sdtype='categorical')

print("Training GaussianCopula Synthesizer on REAL training data...")
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df_train_real)

print("Generating 5,000 synthetic patient records...")
synthetic_data = synthesizer.sample(num_rows=5000)

print("Evaluating synthetic data quality...")
report = QualityReport()
report.generate(df_train_real, synthetic_data, metadata.to_dict())
print(f"Overall Quality Score: {report.get_score()*100:.2f}%")

# Separate features and target from synthetic data
X_train_synth = synthetic_data.drop(columns=["DEATH_EVENT"])
y_train_synth = synthetic_data["DEATH_EVENT"].astype(int)

# ---------------------------------------------------------
# 3. PREPROCESSING PIPELINE
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
# 4. MODEL COMPARISON (Train on Synthetic, Eval on Synthetic CV)
# ---------------------------------------------------------
print("\n--- MODEL COMPARISON (Trained & Cross-Validated on 5000 Synthetic Records) ---")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
    "SVM": SVC(class_weight="balanced", probability=True, random_state=42)
}

stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("Preprocessor", preprocessor),
        ("Model", model)
    ])
    # Cross validation on synthetic data
    scores = cross_val_score(pipeline, X_train_synth, y_train_synth, cv=stratified_cv, scoring="roc_auc")
    results[name] = np.mean(scores)
    print(f"{name} Synthetic CV ROC-AUC: {np.mean(scores):.3f}")

best_model_name = max(results, key=results.get)
print(f"Best model based on Synthetic CV: {best_model_name}")

# We will use XGBoost as the final model for the pipeline (with some basic tuned params)
print("\nTraining Final XGBoost Pipeline on all Synthetic Data...")
final_pipeline = Pipeline([
    ("Preprocessor", preprocessor),
    ("Model", XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        eval_metric="logloss",
        random_state=42
    ))
])

final_pipeline.fit(X_train_synth, y_train_synth)

# ---------------------------------------------------------
# 5. EVALUATION ON HOLD-OUT REAL DATA (TSTR)
# ---------------------------------------------------------
print("\n--- EVALUATION ON UNSEEN REAL TEST SET ---")
y_pred_real = final_pipeline.predict(X_test_real)
y_pred_prob_real = final_pipeline.predict_proba(X_test_real)[:, 1]

roc_auc = roc_auc_score(y_test_real, y_pred_prob_real)
print(f"Final Model ROC-AUC on Real Holdout Set: {roc_auc:.3f}")
print("\nClassification Report on Real Data:")
print(classification_report(y_test_real, y_pred_real))

# ---------------------------------------------------------
# 6. EXPORT
# ---------------------------------------------------------
print("\nSaving model to xgboost_pipeline.pkl...")
joblib.dump(final_pipeline, "xgboost_pipeline.pkl")
print("Done!")
