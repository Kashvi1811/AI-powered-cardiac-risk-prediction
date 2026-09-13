"""
robustness_test.py
------------------
Monte Carlo Cross-Validation (10 random seeds) comparing all 4 models
against two training strategies:
  - TSTR: Train on Synthetic, Test on Real
  - Combined: Train on Real + Synthetic, Test on Real

Key findings that drove the final model selection:
  Combined (Real + Synthetic):
    Logistic Regression: Mean ROC-AUC = 0.769 (±0.061)  <-- WINNER
    Random Forest:       Mean ROC-AUC = 0.734 (±0.045)
    SVM:                 Mean ROC-AUC = 0.736 (±0.059)
    XGBoost:             Mean ROC-AUC = 0.683 (±0.063)  <-- overfits

  XGBoost's seemingly strong 0.789 score at seed=42 was split-specific
  variance. Logistic Regression is genuinely the best model for this dataset.

Usage:
    python robustness_test.py
  (Warning: Takes ~5 minutes as it generates synthetic data 10 times)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from model_utils import log_transform_func
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
if "time" in df.columns:
    df = df.drop(columns=["time"])

X_real = df.drop(columns=["DEATH_EVENT"])
y_real = df["DEATH_EVENT"]

preprocessor = ColumnTransformer(
    transformers=[
        ("standard_scaling", StandardScaler(), ["age", "ejection_fraction", "serum_sodium"]),
        ("robust_scaling", RobustScaler(), ["platelets"]),
        ("log_robust", Pipeline([
            ("log_transform", FunctionTransformer(log_transform_func)),
            ("robust", RobustScaler())
        ]), ["serum_creatinine", "creatinine_phosphokinase"])
    ], remainder="passthrough"
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
    "SVM":                 SVC(class_weight="balanced", probability=True, random_state=42)
}

n_iterations = 10
results_tstr = {name: [] for name in models}
results_combined = {name: [] for name in models}

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(pd.concat([X_real, y_real], axis=1))
for col in ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT"]:
    if col in metadata.columns:
        metadata.update_column(column_name=col, sdtype='categorical')

print(f"Running Monte Carlo CV across {n_iterations} random seeds...")
for seed in range(n_iterations):
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.2, random_state=seed, stratify=y_real
    )
    df_train_real = pd.concat([X_train_real, y_train_real], axis=1)

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df_train_real)
    synth_data = synthesizer.sample(num_rows=2000)

    X_train_synth = synth_data.drop(columns=["DEATH_EVENT"])
    y_train_synth = synth_data["DEATH_EVENT"].astype(int)
    X_train_comb = pd.concat([X_train_real, X_train_synth], ignore_index=True)
    y_train_comb = pd.concat([y_train_real, y_train_synth], ignore_index=True)

    for name, model in models.items():
        pipe_tstr = Pipeline([("Preprocessor", preprocessor), ("Model", model)])
        pipe_tstr.fit(X_train_synth, y_train_synth)
        results_tstr[name].append(roc_auc_score(y_test_real, pipe_tstr.predict_proba(X_test_real)[:, 1]))

        pipe_comb = Pipeline([("Preprocessor", preprocessor), ("Model", model)])
        pipe_comb.fit(X_train_comb, y_train_comb)
        results_combined[name].append(roc_auc_score(y_test_real, pipe_comb.predict_proba(X_test_real)[:, 1]))

    print(f"  Seed {seed} done.")

print("\n--- RESULTS ACROSS 10 SEEDS ---")
print("\nTSTR (Synthetic Only):")
for name in models:
    m, s = np.mean(results_tstr[name]), np.std(results_tstr[name])
    print(f"  {name}: Mean ROC-AUC = {m:.3f} (±{s:.3f})")

print("\nCombined (Real + Synthetic):")
for name in models:
    m, s = np.mean(results_combined[name]), np.std(results_combined[name])
    print(f"  {name}: Mean ROC-AUC = {m:.3f} (±{s:.3f})")
