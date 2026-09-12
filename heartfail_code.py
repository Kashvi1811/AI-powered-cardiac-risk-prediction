import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

df.head()
# Male = 1, Female =0

df.columns

df.dtypes

df.isnull().sum()

df.head()

continous_cols = ["age" , "creatinine_phosphokinase" , "ejection_fraction" , "platelets" , "serum_creatinine" , "serum_sodium" , "time"]
binary_cols = ["anaemia" , "diabetes" , "high_blood_pressure" , "sex" , "smoking"]

for i in continous_cols:
    sns.histplot(x = df[i] , kde = True ,)
    plt.xlabel(i.title())
    plt.title(f"Distribution of {i} ")
    plt.show()

for i in binary_cols:
    ax = sns.countplot(x = df[i] , palette = "viridis" , )
    ax.bar_label(ax.containers[0])
    plt.xlabel(i.title())
    plt.title(f"Plot of {i} ")
    plt.show()

for i in continous_cols:
    sns.boxplot(x = df["DEATH_EVENT"] , y = df[i] , color = "orange")
    plt.xlabel("DEATH_EVENT")
    plt.ylabel(i.title())
    plt.title(f"{i.title()} vs Death Event ")
    plt.show()

for i in binary_cols:
    ax = sns.countplot(x = df[i] , hue = df["DEATH_EVENT"] , palette = "flare")
    for container in ax.containers:
        ax.bar_label(container)
    plt.ylabel("DEATH_EVENT")
    plt.xlabel(i.title())
    plt.title(f"{i.title()} vs Death Event ")
    plt.show()

df_cor = df.corr()
plt.figure(figsize = (10 , 10))
sns.heatmap(data = df_cor , annot = True)
plt.show()

# Numeric columns only (ignore binary + target)
num_cols = ["age", "time", "ejection_fraction", "serum_creatinine",
            "serum_sodium", "platelets", "creatinine_phosphokinase"]

# Check skewness + histograms
for col in num_cols:
    print(f"📊 {col} | Skewness = {df[col].skew():.2f}")
   

from sklearn.preprocessing import StandardScaler , RobustScaler , FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

standard_scaler = ["age", "ejection_fraction", "serum_sodium", "time"]
robust_scaler = ["platelets"]
log_robust = ["serum_creatinine", "creatinine_phosphokinase"]

def log_transform_func(x):
    return np.log1p(x)

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


X_scaled = preprocessor.fit_transform(X)

all_features = standard_scaler + robust_scaler + log_robust + [
    col for col in X.columns if col not in (standard_scaler + robust_scaler + log_robust)
]

scaled_df = pd.DataFrame(X_scaled, columns=all_features)



scaled_df

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df.shape


df.DEATH_EVENT.value_counts()

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42 , stratify = y)
#Stratify is crucial here because you want the ratio of deaths vs survivors to remain consistent in both sets.4

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# XGBoost model (default hyperparameters)
xgb_model_pre = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(eval_metric="logloss", random_state=42)),
])

xgb_model_pre.fit(X_train, y_train)

# Predictions
y_pred = xgb_model_pre.predict(X_test)
y_pred_prob = xgb_model_pre.predict_proba(X_test)[:, 1]

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"XGBoost (Pre-Tuning) ROC-AUC: {roc_auc:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Confusion Matrix (Pre-Tuning)")
plt.show()

# Classification Report
print("Classification report")
print(classification_report(y_test, y_pred))



from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Different models to try
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    "SVM": SVC(class_weight="balanced", probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("Preprocessor", preprocessor),
        ("Model", model)
    ])
    
    # Cross-validation with ROC-AUC (better for imbalanced data)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    results[name] = (np.mean(scores), np.std(scores))

import pandas as pd

df_results = pd.DataFrame(results, index=["Mean ROC-AUC", "Std Dev"]).T
df_results = df_results.round(3)
print(df_results.sort_values(ascending = False, by = "Mean ROC-AUC"))



from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

rf_pipeline = Pipeline([
    ("Preprocessor", preprocessor),
    ("Model", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )),
])

param_dist = {
    "Model__n_estimators": [200, 300, 400, 500],
    "Model__max_depth": [15, 20, None],
    "Model__min_samples_split": [2, 5],
    "Model__min_samples_leaf": [1, 2, 4],
    "Model__max_features": ["sqrt", "log2", 0.5],
}

stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized Search with StratifiedKFold
rf_random = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_dist,
    n_iter=30,
    scoring="roc_auc",
    cv=stratified_cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
)

rf_random.fit(X, y)

results = pd.DataFrame(rf_random.cv_results_)
best_idx = rf_random.best_index_
mean_auc = results.loc[best_idx, "mean_test_score"]
std_auc = results.loc[best_idx, "std_test_score"]

print("Best Parameters:", rf_random.best_params_)
print("Best ROC-AUC (mean):", round(mean_auc, 3))
print("Standard Deviation:", round(std_auc , 3))


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pandas as pd

param_grid_xgb = {
    "Model__n_estimators": [100, 200, 300],
    "Model__max_depth": [3, 5, 7],
    "Model__learning_rate": [0.1, 0.05, 0.01],
    "Model__subsample": [0.7, 1.0],
    "Model__colsample_bytree": [0.7, 1.0],
    "Model__min_child_weight": [1, 3, 5],
}

pipeline_xgb = Pipeline([
    ("Preprocessor", preprocessor),
    ("Model", XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )),
])

stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearch with StratifiedKFold
grid_search_xgb = GridSearchCV(
    pipeline_xgb,
    param_grid=param_grid_xgb,
    cv=stratified_cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2,
)

grid_search_xgb.fit(X, y)

print("Best Parameters:", grid_search_xgb.best_params_)
print("Best ROC-AUC:", grid_search_xgb.best_score_)

results = pd.DataFrame(grid_search_xgb.cv_results_)

best_idx = grid_search_xgb.best_index_

mean_auc = results.loc[best_idx, "mean_test_score"]
std_auc = results.loc[best_idx, "std_test_score"]

print("Best ROC-AUC (mean):", round(mean_auc, 3))
print("Standard Deviation:", round(std_auc, 3))


from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import numpy as np

# Pipeline
svm_pipeline = Pipeline([
    ("Preprocessor", preprocessor),
    ("Model", SVC(probability=True, random_state=42))
])

# Parameter distribution for RandomizedSearch
param_dist_svm = {
    "Model__C": [0.1, 1, 10, 100],
    "Model__kernel": ["linear", "rbf", "poly"],
    "Model__gamma": ["scale", "auto", 0.01, 0.1, 1],
    "Model__degree": [2, 3, 4]  # only used if kernel=poly
}

# Stratified CV
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized Search
svm_random = RandomizedSearchCV(
    svm_pipeline,
    param_distributions=param_dist_svm,
    n_iter=30,                 # number of random parameter sets to try
    scoring="roc_auc",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Fit
svm_random.fit(X, y)

# Results
results = pd.DataFrame(svm_random.cv_results_)
best_idx = svm_random.best_index_
mean_auc = results.loc[best_idx, "mean_test_score"]
std_auc = results.loc[best_idx, "std_test_score"]

print("Best Parameters:", svm_random.best_params_)
print("Best ROC-AUC (mean):", round(mean_auc ,3))
print("Standard Deviation:", round(std_auc , 3))


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

log_reg_pipeline = Pipeline([
    ("Preprocessor", preprocessor),
    ("Model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )),
])

param_grid_log = {
    "Model__C": [0.01, 0.1, 1, 10],          # regularization strength
    "Model__penalty": ["l2"],                # L2 penalty (default)
    "Model__solver": ["lbfgs", "saga"],      # solvers compatible with L2
}

# Stratified CV
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search_log = GridSearchCV(
    log_reg_pipeline,
    param_grid=param_grid_log,
    cv=stratified_cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2
)

grid_search_log.fit(X, y)

results_log = pd.DataFrame(grid_search_log.cv_results_)
best_idx = grid_search_log.best_index_
mean_auc = results_log.loc[best_idx, "mean_test_score"]
std_auc = results_log.loc[best_idx, "std_test_score"]

print("Best Parameters:", grid_search_log.best_params_)
print("Best ROC-AUC (mean):", round(mean_auc, 3))
print("Standard Deviation:", round(std_auc, 3))

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd

# models with tuned hyperparameters
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.01,
        penalty="l2",
        solver="saga",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        eval_metric="logloss",
        random_state=42
    ),
    "SVM": SVC(
        C=0.1,
        kernel="rbf",
        gamma="scale",
        degree=3,
        probability=True,
        class_weight="balanced",
        random_state=42
    ),
}

# Stratified CV for fair evaluation
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("Preprocessor", preprocessor),
        ("Model", model),
    ])

    # Cross-validation with StratifiedKFold
    scores = cross_val_score(pipeline, X, y, cv=stratified_cv, scoring="roc_auc")
    results[name] = (np.mean(scores), np.std(scores))

df_results = pd.DataFrame(results, index=["Mean ROC-AUC", "Std Dev"]).T
df_results = df_results.round(3)

df_results = df_results.sort_values(ascending=False, by="Mean ROC-AUC")
print(df_results)



from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
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

clf.fit(X, y)

joblib.dump(clf, "xgboost_pipeline.pkl")