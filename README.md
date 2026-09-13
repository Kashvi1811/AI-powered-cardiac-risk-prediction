<div align="center">

# 🫀 AI-Powered Cardiac Risk Prediction

### *A production-grade AI healthcare platform for heart failure risk assessment*

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LogisticRegression](https://img.shields.io/badge/Logistic_Regression-Model-189AB4?style=for-the-badge&logo=python&logoColor=white)](https://scikit-learn.org)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)

<br/>

**[🚀 Live Demo](https://ai-powered-cardiac-risk-prediction.streamlit.app)** &nbsp;•&nbsp;
**[📊 Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)** &nbsp;•&nbsp;
**[📬 Contact](https://github.com/Kashvi1811)**

<br/>

![Dashboard Preview](dataset-card.jpg)

</div>

---

## 📖 About The Project

This is a **full-stack AI healthcare application** that combines classical machine learning with a modern, production-grade user interface to predict the risk of heart failure in patients.

The project goes beyond a typical ML notebook by incorporating:

- 🤖 **Google Gemini LLM** integration for AI-powered clinical explanations
- 🔬 **Synthetic data augmentation** (5,000 records via SDV GaussianCopula) to overcome dataset limitations
- 📊 **Interactive Plotly dashboards** with real-time risk gauges
- 🎛️ **Clinical "What-If" Analysis** — a decision support tool for simulating interventions
- 🧠 **Structured JSON AI output** for a real dashboard-style experience

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 **Risk Prediction** | Logistic Regression pipeline predicts heart failure risk with **mean ROC-AUC: 0.769** across 10 random seeds |
| 🧬 **Synthetic Data Augmentation** | SDV GaussianCopula generates 5,000 synthetic patient records combined with real data for training |
| 🛡️ **Medically-Tuned Threshold** | Decision threshold lowered to **0.40** (from default 0.50) to achieve **recall of 0.84** — catching 84% of at-risk patients |
| 🔬 **Robustness-Tested** | Model validated across **10 random seeds** (Monte Carlo CV) with 95% CI reported — not a single lucky split |
| 🤖 **AI Clinical Translation** | Google Gemini explains results in plain English, structured as JSON for a true dashboard experience |
| 📊 **Interactive Risk Gauge** | Plotly speedometer gauge maps risk onto a live Green → Amber → Red spectrum |
| 🎛️ **What-If Analysis** | Adjust 5 clinical parameters and watch risk update in real time |
| 🛡️ **Secure API Key Management** | Gemini API key stored server-side via Streamlit Secrets — never exposed to users |
| 📋 **Tabbed UI** | Clean 3-tab interface: Patient Intake → Clinical Dashboard → Deep Analysis |

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Cardiac Risk System                   │
│                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │  train.py   │───▶│  SDV Synthesizer │──▶│  5,000     │  │
│  │             │    │  (GaussianCopula)│    │  Synthetic │  │
│  └─────────────┘    └──────────────────┘    │  Records   │  │
│         │                                   └─────┬──────┘  │
│         ▼ (10-seed Monte Carlo CV)                ▼         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │     robustness_test.py — 4 Models × 2 Strategies    │    │
│  │   Logistic Regression │ SVM │ Random Forest │ XGB   │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                            ▼ WINNER: Logistic Regression    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │    Logistic Regression (Real + Synthetic Combined)  │    │
│  │    ColumnTransformer → LogisticRegression           │    │
│  │    Mean ROC-AUC: 0.769 | Recall@0.4: 0.84           │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                            │                                │
│              logisticregression_pipeline.pkl                │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                     app.py (Streamlit)              │    │
│  │  Tab 1: Patient Intake                              │    │
│  │  Tab 2: Clinical Dashboard  ←── Google Gemini LLM   │    │
│  │  Tab 3: Deep Analysis (What-If + Plotly Charts)     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Pipeline & Methodology

### 1. Dataset
- **Source:** Heart Failure Clinical Records Dataset (UCI / Kaggle)
- **Size:** 299 real patient records
- **Target:** `DEATH_EVENT` (0 = survived, 1 = death due to heart failure)
- **Target Leakage Removed:** The `time` (follow-up period) column was explicitly dropped to prevent data leakage.

### 2. Synthetic Data Augmentation (Combined Strategy)

Synthetic data is generated using the **SDV GaussianCopula** model and **combined with real training data** for the final model — proven to outperform pure-synthetic-only training across 10 random seeds:

```python
# SDV GaussianCopula learns the joint distribution of real training data
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df_train_real)
synthetic_data = synthesizer.sample(num_rows=5000)  # 5,000 realistic records

# Combine with real training data
X_train_combined = pd.concat([X_train_real, X_train_synth])
```

Synthetic data quality was validated with **sdmetrics QualityReport**: `84.05%` overall score (Column Shapes: 93.27%, Column Pair Trends: 74.83%).

### 3. Multi-Seed Robustness Testing (Monte Carlo CV)

All models were evaluated across **10 random train/test splits** to eliminate single-seed bias:

| Model | Combined Mean ROC-AUC | Std Dev |
|---|---|---|
| ✅ **Logistic Regression** | **0.769** | ±0.061 |
| SVM | 0.736 | ±0.059 |
| Random Forest | 0.734 | ±0.045 |
| XGBoost | 0.683 | ±0.063 |

> ⚠️ **XGBoost's 0.789 score at seed=42 was split-specific variance**, not genuine superiority. Logistic Regression won consistently across all 10 seeds.

### 4. Medical Decision Threshold Tuning

The default threshold of 0.50 produced a recall of only **0.47** — missing 53% of at-risk patients. This is unacceptable in a medical context. The app uses **threshold = 0.40**:

| Threshold | Recall (At-Risk) | Precision (At-Risk) |
|---|---|---|
| 0.50 (default) | 0.47 | 0.75 |
| **0.40 (deployed)** | **0.84** | **0.48** |
| 0.30 | 1.00 | 0.40 |

> In cardiac screening, a false negative (missed fatal case) is catastrophically worse than a false positive (extra follow-up). Tuning to 0.40 is the clinically correct choice.

### 5. Preprocessing Pipeline

```python
ColumnTransformer([
    ("standard_scaling", StandardScaler(), ["age", "ejection_fraction", "serum_sodium"]),
    ("robust_scaling",   RobustScaler(),   ["platelets"]),
    ("log_robust", Pipeline([
        ("log",    FunctionTransformer(log_transform_func)),  # handles skewed distributions
        ("robust", RobustScaler())
    ]), ["serum_creatinine", "creatinine_phosphokinase"])
])
```

---

## 🤖 AI Clinical Translation (Google Gemini)

The application uses **structured JSON output** from Google Gemini to power a true dashboard-style AI summary. This is not just a chatbot — Gemini returns a strict JSON schema:

```json
{
  "risk_overview": "...",
  "positive_factors": [{"factor": "...", "explanation": "..."}],
  "risk_drivers":     [{"factor": "...", "explanation": "..."}],
  "clinical_insight": "...",
  "monitoring":       ["..."],
  "takeaway":         "..."
}
```

Each field is rendered into a dedicated, styled UI component — green cards for protective factors, amber cards for risk drivers, and an info box for the AI's clinical insight.

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Frontend** | Streamlit, Plotly | 1.59.2, 7.0.0 |
| **ML Model** | Logistic Regression, Scikit-learn | 1.8.0 |
| **AI / LLM** | Google Gemini 3.6 Flash (`google-genai`) | 2.23.0 |
| **Synthetic Data** | SDV (GaussianCopula) + sdmetrics | 1.38.3, 0.31.0 |
| **Backend** | Python 3.14, pandas, numpy | 2.3.3, 2.4.4 |
| **Deployment** | Streamlit Community Cloud | — |
| **Version Control** | GitHub | — |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A free Google Gemini API key from [aistudio.google.com](https://aistudio.google.com/apikey)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/Kashvi1811/AI-powered-cardiac-risk-prediction.git
cd AI-powered-cardiac-risk-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
# Create .streamlit/secrets.toml and add:
# GEMINI_API_KEY = "your-api-key-here"

# 4. (Optional) Retrain the model
python train.py

# 5. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application (3-tab UI, threshold=0.40)
├── train.py                        # ML training pipeline (10-seed robustness + final model export)
├── robustness_test.py              # Monte Carlo CV: 4 models × 2 strategies × 10 seeds
├── evaluate_threshold.py           # Decision threshold analysis (0.2 → 0.5 comparison)
├── model_utils.py                  # Shared utility functions (required for pickle serialization)
├── heartfail_code.py               # Original EDA and notebook-style model exploration
├── requirements.txt                # Pinned Python dependencies
├── logisticregression_pipeline.pkl # Trained Logistic Regression pipeline (serialized)
├── heart_failure_clinical_records_dataset.csv
├── .streamlit/
│   ├── config.toml                 # Theme configuration (Medical Light Blue theme)
│   └── secrets.toml                # 🔒 NOT committed — add your Gemini API key here
└── README.md
```

---

## 📜 License

This project is intended for educational and personal use. All rights reserved by **Kashvi1811**.  
*For any usage beyond personal or educational purposes, please contact me in advance.*

## 🤝 Contact & Colloboration

I’m always open to feedback, ideas, and collaboration opportunities! Feel free to reach out:

- **GitHub:** [@Kashvi1811](https://github.com/Kashvi1811)
- **LinkedIn:** https://www.linkedin.com/in/kashvi-soni-6330a92b2/
- **Email:** kashvisoni2005@gmail.com

## ⚠️ Disclaimer

> This application is built for **educational and portfolio purposes only**. It is not a certified medical device and does not constitute medical advice. Always consult a qualified healthcare professional for clinical decisions.

---

<div align="center">

Made with ❤️ by **[Kashvi Soni](https://github.com/Kashvi1811)**

⭐ If you found this project useful, please star the repository!

</div>

