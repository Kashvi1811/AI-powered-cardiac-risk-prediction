<div align="center">

# 🫀 AI-Powered Cardiac Risk Prediction

### *A production-grade AI healthcare platform for heart failure risk assessment*

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-189AB4?style=for-the-badge&logo=python&logoColor=white)](https://xgboost.readthedocs.io)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

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
| 🎯 **Risk Prediction** | XGBoost pipeline predicts heart failure risk with **ROC-AUC: 0.789** on real unseen data |
| 🤖 **AI Clinical Translation** | Google Gemini explains results in plain English, structured as JSON for a true dashboard |
| 📊 **Interactive Risk Gauge** | Plotly speedometer gauge maps risk onto a live Green → Amber → Red spectrum |
| 🎛️ **What-If Analysis** | Adjust 5 clinical parameters (EF, Creatinine, Sodium, BP, Smoking) and watch risk update instantly |
| 🧬 **Synthetic Data Training** | SDV GaussianCopula generates 5,000 realistic synthetic patient records, addressing dataset size limitations |
| 🛡️ **Secure API Key Management** | Gemini API key is stored server-side via Streamlit Secrets — never exposed to users |
| 📋 **Tabbed UI** | Clean 3-tab interface: Patient Intake → Clinical Dashboard → Deep Analysis |

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Cardiac Risk System                    │
│                                                             │
│  ┌─────────────┐    ┌──────────────────┐   ┌────────────┐  │
│  │  train.py   │───▶│  SDV Synthesizer │──▶│  5,000     │  │
│  │             │    │  (GaussianCopula)│   │  Synthetic │  │
│  └─────────────┘    └──────────────────┘   │  Records   │  │
│         │                                  └─────┬──────┘  │
│         ▼                                        ▼          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          Model Comparison (4 Models)                │   │
│  │   Logistic Regression │ SVM │ Random Forest │ XGB   │   │
│  └─────────────────────────────┬───────────────────────┘   │
│                                ▼                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        XGBoost Pipeline (Best on Real Holdout)      │   │
│  │         ColumnTransformer → XGBClassifier           │   │
│  │                 ROC-AUC: 0.789                      │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                    xgboost_pipeline.pkl                     │
│                             │                               │
│                             ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                     app.py (Streamlit)              │   │
│  │  Tab 1: Patient Intake                              │   │
│  │  Tab 2: Clinical Dashboard  ←── Google Gemini LLM  │   │
│  │  Tab 3: Deep Analysis (What-If + Plotly Charts)     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Pipeline & Methodology

### 1. Dataset
- **Source:** Heart Failure Clinical Records Dataset (UCI / Kaggle)
- **Size:** 299 real patient records
- **Target:** `DEATH_EVENT` (0 = survived, 1 = death due to heart failure)
- **Target Leakage Removed:** The `time` (follow-up period) column was explicitly dropped to prevent data leakage.

### 2. Synthetic Data Augmentation (TSTR Framework)

To overcome the 299-record limitation without introducing real patient data, we use **Train on Synthetic, Test on Real (TSTR)**:

```python
# SDV GaussianCopula learns the joint distribution of real training data
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df_train_real)
synthetic_data = synthesizer.sample(num_rows=5000)  # 5,000 realistic records
```

### 3. Model Comparison

| Model | Synthetic CV ROC-AUC |
|---|---|
| **Logistic Regression** | 0.681 |
| SVM | 0.659 |
| Random Forest | 0.637 |
| XGBoost | 0.615 |

> ✅ **Final Model:** XGBoost with tuned hyperparameters, achieving **ROC-AUC: 0.789** on unseen real patient data.

### 4. Preprocessing Pipeline

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

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, Plotly |
| **ML Model** | XGBoost, Scikit-learn |
| **AI / LLM** | Google Gemini 3.6 Flash (`google-genai`) |
| **Synthetic Data** | SDV (Synthetic Data Vault) — GaussianCopula |
| **Backend** | Python 3.14 |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | GitHub |

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
├── app.py                  # Main Streamlit application (3-tab UI)
├── train.py                # ML training pipeline (synthetic data + model comparison)
├── model_utils.py          # Shared utility functions (required for pickle serialization)
├── requirements.txt        # Python dependencies
├── xgboost_pipeline.pkl    # Trained XGBoost pipeline (serialized)
├── heart_failure_clinical_records_dataset.csv
├── .streamlit/
│   ├── config.toml         # Theme configuration (Medical Light Blue theme)
│   └── secrets.toml        # 🔒 NOT committed — add your Gemini API key here
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

