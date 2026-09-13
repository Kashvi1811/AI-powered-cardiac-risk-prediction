import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import joblib # type: ignore
import json
import plotly.graph_objects as go # type: ignore
from pathlib import Path
from model_utils import log_transform_func  # required for pickle to resolve the pipeline's transform step

# ---------------------------
# Setup and model
# ---------------------------
st.set_page_config(page_title="Cardiac Health AI", page_icon="🫀", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("logisticregression_pipeline.pkl")

model = load_model()
gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")

st.markdown("""
<style>
/* ============================================
   ANTI-FLASH LAYER: force background stays blue-grey during reruns
   ============================================ */
html, body {
    background-color: #EBF0F8 !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #EBF0F8 !important;
}
[data-testid="stHeader"] {
    background-color: #EBF0F8 !important;
}
/* Keeps the flashing rerun overlay the same color as our bg */
[data-testid="stStatusWidget"] { display: none; }

/* ============================================
   GENERAL LAYOUT
   ============================================ */
.block-container {
    padding-top: 2rem !important;
    max-width: 1200px;
}

/* ============================================
   TYPOGRAPHY: Bigger, bolder headings
   ============================================ */
h1 {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    color: #1E3A5F !important;
    letter-spacing: -0.5px;
}
h2 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #1E3A5F !important;
    margin-top: 1rem;
}
h3 {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #1D4ED8 !important;
}

/* ============================================
   CARDS: white with blue-tinted shadow
   ============================================ */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 4px 14px rgba(29, 78, 216, 0.10);
    border: 1px solid #BFDBFE;
}

/* ============================================
   TABS: styled for medical UI
   ============================================ */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #BFDBFE;
    gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #1E3A5F !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    background: #DBEAFE;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1D4ED8 !important;
    color: #FFFFFF !important;
}

/* ============================================
   FORM SUBHEADERS
   ============================================ */
.stSubheader {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: #1E3A5F !important;
    border-left: 4px solid #1D4ED8;
    padding-left: 10px;
    margin-top: 1.2rem;
}

/* ============================================
   BUTTON: Premium Blue Gradient
   ============================================ */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    padding: 12px 24px !important;
    box-shadow: 0 4px 14px rgba(29, 78, 216, 0.3);
    transition: filter 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.06) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LLM Clinical Translation (JSON)
# ---------------------------
def generate_clinical_summary_json(patient_data: dict, probability: float, prediction: int, api_key: str) -> dict:
    """Calls Gemini with Structured JSON Output"""
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)

        risk_level = "HIGH" if prediction == 1 else "LOW"
        
        # Format the data cleanly for the prompt
        formatted_data = patient_data.copy()
        formatted_data['sex'] = 'Male' if formatted_data['sex'] == 1 else 'Female'
        formatted_data['smoking'] = 'Yes' if formatted_data['smoking'] == 1 else 'No'
        formatted_data['high_blood_pressure'] = 'Yes' if formatted_data['high_blood_pressure'] == 1 else 'No'
        formatted_data['diabetes'] = 'Yes' if formatted_data['diabetes'] == 1 else 'No'
        formatted_data['anaemia'] = 'Yes' if formatted_data['anaemia'] == 1 else 'No'
        
        prompt = f"""You are an expert AI cardiologist.
Analyze this patient profile:
{json.dumps(formatted_data, indent=2)}

The XGBoost model assessed a {risk_level} RISK of heart failure with a probability of {probability:.1%}.

You MUST return a valid JSON object matching this exact structure, translating medical jargon into plain, reassuring English suitable for a patient dashboard:
{{
  "risk_overview": "1-2 sentences summarizing the overall risk and probability.",
  "positive_factors": [
     {{"factor": "Metric Name", "explanation": "Why it is good (plain English)"}}
  ],
  "risk_drivers": [
     {{"factor": "Metric Name", "explanation": "Why it increases risk (plain English). If none, leave array empty."}}
  ],
  "clinical_insight": "1 short paragraph translating the medical jargon into a simple analogy.",
  "monitoring": ["Recommendation 1", "Recommendation 2"],
  "takeaway": "1 bold sentence summary"
}}
"""
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# State Management
# ---------------------------
feature_names = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking"
]

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

# ---------------------------
# Hero Section
# ---------------------------
st.title("🫀 AI-Assisted Cardiac Risk Assessment")
st.markdown("Advanced predictive analysis for heart failure risk using comprehensive clinical parameters and cutting-edge machine learning.")
st.divider()

# ---------------------------
# Layout: Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📋 Patient Intake", "📊 Clinical Dashboard", "🔍 Deep Analysis"])

# --- TAB 1: DATA ENTRY ---
with tab1:
    st.header("Patient Profile Data Entry")
    
    with st.form("clinical_form"):
        # Group 1: Demographics
        st.subheader("Demographics & Lifestyle")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", 1, 120, 50)
        with col2:
            sex_choice = st.selectbox("Sex", ["Female", "Male"])
            sex = 1 if sex_choice == "Male" else 0
        with col3:
            smoking_choice = st.selectbox("Smoking", ["No", "Yes"])
            smoking = 1 if smoking_choice == "Yes" else 0
            
        # Group 2: Vitals & History
        st.subheader("Clinical History")
        col4, col5, col6 = st.columns(3)
        with col4:
            ejection_fraction = st.number_input("Ejection Fraction (%)", 1, 100, 40)
        with col5:
            high_blood_pressure_choice = st.selectbox("High Blood Pressure", ["No", "Yes"])
            high_blood_pressure = 1 if high_blood_pressure_choice == "Yes" else 0
        with col6:
            diabetes_choice = st.selectbox("Diabetes", ["No", "Yes"])
            diabetes = 1 if diabetes_choice == "Yes" else 0
            
        # Group 3: Blood Work
        st.subheader("Blood Work")
        col7, col8, col9, col10 = st.columns(4)
        with col7:
            serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.0, 20.0, 1.0)
        with col8:
            serum_sodium = st.number_input("Serum Sodium (mEq/L)", 100, 200, 140)
        with col9:
            creatinine_phosphokinase = st.number_input("CPK (U/L)", 0, 10000, 100)
        with col10:
            platelets = st.number_input("Platelets (/µL)", 10000, 1000000, 200000)
            
        anaemia_choice = st.selectbox("Anaemia", ["No", "Yes"])
        anaemia = 1 if anaemia_choice == "Yes" else 0

        st.divider()
        submitted = st.form_submit_button("Analyze Patient Profile", type="primary", use_container_width=True)
        if submitted:
            st.session_state.patient_data = {
                "age": age, "sex": sex, "smoking": smoking,
                "ejection_fraction": ejection_fraction, "high_blood_pressure": high_blood_pressure,
                "diabetes": diabetes, "serum_creatinine": serum_creatinine,
                "serum_sodium": serum_sodium, "creatinine_phosphokinase": creatinine_phosphokinase,
                "platelets": platelets, "anaemia": anaemia
            }
            st.success("✅ Data saved! Navigate to the 'Clinical Dashboard' tab to view results.")

# --- TAB 2: DASHBOARD ---
with tab2:
    if st.session_state.patient_data is None:
        st.info("💡 Please enter patient data in the 'Patient Intake' tab first.")
    else:
        patient_data = st.session_state.patient_data
        df = pd.DataFrame([patient_data])[feature_names] # ensure ordering
        
        pred = model.predict(df)[0]
        prob = float(model.predict_proba(df)[0][1])
        
        # 1. Executive Summary Metric
        dash_col1, dash_col2 = st.columns([1, 1])
        
        with dash_col1:
            st.markdown("### 📊 Risk Assessment Gauge")
            
            # Plotly Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Heart Failure Risk Probability", 'font': {'size': 18}},
                number = {'suffix': "%", 'font': {'size': 40}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': "#10B981"}, # Green
                        {'range': [30, 60], 'color': "#F59E0B"}, # Amber
                        {'range': [60, 100], 'color': "#EF4444"}], # Red
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            if pred == 1:
                st.error("🚨 **HIGH CLINICAL RISK DETECTED**")
            else:
                st.success("✅ **LOW CLINICAL RISK DETECTED**")
                
        with dash_col2:
            st.markdown("### 🤖 AI Clinical Translation")
            if not gemini_api_key:
                st.warning("API Key missing. Add GEMINI_API_KEY to secrets to enable AI summaries.")
            else:
                with st.spinner("Analyzing clinical profile..."):
                    summary_json = generate_clinical_summary_json(patient_data, prob, pred, gemini_api_key)
                    
                if "error" in summary_json:
                    st.error(f"AI Generation Error: {summary_json['error']}")
                else:
                    st.info(summary_json.get("risk_overview", ""))
                    
                    st.subheader("💡 Clinical Insight")
                    st.write(summary_json.get("clinical_insight", ""))
                    
                    st.success(f"**Takeaway:** {summary_json.get('takeaway', '')}")
                    
        # 2. Detailed Breakdown (Cards)
        if 'summary_json' in locals() and "error" not in summary_json:
            st.divider()
            col_pos, col_neg = st.columns(2)
            
            with col_pos:
                st.subheader("🟢 Protective Factors")
                for factor in summary_json.get("positive_factors", []):
                    with st.container(border=True):
                        st.markdown(f"**{factor.get('factor')}**")
                        st.caption(factor.get('explanation'))
                        
            with col_neg:
                st.subheader("⚠️ Risk Drivers")
                drivers = summary_json.get("risk_drivers", [])
                if not drivers:
                    st.success("No major clinical risk drivers identified.")
                else:
                    for factor in drivers:
                        with st.container(border=True):
                            st.markdown(f"**{factor.get('factor')}**")
                            st.caption(factor.get('explanation'))

            st.divider()
            st.subheader("🩺 Recommended Monitoring")
            for rec in summary_json.get("monitoring", []):
                st.markdown(f"- {rec}")
                
            st.caption("*Disclaimer: This assessment is generated by an AI model for informational purposes only and does not constitute medical advice.*")

# --- TAB 3: DEEP ANALYSIS ---
with tab3:
    if st.session_state.patient_data is None:
        st.info("💡 Please enter patient data in the 'Patient Intake' tab first.")
    else:
        patient_data = st.session_state.patient_data
        df = pd.DataFrame([patient_data])[feature_names]
        orig_prob = float(model.predict_proba(df)[0][1])
        
        st.header("🎛️ Interactive Decision Support (What-If Analysis)")
        st.markdown("Simulate clinical interventions. Adjust the variables below to see the real-time impact on the patient's risk profile.")
        
        wi_col1, wi_col2 = st.columns([1, 1.5])
        
        with wi_col1:
            st.subheader("Adjust Clinical Metrics")
            new_ef = st.slider("Ejection Fraction (%)", 1, 100, int(patient_data['ejection_fraction']), help="Percentage of blood leaving the heart each time it contracts.")
            new_scr = st.slider("Serum Creatinine (mg/dL)", 0.0, 10.0, float(patient_data['serum_creatinine']), 0.1, help="Level of creatinine in the blood, indicating kidney function.")
            new_sodium = st.slider("Serum Sodium (mEq/L)", 100, 200, int(patient_data['serum_sodium']), help="Electrolyte balance.")
            
            new_bp = st.toggle("High Blood Pressure", value=bool(patient_data['high_blood_pressure']))
            new_smoking = st.toggle("Smoking", value=bool(patient_data['smoking']))
            
        with wi_col2:
            st.subheader("Risk Impact Visualization")
            
            # Calculate new prediction
            wi_data = patient_data.copy()
            wi_data['ejection_fraction'] = new_ef
            wi_data['serum_creatinine'] = new_scr
            wi_data['serum_sodium'] = new_sodium
            wi_data['high_blood_pressure'] = 1 if new_bp else 0
            wi_data['smoking'] = 1 if new_smoking else 0
            
            wi_df = pd.DataFrame([wi_data])[feature_names]
            wi_prob = float(model.predict_proba(wi_df)[0][1])
            
            # Plotly Bar Chart comparing original vs simulated
            fig_bar = go.Figure(data=[
                go.Bar(
                    name='Baseline Risk',
                    x=['Risk Probability'],
                    y=[orig_prob * 100],
                    marker_color='#94A3B8',
                    text=[f"{orig_prob:.1%}"],
                    textposition='outside',
                    textfont=dict(size=16, color='#1E3A5F', family='Arial Black')
                ),
                go.Bar(
                    name='Simulated Risk',
                    x=['Risk Probability'],
                    y=[wi_prob * 100],
                    marker_color='#1D4ED8',
                    text=[f"{wi_prob:.1%}"],
                    textposition='outside',
                    textfont=dict(size=16, color='#1D4ED8', family='Arial Black')
                )
            ])
            fig_bar.update_layout(
                barmode='group',
                yaxis=dict(title='Probability (%)', range=[0, 115]),
                height=420,
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            delta = wi_prob - orig_prob
            st.metric("Simulated New Probability", f"{wi_prob:.1%}", delta=f"{delta:+.1%} from baseline", delta_color="inverse")
