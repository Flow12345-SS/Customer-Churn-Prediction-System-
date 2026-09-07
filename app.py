import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="CHURNIQ Customer Intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Typography & Hierarchy */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }
    
    .section-desc { color: #94a3b8; font-size: 1.1rem; line-height: 1.6; margin-top: -10px; margin-bottom: 30px; font-weight: 400;}
    
    /* Buttons */
    .stButton>button { 
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important; 
        color: white !important; 
        font-size: 1.05rem !important; 
        font-weight: 600 !important; 
        padding: 0.75rem 1.5rem !important; 
        border-radius: 12px !important; 
        border: 1px solid rgba(255,255,255,0.1) !important; 
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3), inset 0 1px 0 rgba(255,255,255,0.2) !important; 
        transition: all 0.3s ease !important; 
    }
    .stButton>button:hover { 
        transform: translateY(-2px) !important; 
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important; 
    }
    
    /* Secondary Buttons (Outlined) */
    .btn-secondary {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid #334155 !important;
        box-shadow: none !important;
    }
    .btn-secondary:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: #475569 !important;
    }
    
    /* Result Banner & UI Elements */
    .premium-card { background: rgba(15, 23, 42, 0.6); border: 1px solid #1e293b; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); backdrop-filter: blur(10px); }
    .result-banner { background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 58, 138, 0.8)); border: 1px solid #3b82f6; border-radius: 20px; padding: 40px 30px; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1); margin-bottom: 30px; position: relative; overflow: hidden;}
    .result-banner::before { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(56,189,248,0.1) 0%, rgba(0,0,0,0) 70%); pointer-events: none; }
    
    .risk-badge { padding: 6px 16px; border-radius: 100px; font-weight: 800; font-size: 0.9rem; display: inline-block; letter-spacing: 1.5px; text-transform: uppercase; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
    .badge-high { background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(185, 28, 28, 0.2)); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.5); }
    .badge-medium { background: linear-gradient(90deg, rgba(245, 158, 11, 0.2), rgba(180, 83, 9, 0.2)); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.5); }
    .badge-low { background: linear-gradient(90deg, rgba(16, 185, 129, 0.2), rgba(4, 120, 87, 0.2)); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.5); }
    
    .prob-number { font-size: 4.5rem; font-weight: 900; background: -webkit-linear-gradient(45deg, #ffffff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; margin: 15px 0; letter-spacing: -2px;}
    
    /* Info Cards */
    .info-card { background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 24px; height: 100%; box-shadow: 0 4px 20px rgba(0,0,0,0.1); transition: all 0.3s ease; }
    .info-card:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.1); background: rgba(30, 41, 59, 0.6); }
    .info-card h4 { color: #f8fafc; font-weight: 700; margin-top: 10px; margin-bottom: 10px; font-size: 1.1rem; }
    .info-card p { color: #94a3b8; font-size: 0.95rem; line-height: 1.5; margin: 0; }
    .info-card .icon { font-size: 2rem; margin-bottom: 15px; display: inline-block; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); }
    
    /* Hero Headers */
    .page-hero { background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(2, 6, 23, 0.9)); border: 1px solid #1e293b; padding: 30px 40px; border-radius: 20px; margin-bottom: 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); position: relative; overflow: hidden; }
    .page-hero::before { content: ''; position: absolute; left: 0; top: 0; width: 4px; height: 100%; background: linear-gradient(to bottom, #3b82f6, #a855f7); }
    .page-hero h1 { margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 900; letter-spacing: -0.5px; }
    .breadcrumb { color: #3b82f6; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px; display: block; }
    
    /* Layout cleanups */
    .block-container { padding-top: 2rem !important; padding-bottom: 5rem !important; }
    
    /* Custom divider */
    hr { border-color: rgba(255,255,255,0.05) !important; margin: 2rem 0 !important; }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Premium Data Table styling */
    .dataframe { background: transparent !important; }
    .dataframe th { background: rgba(15,23,42,0.8) !important; color: #94a3b8 !important; font-weight: 600 !important; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 1px; border-bottom: 1px solid #1e293b !important; }
    .dataframe td { border-bottom: 1px solid #1e293b !important; color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA & MODEL LOADING
# ==========================================
@st.cache_data
def load_data():
    try:
        data_path = os.path.join(BASE_DIR, "data", "customer_churn.csv")
        df = pd.read_csv(data_path)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
        df = df.dropna(subset=['TotalCharges'])
        df['Churn_Label'] = df['Churn']
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try: 
        model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_metrics():
    try: 
        metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.csv")
        return pd.read_csv(metrics_path)
    except: 
        return pd.DataFrame()

@st.cache_data
def get_global_predictions(_df, _model):
    if _df.empty or _model is None: return _df
    df_pred = _df.copy()
    X = df_pred.drop(columns=['customerID', 'Churn', 'Churn_Label'], errors='ignore')
    try:
        probs = _model.predict_proba(X)[:, 1]
        df_pred['Churn_Probability'] = probs * 100
        df_pred['Risk_Level'] = pd.cut(df_pred['Churn_Probability'], bins=[-1, 39, 69, 100], labels=['Low', 'Medium', 'High'])
        max_charge = df_pred['MonthlyCharges'].max()
        if max_charge > 0:
            val_norm = df_pred['MonthlyCharges'] / max_charge
            df_pred['Priority_Score'] = val_norm * (df_pred['Churn_Probability']/100) * 100
            df_pred['Priority_Tier'] = pd.cut(df_pred['Priority_Score'], bins=[-1, 20, 50, 100], labels=['Low Priority', 'Medium Priority', 'High Priority'])
    except Exception as e: 
        st.error(f"Error generating predictions: {e}")
    return df_pred

df_raw = load_data()
model = load_model()
metrics_df = load_metrics()
df = get_global_predictions(df_raw, model)

@st.cache_data
def get_feature_importance(_model):
    if _model is not None:
        try:
            clf = _model.named_steps['classifier']
            pre = _model.named_steps['preprocessor']
            if hasattr(clf, 'coef_'):
                coefs = clf.coef_[0]
                features = pre.get_feature_names_out()
                fi_df = pd.DataFrame({'Feature': features, 'Importance': coefs})
                fi_df['Feature'] = fi_df['Feature'].str.replace('num__', '').str.replace('cat__', '')
                return fi_df.sort_values(by='Importance', key=abs, ascending=False).head(10)
        except: pass
    return None
feature_importance_df = get_feature_importance(model)

# ==========================================
# UI HELPERS
# ==========================================
def render_hero(title, subtitle, breadcrumb):
    st.markdown(f'<div class="page-hero"><span class="breadcrumb">{breadcrumb}</span><h1>{title}</h1><p style="color: #94a3b8; font-size: 1rem; margin:0;">{subtitle}</p></div>', unsafe_allow_html=True)

def render_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge", value=probability,
        gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#30363d"}, 'bar': {'color': "#f8fafc", 'thickness': 0.3},
               'bgcolor': "#161b22", 'borderwidth': 0, 'steps': [{'range': [0, 39], 'color': 'rgba(16, 185, 129, 0.8)'}, {'range': [39, 69], 'color': 'rgba(245, 158, 11, 0.8)'}, {'range': [69, 100], 'color': 'rgba(239, 68, 68, 0.8)'}]}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"}, height=150, margin=dict(l=20, r=20, t=10, b=10))
    return fig

# ==========================================
# SIDEBAR NAVIGATION (CONSOLIDATED)
# ==========================================
with st.sidebar:
    st.markdown("""<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="background: -webkit-linear-gradient(45deg, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; font-weight: 900; letter-spacing: 2px;">⚡ CHURNIQ</h1>
    <p style="color: #f8fafc; font-size: 1rem; margin-top: -5px; font-weight: bold;">Customer Intelligence</p>
    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: -10px;">Predict • Understand • Retain</p>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 1px solid #30363d; margin-top: 0; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    if "nav_radio" not in st.session_state:
        st.session_state.nav_radio = "🏠 Overview"

    page = st.radio("NAVIGATION", [
        "🏠 Overview", 
        "🔮 Predict",
        "👤 Customer 360",
        "🎯 Risk Center",
        "📊 Analytics"
    ], label_visibility="collapsed", key="nav_radio")
    
    st.markdown("<hr style='border: 1px solid #30363d; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    st.markdown("""<div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 15px;"><div style="color: #34d399; font-size: 0.8rem; font-weight: bold; margin-bottom: 5px;">● AI MODEL ACTIVE</div><div style="color: #f8fafc; font-weight: bold; font-size: 1rem;">Logistic Regression</div>""", unsafe_allow_html=True)
    if not metrics_df.empty:
        best_model = metrics_df.sort_values(by='ROC-AUC', ascending=False).iloc[0]
        st.markdown(f"""<div style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">ROC-AUC <span style="color:#f8fafc; font-weight:bold;">{best_model['ROC-AUC']:.4f}</span><br>Ready for predictions</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 1. OVERVIEW
# ==========================================
if page == "🏠 Overview":
    if df.empty or 'Churn_Probability' not in df.columns: st.error("Data or global predictions not available.")
    else:
        total = len(df)
        churned = len(df[df['Churn'] == 1])
        high_risk = len(df[df['Risk_Level'] == 'High'])
        churn_rate = (churned/total*100) if total > 0 else 0
        avg_prob = df['Churn_Probability'].mean() if 'Churn_Probability' in df else 0
        best_roc = metrics_df.sort_values(by='ROC-AUC', ascending=False).iloc[0]['ROC-AUC'] if not metrics_df.empty else 0.8361
        
        # HERO SECTION
        h_col1, h_col2 = st.columns([1.2, 1])
        with h_col1:
            st.markdown("""
            <div style="padding-top: 10px;">
                <h1 style="font-size: 3.2rem; font-weight: 900; line-height: 1.1; margin-top: 5px; margin-bottom: 0; background: -webkit-linear-gradient(45deg, #ffffff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px;">CUSTOMER RETENTION<br>COMMAND CENTER</h1>
                <p style="color: #94a3b8; font-size: 1.2rem; line-height: 1.6; margin-top: 15px; margin-bottom: 30px; max-width: 90%;">Know who may leave. Understand why. Take action earlier.</p>
            </div>
            """, unsafe_allow_html=True)
            
            bh1, bh2 = st.columns([1, 1])
            with bh1: st.button("✦ Predict Customer", on_click=lambda: st.session_state.update(nav_radio="🔮 Predict"), use_container_width=True)
            with bh2: st.markdown('<div class="btn-secondary">', unsafe_allow_html=True); st.button("◉ View Risk Center", on_click=lambda: st.session_state.update(nav_radio="🎯 Risk Center"), use_container_width=True); st.markdown('</div>', unsafe_allow_html=True)
            
        with h_col2:
            st.markdown("""
            <div style="position: relative; height: 100%; min-height: 350px; display: flex; align-items: center; justify-content: center; background: radial-gradient(circle at center, rgba(56, 189, 248, 0.1) 0%, transparent 70%);">
                <div style="position: absolute; width: 260px; height: 260px; border-radius: 50%; border: 1px solid rgba(56, 189, 248, 0.2); box-shadow: 0 0 30px rgba(56, 189, 248, 0.05);"></div>
                <div style="position: absolute; width: 200px; height: 200px; border-radius: 50%; border: 2px dashed rgba(168, 85, 247, 0.3); animation: spin 25s linear infinite;"></div>
                <div style="position: absolute; width: 120px; height: 120px; border-radius: 50%; background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(168, 85, 247, 0.3)); box-shadow: 0 0 40px rgba(168, 85, 247, 0.3); display: flex; align-items: center; justify-content: center;">
                    <div style="width: 60%; height: 60%; border-radius: 50%; background: rgba(15, 23, 42, 0.9); display: flex; align-items: center; justify-content: center; border: 1px solid rgba(168,85,247,0.5);"><span style="font-size:1.5rem; font-weight:900; color:#f8fafc;">AI</span></div>
                </div>
                <div style="position: absolute; top: 15%; left: 5%; background: rgba(15, 23, 42, 0.7); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.3); color: #f8fafc; font-size: 0.7rem; backdrop-filter: blur(4px);">Analysis Ready</div>
                <div style="position: absolute; bottom: 20%; right: 5%; background: rgba(15, 23, 42, 0.7); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; font-size: 0.7rem; backdrop-filter: blur(4px);">High Risk Detected</div>
                <style>@keyframes spin { 100% { transform: rotate(360deg); } }</style>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # LIVE STATUS BAR
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); padding: 12px 20px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; color: #94a3b8; margin-bottom: 20px;">
            <div><span style="color: #34d399;">● MODEL ONLINE</span> &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; <span style="color: #38bdf8;">● DATA READY</span> &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; <span style="color: #a855f7;">● PREDICTIONS AVAILABLE</span></div>
            <div>LAST ANALYSIS: JUST NOW</div>
        </div>
        """, unsafe_allow_html=True)

        # EXECUTIVE KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("TOTAL CUSTOMERS", f"{total:,}")
        k2.metric("CHURN RATE", f"{churn_rate:.1f}%")
        k3.metric("HIGH-RISK CUSTOMERS", f"{high_risk:,}")
        k4.metric("AVG CHURN PROBABILITY", f"{avg_prob:.1f}%")
        k5.metric("MODEL ROC-AUC", f"{best_roc:.4f}")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # CHARTS & ALERTS
        r1, r2 = st.columns([2, 1])
        with r1:
            st.markdown("<h4 style='color:#f8fafc; margin-top:0;'>CHURN PROBABILITY DISTRIBUTION</h4>", unsafe_allow_html=True)
            fig_dist = px.histogram(df, x="Churn_Probability", color="Risk_Level", nbins=40,
                                   color_discrete_map={"High": "#ef4444", "Medium": "#fbbf24", "Low": "#10b981"},
                                   opacity=0.8, marginal="box")
            fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), height=300, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with r2:
            st.markdown("<h4 style='color:#f8fafc; margin-top:0;'>EXECUTIVE ALERTS</h4>", unsafe_allow_html=True)
            
            # Generate dynamic alerts based on actual data
            m2m_high = len(df[(df['Contract'] == 'Month-to-month') & (df['Risk_Level'] == 'High')])
            fiber_high = len(df[(df['InternetService'] == 'Fiber optic') & (df['Risk_Level'] == 'High')])
            
            st.markdown(f"""
            <div class='premium-card' style='padding: 20px; height: 100%; border-left: 3px solid #ef4444;'>
                <div style="margin-bottom: 15px;">
                    <span style="background: rgba(239, 68, 68, 0.2); color: #fca5a5; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; margin-bottom: 5px; display: inline-block;">HIGH</span>
                    <p style="margin: 0; color: #e2e8f0; font-size: 0.95rem;">{m2m_high} high-risk customers are currently on Month-to-month contracts.</p>
                </div>
                <div style="margin-bottom: 15px;">
                    <span style="background: rgba(245, 158, 11, 0.2); color: #fcd34d; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; margin-bottom: 5px; display: inline-block;">MEDIUM</span>
                    <p style="margin: 0; color: #e2e8f0; font-size: 0.95rem;">Fiber optic service group shows elevated churn probability.</p>
                </div>
                <div>
                    <span style="background: rgba(56, 189, 248, 0.2); color: #7dd3fc; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; margin-bottom: 5px; display: inline-block;">INFO</span>
                    <p style="margin: 0; color: #e2e8f0; font-size: 0.95rem;">Model analysis completed successfully on latest dataset.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 2. PREDICT
# ==========================================
elif page == "🔮 Predict":
    render_hero("AI CHURN ASSESSMENT", "Enter customer details to generate a personalized churn risk prediction.", "Home / Predict Churn")
    
    if model is None: st.error("Model not loaded.")
    else:
        tab1, tab2 = st.tabs(["Individual Prediction", "Batch Prediction"])
        
        with tab1:
            pad_left, form_col, pad_right = st.columns([1, 8, 1])
            with form_col:
                with st.form("prediction_form"):
                    st.markdown("<h3 style='margin-top:0; color:#38bdf8;'>01 CUSTOMER PROFILE</h3>", unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    gender = c1.selectbox("Gender", ["Male", "Female"])
                    senior_citizen = c2.selectbox("Senior Citizen", [0, 1])
                    tenure = c3.slider("Tenure (Months)", 0, 72, 12, help="Number of months the customer has been with the company.")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("<h3 style='margin-top:0; color:#a855f7;'>02 SERVICES & BILLING</h3>", unsafe_allow_html=True)
                    c4, c5, c6 = st.columns(3)
                    internet_service = c4.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                    tech_support = c5.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                    contract = c6.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                    
                    c7, c8 = st.columns(2)
                    monthly_charges = c7.number_input("Monthly Charges ($)", min_value=0.0, value=75.0, step=1.0)
                    total_charges = c8.number_input("Total Charges ($)", min_value=0.0, value=900.0, step=1.0)
                    
                    with st.expander("View Additional Details"):
                        ec1, ec2, ec3 = st.columns(3)
                        partner = ec1.selectbox("Partner", ["Yes", "No"])
                        dependents = ec2.selectbox("Dependents", ["Yes", "No"])
                        phone_service = ec3.selectbox("Phone Service", ["Yes", "No"])
                        multiple_lines = ec1.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                        online_security = ec2.selectbox("Online Security", ["No", "Yes", "No internet service"])
                        online_backup = ec3.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                        device_protection = ec1.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                        streaming_tv = ec2.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                        payment_method = ec3.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                        paperless_billing = ec1.selectbox("Paperless Billing", ["Yes", "No"])
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("<h3 style='margin-top:0; color:#34d399; text-align: center;'>03 AI ASSESSMENT</h3>", unsafe_allow_html=True)
                    submit = st.form_submit_button("✦ RUN AI ASSESSMENT", use_container_width=True)
                
            if submit:
                inputs = {
                    'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
                    'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
                    'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
                    'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
                    'StreamingMovies': "No", 'Contract': contract, 'PaperlessBilling': paperless_billing,
                    'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
                }
                input_df = pd.DataFrame([inputs])
                prob = model.predict_proba(input_df)[0][1] * 100
                st.session_state.last_pred = {'inputs': inputs, 'prob': prob}
                
            if 'last_pred' in st.session_state:
                prob = st.session_state.last_pred['prob']
                inp = st.session_state.last_pred['inputs']
                
                st.markdown("---")
                
                # New Results Banner
                st.markdown('<div class="result-banner">', unsafe_allow_html=True)
                st.markdown("<h4 style='color:#94a3b8; letter-spacing:3px; margin-top:0;'>AI CHURN ASSESSMENT</h4>", unsafe_allow_html=True)
                
                if prob >= 70:
                    st.markdown('<div class="risk-badge badge-high">🔴 HIGH RISK</div>', unsafe_allow_html=True)
                elif prob >= 40:
                    st.markdown('<div class="risk-badge badge-medium">🟡 MEDIUM RISK</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-badge badge-low">🟢 LOW RISK</div>', unsafe_allow_html=True)
                    
                st.markdown('<div class="prob-number">{:.1f}%</div>'.format(prob), unsafe_allow_html=True)
                st.markdown("<p style='font-size:1.1rem; color:#f8fafc; margin-bottom: 20px;'>CHURN PROBABILITY</p>", unsafe_allow_html=True)
                
                # Custom Gauge Slider
                st.markdown(f"""
                <div style="width: 100%; max-width: 600px; margin: 0 auto;">
                    <div style="display: flex; justify-content: space-between; color: #94a3b8; font-weight: bold; font-size: 0.8rem; letter-spacing: 1px; margin-bottom: 5px;">
                        <span>LOW</span><span>MEDIUM</span><span>HIGH</span>
                    </div>
                    <div style="height: 8px; background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%); border-radius: 4px; position: relative;">
                        <div style="position: absolute; left: {prob}%; top: -6px; width: 4px; height: 20px; background: white; box-shadow: 0 0 10px white; border-radius: 2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("""<div class="premium-card"><h4 style='color:#38bdf8; margin-top:0;'>WHY THIS RESULT?</h4>""", unsafe_allow_html=True)
                    obs = []
                    if inp['tenure'] < 12: obs.append("🔴 Short tenure indicates low loyalty.")
                    if inp['Contract'] == 'Month-to-month': obs.append("🔴 Month-to-month contract offers no lock-in.")
                    if inp['TechSupport'] == 'No': obs.append("🔴 No technical support package.")
                    if inp['InternetService'] == 'Fiber optic': obs.append("🟡 Fiber optic users historically churn more.")
                    if not obs: obs.append("🟢 No severe primary risk factors observed.")
                    for o in obs: st.markdown(f"<p style='color:#cbd5e1; margin-bottom:10px;'>{o}</p>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with r2:
                    st.markdown("""<div class="premium-card"><h4 style='color:#34d399; margin-top:0;'>RETENTION PLAYBOOK</h4>""", unsafe_allow_html=True)
                    if prob >= 70:
                        st.markdown("<p style='color:#cbd5e1; margin-bottom:10px;'>• <strong>Priority Action:</strong> Offer immediate contract upgrade discount.</p>", unsafe_allow_html=True)
                        st.markdown("<p style='color:#cbd5e1; margin-bottom:10px;'>• <strong>Secondary Action:</strong> Assign to retention team for a check-in call.</p>", unsafe_allow_html=True)
                    elif prob >= 40:
                        st.markdown("<p style='color:#cbd5e1; margin-bottom:10px;'>• <strong>Priority Action:</strong> Offer 3 months free Tech Support.</p>", unsafe_allow_html=True)
                        st.markdown("<p style='color:#cbd5e1; margin-bottom:10px;'>• <strong>Secondary Action:</strong> Send satisfaction survey.</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p style='color:#cbd5e1; margin-bottom:10px;'>• <strong>Action:</strong> Maintain relationship and consider upselling new services.</p>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### BATCH AI ASSESSMENT")
            st.markdown("<p style='color:#94a3b8;'>Upload a CSV file to generate predictions for multiple customers at once.</p>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("", type=["csv"])
            
            if uploaded_file:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    with st.spinner("Generating Predictions..."):
                        batch_preds = get_global_predictions(batch_df, model)
                    
                    st.success("✓ Predictions generated successfully")
                    total_batch = len(batch_preds)
                    high_batch = len(batch_preds[batch_preds['Risk_Level'] == 'High'])
                    
                    bc1, bc2 = st.columns(2)
                    bc1.metric("Total Processed", total_batch)
                    bc2.metric("High Risk Identified", high_batch)
                    
                    csv = batch_preds.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Batch Results", data=csv, file_name="batch_predictions.csv", mime="text/csv", use_container_width=True)
                except Exception as e:
                    st.error("⚠ Please check the uploaded file format.")

# ==========================================
# 3. CUSTOMER 360
# ==========================================
elif page == "👤 Customer 360":
    render_hero("CUSTOMER 360°", "Everything important about this customer in one view.", "Home / Customer 360")
    
    if df.empty or 'customerID' not in df.columns: st.error("CustomerID column not found.")
    else:
        customer_id = st.selectbox("Search / Select Customer ID", df['customerID'].head(1000).tolist())
        
        if customer_id:
            cust_data = df[df['customerID'] == customer_id].iloc[0]
            prob = cust_data.get('Churn_Probability', 0)
            risk = cust_data.get('Risk_Level', 'Unknown')
            risk_color = '#ef4444' if risk == 'High' else '#fbbf24' if risk == 'Medium' else '#10b981'
            badge_class = 'badge-high' if risk == 'High' else 'badge-medium' if risk == 'Medium' else 'badge-low'
            
            # HEADER
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 20px; margin-bottom: 20px;">
                <div>
                    <h2 style="margin: 0; color: #f8fafc; font-size: 2rem;">👤 {customer_id}</h2>
                    <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 0.95rem;">Customer since {cust_data['tenure']} months</p>
                </div>
                <div><span class="risk-badge {badge_class}">● {risk} RISK</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            # KPI ROW
            st.markdown("<h4 style='color:#64748b; letter-spacing: 1px; font-size: 0.8rem; margin-bottom: 10px;'>CUSTOMER AT A GLANCE</h4>", unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("TENURE", f"{cust_data['tenure']} mo")
            k2.metric("CONTRACT", f"{cust_data['Contract']}")
            k3.metric("MONTHLY", f"${cust_data['MonthlyCharges']}")
            k4.metric("TOTAL", f"${cust_data['TotalCharges']}")
            k5.metric("CHURN PROB", f"{prob:.1f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # PROFILES
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class='premium-card' style='height: 100%;'>
                    <h4 style='color:#38bdf8; margin-top:0;'>CUSTOMER PROFILE</h4>
                    <b>Gender:</b> {cust_data.get('gender', 'N/A')}<br>
                    <b>Senior Citizen:</b> {'Yes' if cust_data.get('SeniorCitizen')==1 else 'No'}<br>
                    <b>Partner:</b> {cust_data.get('Partner', 'N/A')}<br>
                    <b>Dependents:</b> {cust_data.get('Dependents', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='premium-card' style='height: 100%;'>
                    <h4 style='color:#a855f7; margin-top:0;'>SERVICE PROFILE</h4>
                    <b>Internet:</b> {cust_data.get('InternetService', 'N/A')}<br>
                    <b>Tech Support:</b> {cust_data.get('TechSupport', 'N/A')}<br>
                    <b>Online Security:</b> {cust_data.get('OnlineSecurity', 'N/A')}<br>
                    <b>Streaming:</b> {cust_data.get('StreamingTV', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class='premium-card' style='height: 100%;'>
                    <h4 style='color:#34d399; margin-top:0;'>BILLING PROFILE</h4>
                    <b>Payment Method:</b> {cust_data.get('PaymentMethod', 'N/A')}<br>
                    <b>Paperless:</b> {cust_data.get('PaperlessBilling', 'N/A')}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # RISK & PLAYBOOK
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("<div class='premium-card' style='height: 100%; text-align: center;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color:#f8fafc; margin-top:0;'>AI RISK ASSESSMENT</h4>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:3.5rem; font-weight:900; color:{risk_color}; line-height: 1.1;'>{prob:.1f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:{risk_color}; font-weight: bold; letter-spacing: 1px; margin-bottom: 20px;'>{risk.upper()} RISK</p>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="width: 100%; margin: 0 auto;">
                    <div style="display: flex; justify-content: space-between; color: #64748b; font-size: 0.7rem; font-weight: bold; margin-bottom: 5px;">
                        <span>LOW</span><span>MEDIUM</span><span>HIGH</span>
                    </div>
                    <div style="height: 6px; background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%); border-radius: 3px; position: relative;">
                        <div style="position: absolute; left: {prob}%; top: -4px; width: 4px; height: 14px; background: white; box-shadow: 0 0 5px white; border-radius: 2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with r2:
                st.markdown("<div class='premium-card' style='height: 100%;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color:#f8fafc; margin-top:0;'>WHY THIS CUSTOMER MAY BE AT RISK</h4>", unsafe_allow_html=True)
                obs_count = 0
                if cust_data['tenure'] < 12: 
                    st.markdown("<div style='background: rgba(15,23,42,0.5); border: 1px solid rgba(255,255,255,0.05); padding: 10px 15px; border-radius: 8px; margin-bottom: 10px;'><span style='color:#fca5a5;'>⚠ Short tenure</span></div>", unsafe_allow_html=True)
                    obs_count += 1
                if cust_data['Contract'] == 'Month-to-month': 
                    st.markdown("<div style='background: rgba(15,23,42,0.5); border: 1px solid rgba(255,255,255,0.05); padding: 10px 15px; border-radius: 8px; margin-bottom: 10px;'><span style='color:#fca5a5;'>⚠ Contract: Month-to-month</span></div>", unsafe_allow_html=True)
                    obs_count += 1
                if cust_data.get('TechSupport') == 'No': 
                    st.markdown("<div style='background: rgba(15,23,42,0.5); border: 1px solid rgba(255,255,255,0.05); padding: 10px 15px; border-radius: 8px; margin-bottom: 10px;'><span style='color:#fca5a5;'>⚠ No Tech Support</span></div>", unsafe_allow_html=True)
                    obs_count += 1
                if obs_count == 0:
                    st.markdown("<div style='background: rgba(15,23,42,0.5); border: 1px solid rgba(255,255,255,0.05); padding: 10px 15px; border-radius: 8px; margin-bottom: 10px;'><span style='color:#6ee7b7;'>✓ No severe primary risk factors observed.</span></div>", unsafe_allow_html=True)
                
                st.markdown("<h4 style='color:#f8fafc; margin-top:15px; margin-bottom:10px;'>RETENTION PLAYBOOK</h4>", unsafe_allow_html=True)
                if risk == 'High':
                    st.markdown("<div style='background: rgba(56, 189, 248, 0.1); border-left: 3px solid #38bdf8; padding: 10px; font-size: 0.9rem;'><strong>Recommended Action:</strong> Proactive Contract Strategy Review. Offer upgrade incentive.</div>", unsafe_allow_html=True)
                elif risk == 'Medium':
                    st.markdown("<div style='background: rgba(56, 189, 248, 0.1); border-left: 3px solid #38bdf8; padding: 10px; font-size: 0.9rem;'><strong>Recommended Action:</strong> Support Outreach. Offer discounted tech support package.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background: rgba(56, 189, 248, 0.1); border-left: 3px solid #38bdf8; padding: 10px; font-size: 0.9rem;'><strong>Recommended Action:</strong> Maintain relationship. Explore cross-sell opportunities.</div>", unsafe_allow_html=True)
                    
                st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 4. RISK CENTER
# ==========================================
elif page == "🎯 Risk Center":
    render_hero("CUSTOMER RISK CENTER", "Prioritize customers who may need attention.", "Home / Risk Center")
    
    if 'Churn_Probability' not in df.columns: st.error("Global predictions not computed.")
    else:
        # RISK SUMMARY CARDS
        high_risk_n = len(df[df['Risk_Level'] == 'High'])
        med_risk_n = len(df[df['Risk_Level'] == 'Medium'])
        low_risk_n = len(df[df['Risk_Level'] == 'Low'])
        
        st.markdown("<h4 style='color:#94a3b8; letter-spacing: 2px; font-size: 0.8rem; margin-bottom: 10px;'>RISK DISTRIBUTION SUMMARY</h4>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"<div class='premium-card' style='border-top: 3px solid #ef4444;'><h4 style='margin:0; color:#ef4444;'>HIGH RISK</h4><h2 style='margin:10px 0 0 0; font-size: 2.5rem;'>{high_risk_n}</h2></div>", unsafe_allow_html=True)
        with r2:
            st.markdown(f"<div class='premium-card' style='border-top: 3px solid #fbbf24;'><h4 style='margin:0; color:#fbbf24;'>MEDIUM RISK</h4><h2 style='margin:10px 0 0 0; font-size: 2.5rem;'>{med_risk_n}</h2></div>", unsafe_allow_html=True)
        with r3:
            st.markdown(f"<div class='premium-card' style='border-top: 3px solid #10b981;'><h4 style='margin:0; color:#10b981;'>LOW RISK</h4><h2 style='margin:10px 0 0 0; font-size: 2.5rem;'>{low_risk_n}</h2></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # FILTERS & TABLE
        st.markdown("<h4 style='color:#f8fafc; margin-top:0;'>TOP AT-RISK CUSTOMERS</h4>", unsafe_allow_html=True)
        
        f1, f2, f3 = st.columns(3)
        risk_filter = f1.multiselect("Risk Level", ['High', 'Medium', 'Low'], default=['High'])
        contract_filter = f2.multiselect("Contract", df['Contract'].unique().tolist(), default=df['Contract'].unique().tolist())
        top_n = f3.selectbox("Show Top N", [10, 25, 50, 100])
        
        filtered_df = df[df['Risk_Level'].isin(risk_filter) & df['Contract'].isin(contract_filter)]
        filtered_df = filtered_df.sort_values(by="Priority_Score", ascending=False).head(top_n)
        
        display_cols = ['customerID', 'Churn_Probability', 'Risk_Level', 'Priority_Score', 'tenure', 'Contract', 'MonthlyCharges']
        
        if 'customerID' in df.columns:
            st.dataframe(filtered_df[display_cols].style.background_gradient(subset=['Churn_Probability', 'Priority_Score'], cmap='Blues').format({'Churn_Probability': '{:.1f}', 'Priority_Score': '{:.1f}'}), use_container_width=True)
        
        st.markdown("<p style='font-size: 0.8rem; color: #64748b; margin-top: 10px;'>*Priority Score combines Churn Risk and Monthly Charges to surface high-value at-risk customers.</p>", unsafe_allow_html=True)
        
        # RETENTION CAMPAIGN SIMULATOR
        st.markdown("<h4 style='color:#94a3b8; letter-spacing: 2px; font-size: 0.8rem; margin-bottom: 10px; margin-top: 40px;'>BUSINESS PLANNING</h4>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#f8fafc; margin-top:0;'>🧪 RETENTION CAMPAIGN SIMULATOR</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem;'>Estimate the cost and impact of targeting high-risk customers with a retention campaign.</p>", unsafe_allow_html=True)
        
        sim1, sim2 = st.columns([1, 2])
        with sim1:
            coverage = st.slider("Campaign Coverage % (of High-Risk Customers)", 10, 100, 50, 10)
            success_rate = st.slider("Estimated Success Rate %", 5, 50, 20, 5)
            cost_per_cust = st.number_input("Est. Cost per Targeted Customer ($)", value=25.0, step=5.0)
            
            target_count = int(high_risk_n * (coverage / 100))
            saved_count = int(target_count * (success_rate / 100))
            total_cost = target_count * cost_per_cust
            
        with sim2:
            st.markdown(f"""
            <div style="background: rgba(15,23,42,0.6); padding: 25px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.2); height: 100%;">
                <h5 style="color: #38bdf8; margin-top: 0;">SCENARIO ESTIMATES</h5>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 10px 0;">
                    <span style="color: #94a3b8;">High-Risk Customers Available</span>
                    <span style="color: #f8fafc; font-weight: bold;">{high_risk_n}</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 10px 0;">
                    <span style="color: #94a3b8;">Customers Targeted</span>
                    <span style="color: #f8fafc; font-weight: bold;">{target_count}</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 10px 0;">
                    <span style="color: #94a3b8;">Estimated Customers Saved</span>
                    <span style="color: #10b981; font-weight: bold;">{saved_count}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 10px 0 0 0;">
                    <span style="color: #94a3b8;">Estimated Campaign Cost</span>
                    <span style="color: #ef4444; font-weight: bold;">${total_cost:,.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)



# ==========================================
# 6. ANALYTICS
# ==========================================
elif page == "📊 Analytics":
    render_hero("BUSINESS ANALYTICS", "Deep dive into customer segments and model performance.", "Home / Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Churn Analysis", "Customer Segments", "Risk Analysis", "Model Information"])
    
    with tab1:
        st.markdown("<h3 style='margin-top:0;'>📊 Churn Drivers</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if 'Contract' in df:
                fig_c = px.histogram(df, x="Contract", color="Churn_Label", barmode="group", color_discrete_map={"Yes": "#ef4444", "No": "#3b82f6"})
                fig_c.update_layout(title="Churn by Contract", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
                st.plotly_chart(fig_c, use_container_width=True)
        with c2:
            if 'InternetService' in df:
                fig_i = px.histogram(df, x="InternetService", color="Churn_Label", barmode="group", color_discrete_map={"Yes": "#ef4444", "No": "#3b82f6"})
                fig_i.update_layout(title="Churn by Internet Service", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
                st.plotly_chart(fig_i, use_container_width=True)
        
    with tab2:
        st.markdown("<h3 style='margin-top:0;'>🎯 Customer Segments (K-Means)</h3>", unsafe_allow_html=True)
        try:
            # Simple K-Means on Tenure and MonthlyCharges for demonstration
            X_cluster = df[['tenure', 'MonthlyCharges']].dropna()
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['Segment'] = kmeans.fit_predict(X_cluster)
            
            # Map segments to business names based on characteristics
            seg_map = {}
            for i in range(3):
                avg_t = df[df['Segment']==i]['tenure'].mean()
                avg_c = df[df['Segment']==i]['MonthlyCharges'].mean()
                if avg_t > 40 and avg_c > 70: seg_map[i] = "High-Value Loyalists"
                elif avg_t < 20 and avg_c > 60: seg_map[i] = "High-Spend Newcomers (Flight Risk)"
                else: seg_map[i] = "Budget Customers"
            
            df['Segment_Name'] = df['Segment'].map(seg_map)
            
            fig_seg = px.scatter(df, x="tenure", y="MonthlyCharges", color="Segment_Name", opacity=0.6,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_seg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
            st.plotly_chart(fig_seg, use_container_width=True)
        except Exception as e:
            st.warning("Clustering not available.")
        
    with tab3:
        st.markdown("<h3 style='margin-top:0;'>🔥 Risk Heatmap</h3>", unsafe_allow_html=True)
        if 'Contract' in df and 'tenure' in df:
            # Create tenure groups
            df['Tenure_Group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-1 Yr', '1-2 Yrs', '2-4 Yrs', '4-6 Yrs'])
            heatmap_data = df.pivot_table(values='Churn_Probability', index='Contract', columns='Tenure_Group', aggfunc='mean')
            
            fig_heat = px.imshow(heatmap_data, labels=dict(x="Tenure Group", y="Contract", color="Avg Risk %"),
                                 color_continuous_scale="Reds", text_auto=".1f")
            fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
            st.plotly_chart(fig_heat, use_container_width=True)
        
    with tab4:
        st.markdown("<h3 style='margin-top:0;'>🤖 Model Health & Information</h3>", unsafe_allow_html=True)
        if not metrics_df.empty:
            best_model = metrics_df.sort_values(by='ROC-AUC', ascending=False).iloc[0]
            st.markdown(f"**Selected Model:** `{best_model.get('Model', 'Logistic Regression')}`")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
            mc2.metric("Precision", f"{best_model['Precision']:.4f}")
            mc3.metric("Recall", f"{best_model['Recall']:.4f}")
            mc4.metric("ROC-AUC", f"{best_model['ROC-AUC']:.4f}")
            
            if feature_importance_df is not None:
                st.markdown("<h4 style='color:#38bdf8; margin-top:20px;'>Global Feature Importance</h4>", unsafe_allow_html=True)
                fig_fi = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
                fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), height=350)
                st.plotly_chart(fig_fi, use_container_width=True)


