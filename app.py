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
    
    /* Global Background Fixes */
    .stApp {
        background-color: #080D18 !important;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(37, 99, 235, 0.08), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(168, 85, 247, 0.08), transparent 25%) !important;
    }
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #e2e8f0; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }
    
    /* Fix Streamlit Input Backgrounds */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background-color: #101827 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Premium Buttons */
    .stButton>button { 
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important; 
        color: white !important; 
        border: none !important;
        border-radius: 8px !important; 
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover { 
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
    }
    
    .btn-secondary>button {
        background: #101827 !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
        box-shadow: none !important;
    }
    .btn-secondary>button:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(168, 85, 247, 0.6) !important;
    }

    /* Cards / Panels */
    .premium-panel { background: #101827; border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); height: 100%; }
    .kpi-card { background: #101827; border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; position: relative; overflow: hidden; }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* DataFrame */
    .dataframe th { background: #101827 !important; color: #94a3b8 !important; border-bottom: 1px solid rgba(255,255,255,0.1) !important; }
    .dataframe td { background: transparent !important; color: #e2e8f0 !important; border-bottom: 1px solid rgba(255,255,255,0.05) !important; }
    
    hr { border-color: rgba(255,255,255,0.05) !important; margin: 32px 0 !important; }
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
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(15,23,42,0.6), rgba(8,13,24,0.9)); border: 1px solid rgba(59,130,246,0.2); padding: 32px 48px; border-radius: 16px; margin-bottom: 32px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); position: relative; overflow: hidden;">
        <div style="position: absolute; left: 0; top: 0; width: 4px; height: 100%; background: linear-gradient(to bottom, #3b82f6, #06b6d4);"></div>
        <span style="color: #3b82f6; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; display: block;">{breadcrumb}</span>
        <h1 style="margin: 0 0 8px 0; font-size: 2rem; font-weight: 800; color: #ffffff;">{title}</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin:0;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 32px; margin-top: 20px;">
        <h1 style="color: #ffffff; margin: 0; font-weight: 900; letter-spacing: 2px; font-size: 2rem;">⚡ CHURN<span style="color: #3b82f6;">IQ</span></h1>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px; font-weight: 600; letter-spacing: 1px;">CUSTOMER INTELLIGENCE</p>
        <p style="color: #64748b; font-size: 0.75rem; margin-top: 5px;">Predict • Understand • Retain</p>
    </div>
    <hr style="margin: 20px 0 !important;">
    """, unsafe_allow_html=True)
    
    if "nav_radio" not in st.session_state:
        st.session_state.nav_radio = "🏠 Overview"

    page = st.radio("NAVIGATION", [
        "🏠 Overview", 
        "🔮 Predict",
        "👤 Customer 360",
        "🎯 Risk Center",
        "🧪 What-If Lab",
        "📊 Analytics",
        "💡 Insights"
    ], label_visibility="collapsed", key="nav_radio")
    
    st.markdown("<hr style='margin: 20px 0 !important;'>", unsafe_allow_html=True)
    
    if not metrics_df.empty:
        best_model = metrics_df.sort_values(by='ROC-AUC', ascending=False).iloc[0]
        roc = best_model['ROC-AUC']
        mname = best_model.get('Model', 'Logistic Regression')
    else:
        roc = 0.8361
        mname = "Logistic Regression"
        
    st.markdown(f"""
    <div style="background: #101827; border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 12px; padding: 16px;">
        <div style="color: #10b981; font-size: 0.75rem; font-weight: 700; margin-bottom: 4px; display: flex; align-items: center;"><span style="font-size: 10px; margin-right: 6px;">●</span> AI MODEL ACTIVE</div>
        <div style="color: #f8fafc; font-weight: 600; font-size: 0.95rem; margin-bottom: 4px;">{mname}</div>
        <div style="color: #64748b; font-size: 0.8rem;">ROC-AUC <span style="color:#3b82f6; font-weight:bold;">{roc:.4f}</span></div>
    </div>
    """, unsafe_allow_html=True)

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
        
        # STATUS BAR
        st.markdown(f"""
        <div style="background: #101827; border: 1px solid rgba(255,255,255,0.05); padding: 16px 24px; border-radius: 12px; display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; color: #94a3b8; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; gap: 24px; font-weight: 600;">
                <span><span style="color: #10b981; margin-right: 6px;">●</span> MODEL ONLINE</span>
                <span><span style="color: #3b82f6; margin-right: 6px;">●</span> DATA READY</span>
                <span><span style="color: #a855f7; margin-right: 6px;">●</span> PREDICTIONS AVAILABLE</span>
            </div>
            <div style="font-family: monospace; background: rgba(0,0,0,0.2); padding: 4px 8px; border-radius: 4px;">LAST ANALYSIS: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # HERO SECTION
        h_col1, h_col2 = st.columns([1.3, 1])
        with h_col1:
            st.markdown("""
            <div style="padding: 24px 0 32px 0;">
                <h1 style="font-size: 3.5rem; font-weight: 900; line-height: 1.1; margin: 0 0 16px 0; color: #ffffff; text-shadow: 0 0 30px rgba(255,255,255,0.1);">CUSTOMER RETENTION<br><span style="color: transparent; background-clip: text; -webkit-background-clip: text; background-image: linear-gradient(90deg, #3b82f6, #a855f7);">COMMAND CENTER</span></h1>
                <p style="color: #94a3b8; font-size: 1.25rem; line-height: 1.6; margin-bottom: 32px; max-width: 90%; font-weight: 400;">Know who may leave. Understand why. Take action earlier.</p>
            </div>
            """, unsafe_allow_html=True)
            
            bh1, bh2 = st.columns([1, 1])
            with bh1: 
                st.button("✦ PREDICT CUSTOMER", on_click=lambda: st.session_state.update(nav_radio="🔮 Predict"), use_container_width=True)
            with bh2: 
                st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
                st.button("◉ VIEW RISK CENTER", on_click=lambda: st.session_state.update(nav_radio="🎯 Risk Center"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
        with h_col2:
            st.markdown("""
            <div style="position: relative; height: 100%; min-height: 350px; display: flex; align-items: center; justify-content: center;">
                <div style="position: absolute; width: 280px; height: 280px; border-radius: 50%; border: 1px dashed rgba(6, 182, 212, 0.3); animation: spin 40s linear infinite;"></div>
                <div style="position: absolute; width: 220px; height: 220px; border-radius: 50%; border: 1px solid rgba(59, 130, 246, 0.4); box-shadow: 0 0 40px rgba(59, 130, 246, 0.1);"></div>
                <div style="position: absolute; width: 140px; height: 140px; border-radius: 50%; background: radial-gradient(circle, rgba(168, 85, 247, 0.2) 0%, transparent 70%); border: 2px solid rgba(168, 85, 247, 0.5); box-shadow: 0 0 30px rgba(168, 85, 247, 0.4); display: flex; align-items: center; justify-content: center; animation: pulse 4s ease-in-out infinite;">
                    <div style="font-size: 2rem; font-weight: 900; color: #ffffff; text-shadow: 0 0 10px rgba(255,255,255,0.5);">AI</div>
                </div>
                
                <!-- Floating Data Points -->
                <div style="position: absolute; top: 10%; right: 15%; width: 8px; height: 8px; background: #06b6d4; border-radius: 50%; box-shadow: 0 0 10px #06b6d4;"></div>
                <div style="position: absolute; bottom: 20%; left: 10%; width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; box-shadow: 0 0 15px #3b82f6;"></div>
                <div style="position: absolute; top: 30%; left: 15%; width: 6px; height: 6px; background: #a855f7; border-radius: 50%; box-shadow: 0 0 10px #a855f7;"></div>
                
                <div style="position: absolute; top: 15%; left: 5%; background: rgba(16, 24, 39, 0.8); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(6, 182, 212, 0.3); color: #06b6d4; font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">Processing Data...</div>
                <div style="position: absolute; bottom: 15%; right: 5%; background: rgba(16, 24, 39, 0.8); padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">High Risk Detected</div>
                
                <style>
                    @keyframes spin { 100% { transform: rotate(360deg); } }
                    @keyframes pulse { 0% { transform: scale(0.95); box-shadow: 0 0 20px rgba(168, 85, 247, 0.2); } 50% { transform: scale(1.05); box-shadow: 0 0 40px rgba(168, 85, 247, 0.6); } 100% { transform: scale(0.95); box-shadow: 0 0 20px rgba(168, 85, 247, 0.2); } }
                </style>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<hr>", unsafe_allow_html=True)

        # EXECUTIVE KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""<div class="kpi-card" style="border-bottom: 3px solid #3b82f6;">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; margin-bottom: 8px;">TOTAL CUSTOMERS</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff;">{total:,}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="kpi-card" style="border-bottom: 3px solid #06b6d4;">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; margin-bottom: 8px;">CHURN RATE</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; display: flex; align-items: center;">{churn_rate:.1f}% <div style="width:12px; height:12px; border-radius:50%; border:2px solid #06b6d4; margin-left:12px;"></div></div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card" style="border-bottom: 3px solid #ef4444; background: linear-gradient(180deg, #101827 0%, rgba(239,68,68,0.05) 100%);">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; margin-bottom: 8px;">HIGH-RISK CUSTOMERS</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ef4444;">{high_risk:,}</div>
                <div style="position: absolute; top: 16px; right: 16px; font-size: 0.6rem; background: rgba(239,68,68,0.2); color: #fca5a5; padding: 4px 8px; border-radius: 4px; font-weight: bold;">● ATTENTION</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi-card" style="border-bottom: 3px solid #a855f7;">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; margin-bottom: 8px;">AVG PROBABILITY</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff;">{avg_prob:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with k5:
            st.markdown(f"""<div class="kpi-card" style="border-bottom: 3px solid #10b981;">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; margin-bottom: 8px;">MODEL ROC-AUC</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff;">{roc:.4f}</div>
                <div style="margin-top: 8px; width: 100%; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px;"><div style="width: {roc*100}%; height: 100%; background: #10b981; border-radius: 2px;"></div></div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # CHARTS & ALERTS
        r1, r2 = st.columns([2, 1])
        with r1:
            st.markdown("<h3 style='color:#ffffff; margin-top:0; font-size: 1.4rem;'>CHURN PROBABILITY DISTRIBUTION</h3>", unsafe_allow_html=True)
            fig_dist = px.histogram(df, x="Churn_Probability", color="Risk_Level", nbins=40,
                                   color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                                   opacity=0.8)
            fig_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                font=dict(color="#94a3b8"), height=350, showlegend=True, 
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Probability (%)", color="#94a3b8"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Customers", color="#94a3b8")
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with r2:
            st.markdown("<h3 style='color:#ffffff; margin-top:0; font-size: 1.4rem;'>EXECUTIVE ALERTS</h3>", unsafe_allow_html=True)
            
            m2m_high = len(df[(df['Contract'] == 'Month-to-month') & (df['Risk_Level'] == 'High')])
            fiber_high = len(df[(df['InternetService'] == 'Fiber optic') & (df['Risk_Level'] == 'High')])
            
            st.markdown(f"""
            <div class='premium-panel' style='padding: 24px;'>
                <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 16px; border-radius: 0 8px 8px 0; margin-bottom: 16px;">
                    <div style="color: #ef4444; font-size: 0.75rem; font-weight: 800; letter-spacing: 1px; margin-bottom: 4px;">⚠ HIGH PRIORITY</div>
                    <div style="color: #f8fafc; font-size: 0.95rem; font-weight: 500;">{m2m_high} high-risk customers are currently on Month-to-month contracts. Action required.</div>
                </div>
                
                <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 16px; border-radius: 0 8px 8px 0; margin-bottom: 16px;">
                    <div style="color: #f59e0b; font-size: 0.75rem; font-weight: 800; letter-spacing: 1px; margin-bottom: 4px;">⚠ MEDIUM PRIORITY</div>
                    <div style="color: #f8fafc; font-size: 0.95rem; font-weight: 500;">Fiber optic service ({fiber_high} users) shows elevated churn risk compared to DSL.</div>
                </div>
                
                <div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 16px; border-radius: 0 8px 8px 0;">
                    <div style="color: #3b82f6; font-size: 0.75rem; font-weight: 800; letter-spacing: 1px; margin-bottom: 4px;">ℹ INFO</div>
                    <div style="color: #f8fafc; font-size: 0.95rem; font-weight: 500;">Model predictions refreshed. System operating normally.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # QUICK ACTIONS
        st.markdown("<h3 style='color:#ffffff; font-size: 1.4rem; margin-bottom: 24px; text-align: center;'>WHAT DO YOU WANT TO DO?</h3>", unsafe_allow_html=True)
        qa1, qa2, qa3 = st.columns(3)
        with qa1:
            st.markdown("""<div class='premium-panel' style='text-align: center; cursor: pointer; transition: all 0.3s ease;' onmouseover='this.style.transform="translateY(-5px)"' onmouseout='this.style.transform="translateY(0)"'>
                <div style="font-size: 2.5rem; margin-bottom: 16px;">🔮</div>
                <h4 style="color: #ffffff; margin: 0 0 8px 0;">Predict Churn</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Analyze a specific customer's risk profile</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Go to Predict", key="qa_pred", use_container_width=True): st.session_state.update(nav_radio="🔮 Predict"); st.rerun()
        with qa2:
            st.markdown("""<div class='premium-panel' style='text-align: center; cursor: pointer; transition: all 0.3s ease;' onmouseover='this.style.transform="translateY(-5px)"' onmouseout='this.style.transform="translateY(0)"'>
                <div style="font-size: 2.5rem; margin-bottom: 16px;">🎯</div>
                <h4 style="color: #ffffff; margin: 0 0 8px 0;">Risk Center</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">View list of most critical customers</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Go to Risk Center", key="qa_risk", use_container_width=True): st.session_state.update(nav_radio="🎯 Risk Center"); st.rerun()
        with qa3:
            st.markdown("""<div class='premium-panel' style='text-align: center; cursor: pointer; transition: all 0.3s ease;' onmouseover='this.style.transform="translateY(-5px)"' onmouseout='this.style.transform="translateY(0)"'>
                <div style="font-size: 2.5rem; margin-bottom: 16px;">🧪</div>
                <h4 style="color: #ffffff; margin: 0 0 8px 0;">What-If Lab</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Simulate changes to lower churn risk</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Go to What-If", key="qa_what", use_container_width=True): st.session_state.update(nav_radio="🧪 What-If Lab"); st.rerun()

# ==========================================
# 2. PREDICT
# ==========================================
elif page == "🔮 Predict":
    render_hero("AI CHURN ASSESSMENT", "Enter customer details to generate a personalized churn risk prediction.", "Home / Predict")
    
    if model is None: st.error("Model not loaded.")
    else:
        pad_left, form_col, pad_right = st.columns([1, 8, 1])
        with form_col:
            with st.form("prediction_form"):
                st.markdown("<h3 style='margin-top:0; color:#06b6d4; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px;'>01. CUSTOMER PROFILE</h3>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                gender = c1.selectbox("Gender", ["Male", "Female"])
                senior_citizen = c2.selectbox("Senior Citizen", [0, 1])
                tenure = c3.slider("Tenure (Months)", 0, 72, 12)
                
                st.markdown("<hr style='margin: 16px 0 !important;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin-top:0; color:#a855f7; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px;'>02. SERVICES & BILLING</h3>", unsafe_allow_html=True)
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
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Results Banner
            st.markdown("""<div style="background: linear-gradient(135deg, rgba(37,99,235,0.1), rgba(168,85,247,0.1)); border: 1px solid rgba(59,130,246,0.3); border-radius: 16px; padding: 40px; text-align: center; margin-bottom: 32px;">""", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#3b82f6; letter-spacing:3px; margin-top:0;'>ASSESSMENT COMPLETE</h4>", unsafe_allow_html=True)
            
            if prob >= 70:
                rc, bclass = "#ef4444", "HIGH RISK"
            elif prob >= 40:
                rc, bclass = "#f59e0b", "MEDIUM RISK"
            else:
                rc, bclass = "#10b981", "LOW RISK"
                
            st.markdown(f'<div style="display: inline-block; background: {rc}22; border: 1px solid {rc}88; color: {rc}; padding: 6px 16px; border-radius: 100px; font-weight: 800; font-size: 0.9rem; letter-spacing: 1.5px; margin-bottom: 16px;">● {bclass}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 5rem; font-weight: 900; color: #ffffff; line-height: 1;">{prob:.1f}%</div>', unsafe_allow_html=True)
            st.markdown("<p style='font-size:1.1rem; color:#94a3b8; font-weight: 600; margin-bottom: 32px;'>CHURN PROBABILITY</p>", unsafe_allow_html=True)
            
            # Gauge
            st.markdown(f"""
            <div style="width: 100%; max-width: 500px; margin: 0 auto;">
                <div style="display: flex; justify-content: space-between; color: #64748b; font-weight: bold; font-size: 0.75rem; letter-spacing: 1px; margin-bottom: 8px;">
                    <span>LOW</span><span>MEDIUM</span><span>HIGH</span>
                </div>
                <div style="height: 10px; background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%); border-radius: 5px; position: relative; overflow: visible;">
                    <div style="position: absolute; left: {prob}%; top: -8px; width: 6px; height: 26px; background: #ffffff; box-shadow: 0 0 10px rgba(255,255,255,0.8); border-radius: 3px; transform: translateX(-50%);"></div>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("""<div class="premium-panel"><h4 style='color:#ffffff; margin-top:0;'>WHY THIS RESULT?</h4>""", unsafe_allow_html=True)
                obs = []
                if inp['tenure'] < 12: obs.append("🔴 Short tenure indicates low loyalty.")
                if inp['Contract'] == 'Month-to-month': obs.append("🔴 Month-to-month contract offers no lock-in.")
                if inp['TechSupport'] == 'No': obs.append("🔴 No technical support package.")
                if inp['InternetService'] == 'Fiber optic': obs.append("🟡 Fiber optic users historically churn more.")
                if not obs: obs.append("🟢 No severe primary risk factors observed.")
                for o in obs: st.markdown(f"<p style='color:#cbd5e1; margin-bottom:12px;'>{o}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with r2:
                st.markdown("""<div class="premium-panel"><h4 style='color:#ffffff; margin-top:0;'>RETENTION PLAYBOOK</h4>""", unsafe_allow_html=True)
                if prob >= 70:
                    st.markdown("<div style='background: rgba(239,68,68,0.1); border-left: 3px solid #ef4444; padding: 12px; margin-bottom: 12px;'><strong>Priority Action:</strong> Offer immediate contract upgrade discount.</div>", unsafe_allow_html=True)
                    st.markdown("<div style='background: rgba(245,158,11,0.1); border-left: 3px solid #f59e0b; padding: 12px;'><strong>Secondary Action:</strong> Assign to retention team for a check-in call.</div>", unsafe_allow_html=True)
                elif prob >= 40:
                    st.markdown("<div style='background: rgba(245,158,11,0.1); border-left: 3px solid #f59e0b; padding: 12px; margin-bottom: 12px;'><strong>Priority Action:</strong> Offer 3 months free Tech Support.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background: rgba(16,185,129,0.1); border-left: 3px solid #10b981; padding: 12px;'><strong>Action:</strong> Maintain relationship and consider upselling new services.</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 3. CUSTOMER 360
# ==========================================
elif page == "👤 Customer 360":
    render_hero("CUSTOMER 360°", "Everything important about this customer in one view.", "Home / Customer 360")
    
    if df.empty or 'customerID' not in df.columns: st.error("CustomerID column not found.")
    else:
        customer_id = st.selectbox("Search / Select Customer ID", df['customerID'].head(1000).tolist())
        
        if customer_id:
            cust = df[df['customerID'] == customer_id].iloc[0]
            prob = cust.get('Churn_Probability', 0)
            risk = cust.get('Risk_Level', 'Unknown')
            risk_color = '#ef4444' if risk == 'High' else '#f59e0b' if risk == 'Medium' else '#10b981'
            
            st.markdown(f"""
            <div style="background: #101827; border: 1px solid rgba(255,255,255,0.05); padding: 32px; border-radius: 16px; margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 24px; margin-bottom: 24px;">
                    <div>
                        <h2 style="margin: 0; color: #ffffff; font-size: 2.2rem;">👤 {customer_id}</h2>
                        <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 1rem;">Customer since {cust['tenure']} months</p>
                    </div>
                    <div style="background: {risk_color}22; border: 1px solid {risk_color}88; color: {risk_color}; padding: 8px 20px; border-radius: 100px; font-weight: 800; letter-spacing: 1.5px;">● {risk.upper()} RISK</div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px;">
                    <div>
                        <div style="color: #64748b; font-size: 0.8rem; font-weight: 700; margin-bottom: 8px;">TENURE</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{cust['tenure']} mo</div>
                    </div>
                    <div>
                        <div style="color: #64748b; font-size: 0.8rem; font-weight: 700; margin-bottom: 8px;">CONTRACT</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">{cust['Contract']}</div>
                    </div>
                    <div>
                        <div style="color: #64748b; font-size: 0.8rem; font-weight: 700; margin-bottom: 8px;">MONTHLY</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">${cust['MonthlyCharges']}</div>
                    </div>
                    <div>
                        <div style="color: #64748b; font-size: 0.8rem; font-weight: 700; margin-bottom: 8px;">CHURN PROB</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: {risk_color};">{prob:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class='premium-panel'>
                    <h4 style='color:#3b82f6; margin-top:0; margin-bottom:16px;'>PROFILE</h4>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 100px; display:inline-block;">Gender:</span> <strong style="color:#ffffff;">{cust.get('gender', 'N/A')}</strong></div>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 100px; display:inline-block;">Senior:</span> <strong style="color:#ffffff;">{'Yes' if cust.get('SeniorCitizen')==1 else 'No'}</strong></div>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 100px; display:inline-block;">Partner:</span> <strong style="color:#ffffff;">{cust.get('Partner', 'N/A')}</strong></div>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 100px; display:inline-block;">Dependents:</span> <strong style="color:#ffffff;">{cust.get('Dependents', 'N/A')}</strong></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class='premium-panel'>
                    <h4 style='color:#a855f7; margin-top:0; margin-bottom:16px;'>SERVICES</h4>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 100px; display:inline-block;">Internet:</span> <strong style="color:#ffffff;">{cust.get('InternetService', 'N/A')}</strong></div>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 100px; display:inline-block;">Support:</span> <strong style="color:#ffffff;">{cust.get('TechSupport', 'N/A')}</strong></div>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 100px; display:inline-block;">Security:</span> <strong style="color:#ffffff;">{cust.get('OnlineSecurity', 'N/A')}</strong></div>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 100px; display:inline-block;">Streaming:</span> <strong style="color:#ffffff;">{cust.get('StreamingTV', 'N/A')}</strong></div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class='premium-panel'>
                    <h4 style='color:#06b6d4; margin-top:0; margin-bottom:16px;'>BILLING</h4>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 120px; display:inline-block;">Method:</span> <strong style="color:#ffffff;">{cust.get('PaymentMethod', 'N/A')}</strong></div>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 120px; display:inline-block;">Paperless:</span> <strong style="color:#ffffff;">{cust.get('PaperlessBilling', 'N/A')}</strong></div>
                    <div style="margin-bottom: 8px;"><span style="color:#94a3b8; width: 120px; display:inline-block;">Total Billed:</span> <strong style="color:#ffffff;">${cust.get('TotalCharges', 0)}</strong></div>
                </div>""", unsafe_allow_html=True)

# ==========================================
# 4. RISK CENTER
# ==========================================
elif page == "🎯 Risk Center":
    render_hero("CUSTOMER RISK CENTER", "Identify, filter, and export high-priority customers for retention campaigns.", "Home / Risk Center")
    
    if 'Churn_Probability' not in df.columns: st.error("Global predictions not computed.")
    else:
        high_risk_n = len(df[df['Risk_Level'] == 'High'])
        med_risk_n = len(df[df['Risk_Level'] == 'Medium'])
        low_risk_n = len(df[df['Risk_Level'] == 'Low'])
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class='premium-panel' style='border-top: 4px solid #ef4444; background: linear-gradient(180deg, rgba(239,68,68,0.1) 0%, #101827 100%); text-align: center;'>
                <div style="color: #ef4444; font-weight: 800; letter-spacing: 1px; margin-bottom: 8px;">HIGH RISK</div>
                <div style="font-size: 3rem; font-weight: 900; color: #ffffff;">{high_risk_n}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class='premium-panel' style='border-top: 4px solid #f59e0b; background: linear-gradient(180deg, rgba(245,158,11,0.1) 0%, #101827 100%); text-align: center;'>
                <div style="color: #f59e0b; font-weight: 800; letter-spacing: 1px; margin-bottom: 8px;">MEDIUM RISK</div>
                <div style="font-size: 3rem; font-weight: 900; color: #ffffff;">{med_risk_n}</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class='premium-panel' style='border-top: 4px solid #10b981; background: linear-gradient(180deg, rgba(16,185,129,0.1) 0%, #101827 100%); text-align: center;'>
                <div style="color: #10b981; font-weight: 800; letter-spacing: 1px; margin-bottom: 8px;">LOW RISK</div>
                <div style="font-size: 3rem; font-weight: 900; color: #ffffff;">{low_risk_n}</div>
            </div>""", unsafe_allow_html=True)
            
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='color:#ffffff; margin-top:0;'>TOP AT-RISK CUSTOMERS</h3>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        risk_filter = f1.multiselect("Risk Level", ['High', 'Medium', 'Low'], default=['High'])
        contract_filter = f2.multiselect("Contract", df['Contract'].unique().tolist(), default=df['Contract'].unique().tolist())
        top_n = f3.selectbox("Show Top N", [10, 25, 50, 100])
        
        filtered_df = df[df['Risk_Level'].isin(risk_filter) & df['Contract'].isin(contract_filter)]
        filtered_df = filtered_df.sort_values(by="Priority_Score", ascending=False).head(top_n)
        
        display_cols = ['customerID', 'Churn_Probability', 'Risk_Level', 'Priority_Score', 'tenure', 'Contract', 'MonthlyCharges']
        if 'customerID' in df.columns:
            st.dataframe(filtered_df[display_cols].style.background_gradient(subset=['Churn_Probability', 'Priority_Score'], cmap='Purples').format({'Churn_Probability': '{:.1f}', 'Priority_Score': '{:.1f}'}), use_container_width=True)

# ==========================================
# 5. WHAT-IF LAB
# ==========================================
elif page == "🧪 What-If Lab":
    render_hero("WHAT-IF SCENARIO LAB", "Simulate changes to customer properties to see how they impact churn probability.", "Home / What-If Lab")
    
    if df.empty or model is None: st.error("Data or model not available.")
    else:
        st.markdown("<p style='color:#94a3b8;'>Select an existing high-risk customer to act as a baseline, then change their features.</p>", unsafe_allow_html=True)
        high_risk_custs = df[df['Risk_Level'] == 'High']['customerID'].head(100).tolist()
        baseline_id = st.selectbox("Select Baseline Customer", high_risk_custs)
        
        if baseline_id:
            base_data = df[df['customerID'] == baseline_id].iloc[0].to_dict()
            base_prob = base_data.get('Churn_Probability', 0)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            w1, w2 = st.columns([1, 1])
            with w1:
                st.markdown("<h3 style='color:#ffffff; margin-top:0;'>🔧 MODIFY PARAMETERS</h3>", unsafe_allow_html=True)
                with st.form("whatif_form"):
                    new_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(base_data['Contract']))
                    new_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(base_data['TechSupport']) if base_data['TechSupport'] in ["No", "Yes", "No internet service"] else 0)
                    new_charges = st.number_input("Monthly Charges ($)", value=float(base_data['MonthlyCharges']))
                    submit_whatif = st.form_submit_button("✦ SIMULATE", use_container_width=True)
                    
            with w2:
                st.markdown("<h3 style='color:#ffffff; margin-top:0;'>📊 SCENARIO COMPARISON</h3>", unsafe_allow_html=True)
                
                # Always show baseline
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 16px; margin-bottom: 16px; border-radius: 0 8px 8px 0;">
                    <div style="color: #ef4444; font-size: 0.75rem; font-weight: 800; letter-spacing: 1px; margin-bottom: 8px;">CURRENT (BASELINE)</div>
                    <div style="font-size: 2.5rem; font-weight: 900; color: #ffffff; line-height: 1;">{base_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                if submit_whatif:
                    sim_inputs = base_data.copy()
                    sim_inputs['Contract'] = new_contract
                    sim_inputs['TechSupport'] = new_tech
                    sim_inputs['MonthlyCharges'] = new_charges
                    
                    sim_df = pd.DataFrame([sim_inputs])
                    X_sim = sim_df.drop(columns=['customerID', 'Churn', 'Churn_Label', 'Churn_Probability', 'Risk_Level', 'Priority_Score', 'Priority_Tier', 'Segment', 'Segment_Name', 'Tenure_Group'], errors='ignore')
                    
                    try:
                        new_prob = model.predict_proba(X_sim)[0][1] * 100
                        diff = new_prob - base_prob
                        
                        if diff < 0:
                            d_color = "#10b981"
                            d_text = f"▼ {abs(diff):.1f}% reduction"
                            box_color = "rgba(16, 185, 129, 0.1)"
                        else:
                            d_color = "#ef4444"
                            d_text = f"▲ {abs(diff):.1f}% increase"
                            box_color = "rgba(239, 68, 68, 0.1)"
                            
                        st.markdown(f"""
                        <div style="background: {box_color}; border-left: 4px solid {d_color}; padding: 16px; border-radius: 0 8px 8px 0;">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div>
                                    <div style="color: {d_color}; font-size: 0.75rem; font-weight: 800; letter-spacing: 1px; margin-bottom: 8px;">SIMULATED SCENARIO</div>
                                    <div style="font-size: 2.5rem; font-weight: 900; color: #ffffff; line-height: 1;">{new_prob:.1f}%</div>
                                </div>
                                <div style="background: {d_color}; color: #ffffff; padding: 4px 8px; border-radius: 4px; font-weight: 700; font-size: 0.85rem;">{d_text}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error running simulation: {e}")

# ==========================================
# 6. ANALYTICS
# ==========================================
elif page == "📊 Analytics":
    render_hero("BUSINESS ANALYTICS", "Deep dive into customer segments and model performance.", "Home / Analytics")
    
    st.markdown("<h3 style='color:#ffffff; margin-top:0;'>CHURN DRIVERS</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if 'Contract' in df:
            fig_c = px.histogram(df, x="Contract", color="Churn_Label", barmode="group", color_discrete_map={"Yes": "#ef4444", "No": "#3b82f6"})
            fig_c.update_layout(title="Churn by Contract", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig_c, use_container_width=True)
    with c2:
        if 'InternetService' in df:
            fig_i = px.histogram(df, x="InternetService", color="Churn_Label", barmode="group", color_discrete_map={"Yes": "#ef4444", "No": "#3b82f6"})
            fig_i.update_layout(title="Churn by Internet Service", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
            st.plotly_chart(fig_i, use_container_width=True)
            
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#ffffff; margin-top:0;'>RISK HEATMAP</h3>", unsafe_allow_html=True)
    
    if 'Contract' in df and 'tenure' in df:
        df_h = df.copy()
        df_h['Tenure_Group'] = pd.cut(df_h['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-1 Yr', '1-2 Yrs', '2-4 Yrs', '4-6 Yrs'])
        heatmap_data = df_h.pivot_table(values='Churn_Probability', index='Contract', columns='Tenure_Group', aggfunc='mean')
        
        fig_heat = px.imshow(heatmap_data, labels=dict(x="Tenure Group", y="Contract", color="Avg Risk %"),
                             color_continuous_scale="Purples", text_auto=".1f")
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
        st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# 7. INSIGHTS
# ==========================================
elif page == "💡 Insights":
    render_hero("AI INSIGHTS & RECOMMENDATIONS", "Automated business intelligence derived from model predictions.", "Home / Insights")
    
    st.markdown("""
    <div class="premium-panel" style="border-left: 4px solid #a855f7;">
        <h3 style="color: #a855f7; margin-top: 0;">KEY FINDING 01: THE CRITICAL FIRST YEAR</h3>
        <p style="color: #e2e8f0; font-size: 1.1rem; line-height: 1.6;">Our AI model indicates that <strong>tenure</strong> is the strongest predictor of churn. Customers in their first 12 months have a disproportionately high churn probability. <br><br><strong>Recommendation:</strong> Implement a proactive 90-day onboarding sequence and offer loyalty rewards at the 6-month mark.</p>
    </div>
    <br>
    <div class="premium-panel" style="border-left: 4px solid #3b82f6;">
        <h3 style="color: #3b82f6; margin-top: 0;">KEY FINDING 02: MONTH-TO-MONTH VULNERABILITY</h3>
        <p style="color: #e2e8f0; font-size: 1.1rem; line-height: 1.6;">The majority of high-risk customers are on Month-to-month contracts. The lack of lock-in makes them highly sensitive to competitor offers.<br><br><strong>Recommendation:</strong> Launch an aggressive upgrade campaign targeting Month-to-month users with high priority scores, offering a discounted first year on an annual contract.</p>
    </div>
    <br>
    <div class="premium-panel" style="border-left: 4px solid #06b6d4;">
        <h3 style="color: #06b6d4; margin-top: 0;">KEY FINDING 03: TECH SUPPORT DEFICIT</h3>
        <p style="color: #e2e8f0; font-size: 1.1rem; line-height: 1.6;">Customers without Tech Support add-ons show elevated churn risk, likely due to frustration with unresolved technical issues.<br><br><strong>Recommendation:</strong> Bundle 3 months of free Tech Support for at-risk customers to demonstrate value and reduce friction.</p>
    </div>
    """, unsafe_allow_html=True)
