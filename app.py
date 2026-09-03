import os
import sys
import time
import warnings
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="FraudWatch | Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    .badge-low {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .badge-medium {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .badge-high {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E293B;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .result-card {
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-left: 6px solid;
    }
    
    .result-card-low {
        background: #F0FDF4;
        border-left-color: #22C55E;
        border: 1px solid #DCFCE7;
    }
    
    .result-card-medium {
        background: #FFFBEB;
        border-left-color: #F59E0B;
        border: 1px solid #FEF3C7;
    }
    
    .result-card-high {
        background: #FEF2F2;
        border-left-color: #EF4444;
        border: 1px solid #FEE2E2;
    }
</style>
""", unsafe_allow_html=True)

# Load Model Artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, "models", "preprocessing_bundle.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_final_model.pkl")

@st.cache_resource(show_spinner="Loading Preprocessor & Machine Learning Model...")
def load_model_and_bundle():
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"Preprocessing bundle not found at {BUNDLE_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"XGBoost model not found at {MODEL_PATH}")
    
    bundle = joblib.load(BUNDLE_PATH)
    model = joblib.load(MODEL_PATH)
    
    model_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else bundle.get("feature_cols", [])
    cols_to_drop = {'TransactionDT_days', 'TransactionDT', 'Transaction_day'}
    model_features = [c for c in model_features if c not in cols_to_drop]
    
    return bundle, model, model_features

try:
    preprocessing_bundle, xgb_model, model_feature_names = load_model_and_bundle()
    models_ready = True
except Exception as e:
    st.error(f"❌ Failed to load model artifacts: {str(e)}")
    models_ready = False

# Preprocessing Pipeline
def preprocess_dataframe(df_raw: pd.DataFrame, bundle: dict, model_features: list) -> pd.DataFrame:
    df = df_raw.copy()
    
    # Time features
    if "TransactionDT" in df.columns:
        dt = pd.to_numeric(df["TransactionDT"], errors='coerce')
        dt_days = dt / (24 * 60 * 60)
        day = np.floor(dt_days)
        df["Transaction_hour"] = (dt // 3600) % 24
        df["Transaction_weekday"] = day % 7
    else:
        if "Transaction_hour" not in df.columns:
            df["Transaction_hour"] = 12.0
        if "Transaction_weekday" not in df.columns:
            df["Transaction_weekday"] = 3.0
            
    # Transaction amount log transform
    if "TransactionAmt" in df.columns:
        df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors='coerce').fillna(0.0)
        if "TransactionAmt_log" not in df.columns:
            df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].clip(lower=0))
    else:
        df["TransactionAmt"] = 0.0
        df["TransactionAmt_log"] = 0.0

    mappings = bundle.get("encoding_mappings", {})
    
    # Target encodings
    target_enc_map = {
        "ProductCD": "ProductCD_target_enc",
        "card4": "card4_target_enc",
        "card6": "card6_target_enc",
        "DeviceType": "DeviceType_target_enc",
        "P_emaildomain": "P_emaildomain_target_enc"
    }
    
    for orig_col, enc_col in target_enc_map.items():
        if enc_col in mappings:
            mapping = mappings[enc_col]
            global_mean = 0.035
            if hasattr(mapping, "mean"):
                global_mean = float(mapping.mean())
            elif isinstance(mapping, dict) and len(mapping) > 0:
                global_mean = float(sum(mapping.values()) / len(mapping))
                
            if orig_col in df.columns:
                mapping_dict = mapping if isinstance(mapping, dict) else (mapping.to_dict() if hasattr(mapping, 'to_dict') else dict(mapping))
                mapping_dict_lower = {str(k).strip().lower(): float(v) for k, v in mapping_dict.items()}
                df[enc_col] = df[orig_col].astype(str).str.strip().str.lower().map(mapping_dict_lower).fillna(global_mean)
            else:
                df[enc_col] = global_mean

    # Frequency encodings
    for enc_col in bundle.get("frequency_encoded_cols", []):
        orig_col = enc_col.replace("_freq_enc", "")
        if enc_col in mappings:
            mapping = mappings[enc_col]
            if orig_col in df.columns:
                mapping_dict = mapping if isinstance(mapping, dict) else (mapping.to_dict() if hasattr(mapping, 'to_dict') else dict(mapping))
                mapping_dict_lower = {str(k).strip().lower(): float(v) for k, v in mapping_dict.items()}
                df[enc_col] = df[orig_col].astype(str).str.strip().str.lower().map(mapping_dict_lower).fillna(0.0)
            else:
                df[enc_col] = 0.0

    # Align features and impute missing values
    impute_val = float(bundle.get("NUMERIC_IMPUTE_VALUE", -999.0))
    feature_dict = {}
    for col in model_features:
        if col in df.columns:
            feature_dict[col] = pd.to_numeric(df[col], errors='coerce').fillna(impute_val)
        else:
            feature_dict[col] = np.full(len(df), impute_val, dtype=np.float32)
            
    df_aligned = pd.DataFrame(feature_dict, index=df.index)[model_features]
    return df_aligned

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("FraudWatch")
    st.markdown("**Real-Time Transaction Risk Engine**")
    st.divider()
    
    st.subheader("⚙️ Detection Settings")
    threshold = st.slider(
        "Decision Threshold (Fraud Flag)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Transactions with probability >= threshold are flagged as fraud."
    )
    
    st.info(f"Operating Threshold: **{threshold:.2f}**")
    
    st.divider()
    st.subheader("📊 Model Info")
    st.markdown("""
    - **Classifier**: XGBoost (Histogram)
    - **OOF ROC-AUC**: `0.9694`
    - **Fraud Precision**: `90.0%`
    - **Features**: `341`
    """)
    st.divider()
    st.caption("FraudWatch ML Engine")

# Header
st.markdown('<div class="main-title">🛡️ FraudWatch Risk Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time credit card fraud detection powered by gradient boosted decision trees.</div>', unsafe_allow_html=True)

if not models_ready:
    st.stop()

# Tabs
tab_single, tab_batch = st.tabs(["🔍 Single Transaction Analysis", "📁 Batch CSV Prediction"])

# Tab 1: Single Transaction Analysis
with tab_single:
    st.markdown("### Interactive Transaction Risk Scoring")
    
    preset_col1, _ = st.columns([3, 1])
    with preset_col1:
        preset_choice = st.selectbox(
            "⚡ Quick Fill Preset Profile",
            [
                "Custom Input",
                "🟢 Low Risk - Everyday Cardholder Purchase ($45.50)",
                "🟡 Medium Risk - Large Amount, New Region ($850.00)",
                "🔴 High Risk - Rapid Anomaly / High Value Target ($1,850.00)"
            ]
        )
    
    if preset_choice.startswith("🟢"):
        d_amt, d_prod, d_card4, d_card6, d_hour, d_day = 45.50, "W", "visa", "debit", 14, "Wednesday"
        d_email, d_dev, d_addr1, d_addr2, d_dist = "gmail.com", "desktop", 299.0, 87.0, 5.0
        d_v258, d_v70, d_v294, d_c14 = -999.0, 0.0, 0.0, 1.0
    elif preset_choice.startswith("🟡"):
        d_amt, d_prod, d_card4, d_card6, d_hour, d_day = 850.00, "C", "mastercard", "credit", 3, "Saturday"
        d_email, d_dev, d_addr1, d_addr2, d_dist = "hotmail.com", "mobile", 126.0, 87.0, 140.0
        d_v258, d_v70, d_v294, d_c14 = 1.0, 1.0, 0.0, 4.0
    elif preset_choice.startswith("🔴"):
        d_amt, d_prod, d_card4, d_card6, d_hour, d_day = 1850.00, "C", "discover", "credit", 2, "Sunday"
        d_email, d_dev, d_addr1, d_addr2, d_dist = "protonmail.com", "mobile", 441.0, 87.0, 650.0
        d_v258, d_v70, d_v294, d_c14 = 5.0, 3.0, 2.0, 12.0
    else:
        d_amt, d_prod, d_card4, d_card6, d_hour, d_day = 120.00, "W", "visa", "credit", 15, "Tuesday"
        d_email, d_dev, d_addr1, d_addr2, d_dist = "gmail.com", "desktop", 315.0, 87.0, 10.0
        d_v258, d_v70, d_v294, d_c14 = -999.0, 0.0, 0.0, 1.0

    weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

    with st.form("single_transaction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="section-header">💳 Transaction Details</div>', unsafe_allow_html=True)
            tx_amt = st.number_input("Transaction Amount ($ USD)", min_value=0.01, max_value=50000.0, value=float(d_amt), step=10.0)
            product_code = st.selectbox("Product Code", ["W", "C", "R", "H", "S"], index=["W", "C", "R", "H", "S"].index(d_prod))
            tx_hour = st.slider("Transaction Hour (00:00 - 23:00)", min_value=0, max_value=23, value=int(d_hour))
            tx_day_name = st.selectbox("Day of Week", list(weekday_map.keys()), index=list(weekday_map.keys()).index(d_day))
            tx_weekday = weekday_map[tx_day_name]
            
        with col2:
            st.markdown('<div class="section-header">🔒 Card & Identity</div>', unsafe_allow_html=True)
            card_brand = st.selectbox("Card Brand (card4)", ["visa", "mastercard", "discover", "american express", "missing"], index=["visa", "mastercard", "discover", "american express", "missing"].index(d_card4) if d_card4 in ["visa", "mastercard", "discover", "american express", "missing"] else 0)
            card_type = st.selectbox("Card Type (card6)", ["credit", "debit", "charge card", "debit or credit", "missing"], index=["credit", "debit", "charge card", "debit or credit", "missing"].index(d_card6) if d_card6 in ["credit", "debit", "charge card", "debit or credit", "missing"] else 0)
            card1 = st.number_input("Card Issuer ID (card1)", min_value=1000, max_value=20000, value=10045, step=1)
            card2 = st.number_input("Card Security Code / Bank ID (card2)", min_value=-999.0, max_value=1000.0, value=555.0, step=1.0)
            
        with col3:
            st.markdown('<div class="section-header">🌐 Network & Geography</div>', unsafe_allow_html=True)
            email_options = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "protonmail.com", "mail.com", "anonymous.com", "aol.com", "icloud.com", "missing"]
            email_domain = st.selectbox("Purchaser Email Domain", email_options, index=email_options.index(d_email) if d_email in email_options else 0)
            device_type = st.selectbox("Device Type", ["desktop", "mobile", "missing"], index=["desktop", "mobile", "missing"].index(d_dev) if d_dev in ["desktop", "mobile", "missing"] else 0)
            addr1 = st.number_input("Billing Region (addr1)", min_value=-999.0, max_value=600.0, value=float(d_addr1))
            addr2 = st.number_input("Billing Country Code (addr2)", min_value=-999.0, max_value=100.0, value=float(d_addr2))
            dist1 = st.number_input("Distance to Billing Address (dist1 miles)", min_value=-999.0, max_value=5000.0, value=float(d_dist))

        with st.expander("⚡ Advanced Risk Signals (Top XGBoost Indicators)"):
            adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
            with adv_col1:
                v258 = st.number_input("V258 (Anomaly Signal 1)", value=float(d_v258), step=1.0)
            with adv_col2:
                v70 = st.number_input("V70 (Anomaly Signal 2)", value=float(d_v70), step=1.0)
            with adv_col3:
                v294 = st.number_input("V294 (Frequency Spike)", value=float(d_v294), step=1.0)
            with adv_col4:
                c14 = st.number_input("C14 (Transaction Counter)", value=float(d_c14), step=1.0)

        submit_single = st.form_submit_button("⚡ Analyze Transaction Risk", type="primary")

    if submit_single:
        input_data = {
            "TransactionAmt": tx_amt,
            "ProductCD": product_code,
            "card1": card1,
            "card2": card2,
            "card4": card_brand,
            "card6": card_type,
            "addr1": addr1,
            "addr2": addr2,
            "dist1": dist1,
            "P_emaildomain": email_domain,
            "DeviceType": device_type,
            "Transaction_hour": tx_hour,
            "Transaction_weekday": tx_weekday,
            "V258": v258,
            "V70": v70,
            "V294": v294,
            "C14": c14
        }
        
        try:
            df_single_input = pd.DataFrame([input_data])
            X_prepared = preprocess_dataframe(df_single_input, preprocessing_bundle, model_feature_names)
            prob = float(xgb_model.predict_proba(X_prepared)[0, 1])
            is_fraud = prob >= threshold
            
            if prob < 0.20:
                tier_badge = '<span class="badge-low">🟢 LOW RISK</span>'
                card_class = "result-card-low"
                verdict_text = "TRANSACTION APPROVED"
                recommendation = "Standard transaction pattern. No suspicious signals detected."
            elif prob < threshold:
                tier_badge = '<span class="badge-medium">🟡 MODERATE RISK</span>'
                card_class = "result-card-medium"
                verdict_text = "PASSED WITH MONITORING"
                recommendation = "Elevated risk signals detected. Recommend standard 2FA verification."
            else:
                tier_badge = '<span class="badge-high">🔴 HIGH FRAUD RISK</span>'
                card_class = "result-card-high"
                verdict_text = "FLAGGED FOR FRAUD REVIEW"
                recommendation = "High anomaly confidence. Recommend holding payment or requiring step-up authentication."

            st.markdown(f"""
            <div class="result-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <div>{tier_badge}</div>
                    <div style="font-size: 1.1rem; font-weight: 700;">{verdict_text}</div>
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; margin-bottom: 0.4rem;">
                    {prob * 100:.2f}% <span style="font-size: 1rem; font-weight: 400; color: #64748B;">Fraud Probability</span>
                </div>
                <div style="font-size: 0.95rem; color: #334155;">
                    <b>Actionable Advice:</b> {recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(min(max(prob, 0.0), 1.0))
            
            st.markdown("#### 🔍 Key Risk Factor Assessment")
            fact_col1, fact_col2, fact_col3, fact_col4 = st.columns(4)
            with fact_col1:
                st.metric("Amount Exposure", f"${tx_amt:,.2f}", delta="High" if tx_amt > 500 else "Normal", delta_color="inverse")
            with fact_col2:
                st.metric("Product Risk", product_code, delta="Higher Risk" if product_code in ["C", "S"] else "Standard", delta_color="inverse")
            with fact_col3:
                st.metric("Domain Reputation", email_domain, delta="High Risk" if email_domain in ["protonmail.com", "mail.com", "outlook.es"] else "Trusted", delta_color="inverse")
            with fact_col4:
                st.metric("Time Window", f"{tx_hour:02d}:00", delta="Late Night" if tx_hour in [1, 2, 3, 4] else "Daytime", delta_color="inverse")
                
        except Exception as err:
            st.error(f"❌ Error computing single transaction prediction: {str(err)}")

# Tab 2: Batch CSV Prediction
with tab_batch:
    st.markdown("### Batch Transaction Scoring & Bulk CSV Inference")
    st.write("Upload a CSV file containing transactions. The preprocessor will automatically align features, impute missing values, and calculate fraud risk probabilities.")
    
    upload_col, info_col = st.columns([2, 1])
    
    with upload_col:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload raw transaction datasets with transaction columns."
        )
        
    with info_col:
        st.markdown("**Sample Data & Testing:**")
        sample_path = os.path.join(BASE_DIR, "data", "test_transaction.csv")
        sample_exists = os.path.exists(sample_path)
        
        load_sample = st.button("📥 Load 200 Samples from Repository Test Data", disabled=not sample_exists)
        if not sample_exists:
            st.caption("💡 `data/test_transaction.csv` not found.")

    df_raw_batch = None
    source_name = ""
    
    if uploaded_file is not None:
        try:
            df_raw_batch = pd.read_csv(uploaded_file)
            source_name = uploaded_file.name
        except Exception as e:
            st.error(f"❌ Failed to parse uploaded CSV file: {str(e)}")
    elif load_sample and sample_exists:
        try:
            with st.spinner("Loading sample transactions..."):
                df_raw_batch = pd.read_csv(sample_path, nrows=200)
                source_name = "test_transaction.csv (Sample 200 records)"
        except Exception as e:
            st.error(f"❌ Failed to load local sample file: {str(e)}")

    if df_raw_batch is not None:
        st.divider()
        st.markdown(f"#### 📄 Dataset Loaded: `{source_name}`")
        st.write(f"Total Records: **{len(df_raw_batch):,}** | Columns Found: **{len(df_raw_batch.columns)}**")
        
        with st.expander("👁️ Preview Raw Input Data (First 5 Rows)"):
            st.dataframe(df_raw_batch.head(5))

        if st.button("🚀 Run Batch Fraud Risk Inference", type="primary"):
            progress_bar = st.progress(0, text="Initializing Preprocessor Pipeline...")
            t0 = time.time()
            
            try:
                progress_bar.progress(30, text="Transforming & Aligning 341 Features...")
                X_batch_aligned = preprocess_dataframe(df_raw_batch, preprocessing_bundle, model_feature_names)
                
                progress_bar.progress(70, text="Executing XGBoost Inference Engine...")
                probs = xgb_model.predict_proba(X_batch_aligned)[:, 1]
                
                progress_bar.progress(90, text="Compiling Prediction Metrics...")
                
                df_results = df_raw_batch.copy()
                df_results["Fraud_Probability"] = np.round(probs, 4)
                df_results["Fraud_Flag"] = (probs >= threshold).astype(int)
                df_results["Risk_Tier"] = np.where(
                    probs >= threshold, "🔴 HIGH",
                    np.where(probs >= 0.20, "🟡 MEDIUM", "🟢 LOW")
                )
                
                elapsed = time.time() - t0
                progress_bar.progress(100, text=f"Inference Completed in {elapsed:.2f}s!")
                
                total_tx = len(df_results)
                flagged_count = int(df_results["Fraud_Flag"].sum())
                flagged_pct = (flagged_count / total_tx) * 100 if total_tx > 0 else 0
                avg_prob = float(df_results["Fraud_Probability"].mean()) * 100
                
                st.markdown("### 📊 Batch Prediction Overview")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Transactions", f"{total_tx:,}")
                m2.metric("Flagged Fraud (Class 1)", f"{flagged_count:,}", delta=f"{flagged_pct:.1f}%", delta_color="inverse")
                m3.metric("Avg Fraud Probability", f"{avg_prob:.2f}%")
                m4.metric("Inference Time", f"{elapsed:.2f}s", delta=f"{int(total_tx/max(elapsed, 0.001)):,} tx/s")
                
                st.markdown("#### 📈 Risk Distribution Analysis")
                c_col1, c_col2 = st.columns([2, 1])
                
                with c_col1:
                    st.caption("Probability Distribution Histogram")
                    hist_values, bin_edges = np.histogram(probs, bins=20, range=(0, 1))
                    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(hist_values))]
                    chart_df = pd.DataFrame({"Count": hist_values}, index=bin_labels)
                    st.bar_chart(chart_df, height=240)
                    
                with c_col2:
                    st.caption("Risk Tier Proportions")
                    tier_counts = df_results["Risk_Tier"].value_counts()
                    st.dataframe(tier_counts.rename("Transactions"))

                st.markdown("#### 📋 Scored Transactions Table")
                filter_choice = st.radio(
                    "Display Filter:",
                    ["Show All Records", "Show Only Flagged High Risk (Fraud)", "Show Top 50 Riskiest"],
                    horizontal=True
                )
                
                if filter_choice == "Show Only Flagged High Risk (Fraud)":
                    display_df = df_results[df_results["Fraud_Flag"] == 1]
                elif filter_choice == "Show Top 50 Riskiest":
                    display_df = df_results.sort_values(by="Fraud_Probability", ascending=False).head(50)
                else:
                    display_df = df_results

                pred_cols = ["Fraud_Probability", "Fraud_Flag", "Risk_Tier"]
                other_cols = [c for c in display_df.columns if c not in pred_cols]
                display_df = display_df[pred_cols + other_cols]

                st.dataframe(display_df, height=350)
                
                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Enriched Predictions CSV",
                    data=csv_data,
                    file_name=f"fraud_predictions_{int(time.time())}.csv",
                    mime="text/csv",
                    type="primary"
                )

            except Exception as e:
                st.error(f"❌ Batch inference failed: {str(e)}")

# Footer
st.divider()
st.markdown(
    "<center><small style='color: #94A3B8;'>FraudWatch • Machine Learning Risk System</small></center>",
    unsafe_allow_html=True
)
