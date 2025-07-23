import os
import time
import tempfile
os.environ["STREAMLIT_WATCH_MODE"] = "false"
import streamlit as st
from transformers import pipeline  # Hugging Face 🤗
from dotenv import load_dotenv     # For loading AWS keys from .env

# ────────────────────────────────────────────────────────────────────────────────
# CONFIGURE STREAMLIT PAGE
st.set_page_config(page_title="Financial Fraud Detection (BERT)", page_icon="💡", layout="wide")

# ────────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
st.markdown("""
<style>
    .main-header {font-size:44px;color:#4CAF50;text-align:center;
                  font-weight:bold;margin-bottom:10px;}
    .sub-header  {font-size:20px;color:#1E90FF;font-weight:bold;margin-top:10px;}
    .footer      {font-size:12px;text-align:center;color:#696969;margin-top:50px;}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# LOAD AWS CREDENTIALS FROM .env
load_dotenv()

# S3 Configuration
S3_PATH = "s3://fraudet/fraud_detection_model/"
bucket = "fraudet"
prefix = "fraud_detection_model/"

# ────────────────────────────────────────────────────────────────────────────────
# LOAD MODEL FROM S3 (with caching)
@st.cache_resource(show_spinner="🔁 Downloading BERT model from S3…")
def load_model_pipeline():
    import boto3

    s3 = boto3.client("s3")
    tmpdir = tempfile.mkdtemp()

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in response.get("Contents", []):
        s3_key = obj["Key"]
        rel_path = os.path.relpath(s3_key, prefix)
        local_path = os.path.join(tmpdir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, s3_key, local_path)

    return pipeline("text-classification", model=tmpdir, tokenizer=tmpdir)

# ────────────────────────────────────────────────────────────────────────────────
# UI HEADER
st.markdown('<p class="main-header">💡 BERT-based Financial Fraud Detection</p>', unsafe_allow_html=True)

st.sidebar.title("🔍 Features")
st.sidebar.info("""• Detect fraud in financial messages  
• Analyze .txt files or individual entries""")

# ────────────────────────────────────────────────────────────────────────────────
# INTRO
st.write("---")
st.markdown("### 🧠 Introduction")
st.info("Welcome to the **BERT-based Financial Fraud Detection System**! This tool uses a fine-tuned transformer model to analyze and flag potentially fraudulent financial messages.")

# Load the model once
classifier = load_model_pipeline()

# ────────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTIONALITY
with st.expander("💳 Financial Fraud Detection", expanded=True):
    # 1️⃣ FILE UPLOAD
    st.markdown('<p class="sub-header">1️⃣ Upload File</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a .txt file with one message per line", type=["txt"])

    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.text_area("📄 File Preview", file_content, height=150)

        if st.button("🔍 Analyze File"):
            with st.spinner("Analyzing messages…"):
                messages = [line.strip() for line in file_content.splitlines() if line.strip()]
                results = classifier(messages)
                fraud_count = sum(res["label"] == "LABEL_1" for res in results)
                time.sleep(0.5)  # cosmetic delay
            st.success(f"✔️ Detected **{fraud_count}** fraudulent message(s).")

    # 2️⃣ MANUAL ENTRY
    st.markdown('<p class="sub-header">2️⃣ Manual Entry</p>', unsafe_allow_html=True)
    user_input = st.text_area("Enter a message to check for fraud:")

    if st.button("🔍 Analyze Message"):
        if user_input.strip():
            with st.spinner("Analyzing…"):
                result = classifier(user_input)[0]
            label, score = result["label"], result["score"]

            if label == "LABEL_1":
                st.error(f"🚨 Fraud Detected! (Confidence: {score:.2%})")
            else:
                st.success(f"✔️ No fraud detected (Confidence: {score:.2%})")
        else:
            st.warning("⚠️ Please enter a valid message.")

# ────────────────────────────────────────────────────────────────────────────────
# FOOTER
st.write("---")
st.markdown('<p class="footer">Developed with Streamlit & Transformers 🤖 | ©2025</p>', unsafe_allow_html=True)
