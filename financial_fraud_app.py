# Re-run the previous code to regenerate the Streamlit script after kernel reset

# Preparing a revised Streamlit app script using the newly trained BERT model

streamlit_code = """
import streamlit as st
from transformers import pipeline
import time

# Set page configuration
st.set_page_config(page_title="Financial Fraud Detection (BERT)", page_icon="💡", layout="wide")

# Custom CSS for styling
st.markdown(\"\"\"
<style>
    .main-header {
        font-size: 44px;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 20px;
        color: #1E90FF;
        font-weight: bold;
        margin-top: 10px;
    }
    .footer {
        font-size: 12px;
        text-align: center;
        color: #696969;
        margin-top: 50px;
    }
</style>
\"\"\", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model_pipeline():
    return pipeline("text-classification", model="./fraud_detection_model", tokenizer="./fraud_detection_model")

classifier = load_model_pipeline()

# Application title
st.markdown('<p class="main-header">💡 BERT-based Financial Fraud Detection</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔍 Features")
st.sidebar.info(\"\"\"
Use this app to:
- Detect fraud in financial messages
- Analyze .txt files or individual entries
\"\"\")

# Introduction
st.write("---")
st.markdown("### 🧠 Introduction")
st.info(\"\"\"
Welcome to the **BERT-based Financial Fraud Detection System**!  
This tool uses a fine-tuned transformer model to analyze and flag potentially fraudulent financial messages.
\"\"\")

# Financial Fraud Detection Section
with st.expander("💳 Financial Fraud Detection", expanded=True):
    st.markdown('<p class="sub-header">1️⃣ Upload File</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a .txt file with one message per line", type=["txt"])

    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.text_area("📄 File Preview", file_content, height=150)
        if st.button("🔍 Analyze File"):
            with st.spinner("Analyzing messages..."):
                time.sleep(1)
                messages = [line.strip() for line in file_content.split("\\n") if line.strip()]
                results = classifier(messages)
                fraud_count = sum(1 for r in results if r['label'] == 'LABEL_1')
            st.success(f"✔️ Detected {fraud_count} fraudulent message(s)")

    st.markdown('<p class="sub-header">2️⃣ Manual Entry</p>', unsafe_allow_html=True)
    user_input = st.text_area("Enter a message to check for fraud:")

    if st.button("🔍 Analyze Message"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                result = classifier(user_input)
                label = result[0]['label']
                score = result[0]['score']
                if label == 'LABEL_1':
                    st.error(f"🚨 Fraud Detected! (Confidence: {score:.2f})")
                else:
                    st.success(f"✔️ No fraud detected (Confidence: {score:.2f})")
        else:
            st.warning("⚠️ Please enter a valid message.")

# Footer
st.write("---")
st.markdown('<p class="footer">Developed using Transformers 🤖 | © 2024</p>', unsafe_allow_html=True)
"""

# Save it to a file
with open("/mnt/data/bert_fraud_detection_app.py", "w") as f:
    f.write(streamlit_code)

"/mnt/data/bert_fraud_detection_app.py"
