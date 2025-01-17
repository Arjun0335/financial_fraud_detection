import streamlit as st
import pickle
import time

# Set page configuration
st.set_page_config(page_title="Financial Fraud Detection", page_icon="💳", layout="wide")

# Custom CSS for styling
st.markdown("""
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
    .report-section {
        padding: 10px;
        border: 2px solid #F0F0F0;
        border-radius: 10px;
        background-color: #FAFAFA;
        margin-bottom: 15px;
    }
    .custom-expander .streamlit-expanderHeader {
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open("vectorizer.pkl", "rb") as vec_file, open("spam_text.pkl", "rb") as file:
        vectorizer = pickle.load(vec_file)
        model = pickle.load(file)
    return vectorizer, model

vectorizer, model = load_model()

# Application title
st.markdown('<p class="main-header">💳 Financial Fraud Detection System</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🌟 Navigation")
st.sidebar.info("""
Use this application to detect and analyze:
- Uploaded financial messages (files)
- Typed financial messages
""")

# Main layout
st.write("---")
st.markdown("### 🔍 **Introduction**")
st.info(""" 
Welcome to the **Financial Fraud Detection System**!  
Use this tool to analyze financial messages and emails for possible fraudulent activity.  
Select a task below to get started.
""")

# Section 1: File Upload
with st.expander("📂 Upload Financial Message File", expanded=True):
    st.markdown('<p class="sub-header">1️⃣ Upload a File</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a text file containing messages to analyze:", type=["txt"])
    
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.text_area("📄 File Content Preview:", file_content, height=150)
        if st.button("🔍 Analyze File"):
            with st.spinner("Analyzing uploaded file..."):
                time.sleep(2)
            # Placeholder for multiple-line analysis
            messages = [line.strip() for line in file_content.split("\n") if line.strip()]
            transformed_messages = vectorizer.transform(messages)
            predictions = model.predict(transformed_messages)
            fraud_count = (predictions == 'spam').sum()
            st.success(f"✔️ Analysis Complete: Fraud Detected in {fraud_count} message(s).")

# Section 2: Manual Message Input
with st.expander("✍️ Enter a Financial Message", expanded = True):
    st.markdown('<p class="sub-header">2️⃣ Enter a Message</p>', unsafe_allow_html=True)
    user_message = st.text_area(
        "Type or paste your financial message below:",
        placeholder="Enter your message, email, or transaction text here."
    )

    if st.button("🔍 Analyze Message"):
        if user_message.strip():
            with st.spinner("Analyzing message..."):
                time.sleep(1)
            transformed_message = vectorizer.transform([user_message])  # Transform the input text
            prediction = model.predict(transformed_message)
            if prediction == 'spam':
                st.error("🚨 Fraud detected in the message!")
            else:
                st.success("✔️ Message is safe. No fraud detected.")
        else:
            st.warning("⚠️ Please enter a valid message.")

# Footer
st.write("---")
st.markdown('<p class="footer">Developed with ❤️ for fraud detection | © 2024</p>', unsafe_allow_html=True)
