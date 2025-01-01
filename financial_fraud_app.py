import streamlit as st
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

# Application title
st.markdown('<p class="main-header">💳 Financial Fraud Detection System</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🌟 Navigation")
st.sidebar.info("""
Use this application to detect and analyze:
- Uploaded financial messages (files)
- Typed financial messages
- Transaction records
""")

# Main layout
st.write("---")
st.markdown("### 🔍 **Introduction**")
st.info("""
Welcome to the **Financial Fraud Detection System**!  
Use this tool to analyze financial messages, emails, and transactions for possible fraudulent activity.  
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
            st.success("✔️ Placeholder Result: No fraud detected in the uploaded file.")
    else:
        st.markdown('<p class="small-text">No file uploaded yet.</p>', unsafe_allow_html=True)

# Section 2: Manual Message Input
with st.expander("✍️ Enter a Financial Message"):
    st.markdown('<p class="sub-header">2️⃣ Enter a Message</p>', unsafe_allow_html=True)
    user_message = st.text_area(
        "Type or paste your financial message below:",
        placeholder="Enter your message, email, or transaction text here."
    )

    if st.button("🔍 Analyze Message"):
        if user_message.strip():
            with st.spinner("Analyzing message..."):
                time.sleep(2)
            st.success("✔️ Placeholder Result: No fraud detected.")
        else:
            st.warning("⚠️ Please enter a valid message.")

# Section 3: Transaction Details
with st.expander("💵 Enter Transaction Details"):
    st.markdown('<p class="sub-header">3️⃣ Transaction Analysis</p>', unsafe_allow_html=True)
    with st.form("transaction_form"):
        transaction_id = st.text_input("Transaction ID:", placeholder="e.g., TX12345")
        transaction_amount = st.number_input("Transaction Amount (in USD):", min_value=0.0, step=0.01)
        transaction_date = st.date_input("Transaction Date")
        transaction_description = st.text_area("Transaction Description:", placeholder="e.g., Payment for invoice #5678")
        submitted = st.form_submit_button("🔍 Analyze Transaction")
        if submitted:
            if transaction_id.strip():
                with st.spinner("Analyzing transaction..."):
                    time.sleep(2)
                st.success("✔️ Placeholder Result: Transaction is valid. No fraud detected.")
            else:
                st.warning("⚠️ Please enter a valid Transaction ID.")

# Footer
st.write("---")
st.markdown('<p class="footer">Developed with ❤️ for fraud detection | © 2024</p>', unsafe_allow_html=True)
